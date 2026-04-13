"""Tests for L8 async fill reconciliation — FillReconciler.

All tests are offline.  A _FakeLiveWriteClient is used so no CLOB network
call is made.  A real Ledger instance is used so ledger state mutations
are verified against the actual implementation.

Coverage goals:
  1. register is idempotent (duplicate registration is a no-op)
  2. poll on SUBMITTED (size_matched=0): place_limit_order called, no fill applied
  3. poll on PARTIAL fill: fill delta applied, order stays pending
  4. poll on FILLED (matched): fill delta applied, order removed from pending
  5. poll on REJECTED / poll error: order stays pending, no crash
  6. Idempotent second poll with same status: no double-apply
  7. Multiple sequential polls accumulate delta correctly
  8. Runner._dispatch_order registers with reconciler on live submit
  9. Runner._dispatch_order does NOT register on REJECTED report
 10. Runner._poll_live_fills called at start of run_once (fills visible before scan)
 11. fill_reconciler=None by default (paper path unaffected)
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from src.domain.models import (
    ExecutionReport,
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
)
from src.live.broker import LiveBroker
from src.live.client import LiveClientError, LiveOrderResult, LiveOrderStatus, LiveWriteClient
from src.live.reconciler import FillReconciler
from src.paper.ledger import Ledger
from src.runtime.runner import ResearchRunner
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent(
    size: float = 50.0,
    limit_price: float = 0.55,
    side: str = "BUY",
) -> OrderIntent:
    return OrderIntent(
        intent_id=str(uuid4()),
        candidate_id="cand-rec-1",
        mode=OrderMode.LIVE,
        market_slug="rec-market",
        token_id="tok-rec",
        position_id=str(uuid4()),
        side=side,
        order_type=OrderType.LIMIT,
        size=size,
        limit_price=limit_price,
        max_notional_usd=size * limit_price,
        ts=datetime.now(timezone.utc),
    )


class _FakeLiveWriteClient:
    """Minimal LiveWriteClient stand-in for the reconciler tests.

    configure_status() sets the LiveOrderStatus returned by the next
    get_order_status() call.  To simulate a network error, pass
    raises=True to configure_status.
    """

    def __init__(self) -> None:
        self._statuses: dict[str, LiveOrderStatus] = {}
        self._raises: dict[str, Exception] = {}
        self.poll_calls: list[str] = []

    def configure_status(
        self,
        order_id: str,
        *,
        size_matched: float,
        size_remaining: float,
        status: str = "live",
        raises: Exception | None = None,
    ) -> None:
        if raises is not None:
            self._raises[order_id] = raises
        else:
            self._raises.pop(order_id, None)
            self._statuses[order_id] = LiveOrderStatus(
                order_id=order_id,
                status=status,
                size_matched=size_matched,
                size_remaining=size_remaining,
            )

    def get_order_status(self, order_id: str) -> LiveOrderStatus:
        self.poll_calls.append(order_id)
        if order_id in self._raises:
            raise self._raises[order_id]
        return self._statuses[order_id]

    # stub out submit / cancel so isinstance checks pass if needed
    def submit_order(self, **_):  # pragma: no cover
        raise LiveClientError("not implemented in fake")

    def cancel_order(self, _):  # pragma: no cover
        raise LiveClientError("not implemented in fake")


def _reconciler() -> tuple[FillReconciler, _FakeLiveWriteClient]:
    fake = _FakeLiveWriteClient()
    rec = FillReconciler(fake)  # type: ignore[arg-type]
    return rec, fake


def _ledger(cash: float = 10_000.0) -> Ledger:
    return Ledger(cash=cash)


# ---------------------------------------------------------------------------
# 1. register idempotency
# ---------------------------------------------------------------------------

class TestRegisterIdempotency(unittest.TestCase):

    def test_register_once(self) -> None:
        rec, fake = _reconciler()
        intent = _intent()
        rec.register("ord-1", intent)
        self.assertEqual(rec.pending_count, 1)

    def test_register_twice_is_noop(self) -> None:
        rec, fake = _reconciler()
        intent = _intent()
        rec.register("ord-1", intent)
        rec.register("ord-1", intent)
        self.assertEqual(rec.pending_count, 1)

    def test_register_different_ids(self) -> None:
        rec, fake = _reconciler()
        rec.register("ord-1", _intent())
        rec.register("ord-2", _intent())
        self.assertEqual(rec.pending_count, 2)


# ---------------------------------------------------------------------------
# 2. poll — SUBMITTED (size_matched=0): order placed in ledger, no fill
# ---------------------------------------------------------------------------

class TestPollSubmitted(unittest.TestCase):

    def setUp(self) -> None:
        self.rec, self.fake = _reconciler()
        self.ledger = _ledger()
        self.intent = _intent(size=50.0, limit_price=0.55)
        self.fake.configure_status(
            "ord-s1", size_matched=0.0, size_remaining=50.0, status="live"
        )
        self.rec.register("ord-s1", self.intent)

    def test_order_placed_in_ledger(self) -> None:
        self.rec.poll(self.ledger)
        self.assertIn("ord-s1", self.ledger.orders)

    def test_no_fill_applied_when_zero_matched(self) -> None:
        self.rec.poll(self.ledger)
        order = self.ledger.orders["ord-s1"]
        self.assertEqual(order.filled_shares, 0.0)

    def test_order_stays_pending(self) -> None:
        self.rec.poll(self.ledger)
        self.assertEqual(self.rec.pending_count, 1)

    def test_poll_returns_zero_completed(self) -> None:
        result = self.rec.poll(self.ledger)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 3. poll — PARTIAL fill: delta applied, order stays pending
# ---------------------------------------------------------------------------

class TestPollPartial(unittest.TestCase):

    def setUp(self) -> None:
        self.rec, self.fake = _reconciler()
        self.ledger = _ledger()
        self.intent = _intent(size=50.0, limit_price=0.55)
        self.fake.configure_status(
            "ord-p1", size_matched=20.0, size_remaining=30.0, status="live"
        )
        self.rec.register("ord-p1", self.intent)

    def test_fill_applied_for_matched_shares(self) -> None:
        self.rec.poll(self.ledger)
        order = self.ledger.orders["ord-p1"]
        self.assertAlmostEqual(order.filled_shares, 20.0, places=5)

    def test_order_stays_pending_when_partial(self) -> None:
        self.rec.poll(self.ledger)
        self.assertEqual(self.rec.pending_count, 1)

    def test_poll_returns_zero_completed(self) -> None:
        result = self.rec.poll(self.ledger)
        self.assertEqual(result, [])

    def test_position_record_created(self) -> None:
        self.rec.poll(self.ledger)
        # A BUY fill creates a position record
        self.assertTrue(len(self.ledger.position_records) > 0)


# ---------------------------------------------------------------------------
# 4. poll — FILLED (matched): fill applied, order removed from pending
# ---------------------------------------------------------------------------

class TestPollFilled(unittest.TestCase):

    def setUp(self) -> None:
        self.rec, self.fake = _reconciler()
        self.ledger = _ledger()
        self.intent = _intent(size=50.0, limit_price=0.55)
        self.fake.configure_status(
            "ord-f1", size_matched=50.0, size_remaining=0.0, status="matched"
        )
        self.rec.register("ord-f1", self.intent)

    def test_fill_applied_for_full_size(self) -> None:
        self.rec.poll(self.ledger)
        order = self.ledger.orders["ord-f1"]
        self.assertAlmostEqual(order.filled_shares, 50.0, places=5)

    def test_order_removed_from_pending_on_terminal(self) -> None:
        self.rec.poll(self.ledger)
        self.assertEqual(self.rec.pending_count, 0)

    def test_poll_returns_one_completed(self) -> None:
        result = self.rec.poll(self.ledger)
        self.assertEqual(len(result), 1)

    def test_position_visible_in_ledger(self) -> None:
        self.rec.poll(self.ledger)
        self.assertTrue(any(r.is_open for r in self.ledger.position_records.values()))


# ---------------------------------------------------------------------------
# 5. poll — REJECTED / network error: order stays pending, no crash
# ---------------------------------------------------------------------------

class TestPollRejectedOrError(unittest.TestCase):

    def test_stays_pending_on_poll_error(self) -> None:
        rec, fake = _reconciler()
        ledger = _ledger()
        fake.configure_status(
            "ord-e1",
            size_matched=0.0,
            size_remaining=0.0,
            raises=LiveClientError("network down"),
        )
        rec.register("ord-e1", _intent())
        rec.poll(ledger)
        self.assertEqual(rec.pending_count, 1)

    def test_canceled_status_removes_order(self) -> None:
        rec, fake = _reconciler()
        ledger = _ledger()
        fake.configure_status(
            "ord-c1", size_matched=0.0, size_remaining=50.0, status="canceled"
        )
        rec.register("ord-c1", _intent())
        rec.poll(ledger)
        self.assertEqual(rec.pending_count, 0)

    def test_expired_status_removes_order(self) -> None:
        rec, fake = _reconciler()
        ledger = _ledger()
        fake.configure_status(
            "ord-x1", size_matched=0.0, size_remaining=50.0, status="expired"
        )
        rec.register("ord-x1", _intent())
        rec.poll(ledger)
        self.assertEqual(rec.pending_count, 0)


# ---------------------------------------------------------------------------
# 6. Idempotency: second poll with same status does not double-apply
# ---------------------------------------------------------------------------

class TestIdempotentPoll(unittest.TestCase):

    def test_second_poll_same_status_no_double_fill(self) -> None:
        rec, fake = _reconciler()
        ledger = _ledger()
        intent = _intent(size=50.0, limit_price=0.55)
        fake.configure_status(
            "ord-id1", size_matched=20.0, size_remaining=30.0, status="live"
        )
        rec.register("ord-id1", intent)
        rec.poll(ledger)
        rec.poll(ledger)  # same status — delta=0
        order = ledger.orders["ord-id1"]
        self.assertAlmostEqual(order.filled_shares, 20.0, places=5)


# ---------------------------------------------------------------------------
# 7. Sequential polls accumulate delta
# ---------------------------------------------------------------------------

class TestSequentialPolls(unittest.TestCase):

    def test_incremental_fills_accumulated(self) -> None:
        rec, fake = _reconciler()
        ledger = _ledger()
        intent = _intent(size=50.0, limit_price=0.55)
        rec.register("ord-seq1", intent)

        # Poll 1: partial fill
        fake.configure_status(
            "ord-seq1", size_matched=10.0, size_remaining=40.0, status="live"
        )
        rec.poll(ledger)
        self.assertAlmostEqual(ledger.orders["ord-seq1"].filled_shares, 10.0, places=5)

        # Poll 2: more fills arrive
        fake.configure_status(
            "ord-seq1", size_matched=30.0, size_remaining=20.0, status="live"
        )
        rec.poll(ledger)
        self.assertAlmostEqual(ledger.orders["ord-seq1"].filled_shares, 30.0, places=5)
        self.assertEqual(rec.pending_count, 1)

        # Poll 3: fully filled
        fake.configure_status(
            "ord-seq1", size_matched=50.0, size_remaining=0.0, status="matched"
        )
        rec.poll(ledger)
        self.assertAlmostEqual(ledger.orders["ord-seq1"].filled_shares, 50.0, places=5)
        self.assertEqual(rec.pending_count, 0)


# ---------------------------------------------------------------------------
# 8. Runner._dispatch_order registers with reconciler on live submit
# ---------------------------------------------------------------------------

class TestRunnerRegistersWithReconciler(unittest.TestCase):

    def _live_runner_with_fake_clob(self) -> tuple[ResearchRunner, _FakeLiveWriteClient]:
        from src.live.client import LiveWriteClient

        fake_clob = MagicMock()
        fake_clob.create_and_post_order.return_value = {
            "orderID": "clob-reg-1",
            "status": "live",
            "size_matched": "0.0",
        }
        lc = LiveWriteClient(fake_clob, dry_run=False)
        runner = ResearchRunner()
        runner.store = MagicMock()
        runner.opportunity_store = MagicMock()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        runner.live_broker = LiveBroker(lc)
        # Attach reconciler backed by the same LiveWriteClient
        rec, fake_status_client = _reconciler()
        runner.fill_reconciler = rec
        return runner, fake_status_client

    def test_reconciler_registers_order_after_live_dispatch(self) -> None:
        runner, _ = self._live_runner_with_fake_clob()
        intent = _intent()
        runner._dispatch_order(intent, object())
        self.assertEqual(runner.fill_reconciler.pending_count, 1)

    def test_reconciler_registered_order_id_matches_report(self) -> None:
        runner, _ = self._live_runner_with_fake_clob()
        intent = _intent()
        report = runner._dispatch_order(intent, object())
        live_id = report.metadata.get("live_order_id")
        self.assertIn(live_id, runner.fill_reconciler._pending)


# ---------------------------------------------------------------------------
# 9. Runner._dispatch_order does NOT register on REJECTED report
# ---------------------------------------------------------------------------

class TestRunnerDoesNotRegisterOnRejection(unittest.TestCase):

    def test_rejected_report_not_registered(self) -> None:
        from src.live.client import LiveWriteClient

        runner = ResearchRunner()
        runner.store = MagicMock()
        runner.opportunity_store = MagicMock()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False

        # LiveBroker that always returns REJECTED
        mock_broker = MagicMock(spec=LiveBroker)
        mock_broker.submit_limit_order.return_value = ExecutionReport(
            intent_id="x",
            status=OrderStatus.REJECTED,
            ts=datetime.now(timezone.utc),
            metadata={"live_order_id": None, "error": "bad tick size"},
        )
        runner.live_broker = mock_broker

        rec, _ = _reconciler()
        runner.fill_reconciler = rec

        runner._dispatch_order(_intent(), object())
        self.assertEqual(rec.pending_count, 0)


# ---------------------------------------------------------------------------
# 10. Runner._poll_live_fills is called at start of run_once
# ---------------------------------------------------------------------------

class TestRunnerPollsAtCycleStart(unittest.TestCase):

    def test_poll_called_during_run_once(self) -> None:
        runner = ResearchRunner()
        runner.store = MagicMock()
        runner.opportunity_store = MagicMock()

        mock_reconciler = MagicMock(spec=FillReconciler)
        mock_reconciler.poll.return_value = []
        runner.fill_reconciler = mock_reconciler

        with patch("src.runtime.runner.fetch_markets", return_value=[]):
            runner.run_once()

        mock_reconciler.poll.assert_called_once_with(runner.paper_ledger)


# ---------------------------------------------------------------------------
# 11. fill_reconciler is None by default (paper path unaffected)
# ---------------------------------------------------------------------------

class TestReconcilerDefaultNone(unittest.TestCase):

    def test_fill_reconciler_none_by_default(self) -> None:
        runner = ResearchRunner()
        self.assertIsNone(runner.fill_reconciler)

    def test_paper_dispatch_unaffected_when_reconciler_none(self) -> None:
        from src.domain.models import OrderMode
        runner = ResearchRunner()
        runner.store = MagicMock()
        runner.opportunity_store = MagicMock()
        # defaults: paper mode
        intent = _intent()
        intent = OrderIntent(
            intent_id=intent.intent_id,
            candidate_id=intent.candidate_id,
            mode=OrderMode.PAPER,
            market_slug=intent.market_slug,
            token_id=intent.token_id,
            position_id=intent.position_id,
            side=intent.side,
            order_type=intent.order_type,
            size=intent.size,
            limit_price=intent.limit_price,
            max_notional_usd=intent.max_notional_usd,
            ts=intent.ts,
        )
        with patch.object(runner.paper_broker, "submit_limit_order",
                          return_value=ExecutionReport(
                              intent_id=intent.intent_id,
                              status=OrderStatus.SUBMITTED,
                              ts=datetime.now(timezone.utc),
                          )) as mock_paper:
            runner._dispatch_order(intent, object())
        mock_paper.assert_called_once()


if __name__ == "__main__":
    unittest.main()
