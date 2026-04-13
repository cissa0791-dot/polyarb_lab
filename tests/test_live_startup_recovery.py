"""Tests for L9 startup position reconciliation.

Validates that prior-session live orders survive a process restart by being
re-registered with FillReconciler from persistent store data.

Architecture under test:
  ResearchStore.load_pending_live_orders()
      → queries execution_reports + order_intents
      → returns (OrderIntent, live_order_id) for SUBMITTED / PARTIAL live orders

  ResearchRunner._recover_pending_live_orders()
      → calls load_pending_live_orders
      → re-registers each with fill_reconciler
      → no-op when fill_reconciler is None or _effective_live_enabled is False

  ResearchRunner.run_once() startup gate:
      → _recover_pending_live_orders() runs exactly once per process lifetime
      → guarded by _startup_recovery_done flag

Coverage goals:
  Store layer:
    1.  Empty result when no execution_reports exist
    2.  Returns SUBMITTED live order
    3.  Returns PARTIAL live order
    4.  Excludes FILLED live order (terminal)
    5.  Excludes REJECTED live order (terminal)
    6.  Excludes CANCELED live order (terminal)
    7.  Excludes paper-mode orders (mode != live)
    8.  Excludes dry-run rows (live_order_id is None in metadata)
    9.  Returns only the latest report when multiple exist (PARTIAL wins over prior SUBMITTED)
   10.  Skips if latest report is terminal even if an older report was SUBMITTED

  Runner layer:
   11.  _recover_pending_live_orders is no-op when fill_reconciler is None
   12.  _recover_pending_live_orders is no-op when _effective_live_enabled is False
   13.  _recover_pending_live_orders registers all pending orders with reconciler
   14.  Recovery runs only once per process lifetime (_startup_recovery_done flag)
   15.  Second run_once() does not double-register
   16.  After recovery + poll, position is visible in paper ledger
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.domain.models import (
    ExecutionReport,
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
)
from src.live.broker import LiveBroker
from src.live.client import LiveOrderStatus, LiveWriteClient
from src.live.reconciler import FillReconciler
from src.paper.ledger import Ledger
from src.runtime.runner import ResearchRunner
from src.storage.event_store import ResearchStore


# ---------------------------------------------------------------------------
# Store test helpers
# ---------------------------------------------------------------------------

def _mem_store() -> ResearchStore:
    """In-memory SQLite ResearchStore — isolated per test."""
    return ResearchStore("sqlite:///:memory:")


def _intent(
    intent_id: str | None = None,
    mode: OrderMode = OrderMode.LIVE,
    size: float = 50.0,
    limit_price: float = 0.55,
    side: str = "BUY",
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id or str(uuid4()),
        candidate_id="cand-s9",
        mode=mode,
        market_slug="startup-market",
        token_id="tok-s9",
        position_id=str(uuid4()),
        side=side,
        order_type=OrderType.LIMIT,
        size=size,
        limit_price=limit_price,
        max_notional_usd=size * limit_price,
        ts=datetime.now(timezone.utc),
    )


def _report(
    intent_id: str,
    status: OrderStatus,
    live_order_id: str | None = "clob-order-1",
    filled_size: float = 0.0,
) -> ExecutionReport:
    return ExecutionReport(
        intent_id=intent_id,
        status=status,
        filled_size=filled_size,
        metadata={"live_order_id": live_order_id, "live_status": "live"},
        ts=datetime.now(timezone.utc),
    )


def _save_live_order(
    store: ResearchStore,
    *,
    status: OrderStatus,
    live_order_id: str | None = "clob-order-1",
    mode: OrderMode = OrderMode.LIVE,
    filled_size: float = 0.0,
) -> OrderIntent:
    """Save a (intent, report) pair and return the intent."""
    intent = _intent(mode=mode)
    report = _report(intent.intent_id, status, live_order_id, filled_size)
    store.save_order_intent(intent)
    store.save_execution_report(report)
    return intent


# ---------------------------------------------------------------------------
# 1–2. Empty result, SUBMITTED returned
# ---------------------------------------------------------------------------

class TestLoadPendingEmpty(unittest.TestCase):

    def test_empty_when_no_data(self) -> None:
        store = _mem_store()
        result = store.load_pending_live_orders()
        self.assertEqual(result, [])


class TestLoadPendingSubmitted(unittest.TestCase):

    def test_returns_submitted_live_order(self) -> None:
        store = _mem_store()
        intent = _save_live_order(store, status=OrderStatus.SUBMITTED)
        result = store.load_pending_live_orders()
        self.assertEqual(len(result), 1)
        returned_intent, live_order_id = result[0]
        self.assertEqual(returned_intent.intent_id, intent.intent_id)
        self.assertEqual(live_order_id, "clob-order-1")

    def test_intent_fields_reconstructed_correctly(self) -> None:
        store = _mem_store()
        intent = _intent(size=75.0, limit_price=0.62, side="BUY")
        store.save_order_intent(intent)
        store.save_execution_report(_report(intent.intent_id, OrderStatus.SUBMITTED))
        result = store.load_pending_live_orders()
        recovered = result[0][0]
        self.assertEqual(recovered.token_id, intent.token_id)
        self.assertAlmostEqual(recovered.size, 75.0)
        self.assertAlmostEqual(recovered.limit_price, 0.62)
        self.assertEqual(recovered.side, "BUY")
        self.assertEqual(recovered.mode, OrderMode.LIVE)


# ---------------------------------------------------------------------------
# 3. PARTIAL returned
# ---------------------------------------------------------------------------

class TestLoadPendingPartial(unittest.TestCase):

    def test_returns_partial_live_order(self) -> None:
        store = _mem_store()
        _save_live_order(
            store,
            status=OrderStatus.PARTIAL,
            live_order_id="clob-partial-7",
            filled_size=20.0,
        )
        result = store.load_pending_live_orders()
        self.assertEqual(len(result), 1)
        _, live_order_id = result[0]
        self.assertEqual(live_order_id, "clob-partial-7")


# ---------------------------------------------------------------------------
# 4–6. Terminal statuses excluded
# ---------------------------------------------------------------------------

class TestLoadPendingTerminalExcluded(unittest.TestCase):

    def test_filled_order_not_returned(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.FILLED, filled_size=50.0)
        self.assertEqual(store.load_pending_live_orders(), [])

    def test_rejected_order_not_returned(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.REJECTED)
        self.assertEqual(store.load_pending_live_orders(), [])

    def test_canceled_order_not_returned(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.CANCELED)
        self.assertEqual(store.load_pending_live_orders(), [])


# ---------------------------------------------------------------------------
# 7. Paper mode excluded
# ---------------------------------------------------------------------------

class TestLoadPendingPaperExcluded(unittest.TestCase):

    def test_paper_mode_order_not_returned(self) -> None:
        store = _mem_store()
        intent = _intent(mode=OrderMode.PAPER)
        # Paper reports have no live_order_id but save with mode=PAPER
        report = _report(intent.intent_id, OrderStatus.SUBMITTED, live_order_id=None)
        store.save_order_intent(intent)
        store.save_execution_report(report)
        self.assertEqual(store.load_pending_live_orders(), [])

    def test_live_mode_returned_but_paper_excluded_when_both_present(self) -> None:
        store = _mem_store()
        # Paper order — should be excluded
        paper_intent = _intent(mode=OrderMode.PAPER)
        store.save_order_intent(paper_intent)
        store.save_execution_report(
            _report(paper_intent.intent_id, OrderStatus.SUBMITTED, live_order_id=None)
        )
        # Live order — should be included
        live_intent = _save_live_order(store, status=OrderStatus.SUBMITTED)
        result = store.load_pending_live_orders()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0].intent_id, live_intent.intent_id)


# ---------------------------------------------------------------------------
# 8. Dry-run sentinel excluded (live_order_id is None)
# ---------------------------------------------------------------------------

class TestLoadPendingDryRunExcluded(unittest.TestCase):

    def test_null_live_order_id_skipped(self) -> None:
        store = _mem_store()
        intent = _intent(mode=OrderMode.LIVE)
        # dry-run report: live_order_id is None
        report = _report(intent.intent_id, OrderStatus.SUBMITTED, live_order_id=None)
        store.save_order_intent(intent)
        store.save_execution_report(report)
        self.assertEqual(store.load_pending_live_orders(), [])


# ---------------------------------------------------------------------------
# 9. Latest report wins: SUBMITTED → PARTIAL → only PARTIAL seen after restart
# ---------------------------------------------------------------------------

class TestLoadPendingLatestReportWins(unittest.TestCase):

    def test_latest_partial_returned_not_earlier_submitted(self) -> None:
        store = _mem_store()
        intent = _intent()
        store.save_order_intent(intent)
        # First report at submission: SUBMITTED
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.SUBMITTED, filled_size=0.0)
        )
        # Second report after first reconcile cycle: PARTIAL
        store.save_execution_report(
            _report(
                intent.intent_id,
                OrderStatus.PARTIAL,
                live_order_id="clob-seq-1",
                filled_size=20.0,
            )
        )
        result = store.load_pending_live_orders()
        self.assertEqual(len(result), 1)
        _, lid = result[0]
        self.assertEqual(lid, "clob-seq-1")

    def test_order_with_multiple_reports_counts_once(self) -> None:
        store = _mem_store()
        intent = _intent()
        store.save_order_intent(intent)
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.SUBMITTED)
        )
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.PARTIAL, filled_size=15.0)
        )
        result = store.load_pending_live_orders()
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# 10. Latest terminal report wins: PARTIAL then FILLED → nothing returned
# ---------------------------------------------------------------------------

class TestLoadPendingLatestTerminalWins(unittest.TestCase):

    def test_later_filled_report_excludes_order(self) -> None:
        store = _mem_store()
        intent = _intent()
        store.save_order_intent(intent)
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.PARTIAL, filled_size=30.0)
        )
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.FILLED, filled_size=50.0)
        )
        self.assertEqual(store.load_pending_live_orders(), [])

    def test_later_rejected_report_excludes_order(self) -> None:
        store = _mem_store()
        intent = _intent()
        store.save_order_intent(intent)
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.SUBMITTED)
        )
        store.save_execution_report(
            _report(intent.intent_id, OrderStatus.REJECTED)
        )
        self.assertEqual(store.load_pending_live_orders(), [])


# ---------------------------------------------------------------------------
# Runner layer helpers
# ---------------------------------------------------------------------------

class _FakeStatusClient:
    """Minimal status-only client for reconciler wiring tests."""

    def __init__(self, size_matched: float = 0.0, size_remaining: float = 50.0) -> None:
        self._size_matched = size_matched
        self._size_remaining = size_remaining
        self.calls: list[str] = []

    def get_order_status(self, order_id: str) -> LiveOrderStatus:
        self.calls.append(order_id)
        return LiveOrderStatus(
            order_id=order_id,
            status="live",
            size_matched=self._size_matched,
            size_remaining=self._size_remaining,
        )


def _live_runner_with_reconciler(
    store: ResearchStore,
    status_client: _FakeStatusClient | None = None,
) -> ResearchRunner:
    """Runner wired for live mode with a real store and real FillReconciler."""
    runner = ResearchRunner()
    runner.store = store
    runner.opportunity_store = MagicMock()
    runner.config.execution.live_enabled = True
    runner.config.execution.dry_run = False

    # Wire a mock LiveBroker so _dispatch_order doesn't fail in live mode
    mock_live = MagicMock(spec=LiveBroker)
    mock_live.submit_limit_order.return_value = ExecutionReport(
        intent_id="x",
        status=OrderStatus.SUBMITTED,
        ts=datetime.now(timezone.utc),
        metadata={"live_order_id": "clob-new-1"},
    )
    runner.live_broker = mock_live

    sc = status_client or _FakeStatusClient()
    runner.fill_reconciler = FillReconciler(sc)  # type: ignore[arg-type]
    return runner


# ---------------------------------------------------------------------------
# 11. No-op when fill_reconciler is None
# ---------------------------------------------------------------------------

class TestRecoverNoOpNoReconciler(unittest.TestCase):

    def test_no_op_when_reconciler_none(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.SUBMITTED)
        runner = ResearchRunner()
        runner.store = store
        runner.opportunity_store = MagicMock()
        runner.config.execution.live_enabled = True
        runner.config.execution.dry_run = False
        # fill_reconciler is None by default — should not raise
        runner._recover_pending_live_orders()
        self.assertIsNone(runner.fill_reconciler)


# ---------------------------------------------------------------------------
# 12. No-op when _effective_live_enabled is False
# ---------------------------------------------------------------------------

class TestRecoverNoOpPaperMode(unittest.TestCase):

    def test_no_op_when_paper_mode(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.SUBMITTED)
        runner = ResearchRunner()
        runner.store = store
        runner.opportunity_store = MagicMock()
        # paper mode defaults
        sc = _FakeStatusClient()
        runner.fill_reconciler = FillReconciler(sc)  # type: ignore[arg-type]
        runner._recover_pending_live_orders()
        # fill_reconciler has nothing registered — paper store orders not loaded
        self.assertEqual(runner.fill_reconciler.pending_count, 0)


# ---------------------------------------------------------------------------
# 13. Recovery registers all pending orders
# ---------------------------------------------------------------------------

class TestRecoverRegistersOrders(unittest.TestCase):

    def test_single_pending_order_registered(self) -> None:
        store = _mem_store()
        _save_live_order(
            store, status=OrderStatus.SUBMITTED, live_order_id="clob-r1"
        )
        runner = _live_runner_with_reconciler(store)
        runner._recover_pending_live_orders()
        self.assertEqual(runner.fill_reconciler.pending_count, 1)
        self.assertIn("clob-r1", runner.fill_reconciler._pending)

    def test_multiple_pending_orders_all_registered(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-m1")
        _save_live_order(store, status=OrderStatus.PARTIAL, live_order_id="clob-m2")
        runner = _live_runner_with_reconciler(store)
        runner._recover_pending_live_orders()
        self.assertEqual(runner.fill_reconciler.pending_count, 2)

    def test_terminal_orders_not_registered(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-keep")
        _save_live_order(store, status=OrderStatus.FILLED, live_order_id="clob-skip")
        runner = _live_runner_with_reconciler(store)
        runner._recover_pending_live_orders()
        self.assertEqual(runner.fill_reconciler.pending_count, 1)
        self.assertIn("clob-keep", runner.fill_reconciler._pending)

    def test_recovery_is_idempotent_on_repeated_calls(self) -> None:
        """Calling _recover_pending_live_orders twice should not double-register."""
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-idem")
        runner = _live_runner_with_reconciler(store)
        runner._recover_pending_live_orders()
        runner._recover_pending_live_orders()  # FillReconciler.register is idempotent
        self.assertEqual(runner.fill_reconciler.pending_count, 1)


# ---------------------------------------------------------------------------
# 14–15. startup_recovery_done flag: exactly one recovery per process lifetime
# ---------------------------------------------------------------------------

class TestStartupRecoveryDoneFlag(unittest.TestCase):

    def test_flag_false_by_default(self) -> None:
        runner = ResearchRunner()
        self.assertFalse(runner._startup_recovery_done)

    def test_flag_set_after_first_run_once(self) -> None:
        runner = ResearchRunner()
        runner.store = _mem_store()
        runner.opportunity_store = MagicMock()
        with patch("src.runtime.runner.fetch_markets", return_value=[]):
            runner.run_once()
        self.assertTrue(runner._startup_recovery_done)

    def test_store_queried_only_once_across_two_run_once_calls(self) -> None:
        """load_pending_live_orders must be called exactly once across multiple cycles."""
        store = _mem_store()
        runner = _live_runner_with_reconciler(store)
        with patch.object(store, "load_pending_live_orders", wraps=store.load_pending_live_orders) as mock_load:
            with patch("src.runtime.runner.fetch_markets", return_value=[]):
                runner.run_once()
                runner.run_once()
        mock_load.assert_called_once()

    def test_second_run_once_does_not_double_register(self) -> None:
        store = _mem_store()
        _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-d1")
        # Status client that always returns resting (never terminal)
        sc = _FakeStatusClient(size_matched=0.0, size_remaining=50.0)
        runner = _live_runner_with_reconciler(store, sc)
        with patch("src.runtime.runner.fetch_markets", return_value=[]):
            runner.run_once()
            count_after_first = runner.fill_reconciler.pending_count
            runner.run_once()
            count_after_second = runner.fill_reconciler.pending_count
        # Order is still pending (not terminal), count should not grow
        self.assertEqual(count_after_first, count_after_second)


# ---------------------------------------------------------------------------
# 16. After recovery + poll, position is visible in paper ledger
# ---------------------------------------------------------------------------

class TestPositionVisibleAfterRecovery(unittest.TestCase):

    def test_filled_order_creates_position_after_restart(self) -> None:
        """Simulate restart: store has SUBMITTED order, CLOB now shows fully matched."""
        store = _mem_store()
        _save_live_order(
            store,
            status=OrderStatus.SUBMITTED,
            live_order_id="clob-pos-1",
        )
        # CLOB says order is now fully matched
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0)

        class _MatchedStatusClient(_FakeStatusClient):
            def get_order_status(self, order_id: str) -> LiveOrderStatus:
                return LiveOrderStatus(
                    order_id=order_id,
                    status="matched",
                    size_matched=50.0,
                    size_remaining=0.0,
                )

        runner = _live_runner_with_reconciler(store, _MatchedStatusClient())
        runner._recover_pending_live_orders()
        runner._poll_live_fills()

        # Position record should now be visible in paper ledger
        self.assertTrue(
            any(r.is_open for r in runner.paper_ledger.position_records.values()),
            "Expected at least one open position in ledger after fill reconciliation",
        )

    def test_partial_fill_creates_open_position(self) -> None:
        store = _mem_store()
        _save_live_order(
            store,
            status=OrderStatus.SUBMITTED,
            live_order_id="clob-pos-2",
        )

        class _PartialStatusClient(_FakeStatusClient):
            def get_order_status(self, order_id: str) -> LiveOrderStatus:
                return LiveOrderStatus(
                    order_id=order_id,
                    status="live",
                    size_matched=25.0,
                    size_remaining=25.0,
                )

        runner = _live_runner_with_reconciler(store, _PartialStatusClient())
        runner._recover_pending_live_orders()
        runner._poll_live_fills()

        # 25 shares applied → position record exists with shares > 0
        self.assertTrue(
            any(r.is_open for r in runner.paper_ledger.position_records.values()),
        )

    def test_resting_order_no_position_yet(self) -> None:
        """Order recovered but CLOB shows 0 fill — position not yet created."""
        store = _mem_store()
        _save_live_order(
            store,
            status=OrderStatus.SUBMITTED,
            live_order_id="clob-pos-3",
        )
        # CLOB: no fill yet
        sc = _FakeStatusClient(size_matched=0.0, size_remaining=50.0)
        runner = _live_runner_with_reconciler(store, sc)
        runner._recover_pending_live_orders()
        runner._poll_live_fills()

        # Order placed in ledger but no fill — position record not yet open
        open_positions = [
            r for r in runner.paper_ledger.position_records.values() if r.is_open
        ]
        self.assertEqual(len(open_positions), 0)


if __name__ == "__main__":
    unittest.main()
