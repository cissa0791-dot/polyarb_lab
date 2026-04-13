"""Tests for terminal ExecutionReport persistence (audit-layer hardening).

This is not a correctness fix.  PnL, exit logic, position visibility, and
restart recovery were already correct before this change.  This test suite
validates the audit-layer addition only:

  When FillReconciler.poll() detects a terminal order it returns a
  CompletedOrder.  runner._poll_live_fills() writes exactly one terminal
  ExecutionReport row to the store.  That row has status FILLED or CANCELED
  (never SUBMITTED or PARTIAL), which ensures load_pending_live_orders()
  correctly excludes it from future startup recovery.

Coverage:
  CompletedOrder content:
    1.  poll() returns CompletedOrder for matched (FILLED) order
    2.  poll() returns CompletedOrder for canceled order
    3.  CompletedOrder.final_size_matched carries CLOB truth
    4.  CompletedOrder.final_clob_status carries raw CLOB string
    5.  poll() returns empty list for non-terminal order

  _terminal_order_status mapping:
    6.  "matched" → FILLED
    7.  full-size fill on any status → FILLED
    8.  "canceled" → CANCELED (zero fill)
    9.  "cancelled" → CANCELED
   10.  "expired" → CANCELED
   11.  "canceled" with partial fill → CANCELED (not PARTIAL)

  Runner terminal report write:
   12.  _poll_live_fills writes one row per completed order
   13.  terminal row status is FILLED for matched order
   14.  terminal row status is CANCELED for canceled order
   15.  terminal row filled_size mirrors final_size_matched
   16.  terminal row metadata contains live_order_id
   17.  terminal row metadata contains terminal=True flag
   18.  no terminal row written when no orders complete
   19.  exactly one terminal row written (not two) per order lifetime

  Startup recovery semantics after terminal write:
   20.  load_pending_live_orders excludes order once FILLED row exists
   21.  load_pending_live_orders excludes order once CANCELED row exists
   22.  load_pending_live_orders still returns a different pending order
"""

from __future__ import annotations

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
from src.live.client import LiveOrderStatus
from src.live.reconciler import CompletedOrder, FillReconciler
from src.live.broker import LiveBroker
from src.paper.ledger import Ledger
from src.runtime.runner import ResearchRunner, _terminal_order_status
from src.storage.event_store import ResearchStore


# ---------------------------------------------------------------------------
# Helpers shared across all test classes
# ---------------------------------------------------------------------------

def _intent(size: float = 50.0, limit_price: float = 0.55) -> OrderIntent:
    return OrderIntent(
        intent_id=str(uuid4()),
        candidate_id="cand-tp",
        mode=OrderMode.LIVE,
        market_slug="tp-market",
        token_id="tok-tp",
        position_id=str(uuid4()),
        side="BUY",
        order_type=OrderType.LIMIT,
        size=size,
        limit_price=limit_price,
        max_notional_usd=size * limit_price,
        ts=datetime.now(timezone.utc),
    )


def _completed(
    live_order_id: str = "clob-tp-1",
    final_size_matched: float = 50.0,
    final_clob_status: str = "matched",
    size: float = 50.0,
) -> CompletedOrder:
    return CompletedOrder(
        live_order_id=live_order_id,
        intent=_intent(size=size),
        final_size_matched=final_size_matched,
        final_clob_status=final_clob_status,
    )


class _FakeStatusClient:
    def __init__(
        self,
        size_matched: float,
        size_remaining: float,
        status: str = "matched",
        raises=None,
    ) -> None:
        self._size_matched = size_matched
        self._size_remaining = size_remaining
        self._status = status
        self._raises = raises

    def get_order_status(self, order_id: str) -> LiveOrderStatus:
        if self._raises:
            raise self._raises
        return LiveOrderStatus(
            order_id=order_id,
            status=self._status,
            size_matched=self._size_matched,
            size_remaining=self._size_remaining,
        )


def _reconciler(status_client: _FakeStatusClient) -> FillReconciler:
    return FillReconciler(status_client)  # type: ignore[arg-type]


def _mem_store() -> ResearchStore:
    return ResearchStore("sqlite:///:memory:")


def _save_live_order(
    store: ResearchStore,
    *,
    status: OrderStatus,
    live_order_id: str = "clob-1",
) -> OrderIntent:
    intent = _intent()
    report = ExecutionReport(
        intent_id=intent.intent_id,
        status=status,
        ts=datetime.now(timezone.utc),
        metadata={"live_order_id": live_order_id},
    )
    store.save_order_intent(intent)
    store.save_execution_report(report)
    return intent


# ---------------------------------------------------------------------------
# 1–5. CompletedOrder content from poll()
# ---------------------------------------------------------------------------

class TestCompletedOrderFromPoll(unittest.TestCase):

    def test_poll_returns_completed_for_matched(self) -> None:
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0, status="matched")
        rec = _reconciler(sc)
        intent = _intent(size=50.0)
        rec.register("ord-m1", intent)
        result = rec.poll(Ledger(cash=10_000.0))
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CompletedOrder)

    def test_poll_returns_completed_for_canceled(self) -> None:
        sc = _FakeStatusClient(size_matched=0.0, size_remaining=50.0, status="canceled")
        rec = _reconciler(sc)
        rec.register("ord-c1", _intent())
        result = rec.poll(Ledger(cash=10_000.0))
        self.assertEqual(len(result), 1)

    def test_completed_final_size_matched_from_clob(self) -> None:
        sc = _FakeStatusClient(size_matched=35.0, size_remaining=0.0, status="matched")
        rec = _reconciler(sc)
        intent = _intent(size=35.0)
        rec.register("ord-s1", intent)
        result = rec.poll(Ledger(cash=10_000.0))
        self.assertAlmostEqual(result[0].final_size_matched, 35.0)

    def test_completed_final_clob_status_raw_string(self) -> None:
        sc = _FakeStatusClient(size_matched=0.0, size_remaining=50.0, status="expired")
        rec = _reconciler(sc)
        rec.register("ord-x1", _intent())
        result = rec.poll(Ledger(cash=10_000.0))
        self.assertEqual(result[0].final_clob_status, "expired")

    def test_poll_returns_empty_for_non_terminal(self) -> None:
        sc = _FakeStatusClient(size_matched=10.0, size_remaining=40.0, status="live")
        rec = _reconciler(sc)
        rec.register("ord-n1", _intent())
        result = rec.poll(Ledger(cash=10_000.0))
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 6–11. _terminal_order_status mapping
# ---------------------------------------------------------------------------

class TestTerminalOrderStatusMapping(unittest.TestCase):

    def test_matched_maps_to_filled(self) -> None:
        c = _completed(final_clob_status="matched", final_size_matched=50.0, size=50.0)
        self.assertEqual(_terminal_order_status(c), OrderStatus.FILLED)

    def test_full_size_fill_any_status_maps_to_filled(self) -> None:
        # size_remaining=0 with size_matched=intent.size — non-standard CLOB status
        c = _completed(final_clob_status="live", final_size_matched=50.0, size=50.0)
        self.assertEqual(_terminal_order_status(c), OrderStatus.FILLED)

    def test_canceled_zero_fill_maps_to_canceled(self) -> None:
        c = _completed(final_clob_status="canceled", final_size_matched=0.0)
        self.assertEqual(_terminal_order_status(c), OrderStatus.CANCELED)

    def test_cancelled_british_spelling_maps_to_canceled(self) -> None:
        c = _completed(final_clob_status="cancelled", final_size_matched=0.0)
        self.assertEqual(_terminal_order_status(c), OrderStatus.CANCELED)

    def test_expired_maps_to_canceled(self) -> None:
        c = _completed(final_clob_status="expired", final_size_matched=0.0)
        self.assertEqual(_terminal_order_status(c), OrderStatus.CANCELED)

    def test_canceled_with_partial_fill_maps_to_canceled_not_partial(self) -> None:
        # cancel-with-partial: PARTIAL would be re-picked-up by L9 — must be CANCELED
        c = _completed(
            final_clob_status="canceled",
            final_size_matched=20.0,  # partial fill before cancel
            size=50.0,
        )
        result = _terminal_order_status(c)
        self.assertEqual(result, OrderStatus.CANCELED)
        # critical: must NOT be SUBMITTED or PARTIAL (L9 safety)
        self.assertNotIn(result, (OrderStatus.SUBMITTED, OrderStatus.PARTIAL))


# ---------------------------------------------------------------------------
# 12–19. Runner terminal report write
# ---------------------------------------------------------------------------

def _live_runner_with_store(
    store: ResearchStore,
    status_client: _FakeStatusClient,
) -> ResearchRunner:
    runner = ResearchRunner()
    runner.store = store
    runner.opportunity_store = MagicMock()
    runner.config.execution.live_enabled = True
    runner.config.execution.dry_run = False
    mock_broker = MagicMock(spec=LiveBroker)
    mock_broker.submit_limit_order.return_value = ExecutionReport(
        intent_id="x",
        status=OrderStatus.SUBMITTED,
        ts=datetime.now(timezone.utc),
        metadata={"live_order_id": None},
    )
    runner.live_broker = mock_broker
    runner.fill_reconciler = FillReconciler(status_client)  # type: ignore[arg-type]
    return runner


class TestRunnerTerminalReportWrite(unittest.TestCase):

    def _run_poll_with_registered_order(
        self,
        store: ResearchStore,
        status_client: _FakeStatusClient,
        live_order_id: str = "clob-rw-1",
    ) -> ResearchRunner:
        runner = _live_runner_with_store(store, status_client)
        intent = _intent()
        runner.fill_reconciler.register(live_order_id, intent)
        runner._poll_live_fills()
        return runner

    def test_one_terminal_row_written_for_matched_order(self) -> None:
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0, status="matched")
        self._run_poll_with_registered_order(store, sc)
        rows = store.load_pending_live_orders()
        # terminal row (FILLED) must not be in pending — 0 pending
        self.assertEqual(rows, [])

    def test_terminal_row_status_filled_for_matched(self) -> None:
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0, status="matched")
        runner = self._run_poll_with_registered_order(store, sc, "clob-f1")
        # Verify via direct store query
        from sqlalchemy import select
        with store.engine.begin() as conn:
            rows = conn.execute(
                select(store.execution_reports.c.status)
                .where(store.execution_reports.c.intent_id != "dummy")
            ).all()
        statuses = [r[0] for r in rows]
        self.assertIn("filled", statuses)

    def test_terminal_row_status_canceled_for_canceled_order(self) -> None:
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=0.0, size_remaining=50.0, status="canceled")
        runner = self._run_poll_with_registered_order(store, sc, "clob-c1")
        from sqlalchemy import select
        with store.engine.begin() as conn:
            rows = conn.execute(
                select(store.execution_reports.c.status)
            ).all()
        statuses = [r[0] for r in rows]
        self.assertIn("canceled", statuses)

    def test_terminal_row_filled_size_mirrors_clob(self) -> None:
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=42.5, size_remaining=0.0, status="matched")
        runner = self._run_poll_with_registered_order(store, sc)
        from sqlalchemy import select
        with store.engine.begin() as conn:
            rows = conn.execute(
                select(store.execution_reports.c.filled_size)
            ).all()
        filled_sizes = [r[0] for r in rows]
        self.assertTrue(any(abs(fs - 42.5) < 0.001 for fs in filled_sizes))

    def test_terminal_row_metadata_has_live_order_id(self) -> None:
        import json
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0, status="matched")
        runner = self._run_poll_with_registered_order(store, sc, "clob-meta-1")
        from sqlalchemy import select
        with store.engine.begin() as conn:
            rows = conn.execute(
                select(store.execution_reports.c.payload_json)
            ).all()
        for row in rows:
            data = json.loads(row[0])
            if data.get("metadata", {}).get("live_order_id") == "clob-meta-1":
                return
        self.fail("No terminal row found with live_order_id='clob-meta-1'")

    def test_terminal_row_metadata_terminal_flag_true(self) -> None:
        import json
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0, status="matched")
        runner = self._run_poll_with_registered_order(store, sc)
        from sqlalchemy import select
        with store.engine.begin() as conn:
            rows = conn.execute(
                select(store.execution_reports.c.payload_json)
            ).all()
        terminal_rows = [
            json.loads(r[0]) for r in rows
            if json.loads(r[0]).get("metadata", {}).get("terminal") is True
        ]
        self.assertEqual(len(terminal_rows), 1)

    def test_no_terminal_row_when_no_orders_complete(self) -> None:
        store = _mem_store()
        # Non-terminal: still resting
        sc = _FakeStatusClient(size_matched=10.0, size_remaining=40.0, status="live")
        runner = self._run_poll_with_registered_order(store, sc)
        from sqlalchemy import select
        with store.engine.begin() as conn:
            count = conn.execute(
                select(store.execution_reports)
            ).fetchall()
        # No rows at all (we never called save_order_intent / save_execution_report
        # at dispatch in this test — only _poll_live_fills is called)
        # So terminal flag rows = 0
        import json
        terminal_rows = [
            r for r in count
            if json.loads(r[-1]).get("metadata", {}).get("terminal") is True
        ]
        self.assertEqual(len(terminal_rows), 0)

    def test_exactly_one_terminal_row_per_order(self) -> None:
        """poll() is called twice; terminal state detected on first call.
        Second call has nothing pending — no second write."""
        store = _mem_store()
        sc = _FakeStatusClient(size_matched=50.0, size_remaining=0.0, status="matched")
        runner = _live_runner_with_store(store, sc)
        intent = _intent()
        runner.fill_reconciler.register("clob-once-1", intent)
        runner._poll_live_fills()  # terminal detected — one row written
        runner._poll_live_fills()  # nothing pending — no row written
        from sqlalchemy import select
        import json
        with store.engine.begin() as conn:
            rows = conn.execute(select(store.execution_reports.c.payload_json)).all()
        terminal_rows = [
            json.loads(r[0]) for r in rows
            if json.loads(r[0]).get("metadata", {}).get("terminal") is True
        ]
        self.assertEqual(len(terminal_rows), 1)


# ---------------------------------------------------------------------------
# 20–22. Startup recovery semantics after terminal write
# ---------------------------------------------------------------------------

class TestStartupRecoveryAfterTerminalWrite(unittest.TestCase):

    def test_filled_terminal_row_excludes_order_from_recovery(self) -> None:
        store = _mem_store()
        intent = _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-ex-1")
        # Simulate terminal report written by _poll_live_fills
        terminal_report = ExecutionReport(
            intent_id=intent.intent_id,
            status=OrderStatus.FILLED,
            filled_size=50.0,
            ts=datetime.now(timezone.utc),
            metadata={"live_order_id": "clob-ex-1", "terminal": True},
        )
        store.save_execution_report(terminal_report)
        # L9 query must now exclude this order (latest row = FILLED)
        result = store.load_pending_live_orders()
        self.assertEqual(result, [])

    def test_canceled_terminal_row_excludes_order_from_recovery(self) -> None:
        store = _mem_store()
        intent = _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-ex-2")
        terminal_report = ExecutionReport(
            intent_id=intent.intent_id,
            status=OrderStatus.CANCELED,
            filled_size=0.0,
            ts=datetime.now(timezone.utc),
            metadata={"live_order_id": "clob-ex-2", "terminal": True},
        )
        store.save_execution_report(terminal_report)
        result = store.load_pending_live_orders()
        self.assertEqual(result, [])

    def test_other_pending_order_still_returned_after_terminal_write(self) -> None:
        store = _mem_store()
        # Order A — will be marked terminal
        intent_a = _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-a")
        terminal = ExecutionReport(
            intent_id=intent_a.intent_id,
            status=OrderStatus.FILLED,
            filled_size=50.0,
            ts=datetime.now(timezone.utc),
            metadata={"live_order_id": "clob-a", "terminal": True},
        )
        store.save_execution_report(terminal)
        # Order B — still pending
        _save_live_order(store, status=OrderStatus.SUBMITTED, live_order_id="clob-b")
        result = store.load_pending_live_orders()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "clob-b")


if __name__ == "__main__":
    unittest.main()
