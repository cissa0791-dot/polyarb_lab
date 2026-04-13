"""Restart-recovery proof — in-memory SQLite only, zero network calls.

Exercises the exact code path that runner._recover_pending_live_orders() and
FillReconciler take when a process restarts with a resting live BUY order:

  1. Store: insert synthetic SUBMITTED live order (intent + report).
  2. load_pending_live_orders() must return exactly 1 entry.
  3. FillReconciler.register() receives the recovered (intent, live_order_id).
  4. reconciler.poll(ledger) with a mock FILLED status must:
       - return 1 CompletedOrder
       - add a position to ledger.position_records with correct shares
  5. print PASS / FAIL.

Cost: zero (no network, no real orders, no writes outside /tmp).
"""

from __future__ import annotations

import sys
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.models import (
    ExecutionReport,
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
)
from src.live.client import LiveOrderStatus
from src.live.reconciler import FillReconciler
from src.paper.ledger import Ledger
from src.storage.event_store import ResearchStore

_GREEN = "\033[92m"
_RED   = "\033[91m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Minimal mock: replaces LiveWriteClient for the reconciler
# ---------------------------------------------------------------------------

@dataclass
class _MockOrderStatus:
    """Returns a fixed FILLED status for any order_id."""
    status: str
    size_matched: float
    size_remaining: float
    order_id: str = "mock"


class _MockLiveClient:
    """Duck-typed substitute for LiveWriteClient — no network."""

    def __init__(self, size_matched: float, size_remaining: float, status: str = "matched") -> None:
        self._size_matched  = size_matched
        self._size_remaining = size_remaining
        self._status        = status
        self.polled_ids: list[str] = []

    def get_order_status(self, order_id: str) -> LiveOrderStatus:
        self.polled_ids.append(order_id)
        return LiveOrderStatus(
            order_id=order_id,
            status=self._status,
            size_matched=self._size_matched,
            size_remaining=self._size_remaining,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAIL_COUNT = 0


def _assert(condition: bool, label: str) -> None:
    global _FAIL_COUNT
    if condition:
        print(f"  {_GREEN}OK{_RESET}  {label}")
    else:
        print(f"  {_RED}FAIL{_RESET} {label}")
        _FAIL_COUNT += 1


# ---------------------------------------------------------------------------
# Main proof
# ---------------------------------------------------------------------------

def run_proof() -> int:
    print(f"\n{_BOLD}Restart Recovery Proof{_RESET}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Setup: in-memory DB only
    # ------------------------------------------------------------------
    store = ResearchStore("sqlite:///:memory:")

    intent_id  = str(uuid.uuid4())
    pos_id     = str(uuid.uuid4())
    live_oid   = "clob-test-" + str(uuid.uuid4())[:8]
    min_size   = 5.0
    ask_price  = 0.01
    ts_now     = datetime.now(timezone.utc)

    intent = OrderIntent(
        intent_id=intent_id,
        candidate_id="proof_candidate",
        mode=OrderMode.LIVE,
        market_slug="proof-market",
        token_id="0x" + "a" * 64,
        position_id=pos_id,
        side="BUY",
        order_type=OrderType.LIMIT,
        size=min_size,
        limit_price=ask_price,
        max_notional_usd=min_size * ask_price * 1.1,
        ts=ts_now,
    )
    store.save_order_intent(intent)

    # Simulates the SUBMITTED report written immediately after dispatch
    submitted_report = ExecutionReport(
        intent_id=intent_id,
        position_id=pos_id,
        status=OrderStatus.SUBMITTED,
        filled_size=0.0,
        metadata={"live_order_id": live_oid},
        ts=ts_now,
    )
    store.save_execution_report(submitted_report)

    # ------------------------------------------------------------------
    # Step 1: load_pending_live_orders()
    # ------------------------------------------------------------------
    print(f"\n[1] load_pending_live_orders()")
    pending = store.load_pending_live_orders()
    _assert(len(pending) == 1, f"exactly 1 pending order returned (got {len(pending)})")

    if not pending:
        print(f"  {_RED}Cannot continue — no pending orders loaded.{_RESET}")
        return 1

    recovered_intent, recovered_live_oid = pending[0]
    _assert(recovered_intent.intent_id == intent_id, "intent_id matches")
    _assert(recovered_intent.position_id == pos_id,  "position_id matches")
    _assert(recovered_live_oid == live_oid,           "live_order_id matches")
    _assert(recovered_intent.size == min_size,        f"size={recovered_intent.size} matches {min_size}")

    # ------------------------------------------------------------------
    # Step 2: register with FillReconciler
    # ------------------------------------------------------------------
    print(f"\n[2] FillReconciler.register() + poll(ledger) — mock FILLED")
    mock_client = _MockLiveClient(
        size_matched=min_size,
        size_remaining=0.0,
        status="matched",
    )
    reconciler = FillReconciler(client=mock_client)  # type: ignore[arg-type]
    reconciler.register(recovered_live_oid, recovered_intent)
    _assert(reconciler.pending_count == 1, "reconciler has 1 pending order after register()")

    # ------------------------------------------------------------------
    # Step 3: poll() with a fresh ledger (simulates process restart)
    # ------------------------------------------------------------------
    ledger = Ledger()  # fresh — cash=10000.0, no positions
    _assert(len(ledger.position_records) == 0, "ledger starts empty (simulates restart)")

    completed = reconciler.poll(ledger)

    _assert(len(completed) == 1,            "poll() returns 1 CompletedOrder")
    _assert(reconciler.pending_count == 0,  "reconciler pending set is empty after terminal")

    if completed:
        c = completed[0]
        _assert(c.live_order_id == live_oid,     "CompletedOrder.live_order_id correct")
        _assert(c.final_size_matched == min_size, f"final_size_matched={c.final_size_matched} == {min_size}")
        _assert(c.final_clob_status == "matched", f"final_clob_status={c.final_clob_status!r}")

    # ------------------------------------------------------------------
    # Step 4: ledger position rebuilt correctly
    # ------------------------------------------------------------------
    print(f"\n[3] Ledger state after fill replay")
    _assert(len(ledger.position_records) == 1, "position_records has 1 entry")

    if ledger.position_records:
        pos = next(iter(ledger.position_records.values()))
        _assert(pos.position_id == pos_id,              "position_id matches")
        _assert(abs(pos.total_entry_shares - min_size) < 1e-9,
                f"total_entry_shares={pos.total_entry_shares} == {min_size}")
        _assert(abs(pos.avg_entry_price - ask_price) < 1e-9,
                f"avg_entry_price={pos.avg_entry_price} == {ask_price}")
        _assert(pos.is_open,                             "position.is_open == True")

    # ------------------------------------------------------------------
    # Step 5: terminal report would exclude this from next recovery
    # ------------------------------------------------------------------
    print(f"\n[4] Terminal report excludes order from next restart recovery")
    # Simulate what _poll_live_fills does: write a terminal FILLED report
    terminal_report = ExecutionReport(
        intent_id=intent_id,
        position_id=pos_id,
        status=OrderStatus.FILLED,
        filled_size=min_size,
        avg_fill_price=ask_price,
        metadata={"live_order_id": live_oid, "terminal": True},
        ts=datetime.now(timezone.utc),
    )
    store.save_execution_report(terminal_report)

    # Now load_pending_live_orders should return 0 (last report is FILLED)
    pending_after = store.load_pending_live_orders()
    _assert(len(pending_after) == 0,
            f"after terminal report: load_pending returns 0 (got {len(pending_after)})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    if _FAIL_COUNT == 0:
        print(f"{_GREEN}{_BOLD}RESTART_RECOVERY_PROOF: PASS{_RESET}")
    else:
        print(f"{_RED}{_BOLD}RESTART_RECOVERY_PROOF: {_FAIL_COUNT} ASSERTION(S) FAILED{_RESET}")
    print("=" * 60)
    return 0 if _FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(run_proof())
