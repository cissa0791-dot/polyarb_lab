"""Async fill reconciler for live-submitted orders (L8).

After a live order is submitted, its CLOB fill status is polled once per
run_once() cycle.  Any fill delta since the previous poll is mirrored into
the paper ledger so that:

  - paper_ledger.position_records reflects live positions, making them
    visible to _manage_open_positions and exit evaluation.
  - fill/PnL accounting remains consistent with the paper simulation.

Design:
  register(live_order_id, intent)   — called once after successful dispatch
  poll(ledger)                      — called at the top of each run_once()

Idempotency: only the *delta* (new_size_matched - size_applied) is ever
applied to the ledger, so repeated polls with the same status are no-ops.

Terminal detection: orders are removed from the pending set when the CLOB
status string is in _TERMINAL_STATUSES *or* size_remaining == 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.models import OrderIntent
from src.live.client import LiveClientError, LiveWriteClient
from src.monitoring.logger import get_logger
from src.paper.ledger import Ledger

# CLOB status strings that mean no further fills will arrive.
_TERMINAL_STATUSES = frozenset({"matched", "canceled", "cancelled", "expired"})


# ---------------------------------------------------------------------------
# Public result type for completed orders
# ---------------------------------------------------------------------------

@dataclass
class CompletedOrder:
    """Returned by poll() for each order that reached terminal state.

    Runner uses this to write exactly one terminal ExecutionReport row to the
    store, closing the audit gap where async fills had no terminal row.

    Attributes:
        live_order_id:      CLOB order ID.
        intent:             Original OrderIntent as submitted.
        final_size_matched: Shares matched at terminal time (from CLOB).
        final_clob_status:  Raw CLOB status string at terminal detection.
        final_avg_price:    Actual average matched price from CLOB at terminal
                            time; None if the order had no fills or the field
                            was absent in the CLOB response.
    """
    live_order_id: str
    intent: OrderIntent
    final_size_matched: float
    final_clob_status: str
    final_avg_price: float | None = None


# ---------------------------------------------------------------------------
# Internal state per pending order
# ---------------------------------------------------------------------------

@dataclass
class _PendingOrder:
    live_order_id: str
    intent: OrderIntent
    size_applied: float = 0.0       # shares already mirrored into ledger
    ledger_placed: bool = False     # True once place_limit_order succeeded


# ---------------------------------------------------------------------------
# Public reconciler
# ---------------------------------------------------------------------------

class FillReconciler:
    """Polls pending live orders and mirrors fill deltas into the paper ledger.

    Typical lifecycle::

        # 1. After live dispatch:
        reconciler.register(report.metadata["live_order_id"], intent)

        # 2. At the start of each run_once() cycle:
        reconciler.poll(runner.paper_ledger)

    The reconciler is safe to instantiate regardless of mode.  poll() is a
    no-op when there are no pending orders.
    """

    def __init__(self, client: LiveWriteClient) -> None:
        self._client = client
        self._pending: dict[str, _PendingOrder] = {}
        self.logger = get_logger("polyarb.live.reconciler")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, live_order_id: str, intent: OrderIntent) -> None:
        """Register a newly submitted live order for fill tracking.

        Idempotent: registering the same order_id twice is a no-op.
        """
        if live_order_id in self._pending:
            return
        self._pending[live_order_id] = _PendingOrder(
            live_order_id=live_order_id,
            intent=intent,
        )

    def unregister(self, live_order_id: str) -> None:
        """Drop a pending order from tracking after an external cancel."""
        self._pending.pop(live_order_id, None)

    def snapshot_pending(self) -> list[tuple[str, OrderIntent, float, bool]]:
        """Return a stable snapshot of pending orders for refresh/cancel checks."""
        return [
            (pending.live_order_id, pending.intent, pending.size_applied, pending.ledger_placed)
            for pending in self._pending.values()
        ]

    def poll(self, ledger: Ledger) -> list[CompletedOrder]:
        """Fetch fill status for all pending orders and mirror deltas into ledger.

        Returns a list of CompletedOrder for every order that reached terminal
        state during this call.  The caller (runner._poll_live_fills) writes
        exactly one terminal ExecutionReport row per CompletedOrder.

        Empty list when no orders completed or no orders are pending.
        """
        if not self._pending:
            return []

        completed: list[CompletedOrder] = []
        for order_id, pending in list(self._pending.items()):
            result = self._reconcile_one(order_id, pending, ledger)
            if result is not None:
                completed.append(result)

        for c in completed:
            del self._pending[c.live_order_id]

        return completed

    @property
    def pending_count(self) -> int:
        """Number of live orders still awaiting complete reconciliation."""
        return len(self._pending)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reconcile_one(
        self, order_id: str, pending: _PendingOrder, ledger: Ledger
    ) -> CompletedOrder | None:
        """Reconcile one pending order.

        Returns CompletedOrder when terminal state is detected (order done),
        None when the order is still pending.
        """
        intent = pending.intent

        try:
            status = self._client.get_order_status(order_id)
        except LiveClientError as exc:
            self.logger.warning(
                "reconcile poll failed",
                extra={"payload": {"order_id": order_id, "error": str(exc)}},
            )
            return None

        # Mirror the order into the ledger before applying fill deltas.
        if not pending.ledger_placed:
            placed = ledger.place_limit_order(
                order_id=order_id,
                symbol=intent.token_id,
                market_slug=intent.market_slug,
                side=intent.side,
                shares=intent.size,
                limit_price=float(intent.limit_price or 0.0),
                ts=intent.ts.isoformat(),
                candidate_id=intent.candidate_id,
                position_id=intent.position_id,
            )
            if placed:
                pending.ledger_placed = True
            else:
                self.logger.warning(
                    "place_limit_order failed for live order",
                    extra={"payload": {"order_id": order_id}},
                )

        # Apply fill delta (idempotent: only new shares since last poll).
        delta = status.size_matched - pending.size_applied
        if delta > 1e-9 and pending.ledger_placed:
            fill_price = status.avg_price if status.avg_price is not None else float(intent.limit_price or 0.0)
            applied = ledger.apply_fill(
                order_id=order_id,
                shares=delta,
                price=fill_price,
            )
            if applied:
                pending.size_applied += delta
            else:
                self.logger.warning(
                    "apply_fill failed for live order delta",
                    extra={"payload": {"order_id": order_id, "delta": delta}},
                )

        is_terminal = status.status.lower() in _TERMINAL_STATUSES
        is_fully_filled = status.size_remaining <= 1e-9 and status.size_matched > 0.0
        if is_terminal or is_fully_filled:
            return CompletedOrder(
                live_order_id=order_id,
                intent=intent,
                final_size_matched=status.size_matched,
                final_clob_status=status.status,
                final_avg_price=status.avg_price,
            )
        return None
