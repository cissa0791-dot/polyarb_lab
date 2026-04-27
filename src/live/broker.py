"""Live order broker — submits real GTC limit orders to the Polymarket CLOB.

Wraps LiveWriteClient to match the broker interface expected by
ResearchRunner: submit_limit_order(intent) → ExecutionReport.

Unlike PaperBroker:
  * no order-book argument is required
  * the limit_price on the intent is passed directly to the CLOB
  * fills are not known at submission time for resting GTC orders;
    the returned report reflects status at the moment of submission only

All LiveClientError exceptions are caught and returned as REJECTED
ExecutionReports so the runner never needs to handle live-specific
exceptions directly.  Poll get_order_status (LiveWriteClient) for
post-submission fill updates — that is the job of L8.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.domain.models import ExecutionReport, OrderIntent, OrderStatus
from src.live.client import LiveClientError, LiveWriteClient


class LiveBroker:
    """Submit live limit orders to the Polymarket CLOB.

    Construct via ResearchRunner.live_broker after building a
    LiveWriteClient from credentials.

    Attributes:
        client: Authenticated LiveWriteClient (dry_run=False for real orders).
    """

    def __init__(self, client: LiveWriteClient) -> None:
        self.client = client

    def submit_limit_order(self, intent: OrderIntent) -> ExecutionReport:
        """Submit intent to the CLOB and return a normalised ExecutionReport.

        The limit_price on the intent is passed as the resting limit price.
        For GTC orders the CLOB usually returns size_matched=0 at submission;
        the status field reflects what the CLOB reports immediately.

        Any LiveClientError is caught and returned as OrderStatus.REJECTED so
        the runner's existing rejection-recording path handles it uniformly.

        Args:
            intent: The order intent to submit.  intent.mode must be LIVE;
                    this is not enforced here but is guaranteed by the
                    runner's _dispatch_order routing gate.

        Returns:
            ExecutionReport with status SUBMITTED (resting), PARTIAL
            (partially matched at submission), FILLED (fully matched
            immediately), or REJECTED (CLOB error).
        """
        ts = datetime.now(timezone.utc)
        start = time.monotonic()
        try:
            result = self.client.submit_order(
                token_id=intent.token_id,
                side=intent.side,
                price=float(intent.limit_price or 0.0),
                size=intent.size,
                neg_risk=bool(intent.metadata.get("neg_risk", False)),
                tick_size=intent.metadata.get("tick_size") or None,
            )
        except LiveClientError as exc:
            return ExecutionReport(
                intent_id=intent.intent_id,
                position_id=intent.position_id,
                status=OrderStatus.REJECTED,
                metadata={"error": str(exc), "live_order_id": None},
                ts=ts,
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        filled = result.size_matched
        if filled >= intent.size - 1e-9:
            status = OrderStatus.FILLED
        elif filled > 1e-9:
            status = OrderStatus.PARTIAL
        else:
            status = OrderStatus.SUBMITTED

        return ExecutionReport(
            intent_id=intent.intent_id,
            position_id=intent.position_id,
            status=status,
            filled_size=round(filled, 6),
            avg_fill_price=(result.avg_price if result.avg_price is not None else float(intent.limit_price or 0.0)) if filled > 1e-9 else None,
            latency_ms=latency_ms,
            metadata={
                "live_order_id": result.order_id,
                "live_status": result.status,
                "requested_size": intent.size,
            },
            ts=ts,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a live resting order by CLOB order ID."""
        try:
            return self.client.cancel_order(order_id)
        except LiveClientError:
            return False
