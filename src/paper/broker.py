from __future__ import annotations

from datetime import datetime, timezone

from src.domain.models import ExecutionReport, OrderIntent, OrderStatus
from src.paper.ledger import Ledger


class PaperBroker:
    def __init__(self, ledger: Ledger, fee_rate: float = 0.0, fixed_fee_usd: float = 0.0, auto_cancel_unfilled: bool = True):
        self.ledger = ledger
        self.fee_rate = fee_rate
        self.fixed_fee_usd = fixed_fee_usd
        self.auto_cancel_unfilled = auto_cancel_unfilled

    def submit_limit_order(self, intent: OrderIntent, book) -> ExecutionReport:
        created_ts = datetime.now(timezone.utc)
        accepted = self.ledger.place_limit_order(
            order_id=intent.intent_id,
            symbol=intent.token_id,
            market_slug=intent.market_slug,
            side=intent.side,
            shares=intent.size,
            limit_price=float(intent.limit_price or 0.0),
            ts=created_ts.isoformat(),
            candidate_id=intent.candidate_id,
            position_id=intent.position_id,
        )
        if not accepted:
            return ExecutionReport(
                intent_id=intent.intent_id,
                position_id=intent.position_id,
                status=OrderStatus.REJECTED,
                metadata={"reason": "PAPER_ORDER_REJECTED"},
                ts=created_ts,
            )

        if intent.side.upper() == "BUY":
            levels = getattr(book, "asks", [])
            price_filter = lambda px: px <= float(intent.limit_price or 0.0)
        else:
            levels = getattr(book, "bids", [])
            price_filter = lambda px: px >= float(intent.limit_price or 0.0)

        filled = 0.0
        notional = 0.0
        fee_paid = 0.0

        for level in levels:
            price = float(level.price)
            if not price_filter(price):
                continue
            remaining = intent.size - filled
            if remaining <= 1e-9:
                break
            take = min(remaining, float(level.size))
            if take <= 1e-9:
                continue

            fee = (take * price * self.fee_rate) + (self.fixed_fee_usd if filled <= 1e-9 else 0.0)
            if not self.ledger.apply_fill(intent.intent_id, take, price, fee_usd=fee, ts=created_ts.isoformat()):
                break

            filled += take
            notional += take * price
            fee_paid += fee

        order = self.ledger.orders[intent.intent_id]
        report_status = OrderStatus.FILLED if filled >= intent.size - 1e-9 else OrderStatus.SUBMITTED
        canceled_remainder = False
        if order.remaining_shares > 1e-9 and self.auto_cancel_unfilled:
            self.ledger.cancel_order(intent.intent_id)
            if filled <= 1e-9:
                report_status = OrderStatus.CANCELED
            elif filled < intent.size:
                report_status = OrderStatus.PARTIAL
            canceled_remainder = True

        avg_fill_price = (notional / filled) if filled > 1e-9 else None
        return ExecutionReport(
            intent_id=intent.intent_id,
            position_id=order.position_id,
            status=report_status,
            filled_size=round(filled, 6),
            avg_fill_price=round(avg_fill_price, 6) if avg_fill_price is not None else None,
            fee_paid_usd=round(fee_paid, 6),
            latency_ms=0,
            metadata={"requested_size": intent.size, "canceled_remainder": canceled_remainder},
            ts=created_ts,
        )

    def cancel_order(self, intent_id: str) -> bool:
        return self.ledger.cancel_order(intent_id)
