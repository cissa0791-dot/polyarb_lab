from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.core.models import ArbOpportunity, OrderBook


def scan_leq_constraint(name: str, lhs_book: OrderBook, rhs_book: OrderBook, lhs_slug: str, rhs_slug: str, total_buffer_cents: float) -> Optional[ArbOpportunity]:
    """
    Detect violation of P(lhs) <= P(rhs).
    This is research-only detection, not executable trading logic.
    """
    lhs_ask = float(lhs_book.asks[0].price) if lhs_book.asks else None
    rhs_ask = float(rhs_book.asks[0].price) if rhs_book.asks else None

    if lhs_ask is None or rhs_ask is None:
        return None

    violation = lhs_ask - rhs_ask
    if violation <= total_buffer_cents:
        return None

    return ArbOpportunity(
        kind="cross_market",
        name=name,
        edge_cents=round(violation, 6),
        gross_profit=0.0,
        est_fill_cost=0.0,
        est_payout=0.0,
        notional=0.0,
        details={
            "relation": "P(lhs) <= P(rhs)",
            "lhs_slug": lhs_slug,
            "rhs_slug": rhs_slug,
            "lhs_price": lhs_ask,
            "rhs_price": rhs_ask,
        },
        ts=datetime.now(timezone.utc),
    )
