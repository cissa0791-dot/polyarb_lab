from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.core.models import ArbOpportunity, MarketPair, OrderBook
from src.scanner.vwap import buy_cost_from_asks


def scan_yes_no_pair(pair: MarketPair, yes_book: OrderBook, no_book: OrderBook, max_notional: float, total_buffer_cents: float) -> Optional[ArbOpportunity]:
    # Split notional evenly for a conservative paper approximation.
    yes_fill = buy_cost_from_asks(yes_book.asks, max_notional / 2)
    no_fill = buy_cost_from_asks(no_book.asks, max_notional / 2)

    if not yes_fill or not no_fill:
        return None

    pair_cost = yes_fill["vwap"] + no_fill["vwap"]
    edge = 1.0 - pair_cost - total_buffer_cents
    if edge <= 0:
        return None

    shares = min(yes_fill["shares"], no_fill["shares"])
    gross = shares * edge

    return ArbOpportunity(
        kind="single_market",
        name="yes_no_under_1",
        edge_cents=round(edge, 6),
        gross_profit=round(gross, 6),
        est_fill_cost=round(pair_cost * shares, 6),
        est_payout=round(1.0 * shares, 6),
        notional=round(pair_cost * shares, 6),
        details={
            "market_slug": pair.market_slug,
            "question": pair.question,
            "pair_cost": pair_cost,
            "shares": shares,
            "yes_vwap": yes_fill["vwap"],
            "no_vwap": no_fill["vwap"],
        },
        ts=datetime.now(timezone.utc),
    )
