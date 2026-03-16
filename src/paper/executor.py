from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.domain.models import OrderIntent, OrderMode, OrderType
from src.paper.broker import PaperBroker


def simulate_pair_buy(ledger, yes_symbol: str, no_symbol: str, yes_price: float, no_price: float, shares: float) -> bool:
    broker = PaperBroker(ledger)
    ts = datetime.now(timezone.utc)

    yes_intent = OrderIntent(
        intent_id=str(uuid4()),
        candidate_id="legacy-pair",
        mode=OrderMode.PAPER,
        market_slug=yes_symbol,
        token_id=yes_symbol,
        side="BUY",
        order_type=OrderType.LIMIT,
        size=shares,
        limit_price=yes_price,
        max_notional_usd=shares * yes_price,
        ts=ts,
    )
    no_intent = OrderIntent(
        intent_id=str(uuid4()),
        candidate_id="legacy-pair",
        mode=OrderMode.PAPER,
        market_slug=no_symbol,
        token_id=no_symbol,
        side="BUY",
        order_type=OrderType.LIMIT,
        size=shares,
        limit_price=no_price,
        max_notional_usd=shares * no_price,
        ts=ts,
    )

    yes_report = broker.submit_limit_order(yes_intent, type("Book", (), {"asks": [type("Level", (), {"price": yes_price, "size": shares})()]})())
    no_report = broker.submit_limit_order(no_intent, type("Book", (), {"asks": [type("Level", (), {"price": no_price, "size": shares})()]})())
    return yes_report.status.value in {"filled", "partial"} and no_report.status.value in {"filled", "partial"}
