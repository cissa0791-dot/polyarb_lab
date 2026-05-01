from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ArbScanOpportunity:
    status: str
    kind: str
    event_slug: str
    event_title: str | None
    gross_edge: float
    estimated_fee_slippage: float
    executable_edge: float
    required_legs: list[dict[str, Any]]
    resolution_mismatch_risk: str
    estimated_capital_lock_time: str
    confidence_score: float
    decision_trace: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scan_arb_opportunities(
    registry: dict[str, Any],
    *,
    min_edge: float = 0.01,
    max_opportunities: int = 25,
) -> dict[str, Any]:
    """Read-only Polymarket internal arbitrage / relative-value scanner.

    This scanner intentionally does not submit orders. It only flags event-level
    YES baskets where the available top-of-book prices imply a possible
    under/over-pricing across mutually exclusive outcomes.
    """

    min_edge = max(0.0, float(min_edge))
    opportunities: list[ArbScanOpportunity] = []
    events_seen = 0
    markets_seen = 0
    usable_events = 0
    skip_reasons: dict[str, int] = {}

    for event in registry.get("events") or []:
        if not isinstance(event, dict):
            continue
        events_seen += 1
        event_markets = [market for market in event.get("markets") or [] if isinstance(market, dict)]
        markets_seen += len(event_markets)
        usable = [_market_quote(market) for market in event_markets]
        usable = [row for row in usable if row is not None]
        if len(usable) < 2:
            skip_reasons["NOT_ENOUGH_USABLE_MARKETS"] = skip_reasons.get("NOT_ENOUGH_USABLE_MARKETS", 0) + 1
            continue

        usable_events += 1
        event_slug = str(event.get("slug") or "")
        event_title = event.get("title")
        neg_risk = _truthy(event.get("neg_risk")) or _truthy(event.get("enable_neg_risk")) or _truthy(event.get("neg_risk_augmented"))
        structure = _event_structure(event_slug, event_title, usable)
        if structure == "non_exclusive":
            skip_reasons["NON_EXCLUSIVE_EVENT"] = skip_reasons.get("NON_EXCLUSIVE_EVENT", 0) + 1
            continue
        resolution_risk = "low" if neg_risk else "medium"
        base_trace = [
            f"event_markets={len(event_markets)}",
            f"usable_markets={len(usable)}",
            f"neg_risk={neg_risk}",
            f"event_structure={structure}",
        ]

        ask_sum = sum(row["best_ask"] for row in usable)
        bid_sum = sum(row["best_bid"] for row in usable)
        under_edge = 1.0 - ask_sum
        over_edge = bid_sum - 1.0
        fee_slippage = _fee_slippage_estimate(len(usable))

        if under_edge > 0.0 and (neg_risk or structure == "likely_exclusive"):
            executable = under_edge - fee_slippage
            opportunities.append(
                ArbScanOpportunity(
                    status="ARB_CANDIDATE" if executable >= min_edge and neg_risk else "ARB_WATCH",
                    kind="event_yes_basket_under_one",
                    event_slug=event_slug,
                    event_title=event_title if event_title is None else str(event_title),
                    gross_edge=round(under_edge, 6),
                    estimated_fee_slippage=round(fee_slippage, 6),
                    executable_edge=round(executable, 6),
                    required_legs=[
                        {
                            "market_slug": row["market_slug"],
                            "question": row["question"],
                            "token_id": row["token_id"],
                            "side": "BUY",
                            "price": row["best_ask"],
                        }
                        for row in usable
                    ],
                    resolution_mismatch_risk=resolution_risk,
                    estimated_capital_lock_time="until_resolution",
                    confidence_score=_confidence(executable, min_edge, neg_risk),
                    decision_trace=[
                        *base_trace,
                        f"ask_sum={ask_sum:.6f}",
                        f"gross_edge={under_edge:.6f}",
                        f"estimated_fee_slippage={fee_slippage:.6f}",
                    ],
                )
            )
        elif under_edge > 0.0:
            skip_reasons["UNDER_ONE_NOT_EXHAUSTIVE"] = skip_reasons.get("UNDER_ONE_NOT_EXHAUSTIVE", 0) + 1

        if over_edge > 0.0 and neg_risk:
            executable = over_edge - fee_slippage
            opportunities.append(
                ArbScanOpportunity(
                    status="ARB_CANDIDATE" if executable >= min_edge and neg_risk else "ARB_WATCH",
                    kind="event_yes_basket_over_one",
                    event_slug=event_slug,
                    event_title=event_title if event_title is None else str(event_title),
                    gross_edge=round(over_edge, 6),
                    estimated_fee_slippage=round(fee_slippage, 6),
                    executable_edge=round(executable, 6),
                    required_legs=[
                        {
                            "market_slug": row["market_slug"],
                            "question": row["question"],
                            "token_id": row["token_id"],
                            "side": "SELL_OR_SHORT",
                            "price": row["best_bid"],
                        }
                        for row in usable
                    ],
                    resolution_mismatch_risk="high" if not neg_risk else "medium",
                    estimated_capital_lock_time="until_resolution",
                    confidence_score=_confidence(executable, min_edge, neg_risk) * 0.8,
                    decision_trace=[
                        *base_trace,
                        f"bid_sum={bid_sum:.6f}",
                        f"gross_edge={over_edge:.6f}",
                        f"estimated_fee_slippage={fee_slippage:.6f}",
                        "shorting_or_complement_execution_required",
                    ],
                )
            )
        elif over_edge > 0.0:
            skip_reasons["OVER_ONE_NEEDS_SHORT_OR_NEG_RISK"] = skip_reasons.get("OVER_ONE_NEEDS_SHORT_OR_NEG_RISK", 0) + 1

    opportunities.sort(key=lambda item: (item.status == "ARB_CANDIDATE", item.executable_edge), reverse=True)
    rows = [item.to_dict() for item in opportunities[:max(0, int(max_opportunities))]]
    return {
        "events_seen": events_seen,
        "markets_seen": markets_seen,
        "usable_events": usable_events,
        "opportunity_count": len(rows),
        "candidate_count": sum(1 for row in rows if row["status"] == "ARB_CANDIDATE"),
        "watch_count": sum(1 for row in rows if row["status"] == "ARB_WATCH"),
        "skip_reasons": skip_reasons,
        "opportunities": rows,
    }


def _market_quote(market: dict[str, Any]) -> dict[str, Any] | None:
    if not _truthy(market.get("active"), default=True):
        return None
    if _truthy(market.get("closed")):
        return None
    if not _truthy(market.get("enable_orderbook"), default=True):
        return None

    bid = _as_float(market.get("best_bid"))
    ask = _as_float(market.get("best_ask"))
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0 or ask < bid:
        return None
    return {
        "market_slug": str(market.get("slug") or ""),
        "question": market.get("question"),
        "token_id": str(market.get("yes_token_id") or market.get("token_id") or ""),
        "best_bid": round(bid, 6),
        "best_ask": round(ask, 6),
    }


def _event_structure(event_slug: str, event_title: Any, usable_markets: list[dict[str, Any]]) -> str:
    text = " ".join(
        [
            event_slug,
            "" if event_title is None else str(event_title),
            *[str(row.get("market_slug") or "") for row in usable_markets],
            *[str(row.get("question") or "") for row in usable_markets],
        ]
    ).lower()
    non_exclusive_markers = (
        "top 4",
        "top-4",
        "top four",
        "finish in the top",
        "which countries",
        "which cities",
        "which companies",
        "which states",
        "which tokens",
        "which stocks",
        "new trade deal",
        "launch in",
        "will waymo launch",
        "before 2027",
    )
    if any(marker in text for marker in non_exclusive_markers):
        return "non_exclusive"
    likely_exclusive_markers = (
        "winner",
        "who will win",
        "which team will win",
        "which candidate will win",
        "next president",
        "next prime minister",
        "election winner",
        "nominee",
        "champion",
        "champions",
        "mvp",
    )
    if any(marker in text for marker in likely_exclusive_markers):
        return "likely_exclusive"
    return "unknown"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _fee_slippage_estimate(leg_count: int) -> float:
    return round(max(0.0025, 0.0015 * max(1, leg_count)), 6)


def _confidence(edge: float, min_edge: float, neg_risk: bool) -> float:
    if edge <= 0.0:
        return 0.1
    threshold = max(min_edge, 0.000001)
    edge_score = min(0.35, 0.15 * (edge / threshold))
    base = 0.45 if neg_risk else 0.25
    return round(min(0.95, base + edge_score), 6)
