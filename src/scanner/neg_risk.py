from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.core.models import ArbOpportunity


NEG_RISK_EVENT_INELIGIBLE = "NEG_RISK_EVENT_INELIGIBLE"
NEG_RISK_NO_NAMED_MARKETS = "NEG_RISK_NO_NAMED_MARKETS"
NEG_RISK_MISSING_YES_TOKEN = "NEG_RISK_MISSING_YES_TOKEN"
NEG_RISK_EMPTY_BIDS = "NEG_RISK_EMPTY_BIDS"
NEG_RISK_INVALID_ORDERBOOK = "NEG_RISK_INVALID_ORDERBOOK"
NEG_RISK_BASKET_EDGE_NON_POSITIVE = "NEG_RISK_BASKET_EDGE_NON_POSITIVE"


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _market_has_placeholder_title(market: dict[str, Any]) -> bool:
    text = " ".join(
        str(market.get(field) or "")
        for field in ("groupItemTitle", "question", "slug")
    ).strip().lower()
    return any(token in text for token in ("other", "placeholder"))


def _yes_token_id(market: dict[str, Any]) -> str | None:
    outcomes = _parse_json_list(market.get("outcomes"))
    token_ids = _parse_json_list(market.get("clobTokenIds"))
    if len(outcomes) != 2 or len(token_ids) != 2:
        return None
    for outcome, token_id in zip(outcomes, token_ids):
        if str(outcome).strip().upper() == "YES":
            return str(token_id)
    return None


def _best_bid(book) -> float | None:
    bids = list(getattr(book, "bids", []) or [])
    if not bids:
        return None
    level = bids[0]
    price = float(level.price)
    size = float(level.size)
    if price <= 0 or size <= 0:
        return None
    return price


def _best_bid_size(book) -> float | None:
    bids = list(getattr(book, "bids", []) or [])
    if not bids:
        return None
    level = bids[0]
    price = float(level.price)
    size = float(level.size)
    if price <= 0 or size <= 0:
        return None
    return size


def build_eligible_neg_risk_event_groups(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for event in events:
        if not bool(event.get("negRisk")) or not bool(event.get("enableNegRisk")):
            continue
        if bool(event.get("negRiskAugmented")):
            continue

        eligible_markets: list[dict[str, Any]] = []
        for market in list(event.get("markets") or []):
            if not bool(market.get("negRisk")):
                continue
            if bool(market.get("negRiskOther")):
                continue
            if bool(market.get("feesEnabled")):
                continue
            if not bool(market.get("enableOrderBook")):
                continue
            if _market_has_placeholder_title(market):
                continue
            yes_token_id = _yes_token_id(market)
            if yes_token_id is None:
                continue
            eligible_markets.append(
                {
                    "slug": str(market.get("slug") or ""),
                    "question": market.get("question"),
                    "group_item_title": market.get("groupItemTitle"),
                    "yes_token_id": yes_token_id,
                    "best_bid": market.get("bestBid"),
                    "best_ask": market.get("bestAsk"),
                }
            )

        if not eligible_markets:
            continue

        groups.append(
            {
                "event_id": str(event.get("id") or ""),
                "event_slug": str(event.get("slug") or ""),
                "event_title": event.get("title"),
                "neg_risk_market_id": event.get("negRiskMarketID"),
                "markets": eligible_markets,
            }
        )
    return groups


def analyze_neg_risk_rebalancing_event(
    event_group: dict[str, Any],
    books_by_token: dict[str, Any],
    max_notional: float,
) -> dict[str, Any]:
    markets = list(event_group.get("markets") or [])
    if not markets:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_grouping",
                "failure_reason": NEG_RISK_NO_NAMED_MARKETS,
                "event_slug": event_group.get("event_slug"),
            },
        }

    basket_bid_sum = 0.0
    target_shares_ceiling: float | None = None
    bid_details: list[dict[str, Any]] = []

    for market in markets:
        token_id = market.get("yes_token_id")
        if not token_id:
            return {
                "opportunity": None,
                "audit": {
                    "failure_stage": "pre_candidate_grouping",
                    "failure_reason": NEG_RISK_MISSING_YES_TOKEN,
                    "event_slug": event_group.get("event_slug"),
                    "market_slug": market.get("slug"),
                },
            }

        book = books_by_token.get(str(token_id))
        if book is None:
            return {
                "opportunity": None,
                "audit": {
                    "failure_stage": "pre_candidate_execution",
                    "failure_reason": NEG_RISK_EMPTY_BIDS,
                    "event_slug": event_group.get("event_slug"),
                    "market_slug": market.get("slug"),
                    "token_id": token_id,
                },
            }

        bid = _best_bid(book)
        bid_size = _best_bid_size(book)
        if bid is None or bid_size is None:
            return {
                "opportunity": None,
                "audit": {
                    "failure_stage": "pre_candidate_execution",
                    "failure_reason": NEG_RISK_EMPTY_BIDS,
                    "event_slug": event_group.get("event_slug"),
                    "market_slug": market.get("slug"),
                    "token_id": token_id,
                },
            }

        basket_bid_sum += bid
        target_shares_ceiling = bid_size if target_shares_ceiling is None else min(target_shares_ceiling, bid_size)
        bid_details.append(
            {
                "market_slug": market.get("slug"),
                "yes_token_id": token_id,
                "maker_yes_bid": round(bid, 6),
                "maker_yes_bid_size": round(bid_size, 6),
            }
        )

    maker_edge = 1.0 - basket_bid_sum
    if maker_edge <= 0.0:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_pricing",
                "failure_reason": NEG_RISK_BASKET_EDGE_NON_POSITIVE,
                "event_slug": event_group.get("event_slug"),
                "basket_bid_sum": round(basket_bid_sum, 6),
                "maker_edge_cents": round(maker_edge, 6),
                "legs": bid_details,
            },
        }

    if basket_bid_sum <= 0.0 or target_shares_ceiling is None or target_shares_ceiling <= 0.0:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_execution",
                "failure_reason": NEG_RISK_INVALID_ORDERBOOK,
                "event_slug": event_group.get("event_slug"),
                "basket_bid_sum": round(basket_bid_sum, 6),
                "legs": bid_details,
            },
        }

    budget_shares = max_notional / basket_bid_sum
    target_shares = max(0.0, min(target_shares_ceiling, budget_shares))
    est_fill_cost = basket_bid_sum * target_shares
    gross_profit = maker_edge * target_shares

    return {
        "opportunity": ArbOpportunity(
            kind="neg_risk_rebalancing",
            name=str(event_group.get("event_slug") or "neg_risk_rebalancing"),
            edge_cents=round(maker_edge, 6),
            gross_profit=round(gross_profit, 6),
            est_fill_cost=round(est_fill_cost, 6),
            est_payout=round(target_shares, 6),
            notional=round(est_fill_cost, 6),
            details={
                "event_id": event_group.get("event_id"),
                "event_slug": event_group.get("event_slug"),
                "event_title": event_group.get("event_title"),
                "neg_risk_market_id": event_group.get("neg_risk_market_id"),
                "maker_first": True,
                "basket_bid_sum": round(basket_bid_sum, 6),
                "maker_edge_cents": round(maker_edge, 6),
                "shares": round(target_shares, 6),
                "legs": bid_details,
            },
            ts=datetime.now(timezone.utc),
        ),
        "audit": None,
    }
