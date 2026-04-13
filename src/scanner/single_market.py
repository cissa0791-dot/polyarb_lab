from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.core.models import ArbOpportunity, MarketPair, OrderBook
from src.domain.models import RejectionReason
from src.scanner.vwap import analyze_buy_cost_from_asks


YES_BUDGET_UNFILLABLE = "YES_BUDGET_UNFILLABLE"
NO_BUDGET_UNFILLABLE = "NO_BUDGET_UNFILLABLE"
PAIR_EDGE_NON_POSITIVE = "PAIR_EDGE_NON_POSITIVE"
YES_TOUCH_MISSING = "YES_TOUCH_MISSING"
NO_TOUCH_MISSING = "NO_TOUCH_MISSING"
TOUCH_EDGE_NON_POSITIVE = "TOUCH_EDGE_NON_POSITIVE"


def _best_ask(book: OrderBook) -> float | None:
    asks = list(getattr(book, "asks", []) or [])
    return float(asks[0].price) if asks else None


def _has_invalid_asks(book: OrderBook) -> bool:
    asks = list(getattr(book, "asks", []) or [])
    if not asks:
        return False
    previous_price = None
    for level in asks:
        price = float(level.price)
        size = float(level.size)
        if price <= 0 or size <= 0:
            return True
        if previous_price is not None and price < previous_price - 1e-12:
            return True
        previous_price = price
    return False


def _single_market_precheck(
    pair: MarketPair,
    yes_book: OrderBook,
    no_book: OrderBook,
    max_notional: float,
    total_buffer_cents: float,
) -> dict[str, Any] | None:
    yes_asks = list(getattr(yes_book, "asks", []) or [])
    no_asks = list(getattr(no_book, "asks", []) or [])
    yes_best_ask = _best_ask(yes_book)
    no_best_ask = _best_ask(no_book)

    if not yes_asks or not no_asks:
        return {
            "market_slug": pair.market_slug,
            "failure_stage": "pre_candidate_precheck",
            "failure_reason": RejectionReason.EMPTY_ASKS.value,
            "failed_leg": "YES" if not yes_asks else "NO",
            "max_notional": max_notional,
            "total_buffer_cents": total_buffer_cents,
            "yes_best_ask": yes_best_ask,
            "no_best_ask": no_best_ask,
        }

    if _has_invalid_asks(yes_book) or _has_invalid_asks(no_book):
        return {
            "market_slug": pair.market_slug,
            "failure_stage": "pre_candidate_precheck",
            "failure_reason": RejectionReason.INVALID_ORDERBOOK.value,
            "failed_leg": "YES" if _has_invalid_asks(yes_book) else "NO",
            "max_notional": max_notional,
            "total_buffer_cents": total_buffer_cents,
            "yes_best_ask": yes_best_ask,
            "no_best_ask": no_best_ask,
        }

    touch_pair_cost = None
    touch_edge_after_buffer = None
    if yes_best_ask is not None and no_best_ask is not None:
        touch_pair_cost = yes_best_ask + no_best_ask
        touch_edge_after_buffer = 1.0 - touch_pair_cost - total_buffer_cents
        if touch_edge_after_buffer <= 0:
            return {
                "market_slug": pair.market_slug,
                "failure_stage": "pre_candidate_precheck",
                "failure_reason": PAIR_EDGE_NON_POSITIVE,
                "max_notional": max_notional,
                "total_buffer_cents": total_buffer_cents,
                "yes_best_ask": yes_best_ask,
                "no_best_ask": no_best_ask,
                "touch_pair_cost": touch_pair_cost,
                "touch_edge_after_buffer": touch_edge_after_buffer,
            }

    return None


def _single_market_pricing_diagnostics(
    pair: MarketPair,
    yes_book: OrderBook,
    no_book: OrderBook,
    max_notional: float,
    total_buffer_cents: float,
) -> dict[str, Any]:
    # Split notional evenly for a conservative paper approximation.
    split_notional_per_leg = max_notional / 2.0
    yes_fill = analyze_buy_cost_from_asks(yes_book.asks, split_notional_per_leg)
    no_fill = analyze_buy_cost_from_asks(no_book.asks, split_notional_per_leg)

    yes_best_ask = _best_ask(yes_book)
    no_best_ask = _best_ask(no_book)
    yes_budget_shares = float(yes_fill["shares"])
    no_budget_shares = float(no_fill["shares"])
    yes_budget_vwap = float(yes_fill["vwap"]) if yes_fill["vwap"] is not None else None
    no_budget_vwap = float(no_fill["vwap"]) if no_fill["vwap"] is not None else None
    matched_shares = min(yes_budget_shares, no_budget_shares)
    yes_excess_shares_vs_match = yes_budget_shares - matched_shares
    no_excess_shares_vs_match = no_budget_shares - matched_shares

    pair_cost_budget_vwap_sum = None
    if yes_budget_vwap is not None and no_budget_vwap is not None:
        pair_cost_budget_vwap_sum = yes_budget_vwap + no_budget_vwap

    edge_after_buffer = None
    if pair_cost_budget_vwap_sum is not None:
        edge_after_buffer = 1.0 - pair_cost_budget_vwap_sum - total_buffer_cents

    failure_reason = None
    if not yes_fill["filled"]:
        failure_reason = YES_BUDGET_UNFILLABLE
    elif not no_fill["filled"]:
        failure_reason = NO_BUDGET_UNFILLABLE
    elif edge_after_buffer is None or edge_after_buffer <= 0:
        failure_reason = PAIR_EDGE_NON_POSITIVE

    diagnostics = {
        "market_slug": pair.market_slug,
        "failure_stage": "pre_candidate_pricing",
        "failure_reason": failure_reason,
        "max_notional": max_notional,
        "split_notional_per_leg": split_notional_per_leg,
        "total_buffer_cents": total_buffer_cents,
        "yes_best_ask": yes_best_ask,
        "no_best_ask": no_best_ask,
        "yes_budget_shares": yes_budget_shares,
        "no_budget_shares": no_budget_shares,
        "yes_budget_vwap": yes_budget_vwap,
        "no_budget_vwap": no_budget_vwap,
        "matched_shares": matched_shares,
        "yes_excess_shares_vs_match": yes_excess_shares_vs_match,
        "no_excess_shares_vs_match": no_excess_shares_vs_match,
        "pair_cost_budget_vwap_sum": pair_cost_budget_vwap_sum,
        "edge_after_buffer": edge_after_buffer,
        "levels_consumed_yes": int(yes_fill["levels_consumed"]),
        "levels_consumed_no": int(no_fill["levels_consumed"]),
    }
    return diagnostics


def analyze_yes_no_pair(
    pair: MarketPair,
    yes_book: OrderBook,
    no_book: OrderBook,
    max_notional: float,
    total_buffer_cents: float,
) -> dict[str, Any]:
    precheck = _single_market_precheck(
        pair,
        yes_book,
        no_book,
        max_notional=max_notional,
        total_buffer_cents=total_buffer_cents,
    )
    if precheck is not None:
        return {"opportunity": None, "audit": precheck}

    diagnostics = _single_market_pricing_diagnostics(
        pair,
        yes_book,
        no_book,
        max_notional=max_notional,
        total_buffer_cents=total_buffer_cents,
    )
    failure_reason = diagnostics["failure_reason"]
    if failure_reason is not None:
        return {"opportunity": None, "audit": diagnostics}

    pair_cost = float(diagnostics["pair_cost_budget_vwap_sum"])
    edge = float(diagnostics["edge_after_buffer"])
    shares = float(diagnostics["matched_shares"])
    gross = shares * edge

    opportunity = ArbOpportunity(
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
            "yes_vwap": diagnostics["yes_budget_vwap"],
            "no_vwap": diagnostics["no_budget_vwap"],
        },
        ts=datetime.now(timezone.utc),
    )
    return {"opportunity": opportunity, "audit": None}


def _single_market_touch_diagnostics(
    pair: MarketPair,
    yes_book: OrderBook,
    no_book: OrderBook,
    max_notional: float,
    total_buffer_cents: float,
) -> dict[str, Any]:
    split_notional_per_leg = max_notional / 2.0
    yes_best_ask = _best_ask(yes_book)
    no_best_ask = _best_ask(no_book)

    yes_touch_shares = (split_notional_per_leg / yes_best_ask) if yes_best_ask and yes_best_ask > 0 else None
    no_touch_shares = (split_notional_per_leg / no_best_ask) if no_best_ask and no_best_ask > 0 else None
    matched_shares = min(yes_touch_shares or 0.0, no_touch_shares or 0.0) if yes_touch_shares and no_touch_shares else 0.0

    touch_pair_cost = None
    if yes_best_ask is not None and no_best_ask is not None:
        touch_pair_cost = yes_best_ask + no_best_ask

    touch_edge_gross = None
    touch_edge_after_buffer = None
    if touch_pair_cost is not None:
        touch_edge_gross = 1.0 - touch_pair_cost
        touch_edge_after_buffer = touch_edge_gross - total_buffer_cents

    failure_reason = None
    if yes_best_ask is None:
        failure_reason = YES_TOUCH_MISSING
    elif no_best_ask is None:
        failure_reason = NO_TOUCH_MISSING
    elif touch_edge_gross is None or touch_edge_gross <= 0:
        failure_reason = TOUCH_EDGE_NON_POSITIVE

    return {
        "market_slug": pair.market_slug,
        "failure_stage": "pre_candidate_pricing",
        "failure_reason": failure_reason,
        "max_notional": max_notional,
        "split_notional_per_leg": split_notional_per_leg,
        "total_buffer_cents": total_buffer_cents,
        "yes_best_ask": yes_best_ask,
        "no_best_ask": no_best_ask,
        "yes_touch_shares": yes_touch_shares,
        "no_touch_shares": no_touch_shares,
        "matched_shares": matched_shares,
        "touch_pair_cost": touch_pair_cost,
        "touch_edge_gross": touch_edge_gross,
        "touch_edge_after_buffer": touch_edge_after_buffer,
    }


def _single_market_touch_precheck(
    pair: MarketPair,
    yes_book: OrderBook,
    no_book: OrderBook,
    max_notional: float,
    total_buffer_cents: float,
) -> dict[str, Any] | None:
    yes_asks = list(getattr(yes_book, "asks", []) or [])
    no_asks = list(getattr(no_book, "asks", []) or [])
    yes_best_ask = _best_ask(yes_book)
    no_best_ask = _best_ask(no_book)

    if not yes_asks or not no_asks:
        return {
            "market_slug": pair.market_slug,
            "failure_stage": "pre_candidate_precheck",
            "failure_reason": RejectionReason.EMPTY_ASKS.value,
            "failed_leg": "YES" if not yes_asks else "NO",
            "max_notional": max_notional,
            "total_buffer_cents": total_buffer_cents,
            "yes_best_ask": yes_best_ask,
            "no_best_ask": no_best_ask,
        }

    yes_invalid = _has_invalid_asks(yes_book)
    no_invalid = _has_invalid_asks(no_book)
    if yes_invalid or no_invalid:
        return {
            "market_slug": pair.market_slug,
            "failure_stage": "pre_candidate_precheck",
            "failure_reason": RejectionReason.INVALID_ORDERBOOK.value,
            "failed_leg": "YES" if yes_invalid else "NO",
            "max_notional": max_notional,
            "total_buffer_cents": total_buffer_cents,
            "yes_best_ask": yes_best_ask,
            "no_best_ask": no_best_ask,
        }

    touch_pair_cost = None
    touch_edge_gross = None
    touch_edge_after_buffer = None
    if yes_best_ask is not None and no_best_ask is not None:
        touch_pair_cost = yes_best_ask + no_best_ask
        touch_edge_gross = 1.0 - touch_pair_cost
        touch_edge_after_buffer = touch_edge_gross - total_buffer_cents
        if touch_edge_gross <= 0:
            return {
                "market_slug": pair.market_slug,
                "failure_stage": "pre_candidate_precheck",
                "failure_reason": TOUCH_EDGE_NON_POSITIVE,
                "max_notional": max_notional,
                "total_buffer_cents": total_buffer_cents,
                "yes_best_ask": yes_best_ask,
                "no_best_ask": no_best_ask,
                "touch_pair_cost": touch_pair_cost,
                "touch_edge_gross": touch_edge_gross,
                "touch_edge_after_buffer": touch_edge_after_buffer,
            }

    return None


def analyze_yes_no_touch_pair(
    pair: MarketPair,
    yes_book: OrderBook,
    no_book: OrderBook,
    max_notional: float,
    total_buffer_cents: float,
) -> dict[str, Any]:
    precheck = _single_market_touch_precheck(
        pair,
        yes_book,
        no_book,
        max_notional=max_notional,
        total_buffer_cents=total_buffer_cents,
    )
    if precheck is not None:
        return {"opportunity": None, "audit": precheck}

    diagnostics = _single_market_touch_diagnostics(
        pair,
        yes_book,
        no_book,
        max_notional=max_notional,
        total_buffer_cents=total_buffer_cents,
    )
    if diagnostics["failure_reason"] is not None:
        return {"opportunity": None, "audit": diagnostics}

    pair_cost = float(diagnostics["touch_pair_cost"])
    edge = float(diagnostics["touch_edge_gross"])
    shares = float(diagnostics["matched_shares"])
    gross = shares * edge

    opportunity = ArbOpportunity(
        kind="single_market",
        name="yes_no_touch_under_1",
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
            "yes_touch_price": diagnostics["yes_best_ask"],
            "no_touch_price": diagnostics["no_best_ask"],
            "touch_pair_cost": pair_cost,
            "touch_edge_gross": diagnostics["touch_edge_gross"],
            "touch_edge_after_buffer": diagnostics["touch_edge_after_buffer"],
        },
        ts=datetime.now(timezone.utc),
    )
    return {"opportunity": opportunity, "audit": None}


def scan_yes_no_pair(pair: MarketPair, yes_book: OrderBook, no_book: OrderBook, max_notional: float, total_buffer_cents: float) -> Optional[ArbOpportunity]:
    return analyze_yes_no_pair(
        pair,
        yes_book,
        no_book,
        max_notional=max_notional,
        total_buffer_cents=total_buffer_cents,
    )["opportunity"]
