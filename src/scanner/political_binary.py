from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.core.models import ArbOpportunity, OrderBook

POLITICAL_LHS_RELATION_EMPTY_ASKS = "POLITICAL_LHS_RELATION_EMPTY_ASKS"
POLITICAL_RHS_RELATION_EMPTY_ASKS = "POLITICAL_RHS_RELATION_EMPTY_ASKS"
POLITICAL_LHS_EXECUTION_EMPTY_ASKS = "POLITICAL_LHS_EXECUTION_EMPTY_ASKS"
POLITICAL_RHS_EXECUTION_EMPTY_ASKS = "POLITICAL_RHS_EXECUTION_EMPTY_ASKS"
POLITICAL_MUTEX_RELATION_NON_POSITIVE = "POLITICAL_MUTEX_RELATION_NON_POSITIVE"
POLITICAL_IMPLICATION_RELATION_NON_POSITIVE = "POLITICAL_IMPLICATION_RELATION_NON_POSITIVE"
POLITICAL_EXECUTION_GROSS_EDGE_NON_POSITIVE = "POLITICAL_EXECUTION_GROSS_EDGE_NON_POSITIVE"


def analyze_political_mutex_pair(
    rule: dict,
    lhs_relation_book: OrderBook,
    rhs_relation_book: OrderBook,
    lhs_execution_book: OrderBook,
    rhs_execution_book: OrderBook,
    total_buffer_cents: float,
) -> dict[str, object]:
    lhs_relation_ask = _top_ask(lhs_relation_book)
    rhs_relation_ask = _top_ask(rhs_relation_book)
    relation_audit = _relation_missing_audit(rule, lhs_relation_ask, rhs_relation_ask, total_buffer_cents)
    if relation_audit is not None:
        return {"opportunity": None, "audit": relation_audit}

    relation_gap = float(lhs_relation_ask) + float(rhs_relation_ask) - 1.0
    if relation_gap <= 0.0:
        return {
            "opportunity": None,
            "audit": _build_audit(
                rule,
                failure_stage="pre_candidate_relation",
                failure_reason=POLITICAL_MUTEX_RELATION_NON_POSITIVE,
                lhs_relation_ask=lhs_relation_ask,
                rhs_relation_ask=rhs_relation_ask,
                relation_gap=relation_gap,
                total_buffer_cents=total_buffer_cents,
                detection_mode="mutex_yes_yes_gross",
            ),
        }

    lhs_execution_ask = _top_ask(lhs_execution_book)
    rhs_execution_ask = _top_ask(rhs_execution_book)
    execution_audit = _execution_missing_audit(
        rule,
        lhs_relation_ask,
        rhs_relation_ask,
        lhs_execution_ask,
        rhs_execution_ask,
        relation_gap,
        total_buffer_cents,
        detection_mode="mutex_no_no_execution_gross",
    )
    if execution_audit is not None:
        return {"opportunity": None, "audit": execution_audit}

    execution_pair_best_ask_cost = float(lhs_execution_ask) + float(rhs_execution_ask)
    execution_best_ask_edge_cents = 1.0 - execution_pair_best_ask_cost
    if execution_best_ask_edge_cents <= 0.0:
        return {
            "opportunity": None,
            "audit": _build_audit(
                rule,
                failure_stage="pre_candidate_execution",
                failure_reason=POLITICAL_EXECUTION_GROSS_EDGE_NON_POSITIVE,
                lhs_relation_ask=lhs_relation_ask,
                rhs_relation_ask=rhs_relation_ask,
                lhs_execution_ask=lhs_execution_ask,
                rhs_execution_ask=rhs_execution_ask,
                relation_gap=relation_gap,
                execution_pair_best_ask_cost=execution_pair_best_ask_cost,
                execution_best_ask_edge_cents=execution_best_ask_edge_cents,
                total_buffer_cents=total_buffer_cents,
                detection_mode="mutex_no_no_execution_gross",
            ),
        }

    return {
        "opportunity": ArbOpportunity(
            kind="cross_market",
            name=rule["name"],
            edge_cents=round(execution_best_ask_edge_cents, 6),
            gross_profit=0.0,
            est_fill_cost=0.0,
            est_payout=0.0,
            notional=0.0,
            details={
                "relation_type": rule.get("relation_type"),
                "constraint_formula": "P(lhs_yes) + P(rhs_yes) <= 1",
                "lhs_relation_ask": lhs_relation_ask,
                "rhs_relation_ask": rhs_relation_ask,
                "relation_gap": round(relation_gap, 6),
                "lhs_execution_ask": lhs_execution_ask,
                "rhs_execution_ask": rhs_execution_ask,
                "execution_pair_best_ask_cost": round(execution_pair_best_ask_cost, 6),
                "execution_best_ask_edge_cents": round(execution_best_ask_edge_cents, 6),
                "detection_mode": "mutex_no_no_execution_gross",
            },
            ts=datetime.now(timezone.utc),
        ),
        "audit": None,
    }


def analyze_political_implication_pair(
    rule: dict,
    lhs_relation_book: OrderBook,
    rhs_relation_book: OrderBook,
    lhs_execution_book: OrderBook,
    rhs_execution_book: OrderBook,
    total_buffer_cents: float,
) -> dict[str, object]:
    lhs_relation_ask = _top_ask(lhs_relation_book)
    rhs_relation_ask = _top_ask(rhs_relation_book)
    relation_audit = _relation_missing_audit(rule, lhs_relation_ask, rhs_relation_ask, total_buffer_cents)
    if relation_audit is not None:
        return {"opportunity": None, "audit": relation_audit}

    relation_gap = float(lhs_relation_ask) - float(rhs_relation_ask)
    if relation_gap <= 0.0:
        return {
            "opportunity": None,
            "audit": _build_audit(
                rule,
                failure_stage="pre_candidate_relation",
                failure_reason=POLITICAL_IMPLICATION_RELATION_NON_POSITIVE,
                lhs_relation_ask=lhs_relation_ask,
                rhs_relation_ask=rhs_relation_ask,
                relation_gap=relation_gap,
                total_buffer_cents=total_buffer_cents,
                detection_mode="implication_yes_yes_gross",
            ),
        }

    lhs_execution_ask = _top_ask(lhs_execution_book)
    rhs_execution_ask = _top_ask(rhs_execution_book)
    execution_audit = _execution_missing_audit(
        rule,
        lhs_relation_ask,
        rhs_relation_ask,
        lhs_execution_ask,
        rhs_execution_ask,
        relation_gap,
        total_buffer_cents,
        detection_mode="implication_no_yes_execution_gross",
    )
    if execution_audit is not None:
        return {"opportunity": None, "audit": execution_audit}

    execution_pair_best_ask_cost = float(lhs_execution_ask) + float(rhs_execution_ask)
    execution_best_ask_edge_cents = 1.0 - execution_pair_best_ask_cost
    if execution_best_ask_edge_cents <= 0.0:
        return {
            "opportunity": None,
            "audit": _build_audit(
                rule,
                failure_stage="pre_candidate_execution",
                failure_reason=POLITICAL_EXECUTION_GROSS_EDGE_NON_POSITIVE,
                lhs_relation_ask=lhs_relation_ask,
                rhs_relation_ask=rhs_relation_ask,
                lhs_execution_ask=lhs_execution_ask,
                rhs_execution_ask=rhs_execution_ask,
                relation_gap=relation_gap,
                execution_pair_best_ask_cost=execution_pair_best_ask_cost,
                execution_best_ask_edge_cents=execution_best_ask_edge_cents,
                total_buffer_cents=total_buffer_cents,
                detection_mode="implication_no_yes_execution_gross",
            ),
        }

    return {
        "opportunity": ArbOpportunity(
            kind="cross_market",
            name=rule["name"],
            edge_cents=round(execution_best_ask_edge_cents, 6),
            gross_profit=0.0,
            est_fill_cost=0.0,
            est_payout=0.0,
            notional=0.0,
            details={
                "relation_type": rule.get("relation_type"),
                "constraint_formula": "P(lhs_yes) <= P(rhs_yes)",
                "lhs_relation_ask": lhs_relation_ask,
                "rhs_relation_ask": rhs_relation_ask,
                "relation_gap": round(relation_gap, 6),
                "lhs_execution_ask": lhs_execution_ask,
                "rhs_execution_ask": rhs_execution_ask,
                "execution_pair_best_ask_cost": round(execution_pair_best_ask_cost, 6),
                "execution_best_ask_edge_cents": round(execution_best_ask_edge_cents, 6),
                "detection_mode": "implication_no_yes_execution_gross",
            },
            ts=datetime.now(timezone.utc),
        ),
        "audit": None,
    }


def _top_ask(book: OrderBook) -> float | None:
    asks = getattr(book, "asks", [])
    if not asks:
        return None
    return float(asks[0].price)


def _relation_missing_audit(
    rule: dict,
    lhs_relation_ask: float | None,
    rhs_relation_ask: float | None,
    total_buffer_cents: float,
) -> Optional[dict[str, object]]:
    if lhs_relation_ask is not None and rhs_relation_ask is not None:
        return None
    failure_reason = (
        POLITICAL_LHS_RELATION_EMPTY_ASKS
        if lhs_relation_ask is None
        else POLITICAL_RHS_RELATION_EMPTY_ASKS
    )
    return _build_audit(
        rule,
        failure_stage="pre_candidate_relation",
        failure_reason=failure_reason,
        lhs_relation_ask=lhs_relation_ask,
        rhs_relation_ask=rhs_relation_ask,
        total_buffer_cents=total_buffer_cents,
    )


def _execution_missing_audit(
    rule: dict,
    lhs_relation_ask: float | None,
    rhs_relation_ask: float | None,
    lhs_execution_ask: float | None,
    rhs_execution_ask: float | None,
    relation_gap: float,
    total_buffer_cents: float,
    detection_mode: str,
) -> Optional[dict[str, object]]:
    if lhs_execution_ask is not None and rhs_execution_ask is not None:
        return None
    failure_reason = (
        POLITICAL_LHS_EXECUTION_EMPTY_ASKS
        if lhs_execution_ask is None
        else POLITICAL_RHS_EXECUTION_EMPTY_ASKS
    )
    return _build_audit(
        rule,
        failure_stage="pre_candidate_execution",
        failure_reason=failure_reason,
        lhs_relation_ask=lhs_relation_ask,
        rhs_relation_ask=rhs_relation_ask,
        lhs_execution_ask=lhs_execution_ask,
        rhs_execution_ask=rhs_execution_ask,
        relation_gap=relation_gap,
        total_buffer_cents=total_buffer_cents,
        detection_mode=detection_mode,
    )


def _build_audit(
    rule: dict,
    *,
    failure_stage: str,
    failure_reason: str,
    lhs_relation_ask: float | None = None,
    rhs_relation_ask: float | None = None,
    lhs_execution_ask: float | None = None,
    rhs_execution_ask: float | None = None,
    relation_gap: float | None = None,
    execution_pair_best_ask_cost: float | None = None,
    execution_best_ask_edge_cents: float | None = None,
    total_buffer_cents: float,
    detection_mode: str | None = None,
) -> dict[str, object]:
    payload = {
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "constraint_name": rule["name"],
        "relation_type": rule.get("relation_type"),
        "lhs_market_slug": rule["lhs"]["market_slug"],
        "rhs_market_slug": rule["rhs"]["market_slug"],
        "lhs_relation_side": str(rule.get("lhs", {}).get("side", "YES")).upper(),
        "rhs_relation_side": str(rule.get("rhs", {}).get("side", "YES")).upper(),
        "lhs_execution_side": str(rule.get("lhs_execution", {}).get("side", "NO")).upper(),
        "rhs_execution_side": str(rule.get("rhs_execution", {}).get("side", "YES")).upper(),
        "lhs_relation_ask": lhs_relation_ask,
        "rhs_relation_ask": rhs_relation_ask,
        "lhs_execution_ask": lhs_execution_ask,
        "rhs_execution_ask": rhs_execution_ask,
        "relation_gap": round(relation_gap, 6) if relation_gap is not None else None,
        "execution_pair_best_ask_cost": round(execution_pair_best_ask_cost, 6)
        if execution_pair_best_ask_cost is not None
        else None,
        "execution_best_ask_edge_cents": round(execution_best_ask_edge_cents, 6)
        if execution_best_ask_edge_cents is not None
        else None,
        "total_buffer_cents": float(total_buffer_cents),
        "assertion": rule.get("assertion"),
        "notes": rule.get("notes"),
    }
    if detection_mode is not None:
        payload["detection_mode"] = detection_mode
    return payload

