from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.core.models import ArbOpportunity, OrderBook

LHS_RELATION_EMPTY_ASKS = "LHS_RELATION_EMPTY_ASKS"
RHS_RELATION_EMPTY_ASKS = "RHS_RELATION_EMPTY_ASKS"
RELATION_EDGE_NON_POSITIVE = "RELATION_EDGE_NON_POSITIVE"
RELATION_GROSS_EDGE_NON_POSITIVE = "RELATION_GROSS_EDGE_NON_POSITIVE"
LHS_EXECUTION_EMPTY_ASKS = "LHS_EXECUTION_EMPTY_ASKS"
RHS_EXECUTION_EMPTY_ASKS = "RHS_EXECUTION_EMPTY_ASKS"
EXECUTION_GROSS_EDGE_NON_POSITIVE = "EXECUTION_GROSS_EDGE_NON_POSITIVE"


def scan_leq_constraint(name: str, lhs_book: OrderBook, rhs_book: OrderBook, lhs_slug: str, rhs_slug: str, total_buffer_cents: float) -> Optional[ArbOpportunity]:
    analysis = analyze_leq_constraint(name, lhs_book, rhs_book, lhs_slug, rhs_slug, total_buffer_cents)
    return analysis["opportunity"]


def analyze_leq_constraint(
    name: str,
    lhs_book: OrderBook,
    rhs_book: OrderBook,
    lhs_slug: str,
    rhs_slug: str,
    total_buffer_cents: float,
) -> dict[str, object]:
    """
    Detect violation of P(lhs) <= P(rhs).
    This is research-only detection, not executable trading logic.
    """
    return _analyze_leq_constraint(
        name,
        lhs_book,
        rhs_book,
        lhs_slug,
        rhs_slug,
        total_buffer_cents,
        require_buffer_positive=True,
        non_positive_reason=RELATION_EDGE_NON_POSITIVE,
        detection_mode="buffered_relation_gap",
    )


def analyze_leq_constraint_gross(
    name: str,
    lhs_book: OrderBook,
    rhs_book: OrderBook,
    lhs_slug: str,
    rhs_slug: str,
    total_buffer_cents: float,
) -> dict[str, object]:
    """
    Detect gross violation of P(lhs) <= P(rhs) before applying buffers.
    This is additive observability for opportunity-density research only.
    """
    return _analyze_leq_constraint(
        name,
        lhs_book,
        rhs_book,
        lhs_slug,
        rhs_slug,
        total_buffer_cents,
        require_buffer_positive=False,
        non_positive_reason=RELATION_GROSS_EDGE_NON_POSITIVE,
        detection_mode="gross_relation_gap",
    )


def analyze_leq_constraint_execution_gross(
    name: str,
    lhs_relation_book: OrderBook,
    rhs_relation_book: OrderBook,
    lhs_execution_book: OrderBook,
    rhs_execution_book: OrderBook,
    lhs_slug: str,
    rhs_slug: str,
    total_buffer_cents: float,
) -> dict[str, object]:
    relation_analysis = analyze_leq_constraint_gross(
        name,
        lhs_relation_book,
        rhs_relation_book,
        lhs_slug,
        rhs_slug,
        total_buffer_cents,
    )
    if relation_analysis["opportunity"] is None:
        return relation_analysis

    lhs_relation_ask = float(lhs_relation_book.asks[0].price) if lhs_relation_book.asks else None
    rhs_relation_ask = float(rhs_relation_book.asks[0].price) if rhs_relation_book.asks else None
    lhs_execution_ask = float(lhs_execution_book.asks[0].price) if lhs_execution_book.asks else None
    rhs_execution_ask = float(rhs_execution_book.asks[0].price) if rhs_execution_book.asks else None

    if lhs_execution_ask is None or rhs_execution_ask is None:
        failure_reason = LHS_EXECUTION_EMPTY_ASKS if lhs_execution_ask is None else RHS_EXECUTION_EMPTY_ASKS
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_execution",
                "failure_reason": failure_reason,
                "constraint_name": name,
                "lhs_market_slug": lhs_slug,
                "rhs_market_slug": rhs_slug,
                "lhs_relation_ask": lhs_relation_ask,
                "rhs_relation_ask": rhs_relation_ask,
                "relation_gap": round(float(lhs_relation_ask or 0.0) - float(rhs_relation_ask or 0.0), 6)
                if lhs_relation_ask is not None and rhs_relation_ask is not None
                else None,
                "lhs_execution_ask": lhs_execution_ask,
                "rhs_execution_ask": rhs_execution_ask,
                "execution_pair_best_ask_cost": None,
                "execution_best_ask_edge_cents": None,
                "total_buffer_cents": float(total_buffer_cents),
                "edge_after_buffer": round((float(lhs_relation_ask or 0.0) - float(rhs_relation_ask or 0.0)) - total_buffer_cents, 6)
                if lhs_relation_ask is not None and rhs_relation_ask is not None
                else None,
                "detection_mode": "execution_gross_pair_cost",
            },
        }

    execution_pair_best_ask_cost = lhs_execution_ask + rhs_execution_ask
    execution_best_ask_edge_cents = 1.0 - execution_pair_best_ask_cost
    if execution_best_ask_edge_cents <= 0.0:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_execution",
                "failure_reason": EXECUTION_GROSS_EDGE_NON_POSITIVE,
                "constraint_name": name,
                "lhs_market_slug": lhs_slug,
                "rhs_market_slug": rhs_slug,
                "lhs_relation_ask": lhs_relation_ask,
                "rhs_relation_ask": rhs_relation_ask,
                "relation_gap": round(float(lhs_relation_ask or 0.0) - float(rhs_relation_ask or 0.0), 6)
                if lhs_relation_ask is not None and rhs_relation_ask is not None
                else None,
                "lhs_execution_ask": lhs_execution_ask,
                "rhs_execution_ask": rhs_execution_ask,
                "execution_pair_best_ask_cost": round(execution_pair_best_ask_cost, 6),
                "execution_best_ask_edge_cents": round(execution_best_ask_edge_cents, 6),
                "total_buffer_cents": float(total_buffer_cents),
                "edge_after_buffer": round((float(lhs_relation_ask or 0.0) - float(rhs_relation_ask or 0.0)) - total_buffer_cents, 6)
                if lhs_relation_ask is not None and rhs_relation_ask is not None
                else None,
                "detection_mode": "execution_gross_pair_cost",
            },
        }

    relation_opportunity = relation_analysis["opportunity"]
    assert relation_opportunity is not None
    relation_details = dict(relation_opportunity.details)
    relation_details.update(
        {
            "lhs_execution_ask": lhs_execution_ask,
            "rhs_execution_ask": rhs_execution_ask,
            "execution_pair_best_ask_cost": round(execution_pair_best_ask_cost, 6),
            "execution_best_ask_edge_cents": round(execution_best_ask_edge_cents, 6),
            "detection_mode": "execution_gross_pair_cost",
        }
    )
    return {
        "opportunity": ArbOpportunity(
            kind=relation_opportunity.kind,
            name=relation_opportunity.name,
            edge_cents=round(execution_best_ask_edge_cents, 6),
            gross_profit=relation_opportunity.gross_profit,
            est_fill_cost=relation_opportunity.est_fill_cost,
            est_payout=relation_opportunity.est_payout,
            notional=relation_opportunity.notional,
            details=relation_details,
            ts=relation_opportunity.ts,
        ),
        "audit": None,
    }


def _analyze_leq_constraint(
    name: str,
    lhs_book: OrderBook,
    rhs_book: OrderBook,
    lhs_slug: str,
    rhs_slug: str,
    total_buffer_cents: float,
    *,
    require_buffer_positive: bool,
    non_positive_reason: str,
    detection_mode: str,
) -> dict[str, object]:
    lhs_ask = float(lhs_book.asks[0].price) if lhs_book.asks else None
    rhs_ask = float(rhs_book.asks[0].price) if rhs_book.asks else None

    if lhs_ask is None or rhs_ask is None:
        failure_reason = LHS_RELATION_EMPTY_ASKS if lhs_ask is None else RHS_RELATION_EMPTY_ASKS
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_relation",
                "failure_reason": failure_reason,
                "constraint_name": name,
                "lhs_market_slug": lhs_slug,
                "rhs_market_slug": rhs_slug,
                "lhs_relation_ask": lhs_ask,
                "rhs_relation_ask": rhs_ask,
                "relation_gap": None,
                "total_buffer_cents": float(total_buffer_cents),
                "edge_after_buffer": None,
            },
        }

    violation = lhs_ask - rhs_ask
    threshold = total_buffer_cents if require_buffer_positive else 0.0
    if violation <= threshold:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_relation",
                "failure_reason": non_positive_reason,
                "constraint_name": name,
                "lhs_market_slug": lhs_slug,
                "rhs_market_slug": rhs_slug,
                "lhs_relation_ask": lhs_ask,
                "rhs_relation_ask": rhs_ask,
                "relation_gap": round(violation, 6),
                "total_buffer_cents": float(total_buffer_cents),
                "edge_after_buffer": round(violation - total_buffer_cents, 6),
                "detection_mode": detection_mode,
            },
        }

    return {
        "opportunity": ArbOpportunity(
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
                "relation_gap": round(violation, 6),
                "edge_after_buffer": round(violation - total_buffer_cents, 6),
                "detection_mode": detection_mode,
            },
            ts=datetime.now(timezone.utc),
        ),
        "audit": None,
    }
