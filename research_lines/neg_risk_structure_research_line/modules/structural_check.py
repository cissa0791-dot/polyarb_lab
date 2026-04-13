"""
neg_risk_structure_research_line — Module 3: Structural Check
polyarb_lab / research_line / active

Detects possible pricing inconsistencies in normalized NegRiskEvent objects.

Checks performed:
  1. Sum-to-one constraint check (NEG-001)
  2. Per-outcome equivalent position cost check (NEG-002)
  3. Directional bias classification (NEG-004)

This module works entirely on already-normalized data (NegRiskEvent).
No additional API calls are made here.

Read-only. No order submission. No mainline imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from .normalizer import NegRiskEvent, NegRiskOutcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

# Sum constraint thresholds (NEG-001)
GAP_ROUNDING_FLOOR = 0.005       # below this: effectively at constraint (rounding)
GAP_BOUNDARY = 0.010             # above this: non-trivial deviation
GAP_SIGNIFICANT = 0.030          # above this: may survive 1-leg round-trip fee

# Fee estimate for 1-leg round trip (taker fee ~1% each way = 0.02 total)
FEE_ESTIMATE_ROUNDTRIP = 0.020

ConstraintClass = Literal[
    "AT_CONSTRAINT",          # abs_gap < 0.005 — within rounding
    "BOUNDARY",               # 0.005 <= abs_gap < 0.010 — marginal
    "CONSTRAINT_VIOLATION",   # abs_gap >= 0.010 — non-trivial
    "SIGNIFICANT_VIOLATION",  # abs_gap >= 0.030 — may survive fees
    "UNKNOWN",                # could not compute (missing prices)
]

GapDirectionClass = Literal[
    "SUM_ABOVE_ONE",    # implied_sum > 1.0 — collective YES overpriced
    "SUM_BELOW_ONE",    # implied_sum < 1.0 — collective YES underpriced
    "AT_PARITY",        # within rounding — no direction
]

DirectionalBiasClass = Literal[
    "LEADING_OVERPRICED",    # leading outcome contributes most to excess (sum > 1)
    "TRAILING_OVERPRICED",   # trailing outcomes collectively drive excess
    "MIXED",                 # unclear direction of excess
    "NO_VIOLATION",          # no significant gap to classify
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class OutcomeCheckDetail:
    """Per-outcome structural check result."""
    outcome_index: int
    outcome_id: str
    question: str
    yes_mid: float
    no_mid: float

    # Equivalent position check (NEG-002)
    # cost_no_bundle_i = sum(1 - yes_bid_j for j != i)
    # If yes_bid unavailable, uses 1 - yes_mid as approximation
    cost_no_bundle: Optional[float]         # cost to buy NO on all others
    cost_yes_direct: Optional[float]        # cost to buy YES directly (yes_ask)
    equivalent_gap: Optional[float]         # cost_yes_direct - cost_no_bundle
    equivalent_gap_after_fee: Optional[float]  # equivalent_gap - FEE_ESTIMATE_ROUNDTRIP

    # Contribution to implied_sum excess
    fair_share: float                       # 1.0 / N — equal-split fair price
    excess_contribution: float             # yes_mid - fair_share


@dataclass
class StructuralCheckResult:
    """Complete structural check result for one NegRiskEvent."""
    event_id: str
    slug: str
    title: str
    n_outcomes: int
    implied_sum: float
    constraint_gap: float
    abs_gap: float
    has_all_prices: bool

    # Classification
    constraint_class: ConstraintClass
    gap_direction: GapDirectionClass
    directional_bias: DirectionalBiasClass

    # Per-outcome details
    outcome_details: list[OutcomeCheckDetail]

    # Summary metrics
    max_equivalent_gap: Optional[float]       # largest |equivalent_gap| across outcomes
    max_equivalent_gap_outcome_id: Optional[str]
    max_equivalent_gap_after_fee: Optional[float]

    leading_outcome_id: str
    leading_outcome_yes_mid: float
    leading_outcome_excess: float

    # Research flags
    passes_gap_threshold: bool              # abs_gap >= GAP_BOUNDARY
    passes_fee_hurdle: bool                 # abs_gap >= FEE_ESTIMATE_ROUNDTRIP
    research_note: str                      # human-readable summary

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "slug": self.slug,
            "title": self.title,
            "n_outcomes": self.n_outcomes,
            "implied_sum": round(self.implied_sum, 6),
            "constraint_gap": round(self.constraint_gap, 6),
            "abs_gap": round(self.abs_gap, 6),
            "has_all_prices": self.has_all_prices,
            "constraint_class": self.constraint_class,
            "gap_direction": self.gap_direction,
            "directional_bias": self.directional_bias,
            "passes_gap_threshold": self.passes_gap_threshold,
            "passes_fee_hurdle": self.passes_fee_hurdle,
            "leading_outcome_id": self.leading_outcome_id,
            "leading_outcome_yes_mid": round(self.leading_outcome_yes_mid, 6),
            "leading_outcome_excess": round(self.leading_outcome_excess, 6),
            "max_equivalent_gap": (
                round(self.max_equivalent_gap, 6) if self.max_equivalent_gap is not None else None
            ),
            "max_equivalent_gap_after_fee": (
                round(self.max_equivalent_gap_after_fee, 6)
                if self.max_equivalent_gap_after_fee is not None else None
            ),
            "max_equivalent_gap_outcome_id": self.max_equivalent_gap_outcome_id,
            "research_note": self.research_note,
            "outcome_details": [
                {
                    "outcome_index": d.outcome_index,
                    "outcome_id": d.outcome_id,
                    "question": d.question,
                    "yes_mid": round(d.yes_mid, 6),
                    "no_mid": round(d.no_mid, 6),
                    "cost_no_bundle": (
                        round(d.cost_no_bundle, 6) if d.cost_no_bundle is not None else None
                    ),
                    "cost_yes_direct": (
                        round(d.cost_yes_direct, 6) if d.cost_yes_direct is not None else None
                    ),
                    "equivalent_gap": (
                        round(d.equivalent_gap, 6) if d.equivalent_gap is not None else None
                    ),
                    "equivalent_gap_after_fee": (
                        round(d.equivalent_gap_after_fee, 6)
                        if d.equivalent_gap_after_fee is not None else None
                    ),
                    "fair_share": round(d.fair_share, 6),
                    "excess_contribution": round(d.excess_contribution, 6),
                }
                for d in self.outcome_details
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_constraint(abs_gap: float, has_all_prices: bool) -> ConstraintClass:
    if not has_all_prices:
        return "UNKNOWN"
    if abs_gap >= GAP_SIGNIFICANT:
        return "SIGNIFICANT_VIOLATION"
    if abs_gap >= GAP_BOUNDARY:
        return "CONSTRAINT_VIOLATION"
    if abs_gap >= GAP_ROUNDING_FLOOR:
        return "BOUNDARY"
    return "AT_CONSTRAINT"


def _classify_gap_direction(constraint_gap: float, abs_gap: float) -> GapDirectionClass:
    if abs_gap < GAP_ROUNDING_FLOOR:
        return "AT_PARITY"
    return "SUM_ABOVE_ONE" if constraint_gap > 0 else "SUM_BELOW_ONE"


def _classify_directional_bias(
    outcome_details: list[OutcomeCheckDetail],
    constraint_gap: float,
    abs_gap: float,
) -> DirectionalBiasClass:
    if abs_gap < GAP_BOUNDARY:
        return "NO_VIOLATION"
    if not outcome_details:
        return "MIXED"

    # Leading outcome = highest excess_contribution
    leading = max(outcome_details, key=lambda d: d.excess_contribution)
    fair_share = leading.fair_share

    if constraint_gap > 0:
        # Sum above 1: someone is overpriced
        if leading.excess_contribution > 0.01:
            return "LEADING_OVERPRICED"
        # Check if trailing outcomes collectively drive excess
        trailing_excess = sum(
            d.excess_contribution for d in outcome_details
            if d.outcome_id != leading.outcome_id
        )
        if trailing_excess > leading.excess_contribution:
            return "TRAILING_OVERPRICED"
    return "MIXED"


def _compute_equivalent_gap(
    target: NegRiskOutcome,
    all_outcomes: list[NegRiskOutcome],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute the equivalent position gap for outcome i.

    cost_no_bundle_i = sum of NO ask prices for all j != i
      NO ask_j ≈ 1 - YES bid_j (buying NO = selling YES)
      If YES bid unavailable, use 1 - YES mid as approximation.

    cost_yes_direct = YES ask for outcome i
      If YES ask unavailable, use YES mid as approximation.

    Returns (cost_no_bundle, cost_yes_direct, equivalent_gap).
    All may be None if data is insufficient.
    """
    others = [o for o in all_outcomes if o.outcome_id != target.outcome_id]
    if not others:
        return None, None, None

    no_prices: list[float] = []
    for other in others:
        # NO ask for other ≈ 1 - YES bid for other
        if other.yes_bid is not None and other.yes_bid > 0:
            no_prices.append(1.0 - other.yes_bid)
        else:
            # Fallback: use 1 - yes_mid
            no_prices.append(1.0 - other.yes_mid)

    cost_no_bundle = sum(no_prices)

    # YES ask for target
    if target.yes_ask is not None and target.yes_ask > 0:
        cost_yes_direct = target.yes_ask
    else:
        cost_yes_direct = target.yes_mid  # fallback

    equivalent_gap = cost_yes_direct - cost_no_bundle
    equivalent_gap_after_fee = equivalent_gap - FEE_ESTIMATE_ROUNDTRIP

    return cost_no_bundle, cost_yes_direct, equivalent_gap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(event: NegRiskEvent) -> StructuralCheckResult:
    """
    Run all structural checks on a normalized NegRiskEvent.

    Returns a StructuralCheckResult with full details.
    """
    fair_share = 1.0 / event.n_outcomes if event.n_outcomes > 0 else 0.0

    outcome_details: list[OutcomeCheckDetail] = []
    for outcome in event.outcomes:
        cost_no_bundle, cost_yes_direct, eq_gap = _compute_equivalent_gap(
            outcome, event.outcomes
        )
        eq_gap_after_fee = (
            (eq_gap - FEE_ESTIMATE_ROUNDTRIP) if eq_gap is not None else None
        )
        outcome_details.append(OutcomeCheckDetail(
            outcome_index=outcome.outcome_index,
            outcome_id=outcome.outcome_id,
            question=outcome.question,
            yes_mid=outcome.yes_mid,
            no_mid=outcome.no_mid,
            cost_no_bundle=cost_no_bundle,
            cost_yes_direct=cost_yes_direct,
            equivalent_gap=eq_gap,
            equivalent_gap_after_fee=eq_gap_after_fee,
            fair_share=fair_share,
            excess_contribution=outcome.yes_mid - fair_share,
        ))

    constraint_class = _classify_constraint(event.abs_gap, event.has_all_prices)
    gap_direction = _classify_gap_direction(event.constraint_gap, event.abs_gap)
    directional_bias = _classify_directional_bias(
        outcome_details, event.constraint_gap, event.abs_gap
    )

    # Max equivalent gap
    eq_gaps_with_ids = [
        (d.equivalent_gap, d.outcome_id)
        for d in outcome_details
        if d.equivalent_gap is not None
    ]
    if eq_gaps_with_ids:
        max_eq_gap, max_eq_gap_id = max(eq_gaps_with_ids, key=lambda x: abs(x[0]))
        max_eq_gap_after_fee = max_eq_gap - FEE_ESTIMATE_ROUNDTRIP
    else:
        max_eq_gap, max_eq_gap_id, max_eq_gap_after_fee = None, None, None

    # Leading outcome
    leading = event.leading_outcome()

    # Research note
    passes_gap = event.abs_gap >= GAP_BOUNDARY
    passes_fee = event.abs_gap >= FEE_ESTIMATE_ROUNDTRIP

    if constraint_class == "AT_CONSTRAINT":
        note = "No structural inconsistency detected — implied sum within rounding of 1.0."
    elif constraint_class == "BOUNDARY":
        note = (
            f"Marginal gap ({event.abs_gap:.4f}) — above rounding floor but below "
            "non-trivial threshold. Monitor for confirmation."
        )
    elif constraint_class == "CONSTRAINT_VIOLATION":
        note = (
            f"Non-trivial constraint gap ({event.abs_gap:.4f}, "
            f"{gap_direction}, {directional_bias}). "
            "Does not yet clear round-trip fee estimate." if not passes_fee else
            f"Non-trivial constraint gap ({event.abs_gap:.4f}, "
            f"{gap_direction}, {directional_bias}). Clears fee estimate — "
            "classify for execution quality check."
        )
    elif constraint_class == "SIGNIFICANT_VIOLATION":
        note = (
            f"Significant constraint gap ({event.abs_gap:.4f}) clears fee hurdle. "
            f"Direction: {gap_direction}. Bias: {directional_bias}. "
            "High priority for execution quality check."
        )
    else:
        note = "Constraint class UNKNOWN — incomplete price data. Do not escalate."

    return StructuralCheckResult(
        event_id=event.event_id,
        slug=event.slug,
        title=event.title,
        n_outcomes=event.n_outcomes,
        implied_sum=event.implied_sum,
        constraint_gap=event.constraint_gap,
        abs_gap=event.abs_gap,
        has_all_prices=event.has_all_prices,
        constraint_class=constraint_class,
        gap_direction=gap_direction,
        directional_bias=directional_bias,
        outcome_details=outcome_details,
        max_equivalent_gap=max_eq_gap,
        max_equivalent_gap_outcome_id=max_eq_gap_id,
        max_equivalent_gap_after_fee=max_eq_gap_after_fee,
        leading_outcome_id=leading.outcome_id,
        leading_outcome_yes_mid=leading.yes_mid,
        leading_outcome_excess=leading.yes_mid - fair_share,
        passes_gap_threshold=passes_gap,
        passes_fee_hurdle=passes_fee,
        research_note=note,
    )


def check_batch(
    events: list[NegRiskEvent],
) -> list[StructuralCheckResult]:
    """Run structural checks on a batch of normalized events."""
    results: list[StructuralCheckResult] = []
    for event in events:
        try:
            results.append(check(event))
        except Exception as exc:
            logger.warning("Structural check failed for event %s: %s", event.event_id, exc)
    logger.info(
        "Structural check complete: %d results | "
        "significant=%d, violation=%d, boundary=%d, at_constraint=%d, unknown=%d",
        len(results),
        sum(1 for r in results if r.constraint_class == "SIGNIFICANT_VIOLATION"),
        sum(1 for r in results if r.constraint_class == "CONSTRAINT_VIOLATION"),
        sum(1 for r in results if r.constraint_class == "BOUNDARY"),
        sum(1 for r in results if r.constraint_class == "AT_CONSTRAINT"),
        sum(1 for r in results if r.constraint_class == "UNKNOWN"),
    )
    return results
