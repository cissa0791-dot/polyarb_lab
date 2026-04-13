"""
neg_risk_structure_research_line — hypothesis registry
polyarb_lab / research_line / active

Schema and hypothesis registry for neg-risk structural research.
All hypotheses are paper/research-only. No execution path from this file.

Neg-risk structural constraint:
    In a neg-risk market with N outcomes, exactly one resolves YES.
    sum(P_yes_i for i = 1..N) must equal 1.0 under no-arbitrage.
    Deviation = structural inconsistency worth investigating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HypothesisCategory = Literal[
    "sum_constraint",          # sum of yes prices violates no-arbitrage constraint
    "conversion_arbitrage",    # NO→YES adapter conversion creates pricing gap
    "temporal_pattern",        # gap magnitude correlates with time-related variables
    "directional_bias",        # one leg type (leading/trailing) systematically mispriced
    "execution_quality",       # depth/spread/slippage properties of neg-risk markets
    "comparative_structure",   # neg-risk vs binary market structural differences
]

GapDirection = Literal[
    "sum_above_one",    # implied sum > 1.0 — collective YES overpriced
    "sum_below_one",    # implied sum < 1.0 — collective YES underpriced
    "indeterminate",    # direction unknown or not yet measured
]

EvidenceStrength = Literal[
    "none",             # no data yet
    "weak",             # 1–2 observations
    "moderate",         # 3–4 consistent observations
    "strong",           # >= 5 consistent observations across different events
]

FinalClassification = Literal[
    "keep",
    "downgrade",
    "reject",
    "park",
    "escalate",
    "pending",
]


# ---------------------------------------------------------------------------
# Hypothesis schema
# ---------------------------------------------------------------------------

@dataclass
class NegRiskHypothesis:
    hyp_id: str                         # e.g. NEG-001
    category: HypothesisCategory
    source: str                         # where the hypothesis originates
    raw_claim: str                      # original claim, unmodified
    codified_form: str                  # metric / rule / experiment logic
    test_method: str                    # how it will be tested (paper/research only)
    expected_value: str                 # what improvement or insight it might yield
    possible_failure_mode: str          # how it may fail or mislead
    evidence_standard: str              # what counts as supporting evidence
    expected_gap_direction: GapDirection = "indeterminate"
    result: Optional[str] = None        # what happened after testing (None = untested)
    evidence_strength: EvidenceStrength = "none"
    final_classification: FinalClassification = "pending"
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — initial population 2026-03-21
# ---------------------------------------------------------------------------

REGISTRY: list[NegRiskHypothesis] = [

    NegRiskHypothesis(
        hyp_id="NEG-001",
        category="sum_constraint",
        source=(
            "Polymarket neg-risk official docs and neg-risk-ctf-adapter repo. "
            "Structural no-arbitrage requirement: sum of all outcome YES prices = 1.0."
        ),
        raw_claim=(
            "In live neg-risk markets, the sum of all outcome YES bid/ask midpoints "
            "persistently deviates from 1.0 by a measurable amount — not just rounding. "
            "This deviation may be upward (sum > 1.0, collective overpricing) or downward "
            "(sum < 1.0, collective underpricing). "
            "The direction and magnitude are unknown without measurement."
        ),
        codified_form=(
            "metric: implied_sum = sum(mid_yes_i for each outcome i). "
            "  where mid_yes_i = (best_bid_yes_i + best_ask_yes_i) / 2. "
            "  fallback: use outcomePrices from Gamma if CLOB book unavailable. "
            "metric: constraint_gap = implied_sum - 1.0. "
            "metric: abs_gap = |constraint_gap|. "
            "threshold: abs_gap > 0.01 is non-trivial (> rounding error). "
            "threshold: abs_gap > 0.03 may survive 1-leg round-trip fee (~0.02). "
            "classification: "
            "  abs_gap < 0.005  → AT_CONSTRAINT (within rounding) "
            "  abs_gap < 0.01   → BOUNDARY "
            "  abs_gap >= 0.01  → CONSTRAINT_VIOLATION "
        ),
        test_method=(
            "Run discovery.py to fetch all active neg-risk events. "
            "For each event, run normalizer.py to compute implied_sum and constraint_gap. "
            "Use CLOB mid prices where available; fall back to Gamma outcomePrices. "
            "Log result per event with timestamp to data/research/neg_risk/. "
            "Repeat scan >= 5 times across >= 2 calendar days. "
            "Paper only — no order submission."
        ),
        expected_value=(
            "If constraint_gap > 0.01 appears in >= 50% of scanned events, "
            "the sum constraint is structurally loose in this market structure. "
            "This establishes the research foundation for all other hypotheses. "
            "If gaps are consistently near zero, downgrade the entire line."
        ),
        possible_failure_mode=(
            "Gamma outcomePrices may not reflect CLOB mid accurately. "
            "CLOB book for some legs may be empty (no bids or asks). "
            "Small events may have implied_sum near 1.0 trivially due to rounding at 0.5. "
            "Sum constraint may be enforced by the adapter contract making gaps unachievable."
        ),
        evidence_standard=(
            "Minimum: abs_gap > 0.01 observed in >= 5 distinct events across >= 3 scan sessions. "
            "Gap must come from CLOB mid prices, not Gamma outcomePrices alone. "
            "At least 2 of those events must have >= 3 outcomes (not just binary)."
        ),
        expected_gap_direction="indeterminate",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes=(
            "Foundational hypothesis. All other NEG hypotheses depend on this being confirmed. "
            "Test first. If constraint gaps are negligible, park the entire line."
        ),
    ),

    NegRiskHypothesis(
        hyp_id="NEG-002",
        category="conversion_arbitrage",
        source=(
            "Polymarket/neg-risk-ctf-adapter repo — ConvertPositions function. "
            "Official mechanism: holding NO on all outcomes except one is equivalent to YES on that one."
        ),
        raw_claim=(
            "The NO→YES conversion via the neg-risk adapter means that the cost of buying "
            "NO on all outcomes except outcome_i should equal the cost of buying YES on outcome_i. "
            "If cost(NO_all_except_i) != P_yes_i, a pure structural inconsistency exists. "
            "This is distinct from the sum constraint — it is a per-outcome equivalent position check."
        ),
        codified_form=(
            "For each outcome i in a neg-risk event with N outcomes: "
            "  cost_no_bundle_i = sum(ask_no_j for j != i) "
            "    where ask_no_j = 1 - bid_yes_j (NO price ≈ 1 - YES bid). "
            "  cost_yes_i = ask_yes_i (direct YES purchase). "
            "  equivalent_gap_i = cost_yes_i - cost_no_bundle_i. "
            "  If equivalent_gap_i > 0: buying YES directly is more expensive than NO bundle. "
            "  If equivalent_gap_i < 0: buying NO bundle is more expensive than direct YES. "
            "  threshold: |equivalent_gap_i| > 0.02 after 1-leg fee estimate."
        ),
        test_method=(
            "For each normalized NegRiskEvent with >= 3 outcomes: "
            "  compute equivalent_gap_i for all outcomes i. "
            "  log equivalent_gap_i per outcome. "
            "  record which outcome has the largest gap. "
            "Paper only — no order submission. "
            "Requires CLOB book fetch for accurate bid/ask. "
            "Run on >= 10 events across >= 3 scan sessions."
        ),
        expected_value=(
            "If equivalent_gap_i > 0.02 for any outcome in >= 3 events, "
            "the adapter conversion creates a persistent pricing inconsistency. "
            "This is the strongest signal for escalation — it means the structure "
            "allows a cheaper path to the same position."
        ),
        possible_failure_mode=(
            "CLOB bid on YES may be zero (no buyers) making NO price computation inaccurate. "
            "Transaction cost of NO bundle involves N-1 legs vs 1 leg for direct YES — "
            "multi-leg execution cost may exceed the gap. "
            "Adapter conversion has gas cost on-chain which may dwarf the pricing gap."
        ),
        evidence_standard=(
            "Minimum: |equivalent_gap_i| > 0.02 in >= 3 distinct events. "
            "Gap must be computed from CLOB bid/ask (not Gamma outcomePrices). "
            "Must be observed in events with >= 3 outcomes. "
            "Gas cost of adapter conversion must be documented and accounted for."
        ),
        expected_gap_direction="indeterminate",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes=(
            "Depends on NEG-001 confirmation. Test after NEG-001 shows abs_gap > 0.01. "
            "On-chain gas cost is a significant unknown — research this separately. "
            "CLOB-only version (no on-chain conversion) may be structurally different."
        ),
    ),

    NegRiskHypothesis(
        hyp_id="NEG-003",
        category="temporal_pattern",
        source=(
            "General prediction market microstructure: prices converge to {0, 1} near resolution. "
            "Hypothesis: constraint gap may be larger far from resolution and smaller near resolution."
        ),
        raw_claim=(
            "The implied_sum constraint gap in neg-risk markets is not constant over the "
            "event lifecycle. It may be larger when the event has many days remaining "
            "(uncertainty high, prices loosely anchored) and smaller near resolution "
            "(prices converge toward {0, 1} as the outcome becomes clearer). "
            "If this pattern is real, time-to-resolution is a predictor of gap magnitude."
        ),
        codified_form=(
            "metric: days_to_resolution = (end_date - scan_timestamp).days. "
            "metric: abs_gap (from NEG-001). "
            "experiment: bucket events by days_to_resolution: "
            "  bucket_far: > 30 days "
            "  bucket_mid: 7-30 days "
            "  bucket_near: <= 7 days "
            "  bucket_immediate: <= 1 day "
            "measure: mean(abs_gap) per bucket across >= 3 scan sessions. "
            "hypothesis: mean_gap(bucket_far) > mean_gap(bucket_near)."
        ),
        test_method=(
            "From logged scan results, group events by days_to_resolution bucket. "
            "Compute mean(abs_gap) and std(abs_gap) per bucket. "
            "Require >= 5 events per bucket before drawing conclusions. "
            "Run analysis from outputs/tables.py against accumulated log data. "
            "Paper only — analytical pass over logged data."
        ),
        expected_value=(
            "If mean gap is significantly larger in bucket_far (> 30 days), "
            "the optimal window to observe structural inconsistencies is early in the event lifecycle. "
            "This constrains the discovery module to prefer events with > 30 days remaining."
        ),
        possible_failure_mode=(
            "Sample sizes may be too small per bucket in initial scans. "
            "End dates may be unreliable in Gamma API (market resolves early, end date not updated). "
            "Pattern may be event-category specific rather than universal."
        ),
        evidence_standard=(
            "Minimum: >= 5 events per bucket across >= 3 scan sessions. "
            "Mean gap difference between bucket_far and bucket_near must be > 0.005. "
            "Difference must hold in >= 2 of 3 scan sessions independently."
        ),
        expected_gap_direction="indeterminate",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes="Depends on NEG-001 data accumulation. Purely analytical — no new data fetching needed.",
    ),

    NegRiskHypothesis(
        hyp_id="NEG-004",
        category="directional_bias",
        source=(
            "Cross-market logical-constraint family (Family 2) — systematic mispricing direction. "
            "Applied to neg-risk: leading outcome (highest P_yes) may be systematically overpriced."
        ),
        raw_claim=(
            "In neg-risk multi-outcome markets, the outcome with the highest implied probability "
            "(the 'leading outcome' — most likely to resolve YES) may be persistently overpriced "
            "relative to the constraint, while trailing outcomes are collectively underpriced. "
            "This would manifest as: sum > 1.0 AND the leading outcome contributes the most to the excess."
        ),
        codified_form=(
            "For each neg-risk event with constraint_gap > 0.01: "
            "  leading_outcome = outcome with highest mid_yes price. "
            "  leading_contribution = mid_yes(leading) - (1 / N) for N outcomes. "
            "  trailing_sum = sum(mid_yes_i for all non-leading outcomes). "
            "  directional_tag: "
            "    'leading_overpriced' if leading_contribution > 0 AND constraint_gap > 0 "
            "    'trailing_overpriced' if trailing_sum > (N-1)/N AND constraint_gap > 0 "
            "    'mixed' otherwise "
            "aggregate: count per directional_tag across all events with abs_gap > 0.01. "
            "threshold: leading_overpriced_rate > 0.60 across >= 10 events → systematic."
        ),
        test_method=(
            "From logged normalized events with abs_gap > 0.01, "
            "compute directional_tag for each. "
            "Accumulate across scan sessions. "
            "Once >= 10 events with abs_gap > 0.01 are logged, compute leading_overpriced_rate. "
            "Paper only — analytical pass over logged data."
        ),
        expected_value=(
            "If leading_overpriced_rate > 0.60, candidate construction becomes directional: "
            "always consider the leading outcome as the potential sell side. "
            "This constrains the search space and improves signal quality."
        ),
        possible_failure_mode=(
            "Leading outcome may simply reflect real market consensus, not systematic mispricing. "
            "N=2 (binary) events make 'leading' and 'sum > 1.0' trivially correlated. "
            "Sample may be dominated by one event category (e.g., all political)."
        ),
        evidence_standard=(
            "Minimum: >= 10 events with abs_gap > 0.01 across >= 3 scan sessions. "
            "Must include events with N >= 3 outcomes. "
            "leading_overpriced_rate > 0.60, not merely > 0.50."
        ),
        expected_gap_direction="sum_above_one",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes="Depends on NEG-001 producing sufficient events with abs_gap > 0.01.",
    ),

    NegRiskHypothesis(
        hyp_id="NEG-005",
        category="execution_quality",
        source=(
            "Family 2 failure mode: liquidity ceiling prevents closing at meaningful size. "
            "Hypothesis: neg-risk markets may have sufficient depth for paper-scale positions."
        ),
        raw_claim=(
            "Neg-risk multi-outcome markets may have better CLOB depth than expected because "
            "they attract broader participation (more outcomes → more bettor types). "
            "Paper-scale position sizing ($20–$100 per leg) may be feasible in at least "
            "the leading outcome leg. "
            "This is not assumed — it must be measured."
        ),
        codified_form=(
            "For each normalized NegRiskEvent: "
            "  per_leg_depth_metric: "
            "    ask_depth_i = sum(size * price for levels where cumulative_cost <= PAPER_SIZE_USD) "
            "    where PAPER_SIZE_USD = 50.0 (paper position reference). "
            "  min_leg_depth = min(ask_depth_i for all i). "
            "  max_leg_depth = max(ask_depth_i for all i). "
            "  depth_uniformity = min_leg_depth / max_leg_depth. "
            "  liquidity_tag: "
            "    'SUFFICIENT_ALL': min_leg_depth >= 20 "
            "    'SUFFICIENT_LEADING': max_leg_depth >= 20 and min_leg_depth < 20 "
            "    'THIN_ALL': max_leg_depth < 20 "
            "  spread_i = ask_yes_i - bid_yes_i (best levels only). "
            "  max_spread = max(spread_i for all i)."
        ),
        test_method=(
            "For each event passing structural_check (abs_gap > 0.01): "
            "  fetch CLOB order books for all outcome token_ids. "
            "  compute per_leg_depth_metric for all legs. "
            "  assign liquidity_tag. "
            "  log to data/research/neg_risk/. "
            "Run on >= 10 events across >= 3 scan sessions. "
            "Paper only — book fetch only, no order submission."
        ),
        expected_value=(
            "If liquidity_tag = SUFFICIENT_ALL in >= 3 events with abs_gap > 0.01, "
            "the execution quality barrier is lower than expected. "
            "If THIN_ALL is dominant, the line is research value only — not executable."
        ),
        possible_failure_mode=(
            "CLOB depth may be high at the best level but fall off sharply. "
            "Fetching books for N legs per event multiplies API call volume. "
            "Depth may vary significantly between scans due to book refresh timing."
        ),
        evidence_standard=(
            "Minimum: CLOB books fetched for >= 10 distinct events. "
            "SUFFICIENT_ALL rate documented across all fetched events. "
            "Depth must be measured at >= 2 price levels beyond the best."
        ),
        expected_gap_direction="indeterminate",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes=(
            "CLOB book fetch required for this hypothesis — more expensive than Gamma-only scans. "
            "Gate this test on NEG-001 confirmation to avoid wasting API calls on events "
            "with no structural gap."
        ),
    ),

    NegRiskHypothesis(
        hyp_id="NEG-006",
        category="temporal_pattern",
        source=(
            "Family 2 observation persistence requirement: CAD must be stable over >= 3 scans. "
            "Hypothesis: neg-risk gaps, if real, persist over repeated scans on the same event."
        ),
        raw_claim=(
            "If a constraint gap exists in a neg-risk event, it does not instantly disappear. "
            "Due to market structure (low arbitrageur activity, adapter conversion complexity, "
            "on-chain gas cost), the gap may persist for hours or days. "
            "Persistence is a necessary (but not sufficient) condition for exploitability."
        ),
        codified_form=(
            "For each event observed in >= 3 consecutive scans: "
            "  gap_series = [constraint_gap_t1, constraint_gap_t2, constraint_gap_t3, ...] "
            "  persistence_score = count(|gap_ti| > 0.01 for all t) / len(gap_series). "
            "  reversion_rate = count(|gap_ti+1| < |gap_ti|) / (len(gap_series) - 1). "
            "  classification: "
            "    'PERSISTENT': persistence_score >= 0.80 "
            "    'INTERMITTENT': 0.40 <= persistence_score < 0.80 "
            "    'TRANSIENT': persistence_score < 0.40 "
        ),
        test_method=(
            "From logged scan data, match events by event_id across scan sessions. "
            "Compute gap_series for events with >= 3 observations. "
            "Compute persistence_score and reversion_rate. "
            "Report: how many events are PERSISTENT vs INTERMITTENT vs TRANSIENT. "
            "Paper only — purely analytical over logged data."
        ),
        expected_value=(
            "If PERSISTENT events exist, the discovery module can be run less frequently "
            "and still capture the gap. "
            "If gaps are TRANSIENT, the line requires high-frequency scanning to be useful."
        ),
        possible_failure_mode=(
            "Event_id stability across Gamma API calls may not be guaranteed. "
            "Scans may not be frequent enough to distinguish INTERMITTENT from TRANSIENT. "
            "Gap may appear persistent but simply reflect low liquidity (stale book)."
        ),
        evidence_standard=(
            "Minimum: >= 5 events with >= 3 scan observations each. "
            "PERSISTENT events must have fresh book prices (not stale Gamma outcomePrices). "
            "Persistence classification must use timestamps to confirm scan interval."
        ),
        expected_gap_direction="indeterminate",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes=(
            "Purely analytical — requires accumulated scan log data. "
            "Run after >= 5 scan sessions have been logged. "
            "No additional API calls needed."
        ),
    ),

    NegRiskHypothesis(
        hyp_id="NEG-007",
        category="comparative_structure",
        source=(
            "Family 2 classification — constraint-set structure. "
            "Hypothesis: neg-risk markets may be structurally different from "
            "binary two-leg constraint sets in terms of gap frequency and magnitude."
        ),
        raw_claim=(
            "Binary markets (two-outcome: YES + NO summing to 1.0) are a degenerate "
            "special case of the neg-risk structure. "
            "N >= 3 outcome neg-risk markets have more complex pricing dynamics: "
            "more outcome prices must simultaneously satisfy the constraint. "
            "Markets with more outcomes may show larger or more frequent constraint violations "
            "than binary markets with equivalent liquidity."
        ),
        codified_form=(
            "Partition scanned events by outcome count: "
            "  binary: N = 2 "
            "  small_multi: N = 3–5 "
            "  large_multi: N >= 6 "
            "For each partition, compute: "
            "  mean(abs_gap), std(abs_gap), fraction(abs_gap > 0.01). "
            "Hypothesis: fraction(abs_gap > 0.01) increases with N. "
            "Null: fraction is equal across partitions (no structural difference)."
        ),
        test_method=(
            "From logged scan data, group events by outcome count. "
            "Compute gap distribution per partition. "
            "Require >= 10 events per partition before comparing. "
            "Paper only — analytical over logged data."
        ),
        expected_value=(
            "If large_multi events have significantly higher gap frequency, "
            "the discovery module should prioritize N >= 3 events. "
            "If binary events have similar gaps, they are in scope too."
        ),
        possible_failure_mode=(
            "Large_multi events may be rare — insufficient sample size. "
            "Outcome count may correlate with event category (sports vs politics), "
            "making the comparison confounded."
        ),
        evidence_standard=(
            "Minimum: >= 10 events per partition. "
            "Difference in abs_gap > 0.01 fraction must be > 0.15 between partitions. "
            "Must replicate across >= 2 scan sessions."
        ),
        expected_gap_direction="indeterminate",
        result=None,
        evidence_strength="none",
        final_classification="pending",
        notes=(
            "Purely analytical — requires sufficient accumulated log data. "
            "Run after >= 3 scan sessions covering diverse event types."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_by_id(hyp_id: str) -> NegRiskHypothesis:
    for h in REGISTRY:
        if h.hyp_id == hyp_id:
            return h
    raise KeyError(f"No hypothesis with id {hyp_id!r}")


def get_by_category(category: HypothesisCategory) -> list[NegRiskHypothesis]:
    return [h for h in REGISTRY if h.category == category]


def get_by_classification(classification: FinalClassification) -> list[NegRiskHypothesis]:
    return [h for h in REGISTRY if h.final_classification == classification]


def get_pending() -> list[NegRiskHypothesis]:
    return [h for h in REGISTRY if h.final_classification == "pending"]


def get_active() -> list[NegRiskHypothesis]:
    return [h for h in REGISTRY if h.final_classification in ("pending", "keep")]
