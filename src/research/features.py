"""
Pure feature computation functions for the belief-aware ranker.

All functions are stateless unless snapshot_history is passed explicitly.
No imports from Track A execution modules.

Feature families:
  1. Logit-space     — prior vs ask in log-odds space
  2. Uncertainty     — p*(1-p) proxy
  3. Spread/depth    — book quality features
  4. Fragility       — composite book fragility score
  5. Persistence     — how many rounds a slug has appeared
  6. Belief-vol      — price-movement proxy from snapshot history
  7. Composite       — weighted sum with explicit per-component breakdown
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List

from .theory import logit as safe_logit, uncertainty as _theory_uncertainty


# ---------------------------------------------------------------------------
# 1. Logit-space primitives
# (safe_logit is imported from theory.py — canonical implementation)
# ---------------------------------------------------------------------------


def logit_features(p_yes_prior: float, yes_ask: float) -> dict:
    """
    Logit-space representation of prior vs market ask.

    logit_spread > 0: market charges more than prior implies (typical when book is thin).
    logit_spread < 0: market is discounting relative to prior (rare, possible mispricing signal).
    """
    logit_p = safe_logit(p_yes_prior)
    logit_a = safe_logit(yes_ask)
    return {
        "logit_p_yes":   logit_p,
        "logit_ask_yes": logit_a,
        "logit_spread":  logit_a - logit_p,
    }


# ---------------------------------------------------------------------------
# 2. Uncertainty proxy
# (imported from theory.py as _theory_uncertainty; re-exported as uncertainty)
# ---------------------------------------------------------------------------

uncertainty = _theory_uncertainty


# ---------------------------------------------------------------------------
# 3. Spread / depth features
# ---------------------------------------------------------------------------

def spread_features(yes_ask: float, no_ask: float, edge: float) -> dict:
    """
    spread_cents: the cost of buying both sides (negative when edge > 0).
    spread_over_edge_ratio: |spread| / |edge| — how much of the edge is consumed by spread.
    High ratio → even if edge exists, it is largely offset by the bid-ask cost.
    """
    spread_cents = (yes_ask + no_ask - 1.0) * 100.0
    edge_cents   = abs(edge * 100.0)
    ratio        = abs(spread_cents) / max(edge_cents, 0.1)
    return {
        "spread_cents":           spread_cents,
        "spread_over_edge_ratio": ratio,
    }


def depth_near_ask(asks: list, best_ask: float, depth_band: float = 0.05) -> float:
    """
    Sum of share sizes in ask levels within depth_band above best ask.
    More conservative than just taking asks[0].size.
    """
    total = 0.0
    for level in asks:
        try:
            price = float(level.get("price", 0))
            size  = float(level.get("size",  0))
            if price <= best_ask + depth_band:
                total += size
        except (TypeError, ValueError):
            continue
    return total


def depth_features(yes_depth: float, no_depth: float) -> dict:
    """
    depth_imbalance: (+1) = all depth on YES side, (-1) = all depth on NO side.
    Large imbalance may indicate one side is illiquid.
    """
    total     = yes_depth + no_depth
    imbalance = (yes_depth - no_depth) / (total + 1e-6)
    return {
        "depth_imbalance": imbalance,
        "total_depth":     total,
    }


# ---------------------------------------------------------------------------
# 4. Fragility score
# ---------------------------------------------------------------------------

def fragility_score(
    yes_ask: float,
    no_ask: float,
    yes_depth: float,
    no_depth: float,
    min_size: float,
    min_depth_multiple: float = 3.0,
) -> float:
    """
    Composite fragility in [0, 1]. Higher = less trustworthy book.

    Two components weighted equally:
      thinness:   how close yes_ask + no_ask is to 1.0 (zero-edge boundary).
                  A book at exactly combined=1.0 has no edge buffer at all.
      depth_gap:  how much depth falls short of min_size * min_depth_multiple.
                  Underfilled on either side → execution risk.
    """
    combined  = yes_ask + no_ask
    # thinness ramps from 0 (combined far from 1.0) to 1 (combined = 1.0)
    thinness  = max(0.0, min(1.0, 1.0 - abs(combined - 1.0) * 4.0))

    required  = min_size * min_depth_multiple
    yes_short = max(0.0, (required - yes_depth) / (required + 1e-6))
    no_short  = max(0.0, (required - no_depth)  / (required + 1e-6))
    depth_gap = (yes_short + no_short) / 2.0

    return round(0.5 * thinness + 0.5 * depth_gap, 6)


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------

def persistence_rounds(slug: str, history: Dict[str, List[dict]]) -> int:
    """Number of snapshots recorded for this slug. 0 on cold start."""
    return len(history.get(slug, []))


# ---------------------------------------------------------------------------
# 6. Belief-vol proxy
# ---------------------------------------------------------------------------

def belief_vol_proxy(slug: str, history: Dict[str, List[dict]]) -> float:
    """
    Stdev of p_yes across recent snapshots for this slug.
    Returns 0.0 if fewer than 2 snapshots available (cold start).

    Interpretation:
      high belief_vol → price is actively moving → belief is updating → opportunity more likely real
      low belief_vol  → price is stale / stable  → less signal value
    """
    snaps = history.get(slug, [])
    prices = [s.get("p_yes", 0.5) for s in snaps if isinstance(s, dict)]
    if len(prices) < 2:
        return 0.0
    try:
        return round(statistics.stdev(prices), 8)
    except statistics.StatisticsError:
        return 0.0


# ---------------------------------------------------------------------------
# 7. Composite score
# ---------------------------------------------------------------------------

# Weights are explicit constants — adjust here, not buried in scoring logic.
SCORE_WEIGHTS: Dict[str, float] = {
    "edge_cents":       3.0,    # core signal: every cent of positive edge is worth 3 pts
    "uncertainty":     15.0,    # prefer ambiguous markets (max +3.75 at p=0.50)
    "spread_over_edge": -3.0,   # penalize wide spread relative to edge
    "fragility":       -10.0,   # penalize thin / underfilled books
    "persistence":      2.0,    # reward candidates that persist across rounds
    "belief_vol":      40.0,    # reward price movement (active belief updating)
}


def composite_score_and_explanation(
    edge: float,
    uncertainty_val: float,
    spread_over_edge_ratio: float,
    fragility: float,
    persistence: int,
    belief_vol: float,
) -> tuple[float, str]:
    """
    Returns (composite_score, explanation_string).

    Each component's contribution is explicit in both the score and explanation.
    Weights are defined in SCORE_WEIGHTS above for full auditability.
    """
    w = SCORE_WEIGHTS

    c_edge    = edge * 100.0          * w["edge_cents"]
    c_uncert  = uncertainty_val       * w["uncertainty"]
    c_spr     = spread_over_edge_ratio * w["spread_over_edge"]
    c_frag    = fragility             * w["fragility"]
    c_persist = persistence           * w["persistence"]
    c_bvol    = belief_vol            * w["belief_vol"]

    score = c_edge + c_uncert + c_spr + c_frag + c_persist + c_bvol

    expl = (
        f"edge={edge*100:+.2f}¢→{c_edge:+.2f} "
        f"u={uncertainty_val:.3f}→{c_uncert:+.2f} "
        f"spr_ratio={spread_over_edge_ratio:.2f}→{c_spr:+.2f} "
        f"frag={fragility:.3f}→{c_frag:+.2f} "
        f"persist={persistence}→{c_persist:+.2f} "
        f"bvol={belief_vol:.4f}→{c_bvol:+.2f} "
        f"= {score:+.4f}"
    )

    return round(score, 6), expl
