"""
reward_aware_maker_queue_sizing_fillrate_line
polyarb_lab / research_line / analysis-only

Queue-sizing and fill-rate model for the 4 executable-positive survivors.

Inputs:
  - BookSnapshot (from survivor_tracker.py) — one or more rounds
  - RawRewardedMarket (from discovery.py)
  - MarketEVResult (from ev_model.py)

---

ORIGINAL MODEL VIEW (source-of-truth):
  raw_ev = spread_capture + reward_contribution - adverse_selection - inventory
  The EV model (ev_model.py) uses FILL_PROB_BASE = 0.20 as a proxy fill rate.
  Penalties (adverse_selection, inventory) are applied at FULL position size
  regardless of fill probability. This is the confirmed positive-EV basis for
  the 4 survivors.

COUNTERFACTUAL / DIAGNOSTIC HYPOTHESIS (not source-of-truth):
  Observation: with current calibration constants, the fill-component net is:
    fill_net = size × spread × (fill_prob × SPREAD_CAPTURE_FRACTION
                                - ADVERSE_SELECTION_FACTOR - INVENTORY_FACTOR)
             = size × spread × (fill_prob × 0.5 - 0.50)

  At fill_prob = 0.20 (base): fill_net = size × spread × (0.10 - 0.50) < 0.

  Counterfactual: IF actual fill probability is much lower than 0.20 due to
  deep queue ahead, AND IF model penalties over-state real risk for low-fill
  markets, THEN a hypothetical "penalty-corrected" EV would approach
  reward_contribution alone.

  This is labelled cf_reward_only_ev below. It is a DIAGNOSTIC LOWER BOUND
  for the fill contribution — not a confirmed model revision.

  Reasons this counterfactual may be too strong:
  (1) ADVERSE_SELECTION_FACTOR = 0.30 was set conservatively but IS the source-of-truth
      calibration for this probe layer. It has not been replaced.
  (2) Low fill rate ≠ no adverse selection. A rare fill may occur precisely when
      a better-informed taker has edge — the conditional adverse selection on rare
      fills can be HIGHER than the unconditional estimate.
  (3) The penalty structure (not scaled by fill_prob) may intentionally model
      opportunity cost and inventory risk even for unexecuted orders.
  (4) REWARD_POOL_SHARE_FRACTION = 0.05 is itself a rough proxy. If actual reward
      share is lower, the counterfactual EV could also be lower than reward_contribution.

No API calls in this module. No order submission. Pure computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .discovery import RawRewardedMarket
from .ev_model import (
    ADVERSE_SELECTION_FACTOR,
    FILL_PROB_BASE,
    FILL_PROB_MAX,
    INVENTORY_FACTOR,
    REWARD_POOL_SHARE_FRACTION,
    SPREAD_CAPTURE_FRACTION,
    MarketEVResult,
)
from .survivor_tracker import BookSnapshot

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

# Fraction of daily volume assumed to hit the best bid price level.
# Conservative: 25%. Active markets may be 30-40%; thin markets 10-15%.
# Single largest source of uncertainty in fill-rate estimation.
# This is an assumption, not an observed value.
VOLUME_AT_BEST_FRACTION = 0.25

# Size sensitivity ladder (shares)
SIZE_LADDER = [1, 5, 10, 25, 50, 100, 200, 500]

# Minimum rounds for stale detection comparison
STALE_IDENTICAL_ROUNDS = 3


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SizeSensitivityPoint:
    quote_size: float
    p_fill: Optional[float]                # queue-model fill probability at this size
    expected_fill_shares_daily: Optional[float]
    fill_wait_hours: Optional[float]
    # Original model view
    model_fill_net: Optional[float]        # size × spread × (fill_prob × 0.5 - 0.50)
    original_model_ev: Optional[float]     # reward_contribution + model_fill_net
    # Counterfactual view (DIAGNOSTIC — not source-of-truth)
    cf_fill_net_if_penalties_scale: float  # 0.0 if penalties were scaled by fill_prob
    cf_reward_only_ev: float               # reward_contribution + 0 (counterfactual ceiling)


@dataclass
class QueueFillAnalysis:
    """
    Queue-sizing and fill-rate analysis for one survivor.

    Two EV views are tracked:
      original_model_ev  — from ev_model.py (source-of-truth)
      cf_reward_only_ev  — counterfactual: reward_contribution only (DIAGNOSTIC)

    The counterfactual is a hypothesis about what EV would be if the model's
    penalty structure were corrected for low fill rates. It is NOT a replacement
    for the original model view.
    """
    slug: str
    full_slug: str

    # Market parameters
    min_quote_size: float
    reward_max_spread: float
    midpoint: Optional[float]
    reward_daily_rate_usdc: float
    reward_contribution: float             # estimated_reward_contribution from EV model

    # Book state (latest snapshot)
    best_bid: Optional[float]
    best_ask: Optional[float]
    queue_ahead_bid: Optional[float]       # shares at best bid before our order
    queue_ahead_ask: Optional[float]       # shares at best ask before our order
    total_bid_depth: float
    total_ask_depth: float
    bid_levels: int
    ask_levels: int
    bid_levels_in_spread: int
    ask_levels_in_spread: int

    # Volume (from discovery — may be stale or unavailable)
    volume_24hr: Optional[float]
    vol_at_bid_level: Optional[float]      # volume_24hr × VOLUME_AT_BEST_FRACTION

    # Fill model at min_quote_size
    p_fill_min_size: Optional[float]
    expected_fill_shares_daily: Optional[float]
    fill_wait_hours: Optional[float]

    # EV: ORIGINAL MODEL VIEW (source-of-truth)
    original_model_ev: float               # raw_ev from ev_model.py
    model_fill_net_at_min_size: Optional[float]   # fill terms net (negative by calibration)
    model_fill_prob_assumed: float         # fill_prob used by ev_model (0.20 base or 0.30 wide)

    # EV: COUNTERFACTUAL / DIAGNOSTIC (not source-of-truth)
    cf_reward_only_ev: float               # reward_contribution alone (upper bound hypothesis)
    cf_delta_vs_model: float               # cf_reward_only_ev - original_model_ev
    # Reasons the counterfactual may be too strong (populated per-survivor)
    cf_caveats: list[str] = field(default_factory=list)

    # Recommendation
    recommended_quote_size: float = 0.0
    recommended_reasoning: str = ""

    # Size sensitivity
    size_sensitivity: list[SizeSensitivityPoint] = field(default_factory=list)

    # Freshness / stale detection
    freshness_verdict: str = "SINGLE_SNAPSHOT"
    freshness_notes: str = ""
    n_rounds_checked: int = 1
    rounds_identical: bool = False

    # Ranking (reward_contribution primary, p_fill secondary)
    ranking_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "full_slug": self.full_slug,
            "min_quote_size": self.min_quote_size,
            "reward_max_spread": self.reward_max_spread,
            "midpoint": self.midpoint,
            "reward_daily_rate_usdc": self.reward_daily_rate_usdc,
            "reward_contribution": self.reward_contribution,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "queue_ahead_bid": self.queue_ahead_bid,
            "queue_ahead_ask": self.queue_ahead_ask,
            "total_bid_depth": self.total_bid_depth,
            "total_ask_depth": self.total_ask_depth,
            "bid_levels": self.bid_levels,
            "ask_levels": self.ask_levels,
            "bid_levels_in_spread": self.bid_levels_in_spread,
            "ask_levels_in_spread": self.ask_levels_in_spread,
            "volume_24hr": self.volume_24hr,
            "vol_at_bid_level": self.vol_at_bid_level,
            "p_fill_min_size": round(self.p_fill_min_size, 6) if self.p_fill_min_size is not None else None,
            "expected_fill_shares_daily": round(self.expected_fill_shares_daily, 4) if self.expected_fill_shares_daily is not None else None,
            "fill_wait_hours": round(self.fill_wait_hours, 2) if self.fill_wait_hours is not None else None,
            "ev_original_model": {
                "original_model_ev": self.original_model_ev,
                "model_fill_net_at_min_size": self.model_fill_net_at_min_size,
                "model_fill_prob_assumed": self.model_fill_prob_assumed,
            },
            "ev_counterfactual_diagnostic": {
                "cf_reward_only_ev": self.cf_reward_only_ev,
                "cf_delta_vs_model": round(self.cf_delta_vs_model, 6),
                "cf_caveats": self.cf_caveats,
                "label": "COUNTERFACTUAL/DIAGNOSTIC — not source-of-truth",
            },
            "recommended_quote_size": self.recommended_quote_size,
            "recommended_reasoning": self.recommended_reasoning,
            "freshness_verdict": self.freshness_verdict,
            "freshness_notes": self.freshness_notes,
            "n_rounds_checked": self.n_rounds_checked,
            "rounds_identical": self.rounds_identical,
            "ranking_score": round(self.ranking_score, 6),
            "size_sensitivity": [
                {
                    "quote_size": p.quote_size,
                    "p_fill": round(p.p_fill, 4) if p.p_fill is not None else None,
                    "expected_fill_shares_daily": round(p.expected_fill_shares_daily, 2) if p.expected_fill_shares_daily is not None else None,
                    "fill_wait_hours": round(p.fill_wait_hours, 1) if p.fill_wait_hours is not None else None,
                    "original_model_ev": round(p.original_model_ev, 6) if p.original_model_ev is not None else None,
                    "cf_reward_only_ev": round(p.cf_reward_only_ev, 6),
                }
                for p in self.size_sensitivity
            ],
        }


# ---------------------------------------------------------------------------
# Fill model helpers
# ---------------------------------------------------------------------------

def _compute_p_fill(
    queue_ahead: Optional[float],
    vol_at_level: Optional[float],
    our_size: float,
) -> Optional[float]:
    """
    Price-time priority queue fill probability estimate.

    p_fill = min(1.0, vol_at_level / (queue_ahead + our_size))

    This is a MODEL ESTIMATE. Uncertainty sources:
    - vol_at_level is itself estimated (VOLUME_AT_BEST_FRACTION × volume_24hr)
    - volume_24hr from discovery may be stale
    - Price-time priority assumption may not hold exactly
    """
    if queue_ahead is None or vol_at_level is None or vol_at_level <= 0:
        return None
    denom = queue_ahead + our_size
    if denom <= 0:
        return 1.0
    return min(1.0, vol_at_level / denom)


def _fill_wait_hours(p_fill: Optional[float]) -> Optional[float]:
    """Hours to expect one full fill at this fill probability."""
    if p_fill is None or p_fill <= 0:
        return None
    return 24.0 / p_fill


def _model_fill_net(size: float, spread: Optional[float], p_fill: Optional[float]) -> Optional[float]:
    """
    EV model's fill-net term using provided p_fill.

    = size × spread × (p_fill × SPREAD_CAPTURE_FRACTION - ADVERSE_SELECTION_FACTOR - INVENTORY_FACTOR)

    This uses the EV model's penalty factors directly. Result is negative
    for all p_fill < 1.0 at current calibration — this is a property of the
    current parameter set, not a universal truth about market making.
    """
    if spread is None or p_fill is None:
        return None
    return round(
        size * spread * (p_fill * SPREAD_CAPTURE_FRACTION - ADVERSE_SELECTION_FACTOR - INVENTORY_FACTOR),
        6,
    )


# ---------------------------------------------------------------------------
# Stale detection
# ---------------------------------------------------------------------------

def _check_freshness(snapshots: list[BookSnapshot]) -> tuple[str, str, bool]:
    """
    Compare multiple rapid book snapshots to detect cached/stale data.

    Returns: (verdict, notes, all_identical)
      LIVE            — at least one field changed across rounds
      STALE_SUSPECT   — all measured fields identical across all rounds
      SINGLE_SNAPSHOT — fewer than 2 valid snapshots
    """
    valid = [s for s in snapshots if s.fetch_ok]
    if len(valid) < 2:
        return "SINGLE_SNAPSHOT", "Only one valid snapshot — cannot assess freshness.", False

    bid_sizes  = [s.top_bid_size for s in valid]
    best_bids  = [s.best_bid for s in valid]
    bid_levels = [s.bid_levels_count for s in valid]
    ask_levels = [s.ask_levels_count for s in valid]

    all_identical = (
        len(set(bid_sizes)) == 1 and bid_sizes[0] is not None
        and len(set(best_bids)) == 1 and best_bids[0] is not None
        and len(set(bid_levels)) == 1
        and len(set(ask_levels)) == 1
    )

    if all_identical:
        verdict = "STALE_SUSPECT"
        notes = (
            f"All {len(valid)} rounds: top_bid_size={bid_sizes[0]}, "
            f"best_bid={best_bids[0]}, bid_levels={bid_levels[0]}, "
            f"ask_levels={ask_levels[0]} — no change detected. "
            f"Data may be cached."
        )
    else:
        changed = []
        if len(set(bid_sizes)) > 1:
            changed.append(f"top_bid_size {bid_sizes}")
        if len(set(best_bids)) > 1:
            changed.append(f"best_bid {best_bids}")
        if len(set(bid_levels)) > 1:
            changed.append(f"bid_levels {bid_levels}")
        verdict = "LIVE"
        notes = "Variation detected: " + "; ".join(changed)

    return verdict, notes, all_identical


# ---------------------------------------------------------------------------
# Core analysis builder
# ---------------------------------------------------------------------------

def _build_cf_caveats(
    market: RawRewardedMarket,
    p_fill: Optional[float],
    queue_ahead: Optional[float],
    volume_24hr: Optional[float],
) -> list[str]:
    """
    Generate per-survivor reasons why the counterfactual may be too strong.
    These are evidence-based observations, not generic disclaimers.
    """
    caveats = []

    # Caveat 1: probe-layer penalty calibration is not replaced
    caveats.append(
        f"ADVERSE_SELECTION_FACTOR={ADVERSE_SELECTION_FACTOR} and "
        f"INVENTORY_FACTOR={INVENTORY_FACTOR} are the current source-of-truth "
        f"calibration. The counterfactual assumes these over-state risk for low-fill "
        f"markets, but this has not been tested against real fill data."
    )

    # Caveat 2: fill_prob signal
    if p_fill is not None and p_fill > 0.05:
        caveats.append(
            f"Queue model estimates p_fill={p_fill:.3f} — not negligibly small. "
            f"At this fill rate, fill contribution is still non-trivial. "
            f"Counterfactual is weaker when p_fill is not near zero."
        )
    elif p_fill is None:
        caveats.append(
            "Fill probability could not be computed (missing volume_24hr or "
            "queue_ahead data). Counterfactual is unverifiable without fill signal."
        )

    # Caveat 3: rare fills may carry higher adverse selection
    if queue_ahead is not None and volume_24hr is not None and queue_ahead > 0:
        queue_to_vol = queue_ahead / max(volume_24hr, 1.0)
        if queue_to_vol > 0.5:
            caveats.append(
                f"Queue ahead ({queue_ahead:.0f}sh) is {queue_to_vol:.1%} of daily volume. "
                f"Rare fills in deep-queue markets often occur on information events — "
                f"conditional adverse selection on the fill may be higher than the "
                f"unconditional ADVERSE_SELECTION_FACTOR={ADVERSE_SELECTION_FACTOR} estimate."
            )

    # Caveat 4: reward pool share is also a proxy
    caveats.append(
        f"REWARD_POOL_SHARE_FRACTION={REWARD_POOL_SHARE_FRACTION} (5%) is itself a probe-layer "
        f"proxy. Actual reward share depends on competitive quoting intensity at runtime. "
        f"If actual share < 5%, cf_reward_only_ev would also be lower."
    )

    return caveats


def build_queue_fill_analysis(
    slug: str,
    market: RawRewardedMarket,
    ev_result: MarketEVResult,
    snapshots: list[BookSnapshot],
) -> QueueFillAnalysis:
    """Build a QueueFillAnalysis for one survivor."""
    valid_snaps = [s for s in snapshots if s.fetch_ok]
    snap = valid_snaps[-1] if valid_snaps else None

    freshness_verdict, freshness_notes, rounds_identical = _check_freshness(snapshots)

    midpoint        = ev_result.midpoint
    reward_max_spread = market.rewards_max_spread_cents / 100.0
    quoted_spread   = ev_result.quoted_spread or 0.0
    min_size        = market.rewards_min_size
    reward_contribution = ev_result.estimated_reward_contribution
    original_model_ev   = ev_result.reward_adjusted_raw_ev

    # Model fill prob used by ev_model (reconstruct from market_is_wide flag proxy)
    market_is_wide = (
        ev_result.best_ask is not None
        and ev_result.best_bid is not None
        and (float(ev_result.best_ask) - float(ev_result.best_bid)) > reward_max_spread
    )
    model_fill_prob_assumed = min(FILL_PROB_BASE * 1.5, FILL_PROB_MAX) if market_is_wide else FILL_PROB_BASE

    # Fill terms from original model
    model_fill_net_at_min = original_model_ev - reward_contribution - ev_result.estimated_maker_rebate_contribution

    # Counterfactual EV (DIAGNOSTIC — not source-of-truth)
    cf_reward_only_ev = reward_contribution
    cf_delta = cf_reward_only_ev - original_model_ev

    # Book state
    queue_ahead_bid = snap.top_bid_size if snap else None
    queue_ahead_ask = snap.top_ask_size if snap else None
    total_bid   = snap.total_bid_depth_shares if snap else 0.0
    total_ask   = snap.total_ask_depth_shares if snap else 0.0
    bid_levels  = snap.bid_levels_count if snap else 0
    ask_levels  = snap.ask_levels_count if snap else 0
    bid_in_spread = snap.bid_levels_in_reward_spread if snap else 0
    ask_in_spread = snap.ask_levels_in_reward_spread if snap else 0
    best_bid    = snap.best_bid if snap else market.best_bid
    best_ask    = snap.best_ask if snap else market.best_ask

    volume_24hr = market.volume_24hr
    vol_at_bid_level = (
        round(volume_24hr * VOLUME_AT_BEST_FRACTION, 4) if volume_24hr is not None else None
    )

    # Fill model
    p_fill_min = _compute_p_fill(queue_ahead_bid, vol_at_bid_level, min_size)
    fill_wait  = _fill_wait_hours(p_fill_min)
    exp_fill   = round(min_size * p_fill_min, 4) if p_fill_min is not None else None

    # Per-survivor counterfactual caveats
    cf_caveats = _build_cf_caveats(market, p_fill_min, queue_ahead_bid, volume_24hr)

    # Recommended size
    recommended_size = min_size
    if cf_delta > 0:
        rec_reason = (
            f"Minimum size ({min_size:.0f}sh). "
            f"Original model EV is lower than counterfactual CF EV by ${cf_delta:.4f} "
            f"due to fill-term drag. Quoting above minimum increases this drag "
            f"without confirmed proportional reward increase. "
            f"If reward program credits proportionally to size, this conclusion requires revision."
        )
    else:
        rec_reason = (
            f"Minimum size ({min_size:.0f}sh). "
            f"Original model EV is positive. Fill drag at minimum size is "
            f"${abs(model_fill_net_at_min or 0):.4f}. "
            f"Larger sizes increase fill drag without confirmed reward increase."
        )

    # Size sensitivity — two views per size point
    size_points = []
    for sz in SIZE_LADDER:
        p_f = _compute_p_fill(queue_ahead_bid, vol_at_bid_level, sz)
        # Use queue-model p_fill for sensitivity; fall back to model base
        p_for_calc = p_f if p_f is not None else model_fill_prob_assumed
        mfn = _model_fill_net(sz, quoted_spread, p_for_calc)
        total_mev = round(reward_contribution + (mfn or 0.0), 6)
        exp_f  = round(sz * p_f, 4) if p_f is not None else None
        wait   = _fill_wait_hours(p_f)
        size_points.append(SizeSensitivityPoint(
            quote_size=sz,
            p_fill=p_f,
            expected_fill_shares_daily=exp_f,
            fill_wait_hours=wait,
            model_fill_net=mfn,
            original_model_ev=total_mev,
            cf_fill_net_if_penalties_scale=0.0,
            cf_reward_only_ev=round(reward_contribution, 6),
        ))

    ranking_score = reward_contribution * (1.0 + (p_fill_min or 0.0))

    return QueueFillAnalysis(
        slug=slug,
        full_slug=market.market_slug,
        min_quote_size=min_size,
        reward_max_spread=reward_max_spread,
        midpoint=midpoint,
        reward_daily_rate_usdc=market.reward_daily_rate_usdc,
        reward_contribution=reward_contribution,
        best_bid=best_bid,
        best_ask=best_ask,
        queue_ahead_bid=queue_ahead_bid,
        queue_ahead_ask=queue_ahead_ask,
        total_bid_depth=total_bid,
        total_ask_depth=total_ask,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        bid_levels_in_spread=bid_in_spread,
        ask_levels_in_spread=ask_in_spread,
        volume_24hr=volume_24hr,
        vol_at_bid_level=vol_at_bid_level,
        p_fill_min_size=p_fill_min,
        expected_fill_shares_daily=exp_fill,
        fill_wait_hours=fill_wait,
        original_model_ev=original_model_ev,
        model_fill_net_at_min_size=model_fill_net_at_min,
        model_fill_prob_assumed=model_fill_prob_assumed,
        cf_reward_only_ev=cf_reward_only_ev,
        cf_delta_vs_model=cf_delta,
        cf_caveats=cf_caveats,
        recommended_quote_size=recommended_size,
        recommended_reasoning=rec_reason,
        size_sensitivity=size_points,
        freshness_verdict=freshness_verdict,
        freshness_notes=freshness_notes,
        n_rounds_checked=len(snapshots),
        rounds_identical=rounds_identical,
        ranking_score=ranking_score,
    )


def rank_survivors(analyses: list[QueueFillAnalysis]) -> list[QueueFillAnalysis]:
    """Return analyses sorted by ranking_score descending."""
    return sorted(analyses, key=lambda a: a.ranking_score, reverse=True)
