"""
reward_aware_reward_share_validation_line
polyarb_lab / research_line / validation-only

Validates whether REWARD_POOL_SHARE_FRACTION = 0.05 (5%) is a conservative,
fair, or optimistic estimate for the 4 executable-positive survivors.

Mechanism:
  Polymarket reward programs distribute the daily_rate_usdc proportionally
  among makers who are quoting within rewardsMaxSpread.  The share a single
  maker at minimum size receives is approximately:

      implied_share ≈ min_size / (total_eligible_depth_in_spread + min_size)

  Where "eligible depth" = sum of all bid (or ask) shares currently sitting
  within the reward spread window in the live order book.

  The model uses REWARD_POOL_SHARE_FRACTION = 0.05 (5%).  This module
  measures the depth-implied share directly from the book and compares.

Share assumption verdict per survivor:
  CONSERVATIVE : implied_avg_share > model (we would earn more than assumed)
  FAIR         : implied within ±30% of model (0.035 ≤ implied ≤ 0.065)
  OPTIMISTIC   : implied_avg_share < model (we would earn less than assumed)
  UNVERIFIABLE : book data stale or depth unavailable

No order submission.  No mainline imports.  Pure computation + targeted reads.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from .discovery import RawRewardedMarket, _safe_float, CLOB_BOOK_PATH
from .ev_model import REWARD_POOL_SHARE_FRACTION, MarketEVResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

# Verdict bands relative to REWARD_POOL_SHARE_FRACTION (0.05)
# FAIR band: model share × (1 ± FAIR_TOLERANCE)
FAIR_TOLERANCE = 0.30           # ±30% relative → 0.035 … 0.065 at 5% model

# Minimum eligible depth threshold: if total eligible depth < this fraction
# of min_size, the market has near-empty competition → share near 1.0
MIN_ELIGIBLE_DEPTH_RATIO = 0.01  # 1% of min_size = essentially no competition

# Rate stability: reward rate considered stable if delta < this fraction
RATE_STABILITY_THRESHOLD_PCT = 5.0  # 5% variation allowed


# ---------------------------------------------------------------------------
# Per-round snapshot
# ---------------------------------------------------------------------------

@dataclass
class RewardShareSnapshot:
    """
    One book-fetch round for one survivor.

    Computes eligible depth (shares within reward_max_spread window) and
    derives the implied reward share fraction for a min-size quote.
    """
    token_id: str
    round_num: int
    fetched_at: datetime
    fetch_ok: bool

    # Market params (passed in — not fetched from book)
    min_size: float
    reward_max_spread: float
    reward_daily_rate_usdc: float

    # Book state
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]            # best_ask - best_bid
    midpoint: Optional[float]

    # Eligible depth: sum of shares of all levels within reward window
    # Bid side: all bid prices where bid >= best_ask - reward_max_spread
    total_eligible_bid_depth: float    # shares
    eligible_bid_level_count: int      # number of qualifying bid levels

    # Ask side: all ask prices where ask <= best_bid + reward_max_spread
    total_eligible_ask_depth: float    # shares
    eligible_ask_level_count: int      # number of qualifying ask levels

    # Implied share fractions (min_size / (eligible_depth + min_size))
    # If total eligible depth is 0 (no competition), share = 1.0 (we are alone)
    implied_bid_share: Optional[float]   # based on bid side depth
    implied_ask_share: Optional[float]   # based on ask side depth
    implied_avg_share: Optional[float]   # average of bid + ask sides

    # Model comparison
    model_share_fraction: float = REWARD_POOL_SHARE_FRACTION
    share_bias: Optional[float] = None   # implied_avg - model (+ = conservative, - = optimistic)

    # Reward contribution comparison
    model_reward_contribution: float = 0.0     # rate × model_share
    implied_reward_contribution: Optional[float] = None  # rate × implied_avg


def _compute_implied_share(eligible_depth: float, min_size: float) -> float:
    """
    Implied reward share for a maker quoting min_size when eligible_depth
    already sits in the book.

    share = min_size / (eligible_depth + min_size)

    If eligible_depth is effectively zero (no competition), share → 1.0.
    """
    total = eligible_depth + min_size
    if total <= 0:
        return 1.0
    return min_size / total


def fetch_reward_share_snapshot(
    host: str,
    token_id: str,
    min_size: float,
    reward_max_spread: float,
    reward_daily_rate_usdc: float,
    round_num: int,
    client: httpx.Client,
) -> RewardShareSnapshot:
    """
    Fetch one book snapshot and compute reward-eligible depth + implied share.

    Eligible bids: price >= best_ask - reward_max_spread
    Eligible asks: price <= best_bid + reward_max_spread

    These match the same eligibility criteria as survivor_tracker.py's
    bid_levels_in_reward_spread, but summing DEPTH (shares) not just level count.
    """
    fetched_at = datetime.now(timezone.utc)
    model_contrib = round(reward_daily_rate_usdc * REWARD_POOL_SHARE_FRACTION, 6)

    _empty = RewardShareSnapshot(
        token_id=token_id,
        round_num=round_num,
        fetched_at=fetched_at,
        fetch_ok=False,
        min_size=min_size,
        reward_max_spread=reward_max_spread,
        reward_daily_rate_usdc=reward_daily_rate_usdc,
        best_bid=None,
        best_ask=None,
        spread=None,
        midpoint=None,
        total_eligible_bid_depth=0.0,
        eligible_bid_level_count=0,
        total_eligible_ask_depth=0.0,
        eligible_ask_level_count=0,
        implied_bid_share=None,
        implied_ask_share=None,
        implied_avg_share=None,
        model_share_fraction=REWARD_POOL_SHARE_FRACTION,
        share_bias=None,
        model_reward_contribution=model_contrib,
        implied_reward_contribution=None,
    )

    if not token_id:
        return _empty

    url = f"{host.rstrip('/')}{CLOB_BOOK_PATH}"
    try:
        resp = client.get(url, params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []

        # Parse best bid and ask
        best_bid: Optional[float] = None
        best_ask: Optional[float] = None

        for i, lv in enumerate(bids):
            if not isinstance(lv, dict):
                continue
            p = _safe_float(lv.get("price"), default=None)  # type: ignore[arg-type]
            if p is not None and i == 0:
                best_bid = p
                break

        for i, lv in enumerate(asks):
            if not isinstance(lv, dict):
                continue
            p = _safe_float(lv.get("price"), default=None)  # type: ignore[arg-type]
            if p is not None and i == 0:
                best_ask = p
                break

        spread: Optional[float] = None
        midpoint: Optional[float] = None
        if best_bid is not None and best_ask is not None:
            spread = round(best_ask - best_bid, 6)
            midpoint = round((best_bid + best_ask) / 2.0, 6)

        # Eligible bid depth: sum shares of bids >= best_ask - reward_max_spread
        eligible_bid_depth = 0.0
        eligible_bid_count = 0
        if best_ask is not None:
            bid_floor = best_ask - reward_max_spread
            for lv in bids:
                if not isinstance(lv, dict):
                    continue
                p = _safe_float(lv.get("price"), default=None)  # type: ignore[arg-type]
                s = _safe_float(lv.get("size"), default=0.0)
                if p is not None and p >= bid_floor:
                    eligible_bid_depth += s
                    eligible_bid_count += 1

        # Eligible ask depth: sum shares of asks <= best_bid + reward_max_spread
        eligible_ask_depth = 0.0
        eligible_ask_count = 0
        if best_bid is not None:
            ask_ceil = best_bid + reward_max_spread
            for lv in asks:
                if not isinstance(lv, dict):
                    continue
                p = _safe_float(lv.get("price"), default=None)  # type: ignore[arg-type]
                s = _safe_float(lv.get("size"), default=0.0)
                if p is not None and p <= ask_ceil:
                    eligible_ask_depth += s
                    eligible_ask_count += 1

        # Implied shares
        implied_bid: Optional[float] = None
        implied_ask: Optional[float] = None
        implied_avg: Optional[float] = None

        if best_ask is not None:
            implied_bid = _compute_implied_share(eligible_bid_depth, min_size)
        if best_bid is not None:
            implied_ask = _compute_implied_share(eligible_ask_depth, min_size)
        if implied_bid is not None and implied_ask is not None:
            implied_avg = round((implied_bid + implied_ask) / 2.0, 6)
        elif implied_bid is not None:
            implied_avg = implied_bid
        elif implied_ask is not None:
            implied_avg = implied_ask

        share_bias: Optional[float] = None
        implied_contrib: Optional[float] = None
        if implied_avg is not None:
            share_bias = round(implied_avg - REWARD_POOL_SHARE_FRACTION, 6)
            implied_contrib = round(reward_daily_rate_usdc * implied_avg, 6)

        return RewardShareSnapshot(
            token_id=token_id,
            round_num=round_num,
            fetched_at=fetched_at,
            fetch_ok=True,
            min_size=min_size,
            reward_max_spread=reward_max_spread,
            reward_daily_rate_usdc=reward_daily_rate_usdc,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            midpoint=midpoint,
            total_eligible_bid_depth=round(eligible_bid_depth, 4),
            eligible_bid_level_count=eligible_bid_count,
            total_eligible_ask_depth=round(eligible_ask_depth, 4),
            eligible_ask_level_count=eligible_ask_count,
            implied_bid_share=round(implied_bid, 6) if implied_bid is not None else None,
            implied_ask_share=round(implied_ask, 6) if implied_ask is not None else None,
            implied_avg_share=implied_avg,
            model_share_fraction=REWARD_POOL_SHARE_FRACTION,
            share_bias=share_bias,
            model_reward_contribution=model_contrib,
            implied_reward_contribution=implied_contrib,
        )

    except Exception as exc:
        logger.debug("reward_share fetch failed token=%s: %s", token_id, exc)
        return _empty


# ---------------------------------------------------------------------------
# Multi-round analysis
# ---------------------------------------------------------------------------

@dataclass
class RewardShareAnalysis:
    """
    Multi-round reward share analysis for one survivor.

    Aggregates across N book-fetch rounds to assess:
      1. Reward share assumption validity (5% vs implied)
      2. Reward rate persistence (is daily_rate_usdc stable?)
      3. Overall verdict on share assumption
    """
    slug: str
    full_slug: str
    min_size: float
    reward_max_spread: float
    reward_daily_rate_usdc: float
    n_rounds: int

    snapshots: list[RewardShareSnapshot] = field(default_factory=list)

    # Aggregated across valid rounds (mean)
    mean_eligible_bid_depth: Optional[float] = None
    mean_eligible_ask_depth: Optional[float] = None
    mean_implied_bid_share: Optional[float] = None
    mean_implied_ask_share: Optional[float] = None
    mean_implied_avg_share: Optional[float] = None

    # Model comparison
    model_share_fraction: float = REWARD_POOL_SHARE_FRACTION
    mean_share_bias: Optional[float] = None   # positive = conservative, negative = optimistic

    model_reward_contribution: float = 0.0
    mean_implied_reward_contribution: Optional[float] = None

    # Reward rate persistence (rate is fixed for current round from endpoint)
    # Since rewards endpoint is called once per discovery pass, persistence
    # across rounds here means book-implied rate stability not endpoint rate.
    # Rate delta across rounds (endpoint rate is constant per discovery call).
    rate_min: float = 0.0
    rate_max: float = 0.0
    rate_delta_pct: float = 0.0
    rate_stable: bool = True

    # Depth freshness: did eligible depth change across rounds?
    bid_depth_min: float = 0.0
    bid_depth_max: float = 0.0
    bid_depth_delta_pct: float = 0.0
    depth_shows_activity: bool = False   # True if eligible depth changed across rounds

    # Share assumption verdict
    share_assumption_verdict: str = "UNVERIFIABLE"  # CONSERVATIVE / FAIR / OPTIMISTIC / UNVERIFIABLE
    share_assumption_notes: str = ""
    share_assumption_direction: str = ""    # how we would revise reward contribution

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "full_slug": self.full_slug,
            "min_size": self.min_size,
            "reward_max_spread": self.reward_max_spread,
            "reward_daily_rate_usdc": self.reward_daily_rate_usdc,
            "n_rounds": self.n_rounds,
            "model_share_fraction": self.model_share_fraction,
            "mean_eligible_bid_depth": self.mean_eligible_bid_depth,
            "mean_eligible_ask_depth": self.mean_eligible_ask_depth,
            "mean_implied_bid_share": self.mean_implied_bid_share,
            "mean_implied_ask_share": self.mean_implied_ask_share,
            "mean_implied_avg_share": self.mean_implied_avg_share,
            "mean_share_bias": self.mean_share_bias,
            "model_reward_contribution": self.model_reward_contribution,
            "mean_implied_reward_contribution": self.mean_implied_reward_contribution,
            "rate_stable": self.rate_stable,
            "rate_delta_pct": self.rate_delta_pct,
            "depth_shows_activity": self.depth_shows_activity,
            "bid_depth_delta_pct": self.bid_depth_delta_pct,
            "share_assumption_verdict": self.share_assumption_verdict,
            "share_assumption_notes": self.share_assumption_notes,
            "share_assumption_direction": self.share_assumption_direction,
            "snapshots": [
                {
                    "round": s.round_num,
                    "fetch_ok": s.fetch_ok,
                    "best_bid": s.best_bid,
                    "best_ask": s.best_ask,
                    "spread": s.spread,
                    "eligible_bid_depth": s.total_eligible_bid_depth,
                    "eligible_bid_levels": s.eligible_bid_level_count,
                    "eligible_ask_depth": s.total_eligible_ask_depth,
                    "eligible_ask_levels": s.eligible_ask_level_count,
                    "implied_bid_share": s.implied_bid_share,
                    "implied_ask_share": s.implied_ask_share,
                    "implied_avg_share": s.implied_avg_share,
                    "share_bias": s.share_bias,
                    "implied_reward_contribution": s.implied_reward_contribution,
                }
                for s in self.snapshots
            ],
        }


def build_reward_share_analysis(
    slug: str,
    market: RawRewardedMarket,
    snapshots: list[RewardShareSnapshot],
) -> RewardShareAnalysis:
    """
    Aggregate multi-round snapshots into a RewardShareAnalysis.

    Determines share assumption verdict based on mean implied share vs model.
    """
    model_contrib = round(market.reward_daily_rate_usdc * REWARD_POOL_SHARE_FRACTION, 6)
    analysis = RewardShareAnalysis(
        slug=slug,
        full_slug=market.market_slug,
        min_size=market.rewards_min_size,
        reward_max_spread=market.rewards_max_spread_cents / 100.0,
        reward_daily_rate_usdc=market.reward_daily_rate_usdc,
        n_rounds=len(snapshots),
        snapshots=snapshots,
        model_share_fraction=REWARD_POOL_SHARE_FRACTION,
        model_reward_contribution=model_contrib,
        rate_min=market.reward_daily_rate_usdc,
        rate_max=market.reward_daily_rate_usdc,
        rate_delta_pct=0.0,
        rate_stable=True,
    )

    valid = [s for s in snapshots if s.fetch_ok]
    if not valid:
        analysis.share_assumption_verdict = "UNVERIFIABLE"
        analysis.share_assumption_notes = "All book fetches failed — no data."
        return analysis

    # Aggregated means
    bid_depths = [s.total_eligible_bid_depth for s in valid]
    ask_depths = [s.total_eligible_ask_depth for s in valid]

    # GOVERNANCE FIX: empty eligible depth must NOT be treated as CONSERVATIVE.
    # If all valid rounds show zero eligible depth on both sides, the reward spread
    # window is structurally empty — the current book best_bid/best_ask sit outside
    # the reward window (extreme market price, e.g. bid=0.01 ask=0.99 with 4-cent window).
    # implied_share = 100% in this case is a COUNTERFACTUAL UPPER BOUND only.
    # It does NOT mean we would capture 100% of rewards — it means no one is
    # currently quoting in the window at all.  Use official market_competitiveness
    # data (via official_rewards_truth module) to estimate actual competition.
    all_bid_zero = all(d == 0.0 for d in bid_depths)
    all_ask_zero = all(d == 0.0 for d in ask_depths)
    if all_bid_zero and all_ask_zero:
        # Determine why the window is empty
        sample = valid[0]
        spread_str = f"{sample.spread:.4f}" if sample.spread is not None else "N/A"
        window_str = f"{market.rewards_max_spread_cents:.0f}c"
        bid_str = f"{sample.best_bid}" if sample.best_bid is not None else "N/A"
        ask_str = f"{sample.best_ask}" if sample.best_ask is not None else "N/A"
        analysis.share_assumption_verdict = "EMPTY_REWARD_WINDOW"
        analysis.share_assumption_notes = (
            f"All {len(valid)} round(s): eligible_bid_depth=0 / eligible_ask_depth=0. "
            f"Book shows bid={bid_str} ask={ask_str} spread={spread_str}. "
            f"Reward window is {window_str} wide. No current maker quotes within window. "
            f"implied_share=100% is COUNTERFACTUAL UPPER BOUND only — "
            f"not a current reward capture estimate."
        )
        analysis.share_assumption_direction = (
            "UNVERIFIABLE via book-depth proxy. "
            "Use official market_competitiveness from /rewards/markets/{condition_id} "
            "to estimate actual competition level. Do not treat as CONSERVATIVE."
        )
        # Still compute means so caller can display raw numbers
        analysis.mean_eligible_bid_depth = 0.0
        analysis.mean_eligible_ask_depth = 0.0
        # Do NOT set mean_implied_avg_share — leave as None to signal no valid estimate
        return analysis
    bid_shares = [s.implied_bid_share for s in valid if s.implied_bid_share is not None]
    ask_shares = [s.implied_ask_share for s in valid if s.implied_ask_share is not None]
    avg_shares = [s.implied_avg_share for s in valid if s.implied_avg_share is not None]

    analysis.mean_eligible_bid_depth = round(sum(bid_depths) / len(bid_depths), 4)
    analysis.mean_eligible_ask_depth = round(sum(ask_depths) / len(ask_depths), 4)
    if bid_shares:
        analysis.mean_implied_bid_share = round(sum(bid_shares) / len(bid_shares), 6)
    if ask_shares:
        analysis.mean_implied_ask_share = round(sum(ask_shares) / len(ask_shares), 6)
    if avg_shares:
        analysis.mean_implied_avg_share = round(sum(avg_shares) / len(avg_shares), 6)
        analysis.mean_share_bias = round(analysis.mean_implied_avg_share - REWARD_POOL_SHARE_FRACTION, 6)
        analysis.mean_implied_reward_contribution = round(
            market.reward_daily_rate_usdc * analysis.mean_implied_avg_share, 6
        )

    # Depth activity check — did bid eligible depth change across rounds?
    if len(bid_depths) >= 2 and max(bid_depths) > 0:
        analysis.bid_depth_min = min(bid_depths)
        analysis.bid_depth_max = max(bid_depths)
        analysis.bid_depth_delta_pct = round(
            (max(bid_depths) - min(bid_depths)) / max(bid_depths) * 100.0, 2
        )
        analysis.depth_shows_activity = analysis.bid_depth_delta_pct > 1.0

    # Rate stability (endpoint rate is constant per run, but check book spread stability)
    spreads = [s.spread for s in valid if s.spread is not None]
    if spreads:
        analysis.rate_min = min(spreads)
        analysis.rate_max = max(spreads)
        if max(spreads) > 0:
            analysis.rate_delta_pct = round(
                (max(spreads) - min(spreads)) / max(spreads) * 100.0, 2
            )
        analysis.rate_stable = analysis.rate_delta_pct < RATE_STABILITY_THRESHOLD_PCT

    # Determine verdict
    mean_implied = analysis.mean_implied_avg_share
    if mean_implied is None:
        analysis.share_assumption_verdict = "UNVERIFIABLE"
        analysis.share_assumption_notes = (
            "Implied share could not be computed (book data insufficient)."
        )
        analysis.share_assumption_direction = "Unknown — cannot revise reward contribution."
        return analysis

    fair_low  = REWARD_POOL_SHARE_FRACTION * (1 - FAIR_TOLERANCE)  # 0.035
    fair_high = REWARD_POOL_SHARE_FRACTION * (1 + FAIR_TOLERANCE)  # 0.065

    if mean_implied >= fair_high:
        analysis.share_assumption_verdict = "CONSERVATIVE"
        analysis.share_assumption_notes = (
            f"Implied avg share {mean_implied:.1%} exceeds model {REWARD_POOL_SHARE_FRACTION:.1%} "
            f"by {analysis.mean_share_bias:.1%}. "
            f"Low competition in reward spread window: total eligible depth "
            f"({analysis.mean_eligible_bid_depth:.1f}sh bid / {analysis.mean_eligible_ask_depth:.1f}sh ask) "
            f"is thin relative to min_size ({market.rewards_min_size:.0f}sh)."
        )
        analysis.share_assumption_direction = (
            f"Model UNDERSTATES reward contribution. "
            f"Implied: ${analysis.mean_implied_reward_contribution:.4f}/day vs "
            f"model: ${model_contrib:.4f}/day. "
            f"Corrected EV would be higher than original_model_ev."
        )
    elif mean_implied >= fair_low:
        analysis.share_assumption_verdict = "FAIR"
        analysis.share_assumption_notes = (
            f"Implied avg share {mean_implied:.1%} is within ±{FAIR_TOLERANCE:.0%} "
            f"of model {REWARD_POOL_SHARE_FRACTION:.1%} "
            f"(bias = {analysis.mean_share_bias:+.1%}). "
            f"Eligible depth: {analysis.mean_eligible_bid_depth:.1f}sh bid / "
            f"{analysis.mean_eligible_ask_depth:.1f}sh ask."
        )
        analysis.share_assumption_direction = (
            f"Model is approximately correct. "
            f"Implied: ${analysis.mean_implied_reward_contribution:.4f}/day vs "
            f"model: ${model_contrib:.4f}/day."
        )
    else:
        analysis.share_assumption_verdict = "OPTIMISTIC"
        analysis.share_assumption_notes = (
            f"Implied avg share {mean_implied:.1%} is BELOW model {REWARD_POOL_SHARE_FRACTION:.1%} "
            f"(bias = {analysis.mean_share_bias:+.1%}). "
            f"High competition in reward spread window: eligible depth "
            f"({analysis.mean_eligible_bid_depth:.1f}sh bid / {analysis.mean_eligible_ask_depth:.1f}sh ask) "
            f"is large relative to min_size ({market.rewards_min_size:.0f}sh). "
            f"Model overstates reward contribution."
        )
        analysis.share_assumption_direction = (
            f"Model OVERSTATES reward contribution. "
            f"Implied: ${analysis.mean_implied_reward_contribution:.4f}/day vs "
            f"model: ${model_contrib:.4f}/day. "
            f"Corrected EV may be lower. Reassess POSITIVE_RAW_EV classification."
        )

    return analysis


def rank_by_implied_contribution(analyses: list[RewardShareAnalysis]) -> list[RewardShareAnalysis]:
    """Sort by mean_implied_reward_contribution descending. UNVERIFIABLE last."""
    def _key(a: RewardShareAnalysis) -> float:
        if a.mean_implied_reward_contribution is None:
            return -999.0
        return a.mean_implied_reward_contribution
    return sorted(analyses, key=_key, reverse=True)
