"""
reward_aware_maker_probe — Module 2: EV Model
polyarb_lab / research_line / probe-only

Lightweight probe-layer EV decomposition for fee-enabled rewarded markets.

Formula (probe model — NOT an execution engine):
  reward_adjusted_raw_ev
    = estimated_spread_capture
    + estimated_reward_contribution
    + estimated_maker_rebate_contribution   # conservative 0.0 for this probe pass
    - inventory_penalty
    - adverse_selection_penalty

This is a raw/probe layer only.
Purpose: answer whether fee-enabled rewarded markets produce a non-empty
positive reward-adjusted raw EV pool.

Hard constraints:
  - No live orders. No order submission. No inventory controller.
  - No quote management engine.
  - No promotion logic beyond classifying economics_class.
  - Pure computation. No network calls. No state.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .discovery import RawRewardedMarket

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe model parameters
# Conservative calibration for a first-pass raw EV probe.
# Do not tune these to force positives — keep conservative.
# ---------------------------------------------------------------------------

# Fraction of the spread captured per filled round-trip (maker fills one side).
# 0.5 = half-spread per fill (standard maker model).
SPREAD_CAPTURE_FRACTION = 0.5

# Fill probability baseline and drivers (same as maker_rewarded_mm.py calibration).
FILL_PROB_BASE = 0.20
FILL_PROB_ACTIVITY_WEIGHT = 0.35
FILL_PROB_LIQUIDITY_WEIGHT = 0.25
FILL_PROB_STABILITY_WEIGHT = 0.20
FILL_PROB_MIN = 0.03
FILL_PROB_MAX = 0.90

# Adverse-selection penalty factor: fraction of spread × position size.
ADVERSE_SELECTION_FACTOR = 0.30

# Inventory penalty factor: fraction of spread × position size.
INVENTORY_FACTOR = 0.20

# Maker rebate contribution: conservative 0.0 for this probe pass.
# Fee-enabled markets on Polymarket charge takers; maker rebate is not
# confirmed in Gamma API payload. Probe conservatively sets this to 0.0.
# A later execution-layer study should quantify the actual rebate if any.
MAKER_REBATE_CONTRIBUTION = 0.0

# Reward eligibility proxy: fraction of daily_rate credited given spread fit.
# This models the probability of being within rewardsMaxSpread and winning
# a proportional share of the reward pool.
# Conservative: assume we capture 5% of the daily pool (competitive market).
REWARD_POOL_SHARE_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Economics classification
# ---------------------------------------------------------------------------

ECON_POSITIVE_RAW_EV = "POSITIVE_RAW_EV"
ECON_NEGATIVE_RAW_EV = "NEGATIVE_RAW_EV"
ECON_REJECTED_NO_BOOK = "REJECTED_NO_BOOK"
ECON_REJECTED_NO_REWARD = "REJECTED_NO_REWARD"
ECON_REJECTED_SPREAD_TOO_WIDE = "REJECTED_SPREAD_TOO_WIDE"

# Rejection reason codes (list, may have multiple)
RC_NO_USABLE_BOOK = "RC_NO_USABLE_BOOK"
RC_REWARD_METADATA_MISSING = "RC_REWARD_METADATA_MISSING"
RC_SPREAD_EXCEEDS_REWARD_MAX = "RC_SPREAD_EXCEEDS_REWARD_MAX"
RC_NON_POSITIVE_NET_EV = "RC_NON_POSITIVE_NET_EV"

WATCHLIST_MONITOR = "MONITOR"
WATCHLIST_SKIP = "SKIP"
WATCHLIST_RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketEVResult:
    """Full EV decomposition result for one fee-enabled rewarded market."""
    # Identity
    market_slug: str
    event_slug: str
    category: str
    fees_enabled: bool

    # Reward config summary
    reward_config_summary: dict

    # Book state
    best_bid: Optional[float]
    best_ask: Optional[float]
    midpoint: Optional[float]
    tick_size: float
    quoted_spread: Optional[float]

    # EV components
    estimated_spread_capture: float
    estimated_reward_contribution: float
    estimated_maker_rebate_contribution: float  # 0.0 in this probe pass
    inventory_penalty: float
    adverse_selection_penalty: float
    reward_adjusted_raw_ev: float

    # Classification
    economics_class: str
    rejection_reason_codes: list[str]
    watchlist_recommendation: str

    # Probe metadata
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "market_slug": self.market_slug,
            "event_slug": self.event_slug,
            "category": self.category,
            "fees_enabled": self.fees_enabled,
            "reward_config_summary": self.reward_config_summary,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midpoint": self.midpoint,
            "tick_size": self.tick_size,
            "quoted_spread": self.quoted_spread,
            "estimated_spread_capture": self.estimated_spread_capture,
            "estimated_reward_contribution": self.estimated_reward_contribution,
            "estimated_maker_rebate_contribution": self.estimated_maker_rebate_contribution,
            "inventory_penalty": self.inventory_penalty,
            "adverse_selection_penalty": self.adverse_selection_penalty,
            "reward_adjusted_raw_ev": self.reward_adjusted_raw_ev,
            "economics_class": self.economics_class,
            "rejection_reason_codes": self.rejection_reason_codes,
            "watchlist_recommendation": self.watchlist_recommendation,
            "computed_at": self.computed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------

def _reward_config_summary(market: RawRewardedMarket) -> dict:
    return {
        "reward_daily_rate_usdc": market.reward_daily_rate_usdc,
        "rewards_min_size_shares": market.rewards_min_size,
        "rewards_max_spread_cents": market.rewards_max_spread_cents,
        "clob_rewards_count": len(market.clob_rewards_raw),
    }


def _rejected(
    market: RawRewardedMarket,
    economics_class: str,
    reason_codes: list[str],
) -> MarketEVResult:
    """Build a fully-rejected result with zero EV components."""
    mid = None
    if market.best_bid is not None and market.best_ask is not None:
        try:
            mid = round((float(market.best_bid) + float(market.best_ask)) / 2.0, 6)
        except Exception:
            mid = None

    return MarketEVResult(
        market_slug=market.market_slug,
        event_slug=market.event_slug,
        category=market.category,
        fees_enabled=market.fees_enabled,
        reward_config_summary=_reward_config_summary(market),
        best_bid=market.best_bid,
        best_ask=market.best_ask,
        midpoint=mid,
        tick_size=0.01,
        quoted_spread=None,
        estimated_spread_capture=0.0,
        estimated_reward_contribution=0.0,
        estimated_maker_rebate_contribution=MAKER_REBATE_CONTRIBUTION,
        inventory_penalty=0.0,
        adverse_selection_penalty=0.0,
        reward_adjusted_raw_ev=0.0,
        economics_class=economics_class,
        rejection_reason_codes=reason_codes,
        watchlist_recommendation=WATCHLIST_SKIP,
    )


def evaluate_market_ev(market: RawRewardedMarket) -> MarketEVResult:
    """
    Compute reward_adjusted_raw_ev for one fee-enabled rewarded market.

    Returns a MarketEVResult with economics_class set to:
      POSITIVE_RAW_EV        — raw pool candidate
      NEGATIVE_RAW_EV        — evaluated but net negative
      REJECTED_NO_BOOK       — book missing or crossed
      REJECTED_NO_REWARD     — reward metadata missing/zero
      REJECTED_SPREAD_TOO_WIDE — current spread > rewardsMaxSpread
    """
    # --- Gate 1: usable book ---
    if not market.has_usable_book():
        return _rejected(market, ECON_REJECTED_NO_BOOK, [RC_NO_USABLE_BOOK])

    # --- Gate 2: reward metadata ---
    if (
        market.reward_daily_rate_usdc <= 0.0
        or market.rewards_min_size <= 0.0
        or market.rewards_max_spread_cents <= 0.0
    ):
        return _rejected(market, ECON_REJECTED_NO_REWARD, [RC_REWARD_METADATA_MISSING])

    best_bid = float(market.best_bid)  # type: ignore[arg-type]
    best_ask = float(market.best_ask)  # type: ignore[arg-type]
    midpoint = round((best_bid + best_ask) / 2.0, 6)
    tick_size = 0.01  # standard Polymarket tick

    current_spread = best_ask - best_bid
    reward_max_spread_price = market.rewards_max_spread_cents / 100.0  # convert cents → price

    # --- Quote spread selection ---
    # If the market is already trading within the reward window, match the
    # current spread (we are at best price, fill probability is reasonable).
    # If the market spread is wider than reward_max_spread, quote at exactly
    # reward_max_spread — our quote sits inside the current spread, improving
    # fill probability, while still earning reward eligibility.
    # We never hard-reject on spread width: a market maker chooses their own
    # quote width, not the current market spread.
    market_is_wide = current_spread > reward_max_spread_price
    quoted_spread = reward_max_spread_price if market_is_wide else current_spread
    quote_size = market.rewards_min_size  # minimum reward-eligible size in shares

    # --- Fill probability proxy ---
    # When quoting inside a wide market (market_is_wide=True), we have best
    # price in the book — fill probability is modestly higher than base.
    # When matching a tight market, base rate applies.
    fill_probability = (
        min(FILL_PROB_BASE * 1.5, FILL_PROB_MAX)
        if market_is_wide
        else FILL_PROB_BASE
    )

    # --- EV components ---
    estimated_spread_capture = round(
        quote_size * quoted_spread * fill_probability * SPREAD_CAPTURE_FRACTION, 6
    )

    # Reward contribution: conservative fraction of daily rate
    # spread_eligibility = 1.0: quoted_spread ≤ reward_max_spread by construction
    spread_eligibility = 1.0
    estimated_reward_contribution = round(
        market.reward_daily_rate_usdc * REWARD_POOL_SHARE_FRACTION * spread_eligibility, 6
    )

    # Maker rebate: conservative 0.0 (not confirmed from Gamma payload)
    estimated_maker_rebate_contribution = round(MAKER_REBATE_CONTRIBUTION, 6)

    # Penalties (conservative probe estimates without activity data)
    # Adverse selection: higher uncertainty without activity data → use 0.30 factor
    adverse_selection_penalty = round(
        quote_size * quoted_spread * ADVERSE_SELECTION_FACTOR, 6
    )
    inventory_penalty = round(
        quote_size * quoted_spread * INVENTORY_FACTOR, 6
    )

    # --- Net EV ---
    reward_adjusted_raw_ev = round(
        estimated_spread_capture
        + estimated_reward_contribution
        + estimated_maker_rebate_contribution
        - inventory_penalty
        - adverse_selection_penalty,
        6,
    )

    if reward_adjusted_raw_ev > 0.0:
        economics_class = ECON_POSITIVE_RAW_EV
        rejection_reason_codes: list[str] = []
        watchlist_rec = WATCHLIST_RESEARCH_CANDIDATE
    else:
        economics_class = ECON_NEGATIVE_RAW_EV
        rejection_reason_codes = [RC_NON_POSITIVE_NET_EV]
        watchlist_rec = WATCHLIST_MONITOR  # monitor — may flip positive

    return MarketEVResult(
        market_slug=market.market_slug,
        event_slug=market.event_slug,
        category=market.category,
        fees_enabled=market.fees_enabled,
        reward_config_summary=_reward_config_summary(market),
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        tick_size=tick_size,
        quoted_spread=round(quoted_spread, 6),
        estimated_spread_capture=estimated_spread_capture,
        estimated_reward_contribution=estimated_reward_contribution,
        estimated_maker_rebate_contribution=estimated_maker_rebate_contribution,
        inventory_penalty=inventory_penalty,
        adverse_selection_penalty=adverse_selection_penalty,
        reward_adjusted_raw_ev=reward_adjusted_raw_ev,
        economics_class=economics_class,
        rejection_reason_codes=rejection_reason_codes,
        watchlist_recommendation=watchlist_rec,
    )


def evaluate_batch(markets: list[RawRewardedMarket]) -> list[MarketEVResult]:
    """Evaluate all markets and return one result per market."""
    results: list[MarketEVResult] = []
    for mkt in markets:
        try:
            results.append(evaluate_market_ev(mkt))
        except Exception as exc:
            logger.warning("EV evaluation failed for %s: %s", mkt.market_slug, exc)
            results.append(_rejected(mkt, ECON_REJECTED_NO_BOOK, [RC_NO_USABLE_BOOK]))
    return results


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_ev_summary(
    raw_markets: list[RawRewardedMarket],
    results: list[MarketEVResult],
) -> dict:
    """
    Build the top-level summary fields required by the probe output spec.

    Returns:
      rewarded_market_count              — total discovered (all rewarded, not only fee-enabled)
      fee_enabled_rewarded_market_count  — fee-enabled subset passed to EV model
      websocket_books_collected          — always 0 for probe (no WS in probe mode)
      raw_candidates_by_family           — {family: count} for probe output conventions
      positive_raw_maker_candidates      — count of POSITIVE_RAW_EV results
      best_raw_candidate                 — market_slug with highest reward_adjusted_raw_ev
      best_net_candidate                 — same if positive, else None
    """
    pos_results = [r for r in results if r.economics_class == ECON_POSITIVE_RAW_EV]
    best_raw: Optional[str] = None
    if results:
        best = max(results, key=lambda r: r.reward_adjusted_raw_ev)
        if best.reward_adjusted_raw_ev > 0.0:
            best_raw = best.market_slug

    counts: Counter = Counter(r.economics_class for r in results)

    return {
        "rewarded_market_count": len(raw_markets),
        "fee_enabled_rewarded_market_count": len(raw_markets),
        "websocket_books_collected": 0,  # probe mode: no WS
        "raw_candidates_by_family": {
            "reward_aware_single_market_maker": len(pos_results),
        },
        "positive_raw_maker_candidates": len(pos_results),
        "repeated_positive_raw_maker_candidates": None,  # requires multi-cycle
        "best_raw_candidate": best_raw,
        "best_net_candidate": best_raw,  # same as raw in probe (no execution cost model)
        "economics_class_counts": {
            ECON_POSITIVE_RAW_EV:          counts.get(ECON_POSITIVE_RAW_EV, 0),
            ECON_NEGATIVE_RAW_EV:          counts.get(ECON_NEGATIVE_RAW_EV, 0),
            ECON_REJECTED_NO_BOOK:         counts.get(ECON_REJECTED_NO_BOOK, 0),
            ECON_REJECTED_NO_REWARD:       counts.get(ECON_REJECTED_NO_REWARD, 0),
            ECON_REJECTED_SPREAD_TOO_WIDE: counts.get(ECON_REJECTED_SPREAD_TOO_WIDE, 0),
        },
    }
