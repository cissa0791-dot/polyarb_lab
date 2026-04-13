"""
maker_scan_quote_planner.py — Reservation-aware quote planner for maker-first scan.

Wraps InventoryQuoteEngine + evaluate_quote_feasibility into a single call
for use by the wide maker-MM scan path and related maker research tools.

Accepts market dicts in the wide-scan snake_case format:
    rewards_min_size   (float) — min shares for reward eligibility
    rewards_max_spread (float) — max spread from mid in cents
    reward_daily_rate  (float) — USDC/day (already summed from clob_rewards)
    best_bid / best_ask (float)

Outputs: MakerQuotePlan — reservation quote (logit-space) + reward feasibility.

Hard constraints:
  - No network calls. No state. Pure computation.
  - Falls back to ineligible plan on any error.
  - Does NOT modify the input dict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.scanner.inventory_quote_engine import (
    InventoryQuoteEngine,
    QuoteSuggestion,
)
from src.scanner.reward_quote_feasibility import (
    RewardFeasibility,
    evaluate_quote_feasibility,
)
from src.research.reward_eval import RewardConfig

logger = logging.getLogger("polyarb.scanner.maker_scan_quote_planner")

# Conservative prior for x-space belief variance when no live data is available.
# Mirrors belief_var_estimator.PRIOR_VARIANCE — ~±5¢ uncertainty around 50¢ mid.
PRIOR_BELIEF_VAR: float = 0.04

# Minimum reward rate (USDC/day) to consider a market reward-eligible.
_MIN_RATE: float = 5.0

# Polymarket-calibrated engine parameters.
#
# Academic defaults (γ=1.0, k=0.5) produce inventory_penalty_term ≈ 4.4 in
# x-space, which at mid_p=0.50 translates to ~40¢ half-spread — far wider than
# any Polymarket reward max_spread (typically 1.5–4.5¢).
#
# Calibrated values (γ=0.1, k=2.0):
#   - inventory_penalty_term = (2/2.0)*log(1+0.1/2.0) ≈ 0.049 in x-space
#   - At mid_p=0.50: half_spread_p ≈ 0.25 * 0.025 ≈ 0.0025 (0.25¢ per side)
#   - With PRIOR_BELIEF_VAR + T=0.5: total bid/ask spread ≈ 1.3¢ — within bounds
#
# These reflect: (a) low inventory risk aversion on Polymarket (positions reset
# frequently), (b) relatively deep books on reward-eligible markets (moderate k).
MAKER_RISK_AVERSION: float = 0.1
MAKER_K: float = 2.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MakerQuotePlan:
    """
    Reservation-aware maker quote for one market.

    eligible=True means the proposed quote satisfies all reward constraints
    (both spread legs within max_spread_cents AND size >= min_size_shares).
    """
    market_slug        : str
    mid_p              : float
    quote_bid          : float    # sigmoid(bid_x) from reservation formula
    quote_ask          : float    # sigmoid(ask_x)
    quote_size         : float    # max(rewards_min_size, 20)
    bid_spread_cents   : float    # |mid_p - bid_p| * 100
    ask_spread_cents   : float    # |ask_p - mid_p| * 100
    max_spread_cents   : float    # from reward config
    eligible           : bool     # True iff spreads and size within reward bounds
    feasibility_reason : str      # human-readable from RewardFeasibility
    daily_rate_usdc    : float    # from reward config
    belief_var_used    : float    # σ² passed to InventoryQuoteEngine
    horizon_left       : float    # fraction of market lifetime remaining
    inventory          : float    # signed net position used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_quote(
    m             : dict[str, Any],
    inventory     : float = 0.0,
    horizon_left  : float = 0.5,
    belief_var    : Optional[float] = None,
    risk_aversion : float = MAKER_RISK_AVERSION,
    k             : float = MAKER_K,
) -> MakerQuotePlan:
    """
    Compute a reservation-aware maker quote for market m.

    Parameters
    ----------
    m            : market dict with best_bid, best_ask, rewards_min_size,
                   rewards_max_spread, reward_daily_rate (wide-scan snake_case format).
    inventory    : current signed net position (positive=long YES, negative=short).
    horizon_left : fraction of market lifetime remaining [0, 1].
    belief_var   : x-space variance of belief. Uses PRIOR_BELIEF_VAR if None.
    risk_aversion: γ for InventoryQuoteEngine.
    k            : market depth constant for InventoryQuoteEngine.

    Returns an ineligible MakerQuotePlan on any error (never raises).
    """
    try:
        return _plan(m, inventory, horizon_left, belief_var, risk_aversion, k)
    except Exception as exc:
        slug = str(m.get("market_slug") or m.get("slug") or "")
        logger.debug("plan_quote failed for %s: %s", slug, exc)
        return _ineligible_plan(m, reason="evaluation_error")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _plan(
    m             : dict[str, Any],
    inventory     : float,
    horizon_left  : float,
    belief_var    : Optional[float],
    risk_aversion : float,
    k             : float,
) -> "MakerQuotePlan":
    slug     = str(m.get("market_slug") or m.get("slug") or "")
    best_bid = float(m.get("best_bid") or 0.0)
    best_ask = float(m.get("best_ask") or 0.0)

    if best_bid <= 0.0 or best_ask <= best_bid:
        return _ineligible_plan(m, reason="invalid_book")

    mid_p = (best_bid + best_ask) / 2.0

    # Build RewardConfig from snake_case wide-scan fields
    reward_config = _reward_config_from_market(m)
    if reward_config is None or not reward_config.has_rewards:
        return _ineligible_plan(m, reason="no_reward_program")

    quote_size = max(float(reward_config.min_size_shares), 20.0)
    σ2 = belief_var if belief_var is not None else PRIOR_BELIEF_VAR

    # Compute reservation quote in logit space
    engine = InventoryQuoteEngine(risk_aversion=risk_aversion, k=k)
    quote = engine.compute_quote(
        mid_p        = mid_p,
        belief_var   = σ2,
        horizon_left = max(min(horizon_left, 1.0), 0.0),
        inventory    = inventory,
    )
    if quote is None:
        return _ineligible_plan(m, reason="engine_failed")

    # Feasibility: are bid/ask within reward spread bounds AND size eligible?
    feasibility = evaluate_quote_feasibility(
        quote         = quote,
        reward_config = reward_config,
        proposed_size = int(quote_size),
    )

    return MakerQuotePlan(
        market_slug        = slug,
        mid_p              = mid_p,
        quote_bid          = quote.bid_p,
        quote_ask          = quote.ask_p,
        quote_size         = quote_size,
        bid_spread_cents   = feasibility.bid_spread_cents,
        ask_spread_cents   = feasibility.ask_spread_cents,
        max_spread_cents   = feasibility.max_spread_cents,
        eligible           = feasibility.eligible,
        feasibility_reason = feasibility.reason,
        daily_rate_usdc    = feasibility.daily_rate_usdc,
        belief_var_used    = σ2,
        horizon_left       = horizon_left,
        inventory          = inventory,
    )


def _reward_config_from_market(m: dict[str, Any]) -> Optional[RewardConfig]:
    """Build a RewardConfig from a wide-scan snake_case market dict."""
    try:
        min_size   = int(float(m.get("rewards_min_size") or 0))
        max_spread = float(m.get("rewards_max_spread") or 0.0)
        rate       = float(m.get("reward_daily_rate") or 0.0)

        # Fallback: sum from clob_rewards list if reward_daily_rate not present
        if rate <= 0.0:
            clob_rewards = m.get("clob_rewards") or []
            for r in clob_rewards:
                if isinstance(r, dict):
                    rate += float(r.get("rewardsDailyRate") or 0.0)

        has_rewards = (rate >= _MIN_RATE and min_size > 0 and max_spread > 0.0)
        return RewardConfig(
            daily_rate_usdc  = rate,
            max_spread_cents = max_spread,
            min_size_shares  = min_size,
            has_rewards      = has_rewards,
            holding_enabled  = bool(m.get("holding_enabled") or False),
        )
    except Exception as exc:
        logger.debug("_reward_config_from_market failed: %s", exc)
        return None


def _ineligible_plan(m: dict[str, Any], reason: str = "unknown") -> MakerQuotePlan:
    slug     = str(m.get("market_slug") or m.get("slug") or "")
    best_bid = float(m.get("best_bid") or 0.0)
    best_ask = float(m.get("best_ask") or 0.0)
    mid_p    = (best_bid + best_ask) / 2.0 if best_ask > best_bid else 0.0
    rate     = float(m.get("reward_daily_rate") or 0.0)
    max_sp   = float(m.get("rewards_max_spread") or 0.0)
    return MakerQuotePlan(
        market_slug        = slug,
        mid_p              = mid_p,
        quote_bid          = 0.0,
        quote_ask          = 0.0,
        quote_size         = 0.0,
        bid_spread_cents   = 0.0,
        ask_spread_cents   = 0.0,
        max_spread_cents   = max_sp,
        eligible           = False,
        feasibility_reason = reason,
        daily_rate_usdc    = rate,
        belief_var_used    = PRIOR_BELIEF_VAR,
        horizon_left       = 0.0,
        inventory          = 0.0,
    )
