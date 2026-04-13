"""
reward_eval.py — Reward-aware opportunity evaluation.

Parses Polymarket liquidity reward config from Gamma market payloads and
evaluates reward fitness for a candidate after CLOB book fetch.

All reward fields come from the Gamma API market payload (already fetched
by scan_events / scan_slice). No additional API calls.

Confirmed Gamma reward fields (live, 2026-03-21):
  rewardsMinSize   (int)   — min order size in shares for reward eligibility
  rewardsMaxSpread (float) — max spread from midpoint in cents (e.g. 1.5 = 1.5¢)
  clobRewards      (list)  — [{rewardsDailyRate: float, ...}]
                             rewardsDailyRate in USDC/day (sponsored or native)
  holdingRewardsEnabled (bool) — separate holding rewards, not used here

Reward eligibility rules (from official Polymarket docs):
  - Order must be placed within rewardsMaxSpread cents of the midpoint
  - Order must be at least rewardsMinSize shares
  - Distribution proportional to time-weighted liquidity provided

Key computed metrics:
  market_spread_cents  = (yes_ask + no_ask - 1.0) * 100
  spread_fits          = market_spread_cents <= rewardsMaxSpread
                         (current book already within reward threshold)
  size_fits            = market_min_size <= rewardsMinSize
                         (order minimum is achievable for our budget)
  reward_per_capital   = daily_rate / estimated_capital_required
  reward_score         = daily_rate * spread_fitness_factor * size_fitness_factor

Hard constraints:
  - No Track A mutation, no live trading, no network calls.
  - Errors must not propagate to caller (wrap at boundary).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("polyarb.research.reward_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spreads at or above this threshold are considered too saturated for maker quoting.
BOTH_98_SPREAD_CENTS = 96.0

# Reward rate below this (USDC/day) is not worth routing budget toward.
MIN_MEANINGFUL_RATE = 5.0

# rewardsMaxSpread threshold: markets with spread tolerance below this are
# structurally over-competed (tight spread = many existing makers = both_98 risk).
COMPETITIVE_SPREAD_THRESHOLD_CENTS = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardConfig:
    """Parsed reward configuration from a Gamma market payload."""
    daily_rate_usdc    : float   # total daily reward in USDC/day (clobRewards[0].rewardsDailyRate)
    max_spread_cents   : float   # rewardsMaxSpread (e.g. 1.5 means 1.5¢ from mid)
    min_size_shares    : int     # rewardsMinSize
    has_rewards        : bool    # True if any active reward program exists
    holding_enabled    : bool    # holdingRewardsEnabled (informational only)


@dataclass(frozen=True)
class RewardFitness:
    """Computed reward fitness for one candidate after CLOB book fetch."""
    has_rewards          : bool
    spread_fits          : bool   # current market spread <= rewards_max_spread
    size_fits            : bool   # market_min_size <= rewards_min_size (affordable)
    market_spread_cents  : float  # (yes_ask + no_ask - 1.0) * 100
    max_spread_cents     : float  # from RewardConfig
    min_size_shares      : int    # from RewardConfig
    daily_rate_usdc      : float
    reward_per_capital   : float  # daily_rate / estimated_two_sided_capital (%)
    reward_score         : float  # composite reward-weighted score


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_reward_config(m: dict) -> Optional[RewardConfig]:
    """
    Extract reward configuration from a Gamma API market payload.
    Returns None if the market has no reward program.
    Errors are caught; returns None on parse failure.
    """
    try:
        min_size    = int(m.get("rewardsMinSize") or 0)
        max_spread  = float(m.get("rewardsMaxSpread") or 0.0)
        holding     = bool(m.get("holdingRewardsEnabled") or False)

        clob_rewards = m.get("clobRewards") or []
        daily_rate = 0.0
        if clob_rewards and isinstance(clob_rewards, list):
            try:
                daily_rate = float(clob_rewards[0].get("rewardsDailyRate") or 0.0)
            except Exception:
                daily_rate = 0.0

        has_rewards = (daily_rate >= MIN_MEANINGFUL_RATE and min_size > 0 and max_spread > 0)

        return RewardConfig(
            daily_rate_usdc  = daily_rate,
            max_spread_cents = max_spread,
            min_size_shares  = min_size,
            has_rewards      = has_rewards,
            holding_enabled  = holding,
        )
    except Exception as exc:
        logger.debug("parse_reward_config failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_fitness(
    yes_ask       : float,
    no_ask        : float,
    market_min_size: float,     # orderMinSize from Gamma (min trade unit)
    config        : Optional[RewardConfig],
) -> RewardFitness:
    """
    Compute reward fitness for a candidate that cleared the CLOB book fetch.

    yes_ask + no_ask are the current best asks.
    market_min_size is the market's minimum order size (orderMinSize).

    reward_per_capital is an estimate assuming:
      - two-sided quote: (min_size_shares) shares on each side
      - avg price per share ≈ 0.50 USDC (binary market near 50/50)
      - estimate conservative: capital = min_size_shares * 1.0 USDC
        (one full share-pair rather than half-price for each side)
    This gives: reward_per_capital = daily_rate / capital
    """
    market_spread_cents = (yes_ask + no_ask - 1.0) * 100.0

    if config is None:
        return RewardFitness(
            has_rewards=False, spread_fits=False, size_fits=False,
            market_spread_cents=market_spread_cents,
            max_spread_cents=0.0, min_size_shares=0,
            daily_rate_usdc=0.0, reward_per_capital=0.0, reward_score=0.0,
        )

    spread_fits = (
        config.has_rewards
        and config.max_spread_cents > 0
        and market_spread_cents <= config.max_spread_cents
    )

    # size_fits: the market's minimum trade unit is at most our reward min_size
    # (i.e., we can place at least one reward-eligible order at the market minimum)
    size_fits = (
        config.has_rewards
        and config.min_size_shares > 0
        and market_min_size <= config.min_size_shares
    )

    # Capital estimate: min_size_shares shares at ~$0.50 each, two sides
    capital_usdc = max(config.min_size_shares * 1.0, 1.0)
    reward_per_capital = (config.daily_rate_usdc / capital_usdc * 100.0
                          if capital_usdc > 0 else 0.0)

    # Composite reward score
    # Factors: rate (log), spread achievability, size achievability
    # Penalize markets with spread < COMPETITIVE_SPREAD_THRESHOLD (over-competed)
    import math
    rate_factor      = math.log1p(config.daily_rate_usdc)
    spread_factor    = 1.0 if spread_fits else 0.3
    competitive_pen  = 0.5 if config.max_spread_cents < COMPETITIVE_SPREAD_THRESHOLD_CENTS else 1.0
    size_factor      = 1.0 if size_fits else 0.5

    reward_score = rate_factor * spread_factor * competitive_pen * size_factor

    return RewardFitness(
        has_rewards         = config.has_rewards,
        spread_fits         = spread_fits,
        size_fits           = size_fits,
        market_spread_cents = round(market_spread_cents, 3),
        max_spread_cents    = config.max_spread_cents,
        min_size_shares     = config.min_size_shares,
        daily_rate_usdc     = config.daily_rate_usdc,
        reward_per_capital  = round(reward_per_capital, 4),
        reward_score        = round(reward_score, 4),
    )


# ---------------------------------------------------------------------------
# Telemetry formatter
# ---------------------------------------------------------------------------

def format_reward_line(slug: str, fitness: RewardFitness) -> Optional[str]:
    """
    One-line reward telemetry string for print in process_candidates().
    Returns None if market has no reward program (suppress clutter).
    """
    if not fitness.has_rewards:
        return None

    spread_tag = "SPREAD_OK" if fitness.spread_fits else f"SPREAD_WIDE({fitness.market_spread_cents:.1f}>{fitness.max_spread_cents}¢)"
    size_tag   = "SIZE_OK"   if fitness.size_fits   else f"SIZE_MISS(need≥{fitness.min_size_shares})"
    eligible   = fitness.spread_fits and fitness.size_fits

    return (
        f"  [reward] {'ELIGIBLE' if eligible else 'INELIGIBLE':9s}  "
        f"{slug[:40]:<40}  "
        f"rate=${fitness.daily_rate_usdc:.0f}/day  "
        f"{spread_tag}  {size_tag}  "
        f"est_rpc={fitness.reward_per_capital:.2f}%/day  "
        f"score={fitness.reward_score:.2f}"
    )


# ---------------------------------------------------------------------------
# Cohort router bonus helper
# ---------------------------------------------------------------------------

def reward_routing_bonus(m: dict) -> float:
    """
    Return a routing score bonus for the cohort router based on reward config
    in the Gamma market payload.  Returns 0.0 if no reward program.

    Bonus is proportional to daily rate and inversely penalised for
    over-competitive spread thresholds (< COMPETITIVE_SPREAD_THRESHOLD_CENTS).
    """
    config = parse_reward_config(m)
    if config is None or not config.has_rewards:
        return 0.0

    import math
    base = math.log1p(config.daily_rate_usdc) * 2.0  # 0..~18 for rates 0..7500

    # Penalise very tight spread thresholds (over-competed, both_98-prone)
    if config.max_spread_cents < COMPETITIVE_SPREAD_THRESHOLD_CENTS:
        base *= 0.4

    return round(base, 3)
