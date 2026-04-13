"""
reward_quote_feasibility.py — Reward eligibility check for suggested quotes.

Evaluates whether an InventoryQuoteEngine suggestion satisfies Polymarket's
official reward constraints (from the Gamma market payload):

  rewards_max_spread  — max total spread from midpoint in cents
  rewards_min_size    — minimum order size in shares

Authoritative field source: Gamma API market payload (live, confirmed 2026-03-21).
The official Polymarket reward rule: a maker order qualifies for rewards iff:
  - Order is posted within rewardsMaxSpread cents of the midpoint
  - Order size is at least rewardsMinSize shares

This module translates an x-space QuoteSuggestion into p-space ask/bid prices
and checks them against the market's reward thresholds.

Hard constraints:
  - No network calls. No state. Pure computation.
  - Errors must not propagate to caller (returns ineligible result on failure).
  - Official reward rules are authoritative; quote engine formulas are advisory.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.scanner.inventory_quote_engine import QuoteSuggestion
from src.research.reward_eval import RewardConfig

logger = logging.getLogger("polyarb.scanner.reward_quote_feasibility")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardFeasibility:
    """Reward eligibility result for a suggested quote."""
    eligible           : bool
    bid_eligible       : bool    # bid_p within max_spread from mid
    ask_eligible       : bool    # ask_p within max_spread from mid
    size_eligible      : bool    # proposed_size >= rewards_min_size
    bid_spread_cents   : float   # |mid_p - bid_p| * 100
    ask_spread_cents   : float   # |ask_p - mid_p| * 100
    max_spread_cents   : float   # from RewardConfig
    proposed_size      : int     # proposed order size
    min_size_shares    : int     # from RewardConfig
    daily_rate_usdc    : float
    reason             : str     # human-readable eligibility summary


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_quote_feasibility(
    quote         : Optional[QuoteSuggestion],
    reward_config : Optional[RewardConfig],
    proposed_size : int = 0,
) -> RewardFeasibility:
    """
    Check whether a QuoteSuggestion is reward-eligible under the market's
    official reward config.

    proposed_size: the order size we intend to post (shares).
    Returns ineligible result when quote or config is None.
    """
    try:
        return _evaluate(quote, reward_config, proposed_size)
    except Exception as exc:
        logger.debug("evaluate_quote_feasibility failed: %s", exc)
        return _ineligible("evaluation_error", proposed_size, reward_config)


def _evaluate(
    quote         : Optional[QuoteSuggestion],
    reward_config : Optional[RewardConfig],
    proposed_size : int,
) -> RewardFeasibility:

    if quote is None:
        return _ineligible("no_quote", proposed_size, reward_config)
    if reward_config is None or not reward_config.has_rewards:
        return _ineligible("no_reward_program", proposed_size, reward_config)

    max_spread = reward_config.max_spread_cents  # cents
    mid_p      = quote.mid_p

    # Spread of suggested bid/ask from midpoint (in cents)
    bid_spread_cents = abs(mid_p - quote.bid_p) * 100.0
    ask_spread_cents = abs(quote.ask_p - mid_p) * 100.0

    bid_eligible  = bid_spread_cents <= max_spread
    ask_eligible  = ask_spread_cents <= max_spread
    size_eligible = proposed_size >= reward_config.min_size_shares

    eligible = bid_eligible and ask_eligible and size_eligible

    parts = []
    if not bid_eligible:
        parts.append(f"bid_spread={bid_spread_cents:.2f}>{max_spread}¢")
    if not ask_eligible:
        parts.append(f"ask_spread={ask_spread_cents:.2f}>{max_spread}¢")
    if not size_eligible:
        parts.append(f"size={proposed_size}<{reward_config.min_size_shares}")
    reason = "eligible" if eligible else ("ineligible: " + ", ".join(parts))

    return RewardFeasibility(
        eligible         = eligible,
        bid_eligible     = bid_eligible,
        ask_eligible     = ask_eligible,
        size_eligible    = size_eligible,
        bid_spread_cents = round(bid_spread_cents, 3),
        ask_spread_cents = round(ask_spread_cents, 3),
        max_spread_cents = max_spread,
        proposed_size    = proposed_size,
        min_size_shares  = reward_config.min_size_shares,
        daily_rate_usdc  = reward_config.daily_rate_usdc,
        reason           = reason,
    )


def _ineligible(
    reason: str,
    proposed_size: int,
    reward_config: Optional[RewardConfig],
) -> RewardFeasibility:
    return RewardFeasibility(
        eligible         = False,
        bid_eligible     = False,
        ask_eligible     = False,
        size_eligible    = False,
        bid_spread_cents = 0.0,
        ask_spread_cents = 0.0,
        max_spread_cents = reward_config.max_spread_cents if reward_config else 0.0,
        proposed_size    = proposed_size,
        min_size_shares  = reward_config.min_size_shares if reward_config else 0,
        daily_rate_usdc  = reward_config.daily_rate_usdc if reward_config else 0.0,
        reason           = reason,
    )
