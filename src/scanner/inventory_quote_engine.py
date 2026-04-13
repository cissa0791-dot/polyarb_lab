"""
inventory_quote_engine.py — Logit-space reservation quote and spread engine.

Implements the heuristic x-space quoting formulas from the prediction-market
market-maker literature:

  reservation_x = mid_x - inventory * risk_aversion * belief_var * horizon_left

  total_spread_x ≈ risk_aversion * belief_var * horizon_left
                   + (2 / k) * log(1 + risk_aversion / k)

  half_spread_x  = total_spread_x / 2
  bid_x          = reservation_x - half_spread_x
  ask_x          = reservation_x + half_spread_x
  bid_p          = sigmoid(bid_x)
  ask_p          = sigmoid(ask_x)

Simplifications vs paper:
  - k (market depth / order-book impact) is treated as a config constant
    (paper derives it from order flow; we don't have that data)
  - horizon_left is in [0, 1] (fraction of market lifetime remaining)
  - No PIDE, no derivative pricing surface
  - No cross-market hedging
  - No partial fill modelling

Usage:
    engine = InventoryQuoteEngine(risk_aversion=1.0, k=0.5)
    quote  = engine.compute_quote(
        mid_p=0.50, belief_var=0.04, horizon_left=0.5, inventory=0.0
    )
    # quote.bid_p, quote.ask_p, quote.spread_p_approx

Hard constraints:
  - No network calls. No state mutation. Pure computation.
  - Errors must not propagate to caller.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from src.scanner.logit_utils import (
    delta_x_from_p,
    logit_to_prob,
    prob_to_logit,
    spread_x_to_spread_p,
)

logger = logging.getLogger("polyarb.scanner.inventory_quote_engine")

# Default engine parameters — conservative, overridable at construction.
DEFAULT_RISK_AVERSION = 1.0   # γ — higher = wider spread, stronger inventory skew
DEFAULT_K             = 0.5   # market depth constant (order-flow impact proxy)
DEFAULT_HORIZON       = 0.5   # fallback if not provided (50% of market life left)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuoteSuggestion:
    """Computed quote in both x-space and p-space."""
    mid_p             : float   # input midpoint probability
    mid_x             : float   # logit(mid_p)
    reservation_x     : float   # inventory-skewed mid in x-space
    reservation_p     : float   # sigmoid(reservation_x)
    half_spread_x     : float   # half the total x-space spread
    bid_x             : float
    ask_x             : float
    bid_p             : float   # sigmoid(bid_x)
    ask_p             : float   # sigmoid(ask_x)
    spread_p_approx   : float   # first-order p-space spread approximation
    belief_var        : float   # input belief variance used
    horizon_left      : float   # input horizon fraction used
    inventory         : float   # input inventory (signed)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InventoryQuoteEngine:
    """
    Computes inventory-aware reservation quotes and spreads in logit space.

    Parameters
    ----------
    risk_aversion : float
        γ — inventory risk aversion. Higher → wider spread, stronger skew.
    k : float
        Market depth constant (order-book impact proxy). Higher → narrower
        asymptotic spread (less adverse-selection cost modelled).
    """

    def __init__(
        self,
        risk_aversion: float = DEFAULT_RISK_AVERSION,
        k            : float = DEFAULT_K,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.k             = k

    def compute_quote(
        self,
        mid_p       : float,
        belief_var  : float,
        horizon_left: float = DEFAULT_HORIZON,
        inventory   : float = 0.0,
    ) -> Optional[QuoteSuggestion]:
        """
        Compute bid/ask quote suggestion.

        Parameters
        ----------
        mid_p        : current midpoint probability (0 < mid_p < 1)
        belief_var   : x-space variance of belief (from BeliefVarEstimator)
        horizon_left : fraction of market lifetime remaining ∈ [0, 1]
        inventory    : signed net position (positive = long YES, negative = short)

        Returns None on any computation error.
        """
        try:
            return self._compute(mid_p, belief_var, horizon_left, inventory)
        except Exception as exc:
            logger.debug("InventoryQuoteEngine.compute_quote failed: %s", exc)
            return None

    def _compute(
        self,
        mid_p       : float,
        belief_var  : float,
        horizon_left: float,
        inventory   : float,
    ) -> QuoteSuggestion:
        γ = self.risk_aversion
        k = max(self.k, 1e-6)
        σ2 = max(belief_var, 1e-8)
        T  = max(min(horizon_left, 1.0), 0.0)

        mid_x = prob_to_logit(mid_p)

        # Reservation quote: skew toward inventory-reduction side
        reservation_x = mid_x - inventory * γ * σ2 * T

        # Total x-space spread (two additive terms)
        adverse_selection_term = γ * σ2 * T
        inventory_penalty_term = (2.0 / k) * math.log(1.0 + γ / k)
        total_spread_x = adverse_selection_term + inventory_penalty_term
        half_spread_x  = total_spread_x / 2.0

        bid_x = reservation_x - half_spread_x
        ask_x = reservation_x + half_spread_x
        bid_p = logit_to_prob(bid_x)
        ask_p = logit_to_prob(ask_x)

        # First-order p-space spread approximation (at mid)
        spread_p_approx = spread_x_to_spread_p(half_spread_x, mid_p) * 2.0

        reservation_p = logit_to_prob(reservation_x)

        return QuoteSuggestion(
            mid_p           = mid_p,
            mid_x           = round(mid_x, 6),
            reservation_x   = round(reservation_x, 6),
            reservation_p   = round(reservation_p, 6),
            half_spread_x   = round(half_spread_x, 6),
            bid_x           = round(bid_x, 6),
            ask_x           = round(ask_x, 6),
            bid_p           = round(bid_p, 6),
            ask_p           = round(ask_p, 6),
            spread_p_approx = round(spread_p_approx, 6),
            belief_var      = belief_var,
            horizon_left    = horizon_left,
            inventory       = inventory,
        )
