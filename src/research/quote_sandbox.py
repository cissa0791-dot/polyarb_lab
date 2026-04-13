"""
src/research/quote_sandbox.py — Track B Phase 2
Avellaneda-Stoikov inspired quote-width / inventory-skew sandbox.

Background
----------
Avellaneda & Stoikov (2008) derive optimal quotes for a market maker
with CARA utility over inventory risk. Adapted here for a binary
prediction-market context.

Key equations
-------------
Reservation price (inventory-skewed mid):
    r_price = mid - q · γ · σ_b² · T

Optimal half-spread:
    δ = (γ · σ_b² · T) / 2  +  (1/γ) · ln(1 + γ/k)

Full spread:
    s = 2 · δ

Bid/ask quotes:
    ask = r_price + δ
    bid = r_price - δ

where:
    mid    = current mid-price (probability space)
    q      = current inventory (net YES position; positive = long YES)
    γ      = risk aversion coefficient (> 0)
    σ_b    = belief volatility in logit units
    T      = time to expiry in days (or normalised horizon)
    k      = order arrival intensity parameter

Prediction-market adaptations
------------------------------
1. Prices are bounded [0, 1] — all outputs are clamped.
2. σ_b is in logit space; we convert spread back to probability space
   via the local Jacobian: Δp ≈ Δx · p(1-p) where Δx is the logit spread.
3. Inventory q is in shares (can be fractional).
4. k is treated as a free calibration parameter (no live order flow data).

This is a research sandbox — no live orders, no risk limits.
Results are purely informational for theoretical exploration.

No imports from Track A. Research only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

AS_GAMMA_DEFAULT: float = 0.1    # risk aversion (moderate)
AS_K_DEFAULT:     float = 1.5    # order arrival intensity
AS_T_DEFAULT:     float = 1.0    # time horizon (normalised units; 1 day)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ASQuoteResult:
    """
    Output of one Avellaneda-Stoikov quote computation.

    All prices are in probability space [0, 1].

    reservation_price : inventory-adjusted mid (γ=0 → equal to raw mid)
    half_spread       : δ in probability space
    bid               : reservation_price - half_spread
    ask               : reservation_price + half_spread
    spread_prob       : 2 · half_spread (total spread in probability space)
    spread_logit      : spread in logit space (pre-conversion)
    inventory_skew    : q · γ · σ_b² · T (signed; positive = long YES, bid skewed up)
    """
    mid:               float
    reservation_price: float
    half_spread:       float
    bid:               float
    ask:               float
    spread_prob:       float
    spread_logit:      float
    inventory_skew:    float
    gamma:             float
    sigma_b:           float
    T:                 float
    q:                 float
    k:                 float


# ---------------------------------------------------------------------------
# Core A-S computation
# ---------------------------------------------------------------------------

def as_quotes(
    mid:     float,
    sigma_b: float,
    q:       float = 0.0,
    gamma:   float = AS_GAMMA_DEFAULT,
    T:       float = AS_T_DEFAULT,
    k:       float = AS_K_DEFAULT,
) -> ASQuoteResult:
    """
    Compute A-S inspired reservation price and optimal quotes.

    Parameters
    ----------
    mid     : current mid-price (probability space, e.g. 0.55)
    sigma_b : belief volatility (logit space)
    q       : current inventory (net YES shares; positive = long YES)
    gamma   : risk aversion coefficient (> 0)
    T       : time horizon (days or normalised units)
    k       : order arrival intensity

    Returns
    -------
    ASQuoteResult (all prices clamped to [ε, 1-ε])
    """
    eps = 1e-4

    # --- Reservation price in logit space ---
    # inventory_skew_logit = q · γ · σ_b² · T
    var_b = sigma_b * sigma_b
    inventory_skew_logit = q * gamma * var_b * T

    # Convert mid to logit space
    mid_clamped = max(eps, min(1.0 - eps, mid))
    logit_mid   = math.log(mid_clamped / (1.0 - mid_clamped))

    logit_r_price = logit_mid - inventory_skew_logit

    # --- Optimal half-spread in logit space ---
    # δ_logit = (γ · σ_b² · T) / 2  +  (1/γ) · ln(1 + γ/k)
    if gamma < 1e-10:
        half_spread_logit = 0.0
    else:
        half_spread_logit = (
            (gamma * var_b * T) / 2.0
            + (1.0 / gamma) * math.log(1.0 + gamma / max(k, 1e-6))
        )

    # --- Convert back to probability space via Jacobian ---
    # r_price in probability space
    r_price_prob = 1.0 / (1.0 + math.exp(-logit_r_price))

    # Jacobian: dp/dx = sigmoid(x) * (1 - sigmoid(x)) = p*(1-p) at r_price
    jacobian = r_price_prob * (1.0 - r_price_prob)
    half_spread_prob = half_spread_logit * jacobian

    # Quotes
    bid = max(eps, r_price_prob - half_spread_prob)
    ask = min(1.0 - eps, r_price_prob + half_spread_prob)

    # Re-derive actual spread after clamping
    spread_prob = ask - bid

    return ASQuoteResult(
        mid=round(mid, 6),
        reservation_price=round(r_price_prob, 6),
        half_spread=round(half_spread_prob, 6),
        bid=round(bid, 6),
        ask=round(ask, 6),
        spread_prob=round(spread_prob, 6),
        spread_logit=round(2.0 * half_spread_logit, 6),
        inventory_skew=round(inventory_skew_logit, 6),
        gamma=gamma,
        sigma_b=sigma_b,
        T=T,
        q=q,
        k=k,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis helpers
# ---------------------------------------------------------------------------

def spread_vs_sigma_b(
    sigma_b_range: list[float],
    mid:           float = 0.5,
    q:             float = 0.0,
    gamma:         float = AS_GAMMA_DEFAULT,
    T:             float = AS_T_DEFAULT,
    k:             float = AS_K_DEFAULT,
) -> list[dict]:
    """
    Compute spread for each sigma_b value in sigma_b_range.
    Returns list of dicts: {sigma_b, spread_prob, half_spread, bid, ask}.
    Useful for plotting spread curves.
    """
    results = []
    for s in sigma_b_range:
        q_result = as_quotes(mid=mid, sigma_b=s, q=q, gamma=gamma, T=T, k=k)
        results.append({
            "sigma_b":    round(s, 6),
            "spread_prob": q_result.spread_prob,
            "half_spread": q_result.half_spread,
            "bid":         q_result.bid,
            "ask":         q_result.ask,
        })
    return results


def spread_vs_inventory(
    q_range:  list[float],
    mid:      float = 0.5,
    sigma_b:  float = 0.10,
    gamma:    float = AS_GAMMA_DEFAULT,
    T:        float = AS_T_DEFAULT,
    k:        float = AS_K_DEFAULT,
) -> list[dict]:
    """
    Compute reservation price and quotes for each inventory level q.
    Returns list of dicts: {q, reservation_price, bid, ask, inventory_skew}.
    """
    results = []
    for q in q_range:
        q_result = as_quotes(mid=mid, sigma_b=sigma_b, q=q, gamma=gamma, T=T, k=k)
        results.append({
            "q":                q,
            "reservation_price": q_result.reservation_price,
            "bid":               q_result.bid,
            "ask":               q_result.ask,
            "inventory_skew":    q_result.inventory_skew,
        })
    return results


# ---------------------------------------------------------------------------
# Inventory-aware position sizing hint (paper-only)
# ---------------------------------------------------------------------------

def position_size_hint(
    edge:      float,
    sigma_b:   float,
    gamma:     float = AS_GAMMA_DEFAULT,
    max_shares: float = 100.0,
) -> float:
    """
    Kelly-fraction inspired position sizing in the A-S framework.

    Approximate optimal size:
        n* = edge / (γ · σ_b²)

    Clamped to [0, max_shares].
    Returns 0.0 when edge <= 0 (no long position if no edge).

    This is a theoretical approximation — not a live sizing rule.
    """
    if edge <= 0.0 or sigma_b < 1e-10 or gamma < 1e-10:
        return 0.0
    raw = edge / (gamma * sigma_b * sigma_b)
    return min(max(0.0, raw), max_shares)
