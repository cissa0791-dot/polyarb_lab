"""
logit_utils.py — Logit/probability transforms and local Greeks.

Formulas from logit-space market-maker literature:
  x = logit(p) = ln(p / (1 - p))
  p = sigmoid(x) = 1 / (1 + exp(-x))
  delta_x(p) = dp/dx = p * (1 - p)         (chain-rule Jacobian)
  gamma_x(p) = d²p/dx² = p * (1 - p) * (1 - 2*p)

These are used to approximate how midpoint price moves in x-space and to
convert x-space spreads to approximate p-space spreads.

No network calls. No state. Pure functions only.
"""
from __future__ import annotations

import math

# Clamp probability away from 0/1 to avoid log(0) / division-by-zero.
_P_MIN = 1e-6
_P_MAX = 1.0 - 1e-6


def prob_to_logit(p: float) -> float:
    """logit(p) = ln(p / (1-p)).  Clamps p to (_P_MIN, _P_MAX)."""
    p = max(_P_MIN, min(_P_MAX, p))
    return math.log(p / (1.0 - p))


def logit_to_prob(x: float) -> float:
    """sigmoid(x) = 1 / (1 + exp(-x)).  Numerically stable for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def delta_x_from_p(p: float) -> float:
    """dp/dx = p * (1 - p).  Maximum at p=0.5 (=0.25), zero at extremes."""
    p = max(_P_MIN, min(_P_MAX, p))
    return p * (1.0 - p)


def gamma_x_from_p(p: float) -> float:
    """d²p/dx² = p * (1-p) * (1-2p).  Zero at p=0.5; sign flips at 0.5."""
    p = max(_P_MIN, min(_P_MAX, p))
    return p * (1.0 - p) * (1.0 - 2.0 * p)


def spread_x_to_spread_p(half_spread_x: float, p: float) -> float:
    """
    Approximate p-space half-spread from x-space half-spread.
    spread_p ≈ delta_x(p) * half_spread_x  (first-order Taylor).
    """
    return delta_x_from_p(p) * half_spread_x
