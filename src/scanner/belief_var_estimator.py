"""
belief_var_estimator.py — Lightweight x-space belief variance estimator.

Estimates the short-horizon x-space variance of the market mid-probability
from a small window of recent midpoint observations.

This is the belief_var (σ²_belief) term used in the inventory_quote_engine
spread formula:
    total_spread_x ≈ risk_aversion * belief_var * horizon_left + ...

Design constraints:
  - No network calls, no DB writes.
  - Keeps a rolling window of x-values (logit of observed midpoint).
  - Falls back to a sensible prior if fewer than MIN_OBS observations.
  - Errors never propagate to caller.

Usage:
    est = BeliefVarEstimator(window=20)
    est.update(p_yes=0.52)
    est.update(p_yes=0.49)
    var = est.variance()          # x-space variance of recent midpoints
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

from src.scanner.logit_utils import prob_to_logit

logger = logging.getLogger("polyarb.scanner.belief_var_estimator")

# Minimum observations before using the estimated variance (below this, use prior).
MIN_OBS = 3

# Conservative prior for x-space variance when insufficient data.
# Corresponds to roughly ±5¢ uncertainty around 50¢ midpoint.
PRIOR_VARIANCE = 0.04


class BeliefVarEstimator:
    """
    Rolling x-space variance estimator for a single market's midpoint.

    Feed midpoint probability observations via update(); query variance().
    Window is a FIFO of the most recent `window` observations.
    """

    def __init__(self, window: int = 20, prior_var: float = PRIOR_VARIANCE) -> None:
        self._window    : int         = window
        self._prior_var : float       = prior_var
        self._xs        : deque[float] = deque(maxlen=window)

    def update(self, p_yes: float) -> None:
        """Add one midpoint observation (p_yes ∈ (0,1))."""
        try:
            x = prob_to_logit(p_yes)
            self._xs.append(x)
        except Exception as exc:
            logger.debug("BeliefVarEstimator.update failed: %s", exc)

    def variance(self) -> float:
        """
        Return the rolling x-space sample variance.
        Falls back to PRIOR_VARIANCE when fewer than MIN_OBS observations.
        """
        n = len(self._xs)
        if n < MIN_OBS:
            return self._prior_var
        try:
            xs   = list(self._xs)
            mean = sum(xs) / n
            var  = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
            return max(var, 1e-8)   # floor to avoid zero-division downstream
        except Exception:
            return self._prior_var

    def n_obs(self) -> int:
        return len(self._xs)

    def reset(self) -> None:
        self._xs.clear()


def estimate_var_from_series(p_series: list[float], window: int = 20) -> float:
    """
    Convenience: compute x-space variance from a list of p_yes observations.
    Returns PRIOR_VARIANCE if series is too short.
    """
    est = BeliefVarEstimator(window=window)
    for p in p_series:
        est.update(p)
    return est.variance()
