"""
src/research/theory.py — Track B Phase 2
Core mathematical theory primitives for belief dynamics research.

Mathematical foundation
-----------------------
Belief in logit space:
    x = logit(p) = log(p / (1-p))         forward map  (probability → log-odds)
    p = sigmoid(x) = 1 / (1 + exp(-x))    inverse map  (log-odds → probability)

Why logit space?
    The logit map linearizes the bounded [0,1] belief space into ℝ.
    Belief dynamics that are awkward near 0 or 1 in probability space become
    well-behaved arithmetic in logit space.

Belief dynamics (continuous-time SDE):
    dx_t = μ dt + σ_b dW_t + J_t dN_t

    where:
        x_t   = logit(p_t)           latent true belief
        μ     ≈ 0                    drift (martingale assumption: efficient market)
        σ_b   = belief volatility    core state variable; measures how fast belief evolves
        dW_t  = Wiener increment     continuous diffusion
        J_t   = jump magnitude       signed jump when news arrives
        dN_t  = Poisson counter      jump arrival process

Discrete logit return:
    r_t = x_t - x_{t-1} = logit(p_t) - logit(p_{t-1})

    Under pure diffusion and zero drift:
        r_t ~ N(0, σ_b² · Δt)

p(1-p) — the sensitivity / uncertainty proxy:
    d(logit)/dp = 1 / (p(1-p))                 Jacobian of the logit map
    p(1-p) is maximal at p=0.5 (maximum ambiguity)
    p(1-p) → 0 as p → 0 or 1 (near-resolved market, low sensitivity)

    This is the core uncertainty weight. High p(1-p) → small prob changes → large
    logit moves → high sensitivity to new information.

Belief volatility estimator (EWMA):
    σ²_b[t] = λ · σ²_b[t-1] + (1-λ) · r²_t
    σ_b[t]  = sqrt(σ²_b[t])

    λ = 0.94 is a standard choice (RiskMetrics-style).
    Lower λ → faster adaptation to recent moves (better for jump-rich environments).

No imports from Track A. No live trading. Research only.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGIT_CLAMP_EPS: float = 1e-7    # floor/ceiling for p before logit to avoid ±inf
EWMA_LAMBDA_DEFAULT: float = 0.94 # RiskMetrics-style decay
EWMA_VAR_INIT: float = 0.01       # initial variance (σ_b ≈ 0.10 in logit space)


# ---------------------------------------------------------------------------
# 1. Logit / sigmoid transforms
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    """
    logit(p) = log(p / (1-p))   safe, clamped to avoid ±inf.

    p = 0.50  →  x = 0.0
    p = 0.73  →  x ≈ +1.0
    p = 0.27  →  x ≈ -1.0
    p = 0.99  →  x ≈ +4.6
    p = 0.01  →  x ≈ -4.6
    """
    p = max(LOGIT_CLAMP_EPS, min(1.0 - LOGIT_CLAMP_EPS, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """
    sigmoid(x) = 1 / (1 + exp(-x))   numerically stable.

    Two-branch formulation avoids overflow for large |x|.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)


def logit_return(p_prev: float, p_curr: float) -> float:
    """
    Discrete logit return: r = logit(p_curr) - logit(p_prev).

    Interpretation:
        r > 0  →  belief moved up (YES became more likely in logit space)
        r < 0  →  belief moved down
        |r| / σ_b > threshold  →  candidate jump
    """
    return logit(p_curr) - logit(p_prev)


# ---------------------------------------------------------------------------
# 2. Uncertainty proxy
# ---------------------------------------------------------------------------

def uncertainty(p: float) -> float:
    """
    p * (1 - p)  — Bernoulli variance.

    Properties:
        Maximum 0.25 at p = 0.50  (maximum ambiguity)
        Minimum 0.00 at p = 0 or 1  (resolved market)
        Equals 1 / (d(logit)/dp)  — reciprocal of logit sensitivity

    Use as:
        - spread-width driver (wider spread when uncertain)
        - weight in uncertainty index aggregation
        - normalizer for logit move significance
    """
    p = max(LOGIT_CLAMP_EPS, min(1.0 - LOGIT_CLAMP_EPS, float(p)))
    return p * (1.0 - p)


def normalized_uncertainty(p: float) -> float:
    """
    Normalized uncertainty in [0, 100]:  u(p) / 0.25 * 100

    100  at p = 0.50  (maximum uncertainty)
    0    at p = 0 or 1
    """
    return uncertainty(p) / 0.25 * 100.0


# ---------------------------------------------------------------------------
# 3. Belief state dataclass
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    """
    Full belief state for one market at one point in time.
    Populated by the state estimation pipeline.
    """
    slug:             str
    p_observed:       float        # raw outcomePrices[0] from market payload
    logit_observed:   float        # logit(p_observed)
    p_filtered:       float        # Kalman-filtered probability
    logit_filtered:   float        # logit(p_filtered)
    sigma_b:          float        # current belief volatility estimate (in logit units)
    P_var:            float        # Kalman state variance (uncertainty about filtered state)
    n_obs:            int          # number of observations processed
    last_logit_return: float = 0.0 # most recent logit return (for jump detection)
    last_updated:     str = ""     # ISO timestamp

    @property
    def uncertainty(self) -> float:
        return uncertainty(self.p_filtered)

    @property
    def normalized_uncertainty(self) -> float:
        return normalized_uncertainty(self.p_filtered)


# ---------------------------------------------------------------------------
# 4. EWMA belief volatility estimator
# ---------------------------------------------------------------------------

@dataclass
class EWMABeliefVol:
    """
    Exponentially weighted moving average estimator for σ_b.

    Updates on each new logit return observation:
        σ²_b[t] = λ · σ²_b[t-1] + (1-λ) · r²_t

    The EWMA is equivalent to an exponentially decaying window, giving more
    weight to recent observations without discarding older ones entirely.

    State is serializable (float fields only) for snapshot persistence.
    """
    lambda_: float = EWMA_LAMBDA_DEFAULT
    var:     float = EWMA_VAR_INIT
    n_obs:   int   = 0

    def update(self, r: float) -> float:
        """
        Incorporate one new logit return r.
        Returns updated σ_b estimate.
        """
        r2 = r * r
        if self.n_obs == 0:
            # Seed with first squared return; fall back to init if r=0
            self.var = r2 if r2 > 1e-10 else EWMA_VAR_INIT
        else:
            self.var = self.lambda_ * self.var + (1.0 - self.lambda_) * r2
        self.n_obs += 1
        return self.sigma_b

    @property
    def sigma_b(self) -> float:
        """Current σ_b estimate (square root of EWMA variance)."""
        return math.sqrt(max(self.var, 1e-10))

    def reset(self) -> None:
        self.var   = EWMA_VAR_INIT
        self.n_obs = 0

    def to_dict(self) -> dict:
        return {"lambda_": self.lambda_, "var": self.var, "n_obs": self.n_obs}

    @classmethod
    def from_dict(cls, d: dict) -> "EWMABeliefVol":
        return cls(lambda_=d["lambda_"], var=d["var"], n_obs=d["n_obs"])


# ---------------------------------------------------------------------------
# 5. Rolling sigma_b from a price series (batch utility)
# ---------------------------------------------------------------------------

def estimate_sigma_b_from_series(
    p_series: list[float],
    lambda_: float = EWMA_LAMBDA_DEFAULT,
) -> float:
    """
    Compute final σ_b estimate from a full series of probability observations.
    Convenience wrapper around EWMABeliefVol for batch use.

    Returns 0.0 if series has fewer than 2 observations.
    """
    if len(p_series) < 2:
        return 0.0
    ewma = EWMABeliefVol(lambda_=lambda_)
    for i in range(1, len(p_series)):
        r = logit_return(p_series[i - 1], p_series[i])
        ewma.update(r)
    return ewma.sigma_b


def logit_returns_from_series(p_series: list[float]) -> list[float]:
    """
    Convert a probability series to a list of logit returns.
    len(output) == len(input) - 1.
    """
    return [logit_return(p_series[i - 1], p_series[i]) for i in range(1, len(p_series))]
