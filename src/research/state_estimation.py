"""
src/research/state_estimation.py — Track B Phase 2
Scalar Kalman filter for belief denoising in logit space.

Model
-----
State:   x_t = logit(p_t)          (latent true belief)
Obs:     z_t = x_t + v_t           v_t ~ N(0, R)   (noisy market quote)

Martingale prior (zero drift, efficient market):
    x̂_{t|t-1} = x̂_{t-1}
    P_{t|t-1}  = P_{t-1} + Q       Q = process noise (σ_b² per step)

Kalman update:
    K_t        = P_{t|t-1} / (P_{t|t-1} + R)
    x̂_{t|t}   = x̂_{t|t-1} + K_t · (z_t - x̂_{t|t-1})
    P_{t|t}    = (1 - K_t) · P_{t|t-1}

Steady-state gain approximation:
    P_ss = (Q/2) · (1 + sqrt(1 + 4R/Q))   (exact positive root of Riccati)
    K_ss = P_ss / (P_ss + R)

Measurement noise R:
    Derived from bid-ask spread width as a proxy for quote uncertainty.
    R = (spread_logit / 2)²
    where spread_logit = logit(ask) - logit(1 - ask) for a symmetric book.
    Alternatively: pass a fixed constant (e.g. 0.01 = σ_v ≈ 0.10 in logit units).

No imports from Track A. Research only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .theory import logit, sigmoid, LOGIT_CLAMP_EPS


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

KALMAN_Q_DEFAULT: float = 1e-4   # process noise variance per observation
KALMAN_R_DEFAULT: float = 1e-2   # measurement noise variance (σ_v ≈ 0.10 logit)
KALMAN_P_INIT:    float = 1.0    # initial state variance (high uncertainty on cold start)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    """
    Serializable Kalman filter state for one market (logit space).

    Fields
    ------
    x_hat   : filtered logit belief  (the denoised estimate)
    P       : state variance         (uncertainty about x_hat)
    n_obs   : observations processed
    Q       : process noise variance (fixed per instance)
    R       : measurement noise variance (fixed per instance)
    """
    x_hat:   float = 0.0
    P:       float = KALMAN_P_INIT
    n_obs:   int   = 0
    Q:       float = KALMAN_Q_DEFAULT
    R:       float = KALMAN_R_DEFAULT

    @property
    def p_filtered(self) -> float:
        """Filtered probability (sigmoid of x_hat)."""
        return sigmoid(self.x_hat)

    def to_dict(self) -> dict:
        return {
            "x_hat": self.x_hat,
            "P":     self.P,
            "n_obs": self.n_obs,
            "Q":     self.Q,
            "R":     self.R,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KalmanState":
        return cls(
            x_hat=d["x_hat"],
            P=d["P"],
            n_obs=d["n_obs"],
            Q=d.get("Q", KALMAN_Q_DEFAULT),
            R=d.get("R", KALMAN_R_DEFAULT),
        )


@dataclass
class KalmanUpdateResult:
    """
    Full diagnostic from one Kalman step.

    innovation     = z_t - x̂_{t|t-1}    (how surprising the new quote was)
    kalman_gain    = K_t                  (weight on new observation vs prior)
    x_hat_prior    = x̂_{t|t-1}           (predicted state before update)
    x_hat_post     = x̂_{t|t}            (posterior state after update)
    P_prior        = P_{t|t-1}
    P_post         = P_{t|t}
    p_filtered     = sigmoid(x̂_{t|t})   (filtered probability)
    """
    innovation:   float
    kalman_gain:  float
    x_hat_prior:  float
    x_hat_post:   float
    P_prior:      float
    P_post:       float
    p_filtered:   float
    n_obs:        int


# ---------------------------------------------------------------------------
# Scalar Kalman filter
# ---------------------------------------------------------------------------

class ScalarBeliefKalman:
    """
    Scalar Kalman filter operating in logit space for a single market.

    Usage
    -----
    kf = ScalarBeliefKalman(Q=1e-4, R=1e-2)
    result = kf.step(p_observed=0.62)   # returns KalmanUpdateResult
    state  = kf.state                   # KalmanState snapshot

    Initialisation
    --------------
    On the first observation the filter is seeded with x_hat = logit(p_observed)
    to avoid a long convergence tail from x_hat=0.
    """

    def __init__(
        self,
        Q: float = KALMAN_Q_DEFAULT,
        R: float = KALMAN_R_DEFAULT,
        P_init: float = KALMAN_P_INIT,
    ) -> None:
        self._state = KalmanState(x_hat=0.0, P=P_init, n_obs=0, Q=Q, R=R)
        self._initialized = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> KalmanState:
        """Current filter state (read-only snapshot; returns the live object)."""
        return self._state

    def step(self, p_observed: float, R_override: Optional[float] = None) -> KalmanUpdateResult:
        """
        Incorporate one new market probability observation.

        Parameters
        ----------
        p_observed  : raw outcomePrices[0] or best ask from CLOB
        R_override  : optional per-step measurement noise override
                      (e.g. derived from spread width)

        Returns
        -------
        KalmanUpdateResult with full diagnostic fields.
        """
        z = logit(p_observed)
        R = R_override if R_override is not None else self._state.R
        Q = self._state.Q

        # Cold-start seed: avoid a large initial pull from x_hat=0
        if not self._initialized:
            self._state.x_hat = z
            self._initialized = True
            # Still run the full update so P converges from P_init

        # --- Predict ---
        x_prior = self._state.x_hat          # martingale: x̂_{t|t-1} = x̂_{t-1}
        P_prior = self._state.P + Q           # P_{t|t-1} = P_{t-1} + Q

        # --- Update ---
        innovation  = z - x_prior
        K           = P_prior / (P_prior + R)
        x_post      = x_prior + K * innovation
        P_post      = (1.0 - K) * P_prior

        # Write back
        self._state.x_hat = x_post
        self._state.P     = P_post
        self._state.n_obs += 1

        return KalmanUpdateResult(
            innovation=innovation,
            kalman_gain=K,
            x_hat_prior=x_prior,
            x_hat_post=x_post,
            P_prior=P_prior,
            P_post=P_post,
            p_filtered=sigmoid(x_post),
            n_obs=self._state.n_obs,
        )

    def reset(self, P_init: float = KALMAN_P_INIT) -> None:
        """Reset filter to cold-start state (preserve Q and R)."""
        self._state.x_hat = 0.0
        self._state.P     = P_init
        self._state.n_obs = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = self._state.to_dict()
        d["initialized"] = self._initialized
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScalarBeliefKalman":
        kf = cls(Q=d.get("Q", KALMAN_Q_DEFAULT), R=d.get("R", KALMAN_R_DEFAULT))
        kf._state = KalmanState.from_dict(d)
        kf._initialized = d.get("initialized", kf._state.n_obs > 0)
        return kf


# ---------------------------------------------------------------------------
# Measurement noise helper: R from spread
# ---------------------------------------------------------------------------

def R_from_spread(yes_ask: float, no_ask: float) -> float:
    """
    Derive measurement noise R from the observable bid-ask spread.

    Intuition: a wide spread means the quote is less precise → higher R.

    Formula:
        spread_logit = logit(yes_ask) - logit(1 - yes_ask)   [full logit width]
        R = (spread_logit / 4)²

    The /4 divisor treats the half-spread as ±2σ, so σ_v = spread_logit/4.
    Clipped to [1e-4, 1.0] to avoid degenerate filter behaviour.
    """
    # Use mid-point of yes_ask and no_ask to define "the market price"
    mid = (yes_ask + (1.0 - no_ask)) / 2.0
    mid = max(LOGIT_CLAMP_EPS, min(1.0 - LOGIT_CLAMP_EPS, mid))
    half_spread_prob = abs(yes_ask - (1.0 - no_ask)) / 2.0
    half_spread_prob = max(1e-4, half_spread_prob)
    # Logit sensitivity at mid: d(logit)/dp = 1 / (p*(1-p))
    sensitivity = 1.0 / (mid * (1.0 - mid))
    half_spread_logit = half_spread_prob * sensitivity
    R = half_spread_logit ** 2
    return max(1e-4, min(1.0, R))


# ---------------------------------------------------------------------------
# Steady-state Kalman gain utility
# ---------------------------------------------------------------------------

def steady_state_gain(Q: float, R: float) -> tuple[float, float]:
    """
    Exact steady-state Kalman gain and variance for the scalar zero-drift model.

    Riccati equation (scalar):
        P_ss² + Q·P_ss - Q·R = 0
        P_ss = (Q/2) · (-1 + sqrt(1 + 4R/Q))    [positive root]

    Returns
    -------
    (K_ss, P_ss)
    """
    discriminant = math.sqrt(1.0 + 4.0 * R / Q)
    P_ss = (Q / 2.0) * (-1.0 + discriminant)
    K_ss = P_ss / (P_ss + R)
    return K_ss, P_ss


# ---------------------------------------------------------------------------
# Batch utility
# ---------------------------------------------------------------------------

def filter_series(
    p_series: list[float],
    Q: float = KALMAN_Q_DEFAULT,
    R: float = KALMAN_R_DEFAULT,
) -> list[float]:
    """
    Apply Kalman filter to a full probability series.
    Returns list of filtered probabilities (same length as input).
    Useful for offline research / backtesting.
    """
    if not p_series:
        return []
    kf = ScalarBeliefKalman(Q=Q, R=R)
    return [kf.step(p).p_filtered for p in p_series]
