"""
src/research/pipeline.py — Track B Phase 3
Per-slug belief pipeline: integrates Kalman denoising, EWMA σ_b, and EM jump detection.

This is the integration hub for the theory stack. It connects:
    theory.py           — logit / sigmoid / EWMABeliefVol
    state_estimation.py — ScalarBeliefKalman
    jump_detection.py   — EMSeparator

Per-slug state machine
----------------------
For each new observation p_t:
    1. Kalman step     → filtered belief x̂_t, P_t
    2. Logit return    → r_t = x̂_t - x̂_{t-1}  (in filtered logit space)
    3. EWMA update     → σ_b estimate
    4. EM step         → jump_score, is_jump

Using filtered-space logit returns rather than raw-observation returns keeps
the σ_b and jump estimates clean: measurement noise in outcomePrices is
absorbed by the Kalman filter before the EWMA and EM see it.

BeliefPipelineRegistry
-----------------------
Manages one SlugPipeline per market slug.
Serializable to/from dict for snapshot persistence across runs.

No imports from Track A. Research only.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .theory import (
    logit,
    sigmoid,
    BeliefState,
    EWMABeliefVol,
    uncertainty,
    EWMA_LAMBDA_DEFAULT,
)
from .state_estimation import (
    ScalarBeliefKalman,
    KALMAN_Q_DEFAULT,
    KALMAN_R_DEFAULT,
)
from .jump_detection import EMSeparator


# ---------------------------------------------------------------------------
# Per-step result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineStepResult:
    """
    Full output of one pipeline step for a single market observation.

    Combines Kalman filter, EWMA σ_b, and EM jump classification into one record.

    Fields
    ------
    slug            : market slug
    p_observed      : raw input probability (from outcomePrices or CLOB ask)
    p_filtered      : Kalman-filtered probability (denoised)
    logit_observed  : logit(p_observed)
    logit_filtered  : Kalman posterior in logit space (x̂_{t|t})
    logit_ret       : r_t = logit_filtered[t] - logit_filtered[t-1]  (0 on cold start)
    sigma_b         : current EWMA σ_b estimate (logit units)
    P_var           : Kalman posterior variance (uncertainty about filtered state)
    jump_score_val  : P(jump | r_t) from EM separator
    is_jump         : hard classification at EM threshold
    kalman_gain     : K_t from Kalman step
    n_obs           : total observations processed for this slug
    timestamp       : ISO timestamp of this step
    """
    slug:           str
    p_observed:     float
    p_filtered:     float
    logit_observed: float
    logit_filtered: float
    logit_ret:      float      # logit return in filtered space
    sigma_b:        float
    P_var:          float
    jump_score_val: float
    is_jump:        bool
    kalman_gain:    float
    n_obs:          int
    timestamp:      str

    @property
    def uncertainty(self) -> float:
        """p_filtered * (1 - p_filtered)."""
        return uncertainty(self.p_filtered)

    def to_belief_state(self) -> BeliefState:
        """Convert to BeliefState dataclass."""
        return BeliefState(
            slug=self.slug,
            p_observed=self.p_observed,
            logit_observed=self.logit_observed,
            p_filtered=self.p_filtered,
            logit_filtered=self.logit_filtered,
            sigma_b=self.sigma_b,
            P_var=self.P_var,
            n_obs=self.n_obs,
            last_logit_return=self.logit_ret,
            last_updated=self.timestamp,
        )

    def to_dict(self) -> dict:
        return {
            "slug":           self.slug,
            "p_observed":     round(self.p_observed, 6),
            "p_filtered":     round(self.p_filtered, 6),
            "logit_observed": round(self.logit_observed, 6),
            "logit_filtered": round(self.logit_filtered, 6),
            "logit_ret":      round(self.logit_ret, 6),
            "sigma_b":        round(self.sigma_b, 6),
            "P_var":          round(self.P_var, 6),
            "jump_score":     round(self.jump_score_val, 4),
            "is_jump":        self.is_jump,
            "kalman_gain":    round(self.kalman_gain, 6),
            "n_obs":          self.n_obs,
            "timestamp":      self.timestamp,
        }


# ---------------------------------------------------------------------------
# Per-slug pipeline
# ---------------------------------------------------------------------------

class SlugPipeline:
    """
    Per-slug belief pipeline owning:
        ScalarBeliefKalman  — denoises probability observations
        EWMABeliefVol       — estimates σ_b from logit returns
        EMSeparator         — separates diffusion from jump moves

    Logit returns are computed in filtered space (x̂_t - x̂_{t-1}) to avoid
    measurement noise contamination before EWMA and EM see the signal.
    """

    def __init__(
        self,
        Q:           float = KALMAN_Q_DEFAULT,
        R:           float = KALMAN_R_DEFAULT,
        ewma_lambda: float = EWMA_LAMBDA_DEFAULT,
    ) -> None:
        self._kalman = ScalarBeliefKalman(Q=Q, R=R)
        self._ewma   = EWMABeliefVol(lambda_=ewma_lambda)
        self._em     = EMSeparator()
        self._prev_logit_filtered: Optional[float] = None

    # ------------------------------------------------------------------

    def step(
        self,
        p_observed: float,
        R_override: Optional[float] = None,
        timestamp:  Optional[str]   = None,
    ) -> dict:
        """
        Process one probability observation.
        Returns raw dict of outputs (slug filled by BeliefPipelineRegistry).
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Kalman filter
        k_result = self._kalman.step(p_observed, R_override=R_override)

        # Step 2: Logit return in filtered space (not observed space)
        logit_obs = logit(p_observed)
        if self._prev_logit_filtered is None:
            lr = 0.0   # cold-start: no prior filtered value
        else:
            lr = k_result.x_hat_post - self._prev_logit_filtered
        self._prev_logit_filtered = k_result.x_hat_post

        # Step 3: EWMA σ_b
        sigma_b = self._ewma.update(lr)

        # Step 4: EM jump classification
        em_result = self._em.step(lr)

        return dict(
            p_observed=p_observed,
            p_filtered=k_result.p_filtered,
            logit_observed=logit_obs,
            logit_filtered=k_result.x_hat_post,
            logit_ret=lr,
            sigma_b=sigma_b,
            P_var=k_result.P_post,
            jump_score_val=em_result.jump_score,
            is_jump=em_result.is_jump,
            kalman_gain=k_result.kalman_gain,
            n_obs=k_result.n_obs,
            timestamp=timestamp,
        )

    def reset(self) -> None:
        self._kalman.reset()
        self._ewma.reset()
        self._em.reset()
        self._prev_logit_filtered = None

    def to_dict(self) -> dict:
        return {
            "kalman":              self._kalman.to_dict(),
            "ewma":                self._ewma.to_dict(),
            "em":                  self._em.to_dict(),
            "prev_logit_filtered": self._prev_logit_filtered,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlugPipeline":
        obj = cls()
        obj._kalman              = ScalarBeliefKalman.from_dict(d["kalman"])
        obj._ewma                = EWMABeliefVol.from_dict(d["ewma"])
        obj._em                  = EMSeparator.from_dict(d["em"])
        obj._prev_logit_filtered = d.get("prev_logit_filtered")
        return obj


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class BeliefPipelineRegistry:
    """
    Registry of per-slug SlugPipelines.

    Usage
    -----
    registry = BeliefPipelineRegistry()
    result = registry.step("trump-wins-2026", p_observed=0.62)
    result.slug         # "trump-wins-2026"
    result.p_filtered   # Kalman-denoised probability
    result.sigma_b      # EWMA belief volatility (logit units)
    result.is_jump      # True if this move classified as a jump

    Persistence
    -----------
    d    = registry.to_dict()
    reg2 = BeliefPipelineRegistry.from_dict(d)
    """

    def __init__(
        self,
        Q:           float = KALMAN_Q_DEFAULT,
        R:           float = KALMAN_R_DEFAULT,
        ewma_lambda: float = EWMA_LAMBDA_DEFAULT,
    ) -> None:
        self._default_Q      = Q
        self._default_R      = R
        self._default_lambda = ewma_lambda
        self._pipelines: Dict[str, SlugPipeline] = {}

    # ------------------------------------------------------------------

    def step(
        self,
        slug:       str,
        p_observed: float,
        R_override: Optional[float] = None,
        timestamp:  Optional[str]   = None,
    ) -> PipelineStepResult:
        """
        Process one observation for a market slug.
        Creates a new SlugPipeline on first call for this slug.
        """
        if slug not in self._pipelines:
            self._pipelines[slug] = SlugPipeline(
                Q=self._default_Q,
                R=self._default_R,
                ewma_lambda=self._default_lambda,
            )
        raw = self._pipelines[slug].step(
            p_observed, R_override=R_override, timestamp=timestamp
        )
        return PipelineStepResult(slug=slug, **raw)

    def slugs(self) -> List[str]:
        return list(self._pipelines.keys())

    def n_slugs(self) -> int:
        return len(self._pipelines)

    def reset_slug(self, slug: str) -> None:
        if slug in self._pipelines:
            self._pipelines[slug].reset()

    def reset_all(self) -> None:
        self._pipelines.clear()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "default_Q":      self._default_Q,
            "default_R":      self._default_R,
            "default_lambda": self._default_lambda,
            "pipelines": {
                slug: pipe.to_dict()
                for slug, pipe in self._pipelines.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefPipelineRegistry":
        obj = cls(
            Q=d.get("default_Q",      KALMAN_Q_DEFAULT),
            R=d.get("default_R",      KALMAN_R_DEFAULT),
            ewma_lambda=d.get("default_lambda", EWMA_LAMBDA_DEFAULT),
        )
        for slug, pipe_dict in d.get("pipelines", {}).items():
            obj._pipelines[slug] = SlugPipeline.from_dict(pipe_dict)
        return obj
