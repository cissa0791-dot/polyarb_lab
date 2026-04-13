"""
tests/test_research_theory.py
Deterministic tests for src/research/theory.py

Coverage:
  - logit / sigmoid correctness and inverse relationship
  - logit clamp behavior at extremes
  - logit_return sign and magnitude
  - uncertainty (p*(1-p)) shape
  - normalized_uncertainty range
  - BeliefState construction and properties
  - EWMABeliefVol update mechanics
  - estimate_sigma_b_from_series batch utility
  - logit_returns_from_series length invariant
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.theory import (
    logit,
    sigmoid,
    logit_return,
    uncertainty,
    normalized_uncertainty,
    BeliefState,
    EWMABeliefVol,
    estimate_sigma_b_from_series,
    logit_returns_from_series,
    LOGIT_CLAMP_EPS,
    EWMA_LAMBDA_DEFAULT,
    EWMA_VAR_INIT,
)


# ---------------------------------------------------------------------------
# logit / sigmoid
# ---------------------------------------------------------------------------

class TestLogit:
    def test_midpoint(self):
        assert logit(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_known_values(self):
        # logit(0.73) ≈ +1.0  (actually logit(σ(1)) = 1)
        assert logit(sigmoid(1.0)) == pytest.approx(1.0, rel=1e-5)
        assert logit(sigmoid(-1.0)) == pytest.approx(-1.0, rel=1e-5)

    def test_antisymmetry(self):
        # logit(p) = -logit(1-p)
        for p in [0.1, 0.3, 0.7, 0.9]:
            assert logit(p) == pytest.approx(-logit(1.0 - p), rel=1e-9)

    def test_clamping_at_zero(self):
        # logit(0) would be -inf; clamped to finite value
        result = logit(0.0)
        assert math.isfinite(result)
        assert result < 0

    def test_clamping_at_one(self):
        result = logit(1.0)
        assert math.isfinite(result)
        assert result > 0

    def test_monotone(self):
        ps = [0.1, 0.3, 0.5, 0.7, 0.9]
        vals = [logit(p) for p in ps]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]


class TestSigmoid:
    def test_midpoint(self):
        assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_known_values(self):
        assert sigmoid(1.0) == pytest.approx(0.7310585786, rel=1e-6)
        assert sigmoid(-1.0) == pytest.approx(0.2689414214, rel=1e-6)

    def test_large_positive(self):
        # Should not overflow
        result = sigmoid(100.0)
        assert math.isfinite(result)
        assert result > 0.99

    def test_large_negative(self):
        result = sigmoid(-100.0)
        assert math.isfinite(result)
        assert result < 0.01

    def test_inverse_of_logit(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert sigmoid(logit(p)) == pytest.approx(p, rel=1e-6)

    def test_symmetry(self):
        # sigmoid(-x) = 1 - sigmoid(x)
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert sigmoid(-x) == pytest.approx(1.0 - sigmoid(x), rel=1e-10)


class TestLogitReturn:
    def test_zero_when_unchanged(self):
        assert logit_return(0.5, 0.5) == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_moving_up(self):
        assert logit_return(0.4, 0.6) > 0

    def test_negative_when_moving_down(self):
        assert logit_return(0.6, 0.4) < 0

    def test_antisymmetry(self):
        r_up   = logit_return(0.4, 0.6)
        r_down = logit_return(0.6, 0.4)
        assert r_up == pytest.approx(-r_down, rel=1e-9)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_maximum_at_half(self):
        assert uncertainty(0.5) == pytest.approx(0.25, abs=1e-10)

    def test_zero_at_extremes(self):
        # Clamping means it won't be exactly 0, but should be tiny
        assert uncertainty(0.0) < 1e-6
        assert uncertainty(1.0) < 1e-6

    def test_monotone_up_to_half(self):
        ps = [0.1, 0.2, 0.3, 0.4, 0.5]
        vals = [uncertainty(p) for p in ps]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_symmetric(self):
        for p in [0.1, 0.3, 0.4]:
            assert uncertainty(p) == pytest.approx(uncertainty(1.0 - p), rel=1e-9)

    def test_normalized_range(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            nu = normalized_uncertainty(p)
            assert 0.0 <= nu <= 100.0

    def test_normalized_max_at_half(self):
        assert normalized_uncertainty(0.5) == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

class TestBeliefState:
    def _make(self, p_filtered=0.6):
        return BeliefState(
            slug="test-market",
            p_observed=0.62,
            logit_observed=logit(0.62),
            p_filtered=p_filtered,
            logit_filtered=logit(p_filtered),
            sigma_b=0.10,
            P_var=0.01,
            n_obs=5,
        )

    def test_uncertainty_property(self):
        bs = self._make(0.5)
        assert bs.uncertainty == pytest.approx(0.25, abs=1e-10)

    def test_normalized_uncertainty_property(self):
        bs = self._make(0.5)
        assert bs.normalized_uncertainty == pytest.approx(100.0, rel=1e-6)

    def test_non_default_p(self):
        bs = self._make(0.8)
        assert bs.uncertainty == pytest.approx(0.8 * 0.2, rel=1e-9)


# ---------------------------------------------------------------------------
# EWMABeliefVol
# ---------------------------------------------------------------------------

class TestEWMABeliefVol:
    def test_initial_sigma_b(self):
        ewma = EWMABeliefVol()
        # sigma_b before any update = sqrt(EWMA_VAR_INIT)
        assert ewma.sigma_b == pytest.approx(math.sqrt(EWMA_VAR_INIT), rel=1e-9)

    def test_first_update_seeds_variance(self):
        ewma = EWMABeliefVol()
        r = 0.5
        ewma.update(r)
        assert ewma.var == pytest.approx(r * r, rel=1e-9)
        assert ewma.n_obs == 1

    def test_first_update_zero_return_fallback(self):
        ewma = EWMABeliefVol()
        ewma.update(0.0)  # r=0, should fall back to EWMA_VAR_INIT
        assert ewma.var == pytest.approx(EWMA_VAR_INIT, rel=1e-9)

    def test_ewma_formula(self):
        ewma = EWMABeliefVol(lambda_=0.9)
        ewma.update(0.2)   # seeds var = 0.04
        ewma.update(0.1)   # var = 0.9 * 0.04 + 0.1 * 0.01 = 0.036 + 0.001 = 0.037
        expected_var = 0.9 * 0.04 + 0.1 * 0.01
        assert ewma.var == pytest.approx(expected_var, rel=1e-9)

    def test_sigma_b_positive(self):
        ewma = EWMABeliefVol()
        for r in [0.1, -0.2, 0.05, -0.3]:
            sigma = ewma.update(r)
            assert sigma > 0

    def test_n_obs_increments(self):
        ewma = EWMABeliefVol()
        for i in range(5):
            ewma.update(0.1)
        assert ewma.n_obs == 5

    def test_reset(self):
        ewma = EWMABeliefVol()
        ewma.update(0.5)
        ewma.update(0.3)
        ewma.reset()
        assert ewma.n_obs == 0
        assert ewma.var == pytest.approx(EWMA_VAR_INIT, rel=1e-9)

    def test_serialisation_roundtrip(self):
        ewma = EWMABeliefVol(lambda_=0.90)
        ewma.update(0.2)
        ewma.update(0.15)
        d = ewma.to_dict()
        ewma2 = EWMABeliefVol.from_dict(d)
        assert ewma2.var    == pytest.approx(ewma.var,    rel=1e-12)
        assert ewma2.n_obs  == ewma.n_obs
        assert ewma2.lambda_ == ewma.lambda_

    def test_lambda_persistence_effect(self):
        # Higher lambda → slower adaptation to recent shock
        ewma_slow = EWMABeliefVol(lambda_=0.99)
        ewma_fast = EWMABeliefVol(lambda_=0.50)
        big_r = 2.0
        for ewma in [ewma_slow, ewma_fast]:
            ewma.update(0.01)   # seed with small return
        ewma_slow.update(big_r)
        ewma_fast.update(big_r)
        # Fast adapts more to big shock → higher sigma_b
        assert ewma_fast.sigma_b > ewma_slow.sigma_b


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

class TestBatchUtilities:
    def test_estimate_short_series_returns_zero(self):
        assert estimate_sigma_b_from_series([0.5]) == 0.0
        assert estimate_sigma_b_from_series([]) == 0.0

    def test_estimate_constant_series(self):
        # Constant probability → all returns = 0.
        # First update seeds var = EWMA_VAR_INIT; subsequent updates decay it via λ^n.
        # So sigma_b is positive but strictly less than sqrt(EWMA_VAR_INIT).
        series = [0.5] * 10
        sigma = estimate_sigma_b_from_series(series)
        assert sigma > 0.0
        assert sigma < math.sqrt(EWMA_VAR_INIT) + 1e-6

    def test_estimate_volatile_series(self):
        # Alternating between 0.3 and 0.7 → large returns → high sigma_b
        series = [0.3, 0.7] * 10
        sigma = estimate_sigma_b_from_series(series)
        assert sigma > 0.5   # logit(0.7) - logit(0.3) ≈ 1.69

    def test_logit_returns_length(self):
        p_series = [0.4, 0.5, 0.6, 0.55]
        returns = logit_returns_from_series(p_series)
        assert len(returns) == len(p_series) - 1

    def test_logit_returns_signs(self):
        p_series = [0.4, 0.6, 0.5]
        returns = logit_returns_from_series(p_series)
        assert returns[0] > 0   # 0.4 → 0.6: belief moved up
        assert returns[1] < 0   # 0.6 → 0.5: belief moved down
