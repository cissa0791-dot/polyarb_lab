"""
tests/test_research_state_estimation.py
Deterministic tests for src/research/state_estimation.py

Coverage:
  - KalmanState / KalmanUpdateResult construction
  - ScalarBeliefKalman cold-start seeding
  - Single-step update: prior/posterior, Kalman gain, innovation
  - P converges downward from P_init
  - Filtered estimate moves toward observation
  - Reset restores initial state
  - R_from_spread function
  - steady_state_gain analytical formula
  - filter_series batch utility
  - Serialisation roundtrip
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.state_estimation import (
    KalmanState,
    KalmanUpdateResult,
    ScalarBeliefKalman,
    R_from_spread,
    steady_state_gain,
    filter_series,
    KALMAN_Q_DEFAULT,
    KALMAN_R_DEFAULT,
    KALMAN_P_INIT,
)
from src.research.theory import logit, sigmoid


# ---------------------------------------------------------------------------
# KalmanState
# ---------------------------------------------------------------------------

class TestKalmanState:
    def test_p_filtered_property(self):
        state = KalmanState(x_hat=0.0)
        assert state.p_filtered == pytest.approx(0.5, abs=1e-10)

    def test_serialisation_roundtrip(self):
        state = KalmanState(x_hat=1.2, P=0.05, n_obs=10, Q=1e-3, R=2e-2)
        d = state.to_dict()
        state2 = KalmanState.from_dict(d)
        assert state2.x_hat == state.x_hat
        assert state2.P     == state.P
        assert state2.n_obs == state.n_obs
        assert state2.Q     == state.Q
        assert state2.R     == state.R


# ---------------------------------------------------------------------------
# ScalarBeliefKalman — basic mechanics
# ---------------------------------------------------------------------------

class TestScalarBeliefKalman:
    def test_cold_start_seeds_x_hat(self):
        kf = ScalarBeliefKalman()
        result = kf.step(0.7)
        # After cold-start + update, x_hat should be close to logit(0.7)
        assert kf.state.x_hat == pytest.approx(logit(0.7), rel=0.5)

    def test_n_obs_increments(self):
        kf = ScalarBeliefKalman()
        kf.step(0.5)
        kf.step(0.6)
        assert kf.state.n_obs == 2

    def test_kalman_gain_in_range(self):
        kf = ScalarBeliefKalman(Q=1e-4, R=1e-2)
        result = kf.step(0.5)
        assert 0.0 < result.kalman_gain < 1.0

    def test_P_decreases_over_time(self):
        """After many steps with fixed params, P should stabilise below P_init."""
        kf = ScalarBeliefKalman(Q=1e-4, R=1e-2, P_init=1.0)
        for _ in range(50):
            kf.step(0.5)
        assert kf.state.P < KALMAN_P_INIT

    def test_P_lower_bound(self):
        """P should never go to zero (Q keeps injecting noise)."""
        kf = ScalarBeliefKalman(Q=1e-6, R=1e-2)
        for _ in range(200):
            kf.step(0.5)
        assert kf.state.P > 0.0

    def test_filtered_moves_toward_observation(self):
        """
        If observation is consistently 0.8 and filter starts at 0.5,
        p_filtered should converge toward 0.8.
        """
        kf = ScalarBeliefKalman(Q=1e-3, R=1e-2)
        for _ in range(30):
            kf.step(0.8)
        assert kf.state.p_filtered > 0.75

    def test_innovation_is_correct(self):
        """innovation = z_t - x_hat_prior (after cold start)."""
        kf = ScalarBeliefKalman()
        kf.step(0.5)            # cold start: x_hat = logit(0.5) = 0
        result = kf.step(0.7)   # second step
        expected_innovation = logit(0.7) - result.x_hat_prior
        assert result.innovation == pytest.approx(expected_innovation, rel=1e-9)

    def test_posterior_is_weighted_sum(self):
        """x_hat_post = x_hat_prior + K * innovation"""
        kf = ScalarBeliefKalman()
        kf.step(0.5)
        result = kf.step(0.6)
        expected = result.x_hat_prior + result.kalman_gain * result.innovation
        assert result.x_hat_post == pytest.approx(expected, rel=1e-9)

    def test_R_override(self):
        """R_override should change Kalman gain vs default R."""
        kf1 = ScalarBeliefKalman(Q=1e-4, R=1e-2)
        kf2 = ScalarBeliefKalman(Q=1e-4, R=1e-2)
        kf1.step(0.5)  # seed both
        kf2.step(0.5)
        r1 = kf1.step(0.6, R_override=0.001)  # tight R → high K
        r2 = kf2.step(0.6, R_override=1.000)  # loose R → low K
        assert r1.kalman_gain > r2.kalman_gain

    def test_reset(self):
        kf = ScalarBeliefKalman()
        for _ in range(10):
            kf.step(0.7)
        kf.reset()
        assert kf.state.n_obs == 0
        assert kf.state.P == pytest.approx(KALMAN_P_INIT, rel=1e-9)
        # Next step should cold-start again
        result = kf.step(0.5)
        assert result.n_obs == 1

    def test_serialisation_roundtrip(self):
        kf = ScalarBeliefKalman(Q=5e-4, R=2e-2)
        for p in [0.4, 0.5, 0.6, 0.65]:
            kf.step(p)
        d = kf.to_dict()
        kf2 = ScalarBeliefKalman.from_dict(d)
        assert kf2.state.x_hat == pytest.approx(kf.state.x_hat, rel=1e-12)
        assert kf2.state.P     == pytest.approx(kf.state.P,     rel=1e-12)
        assert kf2.state.n_obs == kf.state.n_obs


# ---------------------------------------------------------------------------
# R_from_spread
# ---------------------------------------------------------------------------

class TestRFromSpread:
    def test_wide_spread_gives_larger_R(self):
        R_tight = R_from_spread(0.51, 0.50)
        R_wide  = R_from_spread(0.60, 0.60)
        # Wide spread (0.60+0.60 = 1.20, large gap) → larger R
        assert R_wide > R_tight

    def test_output_is_positive(self):
        assert R_from_spread(0.55, 0.52) > 0.0

    def test_output_is_finite(self):
        assert math.isfinite(R_from_spread(0.99, 0.01))
        assert math.isfinite(R_from_spread(0.50, 0.50))

    def test_clamp_floor(self):
        # Should not produce zero
        assert R_from_spread(0.50, 0.50) >= 1e-4


# ---------------------------------------------------------------------------
# steady_state_gain
# ---------------------------------------------------------------------------

class TestSteadyStateGain:
    def test_gain_in_unit_interval(self):
        K_ss, P_ss = steady_state_gain(Q=1e-4, R=1e-2)
        assert 0.0 < K_ss < 1.0

    def test_P_ss_positive(self):
        _, P_ss = steady_state_gain(Q=1e-4, R=1e-2)
        assert P_ss > 0.0

    def test_riccati_equation(self):
        """Verify P_ss² + Q·P_ss - Q·R ≈ 0"""
        Q, R = 1e-4, 1e-2
        K_ss, P_ss = steady_state_gain(Q, R)
        residual = P_ss ** 2 + Q * P_ss - Q * R
        assert abs(residual) < 1e-12

    def test_larger_Q_gives_larger_K(self):
        K1, _ = steady_state_gain(Q=1e-4, R=1e-2)
        K2, _ = steady_state_gain(Q=1e-2, R=1e-2)
        assert K2 > K1   # more process noise → trust observations more

    def test_larger_R_gives_smaller_K(self):
        K1, _ = steady_state_gain(Q=1e-4, R=1e-2)
        K2, _ = steady_state_gain(Q=1e-4, R=1.0)
        assert K2 < K1   # more measurement noise → trust prior more


# ---------------------------------------------------------------------------
# filter_series batch utility
# ---------------------------------------------------------------------------

class TestFilterSeries:
    def test_empty_returns_empty(self):
        assert filter_series([]) == []

    def test_length_preserved(self):
        p_series = [0.4, 0.5, 0.6, 0.55, 0.65]
        result = filter_series(p_series)
        assert len(result) == len(p_series)

    def test_outputs_in_unit_interval(self):
        p_series = [0.1, 0.9, 0.5, 0.3, 0.7]
        for p_f in filter_series(p_series):
            assert 0.0 < p_f < 1.0

    def test_smoothing_reduces_variance(self):
        """Filtered series should have lower variance than input (noise reduction)."""
        import statistics
        p_series = [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]
        filtered = filter_series(p_series, Q=1e-4, R=1e-2)
        assert statistics.stdev(filtered) < statistics.stdev(p_series)
