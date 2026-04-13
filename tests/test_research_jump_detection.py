"""
tests/test_research_jump_detection.py
Deterministic tests for src/research/jump_detection.py

Coverage:
  - _normal_pdf / _laplace_pdf shape properties
  - jump_score: returns value in [0,1], increases with |r|
  - jump_score: large r → score near 1; small r → score near 0
  - is_jump hard classification
  - EMState serialisation roundtrip
  - EMSeparator: step increments n_obs, parameters stay in range
  - EMSeparator: reset restores defaults
  - EMSeparator serialisation roundtrip
  - detect_jumps_in_series batch utility
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.jump_detection import (
    _normal_pdf,
    _laplace_pdf,
    jump_score,
    is_jump,
    EMState,
    EMSeparator,
    JumpDetectionResult,
    detect_jumps_in_series,
    JUMP_PRIOR_PI,
    JUMP_LAPLACE_B,
    DIFF_SIGMA_INIT,
    JUMP_SCORE_THRESH,
)


# ---------------------------------------------------------------------------
# Density functions
# ---------------------------------------------------------------------------

class TestDensities:
    def test_normal_pdf_peak_at_zero(self):
        assert _normal_pdf(0.0, 0.1) > _normal_pdf(0.5, 0.1)

    def test_normal_pdf_symmetric(self):
        assert _normal_pdf(0.3, 0.1) == pytest.approx(_normal_pdf(-0.3, 0.1), rel=1e-9)

    def test_normal_pdf_zero_sigma(self):
        assert _normal_pdf(0.5, 0.0) == 0.0

    def test_laplace_pdf_peak_at_zero(self):
        assert _laplace_pdf(0.0, 0.3) > _laplace_pdf(0.5, 0.3)

    def test_laplace_pdf_symmetric(self):
        assert _laplace_pdf(0.3, 0.3) == pytest.approx(_laplace_pdf(-0.3, 0.3), rel=1e-9)

    def test_laplace_pdf_zero_b(self):
        assert _laplace_pdf(0.5, 0.0) == 0.0

    def test_laplace_heavier_tail_than_normal(self):
        """At large |r|, Laplace should have higher density than Normal of same scale."""
        r    = 3.0
        sig  = 0.5
        b    = sig / math.sqrt(2.0)   # same second moment
        assert _laplace_pdf(r, b) > _normal_pdf(r, sig)


# ---------------------------------------------------------------------------
# jump_score
# ---------------------------------------------------------------------------

class TestJumpScore:
    def test_output_in_unit_interval(self):
        for r in [-3.0, -1.0, 0.0, 0.5, 2.0]:
            s = jump_score(r)
            assert 0.0 <= s <= 1.0

    def test_near_zero_return_low_score(self):
        s = jump_score(0.0, sigma_d=0.10, b_j=0.30, pi=0.05)
        assert s < 0.10

    def test_large_return_high_score(self):
        # r = 5σ_d is almost certainly a jump
        s = jump_score(5.0, sigma_d=0.10, b_j=0.30, pi=0.05)
        assert s > 0.90

    def test_score_increases_with_abs_r(self):
        scores = [jump_score(r) for r in [0.0, 0.2, 0.5, 1.0, 2.0]]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_symmetric_in_r(self):
        for r in [0.5, 1.0, 2.0]:
            assert jump_score(r) == pytest.approx(jump_score(-r), rel=1e-9)

    def test_higher_pi_gives_higher_score(self):
        r = 0.3
        s_low_pi  = jump_score(r, pi=0.01)
        s_high_pi = jump_score(r, pi=0.20)
        assert s_high_pi > s_low_pi


class TestIsJump:
    def test_large_r_is_jump(self):
        assert is_jump(5.0, sigma_d=0.10, threshold=0.5)

    def test_small_r_not_jump(self):
        assert not is_jump(0.01, sigma_d=0.10, threshold=0.5)

    def test_threshold_respected(self):
        r = 0.5
        # With low threshold should be True
        assert is_jump(r, threshold=0.01)
        # With very high threshold should be False for moderate r
        # (jump_score at r=0.5 with defaults is moderate)
        score = jump_score(r)
        assert is_jump(r, threshold=score - 0.01) is True
        assert is_jump(r, threshold=score + 0.01) is False


# ---------------------------------------------------------------------------
# EMState
# ---------------------------------------------------------------------------

class TestEMState:
    def test_serialisation_roundtrip(self):
        state = EMState(pi=0.08, sigma_d=0.15, b_j=0.25, n_obs=42)
        d = state.to_dict()
        state2 = EMState.from_dict(d)
        assert state2.pi      == state.pi
        assert state2.sigma_d == state.sigma_d
        assert state2.b_j     == state.b_j
        assert state2.n_obs   == state.n_obs


# ---------------------------------------------------------------------------
# EMSeparator
# ---------------------------------------------------------------------------

class TestEMSeparator:
    def test_n_obs_increments(self):
        em = EMSeparator()
        for _ in range(5):
            em.step(0.1)
        assert em.state.n_obs == 5

    def test_parameters_stay_in_range(self):
        em = EMSeparator()
        for r in [-2.0, 0.0, 0.1, 1.5, -0.5, 3.0]:
            em.step(r)
        assert 1e-3 <= em.state.pi <= 0.5
        assert em.state.sigma_d > 0.0
        assert em.state.b_j > 0.0

    def test_many_large_returns_increase_pi(self):
        """Consistent large returns should push pi upward (more jumps detected)."""
        em_base = EMSeparator()
        em_jump = EMSeparator()
        for _ in range(30):
            em_base.step(0.01)   # tiny returns: diffusion-like
            em_jump.step(2.00)   # large returns: jump-like
        assert em_jump.state.pi > em_base.state.pi

    def test_result_fields(self):
        em = EMSeparator()
        result = em.step(0.5)
        assert isinstance(result, JumpDetectionResult)
        assert 0.0 <= result.jump_score <= 1.0
        assert isinstance(result.is_jump, bool)
        assert result.n_obs == 1

    def test_reset(self):
        em = EMSeparator()
        for r in [1.0, 2.0, -1.5]:
            em.step(r)
        em.reset()
        assert em.state.n_obs == 0
        assert em.state.pi == pytest.approx(JUMP_PRIOR_PI, rel=1e-9)
        assert em.state.sigma_d == pytest.approx(DIFF_SIGMA_INIT, rel=1e-9)

    def test_serialisation_roundtrip(self):
        em = EMSeparator(lambda_ewma=0.90)
        for r in [0.1, -0.5, 1.2, 0.3]:
            em.step(r)
        d  = em.to_dict()
        em2 = EMSeparator.from_dict(d)
        assert em2.state.pi      == pytest.approx(em.state.pi,      rel=1e-9)
        assert em2.state.sigma_d == pytest.approx(em.state.sigma_d, rel=1e-9)
        assert em2.state.b_j     == pytest.approx(em.state.b_j,     rel=1e-9)
        assert em2.state.n_obs   == em.state.n_obs


# ---------------------------------------------------------------------------
# detect_jumps_in_series
# ---------------------------------------------------------------------------

class TestDetectJumpsInSeries:
    def test_length_preserved(self):
        returns = [0.1, -0.2, 3.0, 0.05, -2.5]
        results = detect_jumps_in_series(returns)
        assert len(results) == len(returns)

    def test_empty_returns_empty(self):
        assert detect_jumps_in_series([]) == []

    def test_large_returns_classified_as_jumps(self):
        returns = [5.0, -4.5, 6.0]
        results = detect_jumps_in_series(returns, sigma_d=0.10, threshold=0.5)
        assert all(r.is_jump for r in results)

    def test_tiny_returns_not_classified_as_jumps(self):
        returns = [0.001, -0.001, 0.002]
        results = detect_jumps_in_series(returns, sigma_d=0.10, threshold=0.5)
        assert all(not r.is_jump for r in results)

    def test_n_obs_sequential(self):
        returns = [0.1, 0.2, 0.3]
        results = detect_jumps_in_series(returns)
        for i, result in enumerate(results):
            assert result.n_obs == i + 1
