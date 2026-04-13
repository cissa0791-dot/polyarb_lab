"""
tests/test_research_pipeline.py
Deterministic tests for src/research/pipeline.py

Coverage:
  - SlugPipeline cold start
  - SlugPipeline Kalman convergence over repeated observations
  - SlugPipeline sigma_b increases on volatile series
  - SlugPipeline serialisation roundtrip
  - SlugPipeline jump detection on large return
  - BeliefPipelineRegistry slug isolation
  - BeliefPipelineRegistry registry-level serialisation roundtrip
  - PipelineStepResult.to_belief_state() correctness
  - PipelineStepResult.uncertainty property
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.pipeline import (
    SlugPipeline,
    BeliefPipelineRegistry,
    PipelineStepResult,
)
from src.research.theory import sigmoid, logit, uncertainty


# ---------------------------------------------------------------------------
# SlugPipeline tests
# ---------------------------------------------------------------------------

class TestSlugPipelineColdStart:
    def test_first_step_returns_valid_result(self):
        pipe = SlugPipeline()
        raw = pipe.step(0.6)
        assert "p_observed" in raw
        assert raw["p_observed"] == pytest.approx(0.6, abs=1e-10)
        assert math.isfinite(raw["p_filtered"])
        assert math.isfinite(raw["sigma_b"])
        assert raw["n_obs"] == 1

    def test_cold_start_logit_ret_is_zero(self):
        """First observation has no prior filtered state → logit return = 0."""
        pipe = SlugPipeline()
        raw = pipe.step(0.7)
        assert raw["logit_ret"] == pytest.approx(0.0, abs=1e-10)

    def test_p_filtered_close_to_p_observed_on_cold_start(self):
        """Kalman seeds from observation on cold start → p_filtered ≈ p_observed."""
        pipe = SlugPipeline()
        raw = pipe.step(0.65)
        # Should be close (Kalman smoothes but seeds from observation)
        assert abs(raw["p_filtered"] - 0.65) < 0.1

    def test_sigma_b_positive(self):
        pipe = SlugPipeline()
        raw = pipe.step(0.5)
        assert raw["sigma_b"] > 0


class TestSlugPipelineKalmanConvergence:
    def test_p_filtered_converges_on_constant_series(self):
        """Repeated same observation → p_filtered should converge to that value."""
        pipe = SlugPipeline()
        p = 0.72
        raw = None
        for _ in range(30):
            raw = pipe.step(p)
        assert abs(raw["p_filtered"] - p) < 0.01

    def test_P_var_decreases_over_time(self):
        """Kalman variance should shrink as more observations arrive."""
        pipe = SlugPipeline()
        raw_first = pipe.step(0.5)
        P_initial = raw_first["P_var"]
        for _ in range(20):
            pipe.step(0.5)
        raw_last = pipe.step(0.5)
        # Steady-state P should be much smaller than initial P=1.0
        assert raw_last["P_var"] < P_initial

    def test_logit_ret_near_zero_on_stable_series(self):
        """After convergence, logit returns should be near zero on stable p."""
        pipe = SlugPipeline()
        for _ in range(20):
            pipe.step(0.5)
        raw = pipe.step(0.5)
        assert abs(raw["logit_ret"]) < 0.01


class TestSlugPipelineSigmaB:
    def test_sigma_b_higher_on_volatile_series(self):
        """Alternating prices → higher sigma_b than constant series."""
        pipe_stable   = SlugPipeline()
        pipe_volatile = SlugPipeline()
        for i in range(20):
            pipe_stable.step(0.5)
            p = 0.3 if i % 2 == 0 else 0.7
            pipe_volatile.step(p)
        raw_stable   = pipe_stable.step(0.5)
        raw_volatile = pipe_volatile.step(0.4)
        assert raw_volatile["sigma_b"] > raw_stable["sigma_b"]


class TestSlugPipelineJumpDetection:
    def test_large_return_yields_nonzero_jump_score(self):
        """A large logit return should push jump_score above the prior."""
        pipe = SlugPipeline()
        # Warm up filter
        for _ in range(10):
            pipe.step(0.5)
        # Big move: 0.5 → 0.95 in one step → large logit return
        raw = pipe.step(0.95)
        # jump_score should be > prior (0.05) for a large move
        assert raw["jump_score_val"] > 0.05

    def test_small_return_gives_low_jump_score(self):
        """A tiny return should yield a low jump score."""
        pipe = SlugPipeline()
        for _ in range(10):
            pipe.step(0.5)
        raw = pipe.step(0.501)
        # Very small move → mostly diffusion
        assert raw["jump_score_val"] < 0.5


class TestSlugPipelineSerialisation:
    def test_roundtrip_preserves_state(self):
        pipe = SlugPipeline()
        for p in [0.4, 0.5, 0.6, 0.55, 0.5]:
            pipe.step(p)
        d = pipe.to_dict()
        pipe2 = SlugPipeline.from_dict(d)
        # Both should produce identical output on next step
        raw1 = pipe.step(0.52)
        raw2 = pipe2.step(0.52)
        assert raw1["p_filtered"]  == pytest.approx(raw2["p_filtered"],  rel=1e-8)
        assert raw1["sigma_b"]     == pytest.approx(raw2["sigma_b"],     rel=1e-8)
        assert raw1["logit_ret"]   == pytest.approx(raw2["logit_ret"],   rel=1e-8)

    def test_reset_clears_state(self):
        pipe = SlugPipeline()
        for p in [0.3, 0.7, 0.3, 0.7]:
            pipe.step(p)
        raw_before = pipe.step(0.5)
        pipe.reset()
        raw_after = pipe.step(0.5)
        # After reset, logit return should be 0 (cold start again)
        assert raw_after["logit_ret"] == pytest.approx(0.0, abs=1e-10)
        # And n_obs should restart
        assert raw_after["n_obs"] == 1


# ---------------------------------------------------------------------------
# BeliefPipelineRegistry tests
# ---------------------------------------------------------------------------

class TestBeliefPipelineRegistry:
    def test_step_creates_pipeline_on_first_call(self):
        reg = BeliefPipelineRegistry()
        assert reg.n_slugs() == 0
        result = reg.step("slug-a", 0.5)
        assert reg.n_slugs() == 1
        assert result.slug == "slug-a"

    def test_slug_isolation(self):
        """Two slugs must maintain independent state."""
        reg = BeliefPipelineRegistry()
        # Feed different trajectories
        for _ in range(10):
            reg.step("slug-a", 0.3)
        for _ in range(10):
            reg.step("slug-b", 0.7)
        r_a = reg.step("slug-a", 0.3)
        r_b = reg.step("slug-b", 0.7)
        # Filtered beliefs should be near their respective p values
        assert abs(r_a.p_filtered - 0.3) < 0.15
        assert abs(r_b.p_filtered - 0.7) < 0.15
        # They must be different
        assert abs(r_a.p_filtered - r_b.p_filtered) > 0.2

    def test_registry_serialisation_roundtrip(self):
        reg = BeliefPipelineRegistry()
        for p in [0.4, 0.5, 0.6]:
            reg.step("market-x", p)
        d = reg.to_dict()
        reg2 = BeliefPipelineRegistry.from_dict(d)
        r1 = reg.step("market-x", 0.55)
        r2 = reg2.step("market-x", 0.55)
        assert r1.p_filtered == pytest.approx(r2.p_filtered, rel=1e-8)
        assert r1.sigma_b    == pytest.approx(r2.sigma_b,    rel=1e-8)

    def test_reset_slug(self):
        reg = BeliefPipelineRegistry()
        for p in [0.3, 0.7, 0.3]:
            reg.step("market-y", p)
        reg.reset_slug("market-y")
        result = reg.step("market-y", 0.5)
        # After reset, cold start → logit return = 0
        assert result.logit_ret == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# PipelineStepResult tests
# ---------------------------------------------------------------------------

class TestPipelineStepResult:
    def _make_result(self, p=0.6):
        reg = BeliefPipelineRegistry()
        return reg.step("test-slug", p)

    def test_uncertainty_property(self):
        r = self._make_result(0.5)
        # p_filtered ≈ 0.5 on cold start → uncertainty ≈ 0.25
        assert 0.0 < r.uncertainty <= 0.25

    def test_to_belief_state(self):
        r = self._make_result(0.65)
        bs = r.to_belief_state()
        assert bs.slug      == "test-slug"
        assert bs.sigma_b   == pytest.approx(r.sigma_b, rel=1e-10)
        assert bs.p_filtered == pytest.approx(r.p_filtered, rel=1e-10)

    def test_to_dict_keys(self):
        r = self._make_result(0.7)
        d = r.to_dict()
        for key in ["slug", "p_observed", "p_filtered", "sigma_b", "jump_score", "is_jump", "n_obs"]:
            assert key in d

    def test_logit_filtered_consistent_with_p_filtered(self):
        r = self._make_result(0.6)
        # logit_filtered should match logit(p_filtered)
        expected_p = sigmoid(r.logit_filtered)
        assert expected_p == pytest.approx(r.p_filtered, rel=1e-6)
