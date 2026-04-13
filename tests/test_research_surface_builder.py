"""
tests/test_research_surface_builder.py
Deterministic tests for src/research/surface_builder.py

Coverage:
  - _p_bucket_index: boundary mapping
  - _tte_bucket_index: boundary mapping
  - SurfaceCell: add, mean, variance, stdev, to_dict / from_dict
  - BeliefVolSurface: single add, correct cell routing
  - BeliefVolSurface: add_batch increments n_observations
  - BeliefVolSurface: mean_surface returns correct shape
  - BeliefVolSurface: report sorted descending, fields present
  - BeliefVolSurface: top_cells returns at most n rows
  - BeliefVolSurface: serialisation roundtrip
  - BeliefVolSurface: anomalous_cells detects outliers
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.surface_builder import (
    _p_bucket_index,
    _tte_bucket_index,
    p_bucket_label,
    tte_bucket_label,
    SurfaceCell,
    SurfaceObservation,
    BeliefVolSurface,
)


# ---------------------------------------------------------------------------
# Bucket index helpers
# ---------------------------------------------------------------------------

class TestBucketIndexHelpers:
    def test_p_bucket_boundary_values(self):
        assert _p_bucket_index(0.0) == 0
        assert _p_bucket_index(0.05) == 0
        assert _p_bucket_index(0.1) == 1
        assert _p_bucket_index(0.95) == 9
        assert _p_bucket_index(1.0) == 9   # clamped

    def test_p_bucket_midpoints(self):
        for i in range(10):
            mid = i / 10.0 + 0.05
            assert _p_bucket_index(mid) == i

    def test_tte_bucket_boundaries(self):
        assert _tte_bucket_index(0.0) == 0    # < 1d
        assert _tte_bucket_index(0.5) == 0    # < 1d
        assert _tte_bucket_index(1.0) == 1    # 1-7d
        assert _tte_bucket_index(6.9) == 1
        assert _tte_bucket_index(7.0) == 2    # 7-30d
        assert _tte_bucket_index(29.9) == 2
        assert _tte_bucket_index(30.0) == 3   # 30-90d
        assert _tte_bucket_index(89.9) == 3
        assert _tte_bucket_index(90.0) == 4   # >90d
        assert _tte_bucket_index(365.0) == 4

    def test_labels_not_empty(self):
        for i in range(10):
            assert len(p_bucket_label(i)) > 0
        for i in range(5):
            assert len(tte_bucket_label(i)) > 0


# ---------------------------------------------------------------------------
# SurfaceCell
# ---------------------------------------------------------------------------

class TestSurfaceCell:
    def test_empty_cell_mean_nan(self):
        cell = SurfaceCell()
        assert math.isnan(cell.mean)

    def test_single_observation(self):
        cell = SurfaceCell()
        cell.add(0.15)
        assert cell.count == 1
        assert cell.mean == pytest.approx(0.15, rel=1e-9)

    def test_mean_correct(self):
        cell = SurfaceCell()
        for v in [0.10, 0.20, 0.30]:
            cell.add(v)
        assert cell.mean == pytest.approx(0.20, rel=1e-9)

    def test_variance_correct(self):
        cell = SurfaceCell()
        # E[X²] - E[X]² for values [0, 0.4]
        cell.add(0.0)
        cell.add(0.4)
        expected_mean = 0.2
        expected_var  = (0.0 + 0.16) / 2.0 - expected_mean ** 2   # = 0.04
        assert cell.variance == pytest.approx(expected_var, rel=1e-9)

    def test_stdev_positive(self):
        cell = SurfaceCell()
        cell.add(0.1)
        cell.add(0.2)
        assert cell.stdev > 0.0

    def test_single_obs_variance_nan(self):
        cell = SurfaceCell()
        cell.add(0.1)
        assert math.isnan(cell.variance)

    def test_serialisation_roundtrip(self):
        cell = SurfaceCell()
        for v in [0.10, 0.15, 0.20, 0.12]:
            cell.add(v)
        d = cell.to_dict()
        cell2 = SurfaceCell.from_dict(d)
        assert cell2.count       == cell.count
        assert cell2.sum_sigma_b == pytest.approx(cell.sum_sigma_b, rel=1e-12)
        assert cell2.mean        == pytest.approx(cell.mean, rel=1e-9)


# ---------------------------------------------------------------------------
# BeliefVolSurface
# ---------------------------------------------------------------------------

class TestBeliefVolSurface:
    def _obs(self, p_yes=0.5, tte_days=5.0, sigma_b=0.10, slug="m0"):
        return SurfaceObservation(slug=slug, p_yes=p_yes, tte_days=tte_days, sigma_b=sigma_b)

    def test_initial_state_empty(self):
        surface = BeliefVolSurface()
        assert surface.n_observations == 0

    def test_add_increments_n_observations(self):
        surface = BeliefVolSurface()
        surface.add(self._obs())
        assert surface.n_observations == 1

    def test_add_routes_to_correct_cell(self):
        surface = BeliefVolSurface()
        # p=0.5 → p_idx=5; tte=5 → tte_idx=1
        p_idx, tte_idx = surface.add(self._obs(p_yes=0.5, tte_days=5.0, sigma_b=0.15))
        assert p_idx  == 5
        assert tte_idx == 1
        assert surface.cell(5, 1).count == 1
        assert surface.cell(5, 1).mean == pytest.approx(0.15, rel=1e-9)

    def test_add_batch(self):
        surface = BeliefVolSurface()
        obs_list = [self._obs(sigma_b=0.1 * (i + 1)) for i in range(5)]
        surface.add_batch(obs_list)
        assert surface.n_observations == 5

    def test_mean_surface_shape(self):
        surface = BeliefVolSurface()
        grid = surface.mean_surface()
        assert len(grid) == 10
        assert all(len(row) == 5 for row in grid)

    def test_empty_cells_return_none(self):
        surface = BeliefVolSurface()
        grid = surface.mean_surface()
        # All cells should be None when empty
        assert all(cell is None for row in grid for cell in row)

    def test_non_empty_cell_has_float_mean(self):
        surface = BeliefVolSurface()
        surface.add(self._obs(p_yes=0.5, tte_days=5.0, sigma_b=0.12))
        grid = surface.mean_surface()
        assert grid[5][1] == pytest.approx(0.12, rel=1e-9)

    def test_report_sorted_descending(self):
        surface = BeliefVolSurface()
        surface.add(self._obs(p_yes=0.5, tte_days=5.0, sigma_b=0.10, slug="a"))
        surface.add(self._obs(p_yes=0.2, tte_days=50.0, sigma_b=0.50, slug="b"))
        surface.add(self._obs(p_yes=0.8, tte_days=0.5, sigma_b=0.05, slug="c"))
        rows = surface.report()
        means = [r["mean_sigma_b"] for r in rows]
        assert means == sorted(means, reverse=True)

    def test_report_required_keys(self):
        surface = BeliefVolSurface()
        surface.add(self._obs())
        rows = surface.report()
        assert len(rows) > 0
        required = {"p_bucket", "tte_bucket", "mean_sigma_b", "count", "p_idx", "tte_idx"}
        assert required.issubset(rows[0].keys())

    def test_top_cells_at_most_n(self):
        surface = BeliefVolSurface()
        for i in range(10):
            p = 0.05 + i * 0.09
            tte = float(i + 1)
            surface.add(self._obs(p_yes=p, tte_days=tte, sigma_b=0.1 + i * 0.02, slug=f"m{i}"))
        assert len(surface.top_cells(n=3)) <= 3

    def test_min_count_filter_in_report(self):
        surface = BeliefVolSurface()
        surface.add(self._obs(p_yes=0.5, tte_days=5.0, sigma_b=0.10))  # single obs in one cell
        # min_count=2 should exclude single-observation cells
        rows = surface.report(min_count=2)
        assert len(rows) == 0

    def test_serialisation_roundtrip(self):
        surface = BeliefVolSurface()
        for i in range(8):
            surface.add(self._obs(
                p_yes=0.1 + i * 0.1,
                tte_days=float(i + 1),
                sigma_b=0.05 + i * 0.03,
                slug=f"m{i}",
            ))
        d = surface.to_dict()
        surface2 = BeliefVolSurface.from_dict(d)
        assert surface2.n_observations == surface.n_observations
        # Check a specific non-empty cell matches
        for p in range(10):
            for t in range(5):
                c1 = surface.cell(p, t)
                c2 = surface2.cell(p, t)
                assert c2.count == c1.count
                if c1.count > 0:
                    assert c2.mean == pytest.approx(c1.mean, rel=1e-9)

    def test_anomalous_cells_detects_outlier(self):
        surface = BeliefVolSurface()
        # Fill most cells with sigma_b ≈ 0.10 (3 obs each for min_count)
        for p_bucket in range(5):
            p = 0.05 + p_bucket * 0.1
            for tte_bucket in range(3):
                tte = [0.5, 3.0, 15.0][tte_bucket]
                for _ in range(3):
                    surface.add(self._obs(p_yes=p, tte_days=tte, sigma_b=0.10))
        # Add one extremely high sigma_b cell
        for _ in range(3):
            surface.add(self._obs(p_yes=0.85, tte_days=0.5, sigma_b=2.0))  # very high
        anomalies = surface.anomalous_cells(z_thresh=2.0, min_count=3)
        # The high sigma_b cell should appear as anomalous
        assert len(anomalies) >= 1
        assert anomalies[0]["mean_sigma_b"] > 1.0

    def test_anomalous_insufficient_data(self):
        """Should return empty list when not enough cells."""
        surface = BeliefVolSurface()
        surface.add(self._obs())
        anomalies = surface.anomalous_cells(min_count=1)
        # Only one cell, can't compute z-score
        assert anomalies == []
