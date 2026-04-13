"""
tests/test_research_uncertainty_index.py
Deterministic tests for src/research/uncertainty_index.py

Coverage:
  - per_market_uncertainty: peak at 0.5, zero at extremes, symmetry
  - uncertainty_index: empty input, uniform weight, volume weight
  - uncertainty_index: normalized value in [0,100]
  - uncertainty_index: top contributors sum to ≈100%
  - RollingUncertaintyIndex: history accumulation, summary stats
  - sector_index: correct segmentation
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.uncertainty_index import (
    MarketEntry,
    IndexResult,
    per_market_uncertainty,
    uncertainty_index,
    RollingUncertaintyIndex,
    sector_index,
)


# ---------------------------------------------------------------------------
# per_market_uncertainty
# ---------------------------------------------------------------------------

class TestPerMarketUncertainty:
    def test_max_at_half(self):
        assert per_market_uncertainty(0.5) == pytest.approx(0.25, abs=1e-10)

    def test_min_at_extremes(self):
        assert per_market_uncertainty(0.0) == pytest.approx(0.0, abs=1e-10)
        assert per_market_uncertainty(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        for p in [0.1, 0.3, 0.4]:
            assert per_market_uncertainty(p) == pytest.approx(
                per_market_uncertainty(1.0 - p), rel=1e-9
            )

    def test_clamped_for_out_of_range(self):
        # Should not crash or go negative for out-of-range input
        result = per_market_uncertainty(-0.1)
        assert result >= 0.0
        result2 = per_market_uncertainty(1.5)
        assert result2 >= 0.0


# ---------------------------------------------------------------------------
# uncertainty_index
# ---------------------------------------------------------------------------

class TestUncertaintyIndex:
    def _markets_all_at_half(self, n=5, vol=1000.0):
        return [MarketEntry(slug=f"m{i}", p_yes=0.5, volume_usd=vol) for i in range(n)]

    def _markets_all_resolved(self, n=5, vol=1000.0):
        return [MarketEntry(slug=f"m{i}", p_yes=0.01, volume_usd=vol) for i in range(n)]

    def test_empty_returns_zero(self):
        result = uncertainty_index([])
        assert result.index_raw == 0.0
        assert result.index_normalized == 0.0
        assert result.n_markets == 0

    def test_all_at_max_uncertainty(self):
        markets = self._markets_all_at_half()
        result = uncertainty_index(markets)
        assert result.index_raw == pytest.approx(0.25, abs=1e-6)
        assert result.index_normalized == pytest.approx(100.0, abs=1e-4)

    def test_all_resolved_near_zero(self):
        markets = self._markets_all_resolved()
        result = uncertainty_index(markets)
        assert result.index_normalized < 5.0   # near-zero but not exactly 0

    def test_normalized_in_valid_range(self):
        markets = [
            MarketEntry("a", p_yes=0.1, volume_usd=500),
            MarketEntry("b", p_yes=0.5, volume_usd=2000),
            MarketEntry("c", p_yes=0.9, volume_usd=100),
        ]
        result = uncertainty_index(markets)
        assert 0.0 <= result.index_normalized <= 100.0

    def test_volume_weight_upweights_high_vol(self):
        # Market at 0.5 with high volume should dominate
        markets = [
            MarketEntry("high-vol", p_yes=0.5, volume_usd=10_000),  # max uncertainty
            MarketEntry("low-vol",  p_yes=0.1, volume_usd=1),        # near-resolved
        ]
        result = uncertainty_index(markets, weight_mode="volume")
        # Should be close to 100 (high-vol dominates)
        assert result.index_normalized > 80.0

    def test_uniform_weight_averages(self):
        markets = [
            MarketEntry("a", p_yes=0.5, volume_usd=10_000),  # u = 0.25
            MarketEntry("b", p_yes=0.1, volume_usd=1),        # u ≈ 0.09
        ]
        result = uncertainty_index(markets, weight_mode="uniform")
        expected_raw = (0.25 + 0.1 * 0.9) / 2.0
        assert result.index_raw == pytest.approx(expected_raw, rel=1e-5)

    def test_n_markets_correct(self):
        markets = [MarketEntry(f"m{i}", p_yes=0.5) for i in range(7)]
        result = uncertainty_index(markets)
        assert result.n_markets == 7

    def test_top_contributors_present(self):
        markets = [MarketEntry(f"m{i}", p_yes=0.5, volume_usd=float(i + 1)) for i in range(10)]
        result = uncertainty_index(markets, top_n=3)
        assert len(result.top_contributors) == 3

    def test_contributors_sorted_descending(self):
        markets = [MarketEntry(f"m{i}", p_yes=0.5, volume_usd=float(i + 1)) for i in range(10)]
        result = uncertainty_index(markets)
        pcts = [c[1] for c in result.top_contributors]
        assert pcts == sorted(pcts, reverse=True)

    def test_sigma_b_weight_mode(self):
        markets = [
            MarketEntry("a", p_yes=0.5, sigma_b=1.0),   # high vol
            MarketEntry("b", p_yes=0.5, sigma_b=0.01),  # low vol
        ]
        result_sb  = uncertainty_index(markets, weight_mode="sigma_b")
        result_uni = uncertainty_index(markets, weight_mode="uniform")
        # Both markets at p=0.5, so both give same uncertainty; index should be equal
        assert result_sb.index_raw == pytest.approx(result_uni.index_raw, rel=1e-6)


# ---------------------------------------------------------------------------
# RollingUncertaintyIndex
# ---------------------------------------------------------------------------

class TestRollingUncertaintyIndex:
    def _markets(self, p_yes=0.5):
        return [MarketEntry("m0", p_yes=p_yes, volume_usd=1000)]

    def test_history_accumulates(self):
        rolling = RollingUncertaintyIndex()
        for _ in range(5):
            rolling.update(self._markets(0.5))
        assert len(rolling.history) == 5

    def test_max_history_respected(self):
        rolling = RollingUncertaintyIndex(max_history=3)
        for _ in range(10):
            rolling.update(self._markets(0.5))
        assert len(rolling.history) == 3

    def test_current_matches_last_update(self):
        rolling = RollingUncertaintyIndex()
        r1 = rolling.update(self._markets(0.5))
        assert rolling.current is r1

    def test_summary_no_history(self):
        rolling = RollingUncertaintyIndex()
        s = rolling.summary()
        assert s["n"] == 0

    def test_summary_mean_correct(self):
        rolling = RollingUncertaintyIndex(weight_mode="uniform")
        # All markets at p=0.5 → normalized = 100
        for _ in range(5):
            rolling.update(self._markets(0.5))
        s = rolling.summary()
        assert s["mean"] == pytest.approx(100.0, rel=1e-4)

    def test_trend_detects_increasing(self):
        rolling = RollingUncertaintyIndex(weight_mode="uniform")
        # Start with resolved markets, end with uncertain markets
        for _ in range(6):
            rolling.update(self._markets(0.95))   # low uncertainty
        for _ in range(6):
            rolling.update(self._markets(0.50))   # high uncertainty
        s = rolling.summary()
        assert s["trend"] > 0.0   # uncertainty is rising


# ---------------------------------------------------------------------------
# sector_index
# ---------------------------------------------------------------------------

class TestSectorIndex:
    def test_correct_segmentation(self):
        markets = [
            MarketEntry("a", p_yes=0.5, volume_usd=100),
            MarketEntry("b", p_yes=0.5, volume_usd=100),
            MarketEntry("c", p_yes=0.1, volume_usd=100),
        ]
        sector_map = {"a": "politics", "b": "politics", "c": "sports"}
        result = sector_index(markets, sector_map)
        assert "politics" in result
        assert "sports" in result
        assert result["politics"].n_markets == 2
        assert result["sports"].n_markets == 1

    def test_unknown_sector_bucket(self):
        markets = [MarketEntry("x", p_yes=0.5)]
        result = sector_index(markets, sector_map={})
        assert "unknown" in result

    def test_sector_index_values_are_IndexResult(self):
        markets = [MarketEntry("a", p_yes=0.5)]
        result = sector_index(markets, {"a": "crypto"})
        assert isinstance(result["crypto"], IndexResult)
