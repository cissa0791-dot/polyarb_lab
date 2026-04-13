"""
src/research/uncertainty_index.py — Track B Phase 2
Prediction-market VIX / aggregate uncertainty index.

Definition
----------
The uncertainty index is a volume-weighted average of per-market uncertainty:

    U = Σ_i  w_i · p_i(1-p_i)  /  Σ_i  w_i

where:
    p_i    = current probability for market i
    w_i    = volume weight for market i  (or uniform if volumes not available)

This is the prediction-market analogue of the VIX: it measures how much
aggregate belief is concentrated near the ambiguous midpoint vs resolved.

Range
-----
    0   — all markets fully resolved (all p_i near 0 or 1)
    0.25 — all markets maximally uncertain (all p_i = 0.5)

Normalised form (0..100):  U_norm = U / 0.25 * 100

Extensions
----------
    - Sector-sliced index (subset of markets by category)
    - Time-to-expiry weighted (upweight markets expiring soon)
    - Belief-vol weighted (upweight markets with high σ_b, more dynamic)
    - Kalman-filtered probability input (rather than raw outcomePrices)

No imports from Track A. Research only.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Market entry for index computation
# ---------------------------------------------------------------------------

@dataclass
class MarketEntry:
    """
    Minimal input record for the uncertainty index.

    slug        : market identifier
    p_yes       : current probability (raw or filtered)
    volume_usd  : trade volume in USD (used as weight; falls back to 1.0)
    sigma_b     : belief volatility estimate (optional; used if vol_weight='sigma_b')
    """
    slug:       str
    p_yes:      float
    volume_usd: float = 1.0
    sigma_b:    float = 0.0


@dataclass
class IndexResult:
    """
    Output of one index computation pass.

    index_raw        : U in [0, 0.25]
    index_normalized : U_norm in [0, 100]
    n_markets        : number of markets included
    weight_sum       : sum of weights (for diagnostics)
    top_contributors : list of (slug, contribution_pct) sorted descending
    """
    index_raw:        float
    index_normalized: float
    n_markets:        int
    weight_sum:       float
    top_contributors: List[tuple]  # (slug, contribution_pct)


# ---------------------------------------------------------------------------
# Core index functions
# ---------------------------------------------------------------------------

def per_market_uncertainty(p: float) -> float:
    """p*(1-p) for a single market. Clamped to [0, 0.25]."""
    p = max(0.0, min(1.0, p))
    return p * (1.0 - p)


def uncertainty_index(
    markets: Sequence[MarketEntry],
    weight_mode: str = "volume",
    top_n: int = 5,
) -> IndexResult:
    """
    Compute the aggregate uncertainty index over a collection of markets.

    Parameters
    ----------
    markets     : sequence of MarketEntry
    weight_mode : one of 'volume', 'uniform', 'sigma_b'
                  'volume'  — weight by volume_usd
                  'uniform' — equal weight
                  'sigma_b' — weight by belief volatility (requires sigma_b field)
    top_n       : number of top contributors to include in result

    Returns
    -------
    IndexResult
    """
    if not markets:
        return IndexResult(
            index_raw=0.0,
            index_normalized=0.0,
            n_markets=0,
            weight_sum=0.0,
            top_contributors=[],
        )

    weights = []
    uncertainties = []

    for m in markets:
        u = per_market_uncertainty(m.p_yes)
        uncertainties.append(u)

        if weight_mode == "volume":
            w = max(0.0, m.volume_usd)
        elif weight_mode == "sigma_b":
            w = max(0.0, m.sigma_b)
        else:  # uniform
            w = 1.0

        # At least tiny weight so every market counts
        weights.append(max(w, 1e-10))

    w_sum = sum(weights)
    index_raw = sum(w * u for w, u in zip(weights, uncertainties)) / w_sum

    # Per-market contribution to index (as % of total)
    contributions = []
    for m, w, u in zip(markets, weights, uncertainties):
        contrib_abs = w * u / w_sum
        contrib_pct = contrib_abs / index_raw * 100.0 if index_raw > 1e-12 else 0.0
        contributions.append((m.slug, round(contrib_pct, 2)))

    contributions.sort(key=lambda x: x[1], reverse=True)

    return IndexResult(
        index_raw=round(index_raw, 8),
        index_normalized=round(index_raw / 0.25 * 100.0, 4),
        n_markets=len(markets),
        weight_sum=round(w_sum, 4),
        top_contributors=contributions[:top_n],
    )


# ---------------------------------------------------------------------------
# Rolling index tracker
# ---------------------------------------------------------------------------

@dataclass
class RollingUncertaintyIndex:
    """
    Tracks the uncertainty index over time with a rolling history.

    Each call to update() appends one IndexResult to the internal buffer.
    Provides summary statistics (mean, stdev, trend).
    """
    weight_mode: str  = "volume"
    max_history: int  = 100
    _history: List[IndexResult] = field(default_factory=list)

    def update(self, markets: Sequence[MarketEntry]) -> IndexResult:
        """Compute index and append to history. Returns current result."""
        result = uncertainty_index(markets, weight_mode=self.weight_mode)
        self._history.append(result)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        return result

    @property
    def history(self) -> List[IndexResult]:
        return list(self._history)

    @property
    def current(self) -> Optional[IndexResult]:
        return self._history[-1] if self._history else None

    def summary(self) -> dict:
        """
        Mean / stdev / trend of index_normalized over rolling window.
        trend > 0 means uncertainty is rising (markets less resolved).
        """
        if not self._history:
            return {"mean": 0.0, "stdev": 0.0, "trend": 0.0, "n": 0}

        vals = [r.index_normalized for r in self._history]
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) >= 2 else 0.0

        # Trend: slope of last 5 vs first 5 in window
        if len(vals) >= 10:
            recent = statistics.mean(vals[-5:])
            older  = statistics.mean(vals[:5])
            trend  = recent - older
        else:
            trend = 0.0

        return {
            "mean":  round(mean,  4),
            "stdev": round(stdev, 4),
            "trend": round(trend, 4),
            "n":     len(vals),
        }


# ---------------------------------------------------------------------------
# Sector-sliced index helper
# ---------------------------------------------------------------------------

def sector_index(
    markets: Sequence[MarketEntry],
    sector_map: Dict[str, str],
    weight_mode: str = "volume",
) -> Dict[str, IndexResult]:
    """
    Compute uncertainty index per sector.

    Parameters
    ----------
    markets    : all market entries
    sector_map : {slug: sector_name} mapping
    weight_mode: weight scheme

    Returns
    -------
    Dict mapping sector_name → IndexResult
    """
    buckets: Dict[str, List[MarketEntry]] = {}
    for m in markets:
        sector = sector_map.get(m.slug, "unknown")
        buckets.setdefault(sector, []).append(m)

    return {
        sector: uncertainty_index(entries, weight_mode=weight_mode)
        for sector, entries in buckets.items()
    }
