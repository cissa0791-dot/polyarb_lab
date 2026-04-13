"""
src/research/surface_builder.py — Track B Phase 2
Belief volatility surface grid accumulator.

Concept
-------
The belief volatility surface maps (p_bucket, tte_bucket) → mean σ_b.

Axes:
    p_bucket   : probability bucket (e.g. [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0])
    tte_bucket : time-to-expiry bucket (e.g. <1d, 1-7d, 7-30d, 30-90d, >90d)

Each cell accumulates (sum_sigma_b, count) for all observations that
fall into that (p_bucket, tte_bucket) cell.

Mean σ_b per cell:
    surface[p_bucket][tte_bucket] = sum_sigma_b / count

Uses
----
- Visualise how volatility varies across the probability × time space
- Identify which market regimes produce highest σ_b
- Input for A-S spread calibration at different market states
- Detect anomalous cells (unusually high/low σ_b for their regime)

No imports from Track A. Research only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

# Probability buckets: 10 equal-width bins over [0, 1]
P_BUCKET_EDGES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Time-to-expiry buckets in days
TTE_BUCKET_EDGES: List[float] = [0.0, 1.0, 7.0, 30.0, 90.0, float("inf")]
TTE_BUCKET_LABELS: List[str]  = ["<1d", "1-7d", "7-30d", "30-90d", ">90d"]


def _p_bucket_index(p: float) -> int:
    """Map probability p to bucket index [0, 9]."""
    p = max(0.0, min(1.0 - 1e-10, p))
    idx = int(p * 10)
    return min(idx, 9)


def _tte_bucket_index(tte_days: float) -> int:
    """Map time-to-expiry (days) to bucket index [0, 4]."""
    for i, edge in enumerate(TTE_BUCKET_EDGES[1:]):
        if tte_days < edge:
            return i
    return len(TTE_BUCKET_LABELS) - 1


def p_bucket_label(idx: int) -> str:
    """Human-readable label for p_bucket index."""
    lo = P_BUCKET_EDGES[idx]
    hi = P_BUCKET_EDGES[idx + 1]
    return f"[{lo:.1f},{hi:.1f})"


def tte_bucket_label(idx: int) -> str:
    """Human-readable label for tte_bucket index."""
    return TTE_BUCKET_LABELS[idx]


# ---------------------------------------------------------------------------
# Cell accumulator
# ---------------------------------------------------------------------------

@dataclass
class SurfaceCell:
    """Accumulates sigma_b observations for one (p_bucket, tte_bucket) cell."""
    sum_sigma_b:  float = 0.0
    sum_sigma_b2: float = 0.0   # sum of squares for variance computation
    count:        int   = 0

    def add(self, sigma_b: float) -> None:
        self.sum_sigma_b  += sigma_b
        self.sum_sigma_b2 += sigma_b * sigma_b
        self.count        += 1

    @property
    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.sum_sigma_b / self.count

    @property
    def variance(self) -> float:
        if self.count < 2:
            return float("nan")
        m = self.mean
        # E[X²] - E[X]²
        return max(0.0, self.sum_sigma_b2 / self.count - m * m)

    @property
    def stdev(self) -> float:
        v = self.variance
        return math.sqrt(v) if not math.isnan(v) else float("nan")

    def to_dict(self) -> dict:
        return {
            "sum_sigma_b":  self.sum_sigma_b,
            "sum_sigma_b2": self.sum_sigma_b2,
            "count":        self.count,
            "mean":         self.mean,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurfaceCell":
        obj = cls(
            sum_sigma_b=d["sum_sigma_b"],
            sum_sigma_b2=d["sum_sigma_b2"],
            count=d["count"],
        )
        return obj


# ---------------------------------------------------------------------------
# Surface observation
# ---------------------------------------------------------------------------

@dataclass
class SurfaceObservation:
    """
    One data point to add to the surface.

    slug      : market identifier
    p_yes     : current probability
    tte_days  : time to expiry in calendar days (0 = expired/expiring today)
    sigma_b   : current belief volatility estimate (logit units)
    """
    slug:     str
    p_yes:    float
    tte_days: float
    sigma_b:  float


# ---------------------------------------------------------------------------
# Surface builder
# ---------------------------------------------------------------------------

class BeliefVolSurface:
    """
    Accumulates σ_b observations and builds a (p_bucket × tte_bucket) surface.

    Grid dimensions: 10 p_buckets × 5 tte_buckets = 50 cells

    Usage
    -----
    surface = BeliefVolSurface()
    surface.add(SurfaceObservation(slug="market-a", p_yes=0.62, tte_days=5.0, sigma_b=0.12))
    ...
    report = surface.report()
    """

    N_P_BUCKETS:   int = 10
    N_TTE_BUCKETS: int = 5

    def __init__(self) -> None:
        # Grid[p_idx][tte_idx]
        self._grid: List[List[SurfaceCell]] = [
            [SurfaceCell() for _ in range(self.N_TTE_BUCKETS)]
            for _ in range(self.N_P_BUCKETS)
        ]
        self._n_observations: int = 0

    # ------------------------------------------------------------------

    def add(self, obs: SurfaceObservation) -> Tuple[int, int]:
        """
        Add one observation to the surface.
        Returns (p_idx, tte_idx) of the cell that was updated.
        """
        p_idx   = _p_bucket_index(obs.p_yes)
        tte_idx = _tte_bucket_index(obs.tte_days)
        self._grid[p_idx][tte_idx].add(obs.sigma_b)
        self._n_observations += 1
        return p_idx, tte_idx

    def add_batch(self, observations: List[SurfaceObservation]) -> None:
        """Add multiple observations."""
        for obs in observations:
            self.add(obs)

    # ------------------------------------------------------------------

    @property
    def n_observations(self) -> int:
        return self._n_observations

    def cell(self, p_idx: int, tte_idx: int) -> SurfaceCell:
        """Direct cell access (read-only use recommended)."""
        return self._grid[p_idx][tte_idx]

    def mean_surface(self) -> List[List[Optional[float]]]:
        """
        10×5 grid of mean σ_b values.
        None for empty cells.
        Returns grid[p_idx][tte_idx].
        """
        return [
            [
                self._grid[p][t].mean if self._grid[p][t].count > 0 else None
                for t in range(self.N_TTE_BUCKETS)
            ]
            for p in range(self.N_P_BUCKETS)
        ]

    def report(self, min_count: int = 1) -> List[dict]:
        """
        Flat list of non-empty cells as dicts, sorted by mean σ_b descending.

        Each dict contains:
            p_bucket_label, tte_bucket_label, mean_sigma_b,
            stdev_sigma_b, count, p_idx, tte_idx
        """
        rows = []
        for p in range(self.N_P_BUCKETS):
            for t in range(self.N_TTE_BUCKETS):
                cell = self._grid[p][t]
                if cell.count < min_count:
                    continue
                rows.append({
                    "p_bucket":     p_bucket_label(p),
                    "tte_bucket":   tte_bucket_label(t),
                    "mean_sigma_b": round(cell.mean, 6),
                    "stdev_sigma_b": round(cell.stdev, 6) if not math.isnan(cell.stdev) else None,
                    "count":        cell.count,
                    "p_idx":        p,
                    "tte_idx":      t,
                })
        rows.sort(key=lambda r: r["mean_sigma_b"], reverse=True)
        return rows

    def top_cells(self, n: int = 5, min_count: int = 1) -> List[dict]:
        """Top-n cells by mean σ_b."""
        return self.report(min_count=min_count)[:n]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_observations": self._n_observations,
            "grid": [
                [self._grid[p][t].to_dict() for t in range(self.N_TTE_BUCKETS)]
                for p in range(self.N_P_BUCKETS)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefVolSurface":
        obj = cls()
        obj._n_observations = d.get("n_observations", 0)
        raw_grid = d.get("grid", [])
        for p in range(cls.N_P_BUCKETS):
            for t in range(cls.N_TTE_BUCKETS):
                try:
                    obj._grid[p][t] = SurfaceCell.from_dict(raw_grid[p][t])
                except (IndexError, KeyError, TypeError):
                    pass  # leave as empty cell
        return obj

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def anomalous_cells(
        self,
        z_thresh: float = 2.0,
        min_count: int = 3,
    ) -> List[dict]:
        """
        Identify cells whose mean σ_b deviates significantly from the
        global mean (by z_thresh standard deviations).

        Requires min_count observations per cell for inclusion.
        Returns list of anomalous cell dicts with 'z_score' field added.
        """
        rows = self.report(min_count=min_count)
        if len(rows) < 2:
            return []

        means = [r["mean_sigma_b"] for r in rows]
        global_mean = sum(means) / len(means)
        variance    = sum((m - global_mean) ** 2 for m in means) / len(means)
        global_std  = math.sqrt(max(variance, 1e-12))

        anomalies = []
        for row in rows:
            z = (row["mean_sigma_b"] - global_mean) / global_std
            if abs(z) >= z_thresh:
                row = dict(row)
                row["z_score"] = round(z, 3)
                row["global_mean"] = round(global_mean, 6)
                anomalies.append(row)

        anomalies.sort(key=lambda r: abs(r["z_score"]), reverse=True)
        return anomalies
