"""
src/research/derivatives_roadmap.py — Track B Phase 5 (Roadmap Stubs)

Derivative-layer research prototypes for the belief dynamics theory stack.

These are paper-only theoretical constructs — no live trading, no execution.
They represent Phase 5 research objects that the theory stack supports once
the belief volatility surface and pipeline are sufficiently warmed up.

Phase 5 targets
---------------
1. BeliefVarianceSwap       — pays realized σ_b² vs. fixed strike K_var
2. CorridorVariance         — pays variance only when p_filtered in corridor
3. FirstTouchNote           — pays at first time p_filtered crosses a barrier
4. BeliefCorrelationSwap    — pays cross-market logit-return correlation vs. K_rho

Mathematical foundations
------------------------
All derivatives are on the belief process:
    dx_t = σ_b dW_t + J_t dN_t    (martingale, zero drift)

Realized variance:
    V_T = (1/T) Σ_t r_t²           (mean squared logit return)
    payoff: V_T - K_var

Corridor realized variance:
    V_corr = mean(r_t² | p_lower < p_filtered_t < p_upper)
    Only accumulates variance when belief is in the active corridor.

First touch:
    τ = min{t : p_filtered_t crosses barrier B}
    payoff = 1{τ < T}

Correlation:
    ρ_{AB} = cov(r_A, r_B) / (σ_A · σ_B)
    payoff: ρ_{AB} - K_rho

Implementation status: Phase 5 stubs — mathematically specified, not yet
integrated into the ranker or surface builder. Depend on pipeline.py being
warmed up with sufficient observations.

No imports from Track A. Paper research only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Belief variance swap — Phase 5 stub
# ---------------------------------------------------------------------------

@dataclass
class BeliefVarianceSwap:
    """
    Phase 5 stub: realized belief-variance swap.

    Payoff
    ------
    V_T - K_var
    where V_T = (1/T) Σ_t r_t²  (realized mean logit variance)

    Seller receives K_var (fixed leg); buyer receives V_T (floating leg).
    This is the belief-space analogue of a variance swap on a financial asset.

    Depends on: pipeline.logit_ret (filtered-space logit returns)
    """
    slug:          str
    K_var:         float               # variance strike (logit² per step)
    T_steps:       int = 100           # expected number of steps (normalisation)
    _returns:      List[float] = field(default_factory=list, init=False, repr=False)

    def add_return(self, r: float) -> None:
        """Accumulate one logit return observation."""
        self._returns.append(r)

    @property
    def realized_variance(self) -> float:
        """V_T = (1/n) Σ r_t²"""
        if not self._returns:
            return 0.0
        return sum(r * r for r in self._returns) / len(self._returns)

    @property
    def floating_pnl(self) -> float:
        """Current floating PnL = V_T - K_var (paper only)."""
        return self.realized_variance - self.K_var

    @property
    def n_obs(self) -> int:
        return len(self._returns)

    @property
    def fraction_complete(self) -> float:
        """Progress toward T_steps."""
        return min(1.0, self.n_obs / max(1, self.T_steps))


# ---------------------------------------------------------------------------
# 2. Corridor variance — Phase 5 stub
# ---------------------------------------------------------------------------

@dataclass
class CorridorVariance:
    """
    Phase 5 stub: corridor realized variance.

    Payoff
    ------
    V_corr - K_var
    where V_corr = mean(r_t² | p_lower < p_filtered_t < p_upper)

    Only accumulates variance when the filtered belief is within the corridor.
    Markets at extremes (near resolution) contribute nothing.

    This is the belief analogue of a corridor variance swap — useful for
    betting on belief volatility only during the 'live' phase of a market.

    Depends on: pipeline.p_filtered, pipeline.logit_ret
    """
    slug:     str
    K_var:    float         # variance strike
    p_lower:  float = 0.30  # corridor lower bound in probability space
    p_upper:  float = 0.70  # corridor upper bound in probability space

    _sum_r2_in: float = field(default=0.0, init=False, repr=False)
    _n_in:      int   = field(default=0,   init=False, repr=False)
    _n_total:   int   = field(default=0,   init=False, repr=False)

    def add_observation(self, r: float, p_filtered: float) -> bool:
        """
        Add one logit return r with current filtered probability.
        Returns True if this observation was inside the corridor.
        """
        self._n_total += 1
        inside = self.p_lower < p_filtered < self.p_upper
        if inside:
            self._sum_r2_in += r * r
            self._n_in      += 1
        return inside

    @property
    def realized_corridor_variance(self) -> float:
        """Mean of r² for in-corridor observations."""
        if self._n_in == 0:
            return 0.0
        return self._sum_r2_in / self._n_in

    @property
    def corridor_utilisation(self) -> float:
        """Fraction of steps inside the corridor."""
        if self._n_total == 0:
            return 0.0
        return self._n_in / self._n_total

    @property
    def floating_pnl(self) -> float:
        return self.realized_corridor_variance - self.K_var


# ---------------------------------------------------------------------------
# 3. First-touch note — Phase 5 stub
# ---------------------------------------------------------------------------

@dataclass
class FirstTouchNote:
    """
    Phase 5 stub: first-touch binary note.

    Payoff
    ------
    1.0 at first time p_filtered crosses barrier B before T_steps.
    Expires worthless if barrier never touched.

    Captures the timing of belief state changes:
        e.g. 'market crosses 80% for the first time' → pays 1.0 immediately.

    Depends on: pipeline.p_filtered
    """
    slug:      str
    barrier:   float         # probability level to watch
    direction: str  = "up"   # 'up' (crosses above) or 'down' (crosses below)
    T_steps:   int  = 100

    _touched:    bool = field(default=False, init=False, repr=False)
    _touch_step: int  = field(default=-1,    init=False, repr=False)
    _step:       int  = field(default=0,     init=False, repr=False)

    def step(self, p_filtered: float) -> bool:
        """
        Process one filtered probability.
        Returns True if barrier touched on this step (first touch only).
        """
        self._step += 1
        if self._touched:
            return False  # already triggered

        hit = (
            (self.direction == "up"   and p_filtered >= self.barrier) or
            (self.direction == "down" and p_filtered <= self.barrier)
        )
        if hit:
            self._touched    = True
            self._touch_step = self._step
        return hit

    @property
    def payoff(self) -> float:
        """1.0 if touched before expiry, 0.0 otherwise."""
        return 1.0 if self._touched else 0.0

    @property
    def touched(self) -> bool:
        return self._touched

    @property
    def touch_step(self) -> int:
        return self._touch_step

    @property
    def steps_remaining(self) -> int:
        return max(0, self.T_steps - self._step)

    @property
    def expired(self) -> bool:
        return self._step >= self.T_steps


# ---------------------------------------------------------------------------
# 4. Belief correlation swap — Phase 5 stub
# ---------------------------------------------------------------------------

@dataclass
class BeliefCorrelationSwap:
    """
    Phase 5 stub: cross-market belief correlation swap.

    Payoff
    ------
    ρ_{AB}(realized) - K_rho
    where ρ_{AB} = Pearson correlation of logit return series for slugs A and B.

    Buyer bets that markets A and B move together more than K_rho.
    Useful for detecting cross-market belief synchronisation events
    (e.g., correlated political or economic market updates).

    Depends on: pipeline.logit_ret for both slugs
    """
    slug_a:  str
    slug_b:  str
    K_rho:   float = 0.0  # correlation strike

    _r_a: List[float] = field(default_factory=list, init=False, repr=False)
    _r_b: List[float] = field(default_factory=list, init=False, repr=False)

    def add_returns(self, r_a: float, r_b: float) -> None:
        """Add one synchronized return pair (r_a, r_b)."""
        self._r_a.append(r_a)
        self._r_b.append(r_b)

    @property
    def n_obs(self) -> int:
        return min(len(self._r_a), len(self._r_b))

    @property
    def realized_correlation(self) -> float:
        """
        Pearson correlation of accumulated logit return series.
        Returns 0.0 if fewer than 2 observations.
        """
        n = self.n_obs
        if n < 2:
            return 0.0
        ra = self._r_a[:n]
        rb = self._r_b[:n]
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        cov    = sum((a - mean_a) * (b - mean_b) for a, b in zip(ra, rb)) / n
        var_a  = sum((a - mean_a) ** 2 for a in ra) / n
        var_b  = sum((b - mean_b) ** 2 for b in rb) / n
        denom  = math.sqrt(max(var_a, 1e-12) * max(var_b, 1e-12))
        return cov / denom

    @property
    def floating_pnl(self) -> float:
        return self.realized_correlation - self.K_rho


# ---------------------------------------------------------------------------
# Phase 5 roadmap index
# ---------------------------------------------------------------------------

PHASE_5_ROADMAP: Dict[str, dict] = {
    "BeliefVarianceSwap": {
        "status":      "stub",
        "phase":       5,
        "depends_on":  ["pipeline.logit_ret", "theory.EWMABeliefVol"],
        "description": "Realized variance payoff on filtered logit return series",
        "formula":     "V_T = (1/T) Σ r_t² ; payoff = V_T - K_var",
    },
    "CorridorVariance": {
        "status":      "stub",
        "phase":       5,
        "depends_on":  ["pipeline.p_filtered", "pipeline.logit_ret"],
        "description": "Corridor-conditional realized variance",
        "formula":     "V_corr = mean(r_t² | p_lower < p_t < p_upper) ; payoff = V_corr - K_var",
    },
    "FirstTouchNote": {
        "status":      "stub",
        "phase":       5,
        "depends_on":  ["pipeline.p_filtered"],
        "description": "Binary payoff at first barrier crossing in filtered belief",
        "formula":     "payoff = 1{τ < T} where τ = first hit time of barrier B",
    },
    "BeliefCorrelationSwap": {
        "status":      "stub",
        "phase":       5,
        "depends_on":  ["pipeline.logit_ret (both slugs)"],
        "description": "Cross-market logit-return correlation swap",
        "formula":     "payoff = corr(r_A, r_B) - K_rho",
    },
}
