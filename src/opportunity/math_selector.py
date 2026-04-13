"""
src/opportunity/math_selector.py

Research-only math-selector exploration prototype.
NOT a production replacement for ExecutionFeasibilityEvaluator.

Mathematical foundation
-----------------------
Constraint polytope (valid-outcome representation):
    Z = {z ∈ {0,1}^I : A^T z ≥ b}      valid binary payoff vectors
    M = conv(Z)                           arbitrage-free marginal polytope

Bregman projection objective (KL / entropy divergence):
    D(μ ‖ θ) = Σ_i μ_i ln(μ_i / θ_i)   generalised KL divergence
    μ* = argmin_{μ ∈ M} D(μ ‖ θ)        closest arbitrage-free belief

Maximum-profit interpretation:
    max_δ min_ω [δ·φ(ω) − C(θ+δ) + C(θ)] = D(μ* ‖ θ)
    i.e. the divergence at the projection equals max guaranteed arbitrage profit.

Frank-Wolfe skeleton (APPROX-4: capped at MAX_FW_ITERS):
    μ_0   ← centroid of Z_vertices
    for t = 0 … T:
        grad_t = ∇D(μ_t ‖ θ) = ln(μ_t) − ln(θ) + 1   [KL gradient]
        z_t    = argmin_{z ∈ Z} grad_t · z              [LMO oracle]
        γ_t    = 2 / (t + 3)                            [standard step]
        μ_{t+1} = (1 − γ_t) μ_t + γ_t z_t             [FW update]
        gap_t  = grad_t · (μ_t − z_t)                  [duality gap]
        if gap_t ≤ ε: break

Per-leg fee model (replaces flat threshold):
    min_viable_edge_cents = n_legs × (fee_per_leg + slip_per_leg) + target_margin

Liquidity cap (max extractable profit):
    liquidity_cap_profit = gross_edge × min_volume_across_legs

Kelly sizing reference (informational only, never used for gating):
    f = ((b × p − q) / b) × sqrt(p)   where b = edge%, p = fill_prob, q = 1−p

Named approximations
--------------------
APPROX-1  LMO solved by exhaustive search over Z_vertices (valid for |Z_vertices| ≤ 8).
APPROX-2  Z_vertices enumerated by family convention, not full IP column generation.
APPROX-3  Log arguments clipped to [1e-7, 1 − 1e-7] to avoid log(0).
APPROX-4  FW capped at MAX_FW_ITERS = 20 iterations, not exact convergence.
APPROX-5  Surrogate D used for single-market pair: D ≈ max(0, 1 − pair_vwap), the
          linear profit rather than full KL.  Exact KL is used for multi-market.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-9
_LOG_CLIP = 1e-7          # APPROX-3
_MAX_FW_ITERS = 20        # APPROX-4
_FW_TOL = 1e-5            # gap convergence tolerance


# ---------------------------------------------------------------------------
# Constraint Polytope  —  Z and M = conv(Z)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstraintPoly:
    """
    Compact polytope for a candidate's logical structure.

    dim         : number of market-position dimensions
    z_vertices  : extreme points of M (APPROX-2: enumerated by family convention)
    family      : strategy family name (for traceability)

    Each vertex z ∈ {0,1}^I represents a valid joint outcome where z_i = 1
    means leg i generates its payout (the bet on leg i wins).

    Formula mapping:  z_vertices ↔ Z  in  Z = {z ∈ {0,1}^I : A^T z ≥ b}
    """
    dim: int
    z_vertices: list[list[float]]
    family: str

    @classmethod
    def for_single_market(cls, family: str) -> "ConstraintPoly":
        """
        2-leg single-market pair (YES/NO).
        Valid outcomes: {z_YES=1, z_NO=0} or {z_YES=0, z_NO=1}.
        M = simplex {(p, 1-p) : p ∈ [0,1]}.
        """
        return cls(dim=2, z_vertices=[[1.0, 0.0], [0.0, 1.0]], family=family)

    @classmethod
    def for_implication_pair(cls, family: str) -> "ConstraintPoly":
        """
        2-leg cross-market implication  A ⊆ B  (e.g. win election ⊆ win nomination).
        Forbidden outcome: (A=1, B=0).
        Valid binary outcomes in execution-profit space (NO-A pays, YES-B pays):
            (A=0, B=0) → z = (1, 0)   NO-A pays, YES-B doesn't
            (A=0, B=1) → z = (1, 1)   both pay
            (A=1, B=1) → z = (0, 1)   only YES-B pays
        Constraint: z_B ≥ z_A  (encoded in vertex exclusion of (1,0) for YES-B side).

        APPROX-2: three-vertex polytope (not the full simplex product).
        """
        return cls(
            dim=2,
            z_vertices=[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            family=family,
        )

    @classmethod
    def for_general(cls, n_legs: int, family: str) -> "ConstraintPoly":
        """
        Fallback for unknown structure: full {0,1}^n hypercube.
        APPROX-2: assumes no logical constraints (conservative).
        Limited to n_legs ≤ 6 to keep |Z_vertices| ≤ 64.
        """
        n = min(n_legs, 6)
        verts = [
            [float((i >> bit) & 1) for bit in range(n)]
            for i in range(2 ** n)
        ]
        return cls(dim=n, z_vertices=verts, family=family)

    @classmethod
    def for_family(cls, family: str, n_legs: int) -> "ConstraintPoly":
        """Select the appropriate polytope by strategy family."""
        if family in (
            "single_market_mispricing",
            "single_market_touch_mispricing",
        ):
            return cls.for_single_market(family)
        if family in (
            "cross_market_constraint",
            "cross_market_gross_constraint",
            "cross_market_execution_gross_constraint",
        ):
            return cls.for_implication_pair(family)
        return cls.for_general(n_legs, family)


# ---------------------------------------------------------------------------
# Bregman / Frank-Wolfe Projection
# ---------------------------------------------------------------------------

def _clip(x: float) -> float:
    """APPROX-3: clip to avoid log(0) and log singularities."""
    return max(_LOG_CLIP, min(1.0 - _LOG_CLIP, x))


def _kl_div(mu: list[float], theta: list[float]) -> float:
    """
    D(μ ‖ θ) = Σ_i μ_i ln(μ_i / θ_i)
    Returns 0 when μ_i → 0 (convention 0 ln 0 = 0).
    """
    total = 0.0
    for m, t in zip(mu, theta):
        m_c = _clip(m)
        t_c = _clip(t)
        total += m_c * math.log(m_c / t_c)
    return max(0.0, total)


def _kl_gradient(mu: list[float], theta: list[float]) -> list[float]:
    """
    ∇_μ D(μ ‖ θ) = ln(μ_i) − ln(θ_i) + 1
    """
    return [math.log(_clip(m)) - math.log(_clip(t)) + 1.0 for m, t in zip(mu, theta)]


def _lmo(grad: list[float], vertices: list[list[float]]) -> list[float]:
    """
    Linear Minimisation Oracle: argmin_{z ∈ Z} grad · z
    APPROX-1: exhaustive search over Z_vertices.
    Valid when |Z_vertices| is small (≤ 64 in practice here).
    """
    best_val = float("inf")
    best_z = vertices[0]
    for z in vertices:
        val = sum(g * zi for g, zi in zip(grad, z))
        if val < best_val:
            best_val = val
            best_z = z
    return list(best_z)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


@dataclass(frozen=True)
class FWProjectionResult:
    """Output of the Frank-Wolfe projection onto M = conv(Z)."""
    mu_star: list[float]    # approximate argmin_{μ ∈ M} D(μ ‖ θ)
    divergence: float       # D(μ* ‖ θ)  — arbitrage profit per unit stake
    gap: float              # duality gap at μ*  — reliability measure
    iters: int              # FW iterations used
    converged: bool         # True if gap ≤ _FW_TOL


def fw_project(theta: list[float], poly: ConstraintPoly) -> FWProjectionResult:
    """
    Approximate Bregman projection of θ onto M = conv(Z) via Frank-Wolfe.

    Returns μ* = argmin_{μ ∈ M} D(μ ‖ θ) with duality gap estimate.
    Higher divergence D(μ* ‖ θ) = stronger arbitrage signal.
    """
    verts = poly.z_vertices

    # μ_0 ← centroid of Z_vertices
    n = poly.dim
    mu = [sum(v[i] for v in verts) / len(verts) for i in range(n)]
    mu = [_clip(m) for m in mu]

    gap = float("inf")
    iters = 0
    for t in range(_MAX_FW_ITERS):                        # APPROX-4
        grad = _kl_gradient(mu, theta)
        z_star = _lmo(grad, verts)
        gap = _dot(grad, [m - z for m, z in zip(mu, z_star)])
        iters = t + 1
        if gap <= _FW_TOL:
            break
        gamma = 2.0 / (t + 3.0)                          # γ_t = 2/(t+2), 1-indexed
        mu = [_clip((1.0 - gamma) * m + gamma * z) for m, z in zip(mu, z_star)]

    divergence = _kl_div(mu, theta)
    return FWProjectionResult(
        mu_star=mu,
        divergence=divergence,
        gap=gap,
        iters=iters,
        converged=gap <= _FW_TOL,
    )


# ---------------------------------------------------------------------------
# Per-Leg Fee Model  —  replaces flat min_edge_cents
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegAdjustedThreshold:
    """
    Per-leg fee-adjusted minimum viable edge.

    Replaces A's flat min_edge_cents with a leg-count-aware threshold:
        min_viable = n_legs × (fee_per_leg + slip_per_leg) + target_margin

    For 2-leg trades (cross-market): min = 2×0.005 + 2×0.005 + 0.005 = 0.025¢
    For 3-leg trades (neg-risk):     min = 3×0.005 + 3×0.005 + 0.005 = 0.035¢
    This is more restrictive for multi-leg and more permissive for 2-leg vs
    A's flat 0.030¢.
    """
    fee_per_leg_cents: float = 0.005
    slip_per_leg_cents: float = 0.005
    target_margin_cents: float = 0.005

    def min_viable(self, n_legs: int) -> float:
        """Returns the minimum net edge (in cents/share) for n_legs legs."""
        return n_legs * (self.fee_per_leg_cents + self.slip_per_leg_cents) + self.target_margin_cents


# ---------------------------------------------------------------------------
# Liquidity-cap and Kelly helpers
# ---------------------------------------------------------------------------

def _liquidity_cap_profit(gross_edge_cents: float, available_shares: float) -> float:
    """
    Max extractable profit = edge × min_volume_across_legs.
    Here we use the stored available_shares (already the cross-leg minimum).
    Formula: profit = (price_deviation) × min(volume_i)
    """
    return gross_edge_cents * available_shares / 100.0   # convert cents to USD


def _kelly_size_ref(
    b: float,
    p: float,
) -> float | None:
    """
    Kelly-style position size reference (informational only, never used for gating).
    f = ((b × p − q) / b) × sqrt(p)
    where b = edge as fraction, p = fill probability, q = 1 − p.
    Returns None if inputs are degenerate.
    """
    if b <= 0.0 or not (0.0 < p < 1.0):
        return None
    q = 1.0 - p
    raw = ((b * p - q) / b) * math.sqrt(p)
    return round(max(0.0, raw), 6)


# ---------------------------------------------------------------------------
# MathSelectorResult  — output of B
# ---------------------------------------------------------------------------

@dataclass
class MathSelectorResult:
    """
    Output of MathCandidateSelector.evaluate().

    score           : D(μ* ‖ θ) / min_viable_edge  (≥ 1.0 = pass)
    divergence      : D(μ* ‖ θ) in probability units (≈ gross_edge for small edges)
    min_viable_edge : per-leg adjusted threshold in cents/share
    fw_gap          : FW duality gap (lower = more reliable divergence estimate)
    fw_converged    : True if FW reached tolerance before iter cap
    fw_iters        : number of FW iterations
    passed_b        : True if score ≥ 1.0 AND all hard gates pass
    reason_codes_b  : rejection codes under B (empty if passed)
    metadata        : full numeric breakdown for audit
    """
    candidate_id: str
    family: str
    n_legs: int
    score: float
    divergence: float
    min_viable_edge: float
    fw_gap: float
    fw_converged: bool
    fw_iters: int
    passed_b: bool
    reason_codes_b: list[str]
    kelly_ref: float | None
    liquidity_cap_profit_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# MathCandidateSelector  — main B selector
# ---------------------------------------------------------------------------

class MathCandidateSelector:
    """
    Research-only alternative qualifier.

    Replaces A's flat-threshold edge gate with:
      1. Family-specific constraint polytope (Z, M)
      2. FW Bregman projection → divergence score D(μ* ‖ θ)
      3. Per-leg fee-adjusted minimum viable edge threshold
      4. Hard gates reused from A: depth, partial-fill risk, non-atomic risk

    Does NOT replace the mainline runner.  Runs in parallel for comparison.

    Parameters
    ----------
    threshold : LegAdjustedThreshold
        Per-leg fee model.
    max_partial_fill_risk : float
        Reused from OpportunityConfig (default 0.65).
    max_non_atomic_risk : float
        Reused from OpportunityConfig (default 0.60).
    min_net_profit_usd : float
        Hard floor on net USD profit (relaxed vs A's 0.50 default).
    """

    def __init__(
        self,
        threshold: LegAdjustedThreshold | None = None,
        max_partial_fill_risk: float = 0.65,
        max_non_atomic_risk: float = 0.60,
        min_net_profit_usd: float = 0.10,
    ):
        self.threshold = threshold or LegAdjustedThreshold()
        self.max_partial_fill_risk = max_partial_fill_risk
        self.max_non_atomic_risk = max_non_atomic_risk
        self.min_net_profit_usd = min_net_profit_usd

    def evaluate(
        self,
        *,
        candidate_id: str,
        family: str,
        n_legs: int,
        gross_edge_cents: float,
        pair_vwap: float,
        expected_payout_per_share: float,
        leg_vwap_prices: list[float],
        available_shares: float,
        available_depth_usd: float,
        required_depth_usd: float,
        partial_fill_risk_score: float,
        non_atomic_execution_risk_score: float,
        expected_net_profit_usd: float,
    ) -> MathSelectorResult:
        """
        Evaluate a candidate using the FW Bregman projection framework.

        leg_vwap_prices : VWAP execution prices per leg (θ vector, probability units).
        gross_edge_cents: gross_edge_cents from the qualification metadata.
        """
        n = max(1, n_legs)
        reason_codes: list[str] = []

        # -- Constraint polytope (Z, M) for this family  ----------------------
        poly = ConstraintPoly.for_family(family, n)

        # -- θ vector: leg VWAP prices (used for FW/KL structural signal)  -----
        # D(μ*‖θ) is the KL-style information divergence of the market prices
        # from the constraint polytope M.  In a *linear* prediction market the
        # actual arbitrage profit is gross_edge_cents (not D×100); D is kept as
        # a structural misalignment signal and ranking metric.
        # APPROX-5: KL divergence ≠ linear profit.  Primary gate uses
        # gross_edge_cents; D is secondary / informational.
        if len(leg_vwap_prices) == poly.dim and all(v is not None for v in leg_vwap_prices):
            theta = [max(_LOG_CLIP, min(1.0 - _LOG_CLIP, float(v))) for v in leg_vwap_prices]
            fw_result = fw_project(theta, poly)
        else:
            theta = []
            fw_result = FWProjectionResult(
                mu_star=[],
                divergence=float("nan"),
                gap=float("nan"),
                iters=0,
                converged=False,
            )
        divergence = fw_result.divergence   # KL structural signal (not financial profit)

        # -- Per-leg minimum viable edge  --------------------------------------
        min_viable = self.threshold.min_viable(n)   # in cents/share

        # Primary financial score: gross_edge_cents / per-leg-adjusted minimum
        # (replaces A's flat min_edge_cents comparison)
        score = gross_edge_cents / max(min_viable, _EPS)

        # -- Gate 1: edge (replaces A's flat EDGE_BELOW_THRESHOLD)  -----------
        # Uses gross_edge_cents vs per-leg min_viable (not KL divergence).
        if gross_edge_cents < min_viable:
            reason_codes.append("MATH_EDGE_BELOW_LEG_ADJUSTED_THRESHOLD")

        # -- Gate 2: depth (reused from A)  ------------------------------------
        if available_depth_usd + _EPS < required_depth_usd:
            reason_codes.append("INSUFFICIENT_DEPTH")

        # -- Gate 3: partial fill risk (reused from A)  -----------------------
        if partial_fill_risk_score > self.max_partial_fill_risk:
            reason_codes.append("PARTIAL_FILL_RISK_TOO_HIGH")

        # -- Gate 4: non-atomic risk (reused from A)  -------------------------
        if non_atomic_execution_risk_score > self.max_non_atomic_risk:
            reason_codes.append("NON_ATOMIC_RISK_TOO_HIGH")

        # -- Gate 5: net profit floor (relaxed vs A)  -------------------------
        if expected_net_profit_usd < self.min_net_profit_usd:
            reason_codes.append("NET_PROFIT_BELOW_THRESHOLD")

        passed_b = len(reason_codes) == 0

        # -- Kelly size reference (informational only)  ------------------------
        fill_prob = min(1.0, available_shares / max(1.0, available_shares)) if available_shares > 0 else 0.0
        kelly_ref = _kelly_size_ref(b=max(0.0, gross_edge_cents / 100.0), p=max(_EPS, fill_prob))

        # -- Liquidity cap profit  --------------------------------------------
        liq_cap = _liquidity_cap_profit(gross_edge_cents, available_shares)

        return MathSelectorResult(
            candidate_id=candidate_id,
            family=family,
            n_legs=n,
            score=round(score, 6),
            divergence=round(divergence, 8),
            min_viable_edge=round(min_viable, 6),
            fw_gap=round(fw_result.gap, 8) if not math.isnan(fw_result.gap) else float("nan"),
            fw_converged=fw_result.converged,
            fw_iters=fw_result.iters,
            passed_b=passed_b,
            reason_codes_b=sorted(reason_codes),
            kelly_ref=kelly_ref,
            liquidity_cap_profit_usd=round(liq_cap, 6),
            metadata={
                "poly_family": poly.family,
                "poly_dim": poly.dim,
                "n_z_vertices": len(poly.z_vertices),
                "theta": [round(t, 6) for t in theta],
                "mu_star": [round(m, 6) for m in fw_result.mu_star],
                "kl_divergence": round(divergence, 8) if divergence == divergence else None,
                "kl_divergence_note": "KL structural signal, NOT linear profit",
                "min_viable_edge_cents": round(min_viable, 6),
                "score": round(score, 6),
                "fw_gap": fw_result.gap,
                "fw_iters": fw_result.iters,
                "fw_converged": fw_result.converged,
                "gross_edge_cents": gross_edge_cents,
                "available_shares": available_shares,
                "available_depth_usd": available_depth_usd,
                "required_depth_usd": required_depth_usd,
                "partial_fill_risk_score": partial_fill_risk_score,
                "non_atomic_execution_risk_score": non_atomic_execution_risk_score,
                "kelly_ref": kelly_ref,
                "liquidity_cap_profit_usd": liq_cap,
                "approximations_used": [
                    "APPROX-1 (LMO exhaustive search)",
                    "APPROX-2 (Z_vertices by family convention)",
                    "APPROX-3 (log clipping 1e-7)",
                    "APPROX-4 (FW cap 20 iters)",
                ] + (["APPROX-5 (surrogate D)"] if fw_result.iters == 0 else []),
            },
        )

    @classmethod
    def from_stored_qualification_metadata(
        cls,
        selector: "MathCandidateSelector",
        candidate_id: str,
        family: str,
        raw_candidate: dict,
        qual_meta: dict,
    ) -> MathSelectorResult:
        """
        Reconstruct a MathSelectorResult from stored rejection event payload.
        Used in the comparison script to replay B on historical candidates.
        """
        legs = qual_meta.get("legs") or raw_candidate.get("legs", [])
        n_legs = len(legs)
        leg_vwaps = [leg.get("vwap_price") for leg in legs if leg.get("vwap_price") is not None]

        return selector.evaluate(
            candidate_id=candidate_id,
            family=family,
            n_legs=n_legs,
            gross_edge_cents=qual_meta.get("expected_gross_edge_cents", 0.0),
            pair_vwap=qual_meta.get("pair_vwap", 0.0),
            expected_payout_per_share=(
                raw_candidate.get("expected_payout", 0.0) / max(raw_candidate.get("target_shares", 1.0), _EPS)
            ),
            leg_vwap_prices=leg_vwaps,
            available_shares=qual_meta.get("available_shares", 0.0),
            available_depth_usd=qual_meta.get("available_depth_usd", 0.0),
            required_depth_usd=qual_meta.get("required_depth_usd", 0.0),
            partial_fill_risk_score=qual_meta.get("partial_fill_risk_score", 0.0),
            non_atomic_execution_risk_score=qual_meta.get("non_atomic_execution_risk_score", 0.0),
            expected_net_profit_usd=qual_meta.get("expected_net_profit_usd", 0.0),
        )
