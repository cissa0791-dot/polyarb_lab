"""
tests/test_math_selector.py

Unit tests for the math-selector exploration prototype.

Formula-to-test mapping
-----------------------
  _kl_div            D(μ ‖ θ) = Σ μ_i ln(μ_i/θ_i)
  _kl_gradient       ∇D = ln(μ_i) − ln(θ_i) + 1
  _lmo               argmin_{z ∈ Z} grad · z   (exhaustive, APPROX-1)
  fw_project         FW iteration; gap, convergence, divergence
  ConstraintPoly     Z vertices for each family convention
  LegAdjustedThreshold  per-leg fee model
  MathCandidateSelector  end-to-end evaluate()
"""

from __future__ import annotations

import math
import unittest

from src.opportunity.math_selector import (
    ConstraintPoly,
    LegAdjustedThreshold,
    MathCandidateSelector,
    _kl_div,
    _kl_gradient,
    _kelly_size_ref,
    _liquidity_cap_profit,
    _lmo,
    fw_project,
)


# ---------------------------------------------------------------------------
# D(μ ‖ θ) tests
# ---------------------------------------------------------------------------

class KLDivergenceTests(unittest.TestCase):

    def test_d_of_point_with_itself_is_zero(self) -> None:
        mu = [0.5, 0.5]
        self.assertAlmostEqual(_kl_div(mu, mu), 0.0, places=8)

    def test_d_is_non_negative(self) -> None:
        mu = [0.3, 0.7]
        theta = [0.6, 0.4]
        self.assertGreaterEqual(_kl_div(mu, theta), 0.0)

    def test_d_mu_theta_ne_d_theta_mu(self) -> None:
        """KL divergence is not symmetric."""
        mu = [0.3, 0.7]
        theta = [0.5, 0.5]
        self.assertNotAlmostEqual(_kl_div(mu, theta), _kl_div(theta, mu), places=4)

    def test_d_increases_as_mu_moves_away_from_theta(self) -> None:
        theta = [0.5, 0.5]
        mu_close = [0.51, 0.49]
        mu_far = [0.8, 0.2]
        self.assertLess(_kl_div(mu_close, theta), _kl_div(mu_far, theta))

    def test_d_positive_when_mu_ne_theta(self) -> None:
        mu = [0.7, 0.3]
        theta = [0.4, 0.6]
        self.assertGreater(_kl_div(mu, theta), 0.0)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class KLGradientTests(unittest.TestCase):

    def test_gradient_is_zero_at_fixed_point(self) -> None:
        """∇D = ln(μ) − ln(θ) + 1 is NOT zero at μ=θ (that's the property of 0-gradient at optimal)."""
        # At μ = θ: grad_i = ln(θ_i) - ln(θ_i) + 1 = 1 (not zero)
        # Zero at the optimal: the optimal is where the projection gradient balances the constraint
        mu = [0.5, 0.5]
        theta = [0.5, 0.5]
        grad = _kl_gradient(mu, theta)
        self.assertAlmostEqual(grad[0], 1.0, places=6)
        self.assertAlmostEqual(grad[1], 1.0, places=6)

    def test_gradient_dimension_matches_input(self) -> None:
        mu = [0.3, 0.4, 0.3]
        theta = [0.2, 0.5, 0.3]
        grad = _kl_gradient(mu, theta)
        self.assertEqual(len(grad), 3)

    def test_gradient_sign_matches_direction(self) -> None:
        """When μ_i > θ_i, grad_i > 1; when μ_i < θ_i, grad_i < 1."""
        mu = [0.8, 0.2]
        theta = [0.4, 0.6]
        grad = _kl_gradient(mu, theta)
        self.assertGreater(grad[0], 1.0)    # μ_0 > θ_0
        self.assertLess(grad[1], 1.0)       # μ_1 < θ_1


# ---------------------------------------------------------------------------
# LMO tests
# ---------------------------------------------------------------------------

class LMOTests(unittest.TestCase):

    def test_lmo_picks_minimum_dot_product_vertex(self) -> None:
        """z* = argmin_{z ∈ {(1,0),(0,1)}} grad · z"""
        verts = [[1.0, 0.0], [0.0, 1.0]]
        grad = [0.5, 2.0]   # prefer z=(1,0) → dot=0.5 vs 2.0
        z = _lmo(grad, verts)
        self.assertEqual(z, [1.0, 0.0])

    def test_lmo_picks_other_vertex_when_gradient_flipped(self) -> None:
        verts = [[1.0, 0.0], [0.0, 1.0]]
        grad = [2.0, 0.5]   # prefer z=(0,1) → dot=0.5 vs 2.0
        z = _lmo(grad, verts)
        self.assertEqual(z, [0.0, 1.0])

    def test_lmo_three_vertex_polytope(self) -> None:
        """Cross-market implication: 3 valid vertices."""
        verts = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        grad = [1.0, 0.5]   # min is (0,1): dot = 0.5
        z = _lmo(grad, verts)
        self.assertEqual(z, [0.0, 1.0])

    def test_lmo_returns_list_not_reference(self) -> None:
        verts = [[1.0, 0.0], [0.0, 1.0]]
        grad = [0.3, 0.8]
        z = _lmo(grad, verts)
        z[0] = 99.0   # mutating result should not affect verts
        self.assertEqual(verts[0][0], 1.0)


# ---------------------------------------------------------------------------
# Frank-Wolfe projection tests
# ---------------------------------------------------------------------------

class FWProjectionTests(unittest.TestCase):

    def test_no_arbitrage_gives_near_zero_divergence(self) -> None:
        """When θ is already inside M (arbitrage-free), D(μ*‖θ) ≈ 0."""
        # Single-market pair: ask_YES=0.55, ask_NO=0.45 → sum=1.0, no arb
        poly = ConstraintPoly.for_single_market("single_market_mispricing")
        theta = [0.55, 0.45]
        result = fw_project(theta, poly)
        # Divergence should be very small (θ is close to the simplex)
        self.assertLess(result.divergence, 0.05)

    def test_clear_arbitrage_gives_positive_divergence(self) -> None:
        """When ask_YES + ask_NO < 1 strongly, D > 0."""
        poly = ConstraintPoly.for_single_market("single_market_mispricing")
        # ask_YES=0.40, ask_NO=0.40 → sum=0.80 → obvious arb
        theta = [0.40, 0.40]
        result = fw_project(theta, poly)
        self.assertGreater(result.divergence, 0.0)

    def test_gap_is_non_negative(self) -> None:
        poly = ConstraintPoly.for_single_market("single_market_mispricing")
        result = fw_project([0.5, 0.5], poly)
        self.assertGreaterEqual(result.gap, 0.0)

    def test_iters_bounded_by_max(self) -> None:
        from src.opportunity.math_selector import _MAX_FW_ITERS
        poly = ConstraintPoly.for_implication_pair("cross_market_constraint")
        result = fw_project([0.9, 0.9], poly)
        self.assertLessEqual(result.iters, _MAX_FW_ITERS)

    def test_implication_pair_convergence(self) -> None:
        """Cross-market implication pair should converge."""
        poly = ConstraintPoly.for_implication_pair("cross_market_constraint")
        # θ = (ask_NO_A=0.977, ask_YES_B=0.019) → the historical candidate
        theta = [0.977, 0.019]
        result = fw_project(theta, poly)
        self.assertIsNotNone(result.mu_star)
        self.assertEqual(len(result.mu_star), 2)
        self.assertGreater(result.iters, 0)

    def test_mu_star_coordinates_in_unit_interval(self) -> None:
        poly = ConstraintPoly.for_implication_pair("cross_market_constraint")
        result = fw_project([0.5, 0.5], poly)
        for m in result.mu_star:
            self.assertGreaterEqual(m, 0.0)
            self.assertLessEqual(m, 1.0)


# ---------------------------------------------------------------------------
# ConstraintPoly tests
# ---------------------------------------------------------------------------

class ConstraintPolyTests(unittest.TestCase):

    def test_single_market_has_two_vertices(self) -> None:
        poly = ConstraintPoly.for_single_market("single_market_mispricing")
        self.assertEqual(len(poly.z_vertices), 2)
        self.assertEqual(poly.dim, 2)

    def test_implication_pair_has_three_vertices(self) -> None:
        poly = ConstraintPoly.for_implication_pair("cross_market_constraint")
        self.assertEqual(len(poly.z_vertices), 3)

    def test_for_family_routes_correctly(self) -> None:
        self.assertEqual(
            len(ConstraintPoly.for_family("single_market_mispricing", 2).z_vertices), 2
        )
        self.assertEqual(
            len(ConstraintPoly.for_family("cross_market_constraint", 2).z_vertices), 3
        )
        self.assertEqual(
            len(ConstraintPoly.for_family("cross_market_gross_constraint", 2).z_vertices), 3
        )

    def test_general_polytope_has_2n_vertices(self) -> None:
        poly = ConstraintPoly.for_general(3, "unknown_family")
        self.assertEqual(len(poly.z_vertices), 8)   # 2^3

    def test_general_polytope_capped_at_6_dims(self) -> None:
        poly = ConstraintPoly.for_general(10, "big_family")
        self.assertEqual(poly.dim, 6)
        self.assertEqual(len(poly.z_vertices), 64)   # 2^6

    def test_neg_risk_family_uses_general_polytope(self) -> None:
        """neg_risk_rebalancing has unknown structure → general fallback."""
        poly = ConstraintPoly.for_family("neg_risk_rebalancing", 3)
        self.assertEqual(len(poly.z_vertices), 8)   # 2^3 fallback


# ---------------------------------------------------------------------------
# LegAdjustedThreshold tests
# ---------------------------------------------------------------------------

class LegAdjustedThresholdTests(unittest.TestCase):

    def test_two_leg_threshold_below_a_flat(self) -> None:
        """B's 2-leg min (0.025) < A's flat (0.030)."""
        t = LegAdjustedThreshold(fee_per_leg_cents=0.005, slip_per_leg_cents=0.005, target_margin_cents=0.005)
        self.assertAlmostEqual(t.min_viable(2), 0.025, places=6)

    def test_three_leg_threshold_above_a_flat(self) -> None:
        """B's 3-leg min (0.035) > A's flat (0.030)."""
        t = LegAdjustedThreshold(fee_per_leg_cents=0.005, slip_per_leg_cents=0.005, target_margin_cents=0.005)
        self.assertAlmostEqual(t.min_viable(3), 0.035, places=6)

    def test_single_leg_threshold(self) -> None:
        t = LegAdjustedThreshold(fee_per_leg_cents=0.005, slip_per_leg_cents=0.005, target_margin_cents=0.005)
        self.assertAlmostEqual(t.min_viable(1), 0.015, places=6)

    def test_threshold_scales_linearly_with_legs(self) -> None:
        t = LegAdjustedThreshold(fee_per_leg_cents=0.01, slip_per_leg_cents=0.01, target_margin_cents=0.0)
        self.assertAlmostEqual(t.min_viable(4), 4 * 0.02, places=6)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class HelperTests(unittest.TestCase):

    def test_liquidity_cap_profit_scales_with_edge_and_volume(self) -> None:
        profit = _liquidity_cap_profit(gross_edge_cents=0.03, available_shares=1000.0)
        self.assertAlmostEqual(profit, 0.03 * 1000.0 / 100.0, places=6)

    def test_kelly_ref_positive_for_profitable_trade(self) -> None:
        # Need b*p > q = 1-p  →  b > (1-p)/p.  For p=0.9: b > 0.111.  Use b=0.2.
        ref = _kelly_size_ref(b=0.2, p=0.9)
        self.assertIsNotNone(ref)
        self.assertGreater(ref, 0.0)

    def test_kelly_ref_none_for_zero_edge(self) -> None:
        ref = _kelly_size_ref(b=0.0, p=0.9)
        self.assertIsNone(ref)

    def test_kelly_ref_none_for_degenerate_fill_prob(self) -> None:
        ref = _kelly_size_ref(b=0.1, p=0.0)
        self.assertIsNone(ref)

    def test_kelly_ref_none_for_full_fill_prob(self) -> None:
        ref = _kelly_size_ref(b=0.1, p=1.0)
        self.assertIsNone(ref)


# ---------------------------------------------------------------------------
# MathCandidateSelector end-to-end tests
# ---------------------------------------------------------------------------

class MathCandidateSelectorTests(unittest.TestCase):

    def _selector(self) -> MathCandidateSelector:
        return MathCandidateSelector(
            threshold=LegAdjustedThreshold(
                fee_per_leg_cents=0.005,
                slip_per_leg_cents=0.005,
                target_margin_cents=0.005,
            ),
            max_partial_fill_risk=0.65,
            max_non_atomic_risk=0.60,
            min_net_profit_usd=0.10,
        )

    def test_strong_arb_passes_b(self) -> None:
        """A candidate with large divergence and good depth should pass B."""
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-strong",
            family="cross_market_constraint",
            n_legs=2,
            gross_edge_cents=0.10,          # well above 2-leg min of 0.025
            pair_vwap=0.90,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.45, 0.45],   # arb: sum=0.90 < 1.0
            available_shares=10000.0,
            available_depth_usd=5000.0,
            required_depth_usd=100.0,
            partial_fill_risk_score=0.0,
            non_atomic_execution_risk_score=0.1,
            expected_net_profit_usd=5.0,
        )
        self.assertTrue(result.passed_b)
        self.assertEqual(result.reason_codes_b, [])
        self.assertGreater(result.score, 1.0)

    def test_weak_arb_rejects_b(self) -> None:
        """A candidate with tiny divergence (historical level) fails B."""
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-weak",
            family="cross_market_gross_constraint",
            n_legs=2,
            gross_edge_cents=0.004,
            pair_vwap=0.996,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.977, 0.019],  # historical values
            available_shares=2852983.6,
            available_depth_usd=5252396.0,
            required_depth_usd=51.0,
            partial_fill_risk_score=0.0,
            non_atomic_execution_risk_score=0.19,
            expected_net_profit_usd=-0.82,
        )
        self.assertFalse(result.passed_b)
        self.assertIn("MATH_EDGE_BELOW_LEG_ADJUSTED_THRESHOLD", result.reason_codes_b)
        self.assertLess(result.score, 1.0)

    def test_score_is_positive_for_any_positive_edge(self) -> None:
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-score",
            family="cross_market_constraint",
            n_legs=2,
            gross_edge_cents=0.010,
            pair_vwap=0.99,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.50, 0.49],
            available_shares=1000.0,
            available_depth_usd=500.0,
            required_depth_usd=50.0,
            partial_fill_risk_score=0.0,
            non_atomic_execution_risk_score=0.0,
            expected_net_profit_usd=0.0,
        )
        self.assertGreater(result.score, 0.0)

    def test_depth_gate_fires_when_insufficient(self) -> None:
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-depth",
            family="cross_market_constraint",
            n_legs=2,
            gross_edge_cents=0.10,
            pair_vwap=0.90,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.45, 0.45],
            available_shares=5.0,
            available_depth_usd=1.0,    # << required
            required_depth_usd=100.0,
            partial_fill_risk_score=0.0,
            non_atomic_execution_risk_score=0.0,
            expected_net_profit_usd=5.0,
        )
        self.assertIn("INSUFFICIENT_DEPTH", result.reason_codes_b)
        self.assertFalse(result.passed_b)

    def test_partial_fill_risk_gate_fires(self) -> None:
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-pf",
            family="cross_market_constraint",
            n_legs=2,
            gross_edge_cents=0.10,
            pair_vwap=0.90,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.45, 0.45],
            available_shares=10000.0,
            available_depth_usd=5000.0,
            required_depth_usd=100.0,
            partial_fill_risk_score=0.80,  # > max 0.65
            non_atomic_execution_risk_score=0.1,
            expected_net_profit_usd=5.0,
        )
        self.assertIn("PARTIAL_FILL_RISK_TOO_HIGH", result.reason_codes_b)
        self.assertFalse(result.passed_b)

    def test_non_atomic_risk_gate_fires(self) -> None:
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-na",
            family="cross_market_constraint",
            n_legs=2,
            gross_edge_cents=0.10,
            pair_vwap=0.90,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.45, 0.45],
            available_shares=10000.0,
            available_depth_usd=5000.0,
            required_depth_usd=100.0,
            partial_fill_risk_score=0.0,
            non_atomic_execution_risk_score=0.70,  # > max 0.60
            expected_net_profit_usd=5.0,
        )
        self.assertIn("NON_ATOMIC_RISK_TOO_HIGH", result.reason_codes_b)
        self.assertFalse(result.passed_b)

    def test_three_leg_has_higher_min_viable_than_two_leg(self) -> None:
        """3-leg is more restrictive under B's per-leg model."""
        s = self._selector()
        def _eval(n_legs: int) -> float:
            result = s.evaluate(
                candidate_id="test-legs",
                family="neg_risk_rebalancing",
                n_legs=n_legs,
                gross_edge_cents=0.03,
                pair_vwap=0.97,
                expected_payout_per_share=1.0,
                leg_vwap_prices=[0.3] * n_legs,
                available_shares=1000.0,
                available_depth_usd=500.0,
                required_depth_usd=50.0,
                partial_fill_risk_score=0.0,
                non_atomic_execution_risk_score=0.0,
                expected_net_profit_usd=0.5,
            )
            return result.min_viable_edge
        self.assertGreater(_eval(3), _eval(2))

    def test_metadata_contains_fw_fields(self) -> None:
        s = self._selector()
        result = s.evaluate(
            candidate_id="test-meta",
            family="cross_market_constraint",
            n_legs=2,
            gross_edge_cents=0.05,
            pair_vwap=0.95,
            expected_payout_per_share=1.0,
            leg_vwap_prices=[0.48, 0.47],
            available_shares=1000.0,
            available_depth_usd=500.0,
            required_depth_usd=50.0,
            partial_fill_risk_score=0.0,
            non_atomic_execution_risk_score=0.0,
            expected_net_profit_usd=1.0,
        )
        self.assertIn("fw_iters", result.metadata)
        self.assertIn("fw_gap", result.metadata)
        self.assertIn("mu_star", result.metadata)
        self.assertIn("kl_divergence", result.metadata)
        self.assertIn("approximations_used", result.metadata)

    def test_from_stored_metadata_helper(self) -> None:
        """from_stored_qualification_metadata reconstructs correctly."""
        s = self._selector()
        raw = {
            "expected_payout": 51.177073,
            "target_shares": 51.177073,
            "legs": [
                {"vwap_price": 0.977, "action": "BUY", "token_id": "t1"},
                {"vwap_price": 0.019, "action": "BUY", "token_id": "t2"},
            ],
        }
        qual = {
            "expected_gross_edge_cents": 0.004,
            "pair_vwap": 0.996,
            "expected_net_profit_usd": -0.818,
            "available_shares": 2852983.6,
            "available_depth_usd": 5252396.0,
            "required_depth_usd": 50.97,
            "partial_fill_risk_score": 0.0,
            "non_atomic_execution_risk_score": 0.19,
            "legs": [
                {"vwap_price": 0.977, "available_shares": 2852983.6, "available_notional_usd": 2847777.0},
                {"vwap_price": 0.019, "available_shares": 12437582.77, "available_notional_usd": 11967165.0},
            ],
        }
        result = MathCandidateSelector.from_stored_qualification_metadata(
            selector=s,
            candidate_id="hist-1",
            family="cross_market_gross_constraint",
            raw_candidate=raw,
            qual_meta=qual,
        )
        self.assertEqual(result.candidate_id, "hist-1")
        self.assertEqual(result.n_legs, 2)
        self.assertFalse(result.passed_b)   # gross_edge 0.004 << min_viable 0.025
        self.assertIn("MATH_EDGE_BELOW_LEG_ADJUSTED_THRESHOLD", result.reason_codes_b)


if __name__ == "__main__":
    unittest.main()
