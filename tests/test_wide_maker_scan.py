"""Tests for the wide maker-MM scan EV model and stress testing."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_wide_maker_scan import (
    compute_maker_mm_ev,
    q_score_per_side,
    stress_test_plan,
    queue_realism_test,
)


class QScoreTests(unittest.TestCase):
    def test_zero_distance_max_score(self):
        score = q_score_per_side(3.5, 0.0, 20.0)
        self.assertAlmostEqual(score, 20.0)

    def test_at_max_spread_zero_score(self):
        score = q_score_per_side(3.5, 3.5, 20.0)
        self.assertAlmostEqual(score, 0.0)

    def test_half_distance_quadratic(self):
        score = q_score_per_side(3.5, 1.75, 20.0)
        expected = ((3.5 - 1.75) / 3.5) ** 2 * 20.0
        self.assertAlmostEqual(score, expected)

    def test_beyond_max_spread_zero(self):
        score = q_score_per_side(3.5, 4.0, 20.0)
        self.assertAlmostEqual(score, 0.0)

    def test_zero_max_spread(self):
        score = q_score_per_side(0.0, 1.0, 20.0)
        self.assertAlmostEqual(score, 0.0)


class EVModelTests(unittest.TestCase):
    def test_basic_positive_ev(self):
        ev = compute_maker_mm_ev(
            best_bid=0.40,
            best_ask=0.45,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=10.0,
            volume_num=50000.0,
        )
        self.assertGreater(ev["total_ev"], 0)
        self.assertGreater(ev["reward_ev"], 0)
        self.assertGreater(ev["spread_capture_ev"], 0)

    def test_tight_spread_higher_q_score(self):
        tight = compute_maker_mm_ev(
            best_bid=0.49, best_ask=0.51,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=10.0,
        )
        wide = compute_maker_mm_ev(
            best_bid=0.40, best_ask=0.50,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=10.0,
        )
        self.assertGreaterEqual(tight["our_q_score"], wide["our_q_score"])

    def test_zero_reward_still_has_spread_capture(self):
        ev = compute_maker_mm_ev(
            best_bid=0.40, best_ask=0.45,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=0.0,
        )
        self.assertAlmostEqual(ev["reward_ev"], 0.0)
        self.assertGreater(ev["spread_capture_ev"], 0)

    def test_zero_spread_book_low_spread_capture(self):
        ev = compute_maker_mm_ev(
            best_bid=0.50, best_ask=0.50,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=5.0,
        )
        # Spread is 0 but min half-spread forces 0.5 cent each side
        # Spread capture should be very small relative to reward
        self.assertLess(ev["spread_capture_ev"], ev["reward_ev"])

    def test_competition_factor_scales_with_reward_rate(self):
        low = compute_maker_mm_ev(
            best_bid=0.40, best_ask=0.45,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=1.0,
        )
        high = compute_maker_mm_ev(
            best_bid=0.40, best_ask=0.45,
            rewards_min_size=20.0,
            rewards_max_spread_cents=3.5,
            reward_daily_rate=200.0,
        )
        self.assertGreater(high["competition_factor"], low["competition_factor"])


class StressTestTests(unittest.TestCase):
    def _make_plan(self, total_ev=5.0, reward_ev=3.0, spread_ev=1.0):
        return {
            "ev": {
                "total_ev": total_ev,
                "reward_ev": reward_ev,
                "spread_capture_ev": spread_ev,
                "adverse_cost": 0.3,
                "inventory_cost": 0.2,
                "cancel_cost": 0.001,
                "fee_cost": 0.0,
            }
        }

    def test_robust_label(self):
        plan = self._make_plan(total_ev=5.0, reward_ev=4.0, spread_ev=1.0)
        result = stress_test_plan(plan)
        self.assertEqual(result["label"], "ROBUST_PAPER_MM")

    def test_negative_ev_label(self):
        plan = self._make_plan(total_ev=-1.0, reward_ev=0.1, spread_ev=0.01)
        result = stress_test_plan(plan)
        self.assertEqual(result["label"], "NEGATIVE_EV")

    def test_conservative_combined_exists(self):
        plan = self._make_plan()
        result = stress_test_plan(plan)
        self.assertIn("conservative_combined", result["views"])

    def test_all_views_present(self):
        plan = self._make_plan()
        result = stress_test_plan(plan)
        expected = {"spread_haircut_25", "spread_haircut_50", "spread_haircut_75",
                    "one_sided_fill", "adverse_move", "high_competition", "conservative_combined"}
        self.assertEqual(set(result["views"].keys()), expected)


class QueueRealismTests(unittest.TestCase):
    def test_queue_resilient_label(self):
        plan = {
            "ev": {
                "total_ev": 5.0,
                "reward_ev": 4.0,
                "spread_capture_ev": 1.0,
                "adverse_cost": 0.2,
                "inventory_cost": 0.1,
                "cancel_cost": 0.001,
            }
        }
        result = queue_realism_test(plan)
        self.assertEqual(result["label"], "QUEUE_RESILIENT")

    def test_all_views_present(self):
        plan = {
            "ev": {
                "total_ev": 5.0,
                "reward_ev": 4.0,
                "spread_capture_ev": 1.0,
                "adverse_cost": 0.2,
                "inventory_cost": 0.1,
                "cancel_cost": 0.001,
            }
        }
        result = queue_realism_test(plan)
        expected = {"delayed_fill", "quote_replaced", "partial_fill",
                    "reduced_reward_dwell", "conservative_combined"}
        self.assertEqual(set(result["views"].keys()), expected)


if __name__ == "__main__":
    unittest.main()
