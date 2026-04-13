"""
Tests for src.scanner.maker_scan_quote_planner.

Coverage:
  1. Happy path: eligible plan returned for a reward-eligible market
  2. Both spread legs within max_spread_cents when conditions allow
  3. Size eligibility: quote_size >= rewards_min_size
  4. Ineligible: invalid book (bid >= ask)
  5. Ineligible: zero reward rate
  6. Ineligible: reward rate below minimum threshold
  7. Ineligible: missing rewards_min_size
  8. Ineligible: missing rewards_max_spread
  9. Inventory skew: long inventory shifts quote bid/ask down
  10. Inventory skew: short inventory shifts quote bid/ask up
  11. Belief variance scaling: higher var → wider quote spread
  12. horizon_left = 0 → minimal spread (no time value)
  13. horizon_left = 1 → full spread
  14. quote_size is at least 20 even if rewards_min_size < 20
  15. quote_size equals rewards_min_size when it exceeds 20
  16. Fallback belief_var: PRIOR_BELIEF_VAR used when None passed
  17. Explicit belief_var overrides prior
  18. clob_rewards fallback: rate summed from clob_rewards when reward_daily_rate absent
  19. market_slug from 'market_slug' key
  20. market_slug from 'slug' key fallback
  21. Empty dict → ineligible, no exception
  22. None values in dict → ineligible, no exception
  23. mid_p is average of best_bid and best_ask
  24. Return type is always MakerQuotePlan
  25. daily_rate_usdc reflected in plan
  26. max_spread_cents reflected in plan
  27. belief_var_used reflects what was passed
  28. horizon_left reflected in plan
  29. inventory reflected in plan
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scanner.maker_scan_quote_planner import (
    MakerQuotePlan,
    MAKER_K,
    MAKER_RISK_AVERSION,
    PRIOR_BELIEF_VAR,
    plan_quote,
)


def _market(
    *,
    best_bid: float = 0.45,
    best_ask: float = 0.55,
    rewards_min_size: float = 50.0,
    rewards_max_spread: float = 3.5,
    reward_daily_rate: float = 20.0,
    market_slug: str = "test-market",
) -> dict:
    return {
        "market_slug":        market_slug,
        "best_bid":           best_bid,
        "best_ask":           best_ask,
        "rewards_min_size":   rewards_min_size,
        "rewards_max_spread": rewards_max_spread,
        "reward_daily_rate":  reward_daily_rate,
    }


class HappyPathTests(unittest.TestCase):
    def setUp(self):
        self.m = _market()
        self.plan = plan_quote(self.m)

    def test_returns_maker_quote_plan(self):
        self.assertIsInstance(self.plan, MakerQuotePlan)

    def test_eligible_for_standard_market(self):
        """
        Calibrated defaults (γ=0.1, k=2.0) + 3.5¢ max_spread + zero inventory
        + PRIOR_BELIEF_VAR → spread should fit within reward bounds.
        """
        self.assertTrue(self.plan.eligible, msg=self.plan.feasibility_reason)

    def test_quote_bid_below_mid(self):
        self.assertLess(self.plan.quote_bid, self.plan.mid_p)

    def test_quote_ask_above_mid(self):
        self.assertGreater(self.plan.quote_ask, self.plan.mid_p)

    def test_mid_p_is_average(self):
        m = _market(best_bid=0.40, best_ask=0.60)
        p = plan_quote(m)
        self.assertAlmostEqual(p.mid_p, 0.50, places=6)

    def test_spread_cents_positive(self):
        self.assertGreater(self.plan.bid_spread_cents, 0.0)
        self.assertGreater(self.plan.ask_spread_cents, 0.0)

    def test_spread_within_max(self):
        # Standard market (3.5¢) with calibrated defaults → eligible → spread within max
        self.assertTrue(self.plan.eligible, msg=self.plan.feasibility_reason)
        self.assertLessEqual(self.plan.bid_spread_cents, self.plan.max_spread_cents)
        self.assertLessEqual(self.plan.ask_spread_cents, self.plan.max_spread_cents)

    def test_quote_size_at_least_rewards_min_size(self):
        self.assertGreaterEqual(self.plan.quote_size, self.m["rewards_min_size"])

    def test_quote_size_floor_20(self):
        m = _market(rewards_min_size=5.0)
        p = plan_quote(m)
        self.assertGreaterEqual(p.quote_size, 20.0)

    def test_quote_size_uses_rewards_min_when_larger(self):
        m = _market(rewards_min_size=200.0)
        p = plan_quote(m)
        self.assertAlmostEqual(p.quote_size, 200.0)

    def test_daily_rate_reflected(self):
        self.assertAlmostEqual(self.plan.daily_rate_usdc, 20.0)

    def test_max_spread_reflected(self):
        self.assertAlmostEqual(self.plan.max_spread_cents, 3.5)

    def test_market_slug_from_market_slug_key(self):
        m = _market(market_slug="my-slug")
        p = plan_quote(m)
        self.assertEqual(p.market_slug, "my-slug")

    def test_market_slug_from_slug_key_fallback(self):
        m = _market()
        del m["market_slug"]
        m["slug"] = "fallback-slug"
        p = plan_quote(m)
        self.assertEqual(p.market_slug, "fallback-slug")


class IneligibleTests(unittest.TestCase):
    def test_invalid_book_bid_gte_ask(self):
        m = _market(best_bid=0.55, best_ask=0.45)
        p = plan_quote(m)
        self.assertFalse(p.eligible)
        self.assertIn("invalid_book", p.feasibility_reason)

    def test_zero_reward_rate(self):
        m = _market(reward_daily_rate=0.0)
        p = plan_quote(m)
        self.assertFalse(p.eligible)

    def test_rate_below_threshold(self):
        # MIN_RATE is 5.0; 4.9 is below it
        m = _market(reward_daily_rate=4.9)
        p = plan_quote(m)
        self.assertFalse(p.eligible)

    def test_zero_rewards_min_size(self):
        m = _market(rewards_min_size=0.0)
        p = plan_quote(m)
        self.assertFalse(p.eligible)

    def test_zero_rewards_max_spread(self):
        m = _market(rewards_max_spread=0.0)
        p = plan_quote(m)
        self.assertFalse(p.eligible)

    def test_empty_dict_ineligible_no_exception(self):
        p = plan_quote({})
        self.assertIsInstance(p, MakerQuotePlan)
        self.assertFalse(p.eligible)

    def test_none_values_ineligible_no_exception(self):
        m = {
            "market_slug": None,
            "best_bid": None,
            "best_ask": None,
            "rewards_min_size": None,
            "rewards_max_spread": None,
            "reward_daily_rate": None,
        }
        p = plan_quote(m)
        self.assertIsInstance(p, MakerQuotePlan)
        self.assertFalse(p.eligible)


class InventorySkewTests(unittest.TestCase):
    def test_long_inventory_lowers_bid_and_ask(self):
        base   = plan_quote(_market(), inventory=0.0)
        long   = plan_quote(_market(), inventory=5.0)
        self.assertLess(long.quote_bid, base.quote_bid)
        self.assertLess(long.quote_ask, base.quote_ask)

    def test_short_inventory_raises_bid_and_ask(self):
        base  = plan_quote(_market(), inventory=0.0)
        short = plan_quote(_market(), inventory=-5.0)
        self.assertGreater(short.quote_bid, base.quote_bid)
        self.assertGreater(short.quote_ask, base.quote_ask)

    def test_inventory_reflected_in_plan(self):
        p = plan_quote(_market(), inventory=3.0)
        self.assertAlmostEqual(p.inventory, 3.0)


class BeliefVarTests(unittest.TestCase):
    def test_higher_var_wider_spread(self):
        low  = plan_quote(_market(), belief_var=0.01)
        high = plan_quote(_market(), belief_var=0.10)
        low_total  = low.bid_spread_cents  + low.ask_spread_cents
        high_total = high.bid_spread_cents + high.ask_spread_cents
        self.assertGreater(high_total, low_total)

    def test_prior_var_used_when_none(self):
        p = plan_quote(_market(), belief_var=None)
        self.assertAlmostEqual(p.belief_var_used, PRIOR_BELIEF_VAR)

    def test_explicit_var_overrides_prior(self):
        p = plan_quote(_market(), belief_var=0.02)
        self.assertAlmostEqual(p.belief_var_used, 0.02)


class HorizonTests(unittest.TestCase):
    def test_zero_horizon_narrower_than_full(self):
        zero = plan_quote(_market(), horizon_left=0.0)
        full = plan_quote(_market(), horizon_left=1.0)
        # At horizon=0, adverse_selection_term=0, only inventory_penalty_term remains
        zero_total = zero.bid_spread_cents + zero.ask_spread_cents
        full_total = full.bid_spread_cents + full.ask_spread_cents
        self.assertLessEqual(zero_total, full_total)

    def test_horizon_reflected_in_plan(self):
        p = plan_quote(_market(), horizon_left=0.75)
        self.assertAlmostEqual(p.horizon_left, 0.75)


class ClobRewardsFallbackTests(unittest.TestCase):
    def test_rate_from_clob_rewards_when_reward_daily_rate_absent(self):
        m = {
            "market_slug":        "clob-fallback",
            "best_bid":           0.45,
            "best_ask":           0.55,
            "rewards_min_size":   50.0,
            "rewards_max_spread": 3.5,
            # no reward_daily_rate key — rate should be summed from clob_rewards
            "clob_rewards": [{"rewardsDailyRate": 10.0}, {"rewardsDailyRate": 8.0}],
        }
        p = plan_quote(m)
        # rate = 18.0 >= MIN_RATE(5.0) → has_rewards=True
        self.assertAlmostEqual(p.daily_rate_usdc, 18.0)
        # Calibrated defaults produce ~1.3¢ spread, within 3.5¢ → eligible
        self.assertTrue(p.eligible, msg=p.feasibility_reason)


class TightSpreadEligibilityTests(unittest.TestCase):
    def test_ineligible_when_max_spread_very_tight(self):
        """
        Very tight max_spread (0.1¢): calibrated defaults produce ~1.3¢ spread,
        which exceeds 0.1¢ → ineligible.
        """
        m = _market(rewards_max_spread=0.1, reward_daily_rate=20.0)
        p = plan_quote(m)
        # Calibrated defaults: spread ≈ 1.3¢ → exceeds 0.1¢ max → ineligible
        self.assertFalse(p.eligible)
        self.assertIn("ineligible", p.feasibility_reason)

    def test_eligible_with_narrow_belief_var(self):
        """With tiny belief_var, the logit engine produces a very tight quote."""
        m = _market(rewards_max_spread=3.5, reward_daily_rate=20.0)
        p = plan_quote(m, belief_var=1e-5)
        # Calibrated defaults: inventory_penalty_term ≈ 0.049 in x-space → ~0.6¢ at mid
        # Well within 3.5¢ threshold.
        self.assertTrue(p.eligible, msg=p.feasibility_reason)


if __name__ == "__main__":
    unittest.main()
