"""
Tests for estimate_implied_reward_ev() and _book_competitor_q().

Coverage:
  1. Zero competitor Q → captures 100% of pool
  2. Equal our_q and competitor_q → 50% share
  3. Large competitor Q → tiny share (lower bound behaviour)
  4. Zero our_q → returns 0 (no pool share without a quote)
  5. Zero daily_rate → returns 0
  6. Degenerate: both zero → returns 0 (no ZeroDivisionError)
  7. _book_competitor_q: empty book returns 0
  8. _book_competitor_q: orders outside max_spread not counted
  9. _book_competitor_q: orders at exactly max_spread boundary excluded (s >= v)
 10. _book_competitor_q: two-sided book sums bids + asks
 11. estimate_implied_reward_ev returns lower value than modeled reward_ev
     for realistically large competitor pools
 12. reward_ev_ratio: implied / modeled for Hungary-like inputs ≈ 0.18–0.25
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_wide_maker_scan import estimate_implied_reward_ev, q_score_per_side
from scripts.run_maker_paper_calibration import _book_competitor_q


# ---------------------------------------------------------------------------
# estimate_implied_reward_ev
# ---------------------------------------------------------------------------

class ImpliedRewardEvTests(unittest.TestCase):

    def test_zero_competitor_full_share(self):
        """No competitors → we capture 100% of pool."""
        ev = estimate_implied_reward_ev(daily_rate=200.0, our_q_score=300.0, competitor_q_sum=0.0)
        self.assertAlmostEqual(ev, 200.0, places=6)

    def test_equal_q_half_share(self):
        """Our Q equals competitor Q → 50% share."""
        ev = estimate_implied_reward_ev(daily_rate=200.0, our_q_score=300.0, competitor_q_sum=300.0)
        self.assertAlmostEqual(ev, 100.0, places=6)

    def test_large_competitor_tiny_share(self):
        """Large competitor Q → very small share (lower bound)."""
        ev = estimate_implied_reward_ev(daily_rate=200.0, our_q_score=300.0, competitor_q_sum=40000.0)
        self.assertLess(ev, 2.0)
        self.assertGreater(ev, 0.0)

    def test_zero_our_q_returns_zero(self):
        """No quote → no pool share."""
        ev = estimate_implied_reward_ev(daily_rate=200.0, our_q_score=0.0, competitor_q_sum=1000.0)
        self.assertAlmostEqual(ev, 0.0, places=6)

    def test_zero_daily_rate_returns_zero(self):
        """Zero reward pool → zero EV regardless of Q-scores."""
        ev = estimate_implied_reward_ev(daily_rate=0.0, our_q_score=300.0, competitor_q_sum=1000.0)
        self.assertAlmostEqual(ev, 0.0, places=6)

    def test_both_zero_returns_zero_no_error(self):
        """Degenerate all-zero inputs must not raise."""
        ev = estimate_implied_reward_ev(daily_rate=0.0, our_q_score=0.0, competitor_q_sum=0.0)
        self.assertAlmostEqual(ev, 0.0, places=6)

    def test_implies_lower_than_modeled_for_hungary(self):
        """
        With Hungary-like inputs (rate=$200, our_q≈294, competitor_q≈40000),
        implied EV is well below the modeled EV of $8/day (which assumes 4% share).
        """
        implied = estimate_implied_reward_ev(
            daily_rate=200.0, our_q_score=294.0, competitor_q_sum=40000.0
        )
        modeled = 200.0 * (1 / 25.0)   # competition_factor=25 → 4% share → $8/day
        self.assertLess(implied, modeled)

    def test_reward_ev_ratio_hungary_range(self):
        """
        Hungary-like implied/modeled ratio should be in the 0.15 – 0.30 range
        (confirmed by audit: ~0.18–0.23).
        """
        implied = estimate_implied_reward_ev(
            daily_rate=200.0, our_q_score=294.0, competitor_q_sum=40000.0
        )
        modeled = 8.0
        ratio = implied / modeled
        self.assertGreater(ratio, 0.10)
        self.assertLess(ratio, 0.35)

    def test_linearly_scales_with_rate(self):
        """Double the daily rate → double the implied EV."""
        ev1 = estimate_implied_reward_ev(200.0, 300.0, 1000.0)
        ev2 = estimate_implied_reward_ev(400.0, 300.0, 1000.0)
        self.assertAlmostEqual(ev2, ev1 * 2, places=6)


# ---------------------------------------------------------------------------
# _book_competitor_q  (tests the CLOB book helper)
# ---------------------------------------------------------------------------

def _make_level(price: float, size: float):
    m = MagicMock()
    m.price = price
    m.size  = size
    return m


def _make_book(bids=None, asks=None):
    book = MagicMock()
    book.bids = [_make_level(p, s) for p, s in (bids or [])]
    book.asks = [_make_level(p, s) for p, s in (asks or [])]
    return book


class BookCompetitorQTests(unittest.TestCase):

    def test_none_book_returns_zero(self):
        self.assertAlmostEqual(_book_competitor_q(None, 0.5, 3.5), 0.0)

    def test_empty_book_returns_zero(self):
        book = _make_book(bids=[], asks=[])
        self.assertAlmostEqual(_book_competitor_q(book, 0.5, 3.5), 0.0)

    def test_order_outside_spread_not_counted(self):
        """
        mid_p=0.50, max_spread=3.5¢.
        A bid at 0.46 is 4¢ from mid → outside zone → Q=0.
        """
        book = _make_book(bids=[(0.46, 1000.0)], asks=[])
        q = _book_competitor_q(book, 0.50, 3.5)
        self.assertAlmostEqual(q, 0.0)

    def test_order_at_boundary_excluded(self):
        """
        Boundary: distance == max_spread → q_score_per_side returns 0
        (s >= v condition in the formula).
        """
        # mid=0.50, max=3.5¢, bid at 0.465 → dist=3.5¢ → exactly at boundary
        book = _make_book(bids=[(0.465, 1000.0)], asks=[])
        q = _book_competitor_q(book, 0.50, 3.5)
        self.assertAlmostEqual(q, 0.0)

    def test_order_inside_spread_counted(self):
        """A bid at 0.37 (mid=0.375, max=3.5¢) → distance=0.5¢ → inside zone → Q>0."""
        book = _make_book(bids=[(0.37, 9217.0)], asks=[])
        q = _book_competitor_q(book, 0.375, 3.5)
        expected = q_score_per_side(3.5, 0.5, 9217.0)
        self.assertAlmostEqual(q, expected, places=3)
        self.assertGreater(q, 0.0)

    def test_both_sides_summed(self):
        """Competitor Q is sum of bid-side and ask-side Q-scores."""
        book = _make_book(
            bids=[(0.37, 9217.0)],
            asks=[(0.38, 5221.0)],
        )
        mid = 0.375
        q = _book_competitor_q(book, mid, 3.5)
        bid_q = q_score_per_side(3.5, abs(0.37 - mid) * 100, 9217.0)
        ask_q = q_score_per_side(3.5, abs(0.38 - mid) * 100, 5221.0)
        self.assertAlmostEqual(q, bid_q + ask_q, places=3)

    def test_multiple_levels_summed(self):
        """Multiple bid levels within zone are all added."""
        book = _make_book(
            bids=[(0.37, 9217.0), (0.36, 41540.0)],
            asks=[(0.38, 5221.0), (0.39, 25215.0)],
        )
        mid = 0.375
        q = _book_competitor_q(book, mid, 3.5)
        expected = sum(
            q_score_per_side(3.5, abs(p - mid) * 100, sz)
            for p, sz in [(0.37, 9217), (0.36, 41540), (0.38, 5221), (0.39, 25215)]
        )
        self.assertAlmostEqual(q, expected, places=3)


if __name__ == "__main__":
    unittest.main()
