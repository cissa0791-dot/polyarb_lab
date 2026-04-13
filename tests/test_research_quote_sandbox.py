"""
tests/test_research_quote_sandbox.py
Deterministic tests for src/research/quote_sandbox.py

Coverage:
  - as_quotes: output structure (bid < ask, bid > 0, ask < 1)
  - as_quotes: zero inventory → reservation_price ≈ mid
  - as_quotes: positive inventory skews reservation price downward
  - as_quotes: higher sigma_b → wider spread
  - as_quotes: higher gamma → wider spread
  - as_quotes: longer T → wider spread
  - spread_vs_sigma_b: monotone relationship
  - spread_vs_inventory: reservation_price moves with q
  - position_size_hint: edge / (γ · σ²) formula, zero edge, max cap
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.research.quote_sandbox import (
    as_quotes,
    spread_vs_sigma_b,
    spread_vs_inventory,
    position_size_hint,
    ASQuoteResult,
    AS_GAMMA_DEFAULT,
    AS_K_DEFAULT,
    AS_T_DEFAULT,
)


# ---------------------------------------------------------------------------
# as_quotes
# ---------------------------------------------------------------------------

class TestASQuotes:
    def _quote(self, mid=0.5, sigma_b=0.10, q=0.0, gamma=0.1, T=1.0, k=1.5):
        return as_quotes(mid=mid, sigma_b=sigma_b, q=q, gamma=gamma, T=T, k=k)

    def test_returns_ASQuoteResult(self):
        result = self._quote()
        assert isinstance(result, ASQuoteResult)

    def test_bid_below_ask(self):
        result = self._quote()
        assert result.bid < result.ask

    def test_bid_positive(self):
        result = self._quote()
        assert result.bid > 0.0

    def test_ask_below_one(self):
        result = self._quote()
        assert result.ask < 1.0

    def test_zero_inventory_reservation_close_to_mid(self):
        """q=0 → no inventory skew → reservation_price ≈ mid (small spread effect)."""
        result = self._quote(mid=0.5, q=0.0)
        # Reservation price should equal mid when q=0 (no skew)
        assert abs(result.reservation_price - 0.5) < 0.05

    def test_positive_inventory_skews_down(self):
        """Long YES (q>0) → market maker wants to sell YES → lower ask."""
        r_flat = self._quote(mid=0.5, q=0.0)
        r_long = self._quote(mid=0.5, q=50.0)
        assert r_long.reservation_price < r_flat.reservation_price

    def test_negative_inventory_skews_up(self):
        r_flat  = self._quote(mid=0.5, q=0.0)
        r_short = self._quote(mid=0.5, q=-50.0)
        assert r_short.reservation_price > r_flat.reservation_price

    def test_higher_sigma_b_wider_spread(self):
        r_low  = self._quote(sigma_b=0.05)
        r_high = self._quote(sigma_b=0.50)
        assert r_high.spread_prob > r_low.spread_prob

    def test_higher_gamma_wider_spread(self):
        # A-S spread in logit space: δ = (γ·σ²·T)/2 + (1/γ)·ln(1 + γ/k)
        # The (1/γ)·ln(...) term dominates at low γ, making spread non-monotone in γ.
        # The first term (γ·σ²·T)/2 dominates at very high σ_b.
        # Use high sigma_b to ensure first term dominates, making spread larger for high γ.
        r_low  = self._quote(gamma=0.01, sigma_b=2.0)
        r_high = self._quote(gamma=1.00, sigma_b=2.0)
        assert r_high.spread_logit > r_low.spread_logit

    def test_longer_T_wider_spread(self):
        r_short = self._quote(T=0.1)
        r_long  = self._quote(T=10.0)
        assert r_long.spread_prob > r_short.spread_prob

    def test_extreme_p_does_not_crash(self):
        for mid in [0.01, 0.99]:
            result = as_quotes(mid=mid, sigma_b=0.10)
            assert 0.0 < result.bid < result.ask < 1.0

    def test_spread_logit_positive(self):
        result = self._quote()
        assert result.spread_logit > 0.0

    def test_inventory_skew_sign(self):
        """inventory_skew should be positive for long q."""
        result = self._quote(q=10.0)
        assert result.inventory_skew > 0.0

    def test_inventory_skew_zero_for_zero_q(self):
        result = self._quote(q=0.0)
        assert result.inventory_skew == pytest.approx(0.0, abs=1e-10)

    def test_zero_gamma_does_not_crash(self):
        """gamma=0 should not raise ZeroDivisionError."""
        result = as_quotes(mid=0.5, sigma_b=0.1, gamma=0.0)
        assert isinstance(result, ASQuoteResult)

    def test_spread_prob_equals_ask_minus_bid(self):
        result = self._quote()
        assert result.spread_prob == pytest.approx(result.ask - result.bid, abs=1e-6)


# ---------------------------------------------------------------------------
# spread_vs_sigma_b
# ---------------------------------------------------------------------------

class TestSpreadVsSigmaB:
    def test_returns_list_of_correct_length(self):
        sigma_range = [0.05, 0.10, 0.20]
        rows = spread_vs_sigma_b(sigma_range)
        assert len(rows) == 3

    def test_spread_monotone_with_sigma_b(self):
        sigma_range = [0.02, 0.05, 0.10, 0.20, 0.50]
        rows = spread_vs_sigma_b(sigma_range)
        spreads = [r["spread_prob"] for r in rows]
        for i in range(len(spreads) - 1):
            assert spreads[i] <= spreads[i + 1]

    def test_row_keys(self):
        rows = spread_vs_sigma_b([0.10])
        keys = set(rows[0].keys())
        assert {"sigma_b", "spread_prob", "half_spread", "bid", "ask"}.issubset(keys)


# ---------------------------------------------------------------------------
# spread_vs_inventory
# ---------------------------------------------------------------------------

class TestSpreadVsInventory:
    def test_returns_list_of_correct_length(self):
        q_range = [-10.0, 0.0, 10.0]
        rows = spread_vs_inventory(q_range)
        assert len(rows) == 3

    def test_reservation_price_moves_with_q(self):
        q_range = [-20.0, 0.0, 20.0]
        rows = spread_vs_inventory(q_range, mid=0.5, sigma_b=0.10)
        prices = [r["reservation_price"] for r in rows]
        # Should be monotone: long q → lower r_price
        assert prices[0] > prices[1] > prices[2]

    def test_inventory_skew_sign(self):
        rows = spread_vs_inventory([-10.0, 0.0, 10.0])
        assert rows[0]["inventory_skew"] < 0.0   # short → skew down
        assert rows[1]["inventory_skew"] == pytest.approx(0.0, abs=1e-10)
        assert rows[2]["inventory_skew"] > 0.0   # long → skew up


# ---------------------------------------------------------------------------
# position_size_hint
# ---------------------------------------------------------------------------

class TestPositionSizeHint:
    def test_zero_edge(self):
        assert position_size_hint(edge=0.0, sigma_b=0.10) == 0.0

    def test_negative_edge(self):
        assert position_size_hint(edge=-0.05, sigma_b=0.10) == 0.0

    def test_formula(self):
        """n* = edge / (γ · σ_b²)"""
        edge, sigma_b, gamma = 0.05, 0.10, 0.1
        expected = edge / (gamma * sigma_b ** 2)
        result = position_size_hint(edge=edge, sigma_b=sigma_b, gamma=gamma)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_capped_at_max_shares(self):
        result = position_size_hint(edge=10.0, sigma_b=0.001, gamma=0.001, max_shares=50.0)
        assert result == pytest.approx(50.0, rel=1e-6)

    def test_larger_edge_larger_size(self):
        s1 = position_size_hint(edge=0.02, sigma_b=0.10)
        s2 = position_size_hint(edge=0.05, sigma_b=0.10)
        assert s2 > s1

    def test_larger_sigma_smaller_size(self):
        s1 = position_size_hint(edge=0.05, sigma_b=0.10)
        s2 = position_size_hint(edge=0.05, sigma_b=0.30)
        assert s2 < s1
