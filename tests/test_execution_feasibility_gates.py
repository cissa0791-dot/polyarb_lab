"""
tests/test_execution_feasibility_gates.py

Tests for the two new execution feasibility gates:

  Gate 1 — ABSOLUTE_DEPTH_BELOW_FLOOR
      Each leg must have at least OpportunityConfig.min_absolute_leg_depth_usd
      of available dollar notional on the correct side of the book.
      DepthAnalyzer.check_absolute_depth() performs the check.
      ExecutionFeasibilityEvaluator.qualify() appends the rejection reason
      when any leg fails.

  Gate 2 — SIZED_NOTIONAL_TOO_SMALL
      After the quality/risk multipliers in DepthCappedSizer.size() shrink the
      position, the resulting notional must be >= OpportunityConfig.min_sized_notional_usd.
      When it falls below that floor the sizer returns viable=False,
      notional_usd=0.0, shares=0.0 — triggering the runner's existing
      `if sizing.notional_usd <= 1e-9` guard without any runner-side changes.

Both gates default to 0.0 (disabled) for full backward compatibility.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.config_runtime.models import OpportunityConfig, PaperConfig
from src.domain.models import AccountSnapshot
from src.opportunity.models import CandidateLeg, ExecutableCandidate, RawCandidate, StrategyFamily
from src.opportunity.qualification import DepthAnalyzer, ExecutionFeasibilityEvaluator, VWAPCalculator
from src.sizing.engine import DepthCappedSizer, SizingDecision
from src.opportunity.qualification import OpportunityRanker


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _raw(required_shares: float = 5.0, notional: float = 4.0) -> RawCandidate:
    return RawCandidate(
        strategy_id="s1",
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
        candidate_id="raw-gate-test",
        kind="single_market",
        detection_name="yes_no_under_1",
        market_slugs=["mkt-a"],
        gross_edge_cents=0.10,
        expected_payout=required_shares,
        target_notional_usd=notional,
        target_shares=required_shares,
        legs=[
            CandidateLeg(token_id="yes-t", market_slug="mkt-a", action="BUY", side="YES", required_shares=required_shares),
            CandidateLeg(token_id="no-t",  market_slug="mkt-a", action="BUY", side="NO",  required_shares=required_shares),
        ],
        ts=_ts(),
    )


def _base_opp_config(**overrides) -> OpportunityConfig:
    """Permissive base config — individual tests override as needed."""
    defaults = dict(
        min_edge_cents=0.01,
        fee_buffer_cents=0.005,
        slippage_buffer_cents=0.005,
        min_depth_multiple=1.5,
        max_spread_cents=0.20,
        max_partial_fill_risk=0.95,
        max_non_atomic_risk=0.95,
        min_net_profit_usd=0.01,
        min_absolute_leg_depth_usd=0.0,
        min_sized_notional_usd=0.0,
    )
    defaults.update(overrides)
    return OpportunityConfig(**defaults)


def _exec_candidate(**overrides) -> ExecutableCandidate:
    defaults = dict(
        strategy_id="s1",
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
        candidate_id="exec-gate-test",
        kind="single_market",
        market_slugs=["mkt-a"],
        gross_edge_cents=0.12,
        fee_estimate_cents=0.01,
        slippage_estimate_cents=0.01,
        target_notional_usd=10.0,
        estimated_depth_usd=30.0,
        score=0.0,
        estimated_net_profit_usd=1.20,
        ts=_ts(),
        required_depth_usd=10.0,
        available_depth_usd=30.0,
        required_shares=10.0,
        available_shares=30.0,
        partial_fill_risk_score=0.10,
        non_atomic_execution_risk_score=0.10,
    )
    defaults.update(overrides)
    return ExecutableCandidate(**defaults)


# ---------------------------------------------------------------------------
# Gate 1: ABSOLUTE_DEPTH_BELOW_FLOOR
# ---------------------------------------------------------------------------

class AbsoluteDepthFloorTests(unittest.TestCase):

    def test_check_absolute_depth_passes_when_both_legs_meet_floor(self) -> None:
        analyzer = DepthAnalyzer(VWAPCalculator())
        yes_leg = CandidateLeg(token_id="y", market_slug="m", action="BUY", side="YES", required_shares=5.0)
        no_leg  = CandidateLeg(token_id="n", market_slug="m", action="BUY", side="NO",  required_shares=5.0)
        # 10 shares @ $0.40 = $4.00 per leg — floor is $3.00
        yes_book = Book(asks=[Level(0.40, 10.0)], bids=[Level(0.39, 10.0)])
        no_book  = Book(asks=[Level(0.45, 10.0)], bids=[Level(0.44, 10.0)])

        ok, failing = analyzer.check_absolute_depth(
            [(yes_leg, yes_book), (no_leg, no_book)], min_depth_usd=3.0
        )

        self.assertTrue(ok)
        self.assertEqual(failing, [])

    def test_check_absolute_depth_fails_when_one_leg_is_below_floor(self) -> None:
        analyzer = DepthAnalyzer(VWAPCalculator())
        yes_leg = CandidateLeg(token_id="y", market_slug="m", action="BUY", side="YES", required_shares=5.0)
        no_leg  = CandidateLeg(token_id="n", market_slug="m", action="BUY", side="NO",  required_shares=5.0)
        # yes leg: 10 @ 0.40 = $4.00 (passes $3 floor)
        # no  leg: 1  @ 0.45 = $0.45 (fails $3 floor)
        yes_book = Book(asks=[Level(0.40, 10.0)], bids=[Level(0.39, 10.0)])
        thin_no  = Book(asks=[Level(0.45,  1.0)], bids=[Level(0.44,  1.0)])

        ok, failing = analyzer.check_absolute_depth(
            [(yes_leg, yes_book), (no_leg, thin_no)], min_depth_usd=3.0
        )

        self.assertFalse(ok)
        self.assertIn("n", failing)
        self.assertNotIn("y", failing)

    def test_qualify_rejects_with_absolute_depth_below_floor_reason(self) -> None:
        raw = _raw(required_shares=5.0, notional=4.0)
        # Both legs have 3 shares @ $0.45 = $1.35 available — below $20 floor
        thin_book = Book(asks=[Level(0.45, 3.0)], bids=[Level(0.44, 3.0)])
        evaluator = ExecutionFeasibilityEvaluator(
            _base_opp_config(min_absolute_leg_depth_usd=20.0)
        )

        decision = evaluator.qualify(raw, {"yes-t": thin_book, "no-t": thin_book})

        self.assertFalse(decision.passed)
        self.assertIn("ABSOLUTE_DEPTH_BELOW_FLOOR", decision.reason_codes)

    def test_qualify_passes_when_absolute_depth_gate_is_disabled(self) -> None:
        """min_absolute_leg_depth_usd=0.0 skips the check entirely (backward compat)."""
        raw = _raw(required_shares=5.0, notional=4.0)
        # Very thin book — would fail a $20 floor but gate is off
        thin_book = Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)])
        evaluator = ExecutionFeasibilityEvaluator(
            _base_opp_config(min_absolute_leg_depth_usd=0.0)
        )

        decision = evaluator.qualify(raw, {"yes-t": thin_book, "no-t": thin_book})

        self.assertNotIn("ABSOLUTE_DEPTH_BELOW_FLOOR", decision.reason_codes)


# ---------------------------------------------------------------------------
# Gate 2: SIZED_NOTIONAL_TOO_SMALL
# ---------------------------------------------------------------------------

class SizedNotionalMinimumTests(unittest.TestCase):

    def _ranked(self, quality_score: float = 30.0, **overrides) -> "RankedOpportunity":
        from src.opportunity.models import RankedOpportunity
        ec = _exec_candidate(**overrides)
        ranker = OpportunityRanker(_base_opp_config())
        ranked = ranker.rank(ec)
        # Force a specific quality_score to control sizing arithmetic
        payload = ranked.model_dump(mode="python")
        payload["quality_score"] = quality_score
        payload["score"] = quality_score
        payload["ranking_score"] = quality_score
        return RankedOpportunity.model_validate(payload)

    def test_viable_sizing_when_result_meets_minimum(self) -> None:
        # quality_score=80 → quality_multiplier=0.80; risk_mult≈0.80;
        # max_budget = min(10, 100, 40, 20) = 10  (depth_cap=30/1.5=20)
        # sized_notional = 10 * 0.80 * 0.80 = 6.40  — above $5 floor
        ranked = self._ranked(
            quality_score=80.0,
            target_notional_usd=10.0,
            available_depth_usd=30.0,
            required_depth_usd=10.0,
        )
        account = AccountSnapshot(cash=40.0, frozen_cash=0.0, ts=_ts())
        sizer = DepthCappedSizer(
            PaperConfig(max_notional_per_arb=100.0),
            _base_opp_config(min_depth_multiple=1.5, min_sized_notional_usd=5.0),
        )

        decision = sizer.size(ranked, account)

        self.assertTrue(decision.viable)
        self.assertGreater(decision.notional_usd, 5.0)
        self.assertIsNone(decision.rejection_reason)

    def test_non_viable_sizing_when_result_is_below_minimum(self) -> None:
        # quality_score=25 → quality_mult clamped to 0.25; risk_mult ≈ 0.80
        # account cash = $1.50; max_budget = min(10, 100, 1.50, 20) = 1.50
        # sized_notional = 1.50 * 0.25 * 0.80 = 0.30  — below $5 floor
        ranked = self._ranked(
            quality_score=25.0,
            target_notional_usd=10.0,
            available_depth_usd=30.0,
            required_depth_usd=10.0,
        )
        account = AccountSnapshot(cash=1.50, frozen_cash=0.0, ts=_ts())
        sizer = DepthCappedSizer(
            PaperConfig(max_notional_per_arb=100.0),
            _base_opp_config(min_depth_multiple=1.5, min_sized_notional_usd=5.0),
        )

        decision = sizer.size(ranked, account)

        self.assertFalse(decision.viable)
        self.assertEqual(decision.notional_usd, 0.0)
        self.assertEqual(decision.shares, 0.0)
        self.assertEqual(decision.rejection_reason, "SIZED_NOTIONAL_TOO_SMALL")

    def test_min_sized_notional_disabled_at_zero(self) -> None:
        """min_sized_notional_usd=0.0 disables the gate (backward compat)."""
        ranked = self._ranked(
            quality_score=25.0,
            target_notional_usd=10.0,
            available_depth_usd=30.0,
            required_depth_usd=10.0,
        )
        account = AccountSnapshot(cash=1.50, frozen_cash=0.0, ts=_ts())
        sizer = DepthCappedSizer(
            PaperConfig(max_notional_per_arb=100.0),
            _base_opp_config(min_depth_multiple=1.5, min_sized_notional_usd=0.0),
        )

        decision = sizer.size(ranked, account)

        # Gate is off — no rejection regardless of how small the notional is
        self.assertTrue(decision.viable)
        self.assertIsNone(decision.rejection_reason)


# ---------------------------------------------------------------------------
# Combined: conservative config rejects candidates that permissive config passes
# ---------------------------------------------------------------------------

class ConservativeVsPermissiveTests(unittest.TestCase):

    def _qualify(self, raw, yes_book, no_book, **config_overrides):
        cfg = _base_opp_config(**config_overrides)
        return ExecutionFeasibilityEvaluator(cfg).qualify(
            raw, {"yes-t": yes_book, "no-t": no_book}
        )

    def test_thin_book_passes_permissive_config_but_fails_conservative(self) -> None:
        """A book with $2 available per leg passes the ratio check but
        should fail the absolute depth floor when that gate is enabled."""
        raw = _raw(required_shares=2.0, notional=1.6)
        # 2 shares @ $0.40 = $0.80 per leg
        thin = Book(asks=[Level(0.40, 2.0)], bids=[Level(0.39, 2.0)])

        permissive = self._qualify(raw, thin, thin, min_absolute_leg_depth_usd=0.0)
        conservative = self._qualify(raw, thin, thin, min_absolute_leg_depth_usd=20.0)

        # Permissive: gate disabled — result depends only on edge/depth ratio
        self.assertNotIn("ABSOLUTE_DEPTH_BELOW_FLOOR", permissive.reason_codes)
        # Conservative: gate enabled — thin book is explicitly rejected
        self.assertFalse(conservative.passed)
        self.assertIn("ABSOLUTE_DEPTH_BELOW_FLOOR", conservative.reason_codes)


if __name__ == "__main__":
    unittest.main()
