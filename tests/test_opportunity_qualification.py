from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.config_runtime.models import OpportunityConfig, PaperConfig
from src.core.models import MarketPair
from src.domain.models import AccountSnapshot
from src.opportunity.models import CandidateLeg, ExecutableCandidate, RawCandidate, StrategyFamily
from src.opportunity.qualification import ExecutionFeasibilityEvaluator, OpportunityRanker
from src.sizing.engine import DepthCappedSizer
from src.strategies.opportunity_strategies import CrossMarketConstraintStrategy, SingleMarketMispricingStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


def build_raw_candidate(required_shares: float = 5.0, target_notional_usd: float = 4.0) -> RawCandidate:
    return RawCandidate(
        strategy_id="single_market_sum_under_1",
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
        candidate_id="raw-1",
        kind="single_market",
        detection_name="yes_no_under_1",
        market_slugs=["market-a"],
        gross_edge_cents=0.10,
        expected_payout=required_shares,
        target_notional_usd=target_notional_usd,
        target_shares=required_shares,
        gross_profit_usd=0.50,
        est_fill_cost_usd=target_notional_usd,
        legs=[
            CandidateLeg(token_id="yes-token", market_slug="market-a", action="BUY", side="YES", required_shares=required_shares),
            CandidateLeg(token_id="no-token", market_slug="market-a", action="BUY", side="NO", required_shares=required_shares),
        ],
        metadata={"market_slug": "market-a", "pair_cost": 0.80},
        ts=datetime.now(timezone.utc),
    )


class OpportunityQualificationTests(unittest.TestCase):
    def test_raw_candidate_to_executable_and_ranked_opportunity(self) -> None:
        strategy = SingleMarketMispricingStrategy()
        pair = MarketPair(market_slug="market-a", yes_token_id="yes-token", no_token_id="no-token", question="Example?")
        yes_book = Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)])
        no_book = Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)])

        raw_candidate = strategy.detect(pair, yes_book, no_book, max_notional=10.0, total_buffer_cents=0.02)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.strategy_family, StrategyFamily.SINGLE_MARKET_MISPRICING)

        evaluator = ExecutionFeasibilityEvaluator(
            OpportunityConfig(
                min_edge_cents=0.02,
                fee_buffer_cents=0.01,
                slippage_buffer_cents=0.01,
                min_depth_multiple=1.5,
                max_partial_fill_risk=0.9,
                max_non_atomic_risk=0.9,
                min_net_profit_usd=0.05,
            )
        )
        decision = evaluator.qualify(raw_candidate, {"yes-token": yes_book, "no-token": no_book})
        self.assertTrue(decision.passed)
        assert decision.executable_candidate is not None
        self.assertGreater(decision.executable_candidate.estimated_net_profit_usd, 0.0)

        ranker = OpportunityRanker(evaluator.config)
        ranked = ranker.rank(decision.executable_candidate)
        self.assertGreater(ranked.ranking_score, 0.0)
        self.assertEqual(ranked.score, ranked.ranking_score)

    def test_qualification_filters_wide_spread_and_insufficient_depth(self) -> None:
        raw_candidate = build_raw_candidate(required_shares=5.0, target_notional_usd=4.0)
        evaluator = ExecutionFeasibilityEvaluator(
            OpportunityConfig(
                min_edge_cents=0.02,
                fee_buffer_cents=0.01,
                slippage_buffer_cents=0.01,
                min_depth_multiple=1.5,
                max_spread_cents=0.08,
                max_partial_fill_risk=0.95,
                max_non_atomic_risk=0.95,
                min_net_profit_usd=0.05,
            )
        )
        yes_book = Book(asks=[Level(0.40, 3.0)], bids=[Level(0.10, 10.0)])
        no_book = Book(asks=[Level(0.45, 5.0)], bids=[Level(0.44, 10.0)])

        decision = evaluator.qualify(raw_candidate, {"yes-token": yes_book, "no-token": no_book})

        self.assertFalse(decision.passed)
        self.assertIn("SPREAD_TOO_WIDE", decision.reason_codes)
        self.assertIn("INSUFFICIENT_DEPTH", decision.reason_codes)

    def test_qualification_rejects_non_atomic_execution_risk(self) -> None:
        raw_candidate = build_raw_candidate(required_shares=5.0, target_notional_usd=3.6)
        evaluator = ExecutionFeasibilityEvaluator(
            OpportunityConfig(
                min_edge_cents=0.01,
                fee_buffer_cents=0.005,
                slippage_buffer_cents=0.005,
                min_depth_multiple=3.0,
                max_partial_fill_risk=0.8,
                max_non_atomic_risk=0.15,
                min_net_profit_usd=0.01,
            )
        )
        yes_book = Book(asks=[Level(0.10, 1.0), Level(0.50, 14.0)], bids=[Level(0.09, 20.0)])
        no_book = Book(asks=[Level(0.30, 15.0)], bids=[Level(0.29, 20.0)])

        decision = evaluator.qualify(raw_candidate, {"yes-token": yes_book, "no-token": no_book})

        self.assertFalse(decision.passed)
        self.assertIn("NON_ATOMIC_RISK_TOO_HIGH", decision.reason_codes)

    def test_qualification_rejects_missing_book_with_precise_reason(self) -> None:
        raw_candidate = build_raw_candidate(required_shares=5.0, target_notional_usd=4.0)
        evaluator = ExecutionFeasibilityEvaluator(
            OpportunityConfig(
                min_edge_cents=0.02,
                fee_buffer_cents=0.01,
                slippage_buffer_cents=0.01,
                min_depth_multiple=1.5,
                max_partial_fill_risk=0.95,
                max_non_atomic_risk=0.95,
                min_net_profit_usd=0.05,
            )
        )
        yes_book = Book(asks=[Level(0.40, 10.0)], bids=[Level(0.39, 10.0)])

        decision = evaluator.qualify(raw_candidate, {"yes-token": yes_book})

        self.assertFalse(decision.passed)
        self.assertIn("MISSING_ORDERBOOK", decision.reason_codes)
        self.assertNotIn("INVALID_ORDERBOOK", decision.reason_codes)

    def test_ranker_orders_higher_quality_candidate_above_weaker_one(self) -> None:
        ranker = OpportunityRanker(OpportunityConfig(min_edge_cents=0.02, min_net_profit_usd=0.10))
        strong = ExecutableCandidate(
            strategy_id="s1",
            strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
            candidate_id="exec-1",
            kind="single_market",
            market_slugs=["market-a"],
            gross_edge_cents=0.12,
            fee_estimate_cents=0.01,
            slippage_estimate_cents=0.01,
            target_notional_usd=10.0,
            estimated_depth_usd=40.0,
            score=0.0,
            estimated_net_profit_usd=1.20,
            ts=datetime.now(timezone.utc),
            required_depth_usd=10.0,
            available_depth_usd=40.0,
            required_shares=10.0,
            available_shares=40.0,
            partial_fill_risk_score=0.10,
            non_atomic_execution_risk_score=0.10,
        )
        weak = strong.model_copy(
            update={
                "candidate_id": "exec-2",
                "gross_edge_cents": 0.05,
                "estimated_net_profit_usd": 0.20,
                "partial_fill_risk_score": 0.50,
                "non_atomic_execution_risk_score": 0.45,
                "available_depth_usd": 12.0,
            }
        )

        strong_ranked = ranker.rank(strong)
        weak_ranked = ranker.rank(weak)

        self.assertGreater(strong_ranked.ranking_score, weak_ranked.ranking_score)

    def test_depth_capped_sizer_is_conservative(self) -> None:
        ranker = OpportunityRanker(OpportunityConfig(min_edge_cents=0.02, min_net_profit_usd=0.10))
        executable = ExecutableCandidate(
            strategy_id="s1",
            strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
            candidate_id="exec-3",
            kind="single_market",
            market_slugs=["market-a"],
            gross_edge_cents=0.12,
            fee_estimate_cents=0.01,
            slippage_estimate_cents=0.01,
            target_notional_usd=100.0,
            estimated_depth_usd=300.0,
            score=0.0,
            estimated_net_profit_usd=3.0,
            ts=datetime.now(timezone.utc),
            required_depth_usd=100.0,
            available_depth_usd=300.0,
            required_shares=100.0,
            available_shares=300.0,
            partial_fill_risk_score=0.10,
            non_atomic_execution_risk_score=0.10,
        )
        ranked = ranker.rank(executable)
        account = AccountSnapshot(cash=40.0, frozen_cash=0.0, ts=datetime.now(timezone.utc))
        sizer = DepthCappedSizer(PaperConfig(max_notional_per_arb=100.0), OpportunityConfig(min_depth_multiple=3.0))

        sizing = sizer.size(ranked, account)

        self.assertAlmostEqual(sizing.notional_usd, 28.8, places=6)
        self.assertAlmostEqual(sizing.shares, 28.8, places=6)

    def test_cross_market_raw_candidate_flows_into_qualified_research_only_ranked_opportunity(self) -> None:
        strategy = CrossMarketConstraintStrategy()
        raw_candidate = strategy.detect(
            {
                "name": "subset_relation",
                "lhs": {"market_slug": "market-a", "side": "YES"},
                "rhs": {"market_slug": "market-b", "side": "YES"},
            },
            Book(asks=[Level(0.60, 10.0)], bids=[Level(0.59, 10.0)]),
            Book(asks=[Level(0.40, 10.0)], bids=[Level(0.39, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.20, 20.0)], bids=[Level(0.19, 20.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.01,
        )
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None

        evaluator = ExecutionFeasibilityEvaluator(
            OpportunityConfig(
                min_edge_cents=0.01,
                fee_buffer_cents=0.005,
                slippage_buffer_cents=0.005,
                min_depth_multiple=1.5,
                max_partial_fill_risk=0.95,
                max_non_atomic_risk=0.95,
                min_net_profit_usd=0.01,
            )
        )
        decision = evaluator.qualify(
            raw_candidate,
            {
                "market-a-no-token": Book(asks=[Level(0.20, 20.0)], bids=[Level(0.19, 20.0)]),
                "market-b-yes-token": Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            },
        )

        self.assertTrue(decision.passed)
        assert decision.executable_candidate is not None
        self.assertTrue(decision.executable_candidate.research_only)
        self.assertEqual(decision.executable_candidate.execution_mode, "research_only")

        ranked = OpportunityRanker(evaluator.config).rank(decision.executable_candidate)
        self.assertTrue(ranked.research_only)
        self.assertGreater(ranked.ranking_score, 0.0)

    def test_cross_market_qualification_rejects_multi_leg_depth_shortfall(self) -> None:
        strategy = CrossMarketConstraintStrategy()
        raw_candidate = strategy.detect(
            {
                "name": "subset_relation",
                "lhs": {"market_slug": "market-a", "side": "YES"},
                "rhs": {"market_slug": "market-b", "side": "YES"},
            },
            Book(asks=[Level(0.60, 10.0)], bids=[Level(0.59, 10.0)]),
            Book(asks=[Level(0.40, 10.0)], bids=[Level(0.39, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.20, 1.0)], bids=[Level(0.19, 1.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.01,
        )
        assert raw_candidate is not None
        evaluator = ExecutionFeasibilityEvaluator(
            OpportunityConfig(
                min_edge_cents=0.01,
                fee_buffer_cents=0.005,
                slippage_buffer_cents=0.005,
                min_depth_multiple=2.0,
                max_partial_fill_risk=0.5,
                max_non_atomic_risk=0.9,
                min_net_profit_usd=0.01,
            )
        )
        decision = evaluator.qualify(
            raw_candidate,
            {
                "market-a-no-token": Book(asks=[Level(0.20, 1.0)], bids=[Level(0.19, 1.0)]),
                "market-b-yes-token": Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            },
        )

        self.assertFalse(decision.passed)
        self.assertIn("INSUFFICIENT_DEPTH", decision.reason_codes)
        self.assertIn("PARTIAL_FILL_RISK_TOO_HIGH", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
