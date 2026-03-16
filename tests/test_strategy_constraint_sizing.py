from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.beliefs.models import BeliefSnapshot, BeliefVsMarketComparator, ConfidenceScore
from src.constraints.models import (
    ConstraintGraph,
    ExactlyOneConstraint,
    FeasibilityChecker,
    ImplicationConstraint,
    MutualExclusionConstraint,
    SubsetConstraint,
)
from src.core.models import MarketPair
from src.opportunity.models import StrategyFamily
from src.strategies.opportunity_strategies import CrossMarketConstraintStrategy, SingleMarketMispricingStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class StrategyConstraintBeliefTests(unittest.TestCase):
    def test_constraint_graph_reports_violations(self) -> None:
        graph = ConstraintGraph()
        graph.add(ImplicationConstraint(name="rain_implies_cloud", premise="rain", consequence="cloud"))
        graph.add(MutualExclusionConstraint(name="mutual", symbols=("yes", "no")))
        graph.add(ExactlyOneConstraint(name="exactly_one", symbols=("a", "b")))
        graph.add(SubsetConstraint(name="subset", subset_symbol="child", superset_symbol="parent"))

        checker = FeasibilityChecker()
        result = checker.check(
            graph,
            {
                "rain": True,
                "cloud": False,
                "yes": True,
                "no": True,
                "a": False,
                "b": False,
                "child": True,
                "parent": False,
            },
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.violations), 4)

    def test_single_and_cross_market_strategies_emit_raw_candidates(self) -> None:
        single_strategy = SingleMarketMispricingStrategy()
        pair = MarketPair(market_slug="market-a", yes_token_id="yes-token", no_token_id="no-token", question="Example?")
        yes_book = Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)])
        no_book = Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)])

        single_candidate = single_strategy.detect(pair, yes_book, no_book, max_notional=10.0, total_buffer_cents=0.02)
        self.assertIsNotNone(single_candidate)
        assert single_candidate is not None
        self.assertEqual(single_candidate.strategy_family, StrategyFamily.SINGLE_MARKET_MISPRICING)
        self.assertEqual(len(single_candidate.legs), 2)

        cross_strategy = CrossMarketConstraintStrategy()
        cross_candidate = cross_strategy.detect(
            {
                "name": "lhs_leq_rhs",
                "lhs": {"market_slug": "market-a"},
                "rhs": {"market_slug": "market-b"},
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
        self.assertIsNotNone(cross_candidate)
        assert cross_candidate is not None
        self.assertEqual(cross_candidate.strategy_family, StrategyFamily.CROSS_MARKET_CONSTRAINT)
        self.assertEqual(cross_candidate.market_slugs, ["market-a", "market-b"])
        self.assertTrue(cross_candidate.research_only)
        self.assertEqual(len(cross_candidate.legs), 2)

    def test_belief_comparator_is_confidence_weighted(self) -> None:
        comparator = BeliefVsMarketComparator()
        snapshot = BeliefSnapshot(
            source_id="belief-source",
            subject_id="rain-sydney",
            probability=0.70,
            confidence=ConfidenceScore(0.5, "medium"),
            ts=datetime.now(timezone.utc),
        )

        comparison = comparator.compare(snapshot, market_probability=0.40)

        self.assertAlmostEqual(comparison.discrepancy, 0.30, places=6)
        self.assertAlmostEqual(comparison.weighted_discrepancy, 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
