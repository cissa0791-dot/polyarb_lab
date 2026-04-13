from __future__ import annotations

import unittest

from src.scanner.cross_market import EXECUTION_GROSS_EDGE_NON_POSITIVE, RELATION_GROSS_EDGE_NON_POSITIVE
from src.strategies.opportunity_strategies import CrossMarketExecutionGrossConstraintStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class CrossMarketExecutionGrossStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = CrossMarketExecutionGrossConstraintStrategy()
        self.rule = {
            "name": "lhs_leq_rhs",
            "lhs": {"market_slug": "market-a", "side": "YES"},
            "rhs": {"market_slug": "market-b", "side": "YES"},
        }

    def test_relation_must_still_be_positive(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.rule,
            Book(asks=[Level(0.010, 10.0)], bids=[Level(0.009, 10.0)]),
            Book(asks=[Level(0.012, 10.0)], bids=[Level(0.011, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.980, 50.0)], bids=[Level(0.979, 50.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.010, 50.0)], bids=[Level(0.009, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RELATION_GROSS_EDGE_NON_POSITIVE)
        self.assertEqual(audit["strategy_family"], "cross_market_execution_gross_constraint")

    def test_execution_gross_edge_must_be_positive(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.rule,
            Book(asks=[Level(0.030, 10.0)], bids=[Level(0.029, 10.0)]),
            Book(asks=[Level(0.020, 10.0)], bids=[Level(0.019, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.980, 50.0)], bids=[Level(0.979, 50.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.020, 50.0)], bids=[Level(0.019, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], EXECUTION_GROSS_EDGE_NON_POSITIVE)
        self.assertEqual(audit["execution_pair_best_ask_cost"], 1.0)
        self.assertEqual(audit["execution_best_ask_edge_cents"], 0.0)

    def test_success_uses_execution_gross_edge_as_raw_signal(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.rule,
            Book(asks=[Level(0.025, 10.0)], bids=[Level(0.024, 10.0)]),
            Book(asks=[Level(0.019, 10.0)], bids=[Level(0.018, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.977, 50.0)], bids=[Level(0.976, 50.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.019, 50.0)], bids=[Level(0.018, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(audit)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.strategy_family.value, "cross_market_execution_gross_constraint")
        self.assertEqual(raw_candidate.gross_edge_cents, 0.004)
        self.assertEqual(raw_candidate.metadata["relation_gap"], 0.006)
        self.assertEqual(raw_candidate.metadata["execution_best_ask_edge_cents"], 0.004)


if __name__ == "__main__":
    unittest.main()
