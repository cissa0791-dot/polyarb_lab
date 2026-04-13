from __future__ import annotations

import unittest

from src.scanner.cross_market import RELATION_EDGE_NON_POSITIVE, RELATION_GROSS_EDGE_NON_POSITIVE
from src.strategies.opportunity_strategies import CrossMarketConstraintStrategy, CrossMarketGrossConstraintStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class CrossMarketGrossStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buffered = CrossMarketConstraintStrategy()
        self.gross = CrossMarketGrossConstraintStrategy()
        self.rule = {
            "name": "lhs_leq_rhs",
            "lhs": {"market_slug": "market-a", "side": "YES"},
            "rhs": {"market_slug": "market-b", "side": "YES"},
        }

    def test_gross_strategy_emits_raw_when_gap_is_positive_but_below_buffer(self) -> None:
        relation_lhs = Book(asks=[Level(0.025, 10.0)], bids=[Level(0.024, 10.0)])
        relation_rhs = Book(asks=[Level(0.019, 10.0)], bids=[Level(0.018, 10.0)])
        lhs_exec = Book(asks=[Level(0.975, 50.0)], bids=[Level(0.974, 50.0)])
        rhs_exec = Book(asks=[Level(0.019, 50.0)], bids=[Level(0.018, 50.0)])

        buffered_candidate, buffered_audit = self.buffered.detect_with_audit(
            self.rule,
            relation_lhs,
            relation_rhs,
            "market-a-no-token",
            "NO",
            lhs_exec,
            "market-b-yes-token",
            "YES",
            rhs_exec,
            max_notional=10.0,
            total_buffer_cents=0.02,
        )
        gross_candidate, gross_audit = self.gross.detect_with_audit(
            self.rule,
            relation_lhs,
            relation_rhs,
            "market-a-no-token",
            "NO",
            lhs_exec,
            "market-b-yes-token",
            "YES",
            rhs_exec,
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(buffered_candidate)
        self.assertIsNotNone(buffered_audit)
        assert buffered_audit is not None
        self.assertEqual(buffered_audit["failure_reason"], RELATION_EDGE_NON_POSITIVE)

        self.assertIsNotNone(gross_candidate)
        self.assertIsNone(gross_audit)
        assert gross_candidate is not None
        self.assertEqual(gross_candidate.strategy_family.value, "cross_market_gross_constraint")
        self.assertEqual(gross_candidate.gross_edge_cents, 0.006)

    def test_gross_strategy_reports_gross_non_positive_reason(self) -> None:
        raw_candidate, audit = self.gross.detect_with_audit(
            self.rule,
            Book(asks=[Level(0.010, 10.0)], bids=[Level(0.009, 10.0)]),
            Book(asks=[Level(0.012, 10.0)], bids=[Level(0.011, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.990, 50.0)], bids=[Level(0.989, 50.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.012, 50.0)], bids=[Level(0.011, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RELATION_GROSS_EDGE_NON_POSITIVE)
        self.assertEqual(audit["strategy_family"], "cross_market_gross_constraint")
        self.assertLessEqual(audit["relation_gap"], 0.0)

    def test_gross_detect_matches_detect_with_audit_success(self) -> None:
        relation_lhs = Book(asks=[Level(0.030, 10.0)], bids=[Level(0.029, 10.0)])
        relation_rhs = Book(asks=[Level(0.020, 10.0)], bids=[Level(0.019, 10.0)])
        lhs_exec = Book(asks=[Level(0.970, 50.0)], bids=[Level(0.969, 50.0)])
        rhs_exec = Book(asks=[Level(0.020, 50.0)], bids=[Level(0.019, 50.0)])

        raw_candidate, audit = self.gross.detect_with_audit(
            self.rule,
            relation_lhs,
            relation_rhs,
            "market-a-no-token",
            "NO",
            lhs_exec,
            "market-b-yes-token",
            "YES",
            rhs_exec,
            max_notional=10.0,
            total_buffer_cents=0.02,
        )
        direct = self.gross.detect(
            self.rule,
            relation_lhs,
            relation_rhs,
            "market-a-no-token",
            "NO",
            lhs_exec,
            "market-b-yes-token",
            "YES",
            rhs_exec,
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNotNone(raw_candidate)
        self.assertIsNone(audit)
        self.assertIsNotNone(direct)
        assert raw_candidate is not None and direct is not None
        self.assertEqual(raw_candidate.market_slugs, direct.market_slugs)
        self.assertEqual(raw_candidate.gross_edge_cents, direct.gross_edge_cents)


if __name__ == "__main__":
    unittest.main()
