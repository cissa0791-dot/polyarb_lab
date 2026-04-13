from __future__ import annotations

import unittest

from src.scanner.cross_market import (
    LHS_RELATION_EMPTY_ASKS,
    RELATION_EDGE_NON_POSITIVE,
)
from src.strategies.opportunity_strategies import CrossMarketConstraintStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class CrossMarketRelationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = CrossMarketConstraintStrategy()
        self.rule = {
            "name": "lhs_leq_rhs",
            "lhs": {"market_slug": "market-a", "side": "YES"},
            "rhs": {"market_slug": "market-b", "side": "YES"},
        }

    def test_detect_with_audit_reports_missing_lhs_relation_asks(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.rule,
            Book(asks=[], bids=[Level(0.59, 10.0)]),
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

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], LHS_RELATION_EMPTY_ASKS)
        self.assertEqual(audit["failure_stage"], "pre_candidate_relation")
        self.assertEqual(audit["strategy_family"], "cross_market_constraint")

    def test_detect_with_audit_reports_relation_edge_non_positive(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.rule,
            Book(asks=[Level(0.50, 10.0)], bids=[Level(0.49, 10.0)]),
            Book(asks=[Level(0.50, 10.0)], bids=[Level(0.49, 10.0)]),
            "market-a-no-token",
            "NO",
            Book(asks=[Level(0.20, 20.0)], bids=[Level(0.19, 20.0)]),
            "market-b-yes-token",
            "YES",
            Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.01,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RELATION_EDGE_NON_POSITIVE)
        self.assertLessEqual(audit["edge_after_buffer"], 0.0)

    def test_detect_success_path_is_unchanged(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.rule,
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

        direct_candidate = self.strategy.detect(
            self.rule,
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
        self.assertIsNone(audit)
        self.assertIsNotNone(direct_candidate)
        assert raw_candidate is not None and direct_candidate is not None
        self.assertEqual(raw_candidate.market_slugs, direct_candidate.market_slugs)
        self.assertEqual(raw_candidate.gross_edge_cents, direct_candidate.gross_edge_cents)
        self.assertEqual(raw_candidate.target_shares, direct_candidate.target_shares)


if __name__ == "__main__":
    unittest.main()
