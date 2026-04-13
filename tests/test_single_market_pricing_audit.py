from __future__ import annotations

import unittest

from src.core.models import MarketPair
from src.scanner.single_market import (
    NO_BUDGET_UNFILLABLE,
    PAIR_EDGE_NON_POSITIVE,
    YES_BUDGET_UNFILLABLE,
)
from src.domain.models import RejectionReason
from src.strategies.opportunity_strategies import SingleMarketMispricingStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class SingleMarketPricingAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = SingleMarketMispricingStrategy()
        self.pair = MarketPair(
            market_slug="market-a",
            yes_token_id="yes-token",
            no_token_id="no-token",
            question="Example?",
        )

    def test_detect_with_audit_reports_yes_budget_unfillable(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.40, 5.0)], bids=[Level(0.39, 5.0)]),
            Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], YES_BUDGET_UNFILLABLE)
        self.assertEqual(audit["failure_stage"], "pre_candidate_pricing")
        self.assertEqual(audit["strategy_family"], "single_market_mispricing")
        self.assertIn("levels_consumed_yes", audit)
        self.assertIn("pair_cost_budget_vwap_sum", audit)

    def test_detect_with_audit_rejects_empty_asks_early(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[], bids=[Level(0.39, 5.0)]),
            Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RejectionReason.EMPTY_ASKS.value)
        self.assertEqual(audit["failure_stage"], "pre_candidate_precheck")
        self.assertEqual(audit["failed_leg"], "YES")

    def test_detect_with_audit_rejects_invalid_orderbook_early(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.40, 20.0), Level(0.39, 20.0)], bids=[Level(0.38, 20.0)]),
            Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RejectionReason.INVALID_ORDERBOOK.value)
        self.assertEqual(audit["failure_stage"], "pre_candidate_precheck")
        self.assertEqual(audit["failed_leg"], "YES")

    def test_detect_with_audit_rejects_obvious_non_positive_pair_edge_early(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.60, 20.0)], bids=[Level(0.59, 20.0)]),
            Book(asks=[Level(0.43, 20.0)], bids=[Level(0.42, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], PAIR_EDGE_NON_POSITIVE)
        self.assertEqual(audit["failure_stage"], "pre_candidate_precheck")
        self.assertLessEqual(audit["touch_edge_after_buffer"], 0.0)

    def test_detect_with_audit_reports_no_budget_unfillable(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            Book(asks=[Level(0.45, 5.0)], bids=[Level(0.44, 5.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], NO_BUDGET_UNFILLABLE)
        self.assertEqual(audit["failure_stage"], "pre_candidate_pricing")

    def test_detect_with_audit_reports_pair_edge_non_positive(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.50, 20.0)], bids=[Level(0.49, 20.0)]),
            Book(asks=[Level(0.50, 20.0)], bids=[Level(0.49, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], PAIR_EDGE_NON_POSITIVE)
        if "edge_after_buffer" in audit:
            self.assertLessEqual(audit["edge_after_buffer"], 0.0)
        else:
            self.assertLessEqual(audit["touch_edge_after_buffer"], 0.0)

    def test_detect_success_path_is_unchanged(self) -> None:
        raw_candidate, audit = self.strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        direct_candidate = self.strategy.detect(
            self.pair,
            Book(asks=[Level(0.40, 20.0)], bids=[Level(0.39, 20.0)]),
            Book(asks=[Level(0.45, 20.0)], bids=[Level(0.44, 20.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
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
