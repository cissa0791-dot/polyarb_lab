from __future__ import annotations

import unittest

from src.core.models import MarketPair
from src.domain.models import RejectionReason
from src.scanner.single_market import TOUCH_EDGE_NON_POSITIVE
from src.strategies.opportunity_strategies import (
    SingleMarketMispricingStrategy,
    SingleMarketTouchMispricingStrategy,
)


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class SingleMarketTouchStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.touch_strategy = SingleMarketTouchMispricingStrategy()
        self.vwap_strategy = SingleMarketMispricingStrategy()
        self.pair = MarketPair(
            market_slug="market-touch",
            yes_token_id="yes-token",
            no_token_id="no-token",
            question="Example?",
        )

    def test_touch_detect_with_audit_reports_non_positive_touch_edge(self) -> None:
        raw_candidate, audit = self.touch_strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.60, 100.0)], bids=[Level(0.59, 100.0)]),
            Book(asks=[Level(0.45, 100.0)], bids=[Level(0.44, 100.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], TOUCH_EDGE_NON_POSITIVE)
        self.assertEqual(audit["failure_stage"], "pre_candidate_precheck")
        self.assertEqual(audit["strategy_family"], "single_market_touch_mispricing")
        self.assertAlmostEqual(audit["touch_pair_cost"], 1.05)
        self.assertLessEqual(audit["touch_edge_gross"], 0.0)

    def test_touch_detect_with_audit_rejects_empty_asks_early(self) -> None:
        raw_candidate, audit = self.touch_strategy.detect_with_audit(
            self.pair,
            Book(asks=[], bids=[Level(0.39, 50.0)]),
            Book(asks=[Level(0.50, 50.0)], bids=[Level(0.49, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RejectionReason.EMPTY_ASKS.value)
        self.assertEqual(audit["failure_stage"], "pre_candidate_precheck")
        self.assertNotEqual(audit["failure_stage"], "unknown")

    def test_touch_detect_with_audit_rejects_invalid_orderbook_early(self) -> None:
        raw_candidate, audit = self.touch_strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(-0.40, 50.0)], bids=[Level(0.39, 50.0)]),
            Book(asks=[Level(0.50, 50.0)], bids=[Level(0.49, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], RejectionReason.INVALID_ORDERBOOK.value)
        self.assertEqual(audit["failure_stage"], "pre_candidate_precheck")

    def test_touch_strategy_can_emit_candidate_when_budget_vwap_strategy_rejects(self) -> None:
        yes_book = Book(
            asks=[Level(0.01, 1.0), Level(0.99, 1000.0)],
            bids=[Level(0.009, 1.0)],
        )
        no_book = Book(
            asks=[Level(0.95, 1000.0)],
            bids=[Level(0.94, 1000.0)],
        )

        vwap_candidate, vwap_audit = self.vwap_strategy.detect_with_audit(
            self.pair,
            yes_book,
            no_book,
            max_notional=10.0,
            total_buffer_cents=0.02,
        )
        touch_candidate, touch_audit = self.touch_strategy.detect_with_audit(
            self.pair,
            yes_book,
            no_book,
            max_notional=10.0,
            total_buffer_cents=0.02,
        )

        self.assertIsNone(vwap_candidate)
        self.assertIsNotNone(vwap_audit)
        self.assertIsNotNone(touch_candidate)
        self.assertIsNone(touch_audit)
        assert touch_candidate is not None
        self.assertEqual(touch_candidate.strategy_family.value, "single_market_touch_mispricing")
        self.assertAlmostEqual(touch_candidate.gross_edge_cents, 0.04)

    def test_touch_detect_success_path_matches_detect(self) -> None:
        raw_candidate, audit = self.touch_strategy.detect_with_audit(
            self.pair,
            Book(asks=[Level(0.40, 50.0)], bids=[Level(0.39, 50.0)]),
            Book(asks=[Level(0.50, 50.0)], bids=[Level(0.49, 50.0)]),
            max_notional=10.0,
            total_buffer_cents=0.02,
        )
        direct_candidate = self.touch_strategy.detect(
            self.pair,
            Book(asks=[Level(0.40, 50.0)], bids=[Level(0.39, 50.0)]),
            Book(asks=[Level(0.50, 50.0)], bids=[Level(0.49, 50.0)]),
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
