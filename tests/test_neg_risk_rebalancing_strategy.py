from __future__ import annotations

import unittest

from src.scanner.neg_risk import (
    NEG_RISK_BASKET_EDGE_NON_POSITIVE,
    build_eligible_neg_risk_event_groups,
)
from src.strategies.opportunity_strategies import NegRiskRebalancingStrategy


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class NegRiskRebalancingStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = NegRiskRebalancingStrategy()

    def test_eligible_neg_risk_event_grouping_keeps_only_named_markets(self) -> None:
        events = [
            {
                "id": "evt-1",
                "slug": "neg-risk-event",
                "title": "Neg Risk Event",
                "negRisk": True,
                "enableNegRisk": True,
                "negRiskAugmented": False,
                "negRiskMarketID": "nr-1",
                "markets": [
                    {
                        "slug": "named-a",
                        "question": "Named A?",
                        "groupItemTitle": "Named A",
                        "negRisk": True,
                        "negRiskOther": False,
                        "feesEnabled": False,
                        "enableOrderBook": True,
                        "outcomes": '["Yes","No"]',
                        "clobTokenIds": '["tok-a-yes","tok-a-no"]',
                    },
                    {
                        "slug": "named-b",
                        "question": "Named B?",
                        "groupItemTitle": "Named B",
                        "negRisk": True,
                        "negRiskOther": False,
                        "feesEnabled": False,
                        "enableOrderBook": True,
                        "outcomes": '["Yes","No"]',
                        "clobTokenIds": '["tok-b-yes","tok-b-no"]',
                    },
                ],
            }
        ]

        groups = build_eligible_neg_risk_event_groups(events)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["event_slug"], "neg-risk-event")
        self.assertEqual([market["slug"] for market in groups[0]["markets"]], ["named-a", "named-b"])

    def test_grouping_excludes_augmented_other_and_fee_enabled_cases(self) -> None:
        events = [
            {
                "id": "evt-aug",
                "slug": "augmented",
                "title": "Augmented",
                "negRisk": True,
                "enableNegRisk": True,
                "negRiskAugmented": True,
                "markets": [],
            },
            {
                "id": "evt-2",
                "slug": "mixed",
                "title": "Mixed",
                "negRisk": True,
                "enableNegRisk": True,
                "negRiskAugmented": False,
                "markets": [
                    {
                        "slug": "other-leg",
                        "question": "Other?",
                        "groupItemTitle": "Other",
                        "negRisk": True,
                        "negRiskOther": True,
                        "feesEnabled": False,
                        "enableOrderBook": True,
                        "outcomes": '["Yes","No"]',
                        "clobTokenIds": '["tok-o-yes","tok-o-no"]',
                    },
                    {
                        "slug": "fee-leg",
                        "question": "Fee leg?",
                        "groupItemTitle": "Fee Leg",
                        "negRisk": True,
                        "negRiskOther": False,
                        "feesEnabled": True,
                        "enableOrderBook": True,
                        "outcomes": '["Yes","No"]',
                        "clobTokenIds": '["tok-f-yes","tok-f-no"]',
                    },
                    {
                        "slug": "named-ok",
                        "question": "Named ok?",
                        "groupItemTitle": "Named Ok",
                        "negRisk": True,
                        "negRiskOther": False,
                        "feesEnabled": False,
                        "enableOrderBook": True,
                        "outcomes": '["Yes","No"]',
                        "clobTokenIds": '["tok-ok-yes","tok-ok-no"]',
                    },
                ],
            },
        ]

        groups = build_eligible_neg_risk_event_groups(events)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["event_slug"], "mixed")
        self.assertEqual([market["slug"] for market in groups[0]["markets"]], ["named-ok"])

    def test_raw_candidate_created_on_positive_displayed_maker_first_edge(self) -> None:
        event_group = {
            "event_id": "evt-1",
            "event_slug": "neg-risk-event",
            "event_title": "Neg Risk Event",
            "neg_risk_market_id": "nr-1",
            "markets": [
                {"slug": "named-a", "question": "Named A?", "group_item_title": "Named A", "yes_token_id": "tok-a-yes"},
                {"slug": "named-b", "question": "Named B?", "group_item_title": "Named B", "yes_token_id": "tok-b-yes"},
            ],
        }
        books_by_token = {
            "tok-a-yes": Book(bids=[Level(0.40, 20.0)], asks=[Level(0.42, 20.0)]),
            "tok-b-yes": Book(bids=[Level(0.35, 20.0)], asks=[Level(0.37, 20.0)]),
        }

        raw_candidate, audit = self.strategy.detect_with_audit(
            event_group,
            books_by_token,
            max_notional=10.0,
        )

        self.assertIsNone(audit)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.strategy_family.value, "neg_risk_rebalancing")
        self.assertEqual(raw_candidate.market_slugs, ["named-a", "named-b"])
        self.assertAlmostEqual(raw_candidate.gross_edge_cents, 0.25)
        self.assertTrue(raw_candidate.metadata["maker_first"])

    def test_no_candidate_on_non_positive_edge(self) -> None:
        event_group = {
            "event_id": "evt-1",
            "event_slug": "neg-risk-event",
            "event_title": "Neg Risk Event",
            "neg_risk_market_id": "nr-1",
            "markets": [
                {"slug": "named-a", "question": "Named A?", "group_item_title": "Named A", "yes_token_id": "tok-a-yes"},
                {"slug": "named-b", "question": "Named B?", "group_item_title": "Named B", "yes_token_id": "tok-b-yes"},
            ],
        }
        books_by_token = {
            "tok-a-yes": Book(bids=[Level(0.60, 20.0)], asks=[Level(0.62, 20.0)]),
            "tok-b-yes": Book(bids=[Level(0.40, 20.0)], asks=[Level(0.42, 20.0)]),
        }

        raw_candidate, audit = self.strategy.detect_with_audit(
            event_group,
            books_by_token,
            max_notional=10.0,
        )

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], NEG_RISK_BASKET_EDGE_NON_POSITIVE)
        self.assertEqual(audit["strategy_family"], "neg_risk_rebalancing")


if __name__ == "__main__":
    unittest.main()
