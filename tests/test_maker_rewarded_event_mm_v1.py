from __future__ import annotations

import unittest

from src.scanner.maker_rewarded_mm import (
    MAKER_MM_NON_POSITIVE_EV,
    build_eligible_rewarded_market_groups,
)
from src.strategies.opportunity_strategies import MakerRewardedEventMMStrategy


class MakerRewardedEventMMStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = MakerRewardedEventMMStrategy()

    def test_eligible_market_filtering_keeps_only_rewarded_complete_fee_free_markets(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "event-one",
                    "title": "Event One",
                    "markets": [
                        {
                            "slug": "good-market",
                            "question": "Good?",
                            "enable_orderbook": True,
                            "best_bid": 0.45,
                            "best_ask": 0.47,
                            "fees_enabled": False,
                            "is_binary_yes_no": True,
                            "yes_token_id": "yes-a",
                            "no_token_id": "no-a",
                            "rewards_min_size": 20,
                            "rewards_max_spread": 3.5,
                            "clob_rewards": [{"rewardsDailyRate": 2.0}],
                        },
                        {
                            "slug": "bad-market",
                            "question": "Bad?",
                            "enable_orderbook": True,
                            "best_bid": 0.45,
                            "best_ask": 0.47,
                            "fees_enabled": True,
                            "is_binary_yes_no": True,
                            "yes_token_id": "yes-b",
                            "no_token_id": "no-b",
                            "rewards_min_size": 20,
                            "rewards_max_spread": 3.5,
                            "clob_rewards": [{"rewardsDailyRate": 2.0}],
                        },
                    ],
                }
            ]
        }
        watchlist_report = {
            "top_events": [
                {
                    "event_slug": "event-one",
                    "watchlist_score": 10,
                    "top_markets": [
                        {"market_slug": "good-market", "activity_events": 3, "spread_changes": 1, "liquidity_changes": 3, "stability_score": 3, "watchlist_score": 8},
                        {"market_slug": "bad-market", "activity_events": 3, "spread_changes": 1, "liquidity_changes": 3, "stability_score": 3, "watchlist_score": 8},
                    ],
                }
            ]
        }

        groups = build_eligible_rewarded_market_groups(registry=registry, watchlist_report=watchlist_report)

        self.assertEqual(len(groups), 1)
        self.assertEqual([market["market_slug"] for market in groups[0]["markets"]], ["good-market"])

    def test_reward_metadata_is_required(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "event-one",
                    "title": "Event One",
                    "markets": [
                        {
                            "slug": "no-reward-market",
                            "question": "No reward?",
                            "enable_orderbook": True,
                            "best_bid": 0.45,
                            "best_ask": 0.47,
                            "fees_enabled": False,
                            "is_binary_yes_no": True,
                            "yes_token_id": "yes-a",
                            "no_token_id": "no-a",
                            "rewards_min_size": None,
                            "rewards_max_spread": 3.5,
                            "clob_rewards": [],
                        },
                    ],
                }
            ]
        }
        watchlist_report = {
            "top_events": [
                {
                    "event_slug": "event-one",
                    "watchlist_score": 10,
                    "top_markets": [
                        {"market_slug": "no-reward-market", "activity_events": 3, "spread_changes": 1, "liquidity_changes": 3, "stability_score": 3, "watchlist_score": 8},
                    ],
                }
            ]
        }

        groups = build_eligible_rewarded_market_groups(registry=registry, watchlist_report=watchlist_report)
        self.assertEqual(groups, [])

    def test_maker_quote_simulation_creates_candidate_on_stable_books(self) -> None:
        event_group = {
            "event_slug": "event-one",
            "event_title": "Event One",
            "event_watchlist_score": 12.0,
        }
        market = {
            "market_slug": "good-market",
            "question": "Good?",
            "yes_token_id": "yes-a",
            "no_token_id": "no-a",
            "best_bid": 0.45,
            "best_ask": 0.49,
            "rewards_min_size": 20.0,
            "rewards_max_spread": 3.5,
            "reward_daily_rate": 2.0,
            "activity_events": 8,
            "spread_changes": 2,
            "liquidity_changes": 8,
            "stability_score": 8,
            "watchlist_score": 12.0,
        }

        raw_candidate, audit = self.strategy.detect_with_audit(event_group, market)

        self.assertIsNone(audit)
        self.assertIsNotNone(raw_candidate)
        assert raw_candidate is not None
        self.assertEqual(raw_candidate.strategy_family.value, "maker_rewarded_event_mm_v1")
        self.assertGreater(float(raw_candidate.metadata["spread_capture_ev"]), 0.0)
        self.assertGreater(float(raw_candidate.metadata["liquidity_reward_ev"]), 0.0)
        self.assertGreater(float(raw_candidate.metadata["total_ev"]), 0.0)

    def test_no_output_on_ineligible_or_non_positive_ev_market(self) -> None:
        event_group = {
            "event_slug": "event-one",
            "event_title": "Event One",
            "event_watchlist_score": 12.0,
        }
        market = {
            "market_slug": "bad-market",
            "question": "Bad?",
            "yes_token_id": "yes-a",
            "no_token_id": "no-a",
            "best_bid": 0.45,
            "best_ask": 0.46,
            "rewards_min_size": 20.0,
            "rewards_max_spread": 0.1,
            "reward_daily_rate": 0.01,
            "activity_events": 1,
            "spread_changes": 1,
            "liquidity_changes": 0,
            "stability_score": 1,
            "watchlist_score": 1.0,
        }

        raw_candidate, audit = self.strategy.detect_with_audit(event_group, market)

        self.assertIsNone(raw_candidate)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit["failure_reason"], MAKER_MM_NON_POSITIVE_EV)


if __name__ == "__main__":
    unittest.main()
