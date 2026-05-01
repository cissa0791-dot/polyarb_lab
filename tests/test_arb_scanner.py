from __future__ import annotations

import unittest

from src.scanner.arb_scanner import scan_arb_opportunities


class ArbScannerTests(unittest.TestCase):
    def test_detects_neg_risk_yes_basket_under_one(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "winner",
                    "title": "Winner",
                    "neg_risk": True,
                    "markets": [
                        {
                            "slug": "a",
                            "question": "A wins?",
                            "yes_token_id": "tok-a",
                            "active": True,
                            "closed": False,
                            "enable_orderbook": True,
                            "best_bid": 0.30,
                            "best_ask": 0.32,
                        },
                        {
                            "slug": "b",
                            "question": "B wins?",
                            "yes_token_id": "tok-b",
                            "active": True,
                            "closed": False,
                            "enable_orderbook": True,
                            "best_bid": 0.31,
                            "best_ask": 0.33,
                        },
                        {
                            "slug": "c",
                            "question": "C wins?",
                            "yes_token_id": "tok-c",
                            "active": True,
                            "closed": False,
                            "enable_orderbook": True,
                            "best_bid": 0.30,
                            "best_ask": 0.31,
                        },
                    ],
                }
            ]
        }

        report = scan_arb_opportunities(registry, min_edge=0.01)

        self.assertEqual(report["candidate_count"], 1)
        opportunity = report["opportunities"][0]
        self.assertEqual(opportunity["status"], "ARB_CANDIDATE")
        self.assertEqual(opportunity["kind"], "event_yes_basket_under_one")
        self.assertGreater(opportunity["executable_edge"], 0.01)
        self.assertEqual([leg["side"] for leg in opportunity["required_legs"]], ["BUY", "BUY", "BUY"])

    def test_non_neg_risk_under_one_is_watch_not_candidate(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "loose-group",
                    "title": "Election Winner",
                    "neg_risk": False,
                    "markets": [
                        {"slug": "a", "best_bid": 0.20, "best_ask": 0.21, "active": True, "closed": False},
                        {"slug": "b", "best_bid": 0.22, "best_ask": 0.23, "active": True, "closed": False},
                    ],
                }
            ]
        }

        report = scan_arb_opportunities(registry, min_edge=0.01)

        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["watch_count"], 1)
        self.assertEqual(report["opportunities"][0]["resolution_mismatch_risk"], "medium")

    def test_skips_non_exclusive_top_four_over_one(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "epl-top-4-finish",
                    "title": "Premier League Top 4 Finish",
                    "neg_risk": False,
                    "markets": [
                        {
                            "slug": "will-arsenal-finish-in-the-top-4",
                            "question": "Will Arsenal finish in the top 4?",
                            "best_bid": 0.80,
                            "best_ask": 0.82,
                            "active": True,
                            "closed": False,
                        },
                        {
                            "slug": "will-liverpool-finish-in-the-top-4",
                            "question": "Will Liverpool finish in the top 4?",
                            "best_bid": 0.70,
                            "best_ask": 0.72,
                            "active": True,
                            "closed": False,
                        },
                    ],
                }
            ]
        }

        report = scan_arb_opportunities(registry, min_edge=0.01)

        self.assertEqual(report["opportunity_count"], 0)
        self.assertEqual(report["skip_reasons"]["NON_EXCLUSIVE_EVENT"], 1)

    def test_skips_non_neg_risk_over_one_requiring_short(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "unknown-basket",
                    "title": "Unknown Basket",
                    "neg_risk": False,
                    "markets": [
                        {"slug": "a", "best_bid": 0.60, "best_ask": 0.62, "active": True, "closed": False},
                        {"slug": "b", "best_bid": 0.60, "best_ask": 0.62, "active": True, "closed": False},
                    ],
                }
            ]
        }

        report = scan_arb_opportunities(registry, min_edge=0.01)

        self.assertEqual(report["opportunity_count"], 0)
        self.assertEqual(report["skip_reasons"]["OVER_ONE_NEEDS_SHORT_OR_NEG_RISK"], 1)

    def test_large_leg_count_is_watch_not_candidate(self) -> None:
        registry = {
            "events": [
                {
                    "slug": "large-winner-market",
                    "title": "Election Winner",
                    "neg_risk": True,
                    "markets": [
                        {
                            "slug": f"candidate-{idx}",
                            "question": f"Candidate {idx} wins?",
                            "yes_token_id": f"tok-{idx}",
                            "active": True,
                            "closed": False,
                            "enable_orderbook": True,
                            "best_bid": 0.08,
                            "best_ask": 0.09,
                        }
                        for idx in range(7)
                    ],
                }
            ]
        }

        report = scan_arb_opportunities(registry, min_edge=0.02)

        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["watch_count"], 1)
        self.assertIn("candidate_block=LEG_COUNT_GT_6", report["opportunities"][0]["decision_trace"])


if __name__ == "__main__":
    unittest.main()
