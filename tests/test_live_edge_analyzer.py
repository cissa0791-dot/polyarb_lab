from __future__ import annotations

import unittest

from scripts.analyze_live_edge_evidence import build_live_edge_summary


class LiveEdgeAnalyzerTests(unittest.TestCase):
    def test_marks_whitelist_blacklist_and_scale_recommendation(self) -> None:
        rows = [
            {
                "row_type": "market_observation",
                "market_slug": "good",
                "event_slug": "event-good",
                "token_id": "tok-good",
                "actual_reward_usdc": 0.02,
                "spread_realized_usdc": 0.01,
                "verified_net_window_usdc": 0.03,
                "fill_rate_window": 0.5,
                "order_reject_count": 0,
            },
            {
                "row_type": "market_observation",
                "market_slug": "bad",
                "event_slug": "event-bad",
                "token_id": "tok-bad",
                "actual_reward_usdc": 0.0,
                "spread_realized_usdc": 0.0,
                "verified_net_window_usdc": -0.06,
                "fill_rate_window": 0.0,
                "order_reject_count": 1,
            },
        ]

        summary = build_live_edge_summary(rows)

        self.assertEqual(summary["market_count"], 2)
        self.assertEqual(summary["profitable_market_count"], 1)
        self.assertEqual(summary["scale_recommendation"], "DO_NOT_SCALE")
        self.assertEqual(summary["whitelist_candidates"][0]["market_slug"], "good")
        self.assertEqual(summary["blacklist_candidates"][0]["market_slug"], "bad")
        self.assertTrue(summary["market_intel"]["markets"]["bad"]["blocked"])
        self.assertTrue(summary["market_intel"]["markets"]["good"]["allow"])


if __name__ == "__main__":
    unittest.main()
