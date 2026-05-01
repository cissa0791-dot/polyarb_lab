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

    def test_tiny_negative_no_fill_is_no_evidence_not_blacklist(self) -> None:
        rows = [
            {
                "row_type": "market_observation",
                "market_slug": "quiet",
                "event_slug": "event-quiet",
                "token_id": "tok-quiet",
                "actual_reward_usdc": 0.0,
                "spread_realized_usdc": 0.0,
                "verified_net_window_usdc": -0.000052,
                "fill_rate_window": 0.0,
                "order_reject_count": 0,
                "status": "PAUSED",
                "last_cancel_reason": "NOT_SELECTED",
            }
            for _ in range(3)
        ]

        summary = build_live_edge_summary(rows)

        self.assertEqual(summary["blacklist_candidates"], [])
        self.assertEqual(summary["market_intel"]["markets"]["quiet"]["evidence_status"], "NO_EVIDENCE")
        self.assertFalse(summary["market_intel"]["markets"]["quiet"]["blocked"])
        self.assertEqual(summary["scale_recommendation"], "ALLOW_DRY_RUN_FOCUS")

    def test_simulated_positive_edge_is_not_whitelisted(self) -> None:
        rows = [
            {
                "row_type": "market_observation",
                "market_slug": "sim-only",
                "event_slug": "event-sim",
                "token_id": "tok-sim",
                "actual_reward_usdc": 0.0,
                "spread_realized_usdc": 0.05,
                "simulated_spread_usdc": 0.05,
                "verified_net_window_usdc": 0.05,
                "fill_rate_window": 0.5,
                "order_reject_count": 0,
                "evidence_source": "DRY_RUN_SIMULATED",
                "simulated_fill": True,
            }
        ]

        summary = build_live_edge_summary(rows)
        market = summary["market_intel"]["markets"]["sim-only"]

        self.assertEqual(summary["profitable_market_count"], 0)
        self.assertEqual(summary["simulated_profitable_market_count"], 1)
        self.assertEqual(summary["whitelist_candidates"], [])
        self.assertEqual(market["evidence_status"], "NO_EVIDENCE")
        self.assertIn("SIMULATED_PROFIT_ONLY", market["evidence_reasons"])
        self.assertEqual(market["realized_spread_window_usdc"], 0.0)
        self.assertEqual(market["simulated_spread_window_usdc"], 0.05)
        self.assertEqual(summary["scale_recommendation"], "ALLOW_DRY_RUN_FOCUS")


if __name__ == "__main__":
    unittest.main()
