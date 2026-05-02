from __future__ import annotations

import unittest

from scripts.analyze_research_profit_drivers import analyze_profit_drivers, build_markdown_report


class ResearchProfitDriverTests(unittest.TestCase):
    def test_cycle_summary_drives_latest_blocker_report(self) -> None:
        report = analyze_profit_drivers(
            [
                {
                    "row_type": "cycle_summary",
                    "cycle_index": 3,
                    "selected_market_slugs": ["m1"],
                    "selected_market_count": 1,
                    "active_quote_market_count": 1,
                    "eligible_candidate_count": 4,
                    "last_selection_reasons": {"SELECT_PER_MARKET_CAP": 2, "SELECTED": 1},
                    "last_filter_reasons": {"REWARD_MINUS_DRAWDOWN": 99},
                    "scan_diagnostics": {
                        "selection_blocked_candidates": [
                            {
                                "market_slug": "blocked-big",
                                "reason": "SELECT_PER_MARKET_CAP",
                                "capital_basis_usdc": 75.0,
                                "total_score": 0.8,
                            }
                        ]
                    },
                    "verified_net_after_reward_and_cost_usdc": 0.12,
                    "net_after_reward_and_cost_usdc": 0.15,
                    "reward_accrued_estimate_usdc": 0.03,
                    "reward_accrued_actual_usdc": 0.0,
                    "spread_realized_usdc": 0.13,
                    "cost_proxy_usdc": 0.01,
                },
                {
                    "row_type": "market_observation",
                    "cycle_index": 3,
                    "market_slug": "m1",
                    "status": "QUOTING",
                    "action": "PLACE_BID",
                    "verified_net_window_usdc": 0.12,
                    "fill_rate_window": 0.4,
                    "spread_realized_usdc": 0.13,
                    "simulated_fill": True,
                },
            ]
        )

        self.assertEqual(report["latest_cycle"], 3)
        self.assertEqual(report["latest_selected_markets"], 1)
        self.assertEqual(report["latest_filter_reasons"]["REWARD_MINUS_DRAWDOWN"], 99)
        self.assertEqual(report["latest_selection_blockers"][0]["market_slug"], "blocked-big")
        self.assertEqual(report["top_profit_drivers"][0]["profit_quality"], "SIMULATED_ONLY")
        self.assertEqual(report["top_profit_drivers"][0]["recommended_bucket"], "DRY_RUN_FOCUS")
        self.assertTrue(any("ranking only" in action for action in report["strategy_actions"]))
        self.assertTrue(any("higher dry-run-only cap" in action for action in report["strategy_actions"]))

    def test_negative_or_risk_rejected_market_is_avoid(self) -> None:
        report = analyze_profit_drivers(
            [
                {
                    "row_type": "market_observation",
                    "cycle_index": 1,
                    "market_slug": "bad",
                    "verified_net_window_usdc": -0.03,
                    "risk_reject_reason": "RISK_CAP_BELOW_REWARD_MIN_SIZE",
                }
            ]
        )

        self.assertEqual(report["avoid"][0]["market_slug"], "bad")
        self.assertEqual(report["avoid"][0]["recommended_bucket"], "AVOID")

    def test_markdown_report_contains_strategy_actions(self) -> None:
        report = analyze_profit_drivers(
            [
                {
                    "row_type": "market_observation",
                    "cycle_index": 1,
                    "market_slug": "m1",
                    "verified_net_window_usdc": 0.01,
                    "fill_rate_window": 0.2,
                    "simulated_fill": True,
                }
            ]
        )

        markdown = build_markdown_report(report)

        self.assertIn("# Research Profit Drivers", markdown)
        self.assertIn("Top Profit Drivers", markdown)


if __name__ == "__main__":
    unittest.main()
