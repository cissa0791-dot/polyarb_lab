from __future__ import annotations

import sys
import unittest

from scripts.run_autonomous_project_manager import build_autonomous_decision


class AutonomousProjectManagerTests(unittest.TestCase):
    def test_simulated_only_blockers_keep_research_mode(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_DRY_RUN_FOCUS",
                "live_canary_eligible_count": 0,
                "dry_run_focus_count": 3,
                "live_ready_blockers": ["SIMULATED_PROFIT_ONLY"],
            },
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "RUN_RESEARCH")
        self.assertFalse(decision["can_execute_live"])
        self.assertEqual(decision["live_command"], [])

    def test_confirmed_candidate_starts_one_market_canary_with_hard_caps(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 0,
                "live_ready_blockers": [],
            },
            live_pnl={},
            max_live_risk_usdc=40.0,
            live_cycles=12,
            live_interval_sec=5,
        )

        command = decision["live_command"]
        self.assertEqual(decision["decision"], "START_LIVE_CANARY")
        self.assertTrue(decision["can_execute_live"])
        self.assertEqual(decision["target_live_markets"], 1)
        self.assertEqual(decision["max_live_risk_usdc"], 20.0)
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--live", command)
        self.assertEqual(command[command.index("--max-markets") + 1], "1")
        self.assertEqual(command[command.index("--capital") + 1], "20.00")
        self.assertEqual(command[command.index("--max-total-open-buy-usdc") + 1], "20.00")
        self.assertEqual(command[command.index("--max-order-rejects-per-hour") + 1], "1")

    def test_two_market_scale_requires_existing_healthy_live_canary(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_SCALE_TO_2_MARKETS",
                "live_canary_eligible_count": 2,
                "dry_run_focus_count": 0,
                "live_ready_blockers": [],
            },
            live_pnl={
                "summary": {
                    "mode": "LIVE",
                    "verified_net_after_reward_and_cost_usdc": 0.05,
                    "account_inventory_usdc": 10.0,
                    "account_open_buy_usdc": 0.0,
                },
                "markets": [{"order_reject_count": 0}],
            },
            max_live_risk_usdc=40.0,
        )

        command = decision["live_command"]
        self.assertEqual(decision["decision"], "SCALE_LIVE_TO_2_MARKETS")
        self.assertEqual(decision["target_live_markets"], 2)
        self.assertEqual(decision["max_live_risk_usdc"], 40.0)
        self.assertEqual(command[command.index("--max-markets") + 1], "2")
        self.assertEqual(command[command.index("--capital") + 1], "40.00")

    def test_live_order_reject_forces_pause(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 0,
                "live_ready_blockers": [],
            },
            live_pnl={
                "summary": {
                    "mode": "LIVE",
                    "verified_net_after_reward_and_cost_usdc": 0.05,
                    "account_inventory_usdc": 0.0,
                    "account_open_buy_usdc": 0.0,
                },
                "markets": [{"order_reject_count": 1}],
            },
        )

        self.assertEqual(decision["decision"], "PAUSE_LIVE")
        self.assertEqual(decision["reason"], "ORDER_REJECT_STOP")
        self.assertFalse(decision["can_execute_live"])

    def test_human_approval_required_above_risk_limit(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 0,
                "live_ready_blockers": [],
            },
            live_pnl={},
            max_live_risk_usdc=51.0,
        )

        self.assertEqual(decision["decision"], "HOLD_HUMAN_APPROVAL_REQUIRED")
        self.assertTrue(decision["requires_human_approval"])
        self.assertFalse(decision["can_execute_live"])

    def test_contaminated_research_blocks_live(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 0,
                "live_ready_blockers": [],
                "contaminated_state_detected": True,
            },
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "HOLD_NO_LIVE")
        self.assertFalse(decision["can_execute_live"])


if __name__ == "__main__":
    unittest.main()
