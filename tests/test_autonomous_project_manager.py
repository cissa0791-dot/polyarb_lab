from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_autonomous_project_manager import build_autonomous_decision, latest_live_probe_pnl


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

    def test_replay_negative_blocks_live_canary(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 2,
                "live_ready_blockers": [],
            },
            profit_drivers={"live_canary_candidates": [{"profit_quality": "CONFIRMED"}]},
            replay_report={"replayed_market_count": 20, "total_net_pnl_usdc": -0.32},
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "RUN_RESEARCH")
        self.assertIn("REPLAY_NET_NEGATIVE", decision["reason"])
        self.assertFalse(decision["can_execute_live"])

    def test_simulated_positive_edge_allows_capped_micro_live_probe(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_DRY_RUN_FOCUS",
                "live_canary_eligible_count": 0,
                "dry_run_focus_count": 6,
                "actual_reward_confirmed_market_count": 0,
                "live_ready_blockers": [
                    "NO_LIVE_CANARY_ELIGIBLE",
                    "SIMULATED_PROFIT_ONLY",
                    "REPLAY_NOT_CONFIRMED",
                    "NO_ACTUAL_REWARD_CONFIRMED",
                ],
            },
            profit_drivers={
                "top_profit_drivers": [
                    {
                        "market_slug": "probe-me",
                        "verified_net_usdc": 0.2,
                        "profit_quality": "SIMULATED_ONLY",
                    },
                    {
                        "market_slug": "probe-me-too",
                        "verified_net_usdc": 0.1,
                        "profit_quality": "SIMULATED_ONLY",
                    },
                ]
            },
            replay_report={"replayed_market_count": 91, "total_net_pnl_usdc": -0.0775},
            live_pnl={},
            max_live_risk_usdc=20.0,
            live_cycles=12,
            live_interval_sec=5,
        )

        command = decision["live_command"]
        self.assertEqual(decision["decision"], "START_MICRO_LIVE_PROBE")
        self.assertTrue(decision["can_execute_live"])
        self.assertEqual(decision["target_live_markets"], 1)
        self.assertEqual(decision["max_live_risk_usdc"], 20.0)
        self.assertIn("--live", command)
        self.assertEqual(command[command.index("--capital") + 1], "20.00")
        self.assertEqual(command[command.index("--max-markets") + 1], "1")
        self.assertEqual(command[command.index("--inventory-policy") + 1], "auto")
        self.assertIn("--reset-state", command)
        self.assertIn("--state-path", command)
        self.assertIn("live_probe_runs", command[command.index("--state-path") + 1])
        self.assertIn("--pnl-path", command)
        self.assertIn("live_probe_runs", command[command.index("--pnl-path") + 1])
        self.assertTrue(decision["fresh_live_state"])

    def test_live_rate_limit_forces_pause(self) -> None:
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
                    "account_order_sync_error": "Cloudflare error code: 1015 You are being rate limited",
                },
                "markets": [],
            },
        )

        self.assertEqual(decision["decision"], "PAUSE_LIVE")
        self.assertEqual(decision["reason"], "CLOB_RATE_LIMIT")
        self.assertFalse(decision["can_execute_live"])

    def test_old_live_rate_limit_expires_after_cooldown(self) -> None:
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
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
                    "generated_ts": old_ts,
                    "verified_net_after_reward_and_cost_usdc": 0.05,
                    "account_inventory_usdc": 0.0,
                    "account_open_buy_usdc": 0.0,
                    "account_order_sync_error": "Cloudflare error code: 1015 You are being rate limited",
                },
                "markets": [],
            },
        )

        self.assertNotEqual(decision["decision"], "PAUSE_LIVE")
        self.assertFalse(decision["inputs"]["live_health"]["rate_limit_cooldown_active"])

    def test_latest_live_probe_pnl_picks_newest_probe_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "probe-a" / "auto_trade_pnl.json"
            newer = root / "probe-b" / "auto_trade_pnl.json"
            older.parent.mkdir()
            newer.parent.mkdir()
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            os.utime(older, (100.0, 100.0))
            os.utime(newer, (200.0, 200.0))

            self.assertEqual(latest_live_probe_pnl(root), newer)

    def test_severely_negative_replay_blocks_micro_live_probe(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_DRY_RUN_FOCUS",
                "live_canary_eligible_count": 0,
                "dry_run_focus_count": 6,
                "live_ready_blockers": [
                    "NO_LIVE_CANARY_ELIGIBLE",
                    "SIMULATED_PROFIT_ONLY",
                    "REPLAY_NOT_CONFIRMED",
                    "NO_ACTUAL_REWARD_CONFIRMED",
                ],
            },
            profit_drivers={
                "top_profit_drivers": [
                    {"market_slug": "too-risky", "verified_net_usdc": 0.2, "profit_quality": "SIMULATED_ONLY"}
                ]
            },
            replay_report={"replayed_market_count": 20, "total_net_pnl_usdc": -0.32},
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "RUN_RESEARCH")
        self.assertFalse(decision["can_execute_live"])

    def test_profit_driver_run_mismatch_blocks_live_canary(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "run_id": "research-new",
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 0,
                "live_ready_blockers": [],
            },
            profit_drivers={
                "run_id": "research-old",
                "live_canary_candidates": [{"profit_quality": "CONFIRMED"}],
            },
            replay_report={"replayed_market_count": 20, "total_net_pnl_usdc": 0.0},
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "RUN_RESEARCH")
        self.assertIn("PROFIT_DRIVER_RUN_MISMATCH", decision["reason"])

    def test_simulated_profit_driver_blocks_live_canary(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_CANARY_ONLY",
                "live_canary_eligible_count": 1,
                "dry_run_focus_count": 2,
                "live_ready_blockers": [],
            },
            profit_drivers={
                "top_profit_drivers": [
                    {
                        "market_slug": "sim-only",
                        "verified_net_usdc": 0.2,
                        "profit_quality": "SIMULATED_ONLY",
                    }
                ],
                "live_canary_candidates": [],
            },
            replay_report={"replayed_market_count": 20, "total_net_pnl_usdc": 0.0},
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "RUN_RESEARCH")
        self.assertIn("SIMULATED_ONLY_PROFIT_DRIVER", decision["reason"])

    def test_single_market_positive_recommends_broader_dry_run_only(self) -> None:
        decision = build_autonomous_decision(
            research_summary={
                "scale_recommendation": "ALLOW_DRY_RUN_FOCUS",
                "live_canary_eligible_count": 0,
                "dry_run_focus_count": 5,
                "profitable_market_count": 1,
                "verified_net_total": 0.2,
                "live_ready_blockers": ["SIMULATED_PROFIT_ONLY"],
            },
            profit_drivers={"top_profit_drivers": [{"verified_net_usdc": 0.2, "profit_quality": "SIMULATED_ONLY"}]},
            replay_report={"replayed_market_count": 20, "total_net_pnl_usdc": 0.0},
            live_pnl={},
        )

        self.assertEqual(decision["decision"], "RUN_RESEARCH")
        self.assertEqual(decision["research_policy"]["next_research_mode"], "BROADEN_DRY_RUN_ONLY")
        self.assertEqual(decision["research_policy"]["recommended_dry_run_per_market_cap_usdc"], 80.0)

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
            profit_drivers={"live_canary_candidates": [{"profit_quality": "CONFIRMED"}, {"profit_quality": "CONFIRMED"}]},
            replay_report={"replayed_market_count": 20, "total_net_pnl_usdc": 0.0},
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
