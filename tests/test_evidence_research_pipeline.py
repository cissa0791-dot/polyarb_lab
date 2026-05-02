from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from scripts.analyze_live_edge_evidence import build_live_edge_summary, load_jsonl as load_evidence_jsonl
from scripts.replay_live_orderbook_snapshots import build_replay_report, load_jsonl as load_snapshot_jsonl
from scripts.run_evidence_research_pipeline import (
    PipelineInterrupted,
    _scan_command,
    build_research_outputs,
    parse_args,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]


class EvidenceResearchPipelineTests(unittest.TestCase):
    def _fake_research_runner(self) -> tuple[Callable[[list[str]], None], list[list[str]]]:
        commands: list[list[str]] = []

        def fake_runner(command: list[str]) -> None:
            commands.append(command)
            joined = " ".join(command)
            if "run_auto_trade_profit.py" in joined:
                evidence_path = Path(command[command.index("--edge-evidence-path") + 1])
                snapshot_path = Path(command[command.index("--orderbook-snapshot-path") + 1])
                evidence_path.write_text(
                    json.dumps(
                        {
                            "row_type": "market_observation",
                            "market_slug": "m1",
                            "event_slug": "e1",
                            "token_id": "tok-m1",
                            "actual_reward_usdc": 0.0,
                            "spread_realized_usdc": 0.0,
                            "verified_net_window_usdc": -0.00001,
                            "fill_rate_window": 0.0,
                            "order_reject_count": 0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                snapshot_path.write_text("", encoding="utf-8")
                return
            if "analyze_live_edge_evidence.py" in joined:
                evidence_path = Path(command[command.index("--evidence") + 1])
                out_path = Path(command[command.index("--out") + 1])
                intel_path = Path(command[command.index("--market-intel-out") + 1])
                summary = build_live_edge_summary(load_evidence_jsonl(str(evidence_path)))
                out_path.write_text(json.dumps(summary), encoding="utf-8")
                intel_path.write_text(json.dumps(summary["market_intel"]), encoding="utf-8")
                return
            if "replay_live_orderbook_snapshots.py" in joined:
                snapshot_path = Path(command[command.index("--snapshots") + 1])
                out_path = Path(command[command.index("--out") + 1])
                replay = build_replay_report(load_snapshot_jsonl(str(snapshot_path)))
                out_path.write_text(json.dumps(replay), encoding="utf-8")
                return
            raise AssertionError(f"unexpected command: {command}")

        return fake_runner, commands

    def test_research_merger_outputs_allowed_actions(self) -> None:
        evidence_summary = build_live_edge_summary(
            [
                {
                    "row_type": "market_observation",
                    "market_slug": "m1",
                    "verified_net_window_usdc": 0.02,
                    "actual_reward_usdc": 0.01,
                    "spread_realized_usdc": 0.0,
                    "fill_rate_window": 0.5,
                }
            ]
        )
        replay_report = {
            "markets": [
                {
                    "market_slug": "m1",
                    "suitability": "REWARD_MM_CANDIDATE",
                    "net_pnl_usdc": 0.03,
                    "simulated_fill_count": 1,
                }
            ],
            "replayed_market_count": 1,
        }

        outputs = build_research_outputs(evidence_summary=evidence_summary, replay_report=replay_report, arb_rows=[])

        row = outputs["market_intel"]["by_market"]["m1"]
        self.assertEqual(row["recommended_action"], "LIVE_CANARY_ELIGIBLE")
        self.assertEqual(outputs["whitelist"]["markets"][0]["market_slug"], "m1")

    def test_scale_recommendation_clamps_when_no_live_canary_candidates(self) -> None:
        evidence_summary = {
            "market_intel": {"markets": {"arb-only": {"evidence_status": "NO_EVIDENCE"}}},
            "scale_recommendation": "ALLOW_SCALE_TO_2_MARKETS",
            "profitable_market_count": 3,
        }
        replay_report = {"markets": [], "replayed_market_count": 0}
        arb_rows = [{"status": "ARB_CANDIDATE", "required_legs": [{"market_slug": "arb-only"}]}]

        outputs = build_research_outputs(
            evidence_summary=evidence_summary,
            replay_report=replay_report,
            arb_rows=arb_rows,
        )

        self.assertEqual(outputs["summary"]["live_canary_eligible_count"], 0)
        self.assertEqual(outputs["summary"]["dry_run_focus_count"], 1)
        self.assertEqual(outputs["summary"]["scale_recommendation"], "ALLOW_DRY_RUN_FOCUS")
        self.assertEqual(outputs["market_intel"]["by_market"]["arb-only"]["recommended_action"], "DRY_RUN_FOCUS")

    def test_scale_recommendation_clamps_single_live_candidate_to_canary(self) -> None:
        evidence_summary = {
            "market_intel": {"markets": {"m1": {"evidence_status": "WHITELIST_CANDIDATE"}}},
            "scale_recommendation": "ALLOW_SCALE_TO_2_MARKETS",
            "profitable_market_count": 3,
        }
        replay_report = {
            "markets": [{"market_slug": "m1", "suitability": "REWARD_MM_CANDIDATE"}],
            "replayed_market_count": 1,
        }

        outputs = build_research_outputs(
            evidence_summary=evidence_summary,
            replay_report=replay_report,
            arb_rows=[],
        )

        self.assertEqual(outputs["summary"]["live_canary_eligible_count"], 1)
        self.assertEqual(outputs["summary"]["scale_recommendation"], "ALLOW_CANARY_ONLY")

    def test_simulated_only_positive_edge_stays_dry_run_focus(self) -> None:
        evidence_summary = build_live_edge_summary(
            [
                {
                    "row_type": "market_observation",
                    "market_slug": "sim-only",
                    "verified_net_window_usdc": 0.05,
                    "actual_reward_usdc": 0.0,
                    "spread_realized_usdc": 0.05,
                    "simulated_spread_usdc": 0.05,
                    "fill_rate_window": 0.5,
                    "evidence_source": "DRY_RUN_SIMULATED",
                    "simulated_fill": True,
                }
            ]
        )
        replay_report = {
            "markets": [
                {
                    "market_slug": "sim-only",
                    "suitability": "REWARD_MM_CANDIDATE",
                    "net_pnl_usdc": 0.03,
                    "simulated_fill_count": 1,
                }
            ],
            "replayed_market_count": 1,
        }

        outputs = build_research_outputs(evidence_summary=evidence_summary, replay_report=replay_report, arb_rows=[])

        row = outputs["market_intel"]["by_market"]["sim-only"]
        self.assertEqual(row["recommended_action"], "DRY_RUN_FOCUS")
        self.assertEqual(outputs["whitelist"]["markets"], [])
        self.assertEqual(outputs["summary"]["live_canary_eligible_count"], 0)
        self.assertEqual(outputs["summary"]["positive_dry_run_market_count"], 1)
        self.assertEqual(outputs["market_intel"]["positive_dry_run_markets"][0]["market_slug"], "sim-only")
        self.assertEqual(outputs["summary"]["simulated_profitable_market_count"], 1)
        self.assertIn("SIMULATED_PROFIT_ONLY", outputs["summary"]["live_ready_blockers"])
        self.assertEqual(outputs["summary"]["scale_recommendation"], "ALLOW_DRY_RUN_FOCUS")

    def test_confirmed_positive_with_replay_canary_eligible(self) -> None:
        evidence_summary = build_live_edge_summary(
            [
                {
                    "row_type": "market_observation",
                    "market_slug": "confirmed",
                    "verified_net_window_usdc": 0.05,
                    "actual_reward_usdc": 0.02,
                    "spread_realized_usdc": 0.01,
                    "fill_rate_window": 0.5,
                    "evidence_source": "ACTUAL_REWARD",
                }
            ]
        )
        replay_report = {
            "markets": [{"market_slug": "confirmed", "suitability": "REWARD_MM_CANDIDATE"}],
            "replayed_market_count": 1,
        }

        outputs = build_research_outputs(evidence_summary=evidence_summary, replay_report=replay_report, arb_rows=[])

        row = outputs["market_intel"]["by_market"]["confirmed"]
        self.assertEqual(row["recommended_action"], "LIVE_CANARY_ELIGIBLE")
        self.assertEqual(outputs["summary"]["live_ready_blockers"], [])

    def test_pipeline_generates_research_outputs_with_fake_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            default_state = Path(tmpdir) / "auto_trade_profit_state_latest.json"
            default_state.write_text("do-not-delete", encoding="utf-8")
            args = argparse.Namespace(
                cycles=1,
                interval_sec=0,
                event_limit=10,
                market_limit=20,
                snapshot_max_markets=5,
                snapshot_filtered_max=3,
                out_dir=tmpdir,
                verbose=False,
            )

            def fake_runner(command: list[str]) -> None:
                joined = " ".join(command)
                if "run_auto_trade_profit.py" in joined:
                    evidence_path = Path(command[command.index("--edge-evidence-path") + 1])
                    snapshot_path = Path(command[command.index("--orderbook-snapshot-path") + 1])
                    evidence_path.write_text(
                        json.dumps(
                            {
                                "row_type": "market_observation",
                                "market_slug": "m1",
                                "event_slug": "e1",
                                "token_id": "tok-m1",
                                "actual_reward_usdc": 0.0,
                                "spread_realized_usdc": 0.0,
                                "verified_net_window_usdc": -0.00001,
                                "fill_rate_window": 0.0,
                                "order_reject_count": 0,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    snapshot_path.write_text(
                        "\n".join(
                            [
                                json.dumps(
                                    {
                                        "row_type": "orderbook_snapshot",
                                        "ts": "2026-05-01T00:00:00+00:00",
                                        "market_slug": "m1",
                                        "token_id": "tok-m1",
                                        "best_bid": 0.49,
                                        "best_ask": 0.51,
                                    }
                                ),
                                json.dumps(
                                    {
                                        "row_type": "orderbook_snapshot",
                                        "ts": "2026-05-01T00:01:00+00:00",
                                        "market_slug": "m1",
                                        "token_id": "tok-m1",
                                        "best_bid": 0.49,
                                        "best_ask": 0.51,
                                    }
                                ),
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return
                if "analyze_live_edge_evidence.py" in joined:
                    evidence_path = Path(command[command.index("--evidence") + 1])
                    out_path = Path(command[command.index("--out") + 1])
                    intel_path = Path(command[command.index("--market-intel-out") + 1])
                    summary = build_live_edge_summary(load_evidence_jsonl(str(evidence_path)))
                    out_path.write_text(json.dumps(summary), encoding="utf-8")
                    intel_path.write_text(json.dumps(summary["market_intel"]), encoding="utf-8")
                    return
                if "replay_live_orderbook_snapshots.py" in joined:
                    snapshot_path = Path(command[command.index("--snapshots") + 1])
                    out_path = Path(command[command.index("--out") + 1])
                    replay = build_replay_report(load_snapshot_jsonl(str(snapshot_path)))
                    out_path.write_text(json.dumps(replay), encoding="utf-8")
                    return
                raise AssertionError(f"unexpected command: {command}")

            summary = run_pipeline(args, command_runner=fake_runner)

            self.assertEqual(summary["market_count"], 1)
            self.assertTrue(summary["fresh_state"])
            self.assertFalse(summary["contaminated_state_detected"])
            self.assertEqual(Path(summary["state_path"]).name, "research_auto_trade_state_latest.json")
            self.assertEqual(Path(summary["pnl_path"]).name, "research_auto_trade_pnl_latest.json")
            self.assertEqual(default_state.read_text(encoding="utf-8"), "do-not-delete")
            self.assertTrue((Path(tmpdir) / "research_pipeline_summary_latest.json").exists())
            self.assertTrue((Path(tmpdir) / "research_market_intel_latest.json").exists())
            self.assertTrue((Path(tmpdir) / "research_whitelist_latest.json").exists())
            self.assertTrue((Path(tmpdir) / "research_blacklist_latest.json").exists())

    def test_pipeline_marks_contaminated_cycle_one_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                cycles=1,
                interval_sec=0,
                event_limit=10,
                market_limit=20,
                snapshot_max_markets=5,
                snapshot_filtered_max=3,
                out_dir=tmpdir,
                verbose=False,
            )

            def fake_runner(command: list[str]) -> None:
                joined = " ".join(command)
                if "run_auto_trade_profit.py" in joined:
                    evidence_path = Path(command[command.index("--edge-evidence-path") + 1])
                    snapshot_path = Path(command[command.index("--orderbook-snapshot-path") + 1])
                    evidence_path.write_text(
                        json.dumps(
                            {
                                "row_type": "market_observation",
                                "cycle_index": 1,
                                "market_slug": "old-state-market",
                                "bid_order_filled_size": 337.0,
                                "ask_order_filled_size": 35.0,
                                "reward_estimate_usdc": 0.24,
                                "spread_realized_usdc": 0.35,
                                "verified_net_window_usdc": 0.10,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    snapshot_path.write_text("", encoding="utf-8")
                    return
                if "analyze_live_edge_evidence.py" in joined:
                    evidence_path = Path(command[command.index("--evidence") + 1])
                    out_path = Path(command[command.index("--out") + 1])
                    intel_path = Path(command[command.index("--market-intel-out") + 1])
                    summary = build_live_edge_summary(load_evidence_jsonl(str(evidence_path)))
                    summary["scale_recommendation"] = "ALLOW_CANARY_ONLY"
                    out_path.write_text(json.dumps(summary), encoding="utf-8")
                    intel_path.write_text(json.dumps(summary["market_intel"]), encoding="utf-8")
                    return
                if "replay_live_orderbook_snapshots.py" in joined:
                    out_path = Path(command[command.index("--out") + 1])
                    out_path.write_text(json.dumps(build_replay_report([])), encoding="utf-8")
                    return
                raise AssertionError(f"unexpected command: {command}")

            summary = run_pipeline(args, command_runner=fake_runner)

            self.assertTrue(summary["contaminated_state_detected"])
            self.assertIn("CYCLE_1_FILLED_SHARES_GT_5", summary["contamination_reason"])
            self.assertEqual(summary["scale_recommendation"], "DO_NOT_SCALE")

    def test_pipeline_builds_partial_outputs_after_interrupted_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                cycles=120,
                interval_sec=30,
                event_limit=10,
                market_limit=20,
                snapshot_max_markets=5,
                snapshot_filtered_max=3,
                out_dir=tmpdir,
                verbose=False,
            )

            def fake_runner(command: list[str]) -> None:
                joined = " ".join(command)
                if "run_auto_trade_profit.py" in joined:
                    evidence_path = Path(command[command.index("--edge-evidence-path") + 1])
                    snapshot_path = Path(command[command.index("--orderbook-snapshot-path") + 1])
                    evidence_path.write_text(
                        json.dumps(
                            {
                                "row_type": "market_observation",
                                "market_slug": "partial-market",
                                "verified_net_window_usdc": 0.0,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    snapshot_path.write_text("", encoding="utf-8")
                    raise PipelineInterrupted()
                if "analyze_live_edge_evidence.py" in joined:
                    evidence_path = Path(command[command.index("--evidence") + 1])
                    out_path = Path(command[command.index("--out") + 1])
                    intel_path = Path(command[command.index("--market-intel-out") + 1])
                    summary = build_live_edge_summary(load_evidence_jsonl(str(evidence_path)))
                    out_path.write_text(json.dumps(summary), encoding="utf-8")
                    intel_path.write_text(json.dumps(summary["market_intel"]), encoding="utf-8")
                    return
                if "replay_live_orderbook_snapshots.py" in joined:
                    out_path = Path(command[command.index("--out") + 1])
                    out_path.write_text(json.dumps(build_replay_report([])), encoding="utf-8")
                    return
                raise AssertionError(f"unexpected command: {command}")

            summary = run_pipeline(args, command_runner=fake_runner)

            self.assertTrue(summary["partial"])
            self.assertEqual(summary["partial_reason"], "SCAN_INTERRUPTED")
            self.assertTrue((Path(tmpdir) / "research_pipeline_summary_latest.json").exists())

    def test_quick_mode_uses_short_defaults(self) -> None:
        args = parse_args(["--quick"])

        self.assertEqual(args.cycles, 10)
        self.assertEqual(args.interval_sec, 5)
        self.assertEqual(args.max_selected_markets, 3)
        self.assertEqual(args.research_min_reward_minus_drawdown_per_hour, 0.0)
        self.assertEqual(args.research_min_projected_net_at_horizon_usdc, 0.0)
        self.assertEqual(args.research_max_true_break_even_hours, 2.5)
        self.assertEqual(args.research_per_market_cap_usdc, 40.0)

    def test_scan_command_keeps_progress_visible(self) -> None:
        args = argparse.Namespace(
            cycles=1,
            interval_sec=0,
            event_limit=10,
            market_limit=20,
            snapshot_max_markets=5,
            snapshot_filtered_max=3,
            verbose=False,
        )
        paths = {
            "evidence": Path("evidence.jsonl"),
            "snapshots": Path("snapshots.jsonl"),
            "state": Path("research_state.json"),
            "pnl": Path("research_pnl.json"),
        }

        command = _scan_command(args, paths)

        self.assertNotIn("--no-progress", command)
        self.assertIn("--reset-state", command)
        self.assertEqual(command[command.index("--state-path") + 1], "research_state.json")
        self.assertEqual(command[command.index("--pnl-path") + 1], "research_pnl.json")
        self.assertEqual(command[command.index("--max-markets") + 1], "3")
        self.assertEqual(command[command.index("--capital") + 1], "120")
        self.assertEqual(command[command.index("--max-total-open-buy-usdc") + 1], "120")
        self.assertEqual(command[command.index("--max-account-open-buy-orders") + 1], "3")
        self.assertEqual(command[command.index("--per-market-cap") + 1], "40")
        self.assertEqual(command[command.index("--max-inventory-usdc-per-market") + 1], "40")
        self.assertEqual(command[command.index("--min-reward-minus-drawdown-per-hour") + 1], "0.0")
        self.assertEqual(command[command.index("--min-projected-net-at-horizon-usdc") + 1], "0.0")
        self.assertEqual(command[command.index("--max-true-break-even-hours") + 1], "2.5")

    def test_scan_command_allows_single_selected_market_override(self) -> None:
        args = argparse.Namespace(
            cycles=1,
            interval_sec=0,
            event_limit=10,
            market_limit=20,
            snapshot_max_markets=5,
            snapshot_filtered_max=3,
            max_selected_markets=1,
            verbose=False,
        )
        paths = {
            "evidence": Path("evidence.jsonl"),
            "snapshots": Path("snapshots.jsonl"),
            "state": Path("research_state.json"),
            "pnl": Path("research_pnl.json"),
        }

        command = _scan_command(args, paths)

        self.assertEqual(command[command.index("--max-markets") + 1], "1")
        self.assertEqual(command[command.index("--capital") + 1], "40")
        self.assertEqual(command[command.index("--max-total-open-buy-usdc") + 1], "40")
        self.assertEqual(command[command.index("--max-account-open-buy-orders") + 1], "1")

    def test_scan_command_allows_dry_run_cap_override_without_live_changes(self) -> None:
        args = parse_args(["--max-selected-markets", "3", "--research-per-market-cap-usdc", "80"])
        paths = {
            "evidence": Path("evidence.jsonl"),
            "snapshots": Path("snapshots.jsonl"),
            "state": Path("research_state.json"),
            "pnl": Path("research_pnl.json"),
        }

        command = _scan_command(args, paths)

        self.assertEqual(command[command.index("--max-markets") + 1], "3")
        self.assertEqual(command[command.index("--capital") + 1], "240")
        self.assertEqual(command[command.index("--per-market-cap") + 1], "80")
        self.assertEqual(command[command.index("--max-total-open-buy-usdc") + 1], "240")
        self.assertEqual(command[command.index("--max-account-open-buy-orders") + 1], "3")

    def test_scan_command_allows_research_filter_overrides(self) -> None:
        args = parse_args(
            [
                "--research-min-reward-minus-drawdown-per-hour",
                "0.01",
                "--research-min-projected-net-at-horizon-usdc",
                "0.02",
                "--research-max-true-break-even-hours",
                "4",
            ]
        )
        paths = {
            "evidence": Path("evidence.jsonl"),
            "snapshots": Path("snapshots.jsonl"),
            "state": Path("research_state.json"),
            "pnl": Path("research_pnl.json"),
        }

        command = _scan_command(args, paths)

        self.assertEqual(command[command.index("--min-reward-minus-drawdown-per-hour") + 1], "0.01")
        self.assertEqual(command[command.index("--min-projected-net-at-horizon-usdc") + 1], "0.02")
        self.assertEqual(command[command.index("--max-true-break-even-hours") + 1], "4.0")

    def test_pipeline_rejects_live_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_evidence_research_pipeline.py"), "--live"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dry-run only", result.stderr)

    def test_auto_trade_cli_exposes_custom_state_paths(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_auto_trade_profit.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--state-path", result.stdout)
        self.assertIn("--pnl-path", result.stdout)

    def test_quick_pipeline_writes_quick_latest_without_overwriting_full_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            full_latest = Path(tmpdir) / "research_pipeline_summary_latest.json"
            full_latest.write_text('{"keep":"full"}', encoding="utf-8")
            args = parse_args(["--quick", "--run-id", "smoke", "--out-dir", tmpdir])
            fake_runner, _commands = self._fake_research_runner()

            summary = run_pipeline(args, command_runner=fake_runner)

            self.assertTrue(summary["quick"])
            self.assertEqual(summary["run_id"], "quick-smoke")
            self.assertEqual(json.loads(full_latest.read_text(encoding="utf-8")), {"keep": "full"})
            self.assertTrue((Path(tmpdir) / "research_quick_pipeline_summary_latest.json").exists())
            self.assertFalse((Path(tmpdir) / "research_runs" / "quick-smoke" / "research_pipeline_summary_latest.json").exists())
            self.assertTrue((Path(tmpdir) / "research_runs" / "quick-smoke" / "research_quick_pipeline_summary_latest.json").exists())

    def test_pipeline_archives_existing_latest_before_full_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_latest = Path(tmpdir) / "research_pipeline_summary_latest.json"
            old_latest.write_text('{"old":true}', encoding="utf-8")
            args = parse_args(["--cycles", "1", "--interval-sec", "0", "--run-id", "full-r1", "--out-dir", tmpdir])
            fake_runner, _commands = self._fake_research_runner()

            summary = run_pipeline(args, command_runner=fake_runner)

            archive = Path(tmpdir) / "research_runs" / "full-r1" / "archive_before_latest_update" / "research_pipeline_summary_latest.json"
            self.assertEqual(summary["run_id"], "full-r1")
            self.assertTrue(archive.exists())
            self.assertEqual(json.loads(archive.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(json.loads(old_latest.read_text(encoding="utf-8"))["run_id"], "full-r1")

    def test_merge_only_rebuilds_reports_without_running_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence = Path(tmpdir) / "research_edge_observations_latest.jsonl"
            snapshots = Path(tmpdir) / "research_orderbook_snapshots_latest.jsonl"
            evidence.write_text(
                json.dumps({"row_type": "market_observation", "market_slug": "m1", "verified_net_window_usdc": 0.0}) + "\n",
                encoding="utf-8",
            )
            snapshots.write_text("", encoding="utf-8")
            args = parse_args(["--merge-only", "--run-id", "recover-r1", "--out-dir", tmpdir])
            fake_runner, commands = self._fake_research_runner()

            summary = run_pipeline(args, command_runner=fake_runner)

            self.assertTrue(summary["merge_only"])
            self.assertFalse(any("run_auto_trade_profit.py" in " ".join(command) for command in commands))
            self.assertTrue((Path(tmpdir) / "research_runs" / "recover-r1" / "research_edge_observations_latest.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "research_pipeline_summary_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
