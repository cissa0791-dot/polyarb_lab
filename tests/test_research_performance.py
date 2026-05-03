from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_research_performance import analyze_performance, build_markdown_report


class ResearchPerformanceTests(unittest.TestCase):
    def test_performance_uses_state_wall_clock_for_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_run(
                Path(tmpdir),
                verified=0.12,
                modeled=0.15,
                actual_reward=0.0,
                estimated_reward=0.03,
                spread=0.12,
                capital_limit=240.0,
                capital_used=12.0,
                replay_net=0.0,
                started_ts="2026-05-01T00:00:00+00:00",
                updated_ts="2026-05-01T01:00:00+00:00",
                completed=True,
            )

            report = analyze_performance(out_dir=run_dir.parents[1], run_dir=run_dir)

            self.assertEqual(report["duration_source"], "state_wall_clock")
            self.assertEqual(report["verified_usdc_per_hour"], 0.12)
            self.assertEqual(report["verified_roi_on_cap_pct"], 0.05)
            self.assertEqual(report["verified_roi_on_cap_pct_per_hour"], 0.05)
            self.assertEqual(report["verified_roi_on_used_cap_pct"], 1.0)
            self.assertIn("NO_ACTUAL_REWARD_CONFIRMED", report["why_not_live"])

    def test_replay_negative_blocks_live_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_run(
                Path(tmpdir),
                verified=0.25,
                modeled=0.30,
                actual_reward=0.02,
                estimated_reward=0.03,
                spread=0.23,
                capital_limit=100.0,
                capital_used=20.0,
                replay_net=-0.04,
                started_ts="2026-05-01T00:00:00+00:00",
                updated_ts="2026-05-01T00:30:00+00:00",
                completed=True,
                live_canary_eligible_count=1,
            )

            report = analyze_performance(out_dir=run_dir.parents[1], run_dir=run_dir)

            self.assertFalse(report["live_ready"])
            self.assertIn("REPLAY_NEGATIVE", report["why_not_live"])
            self.assertEqual(
                report["recommended_next_step"],
                "continue dry-run/sweep; improve replay-confirmed fill realism before live",
            )

    def test_active_run_without_pipeline_uses_proc_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "reports"
            run_id = "active-run"
            run_dir = out_dir / "research_runs" / run_id
            _write_run(
                root,
                run_id=run_id,
                verified=0.06,
                modeled=0.08,
                actual_reward=0.0,
                estimated_reward=0.02,
                spread=0.06,
                capital_limit=120.0,
                capital_used=6.0,
                replay_net=0.0,
                started_ts=None,
                updated_ts=None,
                completed=False,
            )
            proc_root = root / "proc"
            proc_dir = proc_root / "123"
            proc_dir.mkdir(parents=True)
            argv = [
                "python",
                "scripts/run_evidence_research_pipeline.py",
                "--run-id",
                run_id,
                "--out-dir",
                str(out_dir),
                "--cycles",
                "120",
                "--interval-sec",
                "20",
            ]
            (proc_dir / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")

            report = analyze_performance(out_dir=out_dir, proc_root=proc_root)

            self.assertEqual(report["run_id"], run_id)
            self.assertEqual(report["completed"], False)
            self.assertEqual(report["duration_source"], "cycle_interval_nominal")
            self.assertEqual(report["duration_hours"], 0.166667)
            self.assertEqual(report["verified_usdc_per_hour"], 0.36)
            self.assertIn("DRY_RUN_MODE", report["why_not_live"])

    def test_markdown_includes_core_metrics(self) -> None:
        report = {
            "run_id": "r1",
            "completed": True,
            "partial": False,
            "cycle_index": 10,
            "cycles_requested": 10,
            "progress_pct": 100,
            "selected_markets": 3,
            "active_quote_market_count": 3,
            "eligible_candidates": 8,
            "duration_hours": 1.0,
            "duration_source": "state_wall_clock",
            "verified_net_after_cost_usdc": 0.1,
            "modeled_net_after_cost_usdc": 0.12,
            "verified_usdc_per_hour": 0.1,
            "modeled_usdc_per_hour": 0.12,
            "verified_roi_on_cap_pct": 0.05,
            "verified_roi_on_cap_pct_per_hour": 0.05,
            "verified_roi_on_used_cap_pct": 1.0,
            "verified_roi_on_used_cap_pct_per_hour": 1.0,
            "spread_realized_usdc": 0.08,
            "estimated_reward_usdc": 0.02,
            "actual_reward_usdc": 0.0,
            "replay_total_net_pnl_usdc": -0.01,
            "live_ready": False,
            "why_not_live": ["REPLAY_NEGATIVE"],
            "recommended_next_step": "continue dry-run",
        }

        markdown = build_markdown_report(report)

        self.assertIn("Verified per hour: 0.1 USDC/h", markdown)
        self.assertIn("Why not live", markdown)


def _write_run(
    root: Path,
    *,
    run_id: str = "research-test",
    verified: float,
    modeled: float,
    actual_reward: float,
    estimated_reward: float,
    spread: float,
    capital_limit: float,
    capital_used: float,
    replay_net: float,
    started_ts: str | None,
    updated_ts: str | None,
    completed: bool,
    live_canary_eligible_count: int = 0,
) -> Path:
    out_dir = root / "reports"
    run_dir = out_dir / "research_runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "research_auto_trade_pnl_latest.json").write_text(
        json.dumps(
            {
                "generated_ts": updated_ts or "2026-05-01T00:10:00+00:00",
                "summary": {
                    "mode": "DRY_RUN",
                    "cycle_index": 30,
                    "scan_diagnostics": {"selected_markets": 3},
                    "active_quote_market_count": 3,
                    "last_eligible_candidate_count": 8,
                    "capital_limit_usdc": capital_limit,
                    "capital_in_use_usdc": capital_used,
                    "verified_net_after_reward_and_cost_usdc": verified,
                    "net_after_reward_and_cost_usdc": modeled,
                    "reward_accrued_actual_usdc": actual_reward,
                    "reward_accrued_estimate_usdc": estimated_reward,
                    "spread_realized_usdc": spread,
                    "cost_proxy_usdc": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    state_payload = {"cycle_index": 30}
    if started_ts:
        state_payload["started_ts"] = started_ts
    if updated_ts:
        state_payload["updated_ts"] = updated_ts
    (run_dir / "research_auto_trade_state_latest.json").write_text(json.dumps(state_payload), encoding="utf-8")
    (run_dir / "research_replay_report_latest.json").write_text(
        json.dumps({"total_net_pnl_usdc": replay_net}),
        encoding="utf-8",
    )
    if completed:
        (run_dir / "research_pipeline_summary_latest.json").write_text(
            json.dumps(
                {
                    "scan_cycles_requested": 30,
                    "scan_interval_sec": 120,
                    "partial": False,
                    "scale_recommendation": "ALLOW_DRY_RUN_FOCUS",
                    "live_canary_eligible_count": live_canary_eligible_count,
                }
            ),
            encoding="utf-8",
        )
    return run_dir


if __name__ == "__main__":
    unittest.main()
