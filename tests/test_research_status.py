from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.show_research_status import build_status


class ResearchStatusTests(unittest.TestCase):
    def test_status_prefers_active_run_dir_over_root_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "reports"
            run_id = "research-active"
            run_dir = out_dir / "research_runs" / run_id
            proc_root = root / "proc"
            proc_dir = proc_root / "1234"
            run_dir.mkdir(parents=True)
            proc_dir.mkdir(parents=True)
            (out_dir / "research_pipeline_summary_latest.json").write_text(
                json.dumps({"run_id": "old-root", "scale_recommendation": "ALLOW_DRY_RUN_FOCUS"}),
                encoding="utf-8",
            )
            (run_dir / "research_auto_trade_pnl_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "cycle_index": 7,
                            "last_eligible_candidate_count": 4,
                            "last_selection_reasons": {"SELECT_ZERO_SIZE_REJECT": 2, "SELECTED": 1},
                            "last_filter_reasons": {"REWARD_MINUS_DRAWDOWN": 9},
                            "scan_diagnostics": {"selected_markets": 1},
                            "active_quote_market_count": 1,
                            "verified_net_after_reward_and_cost_usdc": 0.0123,
                            "net_after_reward_and_cost_usdc": 0.017,
                            "bid_order_filled_shares": 3.0,
                            "ask_order_filled_shares": 2.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "research_edge_observations_latest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (run_dir / "research_orderbook_snapshots_latest.jsonl").write_text("{}\n", encoding="utf-8")
            argv = [
                "python",
                "scripts/run_evidence_research_pipeline.py",
                "--run-id",
                run_id,
                "--out-dir",
                str(out_dir),
                "--cycles",
                "240",
                "--interval-sec",
                "30",
            ]
            (proc_dir / "cmdline").write_bytes(b"\0".join(part.encode("utf-8") for part in argv) + b"\0")

            status = build_status(out_dir, proc_root)

            self.assertEqual(status["active_research_process_count"], 1)
            self.assertEqual(status["root_latest_run_id"], "old-root")
            current = status["current_run"]
            self.assertEqual(current["run_id"], run_id)
            self.assertEqual(current["cycle_index"], 7)
            self.assertEqual(current["selected_markets"], 1)
            self.assertEqual(current["active_quote_market_count"], 1)
            self.assertEqual(current["evidence_rows"], 2)
            self.assertEqual(current["snapshot_rows"], 1)
            self.assertEqual(current["last_selection_reasons"]["SELECT_ZERO_SIZE_REJECT"], 2)

    def test_status_falls_back_to_latest_evidence_when_pnl_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "reports"
            run_id = "research-evidence-only"
            run_dir = out_dir / "research_runs" / run_id
            proc_root = root / "proc"
            proc_dir = proc_root / "4321"
            run_dir.mkdir(parents=True)
            proc_dir.mkdir(parents=True)
            rows = [
                {
                    "row_type": "market_observation",
                    "cycle_index": 4,
                    "market_slug": "old",
                    "status": "QUOTING",
                    "verified_net_window_usdc": 0.01,
                },
                {
                    "row_type": "market_observation",
                    "cycle_index": 5,
                    "market_slug": "m1",
                    "status": "QUOTING",
                    "verified_net_window_usdc": 0.02,
                    "net_after_reward_and_cost_usdc": 0.03,
                    "bid_order_filled_size": 2.0,
                    "ask_order_filled_size": 1.5,
                },
                {
                    "row_type": "market_observation",
                    "cycle_index": 5,
                    "market_slug": "m2",
                    "status": "PAUSED",
                    "verified_net_window_usdc": -0.005,
                    "net_after_reward_and_cost_usdc": -0.004,
                    "bid_order_filled_size": 0.0,
                    "ask_order_filled_size": 0.0,
                },
            ]
            (run_dir / "research_edge_observations_latest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (run_dir / "research_orderbook_snapshots_latest.jsonl").write_text("", encoding="utf-8")
            argv = [
                "python",
                "scripts/run_evidence_research_pipeline.py",
                "--run-id",
                run_id,
                "--out-dir",
                str(out_dir),
            ]
            (proc_dir / "cmdline").write_bytes(b"\0".join(part.encode("utf-8") for part in argv) + b"\0")

            current = build_status(out_dir, proc_root)["current_run"]

            self.assertEqual(current["cycle_index"], 5)
            self.assertEqual(current["selected_markets"], 2)
            self.assertEqual(current["active_quote_market_count"], 1)
            self.assertAlmostEqual(current["verified_net_after_cost_usdc"], 0.015)
            self.assertAlmostEqual(current["modeled_net_after_cost_usdc"], 0.026)
            self.assertEqual(current["bid_filled_shares"], 2.0)
            self.assertEqual(current["ask_filled_shares"], 1.5)
            self.assertEqual(current["latest_market_slugs"], ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()
