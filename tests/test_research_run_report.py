from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_research_run_report import build_report


class ResearchRunReportTests(unittest.TestCase):
    def test_report_includes_current_run_and_gate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "reports"
            run_dir = out_dir / "research_runs" / "research-report"
            run_dir.mkdir(parents=True)
            (run_dir / "research_auto_trade_pnl_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "cycle_index": 12,
                            "active_quote_market_count": 1,
                            "last_eligible_candidate_count": 4,
                            "last_selection_reasons": {"SELECT_ZERO_SIZE_REJECT": 2, "SELECTED": 1},
                            "last_filter_reasons": {"REWARD_MINUS_DRAWDOWN": 10},
                            "scan_diagnostics": {"selected_markets": 1},
                            "verified_net_after_reward_and_cost_usdc": 0.03,
                            "net_after_reward_and_cost_usdc": 0.04,
                            "bid_order_filled_shares": 5.0,
                            "ask_order_filled_shares": 4.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "research_pipeline_summary_latest.json").write_text(
                json.dumps(
                    {
                        "scale_recommendation": "ALLOW_DRY_RUN_FOCUS",
                        "live_canary_eligible_count": 0,
                        "dry_run_focus_count": 2,
                        "blacklist_count": 1,
                        "partial": False,
                        "partial_reason": None,
                        "live_ready_blockers": ["NO_ACTUAL_REWARD_CONFIRMED"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "research_market_intel_latest.json").write_text(
                json.dumps(
                    {
                        "markets": [
                            {
                                "market_slug": "m1",
                                "recommended_action": "DRY_RUN_FOCUS",
                                "research_score": 0.25,
                                "reason": "needs more dry-run observation before live",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "research_edge_observations_latest.jsonl").write_text("{}\n", encoding="utf-8")
            (run_dir / "research_orderbook_snapshots_latest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (out_dir / "autonomous_decision_latest.json").write_text(
                json.dumps(
                    {
                        "decision": "START_MICRO_LIVE_PROBE",
                        "reason": "dry-run edge is positive but actual reward requires a capped live probe",
                        "target_live_markets": 1,
                        "max_live_risk_usdc": 20.0,
                        "can_execute_live": True,
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(out_dir=out_dir, run_dir=run_dir, proc_root=root / "missing-proc")

            self.assertIn("# Research Run Report", report)
            self.assertIn("Current run: research-report", report)
            self.assertIn("Selected / active / eligible: 1 / 1 / 4", report)
            self.assertIn("Scale recommendation: ALLOW_DRY_RUN_FOCUS", report)
            self.assertIn("NO_ACTUAL_REWARD_CONFIRMED", report)
            self.assertIn("m1 | DRY_RUN_FOCUS", report)
            self.assertIn("isolated capped micro live probe", report)


if __name__ == "__main__":
    unittest.main()
