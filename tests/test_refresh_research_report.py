from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.refresh_research_report import refresh_reports


class RefreshResearchReportTests(unittest.TestCase):
    def test_refreshes_profit_and_run_reports_for_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            run_dir = out_dir / "research_runs" / "research-test"
            run_dir.mkdir(parents=True)
            evidence = run_dir / "research_edge_observations_latest.jsonl"
            evidence.write_text(
                json.dumps(
                    {
                        "row_type": "cycle_summary",
                        "cycle_index": 4,
                        "selected_market_count": 1,
                        "active_quote_market_count": 1,
                        "eligible_candidate_count": 2,
                        "verified_net_after_reward_and_cost_usdc": 0.02,
                        "net_after_reward_and_cost_usdc": 0.03,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "row_type": "market_observation",
                        "cycle_index": 4,
                        "market_slug": "m1",
                        "verified_net_window_usdc": 0.02,
                        "fill_rate_window": 0.1,
                        "simulated_fill": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = refresh_reports(out_dir=out_dir, run_dir=run_dir, proc_root=out_dir / "missing-proc")

            self.assertTrue(result["refreshed"])
            self.assertEqual(result["latest_cycle"], 4)
            self.assertTrue((run_dir / "research_profit_drivers_latest.json").exists())
            self.assertTrue((run_dir / "research_profit_drivers.md").exists())
            self.assertTrue((run_dir / "research_run_report.md").exists())


if __name__ == "__main__":
    unittest.main()
