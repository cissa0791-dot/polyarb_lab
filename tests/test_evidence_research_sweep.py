from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_evidence_research_sweep import DEFAULT_ARMS, build_sweep_summary, selected_arms


class EvidenceResearchSweepTests(unittest.TestCase):
    def test_default_sweep_has_ten_distinct_arms(self) -> None:
        names = [arm.name for arm in DEFAULT_ARMS]

        self.assertEqual(len(names), 10)
        self.assertEqual(len(set(names)), 10)
        self.assertIn("highraw_3000_sel3_cap80", names)
        self.assertIn("cap120_sel5", names)

    def test_selected_arms_filters_named_defaults(self) -> None:
        arms = selected_arms(["baseline_2000_sel3_cap80", "fast_break_even_1h"])

        self.assertEqual([arm.name for arm in arms], ["baseline_2000_sel3_cap80", "fast_break_even_1h"])

    def test_summary_ranks_verified_positive_above_replay_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep_dir = Path(tmp)
            good = _write_arm(
                sweep_dir,
                "good",
                verified=0.10,
                modeled=0.12,
                replay=0.0,
                active=3,
                selected=3,
            )
            bad_replay = _write_arm(
                sweep_dir,
                "bad_replay",
                verified=0.12,
                modeled=0.13,
                replay=-0.20,
                active=3,
                selected=3,
            )

            summary = build_sweep_summary(sweep_id="sweep-test", sweep_dir=sweep_dir, results=[good, bad_replay])

            self.assertEqual(summary["best_arm"]["arm_name"], "good")
            self.assertGreater(
                summary["arms"][0]["profitability_score"],
                summary["arms"][1]["profitability_score"],
            )


def _write_arm(
    sweep_dir: Path,
    name: str,
    *,
    verified: float,
    modeled: float,
    replay: float,
    active: int,
    selected: int,
) -> dict[str, object]:
    arm_dir = sweep_dir / name
    run_dir = arm_dir / "research_runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "research_pipeline_summary_latest.json").write_text(
        json.dumps({"partial": False, "scale_recommendation": "ALLOW_DRY_RUN_FOCUS"}),
        encoding="utf-8",
    )
    (run_dir / "research_auto_trade_pnl_latest.json").write_text(
        json.dumps(
            {
                "summary": {
                    "verified_net_after_reward_and_cost_usdc": verified,
                    "net_after_reward_and_cost_usdc": modeled,
                    "active_quote_market_count": active,
                    "last_eligible_candidate_count": 8,
                    "scan_diagnostics": {"selected_markets": selected},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "research_replay_report_latest.json").write_text(
        json.dumps({"total_net_pnl_usdc": replay}),
        encoding="utf-8",
    )
    return {
        "arm_name": name,
        "arm_dir": str(arm_dir),
        "run_id": name,
        "return_code": 0,
    }


if __name__ == "__main__":
    unittest.main()
