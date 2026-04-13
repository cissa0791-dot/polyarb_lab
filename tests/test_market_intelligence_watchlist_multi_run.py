from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.intelligence.watchlist_multi_run import (
    build_multi_run_stability_report,
    write_multi_run_stability_report,
)


class MarketIntelligenceWatchlistMultiRunTests(unittest.TestCase):
    def test_build_multi_run_stability_report_aggregates_gaps(self) -> None:
        run_summaries = [
            {
                "run_ts": "2026-03-17T05:00:00+00:00",
                "validation_forward_events": 9,
                "markets_ranked": 10,
                "events_ranked": 4,
                "top5_market_avg_forward_signal_score": 4.8,
                "lower_market_avg_forward_signal_score": 1.2,
                "top3_event_avg_forward_signal_score": 9.3,
                "lower_event_avg_forward_signal_score": 2.0,
                "market_signal_gap": 3.6,
                "event_signal_gap": 7.3,
            },
            {
                "run_ts": "2026-03-17T05:05:00+00:00",
                "validation_forward_events": 8,
                "markets_ranked": 9,
                "events_ranked": 4,
                "top5_market_avg_forward_signal_score": 3.0,
                "lower_market_avg_forward_signal_score": 1.0,
                "top3_event_avg_forward_signal_score": 6.0,
                "lower_event_avg_forward_signal_score": 1.5,
                "market_signal_gap": 2.0,
                "event_signal_gap": 4.5,
            },
        ]

        report = build_multi_run_stability_report(run_summaries)

        self.assertTrue(report["paper_only"])
        self.assertEqual(report["primary_ranking_level"], "event")
        self.assertEqual(report["summary"]["run_count"], 2)
        self.assertEqual(report["summary"]["avg_market_signal_gap"], 2.8)
        self.assertEqual(report["summary"]["positive_market_signal_gap_runs"], 2)
        self.assertTrue(report["summary"]["ranking_edge_stays_positive"])
        self.assertTrue(report["summary"]["event_ranking_edge_stays_positive"])

    def test_write_multi_run_stability_report_outputs_files(self) -> None:
        report = {"paper_only": True, "summary": {"run_count": 1}, "runs": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = write_multi_run_stability_report(out_dir=Path(tmp_dir), report=report)
            self.assertTrue(written["report_path"].exists())
            self.assertTrue(written["latest_report_path"].exists())


if __name__ == "__main__":
    unittest.main()
