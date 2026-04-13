from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.intelligence.watchlist_replay_validator import (
    build_watchlist_validation_report,
    write_watchlist_validation_report,
)


class MarketIntelligenceWatchlistReplayValidatorTests(unittest.TestCase):
    def test_build_watchlist_validation_report_scores_forward_activity(self) -> None:
        registry = {
            "summary": {"events_seen": 2, "markets_seen": 3},
            "events": [
                {
                    "slug": "event-one",
                    "title": "Event One",
                    "markets": [{"slug": "market-a"}, {"slug": "market-b"}],
                },
                {
                    "slug": "event-two",
                    "title": "Event Two",
                    "markets": [{"slug": "market-c"}],
                },
            ],
        }
        live_delta_report = {"summary": {"markets_updated": 3}}
        watchlist_report = {
            "top_markets": [
                {"market_slug": "market-a", "event_slug": "event-one", "watchlist_score": 10},
                {"market_slug": "market-c", "event_slug": "event-two", "watchlist_score": 4},
            ],
            "top_events": [
                {"event_slug": "event-one", "watchlist_score": 12},
                {"event_slug": "event-two", "watchlist_score": 4},
            ],
        }
        delta_events = [
            {"market_slug": "market-a", "observed_ts": "2026-03-17T05:00:00+00:00", "spread_changed": True, "liquidity_changed": True, "completeness_changed": False},
            {"market_slug": "market-b", "observed_ts": "2026-03-17T05:00:01+00:00", "spread_changed": False, "liquidity_changed": True, "completeness_changed": False},
            {"market_slug": "market-a", "observed_ts": "2026-03-17T05:00:02+00:00", "spread_changed": True, "liquidity_changed": True, "completeness_changed": False},
            {"market_slug": "market-c", "observed_ts": "2026-03-17T05:00:03+00:00", "spread_changed": False, "liquidity_changed": True, "completeness_changed": True},
        ]

        report = build_watchlist_validation_report(
            registry=registry,
            live_delta_report=live_delta_report,
            watchlist_report=watchlist_report,
            delta_events=delta_events,
        )

        self.assertTrue(report["paper_only"])
        self.assertEqual(report["primary_ranking_level"], "event")
        self.assertEqual(report["summary"]["live_delta_events"], 4)
        self.assertEqual(report["summary"]["validation_forward_events"], 2)
        self.assertEqual(report["top_event_validation"][0]["event_slug"], "event-one")
        self.assertEqual(report["top_event_validation"][0]["top_market_drilldown"][0]["market_slug"], "market-a")

    def test_write_watchlist_validation_report_outputs_files(self) -> None:
        report = {"paper_only": True, "summary": {"markets_ranked": 1}, "top_market_validation": [], "top_event_validation": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = write_watchlist_validation_report(out_dir=Path(tmp_dir), report=report)
            self.assertTrue(written["report_path"].exists())
            self.assertTrue(written["latest_report_path"].exists())


if __name__ == "__main__":
    unittest.main()
