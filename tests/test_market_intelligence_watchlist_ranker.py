from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.intelligence.watchlist_ranker import build_watchlist_report, write_watchlist_report
from src.storage.event_store import ResearchStore


class MarketIntelligenceWatchlistRankerTests(unittest.TestCase):
    def test_build_watchlist_report_ranks_markets_and_events(self) -> None:
        registry = {
            "summary": {"events_seen": 2, "markets_seen": 3, "tracked_tokens": 6},
            "events": [
                {
                    "slug": "event-one",
                    "title": "Event One",
                    "markets": [
                        {"slug": "market-a", "question": "A?", "best_bid": 0.4, "best_ask": 0.42},
                        {"slug": "market-b", "question": "B?", "best_bid": 0.3, "best_ask": 0.35},
                    ],
                },
                {
                    "slug": "event-two",
                    "title": "Event Two",
                    "markets": [
                        {"slug": "market-c", "question": "C?", "best_bid": 0.5, "best_ask": 0.55},
                    ],
                },
            ],
        }
        snapshot_report = {"summary": {"books_fetched": 3}}
        live_delta_report = {"summary": {"markets_updated": 2, "spread_changes": 2, "liquidity_changes": 3}}
        delta_events = [
            {"market_slug": "market-a", "spread_changed": True, "liquidity_changed": True, "completeness_changed": False},
            {"market_slug": "market-a", "spread_changed": False, "liquidity_changed": True, "completeness_changed": False},
            {"market_slug": "market-c", "spread_changed": True, "liquidity_changed": True, "completeness_changed": True},
        ]

        report = build_watchlist_report(
            registry=registry,
            snapshot_report=snapshot_report,
            live_delta_report=live_delta_report,
            delta_events=delta_events,
        )

        self.assertTrue(report["paper_only"])
        self.assertEqual(report["primary_ranking_level"], "event")
        self.assertEqual(report["summary"]["live_delta_events"], 3)
        self.assertEqual(report["top_events"][0]["event_slug"], "event-one")
        self.assertEqual(report["top_events"][0]["top_markets"][0]["market_slug"], "market-a")

    def test_build_watchlist_report_uses_trade_flow_inputs_when_present(self) -> None:
        registry = {
            "summary": {"events_seen": 1, "markets_seen": 2, "tracked_tokens": 4},
            "events": [
                {
                    "slug": "event-one",
                    "title": "Event One",
                    "markets": [
                        {"slug": "market-a", "question": "A?"},
                        {"slug": "market-b", "question": "B?"},
                    ],
                }
            ],
        }
        report = build_watchlist_report(
            registry=registry,
            snapshot_report={"summary": {"books_fetched": 2}},
            live_delta_report={"summary": {"markets_updated": 2, "spread_changes": 0, "liquidity_changes": 2}},
            delta_events=[
                {"market_slug": "market-a", "spread_changed": False, "liquidity_changed": True, "completeness_changed": False},
                {"market_slug": "market-b", "spread_changed": False, "liquidity_changed": True, "completeness_changed": False},
            ],
            trade_flow_report={
                "top_markets": [
                    {"market_slug": "market-b", "trade_count": 4, "side_imbalance": 0.5, "burst_intensity": 2, "spread_compression_after_trades": 1, "liquidity_refill_after_trades": 1, "trade_flow_score": 8.5}
                ],
                "top_events": [],
            },
        )

        self.assertEqual(report["primary_ranking_level"], "event")
        self.assertEqual(report["top_markets"][0]["market_slug"], "market-b")
        self.assertEqual(report["top_events"][0]["top_markets"][0]["trade_count"], 4)
        self.assertEqual(report["summary"]["trade_flow_markets"], 1)

    def test_write_watchlist_report_outputs_latest_and_timestamped_files(self) -> None:
        report = {"paper_only": True, "summary": {"markets_ranked": 1}, "top_markets": [], "top_events": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = write_watchlist_report(out_dir=Path(tmp_dir), report=report)
            self.assertTrue(written["report_path"].exists())
            self.assertTrue(written["latest_report_path"].exists())

    def test_load_live_delta_events_returns_latest_first_in_chronological_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            store = ResearchStore(f"sqlite:///{db_path.as_posix()}")
            try:
                store.save_raw_snapshot(
                    "clob_live_delta",
                    "tok-1:a",
                    {"market_slug": "market-a", "sequence": 1},
                    datetime(2026, 3, 17, 5, 0, 0, tzinfo=timezone.utc),
                )
                store.save_raw_snapshot(
                    "clob_live_delta",
                    "tok-1:b",
                    {"market_slug": "market-a", "sequence": 2},
                    datetime(2026, 3, 17, 5, 0, 1, tzinfo=timezone.utc),
                )

                from src.intelligence.watchlist_ranker import load_live_delta_events

                events = load_live_delta_events(store, limit=10)
                self.assertEqual([event["sequence"] for event in events], [1, 2])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
