from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.intelligence.live_feed import (
    build_live_delta_report,
    compute_book_delta,
    summarize_book,
    write_live_delta_report,
)


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, token_id: str, bids=None, asks=None):
        self.token_id = token_id
        self.bids = bids or []
        self.asks = asks or []


class MarketIntelligenceLiveFeedTests(unittest.TestCase):
    def test_summarize_book_extracts_top_of_book(self) -> None:
        book = Book(
            token_id="tok-1",
            bids=[Level(0.40, 12.0)],
            asks=[Level(0.42, 10.0)],
        )

        summary = summarize_book(book)

        self.assertEqual(summary["token_id"], "tok-1")
        self.assertEqual(summary["best_bid"], 0.40)
        self.assertEqual(summary["best_ask"], 0.42)
        self.assertEqual(summary["spread"], 0.02)
        self.assertTrue(summary["complete_top_of_book"])

    def test_compute_book_delta_flags_spread_liquidity_and_completeness_changes(self) -> None:
        previous = {
            "best_bid": 0.40,
            "best_ask": 0.42,
            "best_bid_size": 12.0,
            "best_ask_size": 10.0,
            "spread": 0.02,
            "bid_levels": 1,
            "ask_levels": 1,
            "complete_top_of_book": True,
        }
        current = {
            "best_bid": 0.39,
            "best_ask": None,
            "best_bid_size": 8.0,
            "best_ask_size": None,
            "spread": None,
            "bid_levels": 1,
            "ask_levels": 0,
            "complete_top_of_book": False,
        }

        delta = compute_book_delta(
            token_id="tok-1",
            market_slug="market-one",
            previous=previous,
            current=current,
            observed_ts=datetime.now(timezone.utc),
        )

        assert delta is not None
        self.assertTrue(delta["spread_changed"])
        self.assertTrue(delta["liquidity_changed"])
        self.assertTrue(delta["completeness_changed"])

    def test_build_and_write_live_delta_report(self) -> None:
        registry = {
            "summary": {"events_seen": 2, "markets_seen": 3, "tracked_tokens": 6},
        }
        delta_events = [
            {
                "event_type": "book_delta",
                "market_slug": "market-one",
                "spread_changed": True,
                "liquidity_changed": True,
                "completeness_changed": False,
            },
            {
                "event_type": "book_delta",
                "market_slug": "market-one",
                "spread_changed": False,
                "liquidity_changed": True,
                "completeness_changed": True,
            },
        ]

        report = build_live_delta_report(registry=registry, delta_events=delta_events)
        self.assertEqual(report["summary"]["delta_events"], 2)
        self.assertEqual(report["summary"]["markets_updated"], 1)
        self.assertEqual(report["summary"]["spread_changes"], 1)
        self.assertEqual(report["summary"]["liquidity_changes"], 2)
        self.assertEqual(report["summary"]["incomplete_or_missing_book_changes"], 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            written = write_live_delta_report(out_dir=Path(tmp_dir), report=report)
            self.assertTrue(written["report_path"].exists())
            self.assertTrue(written["latest_report_path"].exists())


if __name__ == "__main__":
    unittest.main()
