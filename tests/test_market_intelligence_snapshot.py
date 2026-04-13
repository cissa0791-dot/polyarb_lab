from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.intelligence.market_intelligence import (
    build_daily_paper_report,
    build_event_market_registry,
    collect_registry_token_ids,
    write_snapshot_outputs,
)


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, bids=None, asks=None):
        self.bids = bids or []
        self.asks = asks or []


class MarketIntelligenceSnapshotTests(unittest.TestCase):
    def test_build_event_market_registry_normalizes_binary_markets(self) -> None:
        events = [
            {"id": "evt-1", "slug": "event-one", "title": "Event One", "active": True, "closed": False},
        ]
        markets = [
            {
                "id": "mkt-1",
                "slug": "market-one",
                "question": "Market One?",
                "active": True,
                "closed": False,
                "enableOrderBook": True,
                "bestBid": 0.4,
                "bestAsk": 0.42,
                "liquidityNum": 1234.0,
                "volumeNum": 4321.0,
                "outcomes": '["Yes","No"]',
                "clobTokenIds": '["yes-1","no-1"]',
                "events": [{"id": "evt-1", "slug": "event-one"}],
            }
        ]

        registry = build_event_market_registry(events, markets)

        self.assertEqual(registry["summary"]["events_seen"], 1)
        self.assertEqual(registry["summary"]["markets_seen"], 1)
        self.assertEqual(registry["summary"]["tracked_tokens"], 2)
        event = registry["events"][0]
        self.assertEqual(event["market_count"], 1)
        market = event["markets"][0]
        self.assertTrue(market["is_binary_yes_no"])
        self.assertEqual(market["yes_token_id"], "yes-1")
        self.assertEqual(market["no_token_id"], "no-1")

    def test_report_uses_registry_and_books(self) -> None:
        registry = {
            "summary": {
                "events_seen": 1,
                "markets_seen": 1,
                "events_with_markets": 1,
                "binary_markets": 1,
                "orderbook_enabled_markets": 1,
                "tracked_tokens": 2,
                "neg_risk_events": 0,
            },
            "events": [
                {
                    "slug": "event-one",
                    "title": "Event One",
                    "market_count": 1,
                    "markets": [
                        {
                            "slug": "market-one",
                            "yes_token_id": "yes-1",
                        }
                    ],
                }
            ],
        }
        books = {
            "yes-1": Book(
                bids=[Level(0.40, 10.0)],
                asks=[Level(0.41, 10.0)],
            )
        }

        report = build_daily_paper_report(registry, books)

        self.assertTrue(report["paper_only"])
        self.assertEqual(report["summary"]["books_fetched"], 1)
        self.assertEqual(report["summary"]["markets_with_complete_top_of_book"], 1)
        self.assertEqual(report["top_events_by_book_coverage"][0]["event_slug"], "event-one")

    def test_collect_and_write_outputs(self) -> None:
        registry = {
            "summary": {"events_seen": 0, "markets_seen": 0, "tracked_tokens": 0},
            "events": [
                {
                    "markets": [
                        {"yes_token_id": "yes-1", "no_token_id": "no-1"},
                        {"yes_token_id": "yes-2", "no_token_id": "no-2"},
                    ]
                }
            ],
        }
        report = {"paper_only": True, "summary": {"books_fetched": 0}}

        token_ids = collect_registry_token_ids(registry)
        self.assertEqual(token_ids, ["yes-1", "no-1", "yes-2", "no-2"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            written = write_snapshot_outputs(
                out_dir=Path(tmp_dir),
                registry=registry,
                report=report,
            )
            self.assertTrue(written["registry_path"].exists())
            self.assertTrue(written["report_path"].exists())
            self.assertTrue(written["latest_registry_path"].exists())
            self.assertTrue(written["latest_report_path"].exists())


if __name__ == "__main__":
    unittest.main()
