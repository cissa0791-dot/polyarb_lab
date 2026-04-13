from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.core.models import BookLevel, OrderBook
from src.core.orderbook_validation import build_fetch_failure_validation, validate_orderbook
from src.runtime.runner import ResearchRunner


def build_book(
    token_id: str,
    *,
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    metadata: dict | None = None,
) -> OrderBook:
    bid_levels = [BookLevel(price=price, size=size) for price, size in (bids or [])]
    ask_levels = [BookLevel(price=price, size=size) for price, size in (asks or [])]
    return OrderBook(
        token_id=token_id,
        bids=bid_levels,
        asks=ask_levels,
        ts=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


class StaticClob:
    def __init__(self, books: dict[str, OrderBook]):
        self.books = books

    def get_book(self, token_id: str) -> OrderBook:
        return self.books[token_id]

    def prefetch_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        return {tid: self.books[tid] for tid in token_ids if tid in self.books}


class OrderBookValidationTests(unittest.TestCase):
    def test_buy_validation_accepts_one_sided_book_if_asks_exist(self) -> None:
        book = build_book(
            "yes-token",
            asks=[(0.40, 10.0)],
            metadata={
                "raw_bids_count": 0,
                "raw_asks_count": 1,
                "normalized_bids_count": 0,
                "normalized_asks_count": 1,
            },
        )

        result = validate_orderbook(book, required_action="BUY")

        self.assertTrue(result.passed)
        self.assertIsNone(result.reason_code)

    def test_buy_validation_classifies_empty_asks_precisely(self) -> None:
        book = build_book(
            "no-token",
            bids=[(0.99, 50.0)],
            metadata={
                "raw_bids_count": 50,
                "raw_asks_count": 0,
                "normalized_bids_count": 50,
                "normalized_asks_count": 0,
            },
        )

        result = validate_orderbook(book, required_action="BUY")

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_code, "EMPTY_ASKS")
        self.assertEqual(result.problem_stage, "validate")

    def test_validation_flags_crossed_book(self) -> None:
        book = build_book(
            "crossed-token",
            bids=[(0.60, 10.0)],
            asks=[(0.59, 10.0)],
            metadata={
                "raw_bids_count": 1,
                "raw_asks_count": 1,
                "normalized_bids_count": 1,
                "normalized_asks_count": 1,
            },
        )

        result = validate_orderbook(book, required_action="BUY")

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_code, "CROSSED_BOOK")

    def test_validation_flags_non_monotonic_book(self) -> None:
        book = build_book(
            "non-monotonic-token",
            bids=[(0.40, 10.0), (0.41, 5.0)],
            asks=[(0.45, 10.0), (0.44, 5.0)],
            metadata={
                "raw_bids_count": 2,
                "raw_asks_count": 2,
                "normalized_bids_count": 2,
                "normalized_asks_count": 2,
            },
        )

        result = validate_orderbook(book, required_action="BUY")

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_code, "NON_MONOTONIC_BOOK")
        self.assertEqual(result.problem_stage, "normalize")

    def test_validation_flags_malformed_required_side(self) -> None:
        book = build_book(
            "bad-token",
            metadata={
                "raw_bids_count": 1,
                "raw_asks_count": 2,
                "malformed_ask_levels": 2,
                "normalized_bids_count": 1,
                "normalized_asks_count": 0,
            },
        )

        result = validate_orderbook(book, required_action="BUY")

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_code, "MALFORMED_PRICE_LEVEL")
        self.assertEqual(result.problem_stage, "parse")

    def test_validation_flags_non_positive_touch_depth(self) -> None:
        book = build_book(
            "zero-token",
            metadata={
                "raw_asks_count": 2,
                "normalized_asks_count": 0,
                "non_positive_ask_levels": 2,
            },
        )

        result = validate_orderbook(book, required_action="BUY")

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_code, "NO_TOUCH_DEPTH")

    def test_fetch_failure_classifies_missing_orderbook(self) -> None:
        result = build_fetch_failure_validation("missing-token", Exception("No orderbook exists for the requested token id"))

        self.assertFalse(result.passed)
        self.assertEqual(result.reason_code, "MISSING_ORDERBOOK")
        self.assertEqual(result.problem_stage, "fetch")

    def test_runner_exports_invalid_orderbook_debug_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings_path = tmp_path / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        'gamma_host: "https://gamma-api.polymarket.com"',
                        'clob_host: "https://clob.polymarket.com"',
                        "market_limit: 1",
                        'sqlite_url: "sqlite:///' + str((tmp_path / "test.db").as_posix()) + '"',
                        "starting_cash: 1000.0",
                        'log_level: "DEBUG"',
                    ]
                ),
                encoding="utf-8",
            )
            constraints_path = tmp_path / "constraints.yaml"
            constraints_path.write_text("cross_market: []\n", encoding="utf-8")

            runner = ResearchRunner(
                settings_path=str(settings_path),
                constraints_path=str(constraints_path),
                debug_output_dir=tmp_path / "debug",
            )
            try:
                runner.clob = StaticClob(
                    {
                        "yes-token": build_book(
                            "yes-token",
                            asks=[(0.002, 100.0)],
                            metadata={
                                "raw_bids_count": 0,
                                "raw_asks_count": 1,
                                "normalized_bids_count": 0,
                                "normalized_asks_count": 1,
                            },
                        ),
                        "no-token": build_book(
                            "no-token",
                            bids=[(0.998, 100.0)],
                            metadata={
                                "raw_bids_count": 1,
                                "raw_asks_count": 0,
                                "normalized_bids_count": 1,
                                "normalized_asks_count": 0,
                            },
                        ),
                    }
                )

                markets = [
                    {
                        "slug": "test-market",
                        "question": "Test market?",
                        "outcomes": ["YES", "NO"],
                        "clobTokenIds": ["yes-token", "no-token"],
                    }
                ]

                with patch("src.runtime.runner.fetch_markets", return_value=markets):
                    summary = runner.run_once()

                self.assertEqual(summary.rejection_reason_counts["EMPTY_ASKS"], 1)
                self.assertNotIn("INVALID_ORDERBOOK", summary.rejection_reason_counts)

                export_files = list((tmp_path / "debug").glob("*_invalid_orderbooks.jsonl"))
                self.assertEqual(len(export_files), 1)
                records = [json.loads(line) for line in export_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(record["market_slug"], "test-market")
                self.assertEqual(record["token_id"], "no-token")
                self.assertEqual(record["side"], "NO")
                self.assertEqual(record["reason_code"], "EMPTY_ASKS")
                self.assertEqual(record["problem_stage"], "validate")
                self.assertEqual(record["validation_rule"], "required_ask_side_empty")
                self.assertEqual(record["raw_asks_count"], 0)
                self.assertEqual(record["normalized_asks_count"], 0)
            finally:
                runner.store.close()
                runner.opportunity_store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
