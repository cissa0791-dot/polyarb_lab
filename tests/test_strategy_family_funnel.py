from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.core.models import BookLevel, OrderBook
from src.runtime.runner import ResearchRunner


def build_book(
    token_id: str,
    *,
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    metadata: dict | None = None,
) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=[BookLevel(price=price, size=size) for price, size in (bids or [])],
        asks=[BookLevel(price=price, size=size) for price, size in (asks or [])],
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

    def reset_request_stats(self) -> None:
        pass

    def request_stats_snapshot(self) -> dict:
        return {"books_fetched": 0, "negative_cache_hits": 0, "negative_cache_expired_rechecks": 0}

    def fetch_simplified_markets(self, *, limit: int = 1000) -> list:
        return []

    def fetch_books_batch(self, token_ids: list, chunk_size: int = 100) -> dict:
        return {tid: self.books[tid] for tid in token_ids if tid in self.books}

    def fetch_midpoints_batch(self, token_ids: list) -> dict:
        return {}

    def fetch_spreads_batch(self, token_ids: list) -> dict:
        return {}

    def fetch_prices_batch(self, token_ids: list, side: str = "BUY") -> dict:
        return {}


class StrategyFamilyFunnelTests(unittest.TestCase):
    def test_single_market_family_funnel_is_populated_in_run_summary_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings_path = tmp_path / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        'gamma_host: "https://gamma-api.polymarket.com"',
                        'clob_host: "https://clob.polymarket.com"',
                        "market_limit: 1",
                        'sqlite_url: "sqlite:///' + str((tmp_path / "runner.db").as_posix()) + '"',
                        "starting_cash: 1000.0",
                        'log_level: "INFO"',
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
                            asks=[(0.01, 100.0)],
                            metadata={"raw_asks_count": 1, "normalized_asks_count": 1},
                        ),
                        "no-token": build_book(
                            "no-token",
                            bids=[(0.99, 100.0)],
                            metadata={
                                "raw_bids_count": 1,
                                "normalized_bids_count": 1,
                                "raw_asks_count": 0,
                                "normalized_asks_count": 0,
                            },
                        ),
                    }
                )
                markets = [
                    {
                        "slug": "funnel-market",
                        "question": "Funnel market?",
                        "outcomes": ["YES", "NO"],
                        "clobTokenIds": ["yes-token", "no-token"],
                    }
                ]

                with patch("src.runtime.runner.fetch_events", return_value=[]), \
                     patch("src.runtime.runner.fetch_markets_from_events", return_value=markets):
                    summary = runner.run_once()

                funnel = summary.metadata["strategy_family_funnel"]["single_market_mispricing"]
                self.assertEqual(funnel["markets_considered"], 1)
                self.assertEqual(funnel["books_fetched"], 2)
                self.assertEqual(funnel["books_structurally_valid"], 2)
                self.assertEqual(funnel["books_execution_feasible"], 1)
                self.assertEqual(funnel["raw_candidates_generated"], 0)
                self.assertEqual(funnel["markets_with_any_signal"], 0)
                self.assertEqual(funnel["rejection_reason_counts"], {"EMPTY_ASKS": 1})
            finally:
                runner.store.close()
                runner.opportunity_store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
