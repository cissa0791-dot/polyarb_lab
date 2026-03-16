from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src.core.models import BookLevel, OrderBook
from src.domain.models import RejectionEvent, RunSummary
from src.reporting.analytics import OfflineAnalyticsService
from src.reporting.exporter import export_analytics_report
from src.runtime.runner import ResearchRunner
from src.storage.event_store import ResearchStore


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


class CountingStaticClob:
    def __init__(self, books: dict[str, OrderBook]):
        self.books = books
        self.calls: list[str] = []

    def get_book(self, token_id: str) -> OrderBook:
        self.calls.append(token_id)
        return self.books[token_id]


class OrderbookOperationalAnalyticsTests(unittest.TestCase):
    def test_orderbook_failure_rollups_and_funnel_reports_filter_to_post_fix_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "analytics.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            start = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)
            end = start + timedelta(minutes=5)

            store.save_run_summary(
                RunSummary(
                    run_id="run-fresh",
                    started_ts=start,
                    ended_ts=end,
                    markets_scanned=50,
                    snapshots_stored=100,
                    candidates_generated=0,
                    metadata={
                        "orderbook_funnel": {
                            "books_fetched": 100,
                            "books_structurally_valid": 97,
                            "books_execution_feasible": 94,
                            "raw_candidates_generated": 3,
                            "qualified_candidates": 1,
                            "books_skipped_due_to_recent_empty_asks": 2,
                        }
                    },
                )
            )
            store.save_run_summary(
                RunSummary(
                    run_id="run-old",
                    started_ts=start - timedelta(days=1),
                    ended_ts=end - timedelta(days=1),
                    markets_scanned=10,
                    snapshots_stored=20,
                    candidates_generated=0,
                    metadata={},
                )
            )

            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-fresh",
                    stage="candidate_filter",
                    reason_code="EMPTY_ASKS",
                    metadata={
                        "market_slug": "market-a",
                        "side": "NO",
                        "required_action": "BUY",
                        "strategy_family": "single_market_mispricing",
                        "validation_rule": "required_ask_side_empty",
                        "problem_stage": "validate",
                        "failure_class": "feasibility_failure",
                    },
                    ts=start + timedelta(minutes=1),
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-fresh",
                    stage="candidate_filter",
                    reason_code="EMPTY_ASKS",
                    metadata={
                        "market_slug": "market-a",
                        "side": "NO",
                        "required_action": "BUY",
                        "strategy_family": "single_market_mispricing",
                        "validation_rule": "required_ask_side_empty",
                        "problem_stage": "validate",
                        "failure_class": "feasibility_failure",
                    },
                    ts=start + timedelta(minutes=2),
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-fresh",
                    stage="candidate_filter",
                    reason_code="MALFORMED_PRICE_LEVEL",
                    metadata={
                        "market_slug": "market-b",
                        "side": "YES",
                        "required_action": "BUY",
                        "strategy_family": "single_market_mispricing",
                        "validation_rule": "all_required_side_levels_malformed",
                        "problem_stage": "parse",
                        "failure_class": "integrity_failure",
                    },
                    ts=start + timedelta(minutes=3),
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-old",
                    stage="candidate_filter",
                    reason_code="INVALID_ORDERBOOK",
                    metadata={"market_slug": "old-market", "strategy_family": "single_market_mispricing"},
                    ts=start - timedelta(days=1, minutes=1),
                )
            )

            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            try:
                report = service.build_report()
            finally:
                service.close()

            self.assertEqual(len(report.orderbook_failure_rollups), 2)
            rollups = {(item.market_slug, item.reason_code): item for item in report.orderbook_failure_rollups}
            self.assertEqual(rollups[("market-a", "EMPTY_ASKS")].count, 2)
            self.assertEqual(rollups[("market-a", "EMPTY_ASKS")].failure_class, "feasibility_failure")
            self.assertEqual(rollups[("market-b", "MALFORMED_PRICE_LEVEL")].failure_class, "integrity_failure")
            self.assertAlmostEqual(rollups[("market-a", "EMPTY_ASKS")].pct_of_rejections, 66.666667, places=5)

            self.assertEqual(len(report.orderbook_funnel_reports), 1)
            funnel = report.orderbook_funnel_reports[0]
            self.assertEqual(funnel.run_id, "run-fresh")
            self.assertEqual(funnel.books_fetched, 100)
            self.assertEqual(funnel.books_structurally_valid, 97)
            self.assertEqual(funnel.books_execution_feasible, 94)
            self.assertEqual(funnel.candidates_generated, 3)
            self.assertEqual(funnel.qualified_candidates, 1)
            self.assertEqual(funnel.books_skipped_due_to_recent_empty_asks, 2)

            with tempfile.TemporaryDirectory() as export_dir:
                written = export_analytics_report(report, out_dir=export_dir)
                self.assertIn("orderbook_failures_json", written)
                self.assertIn("orderbook_funnel_json", written)
                self.assertTrue(Path(written["orderbook_failures_json"]).exists())
                self.assertTrue(Path(written["orderbook_funnel_json"]).exists())

    def test_repeated_empty_asks_triggers_temporary_skip_heuristic(self) -> None:
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
                clob = CountingStaticClob(
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
                runner.clob = clob
                markets = [
                    {
                        "slug": "repeat-empty-asks",
                        "question": "Repeat empty asks?",
                        "outcomes": ["YES", "NO"],
                        "clobTokenIds": ["yes-token", "no-token"],
                    }
                ]

                with patch("src.runtime.runner.fetch_markets", return_value=markets):
                    first = runner.run_once()
                    second = runner.run_once()
                    third = runner.run_once()

                self.assertEqual(first.rejection_reason_counts["EMPTY_ASKS"], 1)
                self.assertEqual(second.rejection_reason_counts["EMPTY_ASKS"], 1)
                self.assertEqual(len(clob.calls), 4)
                self.assertEqual(third.metadata["orderbook_funnel"]["books_skipped_due_to_recent_empty_asks"], 1)
                self.assertEqual(third.metadata["orderbook_funnel"]["books_fetched"], 0)
            finally:
                runner.store.close()
                runner.opportunity_store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
