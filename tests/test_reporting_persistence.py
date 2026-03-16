from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone

from src.domain.models import (
    PositionMark,
    PositionState,
    RejectionEvent,
    RunSummary,
    TradeSummary,
)
from src.reporting.summary import aggregate_run_summaries
from src.storage.event_store import ResearchStore


class ReportingPersistenceTests(unittest.TestCase):
    def test_run_summary_aggregation(self) -> None:
        start = datetime.now(timezone.utc)
        summary_a = RunSummary(
            run_id="run-a",
            started_ts=start,
            ended_ts=start,
            markets_scanned=10,
            candidates_generated=2,
            risk_accepted=1,
            risk_rejected=1,
            paper_orders_created=2,
            fills=1,
            partial_fills=1,
            cancellations=1,
            realized_pnl=1.25,
            unrealized_pnl=0.5,
            rejection_reason_counts={"EDGE_BELOW_THRESHOLD": 2},
            top_markets_by_candidates={"market-a": 2},
            top_opportunity_types={"single_market": 2},
        )
        summary_b = RunSummary(
            run_id="run-b",
            started_ts=start,
            ended_ts=start,
            markets_scanned=5,
            candidates_generated=1,
            risk_accepted=1,
            risk_rejected=0,
            paper_orders_created=1,
            fills=1,
            realized_pnl=-0.25,
            rejection_reason_counts={"INSUFFICIENT_DEPTH": 1},
            top_markets_by_candidates={"market-b": 1},
            top_opportunity_types={"single_market": 1},
        )

        daily = aggregate_run_summaries("daily-1", start, start, [summary_a, summary_b])

        self.assertEqual(daily.markets_scanned, 15)
        self.assertEqual(daily.candidates_generated, 3)
        self.assertEqual(daily.rejection_reason_counts["EDGE_BELOW_THRESHOLD"], 2)
        self.assertEqual(daily.rejection_reason_counts["INSUFFICIENT_DEPTH"], 1)
        self.assertAlmostEqual(daily.realized_pnl, 1.0, places=6)

    def test_new_lifecycle_entities_persist_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"sqlite:///{tmp_dir}/test.db"
            store = ResearchStore(db_path)
            now = datetime.now(timezone.utc)

            store.save_position_event(
                position_id="pos-1",
                candidate_id="cand-1",
                event_type="position_opened",
                symbol="token-a",
                market_slug="market-a",
                state=PositionState.OPENED.value,
                reason_code=None,
                payload={"source": "test"},
                ts=now,
            )
            store.save_position_mark(
                PositionMark(
                    position_id="pos-1",
                    candidate_id="cand-1",
                    symbol="token-a",
                    market_slug="market-a",
                    state=PositionState.OPENED,
                    shares=5.0,
                    avg_entry_price=0.4,
                    source_bid=0.5,
                    source_ask=0.51,
                    mark_price=0.5,
                    marked_value_usd=2.5,
                    remaining_entry_cost_usd=2.0,
                    unrealized_pnl_usd=0.5,
                    age_sec=15.0,
                    ts=now,
                )
            )
            store.save_trade_summary(
                TradeSummary(
                    position_id="pos-1",
                    candidate_id="cand-1",
                    symbol="token-a",
                    market_slug="market-a",
                    state=PositionState.CLOSED,
                    entry_cost_usd=2.0,
                    exit_proceeds_usd=2.4,
                    fees_paid_usd=0.1,
                    realized_pnl_usd=0.3,
                    holding_duration_sec=30.0,
                    opened_ts=now,
                    closed_ts=now,
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-1",
                    candidate_id="cand-1",
                    stage="risk",
                    reason_code="EDGE_BELOW_THRESHOLD",
                    metadata={"detail": "test"},
                    ts=now,
                )
            )
            store.save_run_summary(
                RunSummary(
                    run_id="run-1",
                    started_ts=now,
                    ended_ts=now,
                    markets_scanned=1,
                    snapshots_stored=2,
                    candidates_generated=1,
                    risk_accepted=0,
                    risk_rejected=1,
                    near_miss_candidates=1,
                    paper_orders_created=0,
                    fills=0,
                    partial_fills=0,
                    cancellations=0,
                    open_positions=0,
                    closed_positions=1,
                    realized_pnl=0.3,
                    unrealized_pnl=0.0,
                    system_errors=0,
                    rejection_reason_counts={"EDGE_BELOW_THRESHOLD": 1},
                )
            )

            raw_conn = sqlite3.connect(f"{tmp_dir}/test.db")
            cur = raw_conn.cursor()
            counts = {
                table: cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("position_events", "position_marks", "trade_summaries", "rejection_events", "run_summaries")
            }
            raw_conn.close()
            store.close()

        self.assertEqual(counts["position_events"], 1)
        self.assertEqual(counts["position_marks"], 1)
        self.assertEqual(counts["trade_summaries"], 1)
        self.assertEqual(counts["rejection_events"], 1)
        self.assertEqual(counts["run_summaries"], 1)


if __name__ == "__main__":
    unittest.main()
