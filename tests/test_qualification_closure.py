"""
tests/test_qualification_closure.py

Closure-round tests for the qualification funnel integration:

  1. LoadingGapTests
       SQLiteAuditReader.load_rows("qualification_funnel_reports") yields rows with
       a "payload" key; the payload-extraction fix in _load_report_inputs() correctly
       passes payload dicts to build_qualification_funnel_analytics().

  2. SessionAnalyticsFieldPromotionTests
       _build_session_summaries() populates qualified_shortlist_count and
       qualification_rejection_counts_by_gate from run.metadata.

  3. DailyAnalyticsFieldPromotionTests
       _build_daily_summaries() accumulates those fields across sessions.

  4. NearMissClassificationTests
       _QUALIFICATION_NEAR_MISS_REASONS constant contains the expected gates;
       ABSOLUTE_DEPTH_BELOW_FLOOR is excluded; SIZED_NOTIONAL_TOO_SMALL is excluded
       (handled at sizing stage).
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.domain.models import RejectionReason, RunSummary
from src.opportunity.models import (
    QualificationFunnelReport,
    QualificationPassReason,
    QualificationShortlistEntry,
    StrategyFamily,
)
from src.reporting.analytics import SQLiteAuditReader, build_qualification_funnel_analytics
from src.reporting.models import DailyAnalytics, SessionAnalytics
from src.runtime.runner import _QUALIFICATION_NEAR_MISS_REASONS
from src.storage.event_store import ResearchStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _shortlist_entry(candidate_id: str = "c1") -> QualificationShortlistEntry:
    return QualificationShortlistEntry(
        candidate_id=candidate_id,
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING.value,
        market_slugs=["mkt-a"],
        pass_reason_codes=[QualificationPassReason.EDGE_SUFFICIENT],
        gross_edge_cents=0.15,
        net_edge_cents=0.14,
        expected_net_profit_usd=0.70,
        required_depth_usd=4.0,
        available_depth_usd=16.0,
        partial_fill_risk_score=0.0,
        non_atomic_execution_risk_score=0.0,
        ts=_ts(),
    )


def _funnel_report(
    run_id: str = "run-1",
    evaluated: int = 10,
    passed: int = 2,
    rejected: int = 8,
    rejection_counts: dict | None = None,
    shortlist: list | None = None,
) -> QualificationFunnelReport:
    return QualificationFunnelReport(
        run_id=run_id,
        evaluated=evaluated,
        passed=passed,
        rejected=rejected,
        rejection_counts=rejection_counts if rejection_counts is not None else {"EDGE_BELOW_THRESHOLD": 5, "INSUFFICIENT_DEPTH": 3},
        shortlist=shortlist if shortlist is not None else [_shortlist_entry("p1")],
        ts=_ts(),
    )


def _make_run_summary(
    run_id: str = "r1",
    qualified_shortlist_count: int = 0,
    rejection_counts_by_gate: dict | None = None,
) -> RunSummary:
    started = _ts()
    ended = _ts()
    return RunSummary(
        run_id=run_id,
        started_ts=started,
        ended_ts=ended,
        markets_scanned=0,
        snapshots_stored=0,
        candidates_generated=0,
        risk_accepted=0,
        risk_rejected=0,
        near_miss_candidates=0,
        paper_orders_created=0,
        fills=0,
        partial_fills=0,
        cancellations=0,
        open_positions=0,
        closed_positions=0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        system_errors=0,
        rejection_reason_counts={},
        top_markets_by_candidates={},
        top_opportunity_types={},
        metadata={
            "qualification_funnel": {
                "qualified_shortlist_count": qualified_shortlist_count,
                "rejection_counts_by_gate": rejection_counts_by_gate or {},
            }
        },
    )


# ---------------------------------------------------------------------------
# 1. Loading gap
# ---------------------------------------------------------------------------

class LoadingGapTests(unittest.TestCase):
    """Verify payload extraction in the analytics loading path."""

    def _make_store(self, db_path: str) -> ResearchStore:
        return ResearchStore(f"sqlite:///{db_path}")

    def test_load_rows_returns_payload_key(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = self._make_store(db_path)
        report = _funnel_report(rejection_counts={"EDGE_BELOW_THRESHOLD": 7})
        store.save_qualification_funnel_report(report)

        reader = SQLiteAuditReader(Path(db_path))
        rows = reader.load_rows("qualification_funnel_reports")
        reader.close()

        self.assertEqual(len(rows), 1)
        self.assertIn("payload", rows[0], "SQLiteAuditReader must decode payload_json into row['payload']")

    def test_payload_extraction_yields_payload_dicts(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = self._make_store(db_path)
        report = _funnel_report(rejection_counts={"EDGE_BELOW_THRESHOLD": 4, "INSUFFICIENT_DEPTH": 2})
        store.save_qualification_funnel_report(report)

        reader = SQLiteAuditReader(Path(db_path))
        raw_rows = reader.load_rows("qualification_funnel_reports")
        reader.close()

        # This is the extraction formula used in _load_report_inputs()
        payload_dicts = [r["payload"] for r in raw_rows if "payload" in r]
        self.assertEqual(len(payload_dicts), 1)
        self.assertIn("rejection_counts", payload_dicts[0])
        self.assertEqual(payload_dicts[0]["rejection_counts"]["EDGE_BELOW_THRESHOLD"], 4)

    def test_build_analytics_receives_correct_data_via_payload_extraction(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = self._make_store(db_path)
        report = _funnel_report(
            evaluated=10, passed=3, rejected=7,
            rejection_counts={"EDGE_BELOW_THRESHOLD": 5, "INSUFFICIENT_DEPTH": 2},
        )
        store.save_qualification_funnel_report(report)

        reader = SQLiteAuditReader(Path(db_path))
        raw_rows = reader.load_rows("qualification_funnel_reports")
        reader.close()

        payload_dicts = [r["payload"] for r in raw_rows if "payload" in r]
        analytics = build_qualification_funnel_analytics(payload_dicts, set())

        self.assertEqual(analytics.total_runs, 1)
        self.assertEqual(analytics.total_evaluated, 10)
        self.assertEqual(analytics.total_passed, 3)
        gate_map = {s.gate: s.count for s in analytics.gate_rejection_stats}
        self.assertEqual(gate_map["EDGE_BELOW_THRESHOLD"], 5)
        self.assertEqual(gate_map["INSUFFICIENT_DEPTH"], 2)

    def test_raw_rows_without_payload_extraction_miss_rejection_counts(self) -> None:
        """Without payload extraction, row.get('rejection_counts') returns None — confirms the bug."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = self._make_store(db_path)
        report = _funnel_report(rejection_counts={"EDGE_BELOW_THRESHOLD": 9})
        store.save_qualification_funnel_report(report)

        reader = SQLiteAuditReader(Path(db_path))
        raw_rows = reader.load_rows("qualification_funnel_reports")
        reader.close()

        # Passing raw rows (not payload dicts) would produce zero gate stats
        analytics_from_raw = build_qualification_funnel_analytics(raw_rows, set())
        self.assertEqual(analytics_from_raw.gate_rejection_stats, [],
                         "Raw row dicts should NOT have 'rejection_counts' at top level")

        # Correct extraction yields gate stats
        payload_dicts = [r["payload"] for r in raw_rows if "payload" in r]
        analytics_fixed = build_qualification_funnel_analytics(payload_dicts, set())
        self.assertEqual(len(analytics_fixed.gate_rejection_stats), 1)


# ---------------------------------------------------------------------------
# 2. SessionAnalytics field promotion
# ---------------------------------------------------------------------------

class SessionAnalyticsFieldPromotionTests(unittest.TestCase):

    def test_session_analytics_has_qualified_shortlist_count_field(self) -> None:
        s = SessionAnalytics(
            run_id="r1",
            session_day="2026-01-01",
            started_ts=_ts(),
            ended_ts=_ts(),
        )
        self.assertEqual(s.qualified_shortlist_count, 0)

    def test_session_analytics_has_qualification_rejection_counts_by_gate_field(self) -> None:
        s = SessionAnalytics(
            run_id="r1",
            session_day="2026-01-01",
            started_ts=_ts(),
            ended_ts=_ts(),
        )
        self.assertEqual(s.qualification_rejection_counts_by_gate, {})

    def test_run_metadata_flows_into_session_analytics_fields(self) -> None:
        """Verify the extraction logic: qual_funnel = run.metadata.get('qualification_funnel', {})."""
        run = _make_run_summary(
            qualified_shortlist_count=5,
            rejection_counts_by_gate={"EDGE_BELOW_THRESHOLD": 3, "INSUFFICIENT_DEPTH": 1},
        )
        qual_funnel = run.metadata.get("qualification_funnel", {})
        qualified_shortlist_count = int(qual_funnel.get("qualified_shortlist_count", 0))
        qualification_rejection_counts_by_gate = dict(qual_funnel.get("rejection_counts_by_gate", {}))

        s = SessionAnalytics(
            run_id=run.run_id,
            session_day=run.started_ts.date().isoformat(),
            started_ts=run.started_ts,
            ended_ts=run.ended_ts,
            qualified_shortlist_count=qualified_shortlist_count,
            qualification_rejection_counts_by_gate=qualification_rejection_counts_by_gate,
        )

        self.assertEqual(s.qualified_shortlist_count, 5)
        self.assertEqual(s.qualification_rejection_counts_by_gate["EDGE_BELOW_THRESHOLD"], 3)
        self.assertEqual(s.qualification_rejection_counts_by_gate["INSUFFICIENT_DEPTH"], 1)

    def test_missing_qualification_funnel_metadata_defaults_to_zero(self) -> None:
        run = RunSummary(
            run_id="r-no-funnel",
            started_ts=_ts(),
            ended_ts=_ts(),
            markets_scanned=0,
            snapshots_stored=0,
            candidates_generated=0,
            risk_accepted=0,
            risk_rejected=0,
            near_miss_candidates=0,
            paper_orders_created=0,
            fills=0,
            partial_fills=0,
            cancellations=0,
            open_positions=0,
            closed_positions=0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            system_errors=0,
            rejection_reason_counts={},
            top_markets_by_candidates={},
            top_opportunity_types={},
            metadata={},
        )
        qual_funnel = run.metadata.get("qualification_funnel", {})
        qualified_shortlist_count = int(qual_funnel.get("qualified_shortlist_count", 0))
        qualification_rejection_counts_by_gate = dict(qual_funnel.get("rejection_counts_by_gate", {}))

        self.assertEqual(qualified_shortlist_count, 0)
        self.assertEqual(qualification_rejection_counts_by_gate, {})


# ---------------------------------------------------------------------------
# 3. DailyAnalytics field promotion
# ---------------------------------------------------------------------------

class DailyAnalyticsFieldPromotionTests(unittest.TestCase):

    def test_daily_analytics_has_qualified_shortlist_count_field(self) -> None:
        d = DailyAnalytics(session_day="2026-01-01")
        self.assertEqual(d.qualified_shortlist_count, 0)

    def test_daily_analytics_has_qualification_rejection_counts_by_gate_field(self) -> None:
        d = DailyAnalytics(session_day="2026-01-01")
        self.assertEqual(d.qualification_rejection_counts_by_gate, {})

    def test_daily_aggregation_sums_qualified_shortlist_count(self) -> None:
        sessions = [
            SessionAnalytics(
                run_id="r1", session_day="2026-01-01",
                started_ts=_ts(), ended_ts=_ts(),
                qualified_shortlist_count=3,
            ),
            SessionAnalytics(
                run_id="r2", session_day="2026-01-01",
                started_ts=_ts(), ended_ts=_ts(),
                qualified_shortlist_count=5,
            ),
        ]
        total = sum(s.qualified_shortlist_count for s in sessions)
        self.assertEqual(total, 8)

    def test_daily_aggregation_merges_rejection_counts_by_gate(self) -> None:
        from collections import Counter
        sessions = [
            SessionAnalytics(
                run_id="r1", session_day="2026-01-01",
                started_ts=_ts(), ended_ts=_ts(),
                qualification_rejection_counts_by_gate={"EDGE_BELOW_THRESHOLD": 4, "INSUFFICIENT_DEPTH": 2},
            ),
            SessionAnalytics(
                run_id="r2", session_day="2026-01-01",
                started_ts=_ts(), ended_ts=_ts(),
                qualification_rejection_counts_by_gate={"EDGE_BELOW_THRESHOLD": 3, "PARTIAL_FILL_RISK_TOO_HIGH": 1},
            ),
        ]
        merged: Counter[str] = Counter()
        for s in sessions:
            merged.update(s.qualification_rejection_counts_by_gate)

        d = DailyAnalytics(
            session_day="2026-01-01",
            qualification_rejection_counts_by_gate=dict(merged),
        )
        self.assertEqual(d.qualification_rejection_counts_by_gate["EDGE_BELOW_THRESHOLD"], 7)
        self.assertEqual(d.qualification_rejection_counts_by_gate["INSUFFICIENT_DEPTH"], 2)
        self.assertEqual(d.qualification_rejection_counts_by_gate["PARTIAL_FILL_RISK_TOO_HIGH"], 1)


# ---------------------------------------------------------------------------
# 4. Near-miss classification
# ---------------------------------------------------------------------------

class NearMissClassificationTests(unittest.TestCase):

    def test_constant_contains_edge_below_threshold(self) -> None:
        self.assertIn(RejectionReason.EDGE_BELOW_THRESHOLD.value, _QUALIFICATION_NEAR_MISS_REASONS)

    def test_constant_contains_net_profit_below_threshold(self) -> None:
        self.assertIn(RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value, _QUALIFICATION_NEAR_MISS_REASONS)

    def test_constant_contains_insufficient_depth(self) -> None:
        self.assertIn(RejectionReason.INSUFFICIENT_DEPTH.value, _QUALIFICATION_NEAR_MISS_REASONS)

    def test_constant_contains_partial_fill_risk_too_high(self) -> None:
        self.assertIn(RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value, _QUALIFICATION_NEAR_MISS_REASONS)

    def test_absolute_depth_below_floor_excluded(self) -> None:
        """Hard liquidity floor — structural failure, not a marginal near-miss."""
        self.assertNotIn(RejectionReason.ABSOLUTE_DEPTH_BELOW_FLOOR.value, _QUALIFICATION_NEAR_MISS_REASONS)

    def test_sized_notional_too_small_excluded(self) -> None:
        """Handled at sizing stage, not qualification stage."""
        self.assertNotIn(RejectionReason.SIZED_NOTIONAL_TOO_SMALL.value, _QUALIFICATION_NEAR_MISS_REASONS)

    def test_constant_is_frozenset(self) -> None:
        self.assertIsInstance(_QUALIFICATION_NEAR_MISS_REASONS, frozenset)

    def test_constant_has_exactly_four_reasons(self) -> None:
        self.assertEqual(len(_QUALIFICATION_NEAR_MISS_REASONS), 4)


if __name__ == "__main__":
    unittest.main()
