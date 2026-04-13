"""
tests/test_qualification_integration.py

Integration tests for the qualification funnel wiring:

  1. QualificationFunnelPersistenceTests
       save_qualification_funnel_report() / load_qualification_funnel_reports()
       using an in-memory SQLite store.

  2. RunSummaryBuilderFunnelTests
       record_qualification_funnel() populates qualified_shortlist_count and
       qualification_rejection_counts_by_gate; both appear in build() metadata.

  3. QualificationFunnelAnalyticsTests
       build_qualification_funnel_analytics() aggregates across multiple funnel
       rows, ranks gates by rejection count, and computes the shortlist-to-stored
       correlation rate.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.opportunity.models import (
    CandidateLeg,
    QualificationFunnelReport,
    QualificationPassReason,
    QualificationShortlistEntry,
    RawCandidate,
    StrategyFamily,
)
from src.reporting.analytics import build_qualification_funnel_analytics
from src.reporting.summary import RunSummaryBuilder, aggregate_run_summaries
from src.storage.event_store import ResearchStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> datetime:
    return datetime.now(timezone.utc)


def _shortlist_entry(candidate_id: str = "p1") -> QualificationShortlistEntry:
    return QualificationShortlistEntry(
        candidate_id=candidate_id,
        strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING.value,
        market_slugs=["mkt-a"],
        pass_reason_codes=[QualificationPassReason.EDGE_SUFFICIENT, QualificationPassReason.DEPTH_SUFFICIENT],
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
    shortlist: list[QualificationShortlistEntry] | None = None,
) -> QualificationFunnelReport:
    return QualificationFunnelReport(
        run_id=run_id,
        evaluated=evaluated,
        passed=passed,
        rejected=rejected,
        rejection_counts=rejection_counts if rejection_counts is not None else {"EDGE_BELOW_THRESHOLD": 5, "INSUFFICIENT_DEPTH": 3},
        shortlist=shortlist if shortlist is not None else [_shortlist_entry("p1"), _shortlist_entry("p2")],
        ts=_ts(),
    )


def _in_memory_store() -> ResearchStore:
    return ResearchStore("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# 1. Persistence
# ---------------------------------------------------------------------------

class QualificationFunnelPersistenceTests(unittest.TestCase):

    def test_empty_store_returns_empty_list(self) -> None:
        store = _in_memory_store()
        rows = store.load_qualification_funnel_reports()
        self.assertEqual(rows, [])

    def test_save_and_load_roundtrip_preserves_counts(self) -> None:
        store = _in_memory_store()
        report = _funnel_report(run_id="run-rt", evaluated=12, passed=3, rejected=9)
        store.save_qualification_funnel_report(report)

        rows = store.load_qualification_funnel_reports()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_id"], "run-rt")
        self.assertEqual(row["evaluated"], 12)
        self.assertEqual(row["passed"], 3)
        self.assertEqual(row["rejected"], 9)

    def test_save_and_load_roundtrip_preserves_rejection_counts(self) -> None:
        store = _in_memory_store()
        report = _funnel_report(
            rejection_counts={"EDGE_BELOW_THRESHOLD": 7, "PARTIAL_FILL_RISK_TOO_HIGH": 2}
        )
        store.save_qualification_funnel_report(report)

        rows = store.load_qualification_funnel_reports()
        rc = rows[0]["rejection_counts"]
        self.assertEqual(rc["EDGE_BELOW_THRESHOLD"], 7)
        self.assertEqual(rc["PARTIAL_FILL_RISK_TOO_HIGH"], 2)

    def test_save_and_load_roundtrip_preserves_shortlist(self) -> None:
        store = _in_memory_store()
        entry = _shortlist_entry("cand-xyz")
        report = _funnel_report(shortlist=[entry])
        store.save_qualification_funnel_report(report)

        rows = store.load_qualification_funnel_reports()
        shortlist = rows[0]["shortlist"]
        self.assertEqual(len(shortlist), 1)
        self.assertEqual(shortlist[0]["candidate_id"], "cand-xyz")

    def test_multiple_reports_all_loaded(self) -> None:
        store = _in_memory_store()
        store.save_qualification_funnel_report(_funnel_report(run_id="run-a"))
        store.save_qualification_funnel_report(_funnel_report(run_id="run-b"))
        store.save_qualification_funnel_report(_funnel_report(run_id="run-c"))

        rows = store.load_qualification_funnel_reports()
        self.assertEqual(len(rows), 3)
        run_ids = {r["run_id"] for r in rows}
        self.assertEqual(run_ids, {"run-a", "run-b", "run-c"})


# ---------------------------------------------------------------------------
# 2. RunSummaryBuilder
# ---------------------------------------------------------------------------

class RunSummaryBuilderFunnelTests(unittest.TestCase):

    def test_record_qualification_funnel_sets_shortlist_count(self) -> None:
        builder = RunSummaryBuilder(run_id="r1", started_ts=_ts())
        report = _funnel_report(passed=4)
        builder.record_qualification_funnel(report)

        self.assertEqual(builder.qualified_shortlist_count, 4)

    def test_record_qualification_funnel_accumulates_gate_rejection_counts(self) -> None:
        builder = RunSummaryBuilder(run_id="r1", started_ts=_ts())
        report = _funnel_report(
            rejection_counts={"EDGE_BELOW_THRESHOLD": 3, "INSUFFICIENT_DEPTH": 2}
        )
        builder.record_qualification_funnel(report)

        self.assertEqual(builder.qualification_rejection_counts_by_gate["EDGE_BELOW_THRESHOLD"], 3)
        self.assertEqual(builder.qualification_rejection_counts_by_gate["INSUFFICIENT_DEPTH"], 2)

    def test_funnel_data_appears_in_built_metadata(self) -> None:
        builder = RunSummaryBuilder(run_id="r1", started_ts=_ts())
        report = _funnel_report(
            passed=2,
            rejection_counts={"EDGE_BELOW_THRESHOLD": 5}
        )
        builder.record_qualification_funnel(report)
        summary = builder.build(ended_ts=_ts())

        qf = summary.metadata["qualification_funnel"]
        self.assertEqual(qf["qualified_shortlist_count"], 2)
        self.assertEqual(qf["rejection_counts_by_gate"]["EDGE_BELOW_THRESHOLD"], 5)

    def test_aggregate_run_summaries_carries_funnel_data(self) -> None:
        """aggregate_run_summaries must sum qualified_shortlist_count and merge gate counts."""
        def _make_summary(shortlist_count: int, edge_rejections: int) -> None:
            b = RunSummaryBuilder(run_id="rx", started_ts=_ts())
            b.record_qualification_funnel(_funnel_report(
                passed=shortlist_count,
                rejection_counts={"EDGE_BELOW_THRESHOLD": edge_rejections},
            ))
            return b.build(ended_ts=_ts())

        s1 = _make_summary(shortlist_count=3, edge_rejections=4)
        s2 = _make_summary(shortlist_count=2, edge_rejections=6)
        agg = aggregate_run_summaries("agg", s1.started_ts, _ts(), [s1, s2])

        qf = agg.metadata["qualification_funnel"]
        self.assertEqual(qf["qualified_shortlist_count"], 5)
        self.assertEqual(qf["rejection_counts_by_gate"]["EDGE_BELOW_THRESHOLD"], 10)


# ---------------------------------------------------------------------------
# 3. build_qualification_funnel_analytics
# ---------------------------------------------------------------------------

class QualificationFunnelAnalyticsTests(unittest.TestCase):

    def test_empty_funnel_rows_returns_zero_analytics(self) -> None:
        result = build_qualification_funnel_analytics([], set())
        self.assertEqual(result.total_runs, 0)
        self.assertEqual(result.total_evaluated, 0)
        self.assertEqual(result.total_passed, 0)
        self.assertEqual(result.total_rejected, 0)
        self.assertEqual(result.gate_rejection_stats, [])
        self.assertIsNone(result.shortlist_to_candidate_stored_rate)

    def test_pass_rate_computed_correctly(self) -> None:
        row = _funnel_report(evaluated=10, passed=4, rejected=6).model_dump(mode="json")
        result = build_qualification_funnel_analytics([row], set())
        self.assertAlmostEqual(result.pass_rate, 0.4, places=4)

    def test_gate_rejection_stats_sorted_by_count_descending(self) -> None:
        row = _funnel_report(
            rejection_counts={
                "EDGE_BELOW_THRESHOLD": 3,
                "INSUFFICIENT_DEPTH": 7,
                "PARTIAL_FILL_RISK_TOO_HIGH": 1,
            }
        ).model_dump(mode="json")
        result = build_qualification_funnel_analytics([row], set())

        counts = [s.count for s in result.gate_rejection_stats]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(result.gate_rejection_stats[0].gate, "INSUFFICIENT_DEPTH")

    def test_gate_stats_pct_of_rejections_sums_to_100(self) -> None:
        row = _funnel_report(
            rejection_counts={"EDGE_BELOW_THRESHOLD": 6, "INSUFFICIENT_DEPTH": 4}
        ).model_dump(mode="json")
        result = build_qualification_funnel_analytics([row], set())

        total_pct = sum(s.pct_of_rejections for s in result.gate_rejection_stats)
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_shortlist_to_stored_rate_none_when_no_shortlist_entries(self) -> None:
        row = _funnel_report(shortlist=[]).model_dump(mode="json")
        result = build_qualification_funnel_analytics([row], set())
        self.assertIsNone(result.shortlist_to_candidate_stored_rate)

    def test_shortlist_to_stored_rate_computed_correctly(self) -> None:
        entries = [_shortlist_entry("c1"), _shortlist_entry("c2"), _shortlist_entry("c3")]
        row = _funnel_report(shortlist=entries).model_dump(mode="json")
        # c1 and c2 reached sizing and were stored; c3 did not
        stored_ids = {"c1", "c2"}
        result = build_qualification_funnel_analytics([row], stored_ids)
        # 2 of 3 shortlist entries stored → rate = 0.6667
        self.assertAlmostEqual(result.shortlist_to_candidate_stored_rate, 2 / 3, places=3)

    def test_aggregates_across_multiple_runs(self) -> None:
        row1 = _funnel_report(evaluated=10, passed=2, rejection_counts={"EDGE_BELOW_THRESHOLD": 5}).model_dump(mode="json")
        row2 = _funnel_report(evaluated=8, passed=3, rejection_counts={"EDGE_BELOW_THRESHOLD": 3, "INSUFFICIENT_DEPTH": 2}).model_dump(mode="json")
        result = build_qualification_funnel_analytics([row1, row2], set())

        self.assertEqual(result.total_runs, 2)
        self.assertEqual(result.total_evaluated, 18)
        self.assertEqual(result.total_passed, 5)
        gate_map = {s.gate: s.count for s in result.gate_rejection_stats}
        self.assertEqual(gate_map["EDGE_BELOW_THRESHOLD"], 8)
        self.assertEqual(gate_map["INSUFFICIENT_DEPTH"], 2)


if __name__ == "__main__":
    unittest.main()
