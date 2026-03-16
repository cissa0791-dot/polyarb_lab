from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import (
    ExecutionReport,
    OrderStatus,
    PositionMark,
    PositionState,
    RejectionEvent,
    RiskDecision,
    RiskStatus,
    RunSummary,
    TradeSummary,
)
from src.opportunity.models import RankedOpportunity, StrategyFamily
from src.reporting.analytics import OfflineAnalyticsService
from src.reporting.exporter import export_analytics_report
from src.storage.event_store import ResearchStore


class OfflineAnalyticsTests(unittest.TestCase):
    def test_analytics_aggregation_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "analytics.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            run1_start = base
            run1_end = base + timedelta(minutes=10)
            run2_start = base + timedelta(days=1)
            run2_end = run2_start + timedelta(minutes=10)

            store.save_run_summary(
                RunSummary(
                    run_id="run-1",
                    started_ts=run1_start,
                    ended_ts=run1_end,
                    markets_scanned=5,
                    snapshots_stored=11,
                    candidates_generated=2,
                    risk_accepted=1,
                    risk_rejected=1,
                    near_miss_candidates=1,
                    paper_orders_created=2,
                    fills=2,
                    partial_fills=0,
                    cancellations=0,
                    open_positions=0,
                    closed_positions=1,
                    realized_pnl=0.5,
                    unrealized_pnl=0.2,
                    system_errors=1,
                    metadata={
                        "raw_candidates_by_family": {"single_market_mispricing": 2, "cross_market_constraint": 1},
                        "qualified_candidates_by_family": {"single_market_mispricing": 1, "cross_market_constraint": 1},
                        "research_only_candidates_by_family": {"cross_market_constraint": 1},
                        "near_miss_by_family": {"single_market_mispricing": 1},
                        "rejection_reason_counts_by_family": {
                            "single_market_mispricing": {"EDGE_BELOW_THRESHOLD": 1, "INSUFFICIENT_DEPTH": 1}
                        },
                        "strategy_family_funnel": {
                            "single_market_mispricing": {
                                "markets_considered": 5,
                                "books_fetched": 10,
                                "books_structurally_valid": 10,
                                "books_execution_feasible": 9,
                                "raw_candidates_generated": 2,
                                "markets_with_any_signal": 1,
                                "rejection_reason_counts": {"EDGE_BELOW_THRESHOLD": 1, "INSUFFICIENT_DEPTH": 1},
                            },
                            "cross_market_constraint": {
                                "markets_considered": 2,
                                "books_fetched": 4,
                                "books_structurally_valid": 4,
                                "books_execution_feasible": 4,
                                "raw_candidates_generated": 1,
                                "markets_with_any_signal": 1,
                                "rejection_reason_counts": {},
                            },
                        },
                    },
                )
            )
            store.save_run_summary(
                RunSummary(
                    run_id="run-2",
                    started_ts=run2_start,
                    ended_ts=run2_end,
                    markets_scanned=4,
                    snapshots_stored=8,
                    candidates_generated=1,
                    risk_accepted=0,
                    risk_rejected=1,
                    near_miss_candidates=0,
                    paper_orders_created=1,
                    fills=0,
                    partial_fills=1,
                    cancellations=1,
                    open_positions=1,
                    closed_positions=1,
                    realized_pnl=-0.2,
                    unrealized_pnl=0.0,
                    system_errors=0,
                    metadata={
                        "raw_candidates_by_family": {"single_market_mispricing": 1},
                        "qualified_candidates_by_family": {"single_market_mispricing": 1},
                        "near_miss_by_family": {"single_market_mispricing": 1},
                        "rejection_reason_counts_by_family": {
                            "single_market_mispricing": {"NET_PROFIT_BELOW_THRESHOLD": 1}
                        },
                    },
                )
            )

            store.save_execution_report(
                ExecutionReport(
                    intent_id="intent-1",
                    position_id="pos-1",
                    status=OrderStatus.FILLED,
                    filled_size=5.0,
                    avg_fill_price=0.4,
                    ts=run1_start + timedelta(minutes=1),
                )
            )
            store.save_execution_report(
                ExecutionReport(
                    intent_id="intent-2",
                    position_id="pos-2",
                    status=OrderStatus.PARTIAL,
                    filled_size=2.0,
                    avg_fill_price=0.5,
                    ts=run2_start + timedelta(minutes=1),
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
                    exit_proceeds_usd=2.6,
                    fees_paid_usd=0.1,
                    realized_pnl_usd=0.5,
                    holding_duration_sec=120.0,
                    opened_ts=run1_start + timedelta(seconds=30),
                    closed_ts=run1_start + timedelta(minutes=2),
                )
            )
            store.save_trade_summary(
                TradeSummary(
                    position_id="pos-2",
                    candidate_id="cand-2",
                    symbol="token-b",
                    market_slug="market-b",
                    state=PositionState.FORCE_CLOSED,
                    entry_cost_usd=2.5,
                    exit_proceeds_usd=2.4,
                    fees_paid_usd=0.1,
                    realized_pnl_usd=-0.2,
                    holding_duration_sec=180.0,
                    opened_ts=run2_start + timedelta(seconds=20),
                    closed_ts=run2_start + timedelta(minutes=3),
                )
            )

            for age_sec, pnl, cost in ((35.0, 0.2, 2.0), (320.0, 1.0, 2.0)):
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
                        marked_value_usd=cost + pnl,
                        remaining_entry_cost_usd=cost,
                        unrealized_pnl_usd=pnl,
                        age_sec=age_sec,
                        ts=run1_start + timedelta(seconds=age_sec),
                    )
                )

            for age_sec, pnl, cost in ((70.0, -0.1, 2.5), (920.0, 0.3, 2.5)):
                store.save_position_mark(
                    PositionMark(
                        position_id="pos-2",
                        candidate_id="cand-2",
                        symbol="token-b",
                        market_slug="market-b",
                        state=PositionState.OPENED,
                        shares=5.0,
                        avg_entry_price=0.5,
                        source_bid=0.48,
                        source_ask=0.49,
                        mark_price=0.48,
                        marked_value_usd=cost + pnl,
                        remaining_entry_cost_usd=cost,
                        unrealized_pnl_usd=pnl,
                        age_sec=age_sec,
                        ts=run2_start + timedelta(seconds=age_sec),
                    )
                )

            store.save_position_event(
                position_id="pos-1",
                candidate_id="cand-1",
                event_type="position_closed",
                symbol="token-a",
                market_slug="market-a",
                state=PositionState.CLOSED.value,
                reason_code="TAKE_PROFIT",
                payload={"detail": "closed"},
                ts=run1_start + timedelta(minutes=2),
            )
            store.save_position_event(
                position_id="pos-2",
                candidate_id="cand-2",
                event_type="position_force_closed",
                symbol="token-b",
                market_slug="market-b",
                state=PositionState.FORCE_CLOSED.value,
                reason_code="RUN_END_FLATTEN",
                payload={"detail": "force_closed"},
                ts=run2_start + timedelta(minutes=3),
            )

            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-1",
                    stage="candidate_filter",
                    reason_code="EDGE_BELOW_THRESHOLD",
                    metadata={"market_slug": "market-a", "edge_cents": 0.028, "strategy_family": "single_market_mispricing"},
                    ts=run1_start + timedelta(minutes=1),
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-1",
                    candidate_id="cand-depth",
                    stage="risk",
                    reason_code="INSUFFICIENT_DEPTH",
                    metadata={"market_slug": "market-b", "strategy_family": "single_market_mispricing"},
                    ts=run1_start + timedelta(minutes=1, seconds=5),
                )
            )
            store.save_rejection_event(
                RejectionEvent(
                    run_id="run-2",
                    candidate_id="cand-profit",
                    stage="risk",
                    reason_code="NET_PROFIT_BELOW_THRESHOLD",
                    metadata={"market_slug": "market-c", "strategy_family": "single_market_mispricing"},
                    ts=run2_start + timedelta(minutes=1),
                )
            )

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-depth",
                    kind="single_market",
                    market_slugs=["market-b"],
                    gross_edge_cents=0.05,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    target_notional_usd=100.0,
                    estimated_depth_usd=99.5,
                    score=80.0,
                    estimated_net_profit_usd=1.0,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=80.0,
                    sizing_hint_usd=100.0,
                    sizing_hint_shares=50.0,
                    ts=run1_start,
                )
            )
            store.save_risk_decision(
                RiskDecision(
                    candidate_id="cand-depth",
                    status=RiskStatus.BLOCKED,
                    approved_notional_usd=0.0,
                    reason_codes=["INSUFFICIENT_DEPTH"],
                    ts=run1_start,
                )
            )

            store.save_candidate(
                RankedOpportunity(
                    strategy_id="single_market_sum_under_1",
                    strategy_family=StrategyFamily.SINGLE_MARKET_MISPRICING,
                    candidate_id="cand-profit",
                    kind="single_market",
                    market_slugs=["market-c"],
                    gross_edge_cents=0.06,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    target_notional_usd=50.0,
                    estimated_depth_usd=100.0,
                    score=90.0,
                    estimated_net_profit_usd=0.45,
                    execution_mode="paper_eligible",
                    research_only=False,
                    strategy_tag="single_market_mispricing:single_market_sum_under_1",
                    ranking_score=90.0,
                    sizing_hint_usd=50.0,
                    sizing_hint_shares=25.0,
                    ts=run2_start,
                )
            )
            store.save_candidate(
                RankedOpportunity(
                    strategy_id="cross_market_leq",
                    strategy_family=StrategyFamily.CROSS_MARKET_CONSTRAINT,
                    candidate_id="cand-cross",
                    kind="cross_market",
                    market_slugs=["market-a", "market-b"],
                    gross_edge_cents=0.20,
                    fee_estimate_cents=0.01,
                    slippage_estimate_cents=0.01,
                    target_notional_usd=25.0,
                    estimated_depth_usd=80.0,
                    score=88.0,
                    estimated_net_profit_usd=2.0,
                    execution_mode="research_only",
                    research_only=True,
                    strategy_tag="cross_market_constraint:cross_market_leq",
                    ranking_score=88.0,
                    sizing_hint_usd=25.0,
                    sizing_hint_shares=10.0,
                    ts=run1_start + timedelta(minutes=4),
                )
            )
            store.save_risk_decision(
                RiskDecision(
                    candidate_id="cand-profit",
                    status=RiskStatus.BLOCKED,
                    approved_notional_usd=0.0,
                    reason_codes=["NET_PROFIT_BELOW_THRESHOLD"],
                    ts=run2_start,
                )
            )

            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            self.assertEqual(len(report.session_summaries), 2)
            self.assertEqual(len(report.daily_summaries), 2)
            self.assertEqual(report.session_summaries[0].execution_reports_count, 1)
            self.assertEqual(report.session_summaries[0].trade_count, 1)
            self.assertEqual(report.session_summaries[0].win_count, 1)
            self.assertEqual(report.session_summaries[1].loss_count, 1)
            self.assertEqual(report.daily_summaries[0].runs_count, 1)
            self.assertAlmostEqual(report.daily_summaries[0].realized_pnl_total, 0.5, places=6)
            self.assertEqual(report.session_summaries[0].raw_candidates_by_family["cross_market_constraint"], 1)
            self.assertEqual(report.session_summaries[0].qualified_candidates_by_family["cross_market_constraint"], 1)
            self.assertEqual(report.daily_summaries[0].research_only_candidates_by_family["cross_market_constraint"], 1)
            self.assertEqual(len(report.strategy_family_funnel_reports), 2)
            family_funnels = {(item.run_id, item.strategy_family): item for item in report.strategy_family_funnel_reports}
            self.assertEqual(family_funnels[("run-1", "single_market_mispricing")].markets_considered, 5)
            self.assertEqual(family_funnels[("run-1", "single_market_mispricing")].books_execution_feasible, 9)
            self.assertEqual(family_funnels[("run-1", "single_market_mispricing")].raw_candidates_generated, 2)
            self.assertEqual(family_funnels[("run-1", "single_market_mispricing")].markets_with_any_signal, 1)
            self.assertEqual(
                family_funnels[("run-1", "single_market_mispricing")].rejection_reason_counts,
                {"EDGE_BELOW_THRESHOLD": 1, "INSUFFICIENT_DEPTH": 1},
            )
            self.assertEqual(family_funnels[("run-1", "cross_market_constraint")].campaign_label, None)

            reason_counts = {item.reason_code: item.count for item in report.rejection_leaderboard}
            self.assertEqual(reason_counts["EDGE_BELOW_THRESHOLD"], 1)
            self.assertEqual(reason_counts["INSUFFICIENT_DEPTH"], 1)
            self.assertEqual(reason_counts["NET_PROFIT_BELOW_THRESHOLD"], 1)

            near_miss_codes = [item.reason_code for item in report.near_misses]
            self.assertIn("EDGE_BELOW_THRESHOLD", near_miss_codes)
            self.assertIn("INSUFFICIENT_DEPTH", near_miss_codes)
            self.assertIn("NET_PROFIT_BELOW_THRESHOLD", near_miss_codes)

            markout_stats = {item.horizon_sec: item for item in report.markout_horizon_stats}
            self.assertEqual(markout_stats[30].sample_count, 2)
            self.assertEqual(markout_stats[60].sample_count, 2)
            self.assertEqual(markout_stats[300].sample_count, 2)
            self.assertEqual(markout_stats[900].sample_count, 1)
            self.assertAlmostEqual(markout_stats[30].mean_unrealized_pnl_usd or 0.0, 0.05, places=6)

            family_rollups = {item.strategy_family: item for item in report.strategy_family_rollups}
            self.assertEqual(family_rollups["cross_market_constraint"].raw_candidate_count, 1)
            self.assertEqual(family_rollups["cross_market_constraint"].qualified_candidate_count, 1)
            self.assertEqual(family_rollups["cross_market_constraint"].research_only_candidate_count, 1)
            self.assertAlmostEqual(family_rollups["cross_market_constraint"].average_ranking_score or 0.0, 88.0, places=6)

            top_ranked_ids = [item.candidate_id for item in report.top_ranked_opportunities]
            self.assertIn("cand-cross", top_ranked_ids)

            out_dir = Path(tmp_dir) / "reports"
            files = export_analytics_report(report, out_dir=out_dir)
            self.assertTrue(files["full_json"].exists())
            self.assertTrue(files["daily_csv"].exists())
            self.assertTrue(files["markout_stats_csv"].exists())
            self.assertTrue(files["strategy_family_funnel_json"].exists())
            self.assertTrue(files["strategy_family_funnel_csv"].exists())
            self.assertTrue(files["near_misses_csv"].exists())
            self.assertTrue(files["candidate_outcome_stats_csv"].exists())
            self.assertTrue(files["family_outcomes_csv"].exists())
            self.assertTrue(files["shadow_execution_csv"].exists())
            self.assertTrue(files["shadow_family_csv"].exists())
            self.assertTrue(files["readiness_csv"].exists())
            self.assertTrue(files["sample_sufficiency_csv"].exists())
            self.assertTrue(files["stability_csv"].exists())
            self.assertTrue(files["promotion_csv"].exists())
            self.assertTrue(files["watchlist_csv"].exists())
            self.assertTrue(files["promotion_blockers_csv"].exists())
            self.assertTrue(files["campaign_summaries_csv"].exists())
            self.assertTrue(files["campaign_progress_csv"].exists())
            self.assertTrue(files["family_evidence_csv"].exists())
            self.assertTrue(files["coverage_gaps_csv"].exists())
            self.assertTrue(files["collection_recommendations_csv"].exists())
            self.assertTrue(files["campaign_priority_csv"].exists())
            self.assertTrue(files["evidence_targets_csv"].exists())
            self.assertTrue(files["collection_backlog_csv"].exists())
            self.assertTrue(files["strategy_family_rollups_csv"].exists())
            self.assertTrue(files["top_ranked_opportunities_csv"].exists())

            full_report = json.loads(files["full_json"].read_text(encoding="utf-8"))
            self.assertEqual(len(full_report["session_summaries"]), 2)
            self.assertIn("sample_sufficiency_scorecards", full_report)
            self.assertIn("strategy_family_funnel_reports", full_report)
            self.assertIn("promotion_gate_reports", full_report)
            self.assertIn("family_watchlist", full_report)
            self.assertIn("promotion_blockers", full_report)
            self.assertIn("campaign_summaries", full_report)
            self.assertIn("campaign_progress_reports", full_report)
            self.assertIn("family_evidence_reports", full_report)
            self.assertIn("coverage_gap_reports", full_report)
            self.assertIn("collection_recommendations", full_report)
            self.assertIn("evidence_target_trackers", full_report)
            self.assertIn("collection_action_backlog", full_report)

    def test_sparse_database_is_backward_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "empty.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            self.assertEqual(report.session_summaries, [])
            self.assertEqual(report.daily_summaries, [])
            self.assertEqual(report.rejection_leaderboard, [])
            self.assertEqual(report.strategy_family_funnel_reports, [])
            self.assertEqual(report.markout_horizon_stats[0].sample_count, 0)
            self.assertEqual(report.sample_sufficiency_scorecards, [])
            self.assertEqual(report.stability_scorecards, [])
            self.assertEqual(report.promotion_gate_reports, [])

    def test_campaign_reports_distinguish_run_coverage_from_candidate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "campaign_coverage.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            started_ts = datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc)
            ended_ts = started_ts + timedelta(minutes=5)

            store.save_run_summary(
                RunSummary(
                    run_id="run-postfunnel",
                    started_ts=started_ts,
                    ended_ts=ended_ts,
                    markets_scanned=500,
                    snapshots_stored=1000,
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
                    metadata={
                        "campaign_id": "campaign-postfunnel",
                        "campaign_label": "single-market-broad-500-postfunnel",
                        "campaign_target_strategy_families": ["single_market_mispricing"],
                        "campaign_target_parameter_sets": ["runtime_default"],
                        "parameter_set_label": "runtime_default",
                        "strategy_family_funnel": {
                            "single_market_mispricing": {
                                "markets_considered": 500,
                                "books_fetched": 1000,
                                "books_structurally_valid": 1000,
                                "books_execution_feasible": 980,
                                "raw_candidates_generated": 0,
                                "markets_with_any_signal": 0,
                                "rejection_reason_counts": {"EMPTY_ASKS": 20},
                            }
                        },
                    },
                )
            )
            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            summary = next(
                item for item in report.campaign_summaries if item.campaign_label == "single-market-broad-500-postfunnel"
            )
            progress = next(
                item
                for item in report.campaign_progress_reports
                if item.campaign_label == "single-market-broad-500-postfunnel"
            )

            self.assertEqual(summary.qualified_candidate_count, 0)
            self.assertEqual(summary.distinct_parameter_sets_observed, 0)
            self.assertEqual(summary.distinct_families_coverage_observed, 1)
            self.assertEqual(summary.distinct_parameter_sets_coverage_observed, 1)
            self.assertEqual(summary.coverage_family_counts, {"single_market_mispricing": 1})
            self.assertEqual(summary.coverage_parameter_set_counts, {"runtime_default": 1})
            self.assertEqual(summary.metadata["missing_target_families"], ["single_market_mispricing"])
            self.assertEqual(summary.metadata["missing_target_parameter_sets"], ["runtime_default"])
            self.assertEqual(summary.metadata["missing_target_families_by_candidate_evidence"], ["single_market_mispricing"])
            self.assertEqual(summary.metadata["missing_target_parameter_sets_by_candidate_evidence"], ["runtime_default"])
            self.assertEqual(summary.metadata["missing_target_families_by_run_coverage"], [])
            self.assertEqual(summary.metadata["missing_target_parameter_sets_by_run_coverage"], [])

            self.assertEqual(progress.distinct_families_count, 0)
            self.assertEqual(progress.distinct_families_coverage_observed, 1)
            self.assertEqual(progress.distinct_parameter_sets_coverage_observed, 1)
            self.assertEqual(progress.family_qualified_counts, {})
            self.assertEqual(progress.family_coverage_counts, {"single_market_mispricing": 1})
            self.assertEqual(progress.parameter_set_coverage_counts, {"runtime_default": 1})


if __name__ == "__main__":
    unittest.main()
