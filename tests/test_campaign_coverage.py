from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import RunSummary
from src.opportunity.models import CandidateLeg, RankedOpportunity, StrategyFamily
from src.reporting.analytics import OfflineAnalyticsService
from src.storage.event_store import ResearchStore


def _book_payload(token_id: str, bid: float, ask: float) -> dict:
    return {
        "token_id": token_id,
        "bids": [{"price": bid, "size": 100.0}],
        "asks": [{"price": ask, "size": 100.0}],
    }


class CampaignCoverageAnalyticsTests(unittest.TestCase):
    def test_campaign_summaries_and_coverage_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "campaign_coverage.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

            run_specs = [
                {
                    "run_id": "run-a1",
                    "started": base,
                    "campaign_id": "camp-a",
                    "campaign_label": "campaign-a",
                    "experiment_label": "campaign-a:base",
                    "parameter_set_label": "base",
                    "raw_by_family": {"single_market_mispricing": 2},
                    "qualified_by_family": {"single_market_mispricing": 2},
                },
                {
                    "run_id": "run-a2",
                    "started": base + timedelta(days=1),
                    "campaign_id": "camp-a",
                    "campaign_label": "campaign-a",
                    "experiment_label": "campaign-a:strict",
                    "parameter_set_label": "strict",
                    "raw_by_family": {"single_market_mispricing": 1},
                    "qualified_by_family": {"single_market_mispricing": 1},
                },
                {
                    "run_id": "run-b1",
                    "started": base + timedelta(days=1, hours=1),
                    "campaign_id": "camp-b",
                    "campaign_label": "campaign-b",
                    "experiment_label": "campaign-b:base",
                    "parameter_set_label": "base",
                    "raw_by_family": {"cross_market_constraint": 1},
                    "qualified_by_family": {"cross_market_constraint": 1},
                },
                {
                    "run_id": "run-empty",
                    "started": base + timedelta(days=2),
                    "campaign_id": "camp-empty",
                    "campaign_label": "campaign-empty",
                    "experiment_label": "campaign-empty:base",
                    "parameter_set_label": "base",
                    "raw_by_family": {"external_belief": 1},
                    "qualified_by_family": {},
                },
            ]
            for spec in run_specs:
                store.save_run_summary(
                    RunSummary(
                        run_id=spec["run_id"],
                        started_ts=spec["started"],
                        ended_ts=spec["started"] + timedelta(minutes=2),
                        candidates_generated=sum(spec["qualified_by_family"].values()),
                        metadata={
                            "campaign_id": spec["campaign_id"],
                            "campaign_label": spec["campaign_label"],
                            "campaign_purpose": "coverage test",
                            "campaign_target_strategy_families": list(spec["raw_by_family"].keys()),
                            "campaign_target_parameter_sets": ["base", "strict"],
                            "experiment_label": spec["experiment_label"],
                            "parameter_set_label": spec["parameter_set_label"],
                            "raw_candidates_by_family": spec["raw_by_family"],
                            "qualified_candidates_by_family": spec["qualified_by_family"],
                        },
                    )
                )

            candidate_specs = [
                ("cand-a1", "single_market_mispricing", "run-a1", "camp-a", "campaign-a", "campaign-a:base", "base", base, 0.47),
                ("cand-a2", "single_market_mispricing", "run-a1", "camp-a", "campaign-a", "campaign-a:base", "base", base + timedelta(minutes=1), 0.48),
                ("cand-a3", "single_market_mispricing", "run-a2", "camp-a", "campaign-a", "campaign-a:strict", "strict", base + timedelta(days=1), 0.46),
                ("cand-b1", "cross_market_constraint", "run-b1", "camp-b", "campaign-b", "campaign-b:base", "base", base + timedelta(days=1, hours=1), 0.43),
            ]
            for candidate_id, family, run_id, campaign_id, campaign_label, experiment_label, parameter_set_label, ts, future_bid in candidate_specs:
                token_id = f"tok-{candidate_id}"
                market_slug = f"market-{candidate_id}"
                store.save_candidate(
                    RankedOpportunity(
                        strategy_id=f"{family}_strategy",
                        strategy_family=StrategyFamily(family),
                        candidate_id=candidate_id,
                        kind="single_market" if family == "single_market_mispricing" else "cross_market",
                        market_slugs=[market_slug],
                        gross_edge_cents=0.10,
                        fee_estimate_cents=0.01,
                        slippage_estimate_cents=0.01,
                        expected_payout=10.0,
                        target_notional_usd=10.0,
                        estimated_depth_usd=50.0,
                        score=92.0 if family == "single_market_mispricing" else 85.0,
                        estimated_net_profit_usd=0.8,
                        execution_mode="paper_eligible" if family == "single_market_mispricing" else "research_only",
                        research_only=family != "single_market_mispricing",
                        strategy_tag=f"{family}:test",
                        ranking_score=92.0 if family == "single_market_mispricing" else 85.0,
                        sizing_hint_usd=10.0,
                        sizing_hint_shares=10.0,
                        required_shares=10.0,
                        partial_fill_risk_score=0.10,
                        non_atomic_execution_risk_score=0.10,
                        legs=[
                            CandidateLeg(
                                token_id=token_id,
                                market_slug=market_slug,
                                action="BUY",
                                side="YES",
                                required_shares=10.0,
                                best_price=0.45,
                                vwap_price=0.45,
                                spread_cents=0.02,
                            )
                        ],
                        metadata={
                            "run_id": run_id,
                            "campaign_id": campaign_id,
                            "campaign_label": campaign_label,
                            "campaign_purpose": "coverage test",
                            "experiment_label": experiment_label,
                            "parameter_set_label": parameter_set_label,
                            "strategy_family": family,
                            "qualification": {
                                "expected_net_edge_cents": 0.08,
                                "expected_net_profit_usd": 0.8,
                                "legs": [
                                    {
                                        "token_id": token_id,
                                        "market_slug": market_slug,
                                        "action": "BUY",
                                        "side": "YES",
                                        "required_shares": 10.0,
                                        "best_price": 0.45,
                                        "vwap_price": 0.45,
                                        "spread_cents": 0.02,
                                    }
                                ],
                            },
                        },
                        ts=ts,
                    )
                )
                store.save_raw_snapshot("clob", token_id, _book_payload(token_id, 0.44, 0.45), ts)
                store.save_raw_snapshot("clob", token_id, _book_payload(token_id, future_bid, future_bid + 0.01), ts + timedelta(seconds=70))

            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            campaign_by_label = {item.campaign_label: item for item in report.campaign_summaries}
            self.assertEqual(campaign_by_label["campaign-a"].runs_count, 2)
            self.assertEqual(campaign_by_label["campaign-a"].qualified_candidate_count, 3)
            self.assertEqual(campaign_by_label["campaign-a"].distinct_parameter_sets_observed, 2)
            self.assertEqual(campaign_by_label["campaign-a"].recommendation_bucket, "collect_more_data")
            self.assertEqual(campaign_by_label["campaign-empty"].qualified_candidate_count, 0)
            self.assertEqual(campaign_by_label["campaign-empty"].recommendation_bucket, "deprioritize_collection")

            progress_by_label = {item.campaign_label: item for item in report.campaign_progress_reports}
            self.assertEqual(progress_by_label["campaign-a"].delta_qualified_candidate_count, 3)
            self.assertTrue(progress_by_label["campaign-a"].evidence_improved)
            self.assertEqual(progress_by_label["campaign-b"].previous_campaign_label, "campaign-a")
            self.assertEqual(progress_by_label["campaign-empty"].delta_qualified_candidate_count, -1)

            family_by_key = {item.strategy_family: item for item in report.family_evidence_reports}
            self.assertEqual(family_by_key["single_market_mispricing"].qualified_candidate_count, 3)
            self.assertEqual(family_by_key["single_market_mispricing"].distinct_campaigns, 1)
            self.assertEqual(family_by_key["single_market_mispricing"].distinct_parameter_sets, 2)
            self.assertEqual(family_by_key["single_market_mispricing"].raw_candidate_count, 3)
            self.assertEqual(family_by_key["cross_market_constraint"].distinct_campaigns, 1)
            self.assertEqual(family_by_key["cross_market_constraint"].qualified_candidate_count, 1)

            gaps_by_key = {item.subject_key: item for item in report.coverage_gap_reports}
            self.assertEqual(gaps_by_key["single_market_mispricing"].recommendation_bucket, "diversify_time_windows")
            self.assertGreater(gaps_by_key["single_market_mispricing"].missing_campaigns, 0)
            self.assertEqual(gaps_by_key["cross_market_constraint"].recommendation_bucket, "diversify_parameter_sets")

            family_recs = {
                item.subject_key: item
                for item in report.collection_recommendations
                if item.subject_type == "strategy_family"
            }
            self.assertEqual(family_recs["single_market_mispricing"].recommendation_bucket, "diversify_time_windows")
            self.assertEqual(family_recs["cross_market_constraint"].recommendation_bucket, "diversify_parameter_sets")

            tracker_by_key = {item.subject_key: item for item in report.evidence_target_trackers}
            self.assertEqual(tracker_by_key["single_market_mispricing"].missing_outcome_labels, 7)
            self.assertEqual(tracker_by_key["cross_market_constraint"].missing_parameter_sets, 1)
            self.assertIn("single_market_mispricing:base", tracker_by_key)

            backlog_by_key = {
                (item.subject_type, item.subject_key): item
                for item in report.collection_action_backlog
            }
            self.assertEqual(
                backlog_by_key[("strategy_family", "single_market_mispricing")].recommendation_bucket,
                "diversify_collection_needed",
            )
            self.assertEqual(
                backlog_by_key[("campaign", "campaign-empty")].recommendation_bucket,
                "low_yield_deprioritize",
            )

            campaign_recs = {item.subject_key: item for item in report.campaign_priority_list}
            self.assertEqual(campaign_recs["campaign-empty"].recommendation_bucket, "deprioritize_collection")
            self.assertIn("campaign-a", campaign_recs)

    def test_backward_safe_uncategorized_campaign_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "campaign_sparse.db"
            store = ResearchStore(f"sqlite:///{db_path}")
            base = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)
            store.save_run_summary(
                RunSummary(
                    run_id="run-legacy",
                    started_ts=base,
                    ended_ts=base + timedelta(minutes=1),
                    candidates_generated=0,
                    metadata={"raw_candidates_by_family": {"single_market_mispricing": 1}},
                )
            )
            store.close()

            service = OfflineAnalyticsService(db_path=db_path)
            report = service.build_report()
            service.close()

            self.assertEqual(len(report.campaign_summaries), 1)
            self.assertEqual(report.campaign_summaries[0].campaign_label, "uncategorized")
            self.assertEqual(report.campaign_summaries[0].recommendation_bucket, "deprioritize_collection")
            self.assertEqual(report.campaign_progress_reports[0].campaign_label, "uncategorized")
            self.assertEqual(len(report.evidence_target_trackers), 1)
            self.assertEqual(report.evidence_target_trackers[0].subject_key, "single_market_mispricing")


if __name__ == "__main__":
    unittest.main()
