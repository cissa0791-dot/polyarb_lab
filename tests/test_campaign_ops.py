from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.reporting.campaigns import build_collection_action_backlog
from src.reporting.models import (
    CampaignProgressReport,
    CampaignSummaryReport,
    CollectionActionBacklogEntry,
    EvidenceTargetTracker,
)


class CampaignOpsTests(unittest.TestCase):
    def test_collection_action_backlog_buckets_cover_core_ops_states(self) -> None:
        trackers = [
            EvidenceTargetTracker(
                subject_type="strategy_family",
                subject_key="single_market_mispricing",
                strategy_family="single_market_mispricing",
                current_readiness_bucket="research_promising",
                promotion_bucket="research_promising",
                qualified_candidate_count=9,
                outcome_labeled_count=8,
                shadow_labeled_count=8,
                distinct_runs=3,
                distinct_campaigns=2,
                distinct_parameter_sets=2,
                distinct_session_days=2,
                missing_qualified_candidates=1,
                missing_outcome_labels=2,
                missing_shadow_labels=2,
                missing_runs=0,
                missing_campaigns=0,
                missing_parameter_sets=0,
                missing_session_days=1,
                stability_evidence_status="mixed",
                target_progress_score=0.85,
            ),
            EvidenceTargetTracker(
                subject_type="strategy_family",
                subject_key="cross_market_constraint",
                strategy_family="cross_market_constraint",
                promotion_bucket="deprioritize_for_now",
                qualified_candidate_count=0,
                outcome_labeled_count=0,
                shadow_labeled_count=0,
                distinct_runs=1,
                distinct_campaigns=1,
                distinct_parameter_sets=1,
                distinct_session_days=1,
                missing_qualified_candidates=10,
                missing_outcome_labels=10,
                missing_shadow_labels=10,
                missing_runs=2,
                missing_campaigns=1,
                missing_parameter_sets=1,
                missing_session_days=2,
                stability_evidence_status="thin",
                target_progress_score=0.05,
            ),
            EvidenceTargetTracker(
                subject_type="family_parameter_set",
                subject_key="single_market_mispricing:base",
                strategy_family="single_market_mispricing",
                parameter_set_label="base",
                promotion_bucket="research_promising",
                qualified_candidate_count=3,
                outcome_labeled_count=3,
                shadow_labeled_count=3,
                distinct_runs=2,
                distinct_campaigns=1,
                distinct_parameter_sets=1,
                distinct_session_days=2,
                missing_qualified_candidates=2,
                missing_outcome_labels=2,
                missing_shadow_labels=2,
                missing_runs=0,
                missing_campaigns=0,
                missing_parameter_sets=0,
                missing_session_days=0,
                stability_evidence_status="slice_measured",
                target_progress_score=0.75,
            ),
        ]
        now = datetime(2026, 3, 15, tzinfo=timezone.utc)
        campaign_summaries = [
            CampaignSummaryReport(
                campaign_id="camp-a",
                campaign_label="campaign-a",
                runs_count=2,
                qualified_candidate_count=8,
                outcome_labeled_count=8,
                shadow_labeled_count=8,
                distinct_session_days=2,
                distinct_experiments=2,
                distinct_parameter_sets_observed=2,
                recommendation_bucket="collect_more_data",
            ),
            CampaignSummaryReport(
                campaign_id="camp-empty",
                campaign_label="campaign-empty",
                runs_count=1,
                qualified_candidate_count=0,
                outcome_labeled_count=0,
                shadow_labeled_count=0,
                distinct_session_days=1,
                recommendation_bucket="deprioritize_collection",
            ),
        ]
        campaign_progress_reports = [
            CampaignProgressReport(
                campaign_id="camp-a",
                campaign_label="campaign-a",
                first_started_ts=now,
                latest_started_ts=now,
                runs_count=2,
                qualified_candidate_count=8,
                outcome_labeled_count=8,
                shadow_labeled_count=8,
                distinct_families_count=1,
                delta_runs=1,
                delta_qualified_candidate_count=2,
                delta_outcome_labeled_count=2,
                delta_shadow_labeled_count=2,
                delta_distinct_families=0,
                evidence_improved=True,
            ),
            CampaignProgressReport(
                campaign_id="camp-empty",
                campaign_label="campaign-empty",
                previous_campaign_label="campaign-a",
                first_started_ts=now,
                latest_started_ts=now,
                runs_count=1,
                qualified_candidate_count=0,
                outcome_labeled_count=0,
                shadow_labeled_count=0,
                distinct_families_count=0,
                delta_runs=-1,
                delta_qualified_candidate_count=-8,
                delta_outcome_labeled_count=-8,
                delta_shadow_labeled_count=-8,
                delta_distinct_families=-1,
                evidence_improved=False,
            ),
        ]

        backlog = build_collection_action_backlog(campaign_summaries, campaign_progress_reports, trackers)
        backlog_by_key = {(item.subject_type, item.subject_key): item for item in backlog}

        self.assertEqual(
            backlog_by_key[("strategy_family", "single_market_mispricing")].recommendation_bucket,
            "diversify_collection_needed",
        )
        self.assertEqual(
            backlog_by_key[("strategy_family", "cross_market_constraint")].recommendation_bucket,
            "low_yield_deprioritize",
        )
        self.assertEqual(
            backlog_by_key[("family_parameter_set", "single_market_mispricing:base")].recommendation_bucket,
            "close_to_sufficient",
        )
        self.assertEqual(
            backlog_by_key[("campaign", "campaign-empty")].recommendation_bucket,
            "low_yield_deprioritize",
        )

    def test_backlog_is_backward_safe_with_empty_inputs(self) -> None:
        backlog = build_collection_action_backlog([], [], [])
        self.assertEqual(backlog, [])
        self.assertTrue(all(isinstance(item, CollectionActionBacklogEntry) for item in backlog))


if __name__ == "__main__":
    unittest.main()
