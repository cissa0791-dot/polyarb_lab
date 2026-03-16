from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.reporting.campaigns import (
    build_collection_evidence_snapshot,
    compare_collection_evidence_snapshots,
)
from src.reporting.exporter import (
    export_collection_comparison_report,
    export_collection_snapshot,
)
from src.reporting.models import (
    CollectionActionBacklogEntry,
    EvidenceTargetTracker,
    FamilyEvidenceReport,
    OfflineAnalyticsReport,
    PromotionGateReport,
)
from src.runtime.campaigns import (
    build_campaign_manifest_from_preset,
    list_campaign_presets,
    load_campaign_manifest,
    save_campaign_manifest,
)


class ResearchOpsPackagingTests(unittest.TestCase):
    def test_campaign_preset_build_and_save_manifest(self) -> None:
        preset_names = {preset.preset_name for preset in list_campaign_presets()}
        self.assertIn("single_market_focus", preset_names)

        manifest = build_campaign_manifest_from_preset(
            "single_market_focus",
            campaign_label="single-market-week-1",
            cycles=3,
            market_limit=9,
        )

        self.assertEqual(manifest.campaign_label, "single-market-week-1")
        self.assertEqual(manifest.cycles, 3)
        self.assertEqual(manifest.market_limit, 9)
        self.assertEqual(manifest.metadata["campaign_preset"], "single_market_focus")

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "campaign.yaml"
            save_campaign_manifest(manifest, manifest_path)
            loaded = load_campaign_manifest(manifest_path)

        self.assertEqual(loaded.campaign_label, "single-market-week-1")
        self.assertEqual(loaded.cycles, 3)
        self.assertEqual(loaded.metadata["campaign_preset"], "single_market_focus")

    def test_campaign_manifest_rejects_non_positive_market_limit(self) -> None:
        with self.assertRaises(ValidationError):
            build_campaign_manifest_from_preset(
                "single_market_focus",
                campaign_label="bad-market-limit",
                market_limit=0,
            )

    def test_collection_snapshot_and_comparison_exports(self) -> None:
        now = datetime(2026, 3, 15, tzinfo=timezone.utc)
        baseline_report = OfflineAnalyticsReport(
            generated_ts=now,
            db_path="baseline.db",
            family_evidence_reports=[
                FamilyEvidenceReport(
                    strategy_family="single_market_mispricing",
                    current_readiness_bucket="research_promising",
                    promotion_bucket="insufficient_evidence",
                    qualified_candidate_count=4,
                    outcome_labeled_count=3,
                    shadow_labeled_count=3,
                    distinct_runs=2,
                    distinct_campaigns=1,
                    distinct_parameter_sets=1,
                    distinct_session_days=2,
                    blocker_codes=["LOW_OUTCOME_SAMPLE"],
                )
            ],
            promotion_gate_reports=[
                PromotionGateReport(
                    strategy_family="single_market_mispricing",
                    reference_horizon_label="1m",
                    reference_horizon_sec=60,
                    current_readiness_bucket="research_promising",
                    promotion_bucket="insufficient_evidence",
                    blocker_codes=["LOW_OUTCOME_SAMPLE"],
                )
            ],
            evidence_target_trackers=[
                EvidenceTargetTracker(
                    subject_type="strategy_family",
                    subject_key="single_market_mispricing",
                    strategy_family="single_market_mispricing",
                    current_readiness_bucket="research_promising",
                    promotion_bucket="insufficient_evidence",
                    qualified_candidate_count=4,
                    outcome_labeled_count=3,
                    shadow_labeled_count=3,
                    distinct_runs=2,
                    distinct_campaigns=1,
                    distinct_parameter_sets=1,
                    distinct_session_days=2,
                    missing_qualified_candidates=6,
                    missing_outcome_labels=7,
                    missing_shadow_labels=7,
                    missing_runs=1,
                    missing_campaigns=1,
                    missing_parameter_sets=1,
                    missing_session_days=1,
                    stability_evidence_status="thin",
                    target_progress_score=0.35,
                )
            ],
            collection_action_backlog=[
                CollectionActionBacklogEntry(
                    subject_type="strategy_family",
                    subject_key="single_market_mispricing",
                    strategy_family="single_market_mispricing",
                    recommendation_bucket="continue_collection",
                )
            ],
        )
        current_report = OfflineAnalyticsReport(
            generated_ts=now,
            db_path="current.db",
            family_evidence_reports=[
                FamilyEvidenceReport(
                    strategy_family="single_market_mispricing",
                    current_readiness_bucket="candidate_for_future_tiny_live_preparation",
                    promotion_bucket="research_promising",
                    qualified_candidate_count=9,
                    outcome_labeled_count=8,
                    shadow_labeled_count=8,
                    distinct_runs=3,
                    distinct_campaigns=2,
                    distinct_parameter_sets=2,
                    distinct_session_days=3,
                    blocker_codes=[],
                )
            ],
            promotion_gate_reports=[
                PromotionGateReport(
                    strategy_family="single_market_mispricing",
                    reference_horizon_label="1m",
                    reference_horizon_sec=60,
                    current_readiness_bucket="candidate_for_future_tiny_live_preparation",
                    promotion_bucket="research_promising",
                    blocker_codes=[],
                )
            ],
            evidence_target_trackers=[
                EvidenceTargetTracker(
                    subject_type="strategy_family",
                    subject_key="single_market_mispricing",
                    strategy_family="single_market_mispricing",
                    current_readiness_bucket="candidate_for_future_tiny_live_preparation",
                    promotion_bucket="research_promising",
                    qualified_candidate_count=9,
                    outcome_labeled_count=8,
                    shadow_labeled_count=8,
                    distinct_runs=3,
                    distinct_campaigns=2,
                    distinct_parameter_sets=2,
                    distinct_session_days=3,
                    missing_qualified_candidates=1,
                    missing_outcome_labels=2,
                    missing_shadow_labels=2,
                    missing_runs=0,
                    missing_campaigns=0,
                    missing_parameter_sets=0,
                    missing_session_days=0,
                    stability_evidence_status="mixed",
                    target_progress_score=0.82,
                )
            ],
            collection_action_backlog=[
                CollectionActionBacklogEntry(
                    subject_type="strategy_family",
                    subject_key="single_market_mispricing",
                    strategy_family="single_market_mispricing",
                    recommendation_bucket="close_to_sufficient",
                )
            ],
        )

        baseline = build_collection_evidence_snapshot(baseline_report, snapshot_label="baseline")
        current = build_collection_evidence_snapshot(current_report, snapshot_label="post-campaign")
        comparison = compare_collection_evidence_snapshots(baseline, current)

        self.assertEqual(comparison.families_compared, 1)
        self.assertEqual(comparison.families_with_evidence_gain, 1)
        self.assertEqual(comparison.backlog_reduction_count, 1)
        self.assertIn("single_market_mispricing", comparison.newly_promoted_families)
        self.assertEqual(comparison.family_deltas[0].delta_qualified_candidate_count, 5)
        self.assertEqual(comparison.family_deltas[0].delta_missing_outcome_labels, -5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_files = export_collection_snapshot(baseline, Path(tmp_dir) / "baseline")
            comparison_files = export_collection_comparison_report(comparison, Path(tmp_dir) / "comparison")
            self.assertTrue(snapshot_files["full_json"].exists())
            self.assertTrue(snapshot_files["collection_backlog_csv"].exists())
            self.assertTrue(comparison_files["full_json"].exists())
            self.assertTrue(comparison_files["family_deltas_csv"].exists())

    def test_snapshot_comparison_is_backward_safe_when_family_is_missing_in_baseline(self) -> None:
        now = datetime(2026, 3, 15, tzinfo=timezone.utc)
        baseline = build_collection_evidence_snapshot(
            OfflineAnalyticsReport(generated_ts=now, db_path="baseline.db"),
            snapshot_label="baseline",
        )
        current = build_collection_evidence_snapshot(
            OfflineAnalyticsReport(
                generated_ts=now,
                db_path="current.db",
                family_evidence_reports=[
                    FamilyEvidenceReport(
                        strategy_family="cross_market_constraint",
                        promotion_bucket="insufficient_evidence",
                        qualified_candidate_count=1,
                    )
                ],
            ),
            snapshot_label="current",
        )

        comparison = compare_collection_evidence_snapshots(baseline, current)

        self.assertEqual(comparison.families_compared, 1)
        self.assertEqual(comparison.family_deltas[0].strategy_family, "cross_market_constraint")
        self.assertEqual(comparison.family_deltas[0].delta_qualified_candidate_count, 1)


if __name__ == "__main__":
    unittest.main()
