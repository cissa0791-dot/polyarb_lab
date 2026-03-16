from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.reporting.models import (
    CalibrationReport,
    CollectionEvidenceComparisonReport,
    CollectionEvidenceSnapshot,
    OfflineAnalyticsReport,
)


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value, ensure_ascii=False)
        else:
            serialized[key] = value
    return serialized


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    serialized_rows = [_serialize_row(row) for row in rows]
    fieldnames = sorted({key for row in serialized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized_rows)


def export_analytics_report(report: OfflineAnalyticsReport, out_dir: str | Path = "data/reports") -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    files["full_json"] = out_path / "analytics_report.json"
    _write_json(files["full_json"], report.model_dump(mode="json"))

    files["session_json"] = out_path / "session_summaries.json"
    files["session_csv"] = out_path / "session_summaries.csv"
    _write_json(files["session_json"], [item.model_dump(mode="json") for item in report.session_summaries])
    _write_csv(files["session_csv"], [item.model_dump(mode="json") for item in report.session_summaries])

    files["daily_json"] = out_path / "daily_summaries.json"
    files["daily_csv"] = out_path / "daily_summaries.csv"
    _write_json(files["daily_json"], [item.model_dump(mode="json") for item in report.daily_summaries])
    _write_csv(files["daily_csv"], [item.model_dump(mode="json") for item in report.daily_summaries])

    files["rejections_json"] = out_path / "rejection_leaderboard.json"
    files["rejections_csv"] = out_path / "rejection_leaderboard.csv"
    _write_json(files["rejections_json"], [item.model_dump(mode="json") for item in report.rejection_leaderboard])
    _write_csv(files["rejections_csv"], [item.model_dump(mode="json") for item in report.rejection_leaderboard])

    files["orderbook_failures_json"] = out_path / "orderbook_failure_rollups.json"
    files["orderbook_failures_csv"] = out_path / "orderbook_failure_rollups.csv"
    _write_json(files["orderbook_failures_json"], [item.model_dump(mode="json") for item in report.orderbook_failure_rollups])
    _write_csv(files["orderbook_failures_csv"], [item.model_dump(mode="json") for item in report.orderbook_failure_rollups])

    files["orderbook_funnel_json"] = out_path / "orderbook_funnel_reports.json"
    files["orderbook_funnel_csv"] = out_path / "orderbook_funnel_reports.csv"
    _write_json(files["orderbook_funnel_json"], [item.model_dump(mode="json") for item in report.orderbook_funnel_reports])
    _write_csv(files["orderbook_funnel_csv"], [item.model_dump(mode="json") for item in report.orderbook_funnel_reports])

    files["strategy_family_funnel_json"] = out_path / "strategy_family_funnel_reports.json"
    files["strategy_family_funnel_csv"] = out_path / "strategy_family_funnel_reports.csv"
    _write_json(
        files["strategy_family_funnel_json"],
        [item.model_dump(mode="json") for item in report.strategy_family_funnel_reports],
    )
    _write_csv(
        files["strategy_family_funnel_csv"],
        [item.model_dump(mode="json") for item in report.strategy_family_funnel_reports],
    )

    files["near_misses_json"] = out_path / "near_misses.json"
    files["near_misses_csv"] = out_path / "near_misses.csv"
    _write_json(files["near_misses_json"], [item.model_dump(mode="json") for item in report.near_misses])
    _write_csv(files["near_misses_csv"], [item.model_dump(mode="json") for item in report.near_misses])

    files["markout_stats_json"] = out_path / "markout_horizon_stats.json"
    files["markout_stats_csv"] = out_path / "markout_horizon_stats.csv"
    _write_json(files["markout_stats_json"], [item.model_dump(mode="json") for item in report.markout_horizon_stats])
    _write_csv(files["markout_stats_csv"], [item.model_dump(mode="json") for item in report.markout_horizon_stats])

    files["markout_obs_json"] = out_path / "markout_observations.json"
    files["markout_obs_csv"] = out_path / "markout_observations.csv"
    _write_json(files["markout_obs_json"], [item.model_dump(mode="json") for item in report.markout_observations])
    _write_csv(files["markout_obs_csv"], [item.model_dump(mode="json") for item in report.markout_observations])

    files["candidate_outcomes_json"] = out_path / "candidate_outcome_observations.json"
    files["candidate_outcomes_csv"] = out_path / "candidate_outcome_observations.csv"
    _write_json(files["candidate_outcomes_json"], [item.model_dump(mode="json") for item in report.candidate_outcome_observations])
    _write_csv(files["candidate_outcomes_csv"], [item.model_dump(mode="json") for item in report.candidate_outcome_observations])

    files["candidate_outcome_stats_json"] = out_path / "candidate_outcome_horizon_stats.json"
    files["candidate_outcome_stats_csv"] = out_path / "candidate_outcome_horizon_stats.csv"
    _write_json(files["candidate_outcome_stats_json"], [item.model_dump(mode="json") for item in report.candidate_outcome_horizon_stats])
    _write_csv(files["candidate_outcome_stats_csv"], [item.model_dump(mode="json") for item in report.candidate_outcome_horizon_stats])

    files["family_outcomes_json"] = out_path / "family_outcome_scorecards.json"
    files["family_outcomes_csv"] = out_path / "family_outcome_scorecards.csv"
    _write_json(files["family_outcomes_json"], [item.model_dump(mode="json") for item in report.family_outcome_scorecards])
    _write_csv(files["family_outcomes_csv"], [item.model_dump(mode="json") for item in report.family_outcome_scorecards])

    files["opportunity_type_outcomes_json"] = out_path / "opportunity_type_outcome_scorecards.json"
    files["opportunity_type_outcomes_csv"] = out_path / "opportunity_type_outcome_scorecards.csv"
    _write_json(files["opportunity_type_outcomes_json"], [item.model_dump(mode="json") for item in report.opportunity_type_outcome_scorecards])
    _write_csv(files["opportunity_type_outcomes_csv"], [item.model_dump(mode="json") for item in report.opportunity_type_outcome_scorecards])

    files["rank_bucket_outcomes_json"] = out_path / "rank_bucket_outcome_scorecards.json"
    files["rank_bucket_outcomes_csv"] = out_path / "rank_bucket_outcome_scorecards.csv"
    _write_json(files["rank_bucket_outcomes_json"], [item.model_dump(mode="json") for item in report.rank_bucket_outcome_scorecards])
    _write_csv(files["rank_bucket_outcomes_csv"], [item.model_dump(mode="json") for item in report.rank_bucket_outcome_scorecards])

    files["parameter_set_outcomes_json"] = out_path / "parameter_set_outcome_scorecards.json"
    files["parameter_set_outcomes_csv"] = out_path / "parameter_set_outcome_scorecards.csv"
    _write_json(files["parameter_set_outcomes_json"], [item.model_dump(mode="json") for item in report.parameter_set_outcome_scorecards])
    _write_csv(files["parameter_set_outcomes_csv"], [item.model_dump(mode="json") for item in report.parameter_set_outcome_scorecards])

    files["experiment_outcomes_json"] = out_path / "experiment_outcome_scorecards.json"
    files["experiment_outcomes_csv"] = out_path / "experiment_outcome_scorecards.csv"
    _write_json(files["experiment_outcomes_json"], [item.model_dump(mode="json") for item in report.experiment_outcome_scorecards])
    _write_csv(files["experiment_outcomes_csv"], [item.model_dump(mode="json") for item in report.experiment_outcome_scorecards])

    files["candidate_source_outcomes_json"] = out_path / "candidate_source_outcome_scorecards.json"
    files["candidate_source_outcomes_csv"] = out_path / "candidate_source_outcome_scorecards.csv"
    _write_json(files["candidate_source_outcomes_json"], [item.model_dump(mode="json") for item in report.candidate_source_outcome_scorecards])
    _write_csv(files["candidate_source_outcomes_csv"], [item.model_dump(mode="json") for item in report.candidate_source_outcome_scorecards])

    files["shadow_execution_json"] = out_path / "shadow_execution_observations.json"
    files["shadow_execution_csv"] = out_path / "shadow_execution_observations.csv"
    _write_json(files["shadow_execution_json"], [item.model_dump(mode="json") for item in report.shadow_execution_observations])
    _write_csv(files["shadow_execution_csv"], [item.model_dump(mode="json") for item in report.shadow_execution_observations])

    files["shadow_family_json"] = out_path / "family_shadow_execution_scorecards.json"
    files["shadow_family_csv"] = out_path / "family_shadow_execution_scorecards.csv"
    _write_json(files["shadow_family_json"], [item.model_dump(mode="json") for item in report.family_shadow_execution_scorecards])
    _write_csv(files["shadow_family_csv"], [item.model_dump(mode="json") for item in report.family_shadow_execution_scorecards])

    files["shadow_rank_json"] = out_path / "rank_bucket_shadow_execution_scorecards.json"
    files["shadow_rank_csv"] = out_path / "rank_bucket_shadow_execution_scorecards.csv"
    _write_json(files["shadow_rank_json"], [item.model_dump(mode="json") for item in report.rank_bucket_shadow_execution_scorecards])
    _write_csv(files["shadow_rank_csv"], [item.model_dump(mode="json") for item in report.rank_bucket_shadow_execution_scorecards])

    files["shadow_parameter_json"] = out_path / "parameter_set_shadow_execution_scorecards.json"
    files["shadow_parameter_csv"] = out_path / "parameter_set_shadow_execution_scorecards.csv"
    _write_json(files["shadow_parameter_json"], [item.model_dump(mode="json") for item in report.parameter_set_shadow_execution_scorecards])
    _write_csv(files["shadow_parameter_csv"], [item.model_dump(mode="json") for item in report.parameter_set_shadow_execution_scorecards])

    files["readiness_json"] = out_path / "strategy_family_live_readiness.json"
    files["readiness_csv"] = out_path / "strategy_family_live_readiness.csv"
    _write_json(files["readiness_json"], [item.model_dump(mode="json") for item in report.strategy_family_live_readiness])
    _write_csv(files["readiness_csv"], [item.model_dump(mode="json") for item in report.strategy_family_live_readiness])

    files["sample_sufficiency_json"] = out_path / "sample_sufficiency_scorecards.json"
    files["sample_sufficiency_csv"] = out_path / "sample_sufficiency_scorecards.csv"
    _write_json(files["sample_sufficiency_json"], [item.model_dump(mode="json") for item in report.sample_sufficiency_scorecards])
    _write_csv(files["sample_sufficiency_csv"], [item.model_dump(mode="json") for item in report.sample_sufficiency_scorecards])

    files["stability_json"] = out_path / "stability_scorecards.json"
    files["stability_csv"] = out_path / "stability_scorecards.csv"
    _write_json(files["stability_json"], [item.model_dump(mode="json") for item in report.stability_scorecards])
    _write_csv(files["stability_csv"], [item.model_dump(mode="json") for item in report.stability_scorecards])

    files["promotion_json"] = out_path / "promotion_gate_reports.json"
    files["promotion_csv"] = out_path / "promotion_gate_reports.csv"
    _write_json(files["promotion_json"], [item.model_dump(mode="json") for item in report.promotion_gate_reports])
    _write_csv(files["promotion_csv"], [item.model_dump(mode="json") for item in report.promotion_gate_reports])

    files["watchlist_json"] = out_path / "family_watchlist.json"
    files["watchlist_csv"] = out_path / "family_watchlist.csv"
    _write_json(files["watchlist_json"], [item.model_dump(mode="json") for item in report.family_watchlist])
    _write_csv(files["watchlist_csv"], [item.model_dump(mode="json") for item in report.family_watchlist])

    files["promotion_blockers_json"] = out_path / "promotion_blockers.json"
    files["promotion_blockers_csv"] = out_path / "promotion_blockers.csv"
    _write_json(files["promotion_blockers_json"], [item.model_dump(mode="json") for item in report.promotion_blockers])
    _write_csv(files["promotion_blockers_csv"], [item.model_dump(mode="json") for item in report.promotion_blockers])

    files["campaign_summaries_json"] = out_path / "campaign_summaries.json"
    files["campaign_summaries_csv"] = out_path / "campaign_summaries.csv"
    _write_json(files["campaign_summaries_json"], [item.model_dump(mode="json") for item in report.campaign_summaries])
    _write_csv(files["campaign_summaries_csv"], [item.model_dump(mode="json") for item in report.campaign_summaries])

    files["campaign_progress_json"] = out_path / "campaign_progress_reports.json"
    files["campaign_progress_csv"] = out_path / "campaign_progress_reports.csv"
    _write_json(files["campaign_progress_json"], [item.model_dump(mode="json") for item in report.campaign_progress_reports])
    _write_csv(files["campaign_progress_csv"], [item.model_dump(mode="json") for item in report.campaign_progress_reports])

    files["family_evidence_json"] = out_path / "family_evidence_reports.json"
    files["family_evidence_csv"] = out_path / "family_evidence_reports.csv"
    _write_json(files["family_evidence_json"], [item.model_dump(mode="json") for item in report.family_evidence_reports])
    _write_csv(files["family_evidence_csv"], [item.model_dump(mode="json") for item in report.family_evidence_reports])

    files["coverage_gaps_json"] = out_path / "coverage_gap_reports.json"
    files["coverage_gaps_csv"] = out_path / "coverage_gap_reports.csv"
    _write_json(files["coverage_gaps_json"], [item.model_dump(mode="json") for item in report.coverage_gap_reports])
    _write_csv(files["coverage_gaps_csv"], [item.model_dump(mode="json") for item in report.coverage_gap_reports])

    files["collection_recommendations_json"] = out_path / "collection_recommendations.json"
    files["collection_recommendations_csv"] = out_path / "collection_recommendations.csv"
    _write_json(files["collection_recommendations_json"], [item.model_dump(mode="json") for item in report.collection_recommendations])
    _write_csv(files["collection_recommendations_csv"], [item.model_dump(mode="json") for item in report.collection_recommendations])

    files["campaign_priority_json"] = out_path / "campaign_priority_list.json"
    files["campaign_priority_csv"] = out_path / "campaign_priority_list.csv"
    _write_json(files["campaign_priority_json"], [item.model_dump(mode="json") for item in report.campaign_priority_list])
    _write_csv(files["campaign_priority_csv"], [item.model_dump(mode="json") for item in report.campaign_priority_list])

    files["evidence_targets_json"] = out_path / "evidence_target_trackers.json"
    files["evidence_targets_csv"] = out_path / "evidence_target_trackers.csv"
    _write_json(files["evidence_targets_json"], [item.model_dump(mode="json") for item in report.evidence_target_trackers])
    _write_csv(files["evidence_targets_csv"], [item.model_dump(mode="json") for item in report.evidence_target_trackers])

    files["collection_backlog_json"] = out_path / "collection_action_backlog.json"
    files["collection_backlog_csv"] = out_path / "collection_action_backlog.csv"
    _write_json(files["collection_backlog_json"], [item.model_dump(mode="json") for item in report.collection_action_backlog])
    _write_csv(files["collection_backlog_csv"], [item.model_dump(mode="json") for item in report.collection_action_backlog])

    files["market_rollups_json"] = out_path / "market_rollups.json"
    files["market_rollups_csv"] = out_path / "market_rollups.csv"
    _write_json(files["market_rollups_json"], [item.model_dump(mode="json") for item in report.market_rollups])
    _write_csv(files["market_rollups_csv"], [item.model_dump(mode="json") for item in report.market_rollups])

    files["strategy_family_rollups_json"] = out_path / "strategy_family_rollups.json"
    files["strategy_family_rollups_csv"] = out_path / "strategy_family_rollups.csv"
    _write_json(files["strategy_family_rollups_json"], [item.model_dump(mode="json") for item in report.strategy_family_rollups])
    _write_csv(files["strategy_family_rollups_csv"], [item.model_dump(mode="json") for item in report.strategy_family_rollups])

    files["top_ranked_opportunities_json"] = out_path / "top_ranked_opportunities.json"
    files["top_ranked_opportunities_csv"] = out_path / "top_ranked_opportunities.csv"
    _write_json(files["top_ranked_opportunities_json"], [item.model_dump(mode="json") for item in report.top_ranked_opportunities])
    _write_csv(files["top_ranked_opportunities_csv"], [item.model_dump(mode="json") for item in report.top_ranked_opportunities])

    latest_session = report.session_summaries[-1].model_dump(mode="json") if report.session_summaries else {}
    files["latest_session_json"] = out_path / "latest_session_summary.json"
    _write_json(files["latest_session_json"], latest_session)
    return files


def export_collection_snapshot(
    snapshot: CollectionEvidenceSnapshot,
    out_dir: str | Path = "data/reports",
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    files["full_json"] = out_path / "collection_evidence_snapshot.json"
    _write_json(files["full_json"], snapshot.model_dump(mode="json"))

    files["family_evidence_csv"] = out_path / "collection_snapshot_family_evidence.csv"
    _write_csv(files["family_evidence_csv"], [item.model_dump(mode="json") for item in snapshot.family_evidence_reports])

    files["promotion_csv"] = out_path / "collection_snapshot_promotion.csv"
    _write_csv(files["promotion_csv"], [item.model_dump(mode="json") for item in snapshot.promotion_gate_reports])

    files["evidence_targets_csv"] = out_path / "collection_snapshot_evidence_targets.csv"
    _write_csv(files["evidence_targets_csv"], [item.model_dump(mode="json") for item in snapshot.evidence_target_trackers])

    files["collection_backlog_csv"] = out_path / "collection_snapshot_backlog.csv"
    _write_csv(files["collection_backlog_csv"], [item.model_dump(mode="json") for item in snapshot.collection_action_backlog])

    return files


def export_collection_comparison_report(
    report: CollectionEvidenceComparisonReport,
    out_dir: str | Path = "data/reports",
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    files["full_json"] = out_path / "collection_evidence_comparison.json"
    _write_json(files["full_json"], report.model_dump(mode="json"))

    files["family_deltas_csv"] = out_path / "collection_evidence_family_deltas.csv"
    _write_csv(files["family_deltas_csv"], [item.model_dump(mode="json") for item in report.family_deltas])

    files["summary_json"] = out_path / "collection_evidence_comparison_summary.json"
    _write_json(
        files["summary_json"],
        {
            "baseline_label": report.baseline_label,
            "current_label": report.current_label,
            "families_compared": report.families_compared,
            "families_with_evidence_gain": report.families_with_evidence_gain,
            "backlog_reduction_count": report.backlog_reduction_count,
            "newly_promoted_families": report.newly_promoted_families,
            "still_blocked_families": report.still_blocked_families,
        },
    )

    return files


def export_calibration_report(report: CalibrationReport, out_dir: str | Path = "data/reports") -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    files["full_json"] = out_path / "calibration_report.json"
    _write_json(files["full_json"], report.model_dump(mode="json"))

    files["parameter_results_json"] = out_path / "calibration_parameter_results.json"
    files["parameter_results_csv"] = out_path / "calibration_parameter_results.csv"
    parameter_rows = [
        {
            "parameter_set_label": result.parameter_set_label,
            "total_records": result.total_records,
            "qualified_count": result.qualified_count,
            "rejected_count": result.rejected_count,
            "near_miss_count": result.near_miss_count,
            "qualified_by_family": result.qualified_by_family,
            "near_miss_by_family": result.near_miss_by_family,
            "rejection_reason_counts": result.rejection_reason_counts,
        }
        for result in report.parameter_results
    ]
    _write_json(files["parameter_results_json"], parameter_rows)
    _write_csv(files["parameter_results_csv"], parameter_rows)

    family_rows = []
    for result in report.parameter_results:
        for family in result.family_summaries:
            row = family.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            family_rows.append(row)
    files["family_summaries_json"] = out_path / "calibration_family_summaries.json"
    files["family_summaries_csv"] = out_path / "calibration_family_summaries.csv"
    _write_json(files["family_summaries_json"], family_rows)
    _write_csv(files["family_summaries_csv"], family_rows)

    top_rows = []
    for result in report.parameter_results:
        for ranked in result.top_ranked_opportunities:
            row = ranked.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            top_rows.append(row)
    files["top_ranked_json"] = out_path / "calibration_top_ranked.json"
    files["top_ranked_csv"] = out_path / "calibration_top_ranked.csv"
    _write_json(files["top_ranked_json"], top_rows)
    _write_csv(files["top_ranked_csv"], top_rows)

    outcome_rows = []
    family_outcome_rows = []
    rank_bucket_rows = []
    for result in report.parameter_results:
        for stat in result.outcome_horizon_stats:
            row = stat.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            outcome_rows.append(row)
        for card in result.outcome_family_scorecards:
            row = card.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            family_outcome_rows.append(row)
        for card in result.outcome_rank_bucket_scorecards:
            row = card.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            rank_bucket_rows.append(row)

    files["outcome_horizons_json"] = out_path / "calibration_outcome_horizon_stats.json"
    files["outcome_horizons_csv"] = out_path / "calibration_outcome_horizon_stats.csv"
    _write_json(files["outcome_horizons_json"], outcome_rows)
    _write_csv(files["outcome_horizons_csv"], outcome_rows)

    files["outcome_families_json"] = out_path / "calibration_outcome_family_scorecards.json"
    files["outcome_families_csv"] = out_path / "calibration_outcome_family_scorecards.csv"
    _write_json(files["outcome_families_json"], family_outcome_rows)
    _write_csv(files["outcome_families_csv"], family_outcome_rows)

    files["outcome_rank_buckets_json"] = out_path / "calibration_outcome_rank_bucket_scorecards.json"
    files["outcome_rank_buckets_csv"] = out_path / "calibration_outcome_rank_bucket_scorecards.csv"
    _write_json(files["outcome_rank_buckets_json"], rank_bucket_rows)
    _write_csv(files["outcome_rank_buckets_csv"], rank_bucket_rows)

    shadow_summary_rows = []
    shadow_family_rows = []
    shadow_rank_rows = []
    for result in report.parameter_results:
        if result.shadow_execution_summary is not None:
            row = result.shadow_execution_summary.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            shadow_summary_rows.append(row)
        for card in result.shadow_execution_family_scorecards:
            row = card.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            shadow_family_rows.append(row)
        for card in result.shadow_execution_rank_bucket_scorecards:
            row = card.model_dump(mode="json")
            row["parameter_set_label"] = result.parameter_set_label
            shadow_rank_rows.append(row)

    files["shadow_summary_json"] = out_path / "calibration_shadow_execution_summaries.json"
    files["shadow_summary_csv"] = out_path / "calibration_shadow_execution_summaries.csv"
    _write_json(files["shadow_summary_json"], shadow_summary_rows)
    _write_csv(files["shadow_summary_csv"], shadow_summary_rows)

    files["shadow_family_json"] = out_path / "calibration_shadow_execution_family_scorecards.json"
    files["shadow_family_csv"] = out_path / "calibration_shadow_execution_family_scorecards.csv"
    _write_json(files["shadow_family_json"], shadow_family_rows)
    _write_csv(files["shadow_family_csv"], shadow_family_rows)

    files["shadow_rank_json"] = out_path / "calibration_shadow_execution_rank_bucket_scorecards.json"
    files["shadow_rank_csv"] = out_path / "calibration_shadow_execution_rank_bucket_scorecards.csv"
    _write_json(files["shadow_rank_json"], shadow_rank_rows)
    _write_csv(files["shadow_rank_csv"], shadow_rank_rows)

    return files
