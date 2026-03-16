from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from src.domain.models import RunSummary
from src.reporting.models import (
    CampaignProgressReport,
    CampaignSummaryReport,
    CollectionActionBacklogEntry,
    CollectionEvidenceComparisonReport,
    CollectionEvidenceDelta,
    CollectionEvidenceSnapshot,
    CollectionRecommendation,
    CoverageGapReport,
    EvidenceTargetTracker,
    FamilyEvidenceReport,
    OfflineAnalyticsReport,
    PromotionGateReport,
)
from src.reporting.outcomes import OutcomeEvaluationCandidate
from src.reporting.models import CandidateOutcomeObservation, ShadowExecutionObservation


DEFAULT_COVERAGE_TARGETS = {
    "qualified_candidates": 10,
    "outcome_labels": 10,
    "shadow_labels": 10,
    "runs": 3,
    "campaigns": 2,
    "parameter_sets": 2,
    "session_days": 3,
}

DEFAULT_PARAMETER_SET_TARGETS = {
    "qualified_candidates": 5,
    "outcome_labels": 5,
    "shadow_labels": 5,
    "runs": 2,
    "campaigns": 1,
    "parameter_sets": 1,
    "session_days": 2,
}


def build_campaign_summary_reports(
    run_summaries: Sequence[RunSummary],
    qualified_records: Sequence[OutcomeEvaluationCandidate],
    qualified_outcome_observations: Sequence[CandidateOutcomeObservation],
    qualified_shadow_observations: Sequence[ShadowExecutionObservation],
) -> list[CampaignSummaryReport]:
    run_by_id = {run.run_id: run for run in run_summaries}
    records_by_campaign: dict[str, list[OutcomeEvaluationCandidate]] = defaultdict(list)
    runs_by_campaign: dict[str, list[RunSummary]] = defaultdict(list)
    for record in qualified_records:
        records_by_campaign[_campaign_key_for_record(record, run_by_id)].append(record)
    for run in run_summaries:
        runs_by_campaign[_campaign_key(run.metadata)].append(run)

    campaigns = sorted(set(records_by_campaign) | set(runs_by_campaign))
    reports: list[CampaignSummaryReport] = []
    for key in campaigns:
        campaign_records = records_by_campaign.get(key, [])
        candidate_ids = {record.candidate_id for record in campaign_records}
        reference_horizon = _select_reference_horizon(candidate_ids, qualified_outcome_observations)
        campaign_outcomes = [
            observation
            for observation in qualified_outcome_observations
            if observation.candidate_id in candidate_ids and observation.horizon_sec == reference_horizon
        ]
        campaign_shadow = [
            observation
            for observation in qualified_shadow_observations
            if observation.candidate_id in candidate_ids
        ]
        family_counts = Counter(record.strategy_family for record in campaign_records)
        parameter_counts = Counter(_parameter_set_key(record) for record in campaign_records)
        coverage_family_counts = _campaign_coverage_family_counts(runs_by_campaign.get(key, []))
        coverage_parameter_set_counts = _campaign_coverage_parameter_set_counts(runs_by_campaign.get(key, []))
        target_families: set[str] = set()
        target_parameter_sets: set[str] = set()
        purpose = None
        notes = None
        experiments: set[str] = set()
        session_days: set[str] = set()
        for run in runs_by_campaign.get(key, []):
            target_families.update(str(item) for item in run.metadata.get("campaign_target_strategy_families", []) or [])
            target_parameter_sets.update(str(item) for item in run.metadata.get("campaign_target_parameter_sets", []) or [])
            if purpose is None:
                purpose = _string_or_none(run.metadata.get("campaign_purpose"))
            if notes is None:
                notes = _string_or_none(run.metadata.get("campaign_notes"))
            experiments.add(_experiment_key_from_metadata(run.metadata))
            session_days.add(run.started_ts.date().isoformat())
        for record in campaign_records:
            if purpose is None:
                purpose = _string_or_none(record.metadata.get("campaign_purpose"))
            if notes is None:
                notes = _string_or_none(record.metadata.get("campaign_notes"))
            experiments.add(_experiment_key(record))
            session_days.add(record.ts.date().isoformat())

        qualified_count = len(candidate_ids)
        outcome_labeled_count = len({observation.candidate_id for observation in campaign_outcomes})
        shadow_labeled_count = len({observation.candidate_id for observation in campaign_shadow})
        observed_parameter_sets = {value for value in parameter_counts if value != "unknown"}
        observed_coverage_families = set(coverage_family_counts)
        observed_coverage_parameter_sets = {value for value in coverage_parameter_set_counts if value != "unknown"}
        recommendation_bucket = _campaign_recommendation_bucket(
            qualified_candidate_count=qualified_count,
            outcome_labeled_count=outcome_labeled_count,
            shadow_labeled_count=shadow_labeled_count,
            distinct_parameter_sets=max(len(observed_parameter_sets), len(target_parameter_sets)),
            distinct_session_days=len(session_days),
        )
        reports.append(
            CampaignSummaryReport(
                campaign_id=_campaign_id(runs_by_campaign.get(key, []), campaign_records, fallback=key),
                campaign_label=_campaign_label(runs_by_campaign.get(key, []), campaign_records, fallback=key),
                purpose=purpose,
                notes=notes,
                target_strategy_families=sorted(target_families),
                target_parameter_sets=sorted(target_parameter_sets),
                runs_count=len(runs_by_campaign.get(key, [])),
                qualified_candidate_count=qualified_count,
                outcome_labeled_count=outcome_labeled_count,
                shadow_labeled_count=shadow_labeled_count,
                distinct_session_days=len(session_days),
                distinct_experiments=len({item for item in experiments if item != "unknown"}),
                distinct_parameter_sets_observed=len(observed_parameter_sets),
                distinct_families_coverage_observed=len(observed_coverage_families),
                distinct_parameter_sets_coverage_observed=len(observed_coverage_parameter_sets),
                family_candidate_counts=dict(family_counts),
                coverage_family_counts=dict(coverage_family_counts),
                parameter_set_counts=dict(parameter_counts),
                coverage_parameter_set_counts=dict(coverage_parameter_set_counts),
                recommendation_bucket=recommendation_bucket,
                metadata={
                    "missing_target_families": sorted(target_families - set(family_counts)),
                    "missing_target_parameter_sets": sorted(target_parameter_sets - observed_parameter_sets),
                    "missing_target_families_by_candidate_evidence": sorted(target_families - set(family_counts)),
                    "missing_target_parameter_sets_by_candidate_evidence": sorted(
                        target_parameter_sets - observed_parameter_sets
                    ),
                    "missing_target_families_by_run_coverage": sorted(target_families - observed_coverage_families),
                    "missing_target_parameter_sets_by_run_coverage": sorted(
                        target_parameter_sets - observed_coverage_parameter_sets
                    ),
                },
            )
        )
    return reports


def build_campaign_progress_reports(
    run_summaries: Sequence[RunSummary],
    qualified_records: Sequence[OutcomeEvaluationCandidate],
    qualified_outcome_observations: Sequence[CandidateOutcomeObservation],
    qualified_shadow_observations: Sequence[ShadowExecutionObservation],
) -> list[CampaignProgressReport]:
    run_by_id = {run.run_id: run for run in run_summaries}
    records_by_campaign: dict[str, list[OutcomeEvaluationCandidate]] = defaultdict(list)
    runs_by_campaign: dict[str, list[RunSummary]] = defaultdict(list)
    for record in qualified_records:
        records_by_campaign[_campaign_key_for_record(record, run_by_id)].append(record)
    for run in run_summaries:
        runs_by_campaign[_campaign_key(run.metadata)].append(run)

    reports: list[CampaignProgressReport] = []
    previous_snapshot: dict[str, Any] | None = None
    campaigns = sorted(
        set(records_by_campaign) | set(runs_by_campaign),
        key=lambda key: _campaign_sort_key(runs_by_campaign.get(key, []), records_by_campaign.get(key, [])),
    )
    for key in campaigns:
        campaign_records = records_by_campaign.get(key, [])
        campaign_runs = runs_by_campaign.get(key, [])
        snapshot = _campaign_progress_snapshot(
            key=key,
            runs=campaign_runs,
            records=campaign_records,
            outcome_observations=qualified_outcome_observations,
            shadow_observations=qualified_shadow_observations,
        )
        reports.append(_campaign_progress_report(snapshot, previous_snapshot))
        previous_snapshot = snapshot
    return reports


def build_family_evidence_reports(
    run_summaries: Sequence[RunSummary],
    qualified_records: Sequence[OutcomeEvaluationCandidate],
    qualified_outcome_observations: Sequence[CandidateOutcomeObservation],
    qualified_shadow_observations: Sequence[ShadowExecutionObservation],
    promotion_reports: Sequence[PromotionGateReport],
) -> list[FamilyEvidenceReport]:
    raw_counts: Counter[str] = Counter()
    run_by_id = {run.run_id: run for run in run_summaries}
    for run in run_summaries:
        raw_counts.update(run.metadata.get("raw_candidates_by_family", {}))

    promotion_by_family = {report.strategy_family: report for report in promotion_reports}
    records_by_family: dict[str, list[OutcomeEvaluationCandidate]] = defaultdict(list)
    for record in qualified_records:
        records_by_family[record.strategy_family].append(record)

    families = sorted(set(raw_counts) | set(records_by_family) | set(promotion_by_family))
    reports: list[FamilyEvidenceReport] = []
    for family in families:
        family_records = records_by_family.get(family, [])
        candidate_ids = {record.candidate_id for record in family_records}
        reference_horizon = _select_reference_horizon(candidate_ids, qualified_outcome_observations)
        family_outcomes = [
            observation
            for observation in qualified_outcome_observations
            if observation.candidate_id in candidate_ids and observation.horizon_sec == reference_horizon
        ]
        family_shadow = [
            observation
            for observation in qualified_shadow_observations
            if observation.candidate_id in candidate_ids
        ]
        campaign_counts = Counter(_campaign_key_for_record(record, run_by_id) for record in family_records)
        parameter_counts = Counter(_parameter_set_key(record) for record in family_records)
        rank_bucket_counts = Counter(_rank_bucket(record.ranking_score) for record in family_records)
        promotion = promotion_by_family.get(family)
        reports.append(
            FamilyEvidenceReport(
                strategy_family=family,
                current_readiness_bucket=promotion.current_readiness_bucket if promotion is not None else "not_ready",
                promotion_bucket=promotion.promotion_bucket if promotion is not None else "insufficient_evidence",
                raw_candidate_count=raw_counts[family],
                qualified_candidate_count=len(candidate_ids),
                outcome_labeled_count=len({observation.candidate_id for observation in family_outcomes}),
                shadow_labeled_count=len({observation.candidate_id for observation in family_shadow}),
                distinct_runs=len({record.run_id for record in family_records if record.run_id}),
                distinct_campaigns=len({key for key in campaign_counts if key != "uncategorized"}),
                distinct_parameter_sets=len({key for key in parameter_counts if key != "unknown"}),
                distinct_session_days=len({record.ts.date().isoformat() for record in family_records}),
                distinct_rank_buckets=len(rank_bucket_counts),
                campaigns=sorted(campaign_counts),
                parameter_sets=sorted(parameter_counts),
                max_campaign_share=_max_share(campaign_counts),
                max_parameter_set_share=_max_share(parameter_counts),
                blocker_codes=list(promotion.blocker_codes) if promotion is not None else [],
                evidence_needed=list(promotion.evidence_needed) if promotion is not None else [],
                metadata={
                    "reference_horizon_sec": reference_horizon,
                    "campaign_counts": dict(campaign_counts),
                    "parameter_set_counts": dict(parameter_counts),
                    "rank_bucket_counts": dict(rank_bucket_counts),
                },
            )
        )
    return reports


def build_coverage_gap_reports(
    family_reports: Sequence[FamilyEvidenceReport],
) -> list[CoverageGapReport]:
    reports: list[CoverageGapReport] = []
    for family in family_reports:
        recommendation_bucket = _family_collection_bucket(family)
        reports.append(
            CoverageGapReport(
                subject_type="strategy_family",
                subject_key=family.strategy_family,
                recommendation_bucket=recommendation_bucket,
                missing_outcome_labels=max(0, DEFAULT_COVERAGE_TARGETS["outcome_labels"] - family.outcome_labeled_count),
                missing_shadow_labels=max(0, DEFAULT_COVERAGE_TARGETS["shadow_labels"] - family.shadow_labeled_count),
                missing_runs=max(0, DEFAULT_COVERAGE_TARGETS["runs"] - family.distinct_runs),
                missing_campaigns=max(0, DEFAULT_COVERAGE_TARGETS["campaigns"] - family.distinct_campaigns),
                missing_parameter_sets=max(0, DEFAULT_COVERAGE_TARGETS["parameter_sets"] - family.distinct_parameter_sets),
                missing_session_days=max(0, DEFAULT_COVERAGE_TARGETS["session_days"] - family.distinct_session_days),
                blocker_codes=list(family.blocker_codes),
                rationale=_family_gap_rationale(family, recommendation_bucket),
                metadata={
                    "max_campaign_share": family.max_campaign_share,
                    "max_parameter_set_share": family.max_parameter_set_share,
                },
            )
        )
    return reports


def build_collection_recommendations(
    family_reports: Sequence[FamilyEvidenceReport],
    campaign_summaries: Sequence[CampaignSummaryReport],
) -> tuple[list[CollectionRecommendation], list[CollectionRecommendation]]:
    family_recommendations: list[CollectionRecommendation] = []
    for family in family_reports:
        bucket = _family_collection_bucket(family)
        next_focus: list[str] = []
        if bucket == "diversify_parameter_sets":
            next_focus.append("Run this family under an additional parameter set.")
        elif bucket == "diversify_time_windows":
            next_focus.append("Collect more runs on different session days or campaigns.")
        elif bucket == "collect_more_data":
            next_focus.append("Increase qualified and shadow-labeled sample counts.")
        elif bucket == "enough_data_for_research_judgment":
            next_focus.append("You have enough evidence for a research decision; keep monitoring drift only.")
        else:
            next_focus.append("Reduce collection budget unless new strategy changes materially improve quality.")
        family_recommendations.append(
            CollectionRecommendation(
                subject_type="strategy_family",
                subject_key=family.strategy_family,
                recommendation_bucket=bucket,
                priority_score=_family_collection_priority(family, bucket),
                rationale=_family_gap_rationale(family, bucket),
                next_focus=next_focus + list(family.evidence_needed[:2]),
                metadata={
                    "promotion_bucket": family.promotion_bucket,
                    "current_readiness_bucket": family.current_readiness_bucket,
                },
            )
        )

    campaign_recommendations: list[CollectionRecommendation] = []
    for campaign in campaign_summaries:
        next_focus: list[str] = []
        if campaign.recommendation_bucket == "diversify_parameter_sets":
            next_focus.append("Collect another parameter-set slice inside this campaign.")
        elif campaign.recommendation_bucket == "diversify_time_windows":
            next_focus.append("Spread this campaign across more session days.")
        elif campaign.recommendation_bucket == "collect_more_data":
            next_focus.append("Continue batch collection; this campaign is still under-sampled.")
        elif campaign.recommendation_bucket == "enough_data_for_research_judgment":
            next_focus.append("This campaign has enough evidence for a research judgment.")
        else:
            next_focus.append("This campaign is not producing useful evidence at the moment.")
        campaign_recommendations.append(
            CollectionRecommendation(
                subject_type="campaign",
                subject_key=campaign.campaign_label,
                recommendation_bucket=campaign.recommendation_bucket,
                priority_score=_campaign_priority(campaign),
                rationale=_campaign_rationale(campaign),
                next_focus=next_focus,
                metadata={
                    "campaign_id": campaign.campaign_id,
                    "target_strategy_families": campaign.target_strategy_families,
                    "target_parameter_sets": campaign.target_parameter_sets,
                },
            )
        )

    return (
        sorted(family_recommendations, key=lambda item: item.priority_score or 0.0, reverse=True),
        sorted(campaign_recommendations, key=lambda item: item.priority_score or 0.0, reverse=True),
    )


def build_evidence_target_trackers(
    run_summaries: Sequence[RunSummary],
    qualified_records: Sequence[OutcomeEvaluationCandidate],
    qualified_outcome_observations: Sequence[CandidateOutcomeObservation],
    qualified_shadow_observations: Sequence[ShadowExecutionObservation],
    family_reports: Sequence[FamilyEvidenceReport],
    promotion_reports: Sequence[PromotionGateReport],
) -> list[EvidenceTargetTracker]:
    run_by_id = {run.run_id: run for run in run_summaries}
    promotion_by_family = {report.strategy_family: report for report in promotion_reports}
    trackers = [
        _build_family_target_tracker(family_report, promotion_by_family.get(family_report.strategy_family))
        for family_report in family_reports
    ]
    trackers.extend(
        _build_family_parameter_set_trackers(
            run_by_id=run_by_id,
            qualified_records=qualified_records,
            qualified_outcome_observations=qualified_outcome_observations,
            qualified_shadow_observations=qualified_shadow_observations,
            family_promotions=promotion_by_family,
        )
    )
    return sorted(trackers, key=lambda item: (item.subject_type, item.subject_key))


def build_collection_action_backlog(
    campaign_summaries: Sequence[CampaignSummaryReport],
    campaign_progress_reports: Sequence[CampaignProgressReport],
    evidence_target_trackers: Sequence[EvidenceTargetTracker],
) -> list[CollectionActionBacklogEntry]:
    progress_by_label = {report.campaign_label: report for report in campaign_progress_reports}
    backlog: list[CollectionActionBacklogEntry] = []

    for tracker in evidence_target_trackers:
        bucket = _tracker_collection_bucket(tracker)
        backlog.append(
            CollectionActionBacklogEntry(
                subject_type=tracker.subject_type,
                subject_key=tracker.subject_key,
                strategy_family=tracker.strategy_family,
                parameter_set_label=tracker.parameter_set_label,
                recommendation_bucket=bucket,
                priority_score=_tracker_collection_priority(tracker, bucket),
                rationale=_tracker_collection_rationale(tracker, bucket),
                blockers=list(tracker.blocker_codes),
                next_actions=_tracker_next_actions(tracker, bucket),
                metadata={
                    "promotion_bucket": tracker.promotion_bucket,
                    "current_readiness_bucket": tracker.current_readiness_bucket,
                    "target_progress_score": tracker.target_progress_score,
                },
            )
        )

    for summary in campaign_summaries:
        progress = progress_by_label.get(summary.campaign_label)
        bucket = _campaign_collection_bucket(summary, progress)
        backlog.append(
            CollectionActionBacklogEntry(
                subject_type="campaign",
                subject_key=summary.campaign_label,
                campaign_label=summary.campaign_label,
                recommendation_bucket=bucket,
                priority_score=_campaign_collection_priority(summary, progress, bucket),
                rationale=_campaign_collection_rationale(summary, progress, bucket),
                blockers=list(summary.metadata.get("missing_target_families", [])),
                next_actions=_campaign_next_actions(summary, progress, bucket),
                metadata={
                    "campaign_id": summary.campaign_id,
                    "target_strategy_families": summary.target_strategy_families,
                    "target_parameter_sets": summary.target_parameter_sets,
                },
            )
        )

    return sorted(backlog, key=lambda item: (item.priority_score or 0.0), reverse=True)


def build_collection_evidence_snapshot(
    report: OfflineAnalyticsReport,
    *,
    snapshot_label: str,
    metadata: dict[str, Any] | None = None,
) -> CollectionEvidenceSnapshot:
    return CollectionEvidenceSnapshot(
        generated_ts=report.generated_ts,
        snapshot_label=snapshot_label,
        db_path=report.db_path,
        campaign_summaries=list(report.campaign_summaries),
        family_evidence_reports=list(report.family_evidence_reports),
        promotion_gate_reports=list(report.promotion_gate_reports),
        evidence_target_trackers=list(report.evidence_target_trackers),
        collection_action_backlog=list(report.collection_action_backlog),
        metadata={
            "source_report_generated_ts": report.generated_ts.isoformat(),
            **(metadata or {}),
        },
    )


def compare_collection_evidence_snapshots(
    baseline: CollectionEvidenceSnapshot,
    current: CollectionEvidenceSnapshot,
) -> CollectionEvidenceComparisonReport:
    baseline_families = {item.strategy_family: item for item in baseline.family_evidence_reports}
    current_families = {item.strategy_family: item for item in current.family_evidence_reports}
    baseline_promotions = {item.strategy_family: item for item in baseline.promotion_gate_reports}
    current_promotions = {item.strategy_family: item for item in current.promotion_gate_reports}
    baseline_trackers = {
        item.strategy_family: item
        for item in baseline.evidence_target_trackers
        if item.subject_type == "strategy_family"
    }
    current_trackers = {
        item.strategy_family: item
        for item in current.evidence_target_trackers
        if item.subject_type == "strategy_family"
    }
    families = sorted(set(baseline_families) | set(current_families) | set(baseline_promotions) | set(current_promotions))

    family_deltas: list[CollectionEvidenceDelta] = []
    newly_promoted: list[str] = []
    still_blocked: list[str] = []
    families_with_evidence_gain = 0
    backlog_reduction_count = 0

    for family in families:
        delta = _build_family_evidence_delta(
            family=family,
            baseline_family=baseline_families.get(family),
            current_family=current_families.get(family),
            baseline_promotion=baseline_promotions.get(family),
            current_promotion=current_promotions.get(family),
            baseline_tracker=baseline_trackers.get(family),
            current_tracker=current_trackers.get(family),
        )
        family_deltas.append(delta)
        if delta.delta_qualified_candidate_count > 0 or delta.delta_outcome_labeled_count > 0 or delta.delta_shadow_labeled_count > 0:
            families_with_evidence_gain += 1
        if _is_promoted_bucket(delta.current_promotion_bucket) and not _is_promoted_bucket(delta.baseline_promotion_bucket):
            newly_promoted.append(family)
        if delta.current_collection_bucket in {"continue_collection", "diversify_collection_needed", "close_to_sufficient"}:
            still_blocked.append(family)
        if any(value < 0 for value in (
            delta.delta_missing_outcome_labels,
            delta.delta_missing_shadow_labels,
            delta.delta_missing_runs,
            delta.delta_missing_campaigns,
            delta.delta_missing_parameter_sets,
        )):
            backlog_reduction_count += 1

    return CollectionEvidenceComparisonReport(
        generated_ts=current.generated_ts,
        baseline_label=baseline.snapshot_label,
        current_label=current.snapshot_label,
        baseline_generated_ts=baseline.generated_ts,
        current_generated_ts=current.generated_ts,
        families_compared=len(family_deltas),
        families_with_evidence_gain=families_with_evidence_gain,
        backlog_reduction_count=backlog_reduction_count,
        newly_promoted_families=newly_promoted,
        still_blocked_families=still_blocked,
        family_deltas=family_deltas,
        metadata={
            "baseline_db_path": baseline.db_path,
            "current_db_path": current.db_path,
        },
    )


def _campaign_sort_key(
    runs: Sequence[RunSummary],
    records: Sequence[OutcomeEvaluationCandidate],
) -> tuple[str, str]:
    timestamps = [run.started_ts.isoformat() for run in runs]
    timestamps.extend(record.ts.isoformat() for record in records)
    first_ts = min(timestamps) if timestamps else "9999-12-31T00:00:00+00:00"
    return first_ts, _campaign_label(runs, records, fallback="uncategorized")


def _build_family_evidence_delta(
    *,
    family: str,
    baseline_family: FamilyEvidenceReport | None,
    current_family: FamilyEvidenceReport | None,
    baseline_promotion: PromotionGateReport | None,
    current_promotion: PromotionGateReport | None,
    baseline_tracker: EvidenceTargetTracker | None,
    current_tracker: EvidenceTargetTracker | None,
) -> CollectionEvidenceDelta:
    baseline_blockers = set((baseline_family.blocker_codes if baseline_family is not None else []))
    current_blockers = set((current_family.blocker_codes if current_family is not None else []))
    return CollectionEvidenceDelta(
        strategy_family=family,
        baseline_promotion_bucket=baseline_promotion.promotion_bucket if baseline_promotion is not None else "insufficient_evidence",
        current_promotion_bucket=current_promotion.promotion_bucket if current_promotion is not None else "insufficient_evidence",
        baseline_collection_bucket=_tracker_collection_bucket(baseline_tracker) if baseline_tracker is not None else None,
        current_collection_bucket=_tracker_collection_bucket(current_tracker) if current_tracker is not None else None,
        delta_qualified_candidate_count=_family_value(current_family, "qualified_candidate_count") - _family_value(baseline_family, "qualified_candidate_count"),
        delta_outcome_labeled_count=_family_value(current_family, "outcome_labeled_count") - _family_value(baseline_family, "outcome_labeled_count"),
        delta_shadow_labeled_count=_family_value(current_family, "shadow_labeled_count") - _family_value(baseline_family, "shadow_labeled_count"),
        delta_distinct_runs=_family_value(current_family, "distinct_runs") - _family_value(baseline_family, "distinct_runs"),
        delta_distinct_campaigns=_family_value(current_family, "distinct_campaigns") - _family_value(baseline_family, "distinct_campaigns"),
        delta_distinct_parameter_sets=_family_value(current_family, "distinct_parameter_sets") - _family_value(baseline_family, "distinct_parameter_sets"),
        delta_missing_outcome_labels=_tracker_missing_value(current_tracker, "missing_outcome_labels") - _tracker_missing_value(baseline_tracker, "missing_outcome_labels"),
        delta_missing_shadow_labels=_tracker_missing_value(current_tracker, "missing_shadow_labels") - _tracker_missing_value(baseline_tracker, "missing_shadow_labels"),
        delta_missing_runs=_tracker_missing_value(current_tracker, "missing_runs") - _tracker_missing_value(baseline_tracker, "missing_runs"),
        delta_missing_campaigns=_tracker_missing_value(current_tracker, "missing_campaigns") - _tracker_missing_value(baseline_tracker, "missing_campaigns"),
        delta_missing_parameter_sets=_tracker_missing_value(current_tracker, "missing_parameter_sets") - _tracker_missing_value(baseline_tracker, "missing_parameter_sets"),
        delta_target_progress_score=_score_delta(baseline_tracker, current_tracker),
        status_change=_status_change(
            baseline_promotion.promotion_bucket if baseline_promotion is not None else "insufficient_evidence",
            current_promotion.promotion_bucket if current_promotion is not None else "insufficient_evidence",
        ),
        blockers_added=sorted(current_blockers - baseline_blockers),
        blockers_removed=sorted(baseline_blockers - current_blockers),
        metadata={
            "baseline_current_readiness_bucket": baseline_family.current_readiness_bucket if baseline_family is not None else "not_ready",
            "current_current_readiness_bucket": current_family.current_readiness_bucket if current_family is not None else "not_ready",
        },
    )


def _campaign_progress_snapshot(
    *,
    key: str,
    runs: Sequence[RunSummary],
    records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
) -> dict[str, Any]:
    candidate_ids = {record.candidate_id for record in records}
    reference_horizon = _select_reference_horizon(candidate_ids, outcome_observations)
    campaign_outcomes = [
        observation
        for observation in outcome_observations
        if observation.candidate_id in candidate_ids and observation.horizon_sec == reference_horizon
    ]
    campaign_shadow = [
        observation
        for observation in shadow_observations
        if observation.candidate_id in candidate_ids
    ]
    family_qualified_counts = Counter(record.strategy_family for record in records)
    family_outcome_counts = Counter(observation.strategy_family for observation in campaign_outcomes)
    family_shadow_counts = Counter(observation.strategy_family for observation in campaign_shadow)
    family_coverage_counts = _campaign_coverage_family_counts(runs)
    parameter_set_coverage_counts = _campaign_coverage_parameter_set_counts(runs)
    timestamps = [run.started_ts for run in runs]
    timestamps.extend(record.ts for record in records)
    ordered_timestamps = sorted(timestamps)
    return {
        "campaign_id": _campaign_id(runs, records, fallback=key),
        "campaign_label": _campaign_label(runs, records, fallback=key),
        "first_started_ts": ordered_timestamps[0] if ordered_timestamps else None,
        "latest_started_ts": ordered_timestamps[-1] if ordered_timestamps else None,
        "runs_count": len(runs),
        "qualified_candidate_count": len(candidate_ids),
        "outcome_labeled_count": len({observation.candidate_id for observation in campaign_outcomes}),
        "shadow_labeled_count": len({observation.candidate_id for observation in campaign_shadow}),
        "distinct_families_count": len(family_qualified_counts),
        "distinct_families_coverage_observed": len(family_coverage_counts),
        "distinct_parameter_sets_coverage_observed": len(
            {value for value in parameter_set_coverage_counts if value != "unknown"}
        ),
        "family_qualified_counts": dict(family_qualified_counts),
        "family_outcome_counts": dict(family_outcome_counts),
        "family_shadow_counts": dict(family_shadow_counts),
        "family_coverage_counts": dict(family_coverage_counts),
        "parameter_set_coverage_counts": dict(parameter_set_coverage_counts),
        "reference_horizon_sec": reference_horizon,
    }


def _campaign_progress_report(
    snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
) -> CampaignProgressReport:
    previous_label = previous_snapshot["campaign_label"] if previous_snapshot is not None else None
    family_qualified_growth = _counter_delta(snapshot["family_qualified_counts"], previous_snapshot, "family_qualified_counts")
    family_outcome_growth = _counter_delta(snapshot["family_outcome_counts"], previous_snapshot, "family_outcome_counts")
    family_shadow_growth = _counter_delta(snapshot["family_shadow_counts"], previous_snapshot, "family_shadow_counts")
    delta_runs = snapshot["runs_count"] - int(previous_snapshot["runs_count"]) if previous_snapshot is not None else snapshot["runs_count"]
    delta_qualified = _delta_from_previous(snapshot, previous_snapshot, "qualified_candidate_count")
    delta_outcomes = _delta_from_previous(snapshot, previous_snapshot, "outcome_labeled_count")
    delta_shadow = _delta_from_previous(snapshot, previous_snapshot, "shadow_labeled_count")
    delta_families = _delta_from_previous(snapshot, previous_snapshot, "distinct_families_count")
    return CampaignProgressReport(
        campaign_id=snapshot["campaign_id"],
        campaign_label=snapshot["campaign_label"],
        previous_campaign_label=previous_label,
        first_started_ts=snapshot["first_started_ts"],
        latest_started_ts=snapshot["latest_started_ts"],
        runs_count=snapshot["runs_count"],
        qualified_candidate_count=snapshot["qualified_candidate_count"],
        outcome_labeled_count=snapshot["outcome_labeled_count"],
        shadow_labeled_count=snapshot["shadow_labeled_count"],
        distinct_families_count=snapshot["distinct_families_count"],
        distinct_families_coverage_observed=snapshot["distinct_families_coverage_observed"],
        distinct_parameter_sets_coverage_observed=snapshot["distinct_parameter_sets_coverage_observed"],
        family_qualified_counts=snapshot["family_qualified_counts"],
        family_outcome_counts=snapshot["family_outcome_counts"],
        family_shadow_counts=snapshot["family_shadow_counts"],
        family_coverage_counts=snapshot["family_coverage_counts"],
        parameter_set_coverage_counts=snapshot["parameter_set_coverage_counts"],
        delta_runs=delta_runs,
        delta_qualified_candidate_count=delta_qualified,
        delta_outcome_labeled_count=delta_outcomes,
        delta_shadow_labeled_count=delta_shadow,
        delta_distinct_families=delta_families,
        family_qualified_growth=family_qualified_growth,
        family_outcome_growth=family_outcome_growth,
        family_shadow_growth=family_shadow_growth,
        evidence_improved=_campaign_evidence_improved(
            delta_runs=delta_runs,
            delta_qualified=delta_qualified,
            delta_outcomes=delta_outcomes,
            delta_shadow=delta_shadow,
            family_deltas=(family_qualified_growth, family_outcome_growth, family_shadow_growth),
        ),
        metadata={"reference_horizon_sec": snapshot["reference_horizon_sec"]},
    )


def _build_family_target_tracker(
    family_report: FamilyEvidenceReport,
    promotion_report: PromotionGateReport | None,
) -> EvidenceTargetTracker:
    targets = dict(DEFAULT_COVERAGE_TARGETS)
    missing = _missing_counts(
        qualified_candidate_count=family_report.qualified_candidate_count,
        outcome_labeled_count=family_report.outcome_labeled_count,
        shadow_labeled_count=family_report.shadow_labeled_count,
        distinct_runs=family_report.distinct_runs,
        distinct_campaigns=family_report.distinct_campaigns,
        distinct_parameter_sets=family_report.distinct_parameter_sets,
        distinct_session_days=family_report.distinct_session_days,
        targets=targets,
    )
    stability_status = _family_stability_status(promotion_report)
    return EvidenceTargetTracker(
        subject_type="strategy_family",
        subject_key=family_report.strategy_family,
        strategy_family=family_report.strategy_family,
        current_readiness_bucket=family_report.current_readiness_bucket,
        promotion_bucket=family_report.promotion_bucket,
        qualified_candidate_count=family_report.qualified_candidate_count,
        outcome_labeled_count=family_report.outcome_labeled_count,
        shadow_labeled_count=family_report.shadow_labeled_count,
        distinct_runs=family_report.distinct_runs,
        distinct_campaigns=family_report.distinct_campaigns,
        distinct_parameter_sets=family_report.distinct_parameter_sets,
        distinct_session_days=family_report.distinct_session_days,
        missing_qualified_candidates=missing["qualified_candidates"],
        missing_outcome_labels=missing["outcome_labels"],
        missing_shadow_labels=missing["shadow_labels"],
        missing_runs=missing["runs"],
        missing_campaigns=missing["campaigns"],
        missing_parameter_sets=missing["parameter_sets"],
        missing_session_days=missing["session_days"],
        stability_evidence_status=stability_status,
        target_progress_score=_target_progress_score(missing, targets),
        blocker_codes=list(family_report.blocker_codes),
        evidence_needed=list(family_report.evidence_needed),
        metadata={"target_values": targets},
    )


def _build_family_parameter_set_trackers(
    *,
    run_by_id: dict[str, RunSummary],
    qualified_records: Sequence[OutcomeEvaluationCandidate],
    qualified_outcome_observations: Sequence[CandidateOutcomeObservation],
    qualified_shadow_observations: Sequence[ShadowExecutionObservation],
    family_promotions: dict[str, PromotionGateReport],
) -> list[EvidenceTargetTracker]:
    grouped_records: dict[tuple[str, str], list[OutcomeEvaluationCandidate]] = defaultdict(list)
    for record in qualified_records:
        parameter_set = _parameter_set_key(record)
        if parameter_set == "unknown":
            continue
        grouped_records[(record.strategy_family, parameter_set)].append(record)

    trackers: list[EvidenceTargetTracker] = []
    for (family, parameter_set), records in sorted(grouped_records.items()):
        candidate_ids = {record.candidate_id for record in records}
        reference_horizon = _select_reference_horizon(candidate_ids, qualified_outcome_observations)
        outcomes = [
            observation
            for observation in qualified_outcome_observations
            if observation.candidate_id in candidate_ids and observation.horizon_sec == reference_horizon
        ]
        shadows = [
            observation
            for observation in qualified_shadow_observations
            if observation.candidate_id in candidate_ids
        ]
        targets = dict(DEFAULT_PARAMETER_SET_TARGETS)
        missing = _missing_counts(
            qualified_candidate_count=len(candidate_ids),
            outcome_labeled_count=len({observation.candidate_id for observation in outcomes}),
            shadow_labeled_count=len({observation.candidate_id for observation in shadows}),
            distinct_runs=len({record.run_id for record in records if record.run_id}),
            distinct_campaigns=len(
                {
                    _campaign_key_for_record(record, run_by_id)
                    for record in records
                    if _campaign_key_for_record(record, run_by_id) != "uncategorized"
                }
            ),
            distinct_parameter_sets=1,
            distinct_session_days=len({record.ts.date().isoformat() for record in records}),
            targets=targets,
        )
        trackers.append(
            EvidenceTargetTracker(
                subject_type="family_parameter_set",
                subject_key=f"{family}:{parameter_set}",
                strategy_family=family,
                parameter_set_label=parameter_set,
                current_readiness_bucket=family_promotions.get(family).current_readiness_bucket if family in family_promotions else "not_ready",
                promotion_bucket=family_promotions.get(family).promotion_bucket if family in family_promotions else "insufficient_evidence",
                qualified_candidate_count=len(candidate_ids),
                outcome_labeled_count=len({observation.candidate_id for observation in outcomes}),
                shadow_labeled_count=len({observation.candidate_id for observation in shadows}),
                distinct_runs=len({record.run_id for record in records if record.run_id}),
                distinct_campaigns=len(
                    {
                        _campaign_key_for_record(record, run_by_id)
                        for record in records
                        if _campaign_key_for_record(record, run_by_id) != "uncategorized"
                    }
                ),
                distinct_parameter_sets=1,
                distinct_session_days=len({record.ts.date().isoformat() for record in records}),
                missing_qualified_candidates=missing["qualified_candidates"],
                missing_outcome_labels=missing["outcome_labels"],
                missing_shadow_labels=missing["shadow_labels"],
                missing_runs=missing["runs"],
                missing_campaigns=missing["campaigns"],
                missing_parameter_sets=missing["parameter_sets"],
                missing_session_days=missing["session_days"],
                stability_evidence_status="slice_thin" if missing["session_days"] > 0 or missing["runs"] > 0 else "slice_measured",
                target_progress_score=_target_progress_score(missing, targets),
                blocker_codes=[],
                evidence_needed=[f"Collect more evidence for parameter set '{parameter_set}' in family '{family}'."],
                metadata={"target_values": targets, "reference_horizon_sec": reference_horizon},
            )
        )
    return trackers


def _missing_counts(
    *,
    qualified_candidate_count: int,
    outcome_labeled_count: int,
    shadow_labeled_count: int,
    distinct_runs: int,
    distinct_campaigns: int,
    distinct_parameter_sets: int,
    distinct_session_days: int,
    targets: dict[str, int],
) -> dict[str, int]:
    return {
        "qualified_candidates": max(0, targets["qualified_candidates"] - qualified_candidate_count),
        "outcome_labels": max(0, targets["outcome_labels"] - outcome_labeled_count),
        "shadow_labels": max(0, targets["shadow_labels"] - shadow_labeled_count),
        "runs": max(0, targets["runs"] - distinct_runs),
        "campaigns": max(0, targets["campaigns"] - distinct_campaigns),
        "parameter_sets": max(0, targets["parameter_sets"] - distinct_parameter_sets),
        "session_days": max(0, targets["session_days"] - distinct_session_days),
    }


def _target_progress_score(missing: dict[str, int], targets: dict[str, int]) -> float:
    progress_values = []
    for key, target in targets.items():
        if target <= 0:
            progress_values.append(1.0)
            continue
        progress_values.append(max(0.0, min(1.0, 1.0 - (missing[key] / target))))
    return round(sum(progress_values) / len(progress_values), 6) if progress_values else 0.0


def _family_stability_status(promotion_report: PromotionGateReport | None) -> str:
    if promotion_report is None:
        return "unknown"
    if promotion_report.stability_bucket == "stable":
        return "stable"
    if promotion_report.stability_bucket == "mixed":
        return "mixed"
    if promotion_report.stability_bucket == "unstable":
        return "unstable"
    return "thin"


def _tracker_collection_bucket(tracker: EvidenceTargetTracker) -> str:
    if tracker.promotion_bucket == "deprioritize_for_now" or (
        tracker.qualified_candidate_count <= 0 and tracker.outcome_labeled_count <= 0 and tracker.shadow_labeled_count <= 0
    ):
        return "low_yield_deprioritize"
    if _tracker_missing_total(tracker) <= 0 and tracker.stability_evidence_status not in {"thin", "unstable"}:
        return "enough_for_research_judgment"
    if _tracker_needs_diversification(tracker):
        return "diversify_collection_needed"
    if _tracker_is_close_to_sufficient(tracker):
        return "close_to_sufficient"
    return "continue_collection"


def _tracker_collection_priority(tracker: EvidenceTargetTracker, bucket: str) -> float:
    base = {
        "close_to_sufficient": 0.95,
        "continue_collection": 0.80,
        "diversify_collection_needed": 0.75,
        "enough_for_research_judgment": 0.35,
        "low_yield_deprioritize": 0.10,
    }.get(bucket, 0.30)
    if tracker.promotion_bucket == "research_promising":
        base += 0.10
    if tracker.promotion_bucket == "candidate_for_future_tiny_live_preparation":
        base += 0.15
    if tracker.subject_type == "family_parameter_set":
        base -= 0.10
    return round(max(0.0, min(1.0, base)), 6)


def _tracker_collection_rationale(tracker: EvidenceTargetTracker, bucket: str) -> str:
    if bucket == "low_yield_deprioritize":
        return "This subject is not producing enough qualifying evidence to justify more collection budget right now."
    if bucket == "enough_for_research_judgment":
        return "Coverage targets are met and the current evidence is broad enough for a research judgment."
    if bucket == "diversify_collection_needed":
        return "The sample is too concentrated, so the next collection step should broaden slices instead of only adding count."
    if bucket == "close_to_sufficient":
        return "One more focused collection push could be enough to clear the remaining evidence gaps."
    return "Keep collecting; the current sample still falls short of the evidence targets."


def _tracker_next_actions(tracker: EvidenceTargetTracker, bucket: str) -> list[str]:
    actions: list[str] = []
    if bucket == "diversify_collection_needed":
        if tracker.missing_parameter_sets > 0:
            actions.append("Run this family under an additional parameter set.")
        if tracker.missing_campaigns > 0 or tracker.missing_session_days > 0:
            actions.append("Spread collection across more campaigns or session days.")
    else:
        if tracker.missing_qualified_candidates > 0:
            actions.append(f"Collect {tracker.missing_qualified_candidates} more qualified candidates.")
        if tracker.missing_outcome_labels > 0:
            actions.append(f"Collect {tracker.missing_outcome_labels} more outcome-labeled candidates.")
        if tracker.missing_shadow_labels > 0:
            actions.append(f"Collect {tracker.missing_shadow_labels} more shadow-labeled candidates.")
        if tracker.missing_runs > 0:
            actions.append(f"Add {tracker.missing_runs} more distinct run(s).")
    actions.extend(tracker.evidence_needed[:2])
    return list(dict.fromkeys(actions))[:4]


def _campaign_collection_bucket(
    summary: CampaignSummaryReport,
    progress: CampaignProgressReport | None,
) -> str:
    if summary.recommendation_bucket == "deprioritize_collection":
        return "low_yield_deprioritize"
    if summary.recommendation_bucket == "enough_data_for_research_judgment":
        return "enough_for_research_judgment"
    if summary.recommendation_bucket in {"diversify_parameter_sets", "diversify_time_windows"}:
        return "diversify_collection_needed"
    if _campaign_is_close_to_sufficient(summary):
        return "close_to_sufficient"
    if progress is not None and not progress.evidence_improved and summary.runs_count > 0:
        return "low_yield_deprioritize"
    return "continue_collection"


def _campaign_collection_priority(
    summary: CampaignSummaryReport,
    progress: CampaignProgressReport | None,
    bucket: str,
) -> float:
    base = {
        "close_to_sufficient": 0.90,
        "continue_collection": 0.75,
        "diversify_collection_needed": 0.70,
        "enough_for_research_judgment": 0.35,
        "low_yield_deprioritize": 0.10,
    }.get(bucket, 0.30)
    if progress is not None and progress.evidence_improved:
        base += 0.05
    if summary.qualified_candidate_count <= 0:
        base -= 0.10
    return round(max(0.0, min(1.0, base)), 6)


def _campaign_collection_rationale(
    summary: CampaignSummaryReport,
    progress: CampaignProgressReport | None,
    bucket: str,
) -> str:
    if bucket == "low_yield_deprioritize":
        return "This campaign is not improving promotion-relevant evidence enough to justify more collection budget."
    if bucket == "enough_for_research_judgment":
        return "This campaign has already accumulated enough evidence for its current research scope."
    if bucket == "diversify_collection_needed":
        return "This campaign should broaden coverage rather than only repeating the same slice."
    if bucket == "close_to_sufficient":
        return "A small additional batch could move this campaign from under-sampled to decision-ready."
    if progress is not None and progress.evidence_improved:
        return "This campaign is still adding useful evidence and should continue collecting."
    return "This campaign still needs more evidence before it can be judged cleanly."


def _campaign_next_actions(
    summary: CampaignSummaryReport,
    progress: CampaignProgressReport | None,
    bucket: str,
) -> list[str]:
    actions: list[str] = []
    missing_families = summary.metadata.get("missing_target_families", [])
    missing_parameter_sets = summary.metadata.get("missing_target_parameter_sets", [])
    if bucket == "diversify_collection_needed":
        if missing_parameter_sets:
            actions.append("Collect the missing target parameter sets for this campaign.")
        else:
            actions.append("Spread this campaign across more session days.")
    elif bucket == "close_to_sufficient":
        actions.append("Run one more focused collection batch to close the remaining evidence gap.")
    elif bucket == "continue_collection":
        actions.append("Continue collection while monitoring whether qualified and labeled counts keep growing.")
    elif bucket == "low_yield_deprioritize":
        actions.append("Pause this campaign unless the strategy definition changes materially.")
    else:
        actions.append("Hold this campaign at maintenance level only.")
    if missing_families:
        actions.append("Collect evidence for the missing target strategy families.")
    if progress is not None and not progress.evidence_improved:
        actions.append("Review this campaign's scope because the latest batch did not improve evidence.")
    return list(dict.fromkeys(actions))[:4]


def _delta_from_previous(snapshot: dict[str, Any], previous_snapshot: dict[str, Any] | None, key: str) -> int:
    if previous_snapshot is None:
        return int(snapshot[key])
    return int(snapshot[key]) - int(previous_snapshot[key])


def _counter_delta(snapshot_counts: dict[str, int], previous_snapshot: dict[str, Any] | None, key: str) -> dict[str, int]:
    previous_counts = previous_snapshot.get(key, {}) if previous_snapshot is not None else {}
    keys = sorted(set(snapshot_counts) | set(previous_counts))
    return {
        item_key: int(snapshot_counts.get(item_key, 0)) - int(previous_counts.get(item_key, 0))
        for item_key in keys
        if int(snapshot_counts.get(item_key, 0)) - int(previous_counts.get(item_key, 0)) != 0
    }


def _campaign_evidence_improved(
    *,
    delta_runs: int,
    delta_qualified: int,
    delta_outcomes: int,
    delta_shadow: int,
    family_deltas: Sequence[dict[str, int]],
) -> bool:
    if any(delta > 0 for delta in (delta_qualified, delta_outcomes, delta_shadow)):
        return True
    return any(any(value > 0 for value in deltas.values()) for deltas in family_deltas)


def _tracker_missing_total(tracker: EvidenceTargetTracker) -> int:
    return (
        tracker.missing_qualified_candidates
        + tracker.missing_outcome_labels
        + tracker.missing_shadow_labels
        + tracker.missing_runs
        + tracker.missing_campaigns
        + tracker.missing_parameter_sets
        + tracker.missing_session_days
    )


def _tracker_is_close_to_sufficient(tracker: EvidenceTargetTracker) -> bool:
    return (
        tracker.missing_qualified_candidates <= 2
        and tracker.missing_outcome_labels <= 2
        and tracker.missing_shadow_labels <= 2
        and tracker.missing_runs <= 1
        and tracker.missing_campaigns <= 1
        and tracker.missing_session_days <= 1
    )


def _tracker_needs_diversification(tracker: EvidenceTargetTracker) -> bool:
    return (
        tracker.missing_parameter_sets > 0
        or tracker.missing_campaigns > 0
        or tracker.missing_session_days > 0
        or tracker.stability_evidence_status in {"thin", "unstable"}
    )


def _campaign_is_close_to_sufficient(summary: CampaignSummaryReport) -> bool:
    return (
        max(0, DEFAULT_COVERAGE_TARGETS["qualified_candidates"] - summary.qualified_candidate_count) <= 2
        and max(0, DEFAULT_COVERAGE_TARGETS["outcome_labels"] - summary.outcome_labeled_count) <= 2
        and max(0, DEFAULT_COVERAGE_TARGETS["shadow_labels"] - summary.shadow_labeled_count) <= 2
    )


def _family_value(report: FamilyEvidenceReport | None, field_name: str) -> int:
    if report is None:
        return 0
    return int(getattr(report, field_name))


def _tracker_missing_value(tracker: EvidenceTargetTracker | None, field_name: str) -> int:
    if tracker is None:
        return 0
    return int(getattr(tracker, field_name))


def _score_delta(
    baseline_tracker: EvidenceTargetTracker | None,
    current_tracker: EvidenceTargetTracker | None,
) -> float | None:
    baseline_score = baseline_tracker.target_progress_score if baseline_tracker is not None else None
    current_score = current_tracker.target_progress_score if current_tracker is not None else None
    if baseline_score is None and current_score is None:
        return None
    return round((current_score or 0.0) - (baseline_score or 0.0), 6)


def _status_change(baseline_bucket: str, current_bucket: str) -> str:
    if baseline_bucket == current_bucket:
        return "unchanged"
    if _is_promoted_bucket(current_bucket) and not _is_promoted_bucket(baseline_bucket):
        return "promoted"
    if current_bucket == "deprioritize_for_now" and baseline_bucket != "deprioritize_for_now":
        return "downgraded"
    return "changed"


def _is_promoted_bucket(bucket: str) -> bool:
    return bucket in {"research_promising", "candidate_for_future_tiny_live_preparation"}


def _campaign_key(metadata: dict[str, Any]) -> str:
    label = _string_or_none(metadata.get("campaign_label"))
    identifier = _string_or_none(metadata.get("campaign_id"))
    return label or identifier or "uncategorized"


def _campaign_id(runs: Sequence[RunSummary], records: Sequence[OutcomeEvaluationCandidate], *, fallback: str) -> str:
    for run in runs:
        if _string_or_none(run.metadata.get("campaign_id")):
            return str(run.metadata["campaign_id"])
    for record in records:
        if _string_or_none(record.metadata.get("campaign_id")):
            return str(record.metadata["campaign_id"])
    return fallback


def _campaign_label(runs: Sequence[RunSummary], records: Sequence[OutcomeEvaluationCandidate], *, fallback: str) -> str:
    for run in runs:
        if _string_or_none(run.metadata.get("campaign_label")):
            return str(run.metadata["campaign_label"])
    for record in records:
        if _string_or_none(record.metadata.get("campaign_label")):
            return str(record.metadata["campaign_label"])
    return fallback


def _campaign_label_from_record(record: OutcomeEvaluationCandidate) -> str:
    return _campaign_key(record.metadata)


def _campaign_key_for_record(record: OutcomeEvaluationCandidate, run_by_id: dict[str, RunSummary]) -> str:
    if record.metadata:
        key = _campaign_key(record.metadata)
        if key != "uncategorized":
            return key
    if record.run_id and record.run_id in run_by_id:
        return _campaign_key(run_by_id[record.run_id].metadata)
    return "uncategorized"


def _campaign_coverage_family_counts(runs: Sequence[RunSummary]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for run in runs:
        funnel = run.metadata.get("strategy_family_funnel")
        if not isinstance(funnel, dict):
            continue
        for family in funnel:
            if family:
                counts[str(family)] += 1
    return counts


def _campaign_coverage_parameter_set_counts(runs: Sequence[RunSummary]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for run in runs:
        parameter_set_label = _string_or_none(run.metadata.get("parameter_set_label"))
        if parameter_set_label:
            counts[parameter_set_label] += 1
    return counts


def _parameter_set_key(record: OutcomeEvaluationCandidate) -> str:
    value = _string_or_none(record.parameter_set_label) or _string_or_none(record.metadata.get("parameter_set_label"))
    return value or "unknown"


def _experiment_key(record: OutcomeEvaluationCandidate) -> str:
    value = _string_or_none(record.experiment_label) or _string_or_none(record.experiment_id)
    return value or "unknown"


def _experiment_key_from_metadata(metadata: dict[str, Any]) -> str:
    value = _string_or_none(metadata.get("experiment_label")) or _string_or_none(metadata.get("experiment_id"))
    return value or "unknown"


def _rank_bucket(score: float | None) -> str:
    if score is None:
        return "unranked"
    if score >= 90.0:
        return "score_90_plus"
    if score >= 80.0:
        return "score_80_89"
    if score >= 70.0:
        return "score_70_79"
    return "score_below_70"


def _select_reference_horizon(
    candidate_ids: set[str],
    observations: Sequence[CandidateOutcomeObservation],
) -> int:
    for horizon in (60, 30, 300, 900):
        if any(
            observation.horizon_sec == horizon and observation.candidate_id in candidate_ids
            for observation in observations
        ):
            return horizon
    return 60


def _campaign_recommendation_bucket(
    *,
    qualified_candidate_count: int,
    outcome_labeled_count: int,
    shadow_labeled_count: int,
    distinct_parameter_sets: int,
    distinct_session_days: int,
) -> str:
    if qualified_candidate_count <= 0:
        return "deprioritize_collection"
    if outcome_labeled_count >= 10 and shadow_labeled_count >= 10 and distinct_parameter_sets >= 2 and distinct_session_days >= 2:
        return "enough_data_for_research_judgment"
    if distinct_parameter_sets < 2:
        return "diversify_parameter_sets"
    if distinct_session_days < 2:
        return "diversify_time_windows"
    return "collect_more_data"


def _family_collection_bucket(family: FamilyEvidenceReport) -> str:
    if family.promotion_bucket == "deprioritize_for_now":
        return "deprioritize_collection"
    if family.promotion_bucket == "candidate_for_future_tiny_live_preparation" and family.outcome_labeled_count >= 10 and family.shadow_labeled_count >= 10:
        return "enough_data_for_research_judgment"
    if family.distinct_parameter_sets < DEFAULT_COVERAGE_TARGETS["parameter_sets"] or (family.max_parameter_set_share or 0.0) > 0.80:
        return "diversify_parameter_sets"
    if family.distinct_session_days < DEFAULT_COVERAGE_TARGETS["session_days"] or (family.max_campaign_share or 0.0) > 0.80:
        return "diversify_time_windows"
    return "collect_more_data"


def _family_gap_rationale(family: FamilyEvidenceReport, recommendation_bucket: str) -> str:
    if recommendation_bucket == "deprioritize_collection":
        return "Forward quality or execution viability is weak enough that more collection is low priority."
    if recommendation_bucket == "enough_data_for_research_judgment":
        return "This family has enough qualified, labeled, and shadow-evaluable evidence for a research judgment."
    if recommendation_bucket == "diversify_parameter_sets":
        return "Evidence is too concentrated in one parameter set, so the next best step is broader threshold coverage."
    if recommendation_bucket == "diversify_time_windows":
        return "Evidence is too concentrated in one campaign or time window, so broader collection timing matters most."
    return "This family is still under-sampled relative to the promotion gates."


def _family_collection_priority(family: FamilyEvidenceReport, bucket: str) -> float:
    base = {
        "collect_more_data": 0.8,
        "diversify_parameter_sets": 0.75,
        "diversify_time_windows": 0.7,
        "enough_data_for_research_judgment": 0.4,
        "deprioritize_collection": 0.1,
    }.get(bucket, 0.3)
    if family.promotion_bucket in {"research_promising", "continue_research"}:
        base += 0.1
    if family.promotion_bucket == "insufficient_evidence" and family.current_readiness_bucket in {"research_promising", "candidate_for_future_tiny_live_preparation"}:
        base += 0.15
    return round(min(1.0, base), 6)


def _campaign_priority(campaign: CampaignSummaryReport) -> float:
    base = {
        "collect_more_data": 0.8,
        "diversify_parameter_sets": 0.7,
        "diversify_time_windows": 0.65,
        "enough_data_for_research_judgment": 0.4,
        "deprioritize_collection": 0.1,
    }.get(campaign.recommendation_bucket, 0.3)
    if campaign.qualified_candidate_count <= 0:
        base -= 0.1
    return round(max(0.0, min(1.0, base)), 6)


def _campaign_rationale(campaign: CampaignSummaryReport) -> str:
    if campaign.recommendation_bucket == "deprioritize_collection":
        return "This campaign has not produced qualified evidence yet."
    if campaign.recommendation_bucket == "enough_data_for_research_judgment":
        return "This campaign has accumulated enough evidence to judge its targeted scope."
    if campaign.recommendation_bucket == "diversify_parameter_sets":
        return "This campaign should broaden parameter-set coverage before drawing conclusions."
    if campaign.recommendation_bucket == "diversify_time_windows":
        return "This campaign needs broader time coverage to reduce concentration risk."
    return "This campaign is producing evidence, but the sample is still too thin."


def _max_share(counts: Counter[str]) -> float | None:
    filtered = {key: value for key, value in counts.items() if key not in {"unknown", "uncategorized"}}
    total = sum(filtered.values())
    if total <= 0:
        return None
    return max(filtered.values()) / total


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
