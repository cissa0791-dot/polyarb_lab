from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.reporting.models import (
    CandidateOutcomeObservation,
    LiveReadinessScorecard,
    PromotionBlockerEntry,
    PromotionGateReport,
    PromotionWatchlistEntry,
    SampleSufficiencyScorecard,
    ShadowExecutionObservation,
    StabilityScorecard,
)
from src.reporting.outcomes import OutcomeEvaluationCandidate, horizon_label, safe_mean


DEFAULT_REFERENCE_HORIZONS = (60, 30, 300, 900)
DEFAULT_STABILITY_SLICES = ("parameter_set", "experiment", "rank_bucket", "session_day")


@dataclass(frozen=True)
class _GroupSampleData:
    candidate_ids: set[str]
    reference_horizon: int
    group_outcomes: list[CandidateOutcomeObservation]
    group_shadow: list[ShadowExecutionObservation]


@dataclass(frozen=True)
class _RecordDiversity:
    distinct_runs: int
    distinct_session_days: int
    distinct_experiments: int
    distinct_parameter_sets: int


@dataclass(frozen=True)
class _PromotionMetrics:
    outcome_labeled_count: int
    shadow_labeled_count: int
    positive_outcome_ratio: float | None
    shadow_viability_rate: float | None
    shadow_fillability_rate: float | None
    mean_execution_gap_cents: float | None


@dataclass(frozen=True)
class _PromotionContext:
    family: str
    candidate_ids: set[str]
    reference_horizon: int
    current_readiness_bucket: str
    sufficiency_bucket: str
    sufficiency_score: float | None
    stability_bucket: str
    stability_score: float | None
    stability_metadata: dict[str, Any]
    metrics: _PromotionMetrics
    diversity: _RecordDiversity


def build_sample_sufficiency_scorecards(
    records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
    *,
    group_type: str,
    record_group_getter,
    preferred_horizons: Iterable[int] = DEFAULT_REFERENCE_HORIZONS,
) -> list[SampleSufficiencyScorecard]:
    preferred = tuple(dict.fromkeys(int(h) for h in preferred_horizons if int(h) > 0))
    grouped_records = _group_records(records, record_group_getter)

    cards: list[SampleSufficiencyScorecard] = []
    for group_key in sorted(grouped_records):
        group_records = grouped_records[group_key]
        sample_data = _collect_group_sample_data(group_records, outcome_observations, shadow_observations, preferred)
        diversity = _record_diversity(group_records)
        cards.append(
            _build_sample_sufficiency_card(
                group_type=group_type,
                group_key=group_key,
                sample_data=sample_data,
                diversity=diversity,
                preferred=preferred,
            )
        )
    return cards


def build_stability_scorecards(
    records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
    *,
    group_type: str,
    record_group_getter,
    slice_dimensions: Sequence[str] = DEFAULT_STABILITY_SLICES,
    preferred_horizons: Iterable[int] = DEFAULT_REFERENCE_HORIZONS,
) -> list[StabilityScorecard]:
    preferred = tuple(dict.fromkeys(int(h) for h in preferred_horizons if int(h) > 0))
    grouped_records = _group_records(records, record_group_getter)

    cards: list[StabilityScorecard] = []
    for group_key in sorted(grouped_records):
        group_records = grouped_records[group_key]
        sample_data = _collect_group_sample_data(group_records, outcome_observations, shadow_observations, preferred)
        for slice_dimension in slice_dimensions:
            cards.append(
                _build_group_stability_card(
                    group_type=group_type,
                    group_key=group_key,
                    slice_dimension=slice_dimension,
                    group_records=group_records,
                    group_outcomes=sample_data.group_outcomes,
                    group_shadow=sample_data.group_shadow,
                    reference_horizon=sample_data.reference_horizon,
                )
            )
    return cards


def build_promotion_gate_reports(
    records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
    readiness_scorecards: Sequence[LiveReadinessScorecard],
    sample_sufficiency_scorecards: Sequence[SampleSufficiencyScorecard],
    stability_scorecards: Sequence[StabilityScorecard],
) -> tuple[list[PromotionGateReport], list[PromotionWatchlistEntry], list[PromotionBlockerEntry]]:
    readiness_by_family = {card.strategy_family: card for card in readiness_scorecards}
    sufficiency_by_family = {
        card.group_key: card
        for card in sample_sufficiency_scorecards
        if card.group_type == "strategy_family"
    }
    stability_aggregate = _aggregate_family_stability(stability_scorecards)
    grouped_records = _group_records(records, lambda record: record.strategy_family)

    reports: list[PromotionGateReport] = []
    blockers: list[PromotionBlockerEntry] = []
    families = sorted(set(grouped_records) | set(readiness_by_family) | set(sufficiency_by_family))
    for family in families:
        report = _build_promotion_report(
            family=family,
            family_records=grouped_records.get(family, []),
            outcome_observations=outcome_observations,
            shadow_observations=shadow_observations,
            readiness=readiness_by_family.get(family),
            sufficiency=sufficiency_by_family.get(family),
            stability=stability_aggregate.get(family),
        )
        reports.append(report)
        blockers.extend(_build_blocker_entries(report))

    return (
        sorted(
            reports,
            key=lambda item: (_promotion_priority(item.promotion_bucket), item.overall_score or -1.0),
            reverse=True,
        ),
        _build_watchlist(reports),
        blockers,
    )


def _build_group_stability_card(
    *,
    group_type: str,
    group_key: str,
    slice_dimension: str,
    group_records: Sequence[OutcomeEvaluationCandidate],
    group_outcomes: Sequence[CandidateOutcomeObservation],
    group_shadow: Sequence[ShadowExecutionObservation],
    reference_horizon: int,
) -> StabilityScorecard:
    candidate_ids = {record.candidate_id for record in group_records}
    total_candidates = len(candidate_ids)
    slices = _group_records(group_records, lambda record: _slice_key(record, slice_dimension))
    (
        slice_metrics,
        positive_ratios,
        viability_rates,
        execution_gaps,
        contributing_slice_count,
        max_slice_candidate_share,
    ) = _build_stability_slice_metrics(
        slices=slices,
        group_outcomes=group_outcomes,
        group_shadow=group_shadow,
        total_candidates=total_candidates,
    )

    positive_ratio_spread = _spread(positive_ratios)
    viability_rate_spread = _spread(viability_rates)
    execution_gap_spread = _spread(execution_gaps)
    consistency_score = _weighted_mean(
        [
            (_inverse_spread_score(positive_ratio_spread, spread_target=0.35), 0.30),
            (_inverse_spread_score(viability_rate_spread, spread_target=0.35), 0.25),
            (_inverse_spread_score(execution_gap_spread, spread_target=0.08), 0.20),
            (_concentration_score(max_slice_candidate_share), 0.15),
            (_bounded_progress(contributing_slice_count, target=3), 0.10),
        ]
    )
    return StabilityScorecard(
        group_type=group_type,
        group_key=group_key,
        slice_dimension=slice_dimension,
        reference_horizon_label=horizon_label(reference_horizon),
        reference_horizon_sec=reference_horizon,
        total_candidates=total_candidates,
        slice_count=len(slices),
        contributing_slice_count=contributing_slice_count,
        positive_ratio_mean=safe_mean(positive_ratios),
        positive_ratio_spread=positive_ratio_spread,
        viability_rate_mean=safe_mean(viability_rates),
        viability_rate_spread=viability_rate_spread,
        execution_gap_mean=safe_mean(execution_gaps),
        execution_gap_spread=execution_gap_spread,
        max_slice_candidate_share=max_slice_candidate_share,
        consistency_score=consistency_score,
        stability_bucket=_stability_bucket(
            contributing_slice_count=contributing_slice_count,
            consistency_score=consistency_score,
            max_slice_candidate_share=max_slice_candidate_share,
        ),
        metadata={
            "qualified_only": True,
            "slice_metrics": slice_metrics,
        },
    )


def _aggregate_family_stability(scorecards: Sequence[StabilityScorecard]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[StabilityScorecard]] = defaultdict(list)
    for card in scorecards:
        if card.group_type == "strategy_family":
            grouped[card.group_key].append(card)

    aggregate: dict[str, dict[str, Any]] = {}
    for family, cards in grouped.items():
        usable = [card for card in cards if card.consistency_score is not None and card.contributing_slice_count >= 2]
        aggregate_score = safe_mean([card.consistency_score for card in usable if card.consistency_score is not None])
        max_slice_candidate_share = max(
            (card.max_slice_candidate_share for card in usable if card.max_slice_candidate_share is not None),
            default=None,
        )
        if not usable:
            bucket = "insufficient_slices"
        elif aggregate_score is not None and aggregate_score >= 0.75 and (max_slice_candidate_share or 0.0) <= 0.75:
            bucket = "stable"
        elif aggregate_score is not None and aggregate_score >= 0.45:
            bucket = "mixed"
        else:
            bucket = "unstable"
        aggregate[family] = {
            "bucket": bucket,
            "score": aggregate_score,
            "metadata": {
                "dimensions": {
                    card.slice_dimension: {
                        "bucket": card.stability_bucket,
                        "consistency_score": card.consistency_score,
                        "max_slice_candidate_share": card.max_slice_candidate_share,
                    }
                    for card in cards
                }
            },
        }
    return aggregate


def _promotion_blockers(
    *,
    outcome_labeled_count: int,
    shadow_labeled_count: int,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    shadow_fillability_rate: float | None,
    mean_execution_gap_cents: float | None,
    sufficiency_bucket: str,
    stability_bucket: str,
    distinct_runs: int,
    distinct_experiments: int,
    distinct_parameter_sets: int,
    stability_metadata: dict[str, Any],
) -> tuple[list[str], dict[str, Any], list[str]]:
    blockers = []
    blockers.extend(
        _sample_blockers(
            outcome_labeled_count=outcome_labeled_count,
            shadow_labeled_count=shadow_labeled_count,
            sufficiency_bucket=sufficiency_bucket,
            distinct_runs=distinct_runs,
        )
    )
    blockers.extend(
        _quality_blockers(
            positive_outcome_ratio=positive_outcome_ratio,
            shadow_viability_rate=shadow_viability_rate,
            shadow_fillability_rate=shadow_fillability_rate,
            mean_execution_gap_cents=mean_execution_gap_cents,
        )
    )
    blockers.extend(
        _stability_blockers(
            stability_bucket=stability_bucket,
            stability_metadata=stability_metadata,
        )
    )
    blockers.extend(
        _coverage_blockers(
            distinct_experiments=distinct_experiments,
            distinct_parameter_sets=distinct_parameter_sets,
        )
    )
    return _dedupe_blockers(blockers)


def _promotion_bucket(
    *,
    current_readiness_bucket: str,
    sufficiency_bucket: str,
    stability_bucket: str,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    shadow_fillability_rate: float | None,
    mean_execution_gap_cents: float | None,
) -> str:
    if sufficiency_bucket == "insufficient_data":
        return "insufficient_evidence"
    if _meets_deprioritize_gate(
        positive_outcome_ratio=positive_outcome_ratio,
        shadow_viability_rate=shadow_viability_rate,
        mean_execution_gap_cents=mean_execution_gap_cents,
    ):
        return "deprioritize_for_now"
    if _meets_future_tiny_live_gate(
        current_readiness_bucket=current_readiness_bucket,
        sufficiency_bucket=sufficiency_bucket,
        stability_bucket=stability_bucket,
        positive_outcome_ratio=positive_outcome_ratio,
        shadow_viability_rate=shadow_viability_rate,
        shadow_fillability_rate=shadow_fillability_rate,
        mean_execution_gap_cents=mean_execution_gap_cents,
    ):
        return "candidate_for_future_tiny_live_preparation"
    if _meets_research_promising_gate(
        current_readiness_bucket=current_readiness_bucket,
        sufficiency_bucket=sufficiency_bucket,
        stability_bucket=stability_bucket,
        positive_outcome_ratio=positive_outcome_ratio,
        shadow_viability_rate=shadow_viability_rate,
        mean_execution_gap_cents=mean_execution_gap_cents,
    ):
        return "research_promising"
    if sufficiency_bucket == "limited_data":
        return "insufficient_evidence"
    return "continue_research"


def _build_watchlist(reports: Sequence[PromotionGateReport]) -> list[PromotionWatchlistEntry]:
    entries: list[PromotionWatchlistEntry] = []
    for report in reports:
        if report.promotion_bucket == "deprioritize_for_now":
            continue
        next_step = report.evidence_needed[0] if report.evidence_needed else "Keep collecting balanced evidence across runs."
        if report.promotion_bucket == "candidate_for_future_tiny_live_preparation":
            reason = "Forward quality, fillability, and stability all clear the current research gates."
        elif report.promotion_bucket == "research_promising":
            reason = "The family looks interesting, but one or more promotion gates still need more evidence."
        elif report.sample_sufficiency_bucket in {"insufficient_data", "limited_data"} and (report.overall_score or 0.0) >= 0.45:
            reason = "Observed quality looks interesting, but the sample is still too thin to trust."
        else:
            reason = "Keep collecting evidence before deciding whether to promote or deprioritize."
        entries.append(
            PromotionWatchlistEntry(
                strategy_family=report.strategy_family,
                promotion_bucket=report.promotion_bucket,
                watchlist_priority=round(_promotion_priority(report.promotion_bucket) + ((report.overall_score or 0.0) * 0.1), 6),
                current_readiness_bucket=report.current_readiness_bucket,
                sample_sufficiency_bucket=report.sample_sufficiency_bucket,
                stability_bucket=report.stability_bucket,
                watchlist_reason=reason,
                next_step=next_step,
                blocker_codes=list(report.blocker_codes),
            )
        )
    return sorted(entries, key=lambda item: (item.watchlist_priority or 0.0), reverse=True)


def _build_blocker_entries(report: PromotionGateReport) -> list[PromotionBlockerEntry]:
    severity_by_code = {
        "LOW_OUTCOME_SAMPLE": "high",
        "LOW_SHADOW_SAMPLE": "high",
        "SAMPLE_SUFFICIENCY_WEAK": "high",
        "LOW_RUN_DIVERSITY": "medium",
        "FORWARD_QUALITY_WEAK": "high",
        "SHADOW_VIABILITY_WEAK": "high",
        "FULL_SIZE_FILLABILITY_WEAK": "medium",
        "EXECUTION_GAP_TOO_LARGE": "high",
        "STABILITY_UNPROVEN": "medium",
        "RESULTS_UNSTABLE": "high",
        "PARAMETER_CONCENTRATION": "medium",
        "EXPERIMENT_METADATA_THIN": "low",
    }
    entries: list[PromotionBlockerEntry] = []
    for code in report.blocker_codes:
        detail = report.blocker_details.get(code, {})
        evidence_needed = next(
            (item for item in report.evidence_needed if _evidence_matches_blocker(item, code)),
            report.evidence_needed[0] if report.evidence_needed else "",
        )
        entries.append(
            PromotionBlockerEntry(
                strategy_family=report.strategy_family,
                blocker_code=code,
                detail=_detail_text(detail),
                evidence_needed=evidence_needed,
                severity=severity_by_code.get(code, "medium"),
            )
        )
    return entries


def _group_records(
    records: Sequence[OutcomeEvaluationCandidate],
    record_group_getter,
) -> dict[str, list[OutcomeEvaluationCandidate]]:
    grouped_records: dict[str, list[OutcomeEvaluationCandidate]] = defaultdict(list)
    for record in records:
        grouped_records[_group_key(record_group_getter(record))].append(record)
    return grouped_records


def _collect_group_sample_data(
    group_records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
    preferred_horizons: Sequence[int],
) -> _GroupSampleData:
    candidate_ids = {record.candidate_id for record in group_records}
    reference_horizon = _select_reference_horizon(candidate_ids, outcome_observations, preferred_horizons)
    return _GroupSampleData(
        candidate_ids=candidate_ids,
        reference_horizon=reference_horizon,
        group_outcomes=_filter_group_outcomes(
            outcome_observations,
            candidate_ids=candidate_ids,
            reference_horizon=reference_horizon,
        ),
        group_shadow=_filter_group_shadow(shadow_observations, candidate_ids),
    )


def _filter_group_outcomes(
    observations: Sequence[CandidateOutcomeObservation],
    *,
    candidate_ids: set[str],
    reference_horizon: int,
) -> list[CandidateOutcomeObservation]:
    return [
        observation
        for observation in observations
        if observation.horizon_sec == reference_horizon and observation.candidate_id in candidate_ids
    ]


def _filter_group_shadow(
    observations: Sequence[ShadowExecutionObservation],
    candidate_ids: set[str],
) -> list[ShadowExecutionObservation]:
    return [observation for observation in observations if observation.candidate_id in candidate_ids]


def _record_diversity(group_records: Sequence[OutcomeEvaluationCandidate]) -> _RecordDiversity:
    distinct_experiments = {
        experiment
        for experiment in (_experiment_key(record) for record in group_records)
        if experiment != "unknown"
    }
    distinct_parameter_sets = {
        parameter_set
        for parameter_set in (_parameter_set_key(record) for record in group_records)
        if parameter_set != "unknown"
    }
    return _RecordDiversity(
        distinct_runs=len({record.run_id for record in group_records if record.run_id}),
        distinct_session_days=len({record.ts.date().isoformat() for record in group_records}),
        distinct_experiments=len(distinct_experiments),
        distinct_parameter_sets=len(distinct_parameter_sets),
    )


def _build_sample_sufficiency_card(
    *,
    group_type: str,
    group_key: str,
    sample_data: _GroupSampleData,
    diversity: _RecordDiversity,
    preferred: Sequence[int],
) -> SampleSufficiencyScorecard:
    outcome_labeled_count = len({observation.candidate_id for observation in sample_data.group_outcomes})
    shadow_labeled_count = len({observation.candidate_id for observation in sample_data.group_shadow})
    return SampleSufficiencyScorecard(
        group_type=group_type,
        group_key=group_key,
        reference_horizon_label=horizon_label(sample_data.reference_horizon),
        reference_horizon_sec=sample_data.reference_horizon,
        total_candidates=len(sample_data.candidate_ids),
        outcome_labeled_count=outcome_labeled_count,
        shadow_labeled_count=shadow_labeled_count,
        positive_outcome_count=sum(1 for observation in sample_data.group_outcomes if observation.positive_outcome),
        fillable_count=sum(1 for observation in sample_data.group_shadow if observation.full_size_fillable),
        viable_count=sum(1 for observation in sample_data.group_shadow if observation.execution_viable),
        distinct_runs=diversity.distinct_runs,
        distinct_session_days=diversity.distinct_session_days,
        distinct_experiments=diversity.distinct_experiments,
        distinct_parameter_sets=diversity.distinct_parameter_sets,
        sufficiency_score=_sample_sufficiency_score(
            outcome_labeled_count=outcome_labeled_count,
            shadow_labeled_count=shadow_labeled_count,
            distinct_runs=diversity.distinct_runs,
            distinct_session_days=diversity.distinct_session_days,
            distinct_experiments=diversity.distinct_experiments,
            distinct_parameter_sets=diversity.distinct_parameter_sets,
        ),
        sufficiency_bucket=_sample_sufficiency_bucket(
            outcome_labeled_count=outcome_labeled_count,
            shadow_labeled_count=shadow_labeled_count,
            distinct_runs=diversity.distinct_runs,
        ),
        metadata={
            "qualified_only": True,
            "reference_horizon_candidates": list(preferred),
            "source_counts": {"qualified": len(sample_data.candidate_ids)},
        },
    )


def _build_promotion_report(
    *,
    family: str,
    family_records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
    readiness: LiveReadinessScorecard | None,
    sufficiency: SampleSufficiencyScorecard | None,
    stability: dict[str, Any] | None,
) -> PromotionGateReport:
    context = _build_promotion_context(
        family=family,
        family_records=family_records,
        outcome_observations=outcome_observations,
        shadow_observations=shadow_observations,
        readiness=readiness,
        sufficiency=sufficiency,
        stability=stability,
    )
    blocker_codes, blocker_details, evidence_needed = _promotion_blockers(
        outcome_labeled_count=context.metrics.outcome_labeled_count,
        shadow_labeled_count=context.metrics.shadow_labeled_count,
        positive_outcome_ratio=context.metrics.positive_outcome_ratio,
        shadow_viability_rate=context.metrics.shadow_viability_rate,
        shadow_fillability_rate=context.metrics.shadow_fillability_rate,
        mean_execution_gap_cents=context.metrics.mean_execution_gap_cents,
        sufficiency_bucket=context.sufficiency_bucket,
        stability_bucket=context.stability_bucket,
        distinct_runs=context.diversity.distinct_runs,
        distinct_experiments=context.diversity.distinct_experiments,
        distinct_parameter_sets=context.diversity.distinct_parameter_sets,
        stability_metadata=context.stability_metadata,
    )
    return PromotionGateReport(
        strategy_family=family,
        reference_horizon_label=horizon_label(context.reference_horizon),
        reference_horizon_sec=context.reference_horizon,
        current_readiness_bucket=context.current_readiness_bucket,
        sample_sufficiency_bucket=context.sufficiency_bucket,
        stability_bucket=context.stability_bucket,
        promotion_bucket=_promotion_bucket(
            current_readiness_bucket=context.current_readiness_bucket,
            sufficiency_bucket=context.sufficiency_bucket,
            stability_bucket=context.stability_bucket,
            positive_outcome_ratio=context.metrics.positive_outcome_ratio,
            shadow_viability_rate=context.metrics.shadow_viability_rate,
            shadow_fillability_rate=context.metrics.shadow_fillability_rate,
            mean_execution_gap_cents=context.metrics.mean_execution_gap_cents,
        ),
        total_candidates=len(context.candidate_ids),
        outcome_labeled_count=context.metrics.outcome_labeled_count,
        shadow_labeled_count=context.metrics.shadow_labeled_count,
        positive_outcome_ratio=context.metrics.positive_outcome_ratio,
        shadow_viability_rate=context.metrics.shadow_viability_rate,
        shadow_fillability_rate=context.metrics.shadow_fillability_rate,
        mean_execution_gap_cents=context.metrics.mean_execution_gap_cents,
        distinct_runs=context.diversity.distinct_runs,
        distinct_experiments=context.diversity.distinct_experiments,
        distinct_parameter_sets=context.diversity.distinct_parameter_sets,
        sufficiency_score=context.sufficiency_score,
        stability_score=context.stability_score,
        overall_score=_promotion_overall_score(
            positive_outcome_ratio=context.metrics.positive_outcome_ratio,
            shadow_viability_rate=context.metrics.shadow_viability_rate,
            mean_execution_gap_cents=context.metrics.mean_execution_gap_cents,
            sufficiency_score=context.sufficiency_score,
            stability_score=context.stability_score,
        ),
        blocker_codes=blocker_codes,
        blocker_details=blocker_details,
        evidence_needed=evidence_needed,
        metadata={
            "qualified_only": True,
            "stability_dimensions": context.stability_metadata.get("dimensions", {}),
        },
    )


def _build_promotion_context(
    *,
    family: str,
    family_records: Sequence[OutcomeEvaluationCandidate],
    outcome_observations: Sequence[CandidateOutcomeObservation],
    shadow_observations: Sequence[ShadowExecutionObservation],
    readiness: LiveReadinessScorecard | None,
    sufficiency: SampleSufficiencyScorecard | None,
    stability: dict[str, Any] | None,
) -> _PromotionContext:
    candidate_ids = {record.candidate_id for record in family_records}
    stability_data = _resolve_stability_data(stability)
    reference_horizon = _resolve_reference_horizon(sufficiency, readiness)
    family_outcomes = _filter_group_outcomes(
        outcome_observations,
        candidate_ids=candidate_ids,
        reference_horizon=reference_horizon,
    )
    family_shadow = _filter_group_shadow(shadow_observations, candidate_ids)
    return _PromotionContext(
        family=family,
        candidate_ids=candidate_ids,
        reference_horizon=reference_horizon,
        current_readiness_bucket=readiness.recommendation_bucket if readiness is not None else "not_ready",
        sufficiency_bucket=sufficiency.sufficiency_bucket if sufficiency is not None else "insufficient_data",
        sufficiency_score=sufficiency.sufficiency_score if sufficiency is not None else None,
        stability_bucket=str(stability_data["bucket"]),
        stability_score=stability_data["score"],
        stability_metadata=stability_data["metadata"],
        metrics=_promotion_metrics(family_outcomes, family_shadow),
        diversity=_record_diversity(family_records),
    )


def _resolve_stability_data(stability: dict[str, Any] | None) -> dict[str, Any]:
    if stability is not None:
        return stability
    return {
        "bucket": "insufficient_slices",
        "score": None,
        "metadata": {"dimensions": {}},
    }


def _resolve_reference_horizon(
    sufficiency: SampleSufficiencyScorecard | None,
    readiness: LiveReadinessScorecard | None,
) -> int:
    return int(
        (sufficiency.reference_horizon_sec if sufficiency is not None else None)
        or (readiness.reference_horizon_sec if readiness is not None else None)
        or DEFAULT_REFERENCE_HORIZONS[0]
    )


def _promotion_metrics(
    family_outcomes: Sequence[CandidateOutcomeObservation],
    family_shadow: Sequence[ShadowExecutionObservation],
) -> _PromotionMetrics:
    sufficient_shadow = [observation for observation in family_shadow if observation.data_sufficient]
    return _PromotionMetrics(
        outcome_labeled_count=len({observation.candidate_id for observation in family_outcomes}),
        shadow_labeled_count=len({observation.candidate_id for observation in family_shadow}),
        positive_outcome_ratio=_positive_outcome_ratio(family_outcomes),
        shadow_viability_rate=_shadow_ratio(sufficient_shadow, lambda observation: observation.execution_viable),
        shadow_fillability_rate=_shadow_ratio(sufficient_shadow, lambda observation: observation.full_size_fillable),
        mean_execution_gap_cents=safe_mean(
            [
                observation.execution_gap_cents
                for observation in sufficient_shadow
                if observation.execution_gap_cents is not None
            ]
        ),
    )


def _positive_outcome_ratio(observations: Sequence[CandidateOutcomeObservation]) -> float | None:
    if not observations:
        return None
    return sum(1 for observation in observations if observation.positive_outcome) / len(observations)


def _shadow_ratio(
    observations: Sequence[ShadowExecutionObservation],
    predicate,
) -> float | None:
    if not observations:
        return None
    return sum(1 for observation in observations if predicate(observation)) / len(observations)


def _promotion_overall_score(
    *,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    mean_execution_gap_cents: float | None,
    sufficiency_score: float | None,
    stability_score: float | None,
) -> float | None:
    return _weighted_mean(
        [
            (_ratio_progress(positive_outcome_ratio, baseline=0.50, target=0.65), 0.25),
            (shadow_viability_rate, 0.20),
            (_execution_gap_score(mean_execution_gap_cents), 0.15),
            (sufficiency_score, 0.20),
            (stability_score, 0.20),
        ]
    )


def _build_stability_slice_metrics(
    *,
    slices: dict[str, list[OutcomeEvaluationCandidate]],
    group_outcomes: Sequence[CandidateOutcomeObservation],
    group_shadow: Sequence[ShadowExecutionObservation],
    total_candidates: int,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[float], int, float | None]:
    slice_metrics: list[dict[str, Any]] = []
    positive_ratios: list[float] = []
    viability_rates: list[float] = []
    execution_gaps: list[float] = []
    contributing_slice_count = 0
    max_slice_candidate_share = None

    for slice_key, slice_records in sorted(slices.items()):
        metrics = _slice_metric_summary(
            slice_key=slice_key,
            slice_records=slice_records,
            group_outcomes=group_outcomes,
            group_shadow=group_shadow,
            total_candidates=total_candidates,
        )
        slice_metrics.append(metrics["row"])
        if metrics["positive_ratio"] is not None:
            positive_ratios.append(metrics["positive_ratio"])
        if metrics["viability_rate"] is not None:
            viability_rates.append(metrics["viability_rate"])
        if metrics["execution_gap_mean"] is not None:
            execution_gaps.append(metrics["execution_gap_mean"])
        if metrics["contributes"]:
            contributing_slice_count += 1
        if metrics["candidate_share"] is not None:
            max_slice_candidate_share = max(max_slice_candidate_share or 0.0, metrics["candidate_share"])

    return (
        slice_metrics,
        positive_ratios,
        viability_rates,
        execution_gaps,
        contributing_slice_count,
        max_slice_candidate_share,
    )


def _slice_metric_summary(
    *,
    slice_key: str,
    slice_records: Sequence[OutcomeEvaluationCandidate],
    group_outcomes: Sequence[CandidateOutcomeObservation],
    group_shadow: Sequence[ShadowExecutionObservation],
    total_candidates: int,
) -> dict[str, Any]:
    slice_candidate_ids = {record.candidate_id for record in slice_records}
    slice_outcomes = [observation for observation in group_outcomes if observation.candidate_id in slice_candidate_ids]
    slice_shadow = [observation for observation in group_shadow if observation.candidate_id in slice_candidate_ids]
    sufficient_shadow = [observation for observation in slice_shadow if observation.data_sufficient]
    positive_ratio = _positive_outcome_ratio(slice_outcomes)
    viability_rate = _shadow_ratio(sufficient_shadow, lambda observation: observation.execution_viable)
    execution_gap_mean = safe_mean(
        [
            observation.execution_gap_cents
            for observation in sufficient_shadow
            if observation.execution_gap_cents is not None
        ]
    )
    candidate_share = (len(slice_candidate_ids) / total_candidates) if total_candidates > 0 else None
    return {
        "positive_ratio": positive_ratio,
        "viability_rate": viability_rate,
        "execution_gap_mean": execution_gap_mean,
        "candidate_share": candidate_share,
        "contributes": any(
            metric is not None for metric in (positive_ratio, viability_rate, execution_gap_mean)
        ),
        "row": {
            "slice_key": slice_key,
            "candidate_count": len(slice_candidate_ids),
            "candidate_share": round(candidate_share, 6) if candidate_share is not None else None,
            "positive_ratio": round(positive_ratio, 6) if positive_ratio is not None else None,
            "viability_rate": round(viability_rate, 6) if viability_rate is not None else None,
            "execution_gap_mean_cents": round(execution_gap_mean, 6) if execution_gap_mean is not None else None,
        },
    }


def _select_reference_horizon(
    candidate_ids: set[str],
    observations: Sequence[CandidateOutcomeObservation],
    preferred_horizons: Sequence[int],
) -> int:
    for horizon in preferred_horizons:
        if any(
            observation.horizon_sec == horizon and observation.candidate_id in candidate_ids
            for observation in observations
        ):
            return horizon
    return preferred_horizons[0] if preferred_horizons else 60


def _sample_sufficiency_score(
    *,
    outcome_labeled_count: int,
    shadow_labeled_count: int,
    distinct_runs: int,
    distinct_session_days: int,
    distinct_experiments: int,
    distinct_parameter_sets: int,
) -> float:
    return _weighted_mean(
        [
            (_bounded_progress(outcome_labeled_count, target=20), 0.35),
            (_bounded_progress(shadow_labeled_count, target=20), 0.25),
            (_bounded_progress(distinct_runs, target=5), 0.20),
            (_bounded_progress(distinct_session_days, target=4), 0.10),
            (_bounded_progress(max(distinct_experiments, distinct_parameter_sets), target=3), 0.10),
        ]
    ) or 0.0


def _sample_sufficiency_bucket(
    *,
    outcome_labeled_count: int,
    shadow_labeled_count: int,
    distinct_runs: int,
) -> str:
    if outcome_labeled_count < 5 or shadow_labeled_count < 5 or distinct_runs < 2:
        return "insufficient_data"
    if outcome_labeled_count < 10 or shadow_labeled_count < 8 or distinct_runs < 3:
        return "limited_data"
    if outcome_labeled_count < 20 or shadow_labeled_count < 12 or distinct_runs < 4:
        return "moderate_data"
    return "stronger_data"


def _stability_bucket(
    *,
    contributing_slice_count: int,
    consistency_score: float | None,
    max_slice_candidate_share: float | None,
) -> str:
    if contributing_slice_count < 2 or consistency_score is None:
        return "insufficient_slices"
    if consistency_score >= 0.75 and (max_slice_candidate_share or 0.0) <= 0.75:
        return "stable"
    if consistency_score >= 0.45:
        return "mixed"
    return "unstable"


def _slice_key(record: OutcomeEvaluationCandidate, slice_dimension: str) -> str:
    if slice_dimension == "experiment":
        return _experiment_key(record)
    if slice_dimension == "parameter_set":
        return _parameter_set_key(record)
    if slice_dimension == "rank_bucket":
        if record.ranking_score is None:
            return "unranked"
        if record.ranking_score >= 90.0:
            return "score_90_plus"
        if record.ranking_score >= 80.0:
            return "score_80_89"
        if record.ranking_score >= 70.0:
            return "score_70_79"
        return "score_below_70"
    if slice_dimension == "session_day":
        return record.ts.date().isoformat()
    if slice_dimension == "run":
        return record.run_id or "unknown"
    return "unknown"


def _experiment_key(record: OutcomeEvaluationCandidate) -> str:
    return _group_key(record.experiment_label or record.experiment_id)


def _parameter_set_key(record: OutcomeEvaluationCandidate) -> str:
    return _group_key(record.parameter_set_label)


def _bounded_progress(value: int | float, *, target: float) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, min(1.0, float(value) / float(target)))


def _ratio_progress(value: float | None, *, baseline: float, target: float) -> float | None:
    if value is None:
        return None
    if target <= baseline:
        return 1.0 if value >= target else 0.0
    return max(0.0, min(1.0, (value - baseline) / (target - baseline)))


def _execution_gap_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, 1.0 - (max(value, 0.0) / 0.08)))


def _inverse_spread_score(value: float | None, *, spread_target: float) -> float | None:
    if value is None:
        return None
    if spread_target <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (max(value, 0.0) / spread_target)))


def _concentration_score(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0.50:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((value - 0.50) / 0.50)))


def _sample_blockers(
    *,
    outcome_labeled_count: int,
    shadow_labeled_count: int,
    sufficiency_bucket: str,
    distinct_runs: int,
) -> list[tuple[str, Any, str]]:
    blockers: list[tuple[str, Any, str]] = []
    if outcome_labeled_count < 5:
        blockers.append(
            (
                "LOW_OUTCOME_SAMPLE",
                {"current": outcome_labeled_count, "target": 5},
                f"Collect at least {5 - outcome_labeled_count} more qualified outcome labels at the reference horizon.",
            )
        )
    if shadow_labeled_count < 5:
        blockers.append(
            (
                "LOW_SHADOW_SAMPLE",
                {"current": shadow_labeled_count, "target": 5},
                f"Collect at least {5 - shadow_labeled_count} more shadow-execution-evaluable candidates.",
            )
        )
    if sufficiency_bucket in {"insufficient_data", "limited_data"}:
        blockers.append(
            (
                "SAMPLE_SUFFICIENCY_WEAK",
                {"bucket": sufficiency_bucket},
                "Spread collection across more runs before treating this family as promotion-ready.",
            )
        )
    if distinct_runs < 2:
        blockers.append(
            (
                "LOW_RUN_DIVERSITY",
                {"current": distinct_runs, "target": 2},
                "Capture evidence from at least one more distinct run.",
            )
        )
    return blockers


def _quality_blockers(
    *,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    shadow_fillability_rate: float | None,
    mean_execution_gap_cents: float | None,
) -> list[tuple[str, Any, str]]:
    blockers: list[tuple[str, Any, str]] = []
    if positive_outcome_ratio is None or positive_outcome_ratio < 0.55:
        blockers.append(
            (
                "FORWARD_QUALITY_WEAK",
                {"positive_outcome_ratio": positive_outcome_ratio},
                "Improve qualified forward outcome quality or tighten the family thresholds.",
            )
        )
    if shadow_viability_rate is None or shadow_viability_rate < 0.50:
        blockers.append(
            (
                "SHADOW_VIABILITY_WEAK",
                {"shadow_viability_rate": shadow_viability_rate},
                "Increase fillability and execution viability before any promotion step.",
            )
        )
    if shadow_fillability_rate is None or shadow_fillability_rate < 0.40:
        blockers.append(
            (
                "FULL_SIZE_FILLABILITY_WEAK",
                {"shadow_fillability_rate": shadow_fillability_rate},
                "Collect more evidence that full-size entry is consistently fillable.",
            )
        )
    if mean_execution_gap_cents is None or mean_execution_gap_cents > 0.05:
        blockers.append(
            (
                "EXECUTION_GAP_TOO_LARGE",
                {"mean_execution_gap_cents": mean_execution_gap_cents},
                "Reduce execution gap through tighter size/depth assumptions before promotion.",
            )
        )
    return blockers


def _stability_blockers(
    *,
    stability_bucket: str,
    stability_metadata: dict[str, Any],
) -> list[tuple[str, Any, str]]:
    if stability_bucket == "insufficient_slices":
        return [
            (
                "STABILITY_UNPROVEN",
                stability_metadata,
                "Collect evidence across more parameter sets, experiments, or rank buckets.",
            )
        ]
    if stability_bucket == "unstable":
        return [
            (
                "RESULTS_UNSTABLE",
                stability_metadata,
                "Observed quality varies too much across slices; keep the family in research.",
            )
        ]
    return []


def _coverage_blockers(
    *,
    distinct_experiments: int,
    distinct_parameter_sets: int,
) -> list[tuple[str, Any, str]]:
    blockers: list[tuple[str, Any, str]] = []
    if distinct_parameter_sets < 2:
        blockers.append(
            (
                "PARAMETER_CONCENTRATION",
                {"current": distinct_parameter_sets, "target": 2},
                "Validate this family under at least one more parameter set.",
            )
        )
    if distinct_experiments == 0:
        blockers.append(
            (
                "EXPERIMENT_METADATA_THIN",
                {"current": distinct_experiments, "target": 1},
                "Tag future runs with experiment labels so stability can be compared cleanly.",
            )
        )
    return blockers


def _dedupe_blockers(
    blockers: Sequence[tuple[str, Any, str]],
) -> tuple[list[str], dict[str, Any], list[str]]:
    codes: list[str] = []
    details: dict[str, Any] = {}
    evidence_needed: list[str] = []
    for code, detail, evidence in blockers:
        if code not in codes:
            codes.append(code)
        details.setdefault(code, detail)
        if evidence not in evidence_needed:
            evidence_needed.append(evidence)
    return codes, details, evidence_needed


def _meets_deprioritize_gate(
    *,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    mean_execution_gap_cents: float | None,
) -> bool:
    weak_quality_and_viability = (
        positive_outcome_ratio is not None
        and positive_outcome_ratio < 0.45
        and shadow_viability_rate is not None
        and shadow_viability_rate < 0.40
    )
    weak_quality_and_gap = (
        positive_outcome_ratio is not None
        and positive_outcome_ratio < 0.45
        and mean_execution_gap_cents is not None
        and mean_execution_gap_cents > 0.08
    )
    return weak_quality_and_viability or weak_quality_and_gap


def _meets_future_tiny_live_gate(
    *,
    current_readiness_bucket: str,
    sufficiency_bucket: str,
    stability_bucket: str,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    shadow_fillability_rate: float | None,
    mean_execution_gap_cents: float | None,
) -> bool:
    return (
        current_readiness_bucket == "candidate_for_future_tiny_live_preparation"
        and sufficiency_bucket in {"moderate_data", "stronger_data"}
        and stability_bucket == "stable"
        and positive_outcome_ratio is not None
        and positive_outcome_ratio >= 0.60
        and shadow_viability_rate is not None
        and shadow_viability_rate >= 0.70
        and shadow_fillability_rate is not None
        and shadow_fillability_rate >= 0.50
        and mean_execution_gap_cents is not None
        and mean_execution_gap_cents <= 0.05
    )


def _meets_research_promising_gate(
    *,
    current_readiness_bucket: str,
    sufficiency_bucket: str,
    stability_bucket: str,
    positive_outcome_ratio: float | None,
    shadow_viability_rate: float | None,
    mean_execution_gap_cents: float | None,
) -> bool:
    return (
        current_readiness_bucket in {"candidate_for_future_tiny_live_preparation", "research_promising"}
        and sufficiency_bucket in {"limited_data", "moderate_data", "stronger_data"}
        and stability_bucket in {"stable", "mixed"}
        and positive_outcome_ratio is not None
        and positive_outcome_ratio >= 0.55
        and shadow_viability_rate is not None
        and shadow_viability_rate >= 0.50
        and mean_execution_gap_cents is not None
        and mean_execution_gap_cents <= 0.08
    )


def _weighted_mean(weighted_values: Sequence[tuple[float | None, float]]) -> float | None:
    usable = [(value, weight) for value, weight in weighted_values if value is not None]
    if not usable:
        return None
    total_weight = sum(weight for _value, weight in usable)
    if total_weight <= 1e-9:
        return None
    return sum(value * weight for value, weight in usable) / total_weight


def _spread(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _group_key(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    return text if text else "unknown"


def _detail_text(detail: Any) -> str:
    if isinstance(detail, dict):
        parts = [f"{key}={value}" for key, value in sorted(detail.items())]
        return ", ".join(parts)
    return str(detail)


def _evidence_matches_blocker(text: str, blocker_code: str) -> bool:
    rules = {
        "LOW_OUTCOME_SAMPLE": "outcome",
        "LOW_SHADOW_SAMPLE": "shadow",
        "SAMPLE_SUFFICIENCY_WEAK": "runs",
        "LOW_RUN_DIVERSITY": "run",
        "FORWARD_QUALITY_WEAK": "quality",
        "SHADOW_VIABILITY_WEAK": "fillability",
        "FULL_SIZE_FILLABILITY_WEAK": "full-size",
        "EXECUTION_GAP_TOO_LARGE": "execution gap",
        "STABILITY_UNPROVEN": "parameter sets",
        "RESULTS_UNSTABLE": "varies",
        "PARAMETER_CONCENTRATION": "parameter set",
        "EXPERIMENT_METADATA_THIN": "experiment",
    }
    needle = rules.get(blocker_code)
    if needle is None:
        return False
    return needle.lower() in text.lower()


def _promotion_priority(bucket: str) -> float:
    return {
        "candidate_for_future_tiny_live_preparation": 1.0,
        "research_promising": 0.8,
        "continue_research": 0.5,
        "insufficient_evidence": 0.3,
        "deprioritize_for_now": 0.1,
    }.get(bucket, 0.0)
