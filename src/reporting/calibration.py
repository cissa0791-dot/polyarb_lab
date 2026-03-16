from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config_runtime.loader import load_runtime_config
from src.reporting.analytics import DEFAULT_HORIZONS, SQLiteAuditReader, resolve_sqlite_path
from src.reporting.models import (
    CandidateOutcomeObservation,
    CalibrationFamilySummary,
    CalibrationParameterResult,
    CalibrationParameterSet,
    CalibrationReport,
    RankedOpportunityView,
    ShadowExecutionObservation,
    ShadowExecutionScorecard,
)
from src.reporting.outcomes import (
    CandidateOutcomeLabeler,
    build_candidate_outcome_horizon_stats,
    build_outcome_records_from_corpus,
    build_outcome_scorecards,
    rank_bucket,
)
from src.reporting.shadow_execution import (
    ShadowExecutionEvaluator,
    build_shadow_execution_scorecards,
)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _strategy_family_from_payload(payload: dict[str, Any], strategy_id: str | None = None, kind: str | None = None) -> str:
    family = payload.get("strategy_family")
    if isinstance(family, str) and family:
        return family
    strategy_text = (strategy_id or payload.get("strategy_id") or "").lower()
    kind_text = (kind or payload.get("kind") or "").lower()
    if "cross_market" in strategy_text or "cross_market" in kind_text:
        return "cross_market_constraint"
    if "single_market" in strategy_text or "single_market" in kind_text:
        return "single_market_mispricing"
    return "unknown"


@dataclass(frozen=True)
class QualificationCorpusRecord:
    candidate_id: str
    run_id: str | None
    experiment_id: str | None
    experiment_label: str | None
    parameter_set_label: str | None
    strategy_family: str
    strategy_id: str
    kind: str
    market_slugs: list[str]
    gross_edge_cents: float | None
    net_edge_cents: float | None
    expected_payout: float | None
    fee_estimate_cents: float | None
    latency_penalty_cents: float | None
    estimated_net_profit_usd: float | None
    target_notional_usd: float | None
    max_spread_cents: float | None
    required_depth_usd: float | None
    available_depth_usd: float | None
    depth_ratio: float | None
    partial_fill_risk_score: float | None
    non_atomic_execution_risk_score: float | None
    ranking_score: float | None
    research_only: bool
    execution_mode: str
    original_status: str
    original_reason_codes: list[str]
    ts: datetime
    metadata: dict[str, Any]


class ThresholdCalibrationService:
    def __init__(
        self,
        db_path: str | Path,
        settings_path: str = "config/settings.yaml",
        horizons: Iterable[int] = DEFAULT_HORIZONS,
    ):
        self.db_path = Path(db_path)
        self.reader = SQLiteAuditReader(self.db_path)
        self.config = load_runtime_config(settings_path)
        self.horizons = tuple(sorted({int(horizon) for horizon in horizons if int(horizon) > 0}))

    @classmethod
    def from_settings(cls, settings_path: str = "config/settings.yaml", horizons: Iterable[int] = DEFAULT_HORIZONS):
        config = load_runtime_config(settings_path)
        return cls(resolve_sqlite_path(config.storage.sqlite_url), settings_path=settings_path, horizons=horizons)

    def close(self) -> None:
        self.reader.close()

    def default_parameter_sets(self) -> list[CalibrationParameterSet]:
        base = self.config.opportunity
        return [
            CalibrationParameterSet(
                label="base",
                min_net_edge_cents=round(base.min_edge_cents, 6),
                min_net_profit_usd=round(base.min_net_profit_usd, 6),
                max_spread_cents=round(base.max_spread_cents, 6),
                min_depth_ratio=1.0,
                min_target_notional_usd=1.0,
                max_partial_fill_risk=round(base.max_partial_fill_risk, 6),
                max_non_atomic_risk=round(base.max_non_atomic_risk, 6),
            ),
            CalibrationParameterSet(
                label="loose",
                min_net_edge_cents=round(max(0.0, base.min_edge_cents * 0.75), 6),
                min_net_profit_usd=round(max(0.0, base.min_net_profit_usd * 0.75), 6),
                max_spread_cents=round(base.max_spread_cents * 1.25, 6),
                min_depth_ratio=0.9,
                min_target_notional_usd=0.5,
                max_partial_fill_risk=round(min(1.0, base.max_partial_fill_risk + 0.10), 6),
                max_non_atomic_risk=round(min(1.0, base.max_non_atomic_risk + 0.10), 6),
            ),
            CalibrationParameterSet(
                label="strict",
                min_net_edge_cents=round(base.min_edge_cents * 1.25, 6),
                min_net_profit_usd=round(base.min_net_profit_usd * 1.25, 6),
                max_spread_cents=round(base.max_spread_cents * 0.75, 6),
                min_depth_ratio=1.25,
                min_target_notional_usd=2.0,
                max_partial_fill_risk=round(max(0.0, base.max_partial_fill_risk - 0.10), 6),
                max_non_atomic_risk=round(max(0.0, base.max_non_atomic_risk - 0.10), 6),
            ),
        ]

    def build_report(
        self,
        parameter_sets: Iterable[CalibrationParameterSet] | None = None,
        experiment_label: str | None = None,
        experiment_id: str | None = None,
    ) -> CalibrationReport:
        run_rows = self.reader.load_rows("run_summaries")
        selected_run_ids, raw_counts_by_family = self._select_runs(run_rows, experiment_label=experiment_label, experiment_id=experiment_id)
        records = self._load_qualification_corpus(selected_run_ids, experiment_label=experiment_label, experiment_id=experiment_id)
        raw_snapshot_rows = self.reader.load_rows("raw_snapshots")
        outcome_records = build_outcome_records_from_corpus(records)
        outcome_observations = CandidateOutcomeLabeler(raw_snapshot_rows, self.horizons).label_candidates(outcome_records)
        shadow_observations = ShadowExecutionEvaluator(
            raw_snapshot_rows,
            max_non_atomic_risk=self.config.opportunity.max_non_atomic_risk,
        ).label_candidates(outcome_records)
        parameter_sets = list(parameter_sets or self.default_parameter_sets())
        parameter_results = [
            self._evaluate_parameter_set(records, raw_counts_by_family, parameter_set, outcome_observations, shadow_observations)
            for parameter_set in parameter_sets
        ]

        return CalibrationReport(
            generated_ts=datetime.now(timezone.utc),
            db_path=str(self.db_path),
            experiment_id=experiment_id,
            experiment_label=experiment_label,
            record_count=len(records),
            parameter_results=parameter_results,
            methodology={
                "source": "offline qualification replay from persisted opportunity_candidates and qualification-stage rejection_events",
                "limitation": "does not re-fetch order books or rebuild historical execution; outcome labels use persisted raw clob snapshots after candidate detection",
                "near_miss_rule": "candidate fails exactly one calibration rule with normalized distance <= 15%",
                "outcome_selection": "first raw clob snapshot at or after the requested horizon for each candidate leg",
                "shadow_execution_selection": "nearest raw clob snapshot within +/- 5s of candidate detection, preferring the first at or after detection",
            },
        )

    def _select_runs(
        self,
        run_rows: list[dict[str, Any]],
        experiment_label: str | None,
        experiment_id: str | None,
    ) -> tuple[set[str], Counter[str]]:
        selected_run_ids: set[str] = set()
        raw_counts_by_family: Counter[str] = Counter()

        for row in run_rows:
            payload = row.get("payload") or {}
            metadata = payload.get("metadata", {})
            if experiment_label and metadata.get("experiment_label") != experiment_label:
                continue
            if experiment_id and metadata.get("experiment_id") != experiment_id:
                continue
            run_id = payload.get("run_id")
            if run_id:
                selected_run_ids.add(run_id)
            raw_counts_by_family.update(metadata.get("raw_candidates_by_family", {}))

        return selected_run_ids, raw_counts_by_family

    def _load_qualification_corpus(
        self,
        selected_run_ids: set[str],
        experiment_label: str | None,
        experiment_id: str | None,
    ) -> list[QualificationCorpusRecord]:
        accepted_records = self._load_accepted_records(selected_run_ids, experiment_label, experiment_id)
        rejected_records = self._load_rejected_records(selected_run_ids, experiment_label, experiment_id)
        deduped: dict[str, QualificationCorpusRecord] = {}
        for record in accepted_records + rejected_records:
            deduped[f"{record.run_id}:{record.candidate_id}:{record.original_status}"] = record
        return sorted(deduped.values(), key=lambda item: item.ts)

    def _load_accepted_records(
        self,
        selected_run_ids: set[str],
        experiment_label: str | None,
        experiment_id: str | None,
    ) -> list[QualificationCorpusRecord]:
        rows = self.reader.load_rows("opportunity_candidates")
        records: list[QualificationCorpusRecord] = []
        for row in rows:
            payload = row.get("payload") or {}
            metadata = payload.get("metadata", {})
            if not self._matches_filter(selected_run_ids, experiment_label, experiment_id, metadata):
                continue

            qualification = metadata.get("qualification", {})
            legs = payload.get("legs") or qualification.get("legs") or []
            required_depth = self._as_float(payload.get("required_depth_usd", qualification.get("required_depth_usd")))
            available_depth = self._as_float(payload.get("available_depth_usd", qualification.get("available_depth_usd")))
            records.append(
                QualificationCorpusRecord(
                    candidate_id=str(row.get("candidate_id") or payload.get("candidate_id") or "unknown"),
                    run_id=metadata.get("run_id"),
                    experiment_id=metadata.get("experiment_id"),
                    experiment_label=metadata.get("experiment_label"),
                    parameter_set_label=metadata.get("parameter_set_label"),
                    strategy_family=_strategy_family_from_payload(payload, row.get("strategy_id"), row.get("kind")),
                    strategy_id=str(row.get("strategy_id") or payload.get("strategy_id") or "unknown"),
                    kind=str(row.get("kind") or payload.get("kind") or "unknown"),
                    market_slugs=list(payload.get("market_slugs") or []),
                    gross_edge_cents=self._as_float(payload.get("gross_edge_cents")),
                    net_edge_cents=self._net_edge_cents(payload, qualification),
                    expected_payout=self._as_float(payload.get("expected_payout")),
                    fee_estimate_cents=self._as_float(payload.get("fee_estimate_cents")),
                    latency_penalty_cents=self._as_float(payload.get("latency_penalty_cents")),
                    estimated_net_profit_usd=self._as_float(payload.get("estimated_net_profit_usd", qualification.get("expected_net_profit_usd"))),
                    target_notional_usd=self._as_float(payload.get("target_notional_usd")),
                    max_spread_cents=self._max_spread_from_legs(legs),
                    required_depth_usd=required_depth,
                    available_depth_usd=available_depth,
                    depth_ratio=(available_depth / required_depth) if required_depth and required_depth > 1e-9 and available_depth is not None else None,
                    partial_fill_risk_score=self._as_float(payload.get("partial_fill_risk_score", qualification.get("partial_fill_risk_score"))),
                    non_atomic_execution_risk_score=self._as_float(payload.get("non_atomic_execution_risk_score", qualification.get("non_atomic_execution_risk_score"))),
                    ranking_score=self._as_float(payload.get("ranking_score", payload.get("score"))),
                    research_only=bool(payload.get("research_only", False)),
                    execution_mode=str(payload.get("execution_mode", "paper_eligible")),
                    original_status="qualified",
                    original_reason_codes=[],
                    ts=_parse_dt(row.get("ts") or payload.get("ts")),
                    metadata=metadata,
                )
            )
        return records

    def _load_rejected_records(
        self,
        selected_run_ids: set[str],
        experiment_label: str | None,
        experiment_id: str | None,
    ) -> list[QualificationCorpusRecord]:
        grouped: dict[tuple[str | None, str], dict[str, Any]] = {}
        for row in self.reader.load_rows("rejection_events"):
            payload = row.get("payload") or {}
            if payload.get("stage") != "qualification":
                continue
            metadata = payload.get("metadata", {})
            if not self._matches_filter(selected_run_ids, experiment_label, experiment_id, metadata, row_run_id=payload.get("run_id")):
                continue
            candidate_id = payload.get("candidate_id")
            if not candidate_id:
                continue
            key = (payload.get("run_id"), candidate_id)
            entry = grouped.setdefault(
                key,
                {
                    "payload": payload,
                    "reason_codes": set(),
                },
            )
            entry["reason_codes"].add(payload.get("reason_code"))

        records: list[QualificationCorpusRecord] = []
        for (run_id, candidate_id), entry in grouped.items():
            payload = entry["payload"]
            metadata = payload.get("metadata", {})
            raw_candidate = metadata.get("raw_candidate", {})
            qualification = metadata.get("qualification", {})
            legs = qualification.get("legs") or raw_candidate.get("legs") or []
            required_depth = self._as_float(qualification.get("required_depth_usd"))
            available_depth = self._as_float(qualification.get("available_depth_usd"))
            records.append(
                QualificationCorpusRecord(
                    candidate_id=str(candidate_id),
                    run_id=run_id,
                    experiment_id=metadata.get("experiment_id"),
                    experiment_label=metadata.get("experiment_label"),
                    parameter_set_label=metadata.get("parameter_set_label"),
                    strategy_family=str(metadata.get("strategy_family") or _strategy_family_from_payload(raw_candidate)),
                    strategy_id=str(raw_candidate.get("strategy_id") or "unknown"),
                    kind=str(raw_candidate.get("kind") or "unknown"),
                    market_slugs=list(raw_candidate.get("market_slugs") or metadata.get("market_slugs") or []),
                    gross_edge_cents=self._as_float(raw_candidate.get("gross_edge_cents")),
                    net_edge_cents=self._as_float(qualification.get("expected_net_edge_cents")),
                    expected_payout=self._as_float(raw_candidate.get("expected_payout")),
                    fee_estimate_cents=self._as_float(raw_candidate.get("fee_estimate_cents")),
                    latency_penalty_cents=self._as_float(raw_candidate.get("latency_penalty_cents")),
                    estimated_net_profit_usd=self._as_float(qualification.get("expected_net_profit_usd")),
                    target_notional_usd=self._as_float(raw_candidate.get("target_notional_usd")),
                    max_spread_cents=self._max_spread_from_legs(legs),
                    required_depth_usd=required_depth,
                    available_depth_usd=available_depth,
                    depth_ratio=(available_depth / required_depth) if required_depth and required_depth > 1e-9 and available_depth is not None else None,
                    partial_fill_risk_score=self._as_float(qualification.get("partial_fill_risk_score")),
                    non_atomic_execution_risk_score=self._as_float(qualification.get("non_atomic_execution_risk_score")),
                    ranking_score=None,
                    research_only=bool(metadata.get("research_only", raw_candidate.get("research_only", False))),
                    execution_mode=str(metadata.get("execution_mode", raw_candidate.get("execution_mode", "paper_eligible"))),
                    original_status="rejected",
                    original_reason_codes=sorted(code for code in entry["reason_codes"] if code),
                    ts=_parse_dt(payload.get("ts")),
                    metadata=metadata,
                )
            )
        return records

    def _evaluate_parameter_set(
        self,
        records: list[QualificationCorpusRecord],
        raw_counts_by_family: Counter[str],
        parameter_set: CalibrationParameterSet,
        outcome_observations: list[CandidateOutcomeObservation],
        shadow_observations: list[ShadowExecutionObservation],
    ) -> CalibrationParameterResult:
        qualified_records: list[QualificationCorpusRecord] = []
        rejection_counts: Counter[str] = Counter()
        rejection_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
        qualified_by_family: Counter[str] = Counter()
        near_miss_by_family: Counter[str] = Counter()
        near_miss_count = 0

        record_counts_by_family: Counter[str] = Counter(record.strategy_family for record in records)
        for record in records:
            failures = self._evaluate_record(record, parameter_set)
            if not failures:
                qualified_records.append(record)
                qualified_by_family[record.strategy_family] += 1
                continue

            if self._is_near_miss(failures):
                near_miss_count += 1
                near_miss_by_family[record.strategy_family] += 1
            for reason_code, _gap_ratio in failures:
                rejection_counts[reason_code] += 1
                rejection_counts_by_family[record.strategy_family].update([reason_code])

        families = sorted(set(raw_counts_by_family) | set(record_counts_by_family) | set(qualified_by_family) | set(near_miss_by_family))
        family_summaries: list[CalibrationFamilySummary] = []
        for family in families:
            family_qualified = [record for record in qualified_records if record.strategy_family == family]
            raw_count = max(raw_counts_by_family[family], record_counts_by_family[family])
            qualified_count = qualified_by_family[family]
            family_summaries.append(
                CalibrationFamilySummary(
                    strategy_family=family,
                    raw_count=raw_count,
                    qualified_count=qualified_count,
                    near_miss_count=near_miss_by_family[family],
                    conversion_rate=(qualified_count / raw_count) if raw_count > 0 else None,
                    average_gross_edge_cents=_safe_mean([value for value in (record.gross_edge_cents for record in family_qualified) if value is not None]),
                    average_net_edge_cents=_safe_mean([value for value in (record.net_edge_cents for record in family_qualified) if value is not None]),
                    rejection_reason_counts=dict(rejection_counts_by_family[family]),
                )
            )

        top_ranked = sorted(
            [record for record in qualified_records if record.ranking_score is not None],
            key=lambda item: (float(item.ranking_score or 0.0), float(item.estimated_net_profit_usd or 0.0)),
            reverse=True,
        )[:20]
        qualified_outcome_records = build_outcome_records_from_corpus(qualified_records)
        qualified_candidate_ids = {record.candidate_id for record in qualified_outcome_records}
        filtered_outcomes = [
            observation
            for observation in outcome_observations
            if observation.candidate_id in qualified_candidate_ids
        ]
        filtered_shadow = [
            observation
            for observation in shadow_observations
            if observation.candidate_id in qualified_candidate_ids
        ]
        outcome_horizon_stats = build_candidate_outcome_horizon_stats(qualified_outcome_records, filtered_outcomes, self.horizons)
        outcome_family_scorecards = build_outcome_scorecards(
            qualified_outcome_records,
            filtered_outcomes,
            self.horizons,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
            observation_group_getter=lambda observation: observation.strategy_family,
        )
        outcome_rank_bucket_scorecards = build_outcome_scorecards(
            qualified_outcome_records,
            filtered_outcomes,
            self.horizons,
            group_type="rank_bucket",
            record_group_getter=lambda record: rank_bucket(record.ranking_score),
            observation_group_getter=lambda observation: observation.rank_bucket,
        )
        shadow_execution_summary = next(
            iter(
                build_shadow_execution_scorecards(
                    qualified_outcome_records,
                    filtered_shadow,
                    group_type="parameter_set",
                    record_group_getter=lambda _record: parameter_set.label,
                    observation_group_getter=lambda _observation: parameter_set.label,
                )
            ),
            None,
        )
        shadow_execution_family_scorecards = build_shadow_execution_scorecards(
            qualified_outcome_records,
            filtered_shadow,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
            observation_group_getter=lambda observation: observation.strategy_family,
        )
        shadow_execution_rank_bucket_scorecards = build_shadow_execution_scorecards(
            qualified_outcome_records,
            filtered_shadow,
            group_type="rank_bucket",
            record_group_getter=lambda record: rank_bucket(record.ranking_score),
            observation_group_getter=lambda observation: observation.rank_bucket,
        )

        return CalibrationParameterResult(
            parameter_set_label=parameter_set.label,
            total_records=len(records),
            qualified_count=len(qualified_records),
            rejected_count=len(records) - len(qualified_records),
            near_miss_count=near_miss_count,
            qualified_by_family=dict(qualified_by_family),
            near_miss_by_family=dict(near_miss_by_family),
            rejection_reason_counts=dict(rejection_counts),
            rejection_reason_counts_by_family={family: dict(counts) for family, counts in rejection_counts_by_family.items()},
            family_summaries=family_summaries,
            top_ranked_opportunities=[
                RankedOpportunityView(
                    candidate_id=record.candidate_id,
                    strategy_family=record.strategy_family,
                    strategy_id=record.strategy_id,
                    kind=record.kind,
                    market_slugs=record.market_slugs,
                    ranking_score=float(record.ranking_score or 0.0),
                    estimated_net_profit_usd=float(record.estimated_net_profit_usd or 0.0),
                    research_only=record.research_only,
                    execution_mode=record.execution_mode,
                    ts=record.ts,
                    metadata={
                        "parameter_set_label": parameter_set.label,
                        "run_id": record.run_id,
                    },
                )
                for record in top_ranked
            ],
            outcome_horizon_stats=outcome_horizon_stats,
            outcome_family_scorecards=outcome_family_scorecards,
            outcome_rank_bucket_scorecards=outcome_rank_bucket_scorecards,
            shadow_execution_summary=shadow_execution_summary,
            shadow_execution_family_scorecards=shadow_execution_family_scorecards,
            shadow_execution_rank_bucket_scorecards=shadow_execution_rank_bucket_scorecards,
        )

    def _evaluate_record(self, record: QualificationCorpusRecord, parameter_set: CalibrationParameterSet) -> list[tuple[str, float]]:
        failures: list[tuple[str, float]] = []
        if parameter_set.min_net_edge_cents is not None and self._below_min(record.net_edge_cents, parameter_set.min_net_edge_cents):
            failures.append(("EDGE_BELOW_THRESHOLD", self._gap_ratio_min(record.net_edge_cents, parameter_set.min_net_edge_cents)))
        if parameter_set.min_net_profit_usd is not None and self._below_min(record.estimated_net_profit_usd, parameter_set.min_net_profit_usd):
            failures.append(("NET_PROFIT_BELOW_THRESHOLD", self._gap_ratio_min(record.estimated_net_profit_usd, parameter_set.min_net_profit_usd)))
        if parameter_set.max_spread_cents is not None and self._above_max(record.max_spread_cents, parameter_set.max_spread_cents):
            failures.append(("SPREAD_TOO_WIDE", self._gap_ratio_max(record.max_spread_cents, parameter_set.max_spread_cents)))
        if parameter_set.min_depth_ratio is not None and self._below_min(record.depth_ratio, parameter_set.min_depth_ratio):
            failures.append(("INSUFFICIENT_DEPTH", self._gap_ratio_min(record.depth_ratio, parameter_set.min_depth_ratio)))
        if parameter_set.min_target_notional_usd is not None and self._below_min(record.target_notional_usd, parameter_set.min_target_notional_usd):
            failures.append(("SIZE_BELOW_MINIMUM", self._gap_ratio_min(record.target_notional_usd, parameter_set.min_target_notional_usd)))
        if parameter_set.max_partial_fill_risk is not None and self._above_max(record.partial_fill_risk_score, parameter_set.max_partial_fill_risk):
            failures.append(("PARTIAL_FILL_RISK_TOO_HIGH", self._gap_ratio_max(record.partial_fill_risk_score, parameter_set.max_partial_fill_risk)))
        if parameter_set.max_non_atomic_risk is not None and self._above_max(record.non_atomic_execution_risk_score, parameter_set.max_non_atomic_risk):
            failures.append(("NON_ATOMIC_RISK_TOO_HIGH", self._gap_ratio_max(record.non_atomic_execution_risk_score, parameter_set.max_non_atomic_risk)))
        if parameter_set.min_ranking_score is not None and self._below_min(record.ranking_score, parameter_set.min_ranking_score):
            failures.append(("RANKING_BELOW_THRESHOLD", self._gap_ratio_min(record.ranking_score, parameter_set.min_ranking_score)))
        return failures

    def _is_near_miss(self, failures: list[tuple[str, float]]) -> bool:
        return len(failures) == 1 and failures[0][1] <= 0.15

    def _matches_filter(
        self,
        selected_run_ids: set[str],
        experiment_label: str | None,
        experiment_id: str | None,
        metadata: dict[str, Any],
        row_run_id: str | None = None,
    ) -> bool:
        if experiment_label and metadata.get("experiment_label") != experiment_label:
            return False
        if experiment_id and metadata.get("experiment_id") != experiment_id:
            return False
        if selected_run_ids:
            candidate_run_id = metadata.get("run_id") or row_run_id
            return candidate_run_id in selected_run_ids
        return True

    def _net_edge_cents(self, payload: dict[str, Any], qualification: dict[str, Any]) -> float | None:
        value = qualification.get("expected_net_edge_cents")
        if value is not None:
            return self._as_float(value)
        gross = self._as_float(payload.get("gross_edge_cents"))
        fee = self._as_float(payload.get("fee_estimate_cents")) or 0.0
        slippage = self._as_float(payload.get("slippage_estimate_cents")) or 0.0
        latency = self._as_float(payload.get("latency_penalty_cents")) or 0.0
        if gross is None:
            return None
        return gross - fee - slippage - latency

    def _max_spread_from_legs(self, legs: list[dict[str, Any]]) -> float | None:
        spreads = [self._as_float(leg.get("spread_cents")) for leg in legs if self._as_float(leg.get("spread_cents")) is not None]
        if not spreads:
            return None
        return max(spreads)

    def _below_min(self, value: float | None, threshold: float) -> bool:
        return value is None or value < threshold

    def _above_max(self, value: float | None, threshold: float) -> bool:
        return value is None or value > threshold

    def _gap_ratio_min(self, value: float | None, threshold: float) -> float:
        if value is None:
            return 1.0
        return max(0.0, (threshold - value) / max(abs(threshold), 1e-9))

    def _gap_ratio_max(self, value: float | None, threshold: float) -> float:
        if value is None:
            return 1.0
        return max(0.0, (value - threshold) / max(abs(threshold), 1e-9))

    def _as_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
