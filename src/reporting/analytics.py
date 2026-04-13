from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config_runtime.loader import load_runtime_config
from src.core.orderbook_validation import orderbook_failure_class
from src.domain.models import PositionMark, RejectionEvent, RunSummary, TradeSummary
from src.reporting.models import (
    CampaignProgressReport,
    CandidateOutcomeObservation,
    LiveReadinessScorecard,
    CollectionActionBacklogEntry,
    DailyAnalytics,
    CandidateOutcomeHorizonStat,
    CandidateOutcomeScorecard,
    EvidenceTargetTracker,
    MarkoutHorizonStat,
    MarkoutObservation,
    MarketRollup,
    NearMissEntry,
    OfflineAnalyticsReport,
    OrderbookFailureRollup,
    OrderbookFunnelReport,
    QualificationFunnelAnalytics,
    QualificationGateRejectionStat,
    RankedOpportunityView,
    RejectionReasonStat,
    SessionAnalytics,
    ShadowExecutionObservation,
    ShadowExecutionScorecard,
    StrategyFamilyFunnelReport,
    StrategyFamilyRollup,
)
from src.reporting.outcomes import (
    CandidateOutcomeLabeler,
    build_candidate_outcome_horizon_stats,
    build_outcome_records_from_persisted,
    build_outcome_scorecards,
    horizon_label,
    OutcomeEvaluationCandidate,
    parse_dt,
    rank_bucket,
    safe_mean,
    safe_median,
    strategy_family_from_payload,
)
from src.reporting.shadow_execution import (
    ShadowExecutionEvaluator,
    build_live_readiness_scorecards,
    build_shadow_execution_scorecards,
)
from src.reporting.promotion import (
    build_promotion_gate_reports,
    build_sample_sufficiency_scorecards,
    build_stability_scorecards,
)
from src.reporting.campaigns import (
    build_campaign_progress_reports,
    build_campaign_summary_reports,
    build_collection_action_backlog,
    build_collection_recommendations,
    build_coverage_gap_reports,
    build_evidence_target_trackers,
    build_family_evidence_reports,
)


DEFAULT_HORIZONS = (30, 60, 300, 900)


def build_qualification_funnel_analytics(
    funnel_rows: list[dict[str, Any]],
    stored_candidate_ids: set[str],
) -> QualificationFunnelAnalytics:
    """Build aggregated qualification funnel analytics from persisted funnel report rows.

    Args:
        funnel_rows:  Rows loaded from the qualification_funnel_reports table
                      (each row is the full payload_json dict).
        stored_candidate_ids:  candidate_id values present in opportunity_candidates —
                               i.e. candidates that survived sizing and were persisted.

    Returns:
        QualificationFunnelAnalytics with aggregate counts, per-gate rejection
        leaderboard, and shortlist-to-stored correlation rate.
    """
    if not funnel_rows:
        return QualificationFunnelAnalytics()

    total_evaluated = sum(int(r.get("evaluated", 0)) for r in funnel_rows)
    total_passed = sum(int(r.get("passed", 0)) for r in funnel_rows)
    total_rejected = sum(int(r.get("rejected", 0)) for r in funnel_rows)

    gate_counts: Counter[str] = Counter()
    for row in funnel_rows:
        gate_counts.update(row.get("rejection_counts", {}))

    total_gate_rejections = sum(gate_counts.values())
    gate_stats = sorted(
        [
            QualificationGateRejectionStat(
                gate=gate,
                count=count,
                pct_of_rejections=round(count / max(total_gate_rejections, 1) * 100, 2),
                pct_of_evaluated=round(count / max(total_evaluated, 1) * 100, 2),
            )
            for gate, count in gate_counts.items()
        ],
        key=lambda s: s.count,
        reverse=True,
    )

    all_shortlist_entries: list[dict[str, Any]] = []
    for row in funnel_rows:
        all_shortlist_entries.extend(row.get("shortlist", []))

    shortlist_entry_count = len(all_shortlist_entries)
    matched = sum(
        1 for entry in all_shortlist_entries
        if entry.get("candidate_id") in stored_candidate_ids
    )
    shortlist_to_stored_rate: float | None = (
        round(matched / shortlist_entry_count, 4) if shortlist_entry_count > 0 else None
    )

    return QualificationFunnelAnalytics(
        total_runs=len(funnel_rows),
        total_evaluated=total_evaluated,
        total_passed=total_passed,
        total_rejected=total_rejected,
        pass_rate=round(total_passed / max(total_evaluated, 1), 4),
        gate_rejection_stats=gate_stats,
        shortlist_entry_count=shortlist_entry_count,
        shortlist_to_candidate_stored_rate=shortlist_to_stored_rate,
    )


def resolve_sqlite_path(sqlite_url: str) -> Path:
    prefix = "sqlite:///"
    if sqlite_url.startswith(prefix):
        return Path(sqlite_url[len(prefix) :])
    return Path(sqlite_url)


def _update_nested_counter(target: dict[str, Counter[str]], source: dict[str, dict[str, int]]) -> None:
    for outer_key, nested in source.items():
        target[outer_key].update(nested)


@dataclass(frozen=True)
class _LoadedAnalyticsInputs:
    run_summaries: list[RunSummary]
    rejection_events: list[RejectionEvent]
    trade_summaries: list[TradeSummary]
    position_marks: list[PositionMark]
    raw_snapshot_rows: list[dict[str, Any]]
    position_events: list[dict[str, Any]]
    execution_reports: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    risk_decisions: list[dict[str, Any]]
    qualification_funnel_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class _OutcomeAnalyticsBundle:
    outcome_records: list[OutcomeEvaluationCandidate]
    candidate_outcome_observations: list[CandidateOutcomeObservation]
    candidate_outcome_horizon_stats: list[CandidateOutcomeHorizonStat]
    family_outcome_scorecards: list[CandidateOutcomeScorecard]
    opportunity_type_outcome_scorecards: list[CandidateOutcomeScorecard]
    rank_bucket_outcome_scorecards: list[CandidateOutcomeScorecard]
    parameter_set_outcome_scorecards: list[CandidateOutcomeScorecard]
    experiment_outcome_scorecards: list[CandidateOutcomeScorecard]
    candidate_source_outcome_scorecards: list[CandidateOutcomeScorecard]


@dataclass(frozen=True)
class _ShadowAnalyticsBundle:
    shadow_execution_observations: list[ShadowExecutionObservation]
    family_shadow_execution_scorecards: list[ShadowExecutionScorecard]
    rank_bucket_shadow_execution_scorecards: list[ShadowExecutionScorecard]
    parameter_set_shadow_execution_scorecards: list[ShadowExecutionScorecard]
    strategy_family_live_readiness: list[LiveReadinessScorecard]


@dataclass(frozen=True)
class _QualifiedResearchBundle:
    records: list[OutcomeEvaluationCandidate]
    outcome_observations: list[CandidateOutcomeObservation]
    shadow_observations: list[ShadowExecutionObservation]


@dataclass(frozen=True)
class _PromotionAnalyticsBundle:
    sample_sufficiency_scorecards: list[Any]
    stability_scorecards: list[Any]
    promotion_gate_reports: list[Any]
    family_watchlist: list[Any]
    promotion_blockers: list[Any]


@dataclass(frozen=True)
class _CampaignAnalyticsBundle:
    campaign_summaries: list[Any]
    campaign_progress_reports: list[CampaignProgressReport]
    family_evidence_reports: list[Any]
    coverage_gap_reports: list[Any]
    collection_recommendations: list[Any]
    campaign_priority_list: list[Any]
    evidence_target_trackers: list[EvidenceTargetTracker]
    collection_action_backlog: list[CollectionActionBacklogEntry]


class SQLiteAuditReader:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.tables = {
            row["name"]
            for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def load_rows(self, table: str) -> list[dict[str, Any]]:
        if not self.has_table(table):
            return []
        cursor = self.conn.execute(f"SELECT * FROM {table}")
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            payload = row.get("payload_json")
            if payload:
                try:
                    row["payload"] = json.loads(payload)
                except json.JSONDecodeError:
                    row["payload"] = {}
            metadata = row.get("metadata_json")
            if metadata:
                try:
                    row["metadata"] = json.loads(metadata)
                except json.JSONDecodeError:
                    row["metadata"] = {}
            reasons = row.get("reason_codes_json")
            if reasons:
                try:
                    row["reason_codes"] = json.loads(reasons)
                except json.JSONDecodeError:
                    row["reason_codes"] = []
            if table == "opportunity_candidates" and "payload" not in row and "metadata" in row:
                row["payload"] = row["metadata"]
        return rows

    def close(self) -> None:
        self.conn.close()


class OfflineAnalyticsService:
    def __init__(
        self,
        db_path: str | Path,
        settings_path: str = "config/settings.yaml",
        horizons: Iterable[int] = DEFAULT_HORIZONS,
    ):
        self.db_path = Path(db_path)
        self.reader = SQLiteAuditReader(self.db_path)
        self.config = load_runtime_config(settings_path)
        self.horizons = tuple(sorted(set(int(h) for h in horizons if int(h) > 0)))

    @classmethod
    def from_settings(cls, settings_path: str = "config/settings.yaml", horizons: Iterable[int] = DEFAULT_HORIZONS):
        config = load_runtime_config(settings_path)
        return cls(resolve_sqlite_path(config.storage.sqlite_url), settings_path=settings_path, horizons=horizons)

    def close(self) -> None:
        self.reader.close()

    def build_report(self) -> OfflineAnalyticsReport:
        inputs = self._load_report_inputs()
        session_summaries = self._build_session_summaries(
            inputs.run_summaries,
            inputs.rejection_events,
            inputs.trade_summaries,
            inputs.position_events,
            inputs.execution_reports,
        )
        daily_summaries = self._build_daily_summaries(session_summaries, inputs.trade_summaries)
        rejection_leaderboard = self._build_rejection_leaderboard(inputs.rejection_events)
        orderbook_failure_rollups = self._build_orderbook_failure_rollups(inputs.rejection_events)
        orderbook_funnel_reports = self._build_orderbook_funnel_reports(inputs.run_summaries)
        strategy_family_funnel_reports = self._build_strategy_family_funnel_reports(inputs.run_summaries)
        near_misses = self._build_near_misses(inputs.rejection_events, inputs.candidates, inputs.risk_decisions)
        markout_observations, markout_horizon_stats = self._build_markout_stats(inputs.position_marks)
        outcome_bundle = self._build_outcome_analytics(inputs)
        shadow_bundle = self._build_shadow_analytics(outcome_bundle, inputs.raw_snapshot_rows)
        qualified_bundle = self._build_qualified_research_bundle(outcome_bundle, shadow_bundle)
        promotion_bundle = self._build_promotion_analytics(
            qualified_bundle,
            shadow_bundle.strategy_family_live_readiness,
        )
        campaign_bundle = self._build_campaign_analytics(
            inputs.run_summaries,
            qualified_bundle,
            promotion_bundle.promotion_gate_reports,
        )
        market_rollups = self._build_market_rollups(inputs.candidates, inputs.rejection_events)
        strategy_family_rollups = self._build_strategy_family_rollups(
            inputs.run_summaries,
            inputs.candidates,
            inputs.rejection_events,
        )
        top_ranked_opportunities = self._build_top_ranked_opportunities(inputs.candidates)
        stored_candidate_ids = {
            row["candidate_id"] for row in inputs.candidates if row.get("candidate_id")
        }
        qualification_funnel_analytics = build_qualification_funnel_analytics(
            inputs.qualification_funnel_rows,
            stored_candidate_ids,
        )

        return OfflineAnalyticsReport(
            generated_ts=datetime.now(timezone.utc),
            db_path=str(self.db_path),
            session_summaries=session_summaries,
            daily_summaries=daily_summaries,
            rejection_leaderboard=rejection_leaderboard,
            orderbook_failure_rollups=orderbook_failure_rollups,
            orderbook_funnel_reports=orderbook_funnel_reports,
            strategy_family_funnel_reports=strategy_family_funnel_reports,
            near_misses=near_misses,
            markout_horizon_stats=markout_horizon_stats,
            markout_observations=markout_observations,
            candidate_outcome_horizon_stats=outcome_bundle.candidate_outcome_horizon_stats,
            candidate_outcome_observations=outcome_bundle.candidate_outcome_observations,
            family_outcome_scorecards=outcome_bundle.family_outcome_scorecards,
            opportunity_type_outcome_scorecards=outcome_bundle.opportunity_type_outcome_scorecards,
            rank_bucket_outcome_scorecards=outcome_bundle.rank_bucket_outcome_scorecards,
            parameter_set_outcome_scorecards=outcome_bundle.parameter_set_outcome_scorecards,
            experiment_outcome_scorecards=outcome_bundle.experiment_outcome_scorecards,
            candidate_source_outcome_scorecards=outcome_bundle.candidate_source_outcome_scorecards,
            shadow_execution_observations=shadow_bundle.shadow_execution_observations,
            family_shadow_execution_scorecards=shadow_bundle.family_shadow_execution_scorecards,
            rank_bucket_shadow_execution_scorecards=shadow_bundle.rank_bucket_shadow_execution_scorecards,
            parameter_set_shadow_execution_scorecards=shadow_bundle.parameter_set_shadow_execution_scorecards,
            strategy_family_live_readiness=shadow_bundle.strategy_family_live_readiness,
            sample_sufficiency_scorecards=promotion_bundle.sample_sufficiency_scorecards,
            stability_scorecards=promotion_bundle.stability_scorecards,
            promotion_gate_reports=promotion_bundle.promotion_gate_reports,
            family_watchlist=promotion_bundle.family_watchlist,
            promotion_blockers=promotion_bundle.promotion_blockers,
            campaign_summaries=campaign_bundle.campaign_summaries,
            campaign_progress_reports=campaign_bundle.campaign_progress_reports,
            family_evidence_reports=campaign_bundle.family_evidence_reports,
            coverage_gap_reports=campaign_bundle.coverage_gap_reports,
            collection_recommendations=campaign_bundle.collection_recommendations,
            campaign_priority_list=campaign_bundle.campaign_priority_list,
            evidence_target_trackers=campaign_bundle.evidence_target_trackers,
            collection_action_backlog=campaign_bundle.collection_action_backlog,
            market_rollups=market_rollups,
            strategy_family_rollups=strategy_family_rollups,
            top_ranked_opportunities=top_ranked_opportunities,
            qualification_funnel_analytics=qualification_funnel_analytics,
            methodology=self._build_methodology(),
        )

    def _load_report_inputs(self) -> _LoadedAnalyticsInputs:
        return _LoadedAnalyticsInputs(
            run_summaries=self._load_run_summaries(),
            rejection_events=self._load_rejection_events(),
            trade_summaries=self._load_trade_summaries(),
            position_marks=self._load_position_marks(),
            raw_snapshot_rows=self.reader.load_rows("raw_snapshots"),
            position_events=self.reader.load_rows("position_events"),
            execution_reports=self.reader.load_rows("execution_reports"),
            candidates=self.reader.load_rows("opportunity_candidates"),
            risk_decisions=self.reader.load_rows("risk_decisions"),
            qualification_funnel_rows=[
                r["payload"] for r in self.reader.load_rows("qualification_funnel_reports") if "payload" in r
            ],
        )

    def _build_outcome_analytics(self, inputs: _LoadedAnalyticsInputs) -> _OutcomeAnalyticsBundle:
        outcome_records = build_outcome_records_from_persisted(
            inputs.candidates,
            self.reader.load_rows("rejection_events"),
        )
        (
            candidate_outcome_observations,
            candidate_outcome_horizon_stats,
            family_outcome_scorecards,
            opportunity_type_outcome_scorecards,
            rank_bucket_outcome_scorecards,
            parameter_set_outcome_scorecards,
            experiment_outcome_scorecards,
            candidate_source_outcome_scorecards,
        ) = self._build_candidate_outcomes(outcome_records, inputs.raw_snapshot_rows)
        return _OutcomeAnalyticsBundle(
            outcome_records=outcome_records,
            candidate_outcome_observations=candidate_outcome_observations,
            candidate_outcome_horizon_stats=candidate_outcome_horizon_stats,
            family_outcome_scorecards=family_outcome_scorecards,
            opportunity_type_outcome_scorecards=opportunity_type_outcome_scorecards,
            rank_bucket_outcome_scorecards=rank_bucket_outcome_scorecards,
            parameter_set_outcome_scorecards=parameter_set_outcome_scorecards,
            experiment_outcome_scorecards=experiment_outcome_scorecards,
            candidate_source_outcome_scorecards=candidate_source_outcome_scorecards,
        )

    def _build_shadow_analytics(
        self,
        outcome_bundle: _OutcomeAnalyticsBundle,
        raw_snapshot_rows: list[dict[str, Any]],
    ) -> _ShadowAnalyticsBundle:
        (
            shadow_execution_observations,
            family_shadow_execution_scorecards,
            rank_bucket_shadow_execution_scorecards,
            parameter_set_shadow_execution_scorecards,
            strategy_family_live_readiness,
        ) = self._build_shadow_execution(
            outcome_records=outcome_bundle.outcome_records,
            raw_snapshot_rows=raw_snapshot_rows,
            family_outcome_scorecards=outcome_bundle.family_outcome_scorecards,
        )
        return _ShadowAnalyticsBundle(
            shadow_execution_observations=shadow_execution_observations,
            family_shadow_execution_scorecards=family_shadow_execution_scorecards,
            rank_bucket_shadow_execution_scorecards=rank_bucket_shadow_execution_scorecards,
            parameter_set_shadow_execution_scorecards=parameter_set_shadow_execution_scorecards,
            strategy_family_live_readiness=strategy_family_live_readiness,
        )

    def _build_qualified_research_bundle(
        self,
        outcome_bundle: _OutcomeAnalyticsBundle,
        shadow_bundle: _ShadowAnalyticsBundle,
    ) -> _QualifiedResearchBundle:
        return _QualifiedResearchBundle(
            records=[record for record in outcome_bundle.outcome_records if record.source == "qualified"],
            outcome_observations=[
                observation
                for observation in outcome_bundle.candidate_outcome_observations
                if observation.source == "qualified"
            ],
            shadow_observations=[
                observation
                for observation in shadow_bundle.shadow_execution_observations
                if observation.source == "qualified"
            ],
        )

    def _build_promotion_analytics(
        self,
        qualified_bundle: _QualifiedResearchBundle,
        readiness_scorecards: list[LiveReadinessScorecard],
    ) -> _PromotionAnalyticsBundle:
        sample_sufficiency_scorecards = self._build_sample_sufficiency_scorecards(qualified_bundle)
        stability_scorecards = build_stability_scorecards(
            qualified_bundle.records,
            qualified_bundle.outcome_observations,
            qualified_bundle.shadow_observations,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
        )
        promotion_gate_reports, family_watchlist, promotion_blockers = build_promotion_gate_reports(
            qualified_bundle.records,
            qualified_bundle.outcome_observations,
            qualified_bundle.shadow_observations,
            readiness_scorecards,
            sample_sufficiency_scorecards,
            stability_scorecards,
        )
        return _PromotionAnalyticsBundle(
            sample_sufficiency_scorecards=sample_sufficiency_scorecards,
            stability_scorecards=stability_scorecards,
            promotion_gate_reports=promotion_gate_reports,
            family_watchlist=family_watchlist,
            promotion_blockers=promotion_blockers,
        )

    def _build_sample_sufficiency_scorecards(
        self,
        qualified_bundle: _QualifiedResearchBundle,
    ) -> list[Any]:
        group_specs = (
            ("strategy_family", lambda record: record.strategy_family),
            ("rank_bucket", lambda record: rank_bucket(record.ranking_score)),
            ("parameter_set", lambda record: record.parameter_set_label or "unknown"),
            ("experiment", lambda record: record.experiment_label or record.experiment_id or "unknown"),
        )
        scorecards: list[Any] = []
        for group_type, getter in group_specs:
            scorecards.extend(
                build_sample_sufficiency_scorecards(
                    qualified_bundle.records,
                    qualified_bundle.outcome_observations,
                    qualified_bundle.shadow_observations,
                    group_type=group_type,
                    record_group_getter=getter,
                )
            )
        return scorecards

    def _build_campaign_analytics(
        self,
        run_summaries: list[RunSummary],
        qualified_bundle: _QualifiedResearchBundle,
        promotion_gate_reports: list[Any],
    ) -> _CampaignAnalyticsBundle:
        campaign_summaries = build_campaign_summary_reports(
            run_summaries,
            qualified_bundle.records,
            qualified_bundle.outcome_observations,
            qualified_bundle.shadow_observations,
        )
        campaign_progress_reports = build_campaign_progress_reports(
            run_summaries,
            qualified_bundle.records,
            qualified_bundle.outcome_observations,
            qualified_bundle.shadow_observations,
        )
        family_evidence_reports = build_family_evidence_reports(
            run_summaries,
            qualified_bundle.records,
            qualified_bundle.outcome_observations,
            qualified_bundle.shadow_observations,
            promotion_gate_reports,
        )
        coverage_gap_reports = build_coverage_gap_reports(family_evidence_reports)
        collection_recommendations, campaign_priority_list = build_collection_recommendations(
            family_evidence_reports,
            campaign_summaries,
        )
        evidence_target_trackers = build_evidence_target_trackers(
            run_summaries,
            qualified_bundle.records,
            qualified_bundle.outcome_observations,
            qualified_bundle.shadow_observations,
            family_evidence_reports,
            promotion_gate_reports,
        )
        collection_action_backlog = build_collection_action_backlog(
            campaign_summaries,
            campaign_progress_reports,
            evidence_target_trackers,
        )
        return _CampaignAnalyticsBundle(
            campaign_summaries=campaign_summaries,
            campaign_progress_reports=campaign_progress_reports,
            family_evidence_reports=family_evidence_reports,
            coverage_gap_reports=coverage_gap_reports,
            collection_recommendations=collection_recommendations,
            campaign_priority_list=campaign_priority_list,
            evidence_target_trackers=evidence_target_trackers,
            collection_action_backlog=collection_action_backlog,
        )

    def _build_methodology(self) -> dict[str, Any]:
        return {
            "markout_selection": "first persisted mark at or after the requested horizon",
            "markout_missing_data": "positions without a mark at or after the horizon are excluded for that horizon",
            "candidate_outcome_selection": "first raw clob snapshot at or after the requested horizon for each candidate leg",
            "candidate_outcome_positive_rule": "forward_markout_usd > 0",
            "candidate_outcome_missing_data": "candidate horizon labels require all legs to have an entry price and a later raw clob snapshot",
            "orderbook_failure_classification": {
                "integrity_failure": [
                    "INVALID_ORDERBOOK",
                    "MALFORMED_PRICE_LEVEL",
                    "CROSSED_BOOK",
                    "NON_MONOTONIC_BOOK",
                    "MISSING_ORDERBOOK",
                    "ORDERBOOK_FETCH_FAILED",
                ],
                "feasibility_failure": [
                    "EMPTY_ASKS",
                    "EMPTY_BIDS",
                    "NO_TOUCH_DEPTH",
                ],
            },
            "orderbook_failure_rollup_source": "post-fix rejection events with validation_rule metadata only",
            "strategy_family_funnel_source": "run summary metadata strategy_family_funnel records when present",
            "shadow_execution_entry_snapshot_selection": "nearest raw clob snapshot within +/- 5s of candidate ts, preferring the first at or after detection",
            "shadow_execution_viability": "fillable or partially fillable with positive execution-adjusted net edge and acceptable non-atomic fragility",
            "live_readiness_reference_horizon_sec": 60,
            "promotion_reference_population": "qualified candidates only",
            "sample_sufficiency_bucket_method": "count- and run-diversity-based heuristic buckets: insufficient_data, limited_data, moderate_data, stronger_data",
            "stability_slice_dimensions": ["parameter_set", "experiment", "rank_bucket", "session_day"],
            "promotion_bucket_method": "combine readiness, forward quality, shadow viability, execution gap, sample sufficiency, and stability conservatively",
            "campaign_summary_source": "run summary metadata plus qualified candidate / outcome / shadow records",
            "coverage_gap_targets": {
                "qualified_candidates": 10,
                "outcome_labels": 10,
                "shadow_labels": 10,
                "runs": 3,
                "campaigns": 2,
                "parameter_sets": 2,
                "session_days": 3,
            },
            "parameter_set_slice_targets": {
                "qualified_candidates": 5,
                "outcome_labels": 5,
                "shadow_labels": 5,
                "runs": 2,
                "campaigns": 1,
                "parameter_sets": 1,
                "session_days": 2,
            },
            "collection_recommendation_buckets": [
                "collect_more_data",
                "diversify_parameter_sets",
                "diversify_time_windows",
                "deprioritize_collection",
                "enough_data_for_research_judgment",
            ],
            "collection_ops_buckets": [
                "continue_collection",
                "close_to_sufficient",
                "enough_for_research_judgment",
                "low_yield_deprioritize",
                "diversify_collection_needed",
            ],
            "time_zone": "UTC",
            "near_miss_edge_threshold_cents": self.config.opportunity.min_edge_cents,
            "near_miss_net_profit_threshold_usd": self.config.opportunity.min_net_profit_usd,
        }

    def _load_payload_models(self, table: str, model_cls) -> list[Any]:
        return [
            model_cls.model_validate(row.get("payload", {}))
            for row in self.reader.load_rows(table)
            if row.get("payload")
        ]

    def _load_run_summaries(self) -> list[RunSummary]:
        return self._load_payload_models("run_summaries", RunSummary)

    def _load_rejection_events(self) -> list[RejectionEvent]:
        return self._load_payload_models("rejection_events", RejectionEvent)

    def _load_trade_summaries(self) -> list[TradeSummary]:
        return self._load_payload_models("trade_summaries", TradeSummary)

    def _load_position_marks(self) -> list[PositionMark]:
        return self._load_payload_models("position_marks", PositionMark)

    def _build_session_summaries(
        self,
        run_summaries: list[RunSummary],
        rejection_events: list[RejectionEvent],
        trade_summaries: list[TradeSummary],
        position_events: list[dict[str, Any]],
        execution_reports: list[dict[str, Any]],
    ) -> list[SessionAnalytics]:
        sessions: list[SessionAnalytics] = []
        for run in sorted(run_summaries, key=lambda item: item.started_ts):
            run_rejections = [event for event in rejection_events if event.run_id == run.run_id]
            run_trades = [
                trade
                for trade in trade_summaries
                if run.started_ts <= trade.closed_ts <= run.ended_ts
            ]
            run_reports = [
                row
                for row in execution_reports
                if row.get("ts") and run.started_ts <= parse_dt(row["ts"]) <= run.ended_ts
            ]
            run_position_events = [
                row
                for row in position_events
                if row.get("ts") and run.started_ts <= parse_dt(row["ts"]) <= run.ended_ts
            ]
            close_events = [
                row
                for row in run_position_events
                if row.get("event_type") in {"position_closed", "position_force_closed", "position_expired"}
            ]
            rejection_counts = Counter(event.reason_code for event in run_rejections)
            if not rejection_counts:
                rejection_counts.update(run.rejection_reason_counts)
            raw_candidates_by_family = run.metadata.get("raw_candidates_by_family", {})
            qualified_candidates_by_family = run.metadata.get("qualified_candidates_by_family", {})
            research_only_candidates_by_family = run.metadata.get("research_only_candidates_by_family", {})
            near_miss_by_family = run.metadata.get("near_miss_by_family", {})
            rejection_reason_counts_by_family = run.metadata.get("rejection_reason_counts_by_family", {})
            qual_funnel = run.metadata.get("qualification_funnel", {})
            qualified_shortlist_count = int(qual_funnel.get("qualified_shortlist_count", 0))
            qualification_rejection_counts_by_gate = dict(qual_funnel.get("rejection_counts_by_gate", {}))
            realized_pnls = [trade.realized_pnl_usd for trade in run_trades]
            closed_positions_count = len(run_trades) if run_trades else (len(close_events) if close_events else run.closed_positions)
            trade_count = len(run_trades) if run_trades else closed_positions_count

            sessions.append(
                SessionAnalytics(
                    run_id=run.run_id,
                    session_day=run.started_ts.date().isoformat(),
                    started_ts=run.started_ts,
                    ended_ts=run.ended_ts,
                    markets_scanned=run.markets_scanned,
                    snapshots_stored=run.snapshots_stored,
                    candidates_generated=run.candidates_generated,
                    risk_accepted=run.risk_accepted,
                    risk_rejected=run.risk_rejected,
                    paper_orders_created=run.paper_orders_created,
                    execution_reports_count=len(run_reports),
                    open_positions_count=run.open_positions,
                    closed_positions_count=closed_positions_count,
                    realized_pnl_total=run.realized_pnl,
                    unrealized_pnl_end=run.unrealized_pnl,
                    trade_count=trade_count,
                    win_count=sum(1 for pnl in realized_pnls if pnl > 0),
                    loss_count=sum(1 for pnl in realized_pnls if pnl < 0),
                    average_realized_pnl_per_closed_trade=safe_mean(realized_pnls),
                    median_realized_pnl_per_closed_trade=safe_median(realized_pnls),
                    error_count=run.system_errors,
                    rejection_reason_counts=dict(rejection_counts),
                    top_markets_by_candidates=run.top_markets_by_candidates,
                    top_opportunity_types=run.top_opportunity_types,
                    raw_candidates_by_family=raw_candidates_by_family,
                    qualified_candidates_by_family=qualified_candidates_by_family,
                    research_only_candidates_by_family=research_only_candidates_by_family,
                    near_miss_by_family=near_miss_by_family,
                    rejection_reason_counts_by_family=rejection_reason_counts_by_family,
                    qualified_shortlist_count=qualified_shortlist_count,
                    qualification_rejection_counts_by_gate=qualification_rejection_counts_by_gate,
                )
            )
        return sessions

    def _build_daily_summaries(self, session_summaries: list[SessionAnalytics], trade_summaries: list[TradeSummary]) -> list[DailyAnalytics]:
        trades_by_day: dict[str, list[float]] = defaultdict(list)
        for trade in trade_summaries:
            trades_by_day[trade.closed_ts.date().isoformat()].append(trade.realized_pnl_usd)

        sessions_by_day: dict[str, list[SessionAnalytics]] = defaultdict(list)
        for session in session_summaries:
            sessions_by_day[session.session_day].append(session)

        daily: list[DailyAnalytics] = []
        for day in sorted(sessions_by_day):
            sessions = sorted(sessions_by_day[day], key=lambda item: item.ended_ts)
            pnl_values = trades_by_day.get(day, [])
            rejection_counts: Counter[str] = Counter()
            raw_candidates_by_family: Counter[str] = Counter()
            qualified_candidates_by_family: Counter[str] = Counter()
            research_only_candidates_by_family: Counter[str] = Counter()
            near_miss_by_family: Counter[str] = Counter()
            rejection_reason_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
            qualified_shortlist_count = 0
            qualification_rejection_counts_by_gate: Counter[str] = Counter()
            for session in sessions:
                rejection_counts.update(session.rejection_reason_counts)
                raw_candidates_by_family.update(session.raw_candidates_by_family)
                qualified_candidates_by_family.update(session.qualified_candidates_by_family)
                research_only_candidates_by_family.update(session.research_only_candidates_by_family)
                near_miss_by_family.update(session.near_miss_by_family)
                _update_nested_counter(rejection_reason_counts_by_family, session.rejection_reason_counts_by_family)
                qualified_shortlist_count += session.qualified_shortlist_count
                qualification_rejection_counts_by_gate.update(session.qualification_rejection_counts_by_gate)

            daily.append(
                DailyAnalytics(
                    session_day=day,
                    runs_count=len(sessions),
                    markets_scanned=sum(session.markets_scanned for session in sessions),
                    snapshots_stored=sum(session.snapshots_stored for session in sessions),
                    candidates_generated=sum(session.candidates_generated for session in sessions),
                    risk_accepted=sum(session.risk_accepted for session in sessions),
                    risk_rejected=sum(session.risk_rejected for session in sessions),
                    paper_orders_created=sum(session.paper_orders_created for session in sessions),
                    execution_reports_count=sum(session.execution_reports_count for session in sessions),
                    open_positions_count_latest=sessions[-1].open_positions_count,
                    closed_positions_count=sum(session.closed_positions_count for session in sessions),
                    realized_pnl_total=sum(session.realized_pnl_total for session in sessions),
                    unrealized_pnl_latest=sessions[-1].unrealized_pnl_end,
                    trade_count=len(pnl_values),
                    win_count=sum(1 for pnl in pnl_values if pnl > 0),
                    loss_count=sum(1 for pnl in pnl_values if pnl < 0),
                    average_realized_pnl_per_closed_trade=safe_mean(pnl_values),
                    median_realized_pnl_per_closed_trade=safe_median(pnl_values),
                    error_count=sum(session.error_count for session in sessions),
                    rejection_reason_counts=dict(rejection_counts),
                    raw_candidates_by_family=dict(raw_candidates_by_family),
                    qualified_candidates_by_family=dict(qualified_candidates_by_family),
                    research_only_candidates_by_family=dict(research_only_candidates_by_family),
                    near_miss_by_family=dict(near_miss_by_family),
                    rejection_reason_counts_by_family={
                        family: dict(reason_counts)
                        for family, reason_counts in rejection_reason_counts_by_family.items()
                    },
                    qualified_shortlist_count=qualified_shortlist_count,
                    qualification_rejection_counts_by_gate=dict(qualification_rejection_counts_by_gate),
                )
            )
        return daily

    def _build_rejection_leaderboard(self, rejection_events: list[RejectionEvent]) -> list[RejectionReasonStat]:
        counts = Counter(event.reason_code for event in rejection_events)
        total = sum(counts.values())
        if total <= 0:
            return []
        return [
            RejectionReasonStat(
                reason_code=reason_code,
                count=count,
                pct_of_rejections=round((count / total) * 100.0, 6),
            )
            for reason_code, count in counts.most_common()
        ]

    def _build_orderbook_failure_rollups(
        self,
        rejection_events: list[RejectionEvent],
    ) -> list[OrderbookFailureRollup]:
        grouped: dict[tuple[str, str, str, str, str, str], list[datetime]] = defaultdict(list)
        total = 0
        for event in rejection_events:
            metadata = event.metadata or {}
            if "validation_rule" not in metadata:
                continue
            market_slug = str(metadata.get("market_slug") or "")
            side = str(metadata.get("side") or "")
            required_action = str(metadata.get("required_action") or "")
            strategy_family = str(metadata.get("strategy_family") or "unknown")
            if not market_slug or not side or not required_action:
                continue
            failure_class = str(metadata.get("failure_class") or orderbook_failure_class(event.reason_code) or "unknown_failure")
            key = (market_slug, side, required_action, strategy_family, event.reason_code, failure_class)
            grouped[key].append(event.ts)
            total += 1

        if total <= 0:
            return []

        rollups: list[OrderbookFailureRollup] = []
        for key, timestamps in grouped.items():
            market_slug, side, required_action, strategy_family, reason_code, failure_class = key
            sorted_timestamps = sorted(timestamps)
            count = len(sorted_timestamps)
            rollups.append(
                OrderbookFailureRollup(
                    market_slug=market_slug,
                    side=side,
                    required_action=required_action,
                    strategy_family=strategy_family,
                    reason_code=reason_code,
                    failure_class=failure_class,
                    count=count,
                    first_seen=sorted_timestamps[0],
                    last_seen=sorted_timestamps[-1],
                    pct_of_rejections=round((count / total) * 100.0, 6),
                )
            )

        return sorted(
            rollups,
            key=lambda item: (item.count, item.last_seen, item.market_slug, item.side),
            reverse=True,
        )

    def _build_orderbook_funnel_reports(self, run_summaries: list[RunSummary]) -> list[OrderbookFunnelReport]:
        reports: list[OrderbookFunnelReport] = []
        for run in sorted(run_summaries, key=lambda item: item.started_ts):
            funnel = run.metadata.get("orderbook_funnel")
            if not isinstance(funnel, dict):
                continue
            reports.append(
                OrderbookFunnelReport(
                    run_id=run.run_id,
                    started_ts=run.started_ts,
                    ended_ts=run.ended_ts,
                    books_fetched=int(funnel.get("books_fetched", 0)),
                    books_structurally_valid=int(funnel.get("books_structurally_valid", 0)),
                    books_execution_feasible=int(funnel.get("books_execution_feasible", 0)),
                    candidates_generated=int(funnel.get("raw_candidates_generated", 0)),
                    qualified_candidates=int(funnel.get("qualified_candidates", run.candidates_generated)),
                    books_skipped_due_to_recent_empty_asks=int(funnel.get("books_skipped_due_to_recent_empty_asks", 0)),
                    metadata={
                        "execution_mode": run.metadata.get("execution_mode"),
                        "parameter_set_label": run.metadata.get("parameter_set_label"),
                        "campaign_label": run.metadata.get("campaign_label"),
                    },
                )
            )
        return reports

    def _build_strategy_family_funnel_reports(self, run_summaries: list[RunSummary]) -> list[StrategyFamilyFunnelReport]:
        reports: list[StrategyFamilyFunnelReport] = []
        for run in sorted(run_summaries, key=lambda item: item.started_ts):
            funnel = run.metadata.get("strategy_family_funnel")
            if not isinstance(funnel, dict):
                continue
            for strategy_family in sorted(funnel):
                counts = funnel.get(strategy_family)
                if not isinstance(counts, dict):
                    continue
                reports.append(
                    StrategyFamilyFunnelReport(
                        run_id=run.run_id,
                        started_ts=run.started_ts,
                        ended_ts=run.ended_ts,
                        strategy_family=strategy_family,
                        markets_considered=int(counts.get("markets_considered", 0)),
                        books_fetched=int(counts.get("books_fetched", 0)),
                        books_structurally_valid=int(counts.get("books_structurally_valid", 0)),
                        books_execution_feasible=int(counts.get("books_execution_feasible", 0)),
                        raw_candidates_generated=int(counts.get("raw_candidates_generated", 0)),
                        markets_with_any_signal=int(counts.get("markets_with_any_signal", 0)),
                        rejection_reason_counts={
                            str(reason_code): int(count)
                            for reason_code, count in (counts.get("rejection_reason_counts", {}) or {}).items()
                        },
                        execution_mode=run.metadata.get("execution_mode"),
                        parameter_set_label=run.metadata.get("parameter_set_label"),
                        campaign_label=run.metadata.get("campaign_label"),
                    )
                )
        return reports

    def _build_near_misses(
        self,
        rejection_events: list[RejectionEvent],
        candidate_rows: list[dict[str, Any]],
        risk_decision_rows: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[NearMissEntry]:
        near_misses: list[NearMissEntry] = []
        candidate_payloads = {
            row["candidate_id"]: row.get("payload", {})
            for row in candidate_rows
            if row.get("candidate_id") and row.get("payload")
        }

        for event in rejection_events:
            if event.reason_code == "EDGE_BELOW_THRESHOLD":
                qualification = event.metadata.get("qualification", {})
                edge = event.metadata.get("edge_cents")
                if edge is None:
                    edge = qualification.get("expected_net_edge_cents")
                if edge is None:
                    continue
                market_slug = event.metadata.get("market_slug")
                if market_slug is None:
                    market_slugs = event.metadata.get("market_slugs") or []
                    market_slug = market_slugs[0] if market_slugs else None
                near_misses.append(
                    NearMissEntry(
                        source=event.stage,
                        reason_code=event.reason_code,
                        market_slug=market_slug,
                        candidate_id=event.candidate_id,
                        distance_to_pass=round(self.config.opportunity.min_edge_cents - float(edge), 6),
                        metric_value=float(edge),
                        threshold_value=self.config.opportunity.min_edge_cents,
                        ts=event.ts,
                        metadata=event.metadata,
                    )
                )
            elif event.stage == "qualification" and event.reason_code in {"INSUFFICIENT_DEPTH", "NET_PROFIT_BELOW_THRESHOLD"}:
                qualification = event.metadata.get("qualification", {})
                market_slugs = event.metadata.get("market_slugs") or []
                market_slug = event.metadata.get("market_slug") or (market_slugs[0] if market_slugs else None)
                if event.reason_code == "INSUFFICIENT_DEPTH":
                    value = float(qualification.get("available_depth_usd", 0.0))
                    threshold = float(qualification.get("required_depth_usd", 0.0))
                else:
                    value = float(qualification.get("expected_net_profit_usd", 0.0))
                    threshold = float(self.config.opportunity.min_net_profit_usd)
                if threshold <= 0:
                    continue
                near_misses.append(
                    NearMissEntry(
                        source=event.stage,
                        reason_code=event.reason_code,
                        market_slug=market_slug,
                        candidate_id=event.candidate_id,
                        distance_to_pass=round(threshold - value, 6),
                        metric_value=round(value, 6),
                        threshold_value=round(threshold, 6),
                        ts=event.ts,
                        metadata=event.metadata,
                    )
                )

        for row in risk_decision_rows:
            candidate_id = row.get("candidate_id")
            payload = candidate_payloads.get(candidate_id)
            if not payload:
                continue
            try:
                reasons = json.loads(row.get("reason_codes_json") or "[]")
            except json.JSONDecodeError:
                reasons = []
            if len(reasons) != 1:
                continue

            market_slug = None
            market_slugs = payload.get("market_slugs") or []
            if market_slugs:
                market_slug = market_slugs[0]

            if reasons[0] == "INSUFFICIENT_DEPTH":
                value = float(payload.get("estimated_depth_usd", 0.0))
                threshold = float(payload.get("target_notional_usd", 0.0))
            elif reasons[0] == "NET_PROFIT_BELOW_THRESHOLD":
                value = float(payload.get("estimated_net_profit_usd", 0.0))
                threshold = float(self.config.opportunity.min_net_profit_usd)
            else:
                continue

            near_misses.append(
                NearMissEntry(
                    source="risk",
                    reason_code=reasons[0],
                    market_slug=market_slug,
                    candidate_id=candidate_id,
                    distance_to_pass=round(threshold - value, 6),
                    metric_value=round(value, 6),
                    threshold_value=round(threshold, 6),
                    ts=parse_dt(row["ts"]) if row.get("ts") else None,
                    metadata={
                        "candidate_payload": payload,
                        "strategy_family": strategy_family_from_payload(payload, row.get("strategy_id"), row.get("kind")),
                    },
                )
            )

        sorted_near_misses = sorted(
            [item for item in near_misses if item.distance_to_pass is not None],
            key=lambda item: (item.distance_to_pass, item.ts or datetime.max.replace(tzinfo=timezone.utc)),
        )
        return sorted_near_misses[:limit]

    def _build_markout_stats(
        self,
        position_marks: list[PositionMark],
    ) -> tuple[list[MarkoutObservation], list[MarkoutHorizonStat]]:
        marks_by_position: dict[str, list[PositionMark]] = defaultdict(list)
        for mark in position_marks:
            marks_by_position[mark.position_id].append(mark)

        observations: list[MarkoutObservation] = []
        for position_id, marks in marks_by_position.items():
            sorted_marks = sorted(marks, key=lambda item: item.age_sec)
            for horizon in self.horizons:
                selected = next((mark for mark in sorted_marks if mark.age_sec >= horizon), None)
                if selected is None:
                    continue
                denominator = selected.remaining_entry_cost_usd
                return_pct = ((selected.unrealized_pnl_usd / denominator) * 100.0) if denominator > 1e-9 else None
                observations.append(
                    MarkoutObservation(
                        position_id=position_id,
                        candidate_id=selected.candidate_id,
                        symbol=selected.symbol,
                        market_slug=selected.market_slug,
                        horizon_label=horizon_label(horizon),
                        horizon_sec=horizon,
                        selected_mark_age_sec=selected.age_sec,
                        selected_mark_ts=selected.ts,
                        unrealized_pnl_usd=selected.unrealized_pnl_usd,
                        return_pct=round(return_pct, 6) if return_pct is not None else None,
                    )
                )

        stats: list[MarkoutHorizonStat] = []
        for horizon in self.horizons:
            horizon_observations = [obs for obs in observations if obs.horizon_sec == horizon]
            pnl_values = [obs.unrealized_pnl_usd for obs in horizon_observations]
            return_values = [obs.return_pct for obs in horizon_observations if obs.return_pct is not None]
            stats.append(
                MarkoutHorizonStat(
                    horizon_label=horizon_label(horizon),
                    horizon_sec=horizon,
                    sample_count=len(horizon_observations),
                    mean_unrealized_pnl_usd=safe_mean(pnl_values),
                    median_unrealized_pnl_usd=safe_median(pnl_values),
                    positive_ratio=(sum(1 for value in pnl_values if value > 0) / len(pnl_values)) if pnl_values else None,
                    min_unrealized_pnl_usd=min(pnl_values) if pnl_values else None,
                    max_unrealized_pnl_usd=max(pnl_values) if pnl_values else None,
                    mean_return_pct=safe_mean([value for value in return_values if value is not None]),
                )
            )
        return observations, stats

    def _build_market_rollups(
        self,
        candidate_rows: list[dict[str, Any]],
        rejection_events: list[RejectionEvent],
    ) -> list[MarketRollup]:
        candidate_counts: Counter[str] = Counter()
        for row in candidate_rows:
            payload = row.get("payload") or {}
            for market_slug in payload.get("market_slugs", []):
                candidate_counts[market_slug] += 1

        rejection_counts: Counter[str] = Counter()
        rejection_reasons_by_market: dict[str, Counter[str]] = defaultdict(Counter)
        for event in rejection_events:
            market_slug = event.metadata.get("market_slug")
            if market_slug is None:
                candidate_payload = next(
                    (row.get("payload") for row in candidate_rows if row.get("candidate_id") == event.candidate_id),
                    None,
                )
                if candidate_payload:
                    market_slugs = candidate_payload.get("market_slugs") or []
                    market_slug = market_slugs[0] if market_slugs else None
            if market_slug is None:
                continue
            rejection_counts[market_slug] += 1
            rejection_reasons_by_market[market_slug].update([event.reason_code])

        market_slugs = sorted(set(candidate_counts) | set(rejection_counts))
        return [
            MarketRollup(
                market_slug=market_slug,
                candidate_count=candidate_counts[market_slug],
                rejection_count=rejection_counts[market_slug],
                top_rejection_reasons=dict(rejection_reasons_by_market[market_slug].most_common(5)),
            )
            for market_slug in market_slugs
        ]

    def _build_strategy_family_rollups(
        self,
        run_summaries: list[RunSummary],
        candidate_rows: list[dict[str, Any]],
        rejection_events: list[RejectionEvent],
    ) -> list[StrategyFamilyRollup]:
        raw_counts: Counter[str] = Counter()
        qualified_counts: Counter[str] = Counter()
        qualified_counts_from_rows: Counter[str] = Counter()
        research_only_counts: Counter[str] = Counter()
        research_only_counts_from_rows: Counter[str] = Counter()
        near_miss_counts: Counter[str] = Counter()
        gross_edge_values: dict[str, list[float]] = defaultdict(list)
        net_edge_values: dict[str, list[float]] = defaultdict(list)
        ranking_scores: dict[str, list[float]] = defaultdict(list)
        rejection_counts: Counter[str] = Counter()
        rejection_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for run in run_summaries:
            raw_counts.update(run.metadata.get("raw_candidates_by_family", {}))
            qualified_counts.update(run.metadata.get("qualified_candidates_by_family", {}))
            research_only_counts.update(run.metadata.get("research_only_candidates_by_family", {}))
            near_miss_counts.update(run.metadata.get("near_miss_by_family", {}))

        for row in candidate_rows:
            payload = row.get("payload") or {}
            family = strategy_family_from_payload(payload, row.get("strategy_id"), row.get("kind"))
            qualified_counts_from_rows[family] += 1
            if bool(payload.get("research_only", False)):
                research_only_counts_from_rows[family] += 1
            if "gross_edge_cents" in payload:
                gross_edge_values[family].append(float(payload.get("gross_edge_cents", 0.0)))
            net_edge = payload.get("net_edge_cents")
            if net_edge is None:
                gross = float(payload.get("gross_edge_cents", 0.0))
                fee = float(payload.get("fee_estimate_cents", 0.0))
                slip = float(payload.get("slippage_estimate_cents", 0.0))
                latency = float(payload.get("latency_penalty_cents", 0.0))
                net_edge = gross - fee - slip - latency
            net_edge_values[family].append(float(net_edge))
            score = payload.get("ranking_score", payload.get("score"))
            if score is not None:
                ranking_scores[family].append(float(score))

        for event in rejection_events:
            family = str(event.metadata.get("strategy_family") or "unknown")
            rejection_counts[family] += 1
            rejection_reason_counts[family].update([event.reason_code])

        families = sorted(
            set(raw_counts)
            | set(qualified_counts)
            | set(research_only_counts)
            | set(near_miss_counts)
            | set(gross_edge_values)
            | set(rejection_counts)
        )
        return [
            StrategyFamilyRollup(
                strategy_family=family,
                raw_candidate_count=raw_counts[family],
                qualified_candidate_count=max(qualified_counts[family], qualified_counts_from_rows[family]),
                research_only_candidate_count=max(research_only_counts[family], research_only_counts_from_rows[family]),
                rejection_count=rejection_counts[family],
                near_miss_count=near_miss_counts[family],
                average_gross_edge_cents=safe_mean(gross_edge_values.get(family, [])),
                average_net_edge_cents=safe_mean(net_edge_values.get(family, [])),
                average_ranking_score=safe_mean(ranking_scores.get(family, [])),
                top_rejection_reasons=dict(rejection_reason_counts[family].most_common(5)),
            )
            for family in families
        ]

    def _build_top_ranked_opportunities(self, candidate_rows: list[dict[str, Any]], limit: int = 20) -> list[RankedOpportunityView]:
        items: list[RankedOpportunityView] = []
        for row in candidate_rows:
            payload = row.get("payload") or {}
            ranking_score = payload.get("ranking_score", payload.get("score"))
            if ranking_score is None:
                continue
            items.append(
                RankedOpportunityView(
                    candidate_id=row.get("candidate_id") or payload.get("candidate_id") or "unknown",
                    strategy_family=strategy_family_from_payload(payload, row.get("strategy_id"), row.get("kind")),
                    strategy_id=row.get("strategy_id") or payload.get("strategy_id") or "unknown",
                    kind=row.get("kind") or payload.get("kind") or "unknown",
                    market_slugs=list(payload.get("market_slugs") or []),
                    ranking_score=float(ranking_score),
                    estimated_net_profit_usd=float(payload.get("estimated_net_profit_usd", 0.0)),
                    research_only=bool(payload.get("research_only", False)),
                    execution_mode=str(payload.get("execution_mode", "paper_eligible")),
                    ts=parse_dt(row.get("ts") or payload.get("ts")),
                    metadata={
                        "score": payload.get("score"),
                        "strategy_tag": payload.get("strategy_tag"),
                    },
                )
            )
        return sorted(items, key=lambda item: (item.ranking_score, item.estimated_net_profit_usd), reverse=True)[:limit]

    def _build_candidate_outcomes(
        self,
        outcome_records,
        raw_snapshot_rows: list[dict[str, Any]],
    ) -> tuple[
        list[CandidateOutcomeObservation],
        list[CandidateOutcomeHorizonStat],
        list[CandidateOutcomeScorecard],
        list[CandidateOutcomeScorecard],
        list[CandidateOutcomeScorecard],
        list[CandidateOutcomeScorecard],
        list[CandidateOutcomeScorecard],
        list[CandidateOutcomeScorecard],
    ]:
        labeler = CandidateOutcomeLabeler(raw_snapshot_rows, self.horizons)
        observations = labeler.label_candidates(outcome_records)
        horizon_stats = build_candidate_outcome_horizon_stats(outcome_records, observations, self.horizons)

        family_cards = build_outcome_scorecards(
            outcome_records,
            observations,
            self.horizons,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
            observation_group_getter=lambda observation: observation.strategy_family,
        )
        opportunity_type_cards = build_outcome_scorecards(
            outcome_records,
            observations,
            self.horizons,
            group_type="opportunity_type",
            record_group_getter=lambda record: record.kind,
            observation_group_getter=lambda observation: observation.kind,
        )
        rank_bucket_cards = build_outcome_scorecards(
            outcome_records,
            observations,
            self.horizons,
            group_type="rank_bucket",
            record_group_getter=lambda record: rank_bucket(record.ranking_score),
            observation_group_getter=lambda observation: observation.rank_bucket,
        )
        parameter_set_cards = build_outcome_scorecards(
            outcome_records,
            observations,
            self.horizons,
            group_type="parameter_set",
            record_group_getter=lambda record: record.parameter_set_label or "unknown",
            observation_group_getter=lambda observation: observation.parameter_set_label or "unknown",
        )
        experiment_cards = build_outcome_scorecards(
            outcome_records,
            observations,
            self.horizons,
            group_type="experiment",
            record_group_getter=lambda record: record.experiment_label or record.experiment_id or "unknown",
            observation_group_getter=lambda observation: observation.experiment_label or observation.experiment_id or "unknown",
        )
        source_cards = build_outcome_scorecards(
            outcome_records,
            observations,
            self.horizons,
            group_type="candidate_source",
            record_group_getter=lambda record: record.source,
            observation_group_getter=lambda observation: observation.source,
        )
        return (
            observations,
            horizon_stats,
            family_cards,
            opportunity_type_cards,
            rank_bucket_cards,
            parameter_set_cards,
            experiment_cards,
            source_cards,
        )

    def _build_shadow_execution(
        self,
        outcome_records,
        raw_snapshot_rows: list[dict[str, Any]],
        family_outcome_scorecards: list[CandidateOutcomeScorecard],
    ) -> tuple[
        list[ShadowExecutionObservation],
        list[ShadowExecutionScorecard],
        list[ShadowExecutionScorecard],
        list[ShadowExecutionScorecard],
        list[LiveReadinessScorecard],
    ]:
        evaluator = ShadowExecutionEvaluator(
            raw_snapshot_rows,
            max_non_atomic_risk=self.config.opportunity.max_non_atomic_risk,
        )
        observations = evaluator.label_candidates(outcome_records)
        family_cards = build_shadow_execution_scorecards(
            outcome_records,
            observations,
            group_type="strategy_family",
            record_group_getter=lambda record: record.strategy_family,
            observation_group_getter=lambda observation: observation.strategy_family,
        )
        rank_bucket_cards = build_shadow_execution_scorecards(
            outcome_records,
            observations,
            group_type="rank_bucket",
            record_group_getter=lambda record: rank_bucket(record.ranking_score),
            observation_group_getter=lambda observation: observation.rank_bucket,
        )
        parameter_set_cards = build_shadow_execution_scorecards(
            outcome_records,
            observations,
            group_type="parameter_set",
            record_group_getter=lambda record: record.parameter_set_label or "unknown",
            observation_group_getter=lambda observation: observation.parameter_set_label or "unknown",
        )
        readiness = build_live_readiness_scorecards(
            family_outcome_scorecards,
            family_cards,
            preferred_horizon_sec=60,
        )
        return observations, family_cards, rank_bucket_cards, parameter_set_cards, readiness
