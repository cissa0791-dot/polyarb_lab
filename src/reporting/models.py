from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models import RunSummary


class SessionAnalytics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    session_day: str
    started_ts: datetime
    ended_ts: datetime
    markets_scanned: int = 0
    snapshots_stored: int = 0
    candidates_generated: int = 0
    risk_accepted: int = 0
    risk_rejected: int = 0
    paper_orders_created: int = 0
    execution_reports_count: int = 0
    open_positions_count: int = 0
    closed_positions_count: int = 0
    realized_pnl_total: float = 0.0
    unrealized_pnl_end: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    average_realized_pnl_per_closed_trade: float | None = None
    median_realized_pnl_per_closed_trade: float | None = None
    error_count: int = 0
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    top_markets_by_candidates: dict[str, int] = Field(default_factory=dict)
    top_opportunity_types: dict[str, int] = Field(default_factory=dict)
    raw_candidates_by_family: dict[str, int] = Field(default_factory=dict)
    qualified_candidates_by_family: dict[str, int] = Field(default_factory=dict)
    research_only_candidates_by_family: dict[str, int] = Field(default_factory=dict)
    near_miss_by_family: dict[str, int] = Field(default_factory=dict)
    rejection_reason_counts_by_family: dict[str, dict[str, int]] = Field(default_factory=dict)
    qualified_shortlist_count: int = 0
    qualification_rejection_counts_by_gate: dict[str, int] = Field(default_factory=dict)


class DailyAnalytics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_day: str
    runs_count: int = 0
    markets_scanned: int = 0
    snapshots_stored: int = 0
    candidates_generated: int = 0
    risk_accepted: int = 0
    risk_rejected: int = 0
    paper_orders_created: int = 0
    execution_reports_count: int = 0
    open_positions_count_latest: int = 0
    closed_positions_count: int = 0
    realized_pnl_total: float = 0.0
    unrealized_pnl_latest: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    average_realized_pnl_per_closed_trade: float | None = None
    median_realized_pnl_per_closed_trade: float | None = None
    error_count: int = 0
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    raw_candidates_by_family: dict[str, int] = Field(default_factory=dict)
    qualified_candidates_by_family: dict[str, int] = Field(default_factory=dict)
    research_only_candidates_by_family: dict[str, int] = Field(default_factory=dict)
    near_miss_by_family: dict[str, int] = Field(default_factory=dict)
    rejection_reason_counts_by_family: dict[str, dict[str, int]] = Field(default_factory=dict)
    qualified_shortlist_count: int = 0
    qualification_rejection_counts_by_gate: dict[str, int] = Field(default_factory=dict)


class RejectionReasonStat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reason_code: str
    count: int
    pct_of_rejections: float


class OrderbookFailureRollup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_slug: str
    side: str
    required_action: str
    strategy_family: str
    reason_code: str
    failure_class: str
    count: int
    first_seen: datetime
    last_seen: datetime
    pct_of_rejections: float


class OrderbookFunnelReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    started_ts: datetime
    ended_ts: datetime
    books_fetched: int = 0
    books_structurally_valid: int = 0
    books_execution_feasible: int = 0
    candidates_generated: int = 0
    qualified_candidates: int = 0
    books_skipped_due_to_recent_empty_asks: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyFamilyFunnelReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    started_ts: datetime
    ended_ts: datetime
    strategy_family: str
    markets_considered: int = 0
    books_fetched: int = 0
    books_structurally_valid: int = 0
    books_execution_feasible: int = 0
    raw_candidates_generated: int = 0
    markets_with_any_signal: int = 0
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    execution_mode: str | None = None
    parameter_set_label: str | None = None
    campaign_label: str | None = None


class NearMissEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    reason_code: str
    market_slug: str | None = None
    candidate_id: str | None = None
    distance_to_pass: float | None = None
    metric_value: float | None = None
    threshold_value: float | None = None
    ts: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarkoutObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position_id: str
    candidate_id: str | None = None
    symbol: str
    market_slug: str
    horizon_label: str
    horizon_sec: int
    selected_mark_age_sec: float
    selected_mark_ts: datetime
    unrealized_pnl_usd: float
    return_pct: float | None = None


class MarkoutHorizonStat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    horizon_label: str
    horizon_sec: int
    sample_count: int
    mean_unrealized_pnl_usd: float | None = None
    median_unrealized_pnl_usd: float | None = None
    positive_ratio: float | None = None
    min_unrealized_pnl_usd: float | None = None
    max_unrealized_pnl_usd: float | None = None
    mean_return_pct: float | None = None


class CandidateOutcomeObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    source: str
    strategy_family: str
    strategy_id: str
    kind: str
    market_slugs: list[str] = Field(default_factory=list)
    rank_bucket: str = "unranked"
    execution_mode: str = "paper_eligible"
    research_only: bool = False
    experiment_id: str | None = None
    experiment_label: str | None = None
    parameter_set_label: str | None = None
    horizon_label: str
    horizon_sec: int
    selected_mark_age_sec: float
    selected_mark_ts: datetime
    forward_markout_usd: float
    return_pct: float | None = None
    positive_outcome: bool = False
    ranking_score: float | None = None
    gross_edge_cents: float | None = None
    net_edge_cents: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateOutcomeHorizonStat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    horizon_label: str
    horizon_sec: int
    total_candidates: int = 0
    labeled_candidates: int = 0
    positive_ratio: float | None = None
    mean_forward_markout_usd: float | None = None
    median_forward_markout_usd: float | None = None
    min_forward_markout_usd: float | None = None
    max_forward_markout_usd: float | None = None
    mean_return_pct: float | None = None


class CandidateOutcomeScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    group_type: str
    group_key: str
    horizon_label: str
    horizon_sec: int
    total_candidates: int = 0
    labeled_candidates: int = 0
    positive_ratio: float | None = None
    mean_forward_markout_usd: float | None = None
    median_forward_markout_usd: float | None = None
    min_forward_markout_usd: float | None = None
    max_forward_markout_usd: float | None = None
    mean_return_pct: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShadowExecutionLegObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token_id: str
    market_slug: str | None = None
    action: str
    side: str | None = None
    required_shares: float
    fillable_shares: float = 0.0
    fillability_ratio: float = 0.0
    best_price: float | None = None
    estimated_vwap_price: float | None = None
    estimated_notional_usd: float = 0.0
    expected_slippage_cost_usd: float = 0.0
    available_shares: float = 0.0
    snapshot_id: int | None = None
    snapshot_ts: datetime | None = None


class ShadowExecutionObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    source: str
    strategy_family: str
    strategy_id: str
    kind: str
    market_slugs: list[str] = Field(default_factory=list)
    rank_bucket: str = "unranked"
    execution_mode: str = "paper_eligible"
    research_only: bool = False
    experiment_id: str | None = None
    experiment_label: str | None = None
    parameter_set_label: str | None = None
    shadow_status: str
    execution_viable: bool = False
    data_sufficient: bool = False
    full_size_fillable: bool = False
    theoretical_gross_edge_cents: float | None = None
    theoretical_net_edge_cents: float | None = None
    execution_adjusted_gross_edge_cents: float | None = None
    execution_adjusted_net_edge_cents: float | None = None
    execution_gap_cents: float | None = None
    required_bundle_shares: float | None = None
    executable_bundle_shares: float | None = None
    fillability_ratio: float | None = None
    required_entry_notional_usd: float | None = None
    executable_entry_notional_usd: float | None = None
    estimated_entry_bundle_vwap: float | None = None
    expected_slippage_cost_usd: float | None = None
    partial_fill_risk_score: float | None = None
    non_atomic_fragility_score: float | None = None
    snapshot_skew_sec: float | None = None
    unresolved_reason: str | None = None
    ranking_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShadowExecutionScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    group_type: str
    group_key: str
    total_candidates: int = 0
    data_sufficient_count: int = 0
    viable_count: int = 0
    full_size_fillable_count: int = 0
    partially_fillable_count: int = 0
    unresolved_count: int = 0
    fillability_rate: float | None = None
    viability_rate: float | None = None
    average_fillability_ratio: float | None = None
    average_execution_adjusted_net_edge_cents: float | None = None
    average_execution_gap_cents: float | None = None
    average_expected_slippage_cost_usd: float | None = None
    average_non_atomic_fragility_score: float | None = None
    status_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveReadinessScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    reference_horizon_label: str
    reference_horizon_sec: int
    total_candidates: int = 0
    outcome_labeled_count: int = 0
    positive_outcome_ratio: float | None = None
    mean_forward_markout_usd: float | None = None
    shadow_data_sufficient_count: int = 0
    shadow_data_sufficiency_rate: float | None = None
    shadow_viable_count: int = 0
    shadow_viability_rate: float | None = None
    shadow_fillability_rate: float | None = None
    mean_execution_adjusted_net_edge_cents: float | None = None
    mean_execution_gap_cents: float | None = None
    forward_quality_score: float | None = None
    fillability_score: float | None = None
    execution_gap_score: float | None = None
    data_sufficiency_score: float | None = None
    overall_score: float | None = None
    recommendation_bucket: str = "not_ready"
    blocking_reasons: dict[str, int] = Field(default_factory=dict)


class SampleSufficiencyScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    group_type: str
    group_key: str
    reference_horizon_label: str
    reference_horizon_sec: int
    total_candidates: int = 0
    outcome_labeled_count: int = 0
    shadow_labeled_count: int = 0
    positive_outcome_count: int = 0
    fillable_count: int = 0
    viable_count: int = 0
    distinct_runs: int = 0
    distinct_session_days: int = 0
    distinct_experiments: int = 0
    distinct_parameter_sets: int = 0
    sufficiency_score: float | None = None
    sufficiency_bucket: str = "insufficient_data"
    metadata: dict[str, Any] = Field(default_factory=dict)


class StabilityScorecard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    group_type: str
    group_key: str
    slice_dimension: str
    reference_horizon_label: str
    reference_horizon_sec: int
    total_candidates: int = 0
    slice_count: int = 0
    contributing_slice_count: int = 0
    positive_ratio_mean: float | None = None
    positive_ratio_spread: float | None = None
    viability_rate_mean: float | None = None
    viability_rate_spread: float | None = None
    execution_gap_mean: float | None = None
    execution_gap_spread: float | None = None
    max_slice_candidate_share: float | None = None
    consistency_score: float | None = None
    stability_bucket: str = "insufficient_slices"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionGateReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    reference_horizon_label: str
    reference_horizon_sec: int
    current_readiness_bucket: str = "not_ready"
    sample_sufficiency_bucket: str = "insufficient_data"
    stability_bucket: str = "insufficient_slices"
    promotion_bucket: str = "insufficient_evidence"
    total_candidates: int = 0
    outcome_labeled_count: int = 0
    shadow_labeled_count: int = 0
    positive_outcome_ratio: float | None = None
    shadow_viability_rate: float | None = None
    shadow_fillability_rate: float | None = None
    mean_execution_gap_cents: float | None = None
    distinct_runs: int = 0
    distinct_experiments: int = 0
    distinct_parameter_sets: int = 0
    sufficiency_score: float | None = None
    stability_score: float | None = None
    overall_score: float | None = None
    blocker_codes: list[str] = Field(default_factory=list)
    blocker_details: dict[str, Any] = Field(default_factory=dict)
    evidence_needed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromotionWatchlistEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    promotion_bucket: str
    watchlist_priority: float | None = None
    current_readiness_bucket: str = "not_ready"
    sample_sufficiency_bucket: str = "insufficient_data"
    stability_bucket: str = "insufficient_slices"
    watchlist_reason: str = ""
    next_step: str = ""
    blocker_codes: list[str] = Field(default_factory=list)


class PromotionBlockerEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    blocker_code: str
    detail: str = ""
    evidence_needed: str = ""
    severity: str = "medium"


class CampaignSummaryReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    campaign_id: str
    campaign_label: str
    purpose: str | None = None
    notes: str | None = None
    target_strategy_families: list[str] = Field(default_factory=list)
    target_parameter_sets: list[str] = Field(default_factory=list)
    runs_count: int = 0
    qualified_candidate_count: int = 0
    outcome_labeled_count: int = 0
    shadow_labeled_count: int = 0
    distinct_session_days: int = 0
    distinct_experiments: int = 0
    distinct_parameter_sets_observed: int = 0
    distinct_families_coverage_observed: int = 0
    distinct_parameter_sets_coverage_observed: int = 0
    family_candidate_counts: dict[str, int] = Field(default_factory=dict)
    coverage_family_counts: dict[str, int] = Field(default_factory=dict)
    parameter_set_counts: dict[str, int] = Field(default_factory=dict)
    coverage_parameter_set_counts: dict[str, int] = Field(default_factory=dict)
    recommendation_bucket: str = "collect_more_data"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FamilyEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    current_readiness_bucket: str = "not_ready"
    promotion_bucket: str = "insufficient_evidence"
    raw_candidate_count: int = 0
    qualified_candidate_count: int = 0
    outcome_labeled_count: int = 0
    shadow_labeled_count: int = 0
    distinct_runs: int = 0
    distinct_campaigns: int = 0
    distinct_parameter_sets: int = 0
    distinct_session_days: int = 0
    distinct_rank_buckets: int = 0
    campaigns: list[str] = Field(default_factory=list)
    parameter_sets: list[str] = Field(default_factory=list)
    max_campaign_share: float | None = None
    max_parameter_set_share: float | None = None
    blocker_codes: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageGapReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject_type: str
    subject_key: str
    recommendation_bucket: str
    missing_outcome_labels: int = 0
    missing_shadow_labels: int = 0
    missing_runs: int = 0
    missing_campaigns: int = 0
    missing_parameter_sets: int = 0
    missing_session_days: int = 0
    blocker_codes: list[str] = Field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject_type: str
    subject_key: str
    recommendation_bucket: str
    priority_score: float | None = None
    rationale: str = ""
    next_focus: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CampaignProgressReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    campaign_id: str
    campaign_label: str
    previous_campaign_label: str | None = None
    first_started_ts: datetime | None = None
    latest_started_ts: datetime | None = None
    runs_count: int = 0
    qualified_candidate_count: int = 0
    outcome_labeled_count: int = 0
    shadow_labeled_count: int = 0
    distinct_families_count: int = 0
    distinct_families_coverage_observed: int = 0
    distinct_parameter_sets_coverage_observed: int = 0
    family_qualified_counts: dict[str, int] = Field(default_factory=dict)
    family_outcome_counts: dict[str, int] = Field(default_factory=dict)
    family_shadow_counts: dict[str, int] = Field(default_factory=dict)
    family_coverage_counts: dict[str, int] = Field(default_factory=dict)
    parameter_set_coverage_counts: dict[str, int] = Field(default_factory=dict)
    delta_runs: int = 0
    delta_qualified_candidate_count: int = 0
    delta_outcome_labeled_count: int = 0
    delta_shadow_labeled_count: int = 0
    delta_distinct_families: int = 0
    family_qualified_growth: dict[str, int] = Field(default_factory=dict)
    family_outcome_growth: dict[str, int] = Field(default_factory=dict)
    family_shadow_growth: dict[str, int] = Field(default_factory=dict)
    evidence_improved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceTargetTracker(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject_type: str
    subject_key: str
    strategy_family: str
    parameter_set_label: str | None = None
    current_readiness_bucket: str = "not_ready"
    promotion_bucket: str = "insufficient_evidence"
    qualified_candidate_count: int = 0
    outcome_labeled_count: int = 0
    shadow_labeled_count: int = 0
    distinct_runs: int = 0
    distinct_campaigns: int = 0
    distinct_parameter_sets: int = 0
    distinct_session_days: int = 0
    missing_qualified_candidates: int = 0
    missing_outcome_labels: int = 0
    missing_shadow_labels: int = 0
    missing_runs: int = 0
    missing_campaigns: int = 0
    missing_parameter_sets: int = 0
    missing_session_days: int = 0
    stability_evidence_status: str = "unknown"
    target_progress_score: float | None = None
    blocker_codes: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionActionBacklogEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject_type: str
    subject_key: str
    strategy_family: str | None = None
    parameter_set_label: str | None = None
    campaign_label: str | None = None
    recommendation_bucket: str
    priority_score: float | None = None
    rationale: str = ""
    blockers: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionEvidenceSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_ts: datetime
    snapshot_label: str
    db_path: str
    campaign_summaries: list[CampaignSummaryReport] = Field(default_factory=list)
    family_evidence_reports: list[FamilyEvidenceReport] = Field(default_factory=list)
    promotion_gate_reports: list[PromotionGateReport] = Field(default_factory=list)
    evidence_target_trackers: list[EvidenceTargetTracker] = Field(default_factory=list)
    collection_action_backlog: list[CollectionActionBacklogEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionEvidenceDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    baseline_promotion_bucket: str = "insufficient_evidence"
    current_promotion_bucket: str = "insufficient_evidence"
    baseline_collection_bucket: str | None = None
    current_collection_bucket: str | None = None
    delta_qualified_candidate_count: int = 0
    delta_outcome_labeled_count: int = 0
    delta_shadow_labeled_count: int = 0
    delta_distinct_runs: int = 0
    delta_distinct_campaigns: int = 0
    delta_distinct_parameter_sets: int = 0
    delta_missing_outcome_labels: int = 0
    delta_missing_shadow_labels: int = 0
    delta_missing_runs: int = 0
    delta_missing_campaigns: int = 0
    delta_missing_parameter_sets: int = 0
    delta_target_progress_score: float | None = None
    status_change: str = "unchanged"
    blockers_added: list[str] = Field(default_factory=list)
    blockers_removed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionEvidenceComparisonReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_ts: datetime
    baseline_label: str
    current_label: str
    baseline_generated_ts: datetime
    current_generated_ts: datetime
    families_compared: int = 0
    families_with_evidence_gain: int = 0
    backlog_reduction_count: int = 0
    newly_promoted_families: list[str] = Field(default_factory=list)
    still_blocked_families: list[str] = Field(default_factory=list)
    family_deltas: list[CollectionEvidenceDelta] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketRollup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_slug: str
    candidate_count: int = 0
    rejection_count: int = 0
    top_rejection_reasons: dict[str, int] = Field(default_factory=dict)


class StrategyFamilyRollup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    raw_candidate_count: int = 0
    qualified_candidate_count: int = 0
    research_only_candidate_count: int = 0
    rejection_count: int = 0
    near_miss_count: int = 0
    average_gross_edge_cents: float | None = None
    average_net_edge_cents: float | None = None
    average_ranking_score: float | None = None
    top_rejection_reasons: dict[str, int] = Field(default_factory=dict)


class RankedOpportunityView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    strategy_family: str
    strategy_id: str
    kind: str
    market_slugs: list[str] = Field(default_factory=list)
    ranking_score: float = 0.0
    estimated_net_profit_usd: float = 0.0
    research_only: bool = False
    execution_mode: str = "paper_eligible"
    ts: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualificationGateRejectionStat(BaseModel):
    """Rejection count for one qualification gate across all runs in an analytics window."""

    model_config = ConfigDict(extra="ignore")

    gate: str
    count: int
    pct_of_rejections: float
    pct_of_evaluated: float


class QualificationFunnelAnalytics(BaseModel):
    """Aggregated qualification funnel statistics derived from persisted funnel reports.

    Produced by build_qualification_funnel_analytics() and attached to OfflineAnalyticsReport.
    """

    model_config = ConfigDict(extra="ignore")

    total_runs: int = 0
    total_evaluated: int = 0
    total_passed: int = 0
    total_rejected: int = 0
    pass_rate: float = 0.0
    gate_rejection_stats: list[QualificationGateRejectionStat] = Field(default_factory=list)
    shortlist_entry_count: int = 0
    shortlist_to_candidate_stored_rate: float | None = None


class OfflineAnalyticsReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_ts: datetime
    db_path: str
    session_summaries: list[SessionAnalytics] = Field(default_factory=list)
    daily_summaries: list[DailyAnalytics] = Field(default_factory=list)
    rejection_leaderboard: list[RejectionReasonStat] = Field(default_factory=list)
    orderbook_failure_rollups: list[OrderbookFailureRollup] = Field(default_factory=list)
    orderbook_funnel_reports: list[OrderbookFunnelReport] = Field(default_factory=list)
    strategy_family_funnel_reports: list[StrategyFamilyFunnelReport] = Field(default_factory=list)
    near_misses: list[NearMissEntry] = Field(default_factory=list)
    markout_horizon_stats: list[MarkoutHorizonStat] = Field(default_factory=list)
    markout_observations: list[MarkoutObservation] = Field(default_factory=list)
    candidate_outcome_horizon_stats: list[CandidateOutcomeHorizonStat] = Field(default_factory=list)
    candidate_outcome_observations: list[CandidateOutcomeObservation] = Field(default_factory=list)
    family_outcome_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    opportunity_type_outcome_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    rank_bucket_outcome_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    parameter_set_outcome_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    experiment_outcome_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    candidate_source_outcome_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    shadow_execution_observations: list[ShadowExecutionObservation] = Field(default_factory=list)
    family_shadow_execution_scorecards: list[ShadowExecutionScorecard] = Field(default_factory=list)
    rank_bucket_shadow_execution_scorecards: list[ShadowExecutionScorecard] = Field(default_factory=list)
    parameter_set_shadow_execution_scorecards: list[ShadowExecutionScorecard] = Field(default_factory=list)
    strategy_family_live_readiness: list[LiveReadinessScorecard] = Field(default_factory=list)
    sample_sufficiency_scorecards: list[SampleSufficiencyScorecard] = Field(default_factory=list)
    stability_scorecards: list[StabilityScorecard] = Field(default_factory=list)
    promotion_gate_reports: list[PromotionGateReport] = Field(default_factory=list)
    family_watchlist: list[PromotionWatchlistEntry] = Field(default_factory=list)
    promotion_blockers: list[PromotionBlockerEntry] = Field(default_factory=list)
    campaign_summaries: list[CampaignSummaryReport] = Field(default_factory=list)
    campaign_progress_reports: list[CampaignProgressReport] = Field(default_factory=list)
    family_evidence_reports: list[FamilyEvidenceReport] = Field(default_factory=list)
    coverage_gap_reports: list[CoverageGapReport] = Field(default_factory=list)
    collection_recommendations: list[CollectionRecommendation] = Field(default_factory=list)
    campaign_priority_list: list[CollectionRecommendation] = Field(default_factory=list)
    evidence_target_trackers: list[EvidenceTargetTracker] = Field(default_factory=list)
    collection_action_backlog: list[CollectionActionBacklogEntry] = Field(default_factory=list)
    market_rollups: list[MarketRollup] = Field(default_factory=list)
    strategy_family_rollups: list[StrategyFamilyRollup] = Field(default_factory=list)
    top_ranked_opportunities: list[RankedOpportunityView] = Field(default_factory=list)
    qualification_funnel_analytics: QualificationFunnelAnalytics | None = None
    methodology: dict[str, Any] = Field(default_factory=dict)


class BatchExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_id: str | None = None
    experiment_label: str | None = None
    parameter_set_label: str | None = None
    campaign_id: str | None = None
    campaign_label: str | None = None
    started_ts: datetime
    ended_ts: datetime
    cycles_requested: int
    cycles_completed: int
    run_ids: list[str] = Field(default_factory=list)
    aggregate_summary: RunSummary
    per_run_summaries: list[RunSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchCampaignExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    campaign_id: str
    campaign_label: str
    purpose: str | None = None
    notes: str | None = None
    target_strategy_families: list[str] = Field(default_factory=list)
    target_parameter_sets: list[str] = Field(default_factory=list)
    started_ts: datetime
    ended_ts: datetime
    cycles_requested_per_parameter_set: int
    cycles_completed: int
    run_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    batch_summaries: list[BatchExperimentSummary] = Field(default_factory=list)
    aggregate_summary: RunSummary
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalibrationParameterSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    min_net_edge_cents: float | None = None
    min_net_profit_usd: float | None = None
    max_spread_cents: float | None = None
    min_depth_ratio: float | None = None
    min_target_notional_usd: float | None = None
    max_partial_fill_risk: float | None = None
    max_non_atomic_risk: float | None = None
    min_ranking_score: float | None = None


class CalibrationFamilySummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy_family: str
    raw_count: int = 0
    qualified_count: int = 0
    near_miss_count: int = 0
    conversion_rate: float | None = None
    average_gross_edge_cents: float | None = None
    average_net_edge_cents: float | None = None
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)


class CalibrationParameterResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parameter_set_label: str
    total_records: int = 0
    qualified_count: int = 0
    rejected_count: int = 0
    near_miss_count: int = 0
    qualified_by_family: dict[str, int] = Field(default_factory=dict)
    near_miss_by_family: dict[str, int] = Field(default_factory=dict)
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    rejection_reason_counts_by_family: dict[str, dict[str, int]] = Field(default_factory=dict)
    family_summaries: list[CalibrationFamilySummary] = Field(default_factory=list)
    top_ranked_opportunities: list[RankedOpportunityView] = Field(default_factory=list)
    outcome_horizon_stats: list[CandidateOutcomeHorizonStat] = Field(default_factory=list)
    outcome_family_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    outcome_rank_bucket_scorecards: list[CandidateOutcomeScorecard] = Field(default_factory=list)
    shadow_execution_summary: ShadowExecutionScorecard | None = None
    shadow_execution_family_scorecards: list[ShadowExecutionScorecard] = Field(default_factory=list)
    shadow_execution_rank_bucket_scorecards: list[ShadowExecutionScorecard] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_ts: datetime
    db_path: str
    experiment_id: str | None = None
    experiment_label: str | None = None
    record_count: int = 0
    parameter_results: list[CalibrationParameterResult] = Field(default_factory=list)
    methodology: dict[str, Any] = Field(default_factory=dict)
