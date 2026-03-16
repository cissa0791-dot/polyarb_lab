from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable, Sequence

from src.reporting.models import (
    CandidateOutcomeObservation,
    LiveReadinessScorecard,
    ShadowExecutionLegObservation,
    ShadowExecutionObservation,
    ShadowExecutionScorecard,
)
from src.reporting.outcomes import OutcomeEvaluationCandidate, horizon_label, parse_dt, rank_bucket, safe_mean


DEFAULT_ENTRY_SNAPSHOT_TOLERANCE_SEC = 5.0
DEFAULT_MAX_SNAPSHOT_SKEW_SEC = 2.0

FILLABLE_STATUS = "FILLABLE"
PARTIALLY_FILLABLE_STATUS = "PARTIALLY_FILLABLE"
NOT_FILLABLE_STATUS = "NOT_FILLABLE"
INSUFFICIENT_SYNCHRONIZED_DEPTH_STATUS = "INSUFFICIENT_SYNCHRONIZED_DEPTH"
NON_ATOMIC_RISK_TOO_HIGH_STATUS = "NON_ATOMIC_RISK_TOO_HIGH"
EXECUTION_EDGE_NON_POSITIVE_STATUS = "EXECUTION_EDGE_NON_POSITIVE"
UNRESOLVED_MISSING_DATA_STATUS = "UNRESOLVED_MISSING_DATA"


class ShadowExecutionEvaluator:
    def __init__(
        self,
        raw_snapshot_rows: Iterable[dict[str, Any]],
        *,
        entry_snapshot_tolerance_sec: float = DEFAULT_ENTRY_SNAPSHOT_TOLERANCE_SEC,
        max_snapshot_skew_sec: float = DEFAULT_MAX_SNAPSHOT_SKEW_SEC,
        max_non_atomic_risk: float = 1.0,
    ):
        self.entry_snapshot_tolerance_sec = float(entry_snapshot_tolerance_sec)
        self.max_snapshot_skew_sec = float(max_snapshot_skew_sec)
        self.max_non_atomic_risk = float(max_non_atomic_risk)
        self.snapshot_rows_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.snapshot_ts_by_token: dict[str, list[datetime]] = {}
        self._build_snapshot_index(raw_snapshot_rows)

    def label_candidates(self, records: Iterable[OutcomeEvaluationCandidate]) -> list[ShadowExecutionObservation]:
        observations: list[ShadowExecutionObservation] = []
        for record in records:
            if not record.legs:
                continue
            observations.append(self._label_candidate(record))
        return sorted(observations, key=lambda item: (item.strategy_family, item.candidate_id))

    def _build_snapshot_index(self, raw_snapshot_rows: Iterable[dict[str, Any]]) -> None:
        for row in raw_snapshot_rows:
            if str(row.get("source", "")).lower() != "clob":
                continue
            payload = row.get("payload") or {}
            token_id = str(row.get("entity_id") or payload.get("token_id") or "")
            snapshot_ts = row.get("ingest_ts") or row.get("ts")
            if not token_id or snapshot_ts is None:
                continue
            row_copy = dict(row)
            row_copy["payload"] = payload
            row_copy["_snapshot_ts"] = parse_dt(snapshot_ts)
            self.snapshot_rows_by_token[token_id].append(row_copy)

        for token_id, rows in self.snapshot_rows_by_token.items():
            rows.sort(key=lambda item: item["_snapshot_ts"])
            self.snapshot_ts_by_token[token_id] = [item["_snapshot_ts"] for item in rows]

    def _label_candidate(self, record: OutcomeEvaluationCandidate) -> ShadowExecutionObservation:
        leg_observations: list[ShadowExecutionLegObservation] = []
        leg_contexts: list[tuple[dict[str, Any], dict[str, Any], ShadowExecutionLegObservation]] = []
        unresolved_reason: str | None = None
        required_bundle_shares = min(self._as_float(leg.get("required_shares")) or 0.0 for leg in record.legs) if record.legs else None

        for leg in record.legs:
            token_id = str(leg.get("token_id") or "")
            required_shares = self._as_float(leg.get("required_shares")) or 0.0
            snapshot = self._find_entry_snapshot(token_id, record.ts)
            if not token_id or required_shares <= 1e-9 or snapshot is None:
                unresolved_reason = UNRESOLVED_MISSING_DATA_STATUS
                break
            estimate = self._estimate_leg_fill(leg, snapshot, required_shares)
            if estimate is None:
                unresolved_reason = UNRESOLVED_MISSING_DATA_STATUS
                break
            leg_observations.append(estimate)
            leg_contexts.append((leg, snapshot, estimate))

        if unresolved_reason is not None:
            return ShadowExecutionObservation(
                candidate_id=record.candidate_id,
                source=record.source,
                strategy_family=record.strategy_family,
                strategy_id=record.strategy_id,
                kind=record.kind,
                market_slugs=list(record.market_slugs),
                rank_bucket=rank_bucket(record.ranking_score),
                execution_mode=record.execution_mode,
                research_only=record.research_only,
                experiment_id=record.experiment_id,
                experiment_label=record.experiment_label,
                parameter_set_label=record.parameter_set_label,
                shadow_status=UNRESOLVED_MISSING_DATA_STATUS,
                execution_viable=False,
                data_sufficient=False,
                full_size_fillable=False,
                theoretical_gross_edge_cents=record.gross_edge_cents,
                theoretical_net_edge_cents=record.net_edge_cents,
                required_bundle_shares=required_bundle_shares,
                executable_bundle_shares=0.0,
                fillability_ratio=0.0,
                partial_fill_risk_score=record.partial_fill_risk_score,
                non_atomic_fragility_score=record.non_atomic_execution_risk_score,
                unresolved_reason=UNRESOLVED_MISSING_DATA_STATUS,
                ranking_score=record.ranking_score,
                metadata={"legs": [leg for leg in record.legs]},
            )

        snapshot_times = [leg.snapshot_ts for leg in leg_observations if leg.snapshot_ts is not None]
        snapshot_skew_sec = None
        if snapshot_times:
            snapshot_skew_sec = round((max(snapshot_times) - min(snapshot_times)).total_seconds(), 6)

        fillability_ratio = min((leg.fillability_ratio for leg in leg_observations), default=0.0)
        executable_bundle_shares = round((required_bundle_shares or 0.0) * fillability_ratio, 6)
        required_entry_notional_usd = sum(
            (self._entry_price(source_leg) or 0.0) * leg.required_shares
            for source_leg, leg in zip(record.legs, leg_observations, strict=False)
        )
        executable_entry_notional_usd = round(
            sum(
                self._notional_for_shares(
                    (snapshot.get("payload") or {}).get("asks" if str(source_leg.get("action") or "BUY").upper() == "BUY" else "bids") or [],
                    (self._as_float(source_leg.get("required_shares")) or 0.0) * fillability_ratio,
                )
                for source_leg, snapshot, _leg in leg_contexts
            ),
            6,
        )
        estimated_entry_bundle_vwap = (
            round(executable_entry_notional_usd / executable_bundle_shares, 6)
            if executable_bundle_shares and executable_bundle_shares > 1e-9
            else None
        )
        payout_per_bundle_share = (
            (record.expected_payout / required_bundle_shares)
            if record.expected_payout is not None and required_bundle_shares and required_bundle_shares > 1e-9
            else None
        )
        execution_adjusted_gross_edge = (
            round(payout_per_bundle_share - estimated_entry_bundle_vwap, 6)
            if payout_per_bundle_share is not None and estimated_entry_bundle_vwap is not None
            else None
        )
        fee_estimate = record.fee_estimate_cents or 0.0
        latency_penalty = record.latency_penalty_cents or 0.0
        execution_adjusted_net_edge = (
            round(execution_adjusted_gross_edge - fee_estimate - latency_penalty, 6)
            if execution_adjusted_gross_edge is not None
            else None
        )
        theoretical_net_edge = record.net_edge_cents
        execution_gap = (
            round(theoretical_net_edge - execution_adjusted_net_edge, 6)
            if theoretical_net_edge is not None and execution_adjusted_net_edge is not None
            else None
        )
        expected_slippage_cost_usd = round(
            sum(
                self._slippage_cost_for_shares(
                    (snapshot.get("payload") or {}).get("asks" if str(source_leg.get("action") or "BUY").upper() == "BUY" else "bids") or [],
                    (self._as_float(source_leg.get("required_shares")) or 0.0) * fillability_ratio,
                )
                for source_leg, snapshot, _leg in leg_contexts
            ),
            6,
        )
        imbalance = self._imbalance_score(leg_observations)
        partial_fill_risk = round(max(record.partial_fill_risk_score or 0.0, max(0.0, 1.0 - fillability_ratio)), 6)
        skew_penalty = min((snapshot_skew_sec or 0.0) / max(self.max_snapshot_skew_sec, 1e-9), 1.0) if len(leg_observations) > 1 else 0.0
        non_atomic_fragility = round(
            max(record.non_atomic_execution_risk_score or 0.0, partial_fill_risk, imbalance, skew_penalty),
            6,
        )

        full_size_fillable = fillability_ratio >= 1.0 - 1e-9
        data_sufficient = True
        shadow_status = FILLABLE_STATUS if full_size_fillable else PARTIALLY_FILLABLE_STATUS
        execution_viable = True

        if len(leg_observations) > 1 and snapshot_skew_sec is not None and snapshot_skew_sec > self.max_snapshot_skew_sec:
            shadow_status = INSUFFICIENT_SYNCHRONIZED_DEPTH_STATUS
            execution_viable = False
            data_sufficient = False
        elif fillability_ratio <= 1e-9:
            shadow_status = NOT_FILLABLE_STATUS
            execution_viable = False
        elif non_atomic_fragility > self.max_non_atomic_risk:
            shadow_status = NON_ATOMIC_RISK_TOO_HIGH_STATUS
            execution_viable = False
        elif execution_adjusted_net_edge is None or execution_adjusted_net_edge <= 0.0:
            shadow_status = EXECUTION_EDGE_NON_POSITIVE_STATUS
            execution_viable = False

        return ShadowExecutionObservation(
            candidate_id=record.candidate_id,
            source=record.source,
            strategy_family=record.strategy_family,
            strategy_id=record.strategy_id,
            kind=record.kind,
            market_slugs=list(record.market_slugs),
            rank_bucket=rank_bucket(record.ranking_score),
            execution_mode=record.execution_mode,
            research_only=record.research_only,
            experiment_id=record.experiment_id,
            experiment_label=record.experiment_label,
            parameter_set_label=record.parameter_set_label,
            shadow_status=shadow_status,
            execution_viable=execution_viable,
            data_sufficient=data_sufficient,
            full_size_fillable=full_size_fillable,
            theoretical_gross_edge_cents=record.gross_edge_cents,
            theoretical_net_edge_cents=theoretical_net_edge,
            execution_adjusted_gross_edge_cents=execution_adjusted_gross_edge,
            execution_adjusted_net_edge_cents=execution_adjusted_net_edge,
            execution_gap_cents=execution_gap,
            required_bundle_shares=required_bundle_shares,
            executable_bundle_shares=executable_bundle_shares,
            fillability_ratio=round(fillability_ratio, 6),
            required_entry_notional_usd=round(required_entry_notional_usd, 6),
            executable_entry_notional_usd=executable_entry_notional_usd,
            estimated_entry_bundle_vwap=estimated_entry_bundle_vwap,
            expected_slippage_cost_usd=expected_slippage_cost_usd,
            partial_fill_risk_score=partial_fill_risk,
            non_atomic_fragility_score=non_atomic_fragility,
            snapshot_skew_sec=snapshot_skew_sec,
            unresolved_reason=None if data_sufficient else shadow_status,
            ranking_score=record.ranking_score,
            metadata={
                "leg_observations": [leg.model_dump(mode="json") for leg in leg_observations],
                "entry_snapshot_tolerance_sec": self.entry_snapshot_tolerance_sec,
                "max_snapshot_skew_sec": self.max_snapshot_skew_sec,
            },
        )

    def _find_entry_snapshot(self, token_id: str, target_ts: datetime) -> dict[str, Any] | None:
        rows = self.snapshot_rows_by_token.get(token_id, [])
        timestamps = self.snapshot_ts_by_token.get(token_id, [])
        if not rows or not timestamps:
            return None
        idx = bisect_left(timestamps, target_ts)
        candidates: list[tuple[float, int]] = []
        if idx < len(rows):
            delta = abs((timestamps[idx] - target_ts).total_seconds())
            candidates.append((delta, idx))
        if idx > 0:
            delta = abs((timestamps[idx - 1] - target_ts).total_seconds())
            candidates.append((delta, idx - 1))
        if not candidates:
            return None
        delta, chosen_idx = sorted(candidates, key=lambda item: (item[0], 0 if item[1] >= idx else 1))[0]
        if delta > self.entry_snapshot_tolerance_sec:
            return None
        return rows[chosen_idx]

    def _estimate_leg_fill(
        self,
        leg: dict[str, Any],
        snapshot: dict[str, Any],
        required_shares: float,
    ) -> ShadowExecutionLegObservation | None:
        action = str(leg.get("action") or "BUY").upper()
        levels = (snapshot.get("payload") or {}).get("asks" if action == "BUY" else "bids") or []
        if not levels:
            return None

        available_shares = 0.0
        fillable_shares = 0.0
        notional = 0.0
        best_price = self._as_float(levels[0].get("price"))
        if best_price is None or best_price <= 0.0:
            return None
        remaining = required_shares
        for level in levels:
            price = self._as_float(level.get("price"))
            size = self._as_float(level.get("size"))
            if price is None or size is None or price <= 0.0 or size <= 0.0:
                continue
            available_shares += size
            if remaining <= 1e-9:
                continue
            take = min(remaining, size)
            fillable_shares += take
            notional += take * price
            remaining -= take

        estimated_vwap = (notional / fillable_shares) if fillable_shares > 1e-9 else None
        if estimated_vwap is None:
            return None
        slippage_cost = notional - (best_price * fillable_shares)
        return ShadowExecutionLegObservation(
            token_id=str(leg.get("token_id") or "unknown"),
            market_slug=leg.get("market_slug"),
            action=action,
            side=leg.get("side"),
            required_shares=round(required_shares, 6),
            fillable_shares=round(fillable_shares, 6),
            fillability_ratio=round(min(1.0, fillable_shares / max(required_shares, 1e-9)), 6),
            best_price=round(best_price, 6),
            estimated_vwap_price=round(estimated_vwap, 6),
            estimated_notional_usd=round(notional, 6),
            expected_slippage_cost_usd=round(slippage_cost, 6),
            available_shares=round(available_shares, 6),
            snapshot_id=int(snapshot.get("id")) if snapshot.get("id") is not None else None,
            snapshot_ts=snapshot.get("_snapshot_ts"),
        )

    def _entry_price(self, leg: dict[str, Any]) -> float | None:
        return self._as_float(leg.get("vwap_price")) or self._as_float(leg.get("best_price"))

    def _imbalance_score(self, legs: Sequence[ShadowExecutionLegObservation]) -> float:
        ratios = [leg.fillability_ratio for leg in legs]
        if not ratios:
            return 1.0
        return round(max(ratios) - min(ratios), 6)

    def _notional_for_shares(self, levels: list[dict[str, Any]], shares: float) -> float:
        remaining = max(0.0, shares)
        notional = 0.0
        for level in levels:
            price = self._as_float(level.get("price"))
            size = self._as_float(level.get("size"))
            if price is None or size is None or price <= 0.0 or size <= 0.0:
                continue
            take = min(remaining, size)
            notional += take * price
            remaining -= take
            if remaining <= 1e-9:
                break
        return notional

    def _slippage_cost_for_shares(self, levels: list[dict[str, Any]], shares: float) -> float:
        if not levels:
            return 0.0
        best_price = self._as_float(levels[0].get("price")) or 0.0
        if best_price <= 0.0:
            return 0.0
        notional = self._notional_for_shares(levels, shares)
        return max(0.0, notional - (best_price * max(0.0, shares)))

    def _as_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def build_shadow_execution_scorecards(
    records: Sequence[OutcomeEvaluationCandidate],
    observations: Sequence[ShadowExecutionObservation],
    *,
    group_type: str,
    record_group_getter,
    observation_group_getter,
) -> list[ShadowExecutionScorecard]:
    group_candidates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_key = _group_key(record_group_getter(record))
        group_candidates[group_key].add(record.candidate_id)

    cards: list[ShadowExecutionScorecard] = []
    for group_key in sorted(group_candidates):
        candidate_ids = group_candidates[group_key]
        group_observations = [
            observation
            for observation in observations
            if observation.candidate_id in candidate_ids and _group_key(observation_group_getter(observation)) == group_key
        ]
        status_counts = Counter(observation.shadow_status for observation in group_observations)
        sufficient = [observation for observation in group_observations if observation.data_sufficient]
        cards.append(
            ShadowExecutionScorecard(
                group_type=group_type,
                group_key=group_key,
                total_candidates=len(candidate_ids),
                data_sufficient_count=len(sufficient),
                viable_count=sum(1 for observation in group_observations if observation.execution_viable),
                full_size_fillable_count=sum(1 for observation in group_observations if observation.full_size_fillable),
                partially_fillable_count=sum(1 for observation in group_observations if observation.shadow_status == PARTIALLY_FILLABLE_STATUS),
                unresolved_count=sum(1 for observation in group_observations if not observation.data_sufficient),
                fillability_rate=(
                    sum(1 for observation in sufficient if observation.full_size_fillable) / len(sufficient)
                    if sufficient
                    else None
                ),
                viability_rate=(
                    sum(1 for observation in sufficient if observation.execution_viable) / len(sufficient)
                    if sufficient
                    else None
                ),
                average_fillability_ratio=safe_mean([observation.fillability_ratio for observation in sufficient if observation.fillability_ratio is not None]),
                average_execution_adjusted_net_edge_cents=safe_mean([
                    observation.execution_adjusted_net_edge_cents
                    for observation in sufficient
                    if observation.execution_adjusted_net_edge_cents is not None
                ]),
                average_execution_gap_cents=safe_mean([
                    observation.execution_gap_cents
                    for observation in sufficient
                    if observation.execution_gap_cents is not None
                ]),
                average_expected_slippage_cost_usd=safe_mean([
                    observation.expected_slippage_cost_usd
                    for observation in sufficient
                    if observation.expected_slippage_cost_usd is not None
                ]),
                average_non_atomic_fragility_score=safe_mean([
                    observation.non_atomic_fragility_score
                    for observation in sufficient
                    if observation.non_atomic_fragility_score is not None
                ]),
                status_counts=dict(status_counts),
            )
        )
    return cards


def build_live_readiness_scorecards(
    outcome_scorecards: Sequence[Any],
    shadow_scorecards: Sequence[ShadowExecutionScorecard],
    *,
    preferred_horizon_sec: int = 60,
) -> list[LiveReadinessScorecard]:
    outcome_by_family: dict[str, Any] = {}
    candidate_horizons = [preferred_horizon_sec, 30, 300, 900]
    for family in sorted({card.group_key for card in outcome_scorecards if card.group_type == "strategy_family"}):
        chosen = next(
            (
                card
                for horizon in candidate_horizons
                for card in outcome_scorecards
                if card.group_type == "strategy_family" and card.group_key == family and card.horizon_sec == horizon
            ),
            None,
        )
        if chosen is not None:
            outcome_by_family[family] = chosen

    shadow_by_family = {card.group_key: card for card in shadow_scorecards if card.group_type == "strategy_family"}
    families = sorted(set(outcome_by_family) | set(shadow_by_family))
    scorecards: list[LiveReadinessScorecard] = []
    for family in families:
        outcome = outcome_by_family.get(family)
        shadow = shadow_by_family.get(family)
        total_candidates = max(
            int(getattr(outcome, "total_candidates", 0) or 0),
            int(getattr(shadow, "total_candidates", 0) or 0),
        )
        positive_ratio = getattr(outcome, "positive_ratio", None)
        mean_forward_markout = getattr(outcome, "mean_forward_markout_usd", None)
        shadow_sufficiency_rate = (
            (shadow.data_sufficient_count / total_candidates)
            if shadow is not None and total_candidates > 0
            else None
        )
        shadow_viability_rate = (
            (shadow.viable_count / shadow.data_sufficient_count)
            if shadow is not None and shadow.data_sufficient_count > 0
            else None
        )
        shadow_fillability_rate = shadow.fillability_rate if shadow is not None else None

        forward_quality_score = _clamp_ratio(positive_ratio, baseline=0.50, target=0.65)
        fillability_score = shadow_viability_rate
        execution_gap_score = _execution_gap_score(shadow.average_execution_gap_cents if shadow is not None else None)
        data_sufficiency_score = shadow_sufficiency_rate
        overall_score = _weighted_mean(
            [
                (forward_quality_score, 0.35),
                (fillability_score, 0.25),
                (execution_gap_score, 0.20),
                (data_sufficiency_score, 0.20),
            ]
        )
        recommendation_bucket = _recommendation_bucket(
            positive_ratio=positive_ratio,
            viability_rate=shadow_viability_rate,
            data_sufficiency_rate=shadow_sufficiency_rate,
            execution_adjusted_net_edge=shadow.average_execution_adjusted_net_edge_cents if shadow is not None else None,
        )
        blocking_reasons = {}
        if shadow is not None:
            blocking_reasons = {
                reason: count
                for reason, count in shadow.status_counts.items()
                if reason not in {FILLABLE_STATUS, PARTIALLY_FILLABLE_STATUS}
            }

        reference_horizon_sec = int(getattr(outcome, "horizon_sec", preferred_horizon_sec) or preferred_horizon_sec)
        scorecards.append(
            LiveReadinessScorecard(
                strategy_family=family,
                reference_horizon_label=horizon_label(reference_horizon_sec),
                reference_horizon_sec=reference_horizon_sec,
                total_candidates=total_candidates,
                outcome_labeled_count=int(getattr(outcome, "labeled_candidates", 0) or 0),
                positive_outcome_ratio=positive_ratio,
                mean_forward_markout_usd=mean_forward_markout,
                shadow_data_sufficient_count=int(getattr(shadow, "data_sufficient_count", 0) or 0),
                shadow_data_sufficiency_rate=shadow_sufficiency_rate,
                shadow_viable_count=int(getattr(shadow, "viable_count", 0) or 0),
                shadow_viability_rate=shadow_viability_rate,
                shadow_fillability_rate=shadow_fillability_rate,
                mean_execution_adjusted_net_edge_cents=shadow.average_execution_adjusted_net_edge_cents if shadow is not None else None,
                mean_execution_gap_cents=shadow.average_execution_gap_cents if shadow is not None else None,
                forward_quality_score=forward_quality_score,
                fillability_score=fillability_score,
                execution_gap_score=execution_gap_score,
                data_sufficiency_score=data_sufficiency_score,
                overall_score=overall_score,
                recommendation_bucket=recommendation_bucket,
                blocking_reasons=blocking_reasons,
            )
        )
    return scorecards


def _clamp_ratio(value: float | None, *, baseline: float, target: float) -> float | None:
    if value is None:
        return None
    if target <= baseline:
        return 1.0 if value >= target else 0.0
    return max(0.0, min(1.0, (value - baseline) / (target - baseline)))


def _execution_gap_score(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, 1.0 - (max(value, 0.0) / 0.05)))


def _weighted_mean(weighted_values: Sequence[tuple[float | None, float]]) -> float | None:
    usable = [(value, weight) for value, weight in weighted_values if value is not None]
    if not usable:
        return None
    total_weight = sum(weight for _value, weight in usable)
    if total_weight <= 1e-9:
        return None
    return sum(value * weight for value, weight in usable) / total_weight


def _recommendation_bucket(
    *,
    positive_ratio: float | None,
    viability_rate: float | None,
    data_sufficiency_rate: float | None,
    execution_adjusted_net_edge: float | None,
) -> str:
    if (
        positive_ratio is not None
        and viability_rate is not None
        and data_sufficiency_rate is not None
        and execution_adjusted_net_edge is not None
        and positive_ratio >= 0.60
        and viability_rate >= 0.75
        and data_sufficiency_rate >= 0.70
        and execution_adjusted_net_edge > 0.0
    ):
        return "candidate_for_future_tiny_live_preparation"
    if (
        positive_ratio is not None
        and viability_rate is not None
        and data_sufficiency_rate is not None
        and execution_adjusted_net_edge is not None
        and positive_ratio >= 0.55
        and viability_rate >= 0.50
        and data_sufficiency_rate >= 0.50
        and execution_adjusted_net_edge > 0.0
    ):
        return "research_promising"
    return "not_ready"


def _group_key(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    return text if text else "unknown"
