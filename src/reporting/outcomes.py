from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable, Sequence

from src.reporting.models import (
    CandidateOutcomeHorizonStat,
    CandidateOutcomeObservation,
    CandidateOutcomeScorecard,
)


NEAR_MISS_REASON_CODES = {
    "EDGE_BELOW_THRESHOLD",
    "NET_PROFIT_BELOW_THRESHOLD",
    "INSUFFICIENT_DEPTH",
    "SPREAD_TOO_WIDE",
    "PARTIAL_FILL_RISK_TOO_HIGH",
    "NON_ATOMIC_RISK_TOO_HIGH",
    "ORDER_SIZE_LIMIT",
}


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def safe_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def strategy_family_from_payload(payload: dict[str, Any], strategy_id: str | None = None, kind: str | None = None) -> str:
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


def horizon_label(horizon_sec: int) -> str:
    if horizon_sec < 60:
        return f"{horizon_sec}s"
    if horizon_sec % 60 == 0:
        return f"{horizon_sec // 60}m"
    return f"{horizon_sec}s"


def rank_bucket(ranking_score: float | None) -> str:
    if ranking_score is None:
        return "unranked"
    if ranking_score >= 90.0:
        return "score_90_plus"
    if ranking_score >= 80.0:
        return "score_80_89"
    if ranking_score >= 70.0:
        return "score_70_79"
    return "score_below_70"


@dataclass(frozen=True)
class OutcomeEvaluationCandidate:
    candidate_id: str
    source: str
    strategy_family: str
    strategy_id: str
    kind: str
    market_slugs: list[str]
    run_id: str | None
    legs: list[dict[str, Any]]
    ts: datetime
    ranking_score: float | None
    gross_edge_cents: float | None
    net_edge_cents: float | None
    expected_payout: float | None
    fee_estimate_cents: float | None
    latency_penalty_cents: float | None
    partial_fill_risk_score: float | None
    non_atomic_execution_risk_score: float | None
    execution_mode: str
    research_only: bool
    experiment_id: str | None
    experiment_label: str | None
    parameter_set_label: str | None
    metadata: dict[str, Any]


class CandidateOutcomeLabeler:
    def __init__(self, raw_snapshot_rows: Iterable[dict[str, Any]], horizons: Iterable[int]):
        self.horizons = tuple(sorted({int(horizon) for horizon in horizons if int(horizon) > 0}))
        self.snapshot_rows_by_token: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.snapshot_ts_by_token: dict[str, list[datetime]] = {}
        self._build_snapshot_index(raw_snapshot_rows)

    def label_candidates(self, records: Iterable[OutcomeEvaluationCandidate]) -> list[CandidateOutcomeObservation]:
        observations: list[CandidateOutcomeObservation] = []
        for record in records:
            if not record.legs:
                continue
            for horizon in self.horizons:
                observation = self._label_candidate_at_horizon(record, horizon)
                if observation is not None:
                    observations.append(observation)
        return sorted(
            observations,
            key=lambda item: (item.horizon_sec, item.selected_mark_ts, item.candidate_id),
        )

    def _build_snapshot_index(self, raw_snapshot_rows: Iterable[dict[str, Any]]) -> None:
        for row in raw_snapshot_rows:
            if str(row.get("source", "")).lower() != "clob":
                continue
            payload = row.get("payload") or {}
            token_id = str(row.get("entity_id") or payload.get("token_id") or "")
            if not token_id:
                continue
            row_copy = dict(row)
            row_copy["payload"] = payload
            row_copy["_snapshot_ts"] = parse_dt(row.get("ingest_ts") or row.get("ts"))
            self.snapshot_rows_by_token[token_id].append(row_copy)

        for token_id, rows in self.snapshot_rows_by_token.items():
            rows.sort(key=lambda item: item["_snapshot_ts"])
            self.snapshot_ts_by_token[token_id] = [item["_snapshot_ts"] for item in rows]

    def _label_candidate_at_horizon(
        self,
        record: OutcomeEvaluationCandidate,
        horizon_sec: int,
    ) -> CandidateOutcomeObservation | None:
        target_ts = record.ts.timestamp() + horizon_sec
        total_markout = 0.0
        total_entry_notional = 0.0
        selected_ts: list[datetime] = []
        leg_marks: list[dict[str, Any]] = []

        for leg in record.legs:
            token_id = str(leg.get("token_id") or "")
            shares = self._as_float(leg.get("required_shares"))
            entry_price = self._entry_price(leg)
            action = str(leg.get("action") or "BUY").upper()
            if not token_id or shares is None or shares <= 1e-9 or entry_price is None:
                return None

            snapshot = self._find_snapshot(token_id, target_ts)
            if snapshot is None:
                return None

            liquidation_price = self._liquidation_price(snapshot.get("payload") or {}, action)
            if liquidation_price is None:
                return None

            direction = 1.0 if action == "BUY" else -1.0
            leg_markout = direction * shares * (liquidation_price - entry_price)
            total_markout += leg_markout
            total_entry_notional += abs(shares * entry_price)
            snapshot_ts = snapshot["_snapshot_ts"]
            selected_ts.append(snapshot_ts)
            leg_marks.append(
                {
                    "token_id": token_id,
                    "market_slug": leg.get("market_slug"),
                    "action": action,
                    "side": leg.get("side"),
                    "shares": shares,
                    "entry_price": entry_price,
                    "liquidation_price": liquidation_price,
                    "snapshot_id": snapshot.get("id"),
                    "snapshot_ts": snapshot_ts.isoformat(),
                    "age_sec": round((snapshot_ts - record.ts).total_seconds(), 6),
                    "leg_markout_usd": round(leg_markout, 6),
                }
            )

        if not selected_ts:
            return None

        max_ts = max(selected_ts)
        max_age_sec = round((max_ts - record.ts).total_seconds(), 6)
        return_pct = (total_markout / total_entry_notional * 100.0) if total_entry_notional > 1e-9 else None
        return CandidateOutcomeObservation(
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
            horizon_label=horizon_label(horizon_sec),
            horizon_sec=horizon_sec,
            selected_mark_age_sec=max_age_sec,
            selected_mark_ts=max_ts,
            forward_markout_usd=round(total_markout, 6),
            return_pct=round(return_pct, 6) if return_pct is not None else None,
            positive_outcome=total_markout > 0.0,
            ranking_score=record.ranking_score,
            gross_edge_cents=record.gross_edge_cents,
            net_edge_cents=record.net_edge_cents,
            metadata={
                "selection_method": "first_raw_snapshot_at_or_after_horizon_per_leg",
                "entry_notional_usd": round(total_entry_notional, 6),
                "leg_marks": leg_marks,
                "candidate_metadata": record.metadata,
            },
        )

    def _find_snapshot(self, token_id: str, target_ts: float) -> dict[str, Any] | None:
        rows = self.snapshot_rows_by_token.get(token_id, [])
        timestamps = self.snapshot_ts_by_token.get(token_id, [])
        if not rows or not timestamps:
            return None
        target_dt = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        idx = bisect_left(timestamps, target_dt)
        if idx >= len(rows):
            return None
        return rows[idx]

    def _entry_price(self, leg: dict[str, Any]) -> float | None:
        return self._as_float(leg.get("vwap_price")) or self._as_float(leg.get("best_price"))

    def _liquidation_price(self, payload: dict[str, Any], action: str) -> float | None:
        levels = payload.get("bids") if action == "BUY" else payload.get("asks")
        if not levels:
            return None
        return self._as_float(levels[0].get("price"))

    def _as_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def build_outcome_records_from_persisted(
    candidate_rows: Iterable[dict[str, Any]],
    rejection_rows: Iterable[dict[str, Any]],
) -> list[OutcomeEvaluationCandidate]:
    records: list[OutcomeEvaluationCandidate] = []

    for row in candidate_rows:
        payload = row.get("payload") or {}
        metadata = payload.get("metadata", {})
        records.append(
            OutcomeEvaluationCandidate(
                candidate_id=str(row.get("candidate_id") or payload.get("candidate_id") or "unknown"),
                source="qualified",
                strategy_family=str(strategy_family_from_payload(payload, row.get("strategy_id"), row.get("kind"))),
                strategy_id=str(row.get("strategy_id") or payload.get("strategy_id") or "unknown"),
                kind=str(row.get("kind") or payload.get("kind") or "unknown"),
                market_slugs=list(payload.get("market_slugs") or []),
                run_id=_string_or_none(metadata.get("run_id")),
                legs=list(payload.get("legs") or metadata.get("qualification", {}).get("legs") or []),
                ts=parse_dt(row.get("ts") or payload.get("ts")),
                ranking_score=_as_float(payload.get("ranking_score", payload.get("score"))),
                gross_edge_cents=_as_float(payload.get("gross_edge_cents")),
                net_edge_cents=_net_edge_cents(payload, metadata.get("qualification", {})),
                expected_payout=_as_float(payload.get("expected_payout")),
                fee_estimate_cents=_as_float(payload.get("fee_estimate_cents")),
                latency_penalty_cents=_as_float(payload.get("latency_penalty_cents")),
                partial_fill_risk_score=_as_float(payload.get("partial_fill_risk_score", metadata.get("qualification", {}).get("partial_fill_risk_score"))),
                non_atomic_execution_risk_score=_as_float(payload.get("non_atomic_execution_risk_score", metadata.get("qualification", {}).get("non_atomic_execution_risk_score"))),
                execution_mode=str(payload.get("execution_mode", "paper_eligible")),
                research_only=bool(payload.get("research_only", False)),
                experiment_id=_string_or_none(metadata.get("experiment_id")),
                experiment_label=_string_or_none(metadata.get("experiment_label")),
                parameter_set_label=_string_or_none(metadata.get("parameter_set_label")),
                metadata=metadata,
            )
        )

    grouped_rejections: dict[tuple[str | None, str], dict[str, Any]] = {}
    for row in rejection_rows:
        payload = row.get("payload") or {}
        if payload.get("stage") != "qualification":
            continue
        candidate_id = payload.get("candidate_id")
        if not candidate_id:
            continue
        key = (payload.get("run_id"), str(candidate_id))
        entry = grouped_rejections.setdefault(
            key,
            {
                "payload": payload,
                "reason_codes": set(),
            },
        )
        reason_code = payload.get("reason_code")
        if reason_code:
            entry["reason_codes"].add(str(reason_code))

    for entry in grouped_rejections.values():
        payload = entry["payload"]
        reason_codes = sorted(entry["reason_codes"])
        if len(reason_codes) != 1 or reason_codes[0] not in NEAR_MISS_REASON_CODES:
            continue
        metadata = payload.get("metadata", {})
        raw_candidate = metadata.get("raw_candidate", {})
        qualification = metadata.get("qualification", {})
        records.append(
            OutcomeEvaluationCandidate(
                candidate_id=str(payload.get("candidate_id") or "unknown"),
                source="near_miss",
                strategy_family=str(metadata.get("strategy_family") or strategy_family_from_payload(raw_candidate)),
                strategy_id=str(raw_candidate.get("strategy_id") or "unknown"),
                kind=str(raw_candidate.get("kind") or "unknown"),
                market_slugs=list(raw_candidate.get("market_slugs") or metadata.get("market_slugs") or []),
                run_id=_string_or_none(metadata.get("run_id") or payload.get("run_id")),
                legs=list(qualification.get("legs") or raw_candidate.get("legs") or []),
                ts=parse_dt(payload.get("ts")),
                ranking_score=None,
                gross_edge_cents=_as_float(raw_candidate.get("gross_edge_cents")),
                net_edge_cents=_as_float(qualification.get("expected_net_edge_cents")),
                expected_payout=_as_float(raw_candidate.get("expected_payout")),
                fee_estimate_cents=_as_float(raw_candidate.get("fee_estimate_cents")),
                latency_penalty_cents=_as_float(raw_candidate.get("latency_penalty_cents")),
                partial_fill_risk_score=_as_float(qualification.get("partial_fill_risk_score")),
                non_atomic_execution_risk_score=_as_float(qualification.get("non_atomic_execution_risk_score")),
                execution_mode=str(metadata.get("execution_mode", raw_candidate.get("execution_mode", "paper_eligible"))),
                research_only=bool(metadata.get("research_only", raw_candidate.get("research_only", False))),
                experiment_id=_string_or_none(metadata.get("experiment_id")),
                experiment_label=_string_or_none(metadata.get("experiment_label")),
                parameter_set_label=_string_or_none(metadata.get("parameter_set_label")),
                metadata={
                    **metadata,
                    "reason_codes": reason_codes,
                },
            )
        )

    deduped: dict[tuple[str, str], OutcomeEvaluationCandidate] = {}
    for record in records:
        deduped[(record.source, record.candidate_id)] = record
    return sorted(deduped.values(), key=lambda item: (item.ts, item.source, item.candidate_id))


def build_outcome_records_from_corpus(records: Iterable[Any]) -> list[OutcomeEvaluationCandidate]:
    converted: list[OutcomeEvaluationCandidate] = []
    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        qualification = metadata.get("qualification", {})
        raw_candidate = metadata.get("raw_candidate", {})
        converted.append(
            OutcomeEvaluationCandidate(
                candidate_id=str(getattr(record, "candidate_id", "unknown")),
                source=str(getattr(record, "original_status", "qualified")),
                strategy_family=str(getattr(record, "strategy_family", "unknown")),
                strategy_id=str(getattr(record, "strategy_id", "unknown")),
                kind=str(getattr(record, "kind", "unknown")),
                market_slugs=list(getattr(record, "market_slugs", []) or []),
                run_id=_string_or_none(getattr(record, "run_id", None)),
                legs=list(qualification.get("legs") or raw_candidate.get("legs") or getattr(record, "legs", []) or []),
                ts=getattr(record, "ts"),
                ranking_score=_as_float(getattr(record, "ranking_score", None)),
                gross_edge_cents=_as_float(getattr(record, "gross_edge_cents", None)),
                net_edge_cents=_as_float(getattr(record, "net_edge_cents", None)),
                expected_payout=_as_float(getattr(record, "expected_payout", None)),
                fee_estimate_cents=_as_float(getattr(record, "fee_estimate_cents", None)),
                latency_penalty_cents=_as_float(getattr(record, "latency_penalty_cents", None)),
                partial_fill_risk_score=_as_float(getattr(record, "partial_fill_risk_score", None)),
                non_atomic_execution_risk_score=_as_float(getattr(record, "non_atomic_execution_risk_score", None)),
                execution_mode=str(getattr(record, "execution_mode", "paper_eligible")),
                research_only=bool(getattr(record, "research_only", False)),
                experiment_id=_string_or_none(getattr(record, "experiment_id", None)),
                experiment_label=_string_or_none(getattr(record, "experiment_label", None)),
                parameter_set_label=_string_or_none(getattr(record, "parameter_set_label", None)),
                metadata=metadata,
            )
        )
    return converted


def build_candidate_outcome_horizon_stats(
    records: Sequence[OutcomeEvaluationCandidate],
    observations: Sequence[CandidateOutcomeObservation],
    horizons: Iterable[int],
) -> list[CandidateOutcomeHorizonStat]:
    total_candidates = len(records)
    stats: list[CandidateOutcomeHorizonStat] = []
    for horizon in horizons:
        horizon_obs = [obs for obs in observations if obs.horizon_sec == horizon]
        pnl_values = [obs.forward_markout_usd for obs in horizon_obs]
        return_values = [obs.return_pct for obs in horizon_obs if obs.return_pct is not None]
        stats.append(
            CandidateOutcomeHorizonStat(
                horizon_label=horizon_label(horizon),
                horizon_sec=horizon,
                total_candidates=total_candidates,
                labeled_candidates=len({obs.candidate_id for obs in horizon_obs}),
                positive_ratio=(sum(1 for value in pnl_values if value > 0.0) / len(pnl_values)) if pnl_values else None,
                mean_forward_markout_usd=safe_mean(pnl_values),
                median_forward_markout_usd=safe_median(pnl_values),
                min_forward_markout_usd=min(pnl_values) if pnl_values else None,
                max_forward_markout_usd=max(pnl_values) if pnl_values else None,
                mean_return_pct=safe_mean(return_values),
            )
        )
    return stats


def build_outcome_scorecards(
    records: Sequence[OutcomeEvaluationCandidate],
    observations: Sequence[CandidateOutcomeObservation],
    horizons: Iterable[int],
    group_type: str,
    record_group_getter,
    observation_group_getter,
) -> list[CandidateOutcomeScorecard]:
    group_candidates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_key = _group_key(record_group_getter(record))
        group_candidates[group_key].add(record.candidate_id)

    cards: list[CandidateOutcomeScorecard] = []
    for group_key in sorted(group_candidates):
        candidate_ids = group_candidates[group_key]
        for horizon in horizons:
            horizon_obs = [
                obs
                for obs in observations
                if obs.horizon_sec == horizon and obs.candidate_id in candidate_ids and _group_key(observation_group_getter(obs)) == group_key
            ]
            pnl_values = [obs.forward_markout_usd for obs in horizon_obs]
            return_values = [obs.return_pct for obs in horizon_obs if obs.return_pct is not None]
            cards.append(
                CandidateOutcomeScorecard(
                    group_type=group_type,
                    group_key=group_key,
                    horizon_label=horizon_label(horizon),
                    horizon_sec=horizon,
                    total_candidates=len(candidate_ids),
                    labeled_candidates=len({obs.candidate_id for obs in horizon_obs}),
                    positive_ratio=(sum(1 for value in pnl_values if value > 0.0) / len(pnl_values)) if pnl_values else None,
                    mean_forward_markout_usd=safe_mean(pnl_values),
                    median_forward_markout_usd=safe_median(pnl_values),
                    min_forward_markout_usd=min(pnl_values) if pnl_values else None,
                    max_forward_markout_usd=max(pnl_values) if pnl_values else None,
                    mean_return_pct=safe_mean(return_values),
                )
            )
    return cards


def _net_edge_cents(payload: dict[str, Any], qualification: dict[str, Any]) -> float | None:
    explicit = _as_float(qualification.get("expected_net_edge_cents"))
    if explicit is not None:
        return explicit
    gross = _as_float(payload.get("gross_edge_cents"))
    fee = _as_float(payload.get("fee_estimate_cents")) or 0.0
    slippage = _as_float(payload.get("slippage_estimate_cents")) or 0.0
    latency = _as_float(payload.get("latency_penalty_cents")) or 0.0
    if gross is None:
        return None
    return gross - fee - slippage - latency


def _group_key(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    return text if text else "unknown"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
