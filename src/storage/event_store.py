from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, select
from sqlalchemy.sql import insert


def _payload(value: Any) -> str:
    if hasattr(value, "model_dump"):
        serializable = value.model_dump(mode="json")
    else:
        serializable = value
    return json.dumps(serializable, ensure_ascii=False)


class ResearchStore:
    def __init__(self, sqlite_url: str):
        self.engine = create_engine(sqlite_url, future=True)
        self.meta = MetaData()

        self.raw_snapshots = Table(
            "raw_snapshots",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("source", String(32), nullable=False),
            Column("entity_id", String(128), nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ingest_ts", DateTime, nullable=False),
        )
        self.opportunity_candidates = Table(
            "opportunity_candidates",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("candidate_id", String(128), nullable=False),
            Column("strategy_id", String(64), nullable=False),
            Column("kind", String(64), nullable=False),
            Column("score", Float, nullable=False),
            Column("net_edge_cents", Float, nullable=False),
            Column("target_notional_usd", Float, nullable=False),
            Column("metadata_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.risk_decisions = Table(
            "risk_decisions",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("candidate_id", String(128), nullable=False),
            Column("status", String(32), nullable=False),
            Column("approved_notional_usd", Float, nullable=False),
            Column("reason_codes_json", Text, nullable=False),
            Column("metadata_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.order_intents = Table(
            "order_intents",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("intent_id", String(128), nullable=False),
            Column("candidate_id", String(128), nullable=False),
            Column("mode", String(16), nullable=False),
            Column("market_slug", String(256), nullable=False),
            Column("token_id", String(128), nullable=False),
            Column("side", String(16), nullable=False),
            Column("order_type", String(16), nullable=False),
            Column("size", Float, nullable=False),
            Column("limit_price", Float),
            Column("max_notional_usd", Float, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.execution_reports = Table(
            "execution_reports",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("intent_id", String(128), nullable=False),
            Column("status", String(32), nullable=False),
            Column("filled_size", Float, nullable=False),
            Column("avg_fill_price", Float),
            Column("fee_paid_usd", Float, nullable=False),
            Column("latency_ms", Integer, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.account_snapshots = Table(
            "account_snapshots",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("cash", Float, nullable=False),
            Column("frozen_cash", Float, nullable=False),
            Column("realized_pnl", Float, nullable=False),
            Column("unrealized_pnl", Float, nullable=False),
            Column("daily_pnl", Float, nullable=False),
            Column("open_positions", Integer, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.position_events = Table(
            "position_events",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("position_id", String(128), nullable=False),
            Column("candidate_id", String(128)),
            Column("event_type", String(64), nullable=False),
            Column("symbol", String(128), nullable=False),
            Column("market_slug", String(256), nullable=False),
            Column("state", String(32), nullable=False),
            Column("reason_code", String(64)),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.position_marks = Table(
            "position_marks",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("position_id", String(128), nullable=False),
            Column("candidate_id", String(128)),
            Column("symbol", String(128), nullable=False),
            Column("market_slug", String(256), nullable=False),
            Column("state", String(32), nullable=False),
            Column("mark_price", Float, nullable=False),
            Column("marked_value_usd", Float, nullable=False),
            Column("unrealized_pnl_usd", Float, nullable=False),
            Column("age_sec", Float, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.trade_summaries = Table(
            "trade_summaries",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("position_id", String(128), nullable=False),
            Column("candidate_id", String(128)),
            Column("symbol", String(128), nullable=False),
            Column("market_slug", String(256), nullable=False),
            Column("state", String(32), nullable=False),
            Column("entry_cost_usd", Float, nullable=False),
            Column("exit_proceeds_usd", Float, nullable=False),
            Column("fees_paid_usd", Float, nullable=False),
            Column("realized_pnl_usd", Float, nullable=False),
            Column("holding_duration_sec", Float, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("opened_ts", DateTime, nullable=False),
            Column("closed_ts", DateTime, nullable=False),
        )
        self.rejection_events = Table(
            "rejection_events",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("run_id", String(128), nullable=False),
            Column("candidate_id", String(128)),
            Column("stage", String(64), nullable=False),
            Column("reason_code", String(64), nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )
        self.run_summaries = Table(
            "run_summaries",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("run_id", String(128), nullable=False),
            Column("started_ts", DateTime, nullable=False),
            Column("ended_ts", DateTime, nullable=False),
            Column("markets_scanned", Integer, nullable=False),
            Column("snapshots_stored", Integer, nullable=False),
            Column("candidates_generated", Integer, nullable=False),
            Column("risk_accepted", Integer, nullable=False),
            Column("risk_rejected", Integer, nullable=False),
            Column("near_miss_candidates", Integer, nullable=False),
            Column("paper_orders_created", Integer, nullable=False),
            Column("fills", Integer, nullable=False),
            Column("partial_fills", Integer, nullable=False),
            Column("cancellations", Integer, nullable=False),
            Column("open_positions", Integer, nullable=False),
            Column("closed_positions", Integer, nullable=False),
            Column("realized_pnl", Float, nullable=False),
            Column("unrealized_pnl", Float, nullable=False),
            Column("system_errors", Integer, nullable=False),
            Column("payload_json", Text, nullable=False),
        )
        self.system_events = Table(
            "system_events",
            self.meta,
            Column("id", Integer, primary_key=True),
            Column("event_type", String(64), nullable=False),
            Column("severity", String(16), nullable=False),
            Column("entity_id", String(128)),
            Column("message", Text, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("ts", DateTime, nullable=False),
        )

        self.meta.create_all(self.engine)

    def save_raw_snapshot(self, source: str, entity_id: str, payload: Any, ingest_ts: datetime) -> None:
        self._execute(
            self.raw_snapshots,
            source=source,
            entity_id=entity_id,
            payload_json=_payload(payload),
            ingest_ts=ingest_ts,
        )

    def save_candidate(self, candidate: Any) -> None:
        self._execute(
            self.opportunity_candidates,
            candidate_id=candidate.candidate_id,
            strategy_id=candidate.strategy_id,
            kind=candidate.kind,
            score=candidate.score,
            net_edge_cents=candidate.net_edge_cents,
            target_notional_usd=candidate.target_notional_usd,
            metadata_json=_payload(candidate.model_dump(mode="json")),
            ts=candidate.ts,
        )

    def save_risk_decision(self, decision: Any) -> None:
        self._execute(
            self.risk_decisions,
            candidate_id=decision.candidate_id,
            status=decision.status.value if hasattr(decision.status, "value") else str(decision.status),
            approved_notional_usd=decision.approved_notional_usd,
            reason_codes_json=_payload(decision.reason_codes),
            metadata_json=_payload(decision.metadata),
            ts=decision.ts,
        )

    def save_order_intent(self, intent: Any) -> None:
        self._execute(
            self.order_intents,
            intent_id=intent.intent_id,
            candidate_id=intent.candidate_id,
            mode=intent.mode.value if hasattr(intent.mode, "value") else str(intent.mode),
            market_slug=intent.market_slug,
            token_id=intent.token_id,
            side=intent.side,
            order_type=intent.order_type.value if hasattr(intent.order_type, "value") else str(intent.order_type),
            size=intent.size,
            limit_price=intent.limit_price,
            max_notional_usd=intent.max_notional_usd,
            payload_json=_payload(intent.model_dump(mode="json")),
            ts=intent.ts,
        )

    def save_execution_report(self, report: Any) -> None:
        self._execute(
            self.execution_reports,
            intent_id=report.intent_id,
            status=report.status.value if hasattr(report.status, "value") else str(report.status),
            filled_size=report.filled_size,
            avg_fill_price=report.avg_fill_price,
            fee_paid_usd=report.fee_paid_usd,
            latency_ms=report.latency_ms,
            payload_json=_payload(report.model_dump(mode="json")),
            ts=report.ts,
        )

    def save_account_snapshot(self, snapshot: Any) -> None:
        self._execute(
            self.account_snapshots,
            cash=snapshot.cash,
            frozen_cash=snapshot.frozen_cash,
            realized_pnl=snapshot.realized_pnl,
            unrealized_pnl=snapshot.unrealized_pnl,
            daily_pnl=snapshot.daily_pnl,
            open_positions=snapshot.open_positions,
            payload_json=_payload(snapshot.model_dump(mode="json")),
            ts=snapshot.ts,
        )

    def save_position_event(
        self,
        position_id: str,
        candidate_id: str | None,
        event_type: str,
        symbol: str,
        market_slug: str,
        state: str,
        reason_code: str | None,
        payload: Any,
        ts: datetime,
    ) -> None:
        self._execute(
            self.position_events,
            position_id=position_id,
            candidate_id=candidate_id,
            event_type=event_type,
            symbol=symbol,
            market_slug=market_slug,
            state=state,
            reason_code=reason_code,
            payload_json=_payload(payload),
            ts=ts,
        )

    def save_position_mark(self, mark: Any) -> None:
        self._execute(
            self.position_marks,
            position_id=mark.position_id,
            candidate_id=mark.candidate_id,
            symbol=mark.symbol,
            market_slug=mark.market_slug,
            state=mark.state.value if hasattr(mark.state, "value") else str(mark.state),
            mark_price=mark.mark_price,
            marked_value_usd=mark.marked_value_usd,
            unrealized_pnl_usd=mark.unrealized_pnl_usd,
            age_sec=mark.age_sec,
            payload_json=_payload(mark.model_dump(mode="json")),
            ts=mark.ts,
        )

    def save_trade_summary(self, trade_summary: Any) -> None:
        self._execute(
            self.trade_summaries,
            position_id=trade_summary.position_id,
            candidate_id=trade_summary.candidate_id,
            symbol=trade_summary.symbol,
            market_slug=trade_summary.market_slug,
            state=trade_summary.state.value if hasattr(trade_summary.state, "value") else str(trade_summary.state),
            entry_cost_usd=trade_summary.entry_cost_usd,
            exit_proceeds_usd=trade_summary.exit_proceeds_usd,
            fees_paid_usd=trade_summary.fees_paid_usd,
            realized_pnl_usd=trade_summary.realized_pnl_usd,
            holding_duration_sec=trade_summary.holding_duration_sec,
            payload_json=_payload(trade_summary.model_dump(mode="json")),
            opened_ts=trade_summary.opened_ts,
            closed_ts=trade_summary.closed_ts,
        )

    def save_rejection_event(self, rejection: Any) -> None:
        self._execute(
            self.rejection_events,
            run_id=rejection.run_id,
            candidate_id=rejection.candidate_id,
            stage=rejection.stage,
            reason_code=rejection.reason_code,
            payload_json=_payload(rejection.model_dump(mode="json")),
            ts=rejection.ts,
        )

    def save_run_summary(self, summary: Any) -> None:
        self._execute(
            self.run_summaries,
            run_id=summary.run_id,
            started_ts=summary.started_ts,
            ended_ts=summary.ended_ts,
            markets_scanned=summary.markets_scanned,
            snapshots_stored=summary.snapshots_stored,
            candidates_generated=summary.candidates_generated,
            risk_accepted=summary.risk_accepted,
            risk_rejected=summary.risk_rejected,
            near_miss_candidates=summary.near_miss_candidates,
            paper_orders_created=summary.paper_orders_created,
            fills=summary.fills,
            partial_fills=summary.partial_fills,
            cancellations=summary.cancellations,
            open_positions=summary.open_positions,
            closed_positions=summary.closed_positions,
            realized_pnl=summary.realized_pnl,
            unrealized_pnl=summary.unrealized_pnl,
            system_errors=summary.system_errors,
            payload_json=_payload(summary.model_dump(mode="json")),
        )

    def save_system_event(self, event: Any) -> None:
        self._execute(
            self.system_events,
            event_type=event.event_type,
            severity=event.severity.value if hasattr(event.severity, "value") else str(event.severity),
            entity_id=event.entity_id,
            message=event.message,
            payload_json=_payload(event.payload),
            ts=event.ts,
        )

    def load_run_summaries(self) -> list[dict[str, Any]]:
        with self.engine.begin() as connection:
            rows = connection.execute(select(self.run_summaries.c.payload_json)).all()
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        self.engine.dispose()

    def _execute(self, table: Table, **values: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(insert(table).values(**values))
