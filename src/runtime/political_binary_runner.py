from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.config_runtime.loader import load_runtime_config
from src.constraints.political_rules import build_constraint_scan_plan
from src.core.normalize import build_yes_no_pairs
from src.core.orderbook_validation import validate_orderbook
from src.domain.models import OrderStatus, RejectionReason, Severity
from src.ingest.gamma import fetch_markets
from src.paper.broker import PaperBroker
from src.paper.ledger import Ledger
from src.reporting.summary import RunSummaryBuilder
from src.runtime.runner import ResearchRunner, _read_yaml, _token_map_from_pairs
from src.storage.event_store import ResearchStore
from src.strategies.opportunity_strategies import PoliticalBinaryConstraintPaperStrategy
from src.utils.db import OpportunityStore


class PoliticalBinaryPaperRunner(ResearchRunner):
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        constraints_path: str = "config/political_constraint_rules.yaml",
        debug_output_dir=None,
    ):
        super().__init__(
            settings_path=settings_path,
            constraints_path=constraints_path,
            debug_output_dir=debug_output_dir,
        )
        self._base_config = load_runtime_config(settings_path)
        self.config = self._base_config.model_copy(deep=True)
        self.store = ResearchStore(self.config.storage.sqlite_url)
        self.opportunity_store = OpportunityStore(self.config.storage.sqlite_url)
        self.paper_ledger = Ledger(cash=self.config.paper.starting_cash)
        self.paper_broker = PaperBroker(self.paper_ledger)
        self.political_binary_strategy = PoliticalBinaryConstraintPaperStrategy()
        self._rebuild_runtime_components()

    def run_once(self, experiment_context: dict[str, object] | None = None):
        if self.config.execution.mode != "paper" or self.config.execution.live_enabled:
            raise ValueError("PoliticalBinaryPaperRunner only supports execution.mode=paper with live disabled.")

        cycle_started = datetime.now(timezone.utc)
        self._run_sequence += 1
        self._current_run_id = str(uuid4())
        self._current_summary = RunSummaryBuilder(run_id=self._current_run_id, started_ts=cycle_started)
        self._invalid_orderbook_export_path = self._prepare_invalid_orderbook_export_path(cycle_started)
        if experiment_context is not None:
            self.set_experiment_context(**experiment_context)
        if self._current_summary is not None:
            self._current_summary.metadata.update(self._build_experiment_metadata())
            self._current_summary.metadata["invalid_orderbooks_export_path"] = str(self._invalid_orderbook_export_path)

        constraint_plan = build_constraint_scan_plan(_read_yaml(self.constraints_path))
        approved_market_slugs = {
            str(slug)
            for slug in constraint_plan.get("approved_market_slugs", [])
            if str(slug).strip()
        }
        self._current_summary.metadata.update(
            {
                "constraint_model_id": constraint_plan.get("constraint_model_id"),
                "constraint_domain": constraint_plan.get("constraint_domain"),
                "approved_rule_count": constraint_plan.get("approved_rule_count"),
                "rule_count": constraint_plan.get("rule_count"),
                "approved_universe_size": len(approved_market_slugs),
                "execution_mode": self.config.execution.mode,
            }
        )

        book_cache: dict[str, object] = {}

        try:
            markets = fetch_markets(self.config.market_data.gamma_host, self.config.market_data.market_limit)
            self._save_raw_snapshot("gamma", "markets", markets, cycle_started)
            pairs = build_yes_no_pairs(markets)
            if approved_market_slugs:
                pairs = [pair for pair in pairs if pair.market_slug in approved_market_slugs]
            token_map = _token_map_from_pairs(pairs)
            self._current_summary.markets_scanned = len(pairs)
            self.logger.info(
                "loaded political approved market pairs",
                extra={
                    "payload": {
                        "pairs": len(pairs),
                        "approved_universe_size": len(approved_market_slugs),
                        "run_id": self._current_run_id,
                    }
                },
            )
        except Exception as exc:
            self._record_event(
                "market_fetch_failed",
                Severity.ERROR,
                "Failed to fetch or normalize political approved markets",
                {"error": str(exc)},
            )
            raise

        self._run_political_constraint_scan(constraint_plan, token_map, cycle_started, book_cache)
        self._manage_open_positions(
            cycle_started,
            book_cache,
            force_reason="RUN_END_FLATTEN" if self.config.paper.flatten_on_run_end else None,
        )

        final_snapshot = self.paper_ledger.snapshot()
        self.store.save_account_snapshot(final_snapshot)
        self._current_summary.open_positions = final_snapshot.open_positions
        self._current_summary.realized_pnl = final_snapshot.realized_pnl
        self._current_summary.unrealized_pnl = final_snapshot.unrealized_pnl

        ended_ts = datetime.now(timezone.utc)
        summary = self._current_summary.build(ended_ts=ended_ts)
        self.store.save_run_summary(summary)
        if self.config.monitoring.emit_console_summary:
            self.logger.info("run summary", extra={"payload": summary.model_dump(mode="json")})
        self._current_summary = None
        self._current_run_id = None
        self._invalid_orderbook_export_path = None
        return summary

    def _active_political_strategies(self) -> list[PoliticalBinaryConstraintPaperStrategy]:
        targets = self._target_strategy_families()
        if targets is None:
            return [self.political_binary_strategy]
        if self.political_binary_strategy.strategy_family.value in targets:
            return [self.political_binary_strategy]
        return []

    def _run_political_constraint_scan(
        self,
        constraints: dict,
        token_map: dict[tuple[str, str], str],
        cycle_started: datetime,
        book_cache: dict[str, object],
    ) -> None:
        active_strategies = self._active_political_strategies()
        if not active_strategies:
            return

        for rule in constraints.get("cross_market", []):
            if rule.get("relation") != "political_binary":
                continue

            lhs_slug = rule["lhs"]["market_slug"]
            rhs_slug = rule["rhs"]["market_slug"]
            for strategy in active_strategies:
                self._record_strategy_family_market_considered(strategy.strategy_family.value, lhs_slug)
                self._record_strategy_family_market_considered(strategy.strategy_family.value, rhs_slug)

            lhs_side = rule["lhs"].get("side", "YES").upper()
            rhs_side = rule["rhs"].get("side", "YES").upper()
            lhs_exec_side = rule.get("lhs_execution", {}).get("side", "NO").upper()
            rhs_exec_side = rule.get("rhs_execution", {}).get("side", "YES").upper()
            lhs_token = token_map.get((lhs_slug, lhs_side))
            rhs_token = token_map.get((rhs_slug, rhs_side))
            lhs_exec_token = token_map.get((lhs_slug, lhs_exec_side))
            rhs_exec_token = token_map.get((rhs_slug, rhs_exec_side))
            if not lhs_token or not rhs_token or not lhs_exec_token or not rhs_exec_token:
                for strategy in active_strategies:
                    self._record_rejection(
                        stage="candidate_filter",
                        reason_code="MISSING_CONSTRAINT_TOKEN",
                        candidate_id=None,
                        metadata={
                            "strategy_family": strategy.strategy_family.value,
                            "failure_stage": "pre_candidate_relation",
                            "constraint_name": rule.get("name"),
                            "lhs_market_slug": lhs_slug,
                            "rhs_market_slug": rhs_slug,
                            "lhs_relation_side": lhs_side,
                            "rhs_relation_side": rhs_side,
                            "lhs_execution_side": lhs_exec_side,
                            "rhs_execution_side": rhs_exec_side,
                        },
                    )
                continue

            try:
                relation_lhs_book = self.clob.get_book(lhs_token)
                self._register_book_fetch()
                relation_rhs_book = self.clob.get_book(rhs_token)
                self._register_book_fetch()
                lhs_exec_book = self.clob.get_book(lhs_exec_token)
                self._register_book_fetch()
                rhs_exec_book = self.clob.get_book(rhs_exec_token)
                self._register_book_fetch()

                for strategy in active_strategies:
                    for _ in range(4):
                        self._record_strategy_family_book_fetch(strategy.strategy_family.value)

                books_to_save = {
                    lhs_token: relation_lhs_book,
                    rhs_token: relation_rhs_book,
                    lhs_exec_token: lhs_exec_book,
                    rhs_exec_token: rhs_exec_book,
                }
                for token_id, book in books_to_save.items():
                    book_cache[token_id] = book
                    self._save_raw_snapshot("clob", token_id, book.model_dump(mode="json"), cycle_started)

                for book in (relation_lhs_book, relation_rhs_book, lhs_exec_book, rhs_exec_book):
                    validation = validate_orderbook(book, required_action="BUY")
                    self._record_book_validation_result(validation)
                    for strategy in active_strategies:
                        self._record_strategy_family_book_validation_result(strategy.strategy_family.value, validation)

                for strategy in active_strategies:
                    raw_candidate, relation_audit = strategy.detect_with_audit(
                        rule,
                        relation_lhs_book,
                        relation_rhs_book,
                        lhs_exec_token,
                        lhs_exec_side,
                        lhs_exec_book,
                        rhs_exec_token,
                        rhs_exec_side,
                        rhs_exec_book,
                        self.config.paper.max_notional_per_arb,
                        self.total_buffer,
                    )
                    if raw_candidate is None:
                        if relation_audit is not None:
                            self._record_cross_market_pre_candidate_failure(relation_audit)
                        continue

                    raw_candidate = self._decorate_raw_candidate(raw_candidate)
                    self._record_raw_candidate(raw_candidate)

                    account_snapshot = self.paper_ledger.snapshot(ts=cycle_started)
                    candidate = self._qualify_and_rank_candidate(
                        raw_candidate=raw_candidate,
                        books_by_token={
                            lhs_exec_token: lhs_exec_book,
                            rhs_exec_token: rhs_exec_book,
                        },
                        account_snapshot=account_snapshot,
                    )
                    if candidate is None:
                        continue

                    self._record_qualified_candidate(candidate)
                    self._current_summary.candidates_generated += 1
                    for market_slug in candidate.market_slugs:
                        self._current_summary.market_counts[market_slug] += 1
                    self._current_summary.opportunity_type_counts[candidate.kind] += 1
                    self.store.save_candidate(candidate)
                    self.opportunity_store.save(raw_candidate.to_legacy_opportunity())

                    try:
                        decision = self.risk.evaluate(candidate, account_snapshot)
                    except Exception as exc:
                        self._record_rejection(
                            stage="risk",
                            reason_code=RejectionReason.RISK_ENGINE_ERROR.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "constraint_name": rule["name"],
                                "error": str(exc),
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                        continue

                    self.store.save_risk_decision(decision)
                    if decision.status.name in {"BLOCKED", "HALTED"}:
                        self._current_summary.risk_rejected += 1
                        for reason_code in decision.reason_codes:
                            self._record_rejection(
                                stage="risk",
                                reason_code=reason_code,
                                candidate_id=candidate.candidate_id,
                                metadata={
                                    "constraint_name": rule["name"],
                                    "strategy_family": candidate.strategy_family.value,
                                },
                            )
                        continue

                    self._current_summary.risk_accepted += 1
                    try:
                        intents, reports = self._submit_candidate_orders(
                            candidate,
                            books_by_token={
                                lhs_exec_token: lhs_exec_book,
                                rhs_exec_token: rhs_exec_book,
                            },
                            shares=candidate.sizing_hint_shares,
                        )
                    except Exception as exc:
                        self._record_rejection(
                            stage="execution",
                            reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "constraint_name": rule["name"],
                                "error": str(exc),
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                        continue

                    self._current_summary.paper_orders_created += len(intents)
                    for intent in intents:
                        self.store.save_order_intent(intent)
                    for report in reports:
                        self.store.save_execution_report(report)
                        self._record_report_stats(report)

                    for report in reports:
                        if report.filled_size > 0 and report.position_id:
                            position = self.paper_ledger.position_records.get(report.position_id)
                            if position is not None:
                                self.store.save_position_event(
                                    position_id=position.position_id,
                                    candidate_id=position.candidate_id,
                                    event_type="position_opened",
                                    symbol=position.symbol,
                                    market_slug=position.market_slug,
                                    state=position.state.value,
                                    reason_code=None,
                                    payload={
                                        "filled_size": report.filled_size,
                                        "avg_fill_price": report.avg_fill_price,
                                        "intent_id": report.intent_id,
                                    },
                                    ts=report.ts,
                                )

                    snapshot = self.paper_ledger.snapshot()
                    self.store.save_account_snapshot(snapshot)

                    if any(report.status != OrderStatus.FILLED for report in reports):
                        self._record_event(
                            "paper_candidate_incomplete_fill",
                            Severity.WARNING,
                            f"Incomplete candidate fill for {rule['name']}",
                            {
                                "candidate_id": candidate.candidate_id,
                                "constraint_name": rule["name"],
                                "reports": [report.model_dump(mode="json") for report in reports],
                            },
                        )
            except Exception as exc:
                self._record_event(
                    "political_constraint_scan_failed",
                    Severity.WARNING,
                    f"Failed to scan political relation {rule['name']}",
                    {"rule": rule["name"], "error": str(exc)},
                )
