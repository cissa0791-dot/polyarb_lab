from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from src.config_runtime.loader import load_runtime_config
from src.core.fees import total_buffer_cents
from src.core.normalize import build_yes_no_pairs
from src.core.orderbook_validation import (
    FEASIBILITY_FAILURE,
    build_fetch_failure_validation,
    orderbook_failure_class,
    validate_orderbook,
)
from src.domain.models import (
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
    PositionState,
    RejectionEvent,
    RejectionReason,
    Severity,
    SystemEvent,
)
from src.ingest.clob import ReadOnlyClob
from src.ingest.gamma import fetch_markets
from src.monitoring.logger import configure_logging, get_logger
from src.opportunity.models import RankedOpportunity
from src.opportunity.qualification import ExecutionFeasibilityEvaluator, OpportunityRanker
from src.paper.broker import PaperBroker
from src.paper.exit_policy import evaluate_exit
from src.paper.ledger import Ledger
from src.reporting.summary import RunSummaryBuilder
from src.risk.manager import RiskManager
from src.sizing.engine import DepthCappedSizer
from src.storage.event_store import ResearchStore
from src.strategies.opportunity_strategies import CrossMarketConstraintStrategy, SingleMarketMispricingStrategy
from src.utils.db import OpportunityStore


def _read_yaml(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _token_map_from_pairs(pairs: list) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for pair in pairs:
        out[(pair.market_slug, "YES")] = pair.yes_token_id
        out[(pair.market_slug, "NO")] = pair.no_token_id
    return out


def _limit_price_for_target_shares(levels: list, shares: float) -> float | None:
    remaining = float(shares)
    last_price: float | None = None
    for level in levels:
        size = float(level.size)
        if size <= 0:
            continue
        last_price = float(level.price)
        remaining -= min(remaining, size)
        if remaining <= 1e-9:
            return last_price
    return None


def _complement_side(side: str) -> str:
    return "NO" if str(side).upper() == "YES" else "YES"


class ResearchRunner:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        constraints_path: str = "config/constraints.yaml",
        debug_output_dir: str | Path | None = None,
    ):
        self.config = load_runtime_config(settings_path)
        configure_logging(
            level=self.config.monitoring.log_level,
            json_logs=self.config.monitoring.json_logs,
        )
        self.logger = get_logger("polyarb.runtime")
        self.constraints_path = constraints_path
        self.store = ResearchStore(self.config.storage.sqlite_url)
        self.opportunity_store = OpportunityStore(self.config.storage.sqlite_url)
        self.clob = ReadOnlyClob(self.config.market_data.clob_host)
        self.risk = RiskManager(self.config.risk, self.config.opportunity, self.config.execution)
        self.paper_ledger = Ledger(cash=self.config.paper.starting_cash)
        self.paper_broker = PaperBroker(self.paper_ledger)
        self.single_market_strategy = SingleMarketMispricingStrategy()
        self.cross_market_strategy = CrossMarketConstraintStrategy()
        self.feasibility = ExecutionFeasibilityEvaluator(self.config.opportunity)
        self.ranker = OpportunityRanker(self.config.opportunity)
        self.sizer = DepthCappedSizer(self.config.paper, self.config.opportunity)
        self.total_buffer = total_buffer_cents(
            {
                "fee_buffer_cents": self.config.opportunity.fee_buffer_cents,
                "slippage_buffer_cents": self.config.opportunity.slippage_buffer_cents,
            }
        )
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir is not None else Path("data/reports/orderbook_debug")
        self._current_run_id: str | None = None
        self._current_summary: RunSummaryBuilder | None = None
        self._experiment_context: dict[str, object] = {}
        self._invalid_orderbook_export_path: Path | None = None
        self._run_sequence = 0
        self._liquidity_skip_state: dict[tuple[str, str, str], dict[str, int]] = {}
        self._empty_asks_skip_threshold = 2
        self._empty_asks_skip_cooldown_runs = 2

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config.market_data.scan_interval_sec)

    def set_experiment_context(self, **context) -> None:
        self._experiment_context = {
            key: value
            for key, value in context.items()
            if value is not None
        }

    def run_once(self, experiment_context: dict[str, object] | None = None):
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
        constraints = _read_yaml(self.constraints_path)
        book_cache: dict[str, object] = {}

        try:
            markets = fetch_markets(self.config.market_data.gamma_host, self.config.market_data.market_limit)
            self._save_raw_snapshot("gamma", "markets", markets, cycle_started)
            pairs = build_yes_no_pairs(markets)
            token_map = _token_map_from_pairs(pairs)
            self._current_summary.markets_scanned = len(pairs)
            self.logger.info("loaded market pairs", extra={"payload": {"pairs": len(pairs), "run_id": self._current_run_id}})
        except Exception as exc:
            self._record_event("market_fetch_failed", Severity.ERROR, "Failed to fetch or normalize markets", {"error": str(exc)})
            raise

        self._run_single_market_scan(pairs, cycle_started, book_cache)
        self._run_cross_market_scan(constraints, token_map, cycle_started, book_cache)
        self._manage_open_positions(cycle_started, book_cache, force_reason="RUN_END_FLATTEN" if self.config.paper.flatten_on_run_end else None)

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

    def force_flatten_open_positions(self, reason: str = "MANUAL_FORCE_FLATTEN"):
        started_ts = datetime.now(timezone.utc)
        self._current_run_id = self._current_run_id or str(uuid4())
        self._current_summary = self._current_summary or RunSummaryBuilder(run_id=self._current_run_id, started_ts=started_ts)
        self._invalid_orderbook_export_path = self._invalid_orderbook_export_path or self._prepare_invalid_orderbook_export_path(started_ts)
        self._manage_open_positions(started_ts, book_cache={}, force_reason=reason)
        snapshot = self.paper_ledger.snapshot()
        self.store.save_account_snapshot(snapshot)
        self._current_summary.open_positions = snapshot.open_positions
        self._current_summary.realized_pnl = snapshot.realized_pnl
        self._current_summary.unrealized_pnl = snapshot.unrealized_pnl
        summary = self._current_summary.build(ended_ts=datetime.now(timezone.utc))
        self.store.save_run_summary(summary)
        self._current_summary = None
        self._current_run_id = None
        self._invalid_orderbook_export_path = None
        return summary

    def _run_single_market_scan(self, pairs: list, cycle_started: datetime, book_cache: dict[str, object]) -> None:
        strategy_family = self.single_market_strategy.strategy_family.value
        for pair in pairs:
            self._record_strategy_family_market_considered(strategy_family, pair.market_slug)
            yes_skip_key = (pair.market_slug, "YES", "BUY")
            no_skip_key = (pair.market_slug, "NO", "BUY")
            if self._should_skip_market_leg(*yes_skip_key) or self._should_skip_market_leg(*no_skip_key):
                if self._current_summary is not None:
                    self._current_summary.books_skipped_due_to_recent_empty_asks += 1
                self.logger.info(
                    "skipping market due to repeated empty asks",
                    extra={
                        "payload": {
                            "market_slug": pair.market_slug,
                            "skip_keys": [
                                key
                                for key in (yes_skip_key, no_skip_key)
                                if self._should_skip_market_leg(*key)
                            ],
                            "run_id": self._current_run_id,
                        }
                    },
                )
                continue
            try:
                try:
                    yes_book = self.clob.get_book(pair.yes_token_id)
                    self._register_book_fetch()
                    self._record_strategy_family_book_fetch(strategy_family)
                except Exception as exc:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=build_fetch_failure_validation(pair.yes_token_id, exc),
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.yes_token_id,
                            "side": "YES",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue

                try:
                    no_book = self.clob.get_book(pair.no_token_id)
                    self._register_book_fetch()
                    self._record_strategy_family_book_fetch(strategy_family)
                except Exception as exc:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=build_fetch_failure_validation(pair.no_token_id, exc),
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.no_token_id,
                            "side": "NO",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue
                book_cache[pair.yes_token_id] = yes_book
                book_cache[pair.no_token_id] = no_book

                self._save_raw_snapshot("clob", pair.yes_token_id, yes_book.model_dump(mode="json"), cycle_started)
                self._save_raw_snapshot("clob", pair.no_token_id, no_book.model_dump(mode="json"), cycle_started)

                yes_validation = validate_orderbook(yes_book, required_action="BUY")
                self._record_book_validation_result(yes_validation)
                self._record_strategy_family_book_validation_result(strategy_family, yes_validation)
                if not yes_validation.passed:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=yes_validation,
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.yes_token_id,
                            "side": "YES",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue
                self._update_liquidity_skip_state(pair.market_slug, "YES", "BUY", "PASSED", "candidate_filter")

                no_validation = validate_orderbook(no_book, required_action="BUY")
                self._record_book_validation_result(no_validation)
                self._record_strategy_family_book_validation_result(strategy_family, no_validation)
                if not no_validation.passed:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=no_validation,
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.no_token_id,
                            "side": "NO",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue
                self._update_liquidity_skip_state(pair.market_slug, "NO", "BUY", "PASSED", "candidate_filter")

                raw_candidate = self.single_market_strategy.detect(
                    pair,
                    yes_book,
                    no_book,
                    max_notional=self.config.paper.max_notional_per_arb,
                    total_buffer_cents=self.total_buffer,
                )
                if raw_candidate is None:
                    continue
                raw_candidate = self._decorate_raw_candidate(raw_candidate)
                self._record_raw_candidate(raw_candidate)

                account_snapshot = self.paper_ledger.snapshot(ts=cycle_started)
                candidate = self._qualify_and_rank_candidate(
                    raw_candidate=raw_candidate,
                    books_by_token={pair.yes_token_id: yes_book, pair.no_token_id: no_book},
                    account_snapshot=account_snapshot,
                )
                if candidate is None:
                    continue
                self._record_qualified_candidate(candidate)

                self._current_summary.candidates_generated += 1
                self._current_summary.market_counts[pair.market_slug] += 1
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
                            "market_slug": pair.market_slug,
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
                                "market_slug": pair.market_slug,
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                    self.logger.warning(
                        "candidate blocked by risk",
                        extra={"payload": {"candidate_id": candidate.candidate_id, "reasons": decision.reason_codes}},
                    )
                    continue

                self._current_summary.risk_accepted += 1
                try:
                    intents, reports = self._submit_pair_orders(
                        candidate,
                        pair.market_slug,
                        pair.yes_token_id,
                        pair.no_token_id,
                        yes_book,
                        no_book,
                        candidate.sizing_hint_shares,
                    )
                except Exception as exc:
                    self._record_rejection(
                        stage="execution",
                        reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                        candidate_id=candidate.candidate_id,
                        metadata={
                            "market_slug": pair.market_slug,
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
                        "paper_pair_incomplete_fill",
                        Severity.WARNING,
                        f"Incomplete pair fill for {pair.market_slug}",
                        {
                            "candidate_id": candidate.candidate_id,
                            "reports": [report.model_dump(mode="json") for report in reports],
                        },
                    )
            except Exception as exc:
                self._record_event(
                    "single_market_scan_failed",
                    Severity.WARNING,
                    f"Failed to scan market {pair.market_slug}",
                    {"market_slug": pair.market_slug, "error": str(exc)},
                )

    def _submit_pair_orders(
        self,
        candidate: RankedOpportunity,
        market_slug: str,
        yes_token_id: str,
        no_token_id: str,
        yes_book,
        no_book,
        shares: float,
    ) -> tuple[list[OrderIntent], list]:
        if shares <= 0:
            raise ValueError(f"Invalid paper order size {shares} for {market_slug}")

        ts = datetime.now(timezone.utc)
        yes_limit = _limit_price_for_target_shares(getattr(yes_book, "asks", []), shares)
        no_limit = _limit_price_for_target_shares(getattr(no_book, "asks", []), shares)
        if yes_limit is None or no_limit is None:
            raise ValueError(f"Insufficient depth to simulate pair fill for {market_slug}")

        yes_intent = OrderIntent(
            intent_id=str(uuid4()),
            candidate_id=candidate.candidate_id,
            mode=OrderMode.PAPER,
            market_slug=market_slug,
            token_id=yes_token_id,
            position_id=str(uuid4()),
            side="BUY",
            order_type=OrderType.LIMIT,
            size=shares,
            limit_price=yes_limit,
            max_notional_usd=shares * yes_limit,
            ts=ts,
        )
        no_intent = OrderIntent(
            intent_id=str(uuid4()),
            candidate_id=candidate.candidate_id,
            mode=OrderMode.PAPER,
            market_slug=market_slug,
            token_id=no_token_id,
            position_id=str(uuid4()),
            side="BUY",
            order_type=OrderType.LIMIT,
            size=shares,
            limit_price=no_limit,
            max_notional_usd=shares * no_limit,
            ts=ts,
        )

        yes_report = self.paper_broker.submit_limit_order(yes_intent, yes_book)
        no_report = self.paper_broker.submit_limit_order(no_intent, no_book)
        return [yes_intent, no_intent], [yes_report, no_report]

    def _run_cross_market_scan(self, constraints: dict, token_map: dict[tuple[str, str], str], cycle_started: datetime, book_cache: dict[str, object]) -> None:
        for rule in constraints.get("cross_market", []):
            if rule.get("relation") != "leq":
                continue

            lhs_slug = rule["lhs"]["market_slug"]
            rhs_slug = rule["rhs"]["market_slug"]
            lhs_side = rule["lhs"].get("side", "YES").upper()
            rhs_side = rule["rhs"].get("side", "YES").upper()
            lhs_token = token_map.get((lhs_slug, lhs_side))
            rhs_token = token_map.get((rhs_slug, rhs_side))
            lhs_exec_side = _complement_side(lhs_side)
            rhs_exec_side = rhs_side
            lhs_exec_token = token_map.get((lhs_slug, lhs_exec_side))
            rhs_exec_token = token_map.get((rhs_slug, rhs_exec_side))
            if not lhs_token or not rhs_token or not lhs_exec_token or not rhs_exec_token:
                continue

            try:
                relation_lhs_book = self.clob.get_book(lhs_token)
                relation_rhs_book = self.clob.get_book(rhs_token)
                lhs_exec_book = self.clob.get_book(lhs_exec_token)
                rhs_exec_book = self.clob.get_book(rhs_exec_token)
                books_to_save = {
                    lhs_token: relation_lhs_book,
                    rhs_token: relation_rhs_book,
                    lhs_exec_token: lhs_exec_book,
                    rhs_exec_token: rhs_exec_book,
                }
                for token_id, book in books_to_save.items():
                    book_cache[token_id] = book
                    self._save_raw_snapshot("clob", token_id, book.model_dump(mode="json"), cycle_started)

                raw_candidate = self.cross_market_strategy.detect(
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
            except Exception as exc:
                self._record_event(
                    "cross_market_scan_failed",
                    Severity.WARNING,
                    f"Failed to scan relation {rule['name']}",
                    {"rule": rule["name"], "error": str(exc)},
                )

    def _manage_open_positions(self, cycle_started: datetime, book_cache: dict[str, object], force_reason: str | None = None) -> None:
        open_positions = list(self.paper_ledger.get_open_positions())
        for position in open_positions:
            try:
                book = book_cache.get(position.symbol)
                if book is None:
                    try:
                        book = self.clob.get_book(position.symbol)
                    except Exception as exc:
                        self._record_invalid_orderbook(
                            stage="markout",
                            validation=build_fetch_failure_validation(position.symbol, exc),
                            metadata={
                                "candidate_id": position.candidate_id,
                                "position_id": position.position_id,
                                "market_slug": position.market_slug,
                                "token_id": position.symbol,
                                "side": "SELL",
                            },
                        )
                        continue
                    book_cache[position.symbol] = book
                    self._save_raw_snapshot("clob", position.symbol, book.model_dump(mode="json"), cycle_started)

                markout_validation = validate_orderbook(book, required_action="SELL")
                if not markout_validation.passed:
                    self._record_invalid_orderbook(
                        stage="markout",
                        validation=markout_validation,
                        metadata={
                            "candidate_id": position.candidate_id,
                            "position_id": position.position_id,
                            "market_slug": position.market_slug,
                            "token_id": position.symbol,
                            "side": "SELL",
                        },
                    )
                    continue

                best_bid = float(book.bids[0].price) if getattr(book, "bids", []) else None
                best_ask = float(book.asks[0].price) if getattr(book, "asks", []) else None

                mark = self.paper_ledger.mark_position(
                    position_id=position.position_id,
                    mark_price=best_bid,
                    ts=datetime.now(timezone.utc),
                    source_bid=best_bid,
                    source_ask=best_ask,
                )
                if mark is None:
                    continue

                self.store.save_position_mark(mark)
                self.store.save_position_event(
                    position_id=mark.position_id,
                    candidate_id=mark.candidate_id,
                    event_type="position_marked",
                    symbol=mark.symbol,
                    market_slug=mark.market_slug,
                    state=mark.state.value,
                    reason_code=None,
                    payload=mark.model_dump(mode="json"),
                    ts=mark.ts,
                )

                exit_signal = evaluate_exit(mark, self.config.paper, force_reason=force_reason)
                if exit_signal is None:
                    continue

                self.store.save_position_event(
                    position_id=exit_signal.position_id,
                    candidate_id=exit_signal.candidate_id,
                    event_type="exit_signal_generated",
                    symbol=exit_signal.symbol,
                    market_slug=exit_signal.market_slug,
                    state=position.state.value,
                    reason_code=exit_signal.reason_code,
                    payload=exit_signal.model_dump(mode="json"),
                    ts=exit_signal.ts,
                )

                limit_price = _limit_price_for_target_shares(getattr(book, "bids", []), position.remaining_shares)
                if limit_price is None:
                    self._record_rejection(
                        stage="execution",
                        reason_code=RejectionReason.INSUFFICIENT_DEPTH.value,
                        candidate_id=position.candidate_id,
                        metadata={"position_id": position.position_id, "market_slug": position.market_slug},
                    )
                    continue

                intent = OrderIntent(
                    intent_id=str(uuid4()),
                    candidate_id=position.candidate_id or "unknown",
                    mode=OrderMode.PAPER,
                    market_slug=position.market_slug,
                    token_id=position.symbol,
                    position_id=position.position_id,
                    side="SELL",
                    order_type=OrderType.LIMIT,
                    size=position.remaining_shares,
                    limit_price=limit_price,
                    max_notional_usd=position.remaining_shares * limit_price,
                    ts=exit_signal.ts,
                )
                report = self.paper_broker.submit_limit_order(intent, book)
                self.store.save_order_intent(intent)
                self.store.save_execution_report(report)
                self._current_summary.paper_orders_created += 1
                self._record_report_stats(report)

                updated_position = self.paper_ledger.position_records.get(position.position_id)
                if report.filled_size <= 0:
                    self._record_rejection(
                        stage="execution",
                        reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                        candidate_id=position.candidate_id,
                        metadata={"position_id": position.position_id, "status": report.status.value},
                    )
                    continue

                if updated_position is not None and updated_position.is_open:
                    self.store.save_position_event(
                        position_id=updated_position.position_id,
                        candidate_id=updated_position.candidate_id,
                        event_type="position_reduced",
                        symbol=updated_position.symbol,
                        market_slug=updated_position.market_slug,
                        state=updated_position.state.value,
                        reason_code=exit_signal.reason_code,
                        payload=report.model_dump(mode="json"),
                        ts=report.ts,
                    )
                    continue

                final_state = self._final_position_state(exit_signal.reason_code)
                self.paper_ledger.set_position_state(position.position_id, final_state, reason_code=exit_signal.reason_code, ts=report.ts)
                trade_summary = self.paper_ledger.build_trade_summary(position.position_id)
                if trade_summary is not None:
                    self.store.save_trade_summary(trade_summary)
                    if self._current_summary is not None:
                        self._current_summary.closed_positions += 1
                    self.store.save_position_event(
                        position_id=trade_summary.position_id,
                        candidate_id=trade_summary.candidate_id,
                        event_type=self._close_event_type(final_state),
                        symbol=trade_summary.symbol,
                        market_slug=trade_summary.market_slug,
                        state=trade_summary.state.value,
                        reason_code=exit_signal.reason_code,
                        payload=trade_summary.model_dump(mode="json"),
                        ts=trade_summary.closed_ts,
                    )
            except Exception as exc:
                self._record_event(
                    "position_management_failed",
                    Severity.WARNING,
                    f"Failed to manage paper position {position.position_id}",
                    {"position_id": position.position_id, "error": str(exc)},
                )

    def _record_raw_candidate(self, raw_candidate) -> None:
        if self._current_summary is None:
            return
        strategy_family = raw_candidate.strategy_family.value
        self._current_summary.raw_candidate_family_counts[strategy_family] += 1
        self._current_summary.record_strategy_family_signal(strategy_family, list(raw_candidate.market_slugs))

    def _record_qualified_candidate(self, candidate: RankedOpportunity) -> None:
        if self._current_summary is None:
            return
        self._current_summary.qualified_candidate_family_counts[candidate.strategy_family.value] += 1
        if candidate.research_only:
            self._current_summary.research_only_family_counts[candidate.strategy_family.value] += 1

    def _decorate_raw_candidate(self, raw_candidate):
        return raw_candidate.model_copy(
            update={
                "metadata": {
                    **raw_candidate.metadata,
                    **self._runtime_metadata(),
                    "strategy_family": raw_candidate.strategy_family.value,
                    "execution_mode": raw_candidate.execution_mode,
                    "research_only": raw_candidate.research_only,
                }
            }
        )

    def _qualify_and_rank_candidate(self, raw_candidate, books_by_token: dict[str, object], account_snapshot) -> RankedOpportunity | None:
        decision = self.feasibility.qualify(raw_candidate, books_by_token)
        if not decision.passed or decision.executable_candidate is None:
            if self._current_summary is not None and any(
                reason in {
                    RejectionReason.EDGE_BELOW_THRESHOLD.value,
                    RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value,
                    RejectionReason.INSUFFICIENT_DEPTH.value,
                    RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value,
                }
                for reason in decision.reason_codes
            ):
                self._current_summary.near_miss_candidates += 1
                self._current_summary.near_miss_family_counts[raw_candidate.strategy_family.value] += 1
            for reason_code in decision.reason_codes:
                self._record_rejection(
                    stage="qualification",
                    reason_code=reason_code,
                    candidate_id=raw_candidate.candidate_id,
                    metadata={
                        "strategy_family": raw_candidate.strategy_family.value,
                        "execution_mode": raw_candidate.execution_mode,
                        "research_only": raw_candidate.research_only,
                        "market_slugs": raw_candidate.market_slugs,
                        "raw_candidate": raw_candidate.model_dump(mode="json"),
                        "qualification": decision.metadata,
                    },
                )
            return None

        ranked = self.ranker.rank(decision.executable_candidate)
        sizing = self.sizer.size(ranked, account_snapshot)
        if sizing.notional_usd <= 1e-9 or sizing.shares <= 1e-9:
            self._record_rejection(
                stage="qualification",
                reason_code=RejectionReason.ORDER_SIZE_LIMIT.value,
                candidate_id=raw_candidate.candidate_id,
                metadata={
                    "strategy_family": raw_candidate.strategy_family.value,
                    "execution_mode": raw_candidate.execution_mode,
                    "research_only": raw_candidate.research_only,
                    "market_slugs": raw_candidate.market_slugs,
                    "sizing": {
                        "notional_usd": sizing.notional_usd,
                        "shares": sizing.shares,
                        "reason": sizing.reason,
                        "metadata": sizing.metadata,
                    },
                },
            )
            return None

        scale = sizing.notional_usd / max(ranked.target_notional_usd, 1e-9)
        scaled_legs = [
            leg.model_copy(update={"required_shares": sizing.shares})
            for leg in ranked.legs
        ]
        return ranked.model_copy(
            update={
                "target_notional_usd": sizing.notional_usd,
                "expected_payout": round(ranked.expected_payout * scale, 6),
                "estimated_net_profit_usd": round(ranked.estimated_net_profit_usd * scale, 6),
                "expected_gross_profit_usd": round(ranked.expected_gross_profit_usd * scale, 6),
                "expected_fee_usd": round(ranked.expected_fee_usd * scale, 6),
                "expected_slippage_usd": round(ranked.expected_slippage_usd * scale, 6),
                "required_depth_usd": round(ranked.required_depth_usd * scale, 6),
                "required_shares": sizing.shares,
                "sizing_hint_usd": sizing.notional_usd,
                "sizing_hint_shares": sizing.shares,
                "legs": scaled_legs,
                "metadata": {
                    **ranked.metadata,
                    **self._runtime_metadata(),
                    "strategy_family": ranked.strategy_family.value,
                    "execution_mode": ranked.execution_mode,
                    "research_only": ranked.research_only,
                    "qualification": ranked.qualification_metadata,
                    "sizing_decision": {
                        "notional_usd": sizing.notional_usd,
                        "shares": sizing.shares,
                        "reason": sizing.reason,
                        "metadata": sizing.metadata,
                    },
                },
            }
        )

    def _build_experiment_metadata(self) -> dict[str, object]:
        context = self._experiment_context.copy()
        context.setdefault("execution_mode", self.config.execution.mode)
        context["parameter_set"] = self._parameter_snapshot()
        context["scan_scope"] = {
            "market_limit": self.config.market_data.market_limit,
            "scan_interval_sec": self.config.market_data.scan_interval_sec,
        }
        return context

    def _parameter_snapshot(self) -> dict[str, float | int | bool]:
        return {
            "min_edge_cents": self.config.opportunity.min_edge_cents,
            "fee_buffer_cents": self.config.opportunity.fee_buffer_cents,
            "slippage_buffer_cents": self.config.opportunity.slippage_buffer_cents,
            "min_depth_multiple": self.config.opportunity.min_depth_multiple,
            "max_spread_cents": self.config.opportunity.max_spread_cents,
            "min_net_profit_usd": self.config.opportunity.min_net_profit_usd,
            "max_partial_fill_risk": self.config.opportunity.max_partial_fill_risk,
            "max_non_atomic_risk": self.config.opportunity.max_non_atomic_risk,
            "max_notional_per_arb": self.config.paper.max_notional_per_arb,
        }

    def _runtime_metadata(self) -> dict[str, object]:
        metadata = self._build_experiment_metadata()
        if self._current_run_id is not None:
            metadata["run_id"] = self._current_run_id
        return metadata

    def _save_raw_snapshot(self, source: str, entity_id: str, payload, ts: datetime) -> None:
        self.store.save_raw_snapshot(source, entity_id, payload, ts)
        if self._current_summary is not None:
            self._current_summary.snapshots_stored += 1

    def _record_report_stats(self, report) -> None:
        if self._current_summary is None:
            return
        if report.status == OrderStatus.FILLED:
            self._current_summary.fills += 1
        elif report.status == OrderStatus.PARTIAL:
            self._current_summary.partial_fills += 1
            self._current_summary.cancellations += 1
            self._record_rejection(
                stage="execution",
                reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                candidate_id=None,
                metadata={"intent_id": report.intent_id, "status": report.status.value},
            )
        elif report.status == OrderStatus.CANCELED:
            self._current_summary.cancellations += 1
            self._record_rejection(
                stage="execution",
                reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                candidate_id=None,
                metadata={"intent_id": report.intent_id, "status": report.status.value},
            )
        elif report.status == OrderStatus.REJECTED:
            self._record_rejection(
                stage="execution",
                reason_code=RejectionReason.PAPER_ORDER_REJECTED.value,
                candidate_id=None,
                metadata={"intent_id": report.intent_id},
            )

    def _record_rejection(self, stage: str, reason_code: str, candidate_id: str | None, metadata: dict) -> None:
        metadata = {**self._runtime_metadata(), **metadata}
        if self._current_summary is not None:
            self._current_summary.rejection_reason_counts[reason_code] += 1
            strategy_family = metadata.get("strategy_family")
            if strategy_family:
                self._current_summary.rejection_counts_by_family[str(strategy_family)][reason_code] += 1
        if self._current_run_id is None:
            return
        self.store.save_rejection_event(
            RejectionEvent(
                run_id=self._current_run_id,
                candidate_id=candidate_id,
                stage=stage,
                reason_code=reason_code,
                metadata=metadata,
                ts=datetime.now(timezone.utc),
            )
        )

    def _record_invalid_orderbook(self, stage: str, validation, metadata: dict) -> None:
        reason_code = validation.reason_code or RejectionReason.INVALID_ORDERBOOK.value
        failure_class = orderbook_failure_class(reason_code)
        debug_payload = {
            **self._runtime_metadata(),
            **metadata,
            **validation.to_debug_payload(),
            "failure_class": failure_class,
            "reason_code": reason_code,
            "stage": stage,
        }
        self._update_liquidity_skip_state(
            market_slug=str(metadata.get("market_slug") or ""),
            side=str(metadata.get("side") or ""),
            required_action=str(validation.required_action),
            reason_code=reason_code,
            stage=stage,
        )
        self._append_invalid_orderbook_export(debug_payload)
        self.logger.warning("invalid orderbook rejected", extra={"payload": debug_payload})
        self._record_rejection(
            stage=stage,
            reason_code=reason_code,
            candidate_id=metadata.get("candidate_id"),
            metadata=debug_payload,
        )

    def _record_event(self, event_type: str, severity: Severity, message: str, payload: dict) -> None:
        self.store.save_system_event(
            SystemEvent(
                event_type=event_type,
                severity=severity,
                message=message,
                payload=payload,
                ts=datetime.now(timezone.utc),
            )
        )
        if self._current_summary is not None and severity in {Severity.WARNING, Severity.ERROR, Severity.CRITICAL}:
            self._current_summary.system_errors += 1
        log_method = getattr(self.logger, severity.value.lower(), logging.info)
        log_method(message, extra={"payload": payload})

    def _register_book_fetch(self) -> None:
        if self._current_summary is not None:
            self._current_summary.books_fetched += 1

    def _record_book_validation_result(self, validation) -> None:
        if self._current_summary is None:
            return
        failure_class = orderbook_failure_class(validation.reason_code)
        if validation.passed or failure_class == FEASIBILITY_FAILURE:
            self._current_summary.books_structurally_valid += 1
        if validation.passed:
            self._current_summary.books_execution_feasible += 1

    def _record_strategy_family_market_considered(self, strategy_family: str, market_slug: str) -> None:
        if self._current_summary is not None:
            self._current_summary.record_strategy_family_market_considered(strategy_family, market_slug)

    def _record_strategy_family_book_fetch(self, strategy_family: str) -> None:
        if self._current_summary is not None:
            self._current_summary.record_strategy_family_book_fetch(strategy_family)

    def _record_strategy_family_book_validation_result(self, strategy_family: str, validation) -> None:
        if self._current_summary is None:
            return
        failure_class = orderbook_failure_class(validation.reason_code)
        self._current_summary.record_strategy_family_book_validation(
            strategy_family,
            structurally_valid=bool(validation.passed or failure_class == FEASIBILITY_FAILURE),
            execution_feasible=bool(validation.passed),
        )

    def _should_skip_market_leg(self, market_slug: str, side: str, required_action: str) -> bool:
        state = self._liquidity_skip_state.get((market_slug, side, required_action))
        if not state:
            return False
        return self._run_sequence < int(state.get("next_allowed_run_sequence", 0))

    def _update_liquidity_skip_state(
        self,
        market_slug: str,
        side: str,
        required_action: str,
        reason_code: str,
        stage: str,
    ) -> None:
        if stage != "candidate_filter" or not market_slug or not side:
            return

        key = (market_slug, side, required_action)
        if reason_code == RejectionReason.EMPTY_ASKS.value:
            state = self._liquidity_skip_state.get(key, {"consecutive_failures": 0, "next_allowed_run_sequence": 0})
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
            if state["consecutive_failures"] >= self._empty_asks_skip_threshold:
                state["next_allowed_run_sequence"] = self._run_sequence + self._empty_asks_skip_cooldown_runs
            self._liquidity_skip_state[key] = state
            return

        self._liquidity_skip_state.pop(key, None)

    def _prepare_invalid_orderbook_export_path(self, started_ts: datetime) -> Path:
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        run_id = self._current_run_id or "unknown-run"
        filename = f"{started_ts.strftime('%Y%m%dT%H%M%S')}_{run_id}_invalid_orderbooks.jsonl"
        return self.debug_output_dir / filename

    def _append_invalid_orderbook_export(self, payload: dict[str, object]) -> None:
        if self._invalid_orderbook_export_path is None:
            return
        with self._invalid_orderbook_export_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def _final_position_state(self, reason_code: str) -> PositionState:
        if reason_code == "RUN_END_FLATTEN" or reason_code == "MANUAL_FORCE_FLATTEN":
            return PositionState.FORCE_CLOSED
        if reason_code == "MAX_HOLDING_AGE":
            return PositionState.EXPIRED
        return PositionState.CLOSED

    def _close_event_type(self, state: PositionState) -> str:
        if state == PositionState.FORCE_CLOSED:
            return "position_force_closed"
        if state == PositionState.EXPIRED:
            return "position_expired"
        return "position_closed"
