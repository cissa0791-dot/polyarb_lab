from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from src.live.client import LiveClientError, clean_live_error_message
from src.live.one_time_auth_token import (
    DEFAULT_MODE,
    DEFAULT_SIDE,
    DEFAULT_TOKEN_PATH,
    build_authorization_report,
    load_authorization_token,
    mark_token_expended,
    write_authorization_token,
)


REPORT_SCHEMA_VERSION = "single_side_bid_probe.v1"
REPORT_TYPE = "single_side_bid_probe"

DRY_RUN_READY = "SINGLE_SIDE_BID_PROBE_DRY_RUN_READY"
BLOCKED = "SINGLE_SIDE_BID_PROBE_BLOCKED"
COMPLETED = "SINGLE_SIDE_BID_PROBE_COMPLETED"
EMERGENCY_REVIEW = "SINGLE_SIDE_BID_PROBE_EMERGENCY_REVIEW_REQUIRED"
ABORTED_CANCEL_CONFIRMED = "SINGLE_SIDE_BID_PROBE_ABORTED_CANCEL_CONFIRMED"

DEFAULT_HOLD_SECONDS = 30.0
DEFAULT_STATUS_POLL_SECONDS = 5.0

ACTIVE_ORDER_STATUSES = {"LIVE", "OPEN", "ACTIVE", "PENDING", "PLACED"}
TERMINAL_ORDER_STATUSES = {"CANCELED", "CANCELLED", "EXPIRED", "FILLED", "MATCHED", "DEAD"}

GuardSnapshotProvider = Callable[[], dict[str, Any]]


class SingleSideProbeClient(Protocol):
    def submit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        *,
        neg_risk: bool = False,
        tick_size: str | None = None,
        fee_rate_bps: int = 0,
    ) -> Any:
        ...

    def get_order_status(self, order_id: str) -> Any:
        ...

    def cancel_order(self, order_id: str) -> bool:
        ...

    def get_open_orders(self, token_id: str) -> list[Any]:
        ...


def load_json_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_probe_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def build_target_context(
    *,
    health: dict[str, Any] | None = None,
    market_microstructure: dict[str, Any] | None = None,
    rehearsal: dict[str, Any] | None = None,
    token_report: dict[str, Any] | None = None,
    market_slug: str | None = None,
    quote_price: float | None = None,
    quote_size: float | None = None,
) -> dict[str, Any]:
    health = health or {}
    market_microstructure = market_microstructure or {}
    rehearsal = rehearsal or {}
    token_report = token_report or {}
    health_target = health.get("target_market") if isinstance(health.get("target_market"), dict) else {}

    return {
        "market_slug": (
            market_slug
            or rehearsal.get("target_market_slug")
            or token_report.get("market_slug")
            or market_microstructure.get("market_slug")
            or health_target.get("market_slug")
        ),
        "token_id": market_microstructure.get("token_id") or health_target.get("token_id"),
        "side": "BUY",
        "order_side_selected": "BID_ONLY",
        "quote_price": _round(quote_price if quote_price is not None else rehearsal.get("quote_price")),
        "quote_size": _round(quote_size if quote_size is not None else rehearsal.get("quote_size")),
        "tick_size": _string_or_none(market_microstructure.get("tick_size") or health_target.get("tick_size")),
        "neg_risk": bool(market_microstructure.get("neg_risk") or health_target.get("neg_risk")),
        "best_bid": _round(market_microstructure.get("best_bid") or health_target.get("best_bid")),
        "best_ask": _round(market_microstructure.get("best_ask") or health_target.get("best_ask")),
    }


def build_probe_preflight(
    *,
    gate: dict[str, Any],
    rehearsal: dict[str, Any],
    token_report: dict[str, Any],
    target: dict[str, Any],
    max_live_risk_usdc: float,
    now: datetime | None = None,
    max_report_age_minutes: float | None = 2.0,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    blockers: list[str] = []

    gate_checks = {
        "gate_live_ready_approved": gate.get("status") == "LIVE_READY_APPROVED",
        "gate_12_of_12_passed": gate.get("asserts_passed") == 12 and gate.get("asserts_failed") == 0,
        "gate_has_no_blockers": not (gate.get("blockers") or []),
        "gate_remains_read_only": gate.get("can_submit_order") is False and gate.get("live_order_sent") is False,
        "gate_report_fresh": _report_fresh(gate, now=now, max_report_age_minutes=max_report_age_minutes),
    }
    _append_failed(blockers, gate_checks, "GATE")

    rehearsal_checks = {
        "rehearsal_review_ready": rehearsal.get("status") == "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_READY",
        "rehearsal_prelive_ready": rehearsal.get("PRELIVE_READY") is True,
        "rehearsal_does_not_authorize_execution": rehearsal.get("EXECUTION_AUTHORIZED") is False,
        "rehearsal_can_submit_order_false": rehearsal.get("CAN_SUBMIT_ORDER") is False,
        "rehearsal_live_order_sent_false": rehearsal.get("LIVE_ORDER_SENT") is False,
        "rehearsal_bid_only": rehearsal.get("order_side_selected") == "BID_ONLY",
        "rehearsal_rejects_both_sides": (rehearsal.get("final_decision") or {}).get("rejected_next_mode") == "MAKER_BOTH_SIDES_LIVE",
        "rehearsal_report_fresh": _report_fresh(rehearsal, now=now, max_report_age_minutes=max_report_age_minutes),
    }
    _append_failed(blockers, rehearsal_checks, "REHEARSAL")

    token_checks = {
        "token_report_ready": token_report.get("status") == "SINGLE_SIDE_PROBE_AUTHORIZATION_READY",
        "token_valid": token_report.get("authorization_token_valid") is True,
        "token_release_ready": token_report.get("execution_release_ready") is True,
        "token_unused": token_report.get("token_status") == "ISSUED_UNUSED",
        "token_report_does_not_submit": token_report.get("can_submit_order") is False and token_report.get("live_order_sent") is False,
    }
    _append_failed(blockers, token_checks, "TOKEN")

    target_checks = {
        "target_market_matches_gate": bool(target.get("market_slug")) and target.get("market_slug") == gate.get("target_market_slug"),
        "target_risk_matches_gate": _same_float(max_live_risk_usdc, gate.get("max_live_risk_usdc")),
        "target_token_id_present": bool(target.get("token_id")),
        "target_quote_price_present": _optional_float(target.get("quote_price")) is not None,
        "target_quote_size_present": _optional_float(target.get("quote_size")) is not None,
        "target_side_bid_only": target.get("order_side_selected") == "BID_ONLY" and target.get("side") == "BUY",
    }
    _append_failed(blockers, target_checks, "TARGET")

    ready = not blockers
    return {
        "ready": ready,
        "status": "PROBE_PREFLIGHT_READY" if ready else "PROBE_PREFLIGHT_BLOCKED",
        "generated_at_utc": now.isoformat(),
        "max_report_age_minutes": max_report_age_minutes,
        "checks": {
            "gate": gate_checks,
            "rehearsal": rehearsal_checks,
            "token": token_checks,
            "target": target_checks,
        },
        "blockers": _unique(blockers),
    }


def run_single_side_bid_probe(
    *,
    gate: dict[str, Any],
    rehearsal: dict[str, Any],
    token_report: dict[str, Any],
    health: dict[str, Any] | None,
    market_microstructure: dict[str, Any] | None,
    max_live_risk_usdc: float,
    quote_price: float,
    quote_size: float,
    token_file: str | Path = DEFAULT_TOKEN_PATH,
    execute_live_probe: bool = False,
    consume_token: bool = False,
    acknowledge_live_risk: bool = False,
    confirm_single_side_bid_probe: bool = False,
    hold_seconds: float = DEFAULT_HOLD_SECONDS,
    status_poll_seconds: float = DEFAULT_STATUS_POLL_SECONDS,
    client: SingleSideProbeClient | None = None,
    now_fn: Callable[[], datetime] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    enable_abort_guards: bool = False,
    guard_snapshot_fn: GuardSnapshotProvider | None = None,
    max_report_age_minutes: float | None = 2.0,
) -> dict[str, Any]:
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    monotonic_fn = monotonic_fn or time.monotonic
    sleep_fn = sleep_fn or time.sleep
    now = now_fn()

    token, token_load_error = load_authorization_token(token_file)
    live_token_report = build_authorization_report(
        token=token,
        token_file=token_file,
        expected_market_slug=str(rehearsal.get("target_market_slug") or gate.get("target_market_slug") or ""),
        expected_max_live_risk_usdc=max_live_risk_usdc,
        expected_quote_price=quote_price,
        expected_quote_size=quote_size,
        now=now,
        load_error=token_load_error,
    )
    if token_report:
        # Prefer a freshly revalidated token report for execution decisions, but
        # keep caller-provided report fields in the audit surface.
        token_report = {**token_report, "_runtime_revalidation": live_token_report}
    else:
        token_report = live_token_report

    target = build_target_context(
        health=health,
        market_microstructure=market_microstructure,
        rehearsal=rehearsal,
        token_report=live_token_report,
        quote_price=quote_price,
        quote_size=quote_size,
    )
    preflight = build_probe_preflight(
        gate=gate,
        rehearsal=rehearsal,
        token_report=live_token_report,
        target=target,
        max_live_risk_usdc=max_live_risk_usdc,
        now=now,
        max_report_age_minutes=max_report_age_minutes,
    )

    report = _base_report(
        now=now,
        gate=gate,
        rehearsal=rehearsal,
        token_report=token_report,
        runtime_token_report=live_token_report,
        target=target,
        max_live_risk_usdc=max_live_risk_usdc,
        execute_live_probe=execute_live_probe,
        consume_token=consume_token,
        acknowledge_live_risk=acknowledge_live_risk,
        confirm_single_side_bid_probe=confirm_single_side_bid_probe,
        hold_seconds=hold_seconds,
        status_poll_seconds=status_poll_seconds,
        preflight=preflight,
    )

    if not execute_live_probe:
        status = DRY_RUN_READY if preflight["ready"] else BLOCKED
        report.update(
            {
                "status": status,
                "can_submit_order": False,
                "live_order_sent": False,
                "token_consumed": False,
                "execution_window": {
                    "can_submit_order_during_atomic_submit": False,
                    "reason": "DRY_RUN_ONLY_NO_LIVE_ORDER_PATH_ENTERED",
                },
                "one_line_verdict": _one_line_verdict(status, preflight.get("blockers") or []),
            }
        )
        return report

    live_flag_checks = {
        "consume_token_flag": consume_token is True,
        "acknowledge_live_risk_flag": acknowledge_live_risk is True,
        "confirm_single_side_bid_probe_flag": confirm_single_side_bid_probe is True,
        "client_present": client is not None,
    }
    flag_blockers: list[str] = []
    _append_failed(flag_blockers, live_flag_checks, "LIVE_FLAG")
    if not preflight["ready"] or flag_blockers:
        blockers = _unique((preflight.get("blockers") or []) + flag_blockers)
        report.update(
            {
                "status": BLOCKED,
                "blockers": blockers,
                "can_submit_order": False,
                "live_order_sent": False,
                "token_consumed": False,
                "execution_window": {
                    "can_submit_order_during_atomic_submit": False,
                    "reason": "LIVE_EXECUTION_FLAGS_OR_PREFLIGHT_NOT_READY",
                },
                "one_line_verdict": _one_line_verdict(BLOCKED, blockers),
            }
        )
        return report

    assert client is not None  # for type-checkers; guarded above
    order_id: str | None = None
    cancel_attempted = False
    cancel_confirmed = False
    status_rows: list[dict[str, Any]] = []
    event_log: list[dict[str, Any]] = []

    try:
        expended = mark_token_expended(token, now=now_fn())
        write_authorization_token(expended, token_file)
        report["token_consumed"] = True
        report["token_consumed_before_submit"] = True
        event_log.append(_event("TOKEN_EXPENDED_BEFORE_SUBMIT", now_fn()))

        submit_started = monotonic_fn()
        request_sent_at = now_fn()
        submit = client.submit_order(
            token_id=str(target["token_id"]),
            side="BUY",
            price=float(quote_price),
            size=float(quote_size),
            neg_risk=bool(target.get("neg_risk")),
            tick_size=target.get("tick_size"),
        )
        response_received_at = now_fn()
        submit_latency_ms = round(max(0.0, monotonic_fn() - submit_started) * 1000.0, 6)
        order_id = str(getattr(submit, "order_id", "") or "")
        report["submit_result"] = {
            "request_sent_at_utc": request_sent_at.isoformat(),
            "response_received_at_utc": response_received_at.isoformat(),
            "latency_ms": submit_latency_ms,
            "order_id": order_id or None,
            "status": getattr(submit, "status", None),
            "size_matched": _round(getattr(submit, "size_matched", 0.0)),
            "avg_price": _round(getattr(submit, "avg_price", None)),
        }
        report["live_order_sent"] = bool(order_id)
        event_log.append(_event("BID_ORDER_SUBMIT_RESPONSE_RECEIVED", response_received_at, order_id=order_id or None))
        if not order_id:
            raise RuntimeError("LIVE_SUBMIT_RETURNED_NO_ORDER_ID")

        observation = _observe_order(
            client=client,
            token_id=str(target["token_id"]),
            order_id=order_id,
            hold_seconds=hold_seconds,
            poll_seconds=status_poll_seconds,
            now_fn=now_fn,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            status_rows=status_rows,
            event_log=event_log,
            enable_abort_guards=enable_abort_guards,
            guard_snapshot_fn=guard_snapshot_fn,
        )
        report["hold_observation"] = {
            "target_hold_seconds": _round(hold_seconds),
            "observed_seconds": _round(observation.get("observed_seconds")),
            "aborted": observation.get("aborted") is True,
            "abort_condition": observation.get("abort_condition"),
            "abort_snapshot": observation.get("abort_snapshot"),
            "status_polls": status_rows,
        }

        cancel_started = monotonic_fn()
        cancel_sent_at = now_fn()
        cancel_attempted = True
        cancel_ok = client.cancel_order(order_id)
        cancel_response_at = now_fn()
        cancel_latency_ms = round(max(0.0, monotonic_fn() - cancel_started) * 1000.0, 6)
        event_log.append(_event("CANCEL_BY_ORDER_ID_RESPONSE_RECEIVED", cancel_response_at, order_id=order_id, ok=bool(cancel_ok)))
        cancel_confirmed = _confirm_order_not_open(
            client=client,
            token_id=str(target["token_id"]),
            order_id=order_id,
            poll_seconds=status_poll_seconds,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
        )
        report["cancel_result"] = {
            "cancel_sent_at_utc": cancel_sent_at.isoformat(),
            "cancel_response_at_utc": cancel_response_at.isoformat(),
            "latency_ms": cancel_latency_ms,
            "order_id": order_id,
            "cancel_request_accepted": bool(cancel_ok),
            "cancel_confirmed_not_open": bool(cancel_confirmed),
        }

        abort_condition = observation.get("abort_condition")
        if cancel_ok and cancel_confirmed and abort_condition:
            final_status = ABORTED_CANCEL_CONFIRMED
            blockers = [str(abort_condition)]
        elif cancel_ok and cancel_confirmed:
            final_status = COMPLETED
            blockers = []
        else:
            final_status = EMERGENCY_REVIEW
            blockers = ["CANCEL_NOT_CONFIRMED_ORDER_RECONCILIATION_REQUIRED"]
        report.update(
            {
                "status": final_status,
                "abort_condition": abort_condition,
                "blockers": blockers,
                "can_submit_order": False,
                "execution_window": {
                    "can_submit_order_during_atomic_submit": True,
                    "auto_retry": False,
                    "max_order_count": 1,
                    "order_side": "BID_ONLY",
                },
                "one_line_verdict": _one_line_verdict(final_status, blockers),
            }
        )
        return report
    except Exception as exc:
        reason = clean_live_error_message(exc) or str(exc) or type(exc).__name__
        report.update(
            {
                "status": EMERGENCY_REVIEW,
                "reason": reason,
                "blockers": _unique([reason]),
                "can_submit_order": False,
                "execution_window": {
                    "can_submit_order_during_atomic_submit": bool(report.get("token_consumed")),
                    "auto_retry": False,
                    "max_order_count": 1,
                    "order_side": "BID_ONLY",
                },
                "one_line_verdict": _one_line_verdict(EMERGENCY_REVIEW, [reason]),
            }
        )
        if order_id and not cancel_attempted:
            try:
                cancel_attempted = True
                ok = client.cancel_order(order_id)
                cancel_confirmed = _confirm_order_not_open(
                    client=client,
                    token_id=str(target["token_id"]),
                    order_id=order_id,
                    poll_seconds=status_poll_seconds,
                    sleep_fn=sleep_fn,
                    now_fn=now_fn,
                )
                report["emergency_cancel_result"] = {
                    "order_id": order_id,
                    "cancel_request_accepted": bool(ok),
                    "cancel_confirmed_not_open": bool(cancel_confirmed),
                }
            except Exception as cancel_exc:
                report["emergency_cancel_result"] = {
                    "order_id": order_id,
                    "cancel_request_accepted": False,
                    "cancel_confirmed_not_open": False,
                    "error": clean_live_error_message(cancel_exc) or str(cancel_exc),
                }
        return report
    finally:
        report["event_log"] = event_log
        report["post_probe_required_report_sequence"] = [
            "build_inventory_state_report.py",
            "build_order_mutex_readiness_report.py",
            "build_fee_reconciliation_report.py",
            "build_api_heartbeat_report.py",
            "build_live_readiness_gate_report.py",
        ]
        report["safety_invariants"] = {
            "maker_both_sides_live_allowed": False,
            "auto_retry_allowed": False,
            "uses_cancel_by_exact_order_id": bool(cancel_attempted) if order_id else True,
            "final_top_level_can_submit_order_false": report.get("can_submit_order") is False,
            "pending_rewards_not_counted_as_cash_pnl": True,
            "estimated_profit_not_counted_as_realized_cash_pnl": True,
        }


def _observe_order(
    *,
    client: SingleSideProbeClient,
    token_id: str,
    order_id: str,
    hold_seconds: float,
    poll_seconds: float,
    now_fn: Callable[[], datetime],
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    status_rows: list[dict[str, Any]],
    event_log: list[dict[str, Any]],
    enable_abort_guards: bool = False,
    guard_snapshot_fn: GuardSnapshotProvider | None = None,
) -> dict[str, Any]:
    start = monotonic_fn()
    deadline = start + max(0.0, float(hold_seconds))
    poll_interval = max(0.0, float(poll_seconds))
    while True:
        row: dict[str, Any] = {}
        try:
            status = client.get_order_status(order_id)
            row = _status_row(status, observed_at=now_fn())
            status_rows.append(row)
            event_log.append(_event("ORDER_STATUS_POLLED", now_fn(), order_id=order_id))
        except LiveClientError as exc:
            row = {"observed_at_utc": now_fn().isoformat(), "error": clean_live_error_message(exc)}
            status_rows.append(row)
            if enable_abort_guards:
                return _abort_observation(
                    start=start,
                    monotonic_fn=monotonic_fn,
                    event_log=event_log,
                    now_fn=now_fn,
                    condition="ORDER_STATUS_READ_FAILED",
                    snapshot=row,
                    order_id=order_id,
                )
        if enable_abort_guards:
            abort = _abort_from_order_status(row)
            open_order_snapshot: dict[str, Any] | None = None
            if abort is None:
                open_order_snapshot = _open_order_guard_snapshot(
                    client=client,
                    token_id=token_id,
                    order_id=order_id,
                    row=row,
                    now_fn=now_fn,
                )
                abort = _abort_from_open_order_snapshot(open_order_snapshot)
            guard_snapshot = guard_snapshot_fn() if guard_snapshot_fn is not None and abort is None else {}
            if guard_snapshot:
                row["guard_snapshot"] = _compact_guard_snapshot(guard_snapshot)
                abort = _abort_from_guard_snapshot(guard_snapshot, row)
            if abort is not None:
                snapshot = {
                    "status_row": row,
                    "open_order_guard": open_order_snapshot,
                    "guard_snapshot": _compact_guard_snapshot(guard_snapshot),
                }
                return _abort_observation(
                    start=start,
                    monotonic_fn=monotonic_fn,
                    event_log=event_log,
                    now_fn=now_fn,
                    condition=abort,
                    snapshot=snapshot,
                    order_id=order_id,
                )
        now_mono = monotonic_fn()
        if now_mono >= deadline:
            return {
                "observed_seconds": round(max(0.0, now_mono - start), 6),
                "aborted": False,
                "abort_condition": None,
                "abort_snapshot": None,
            }
        sleep_for = min(poll_interval, max(0.0, deadline - now_mono))
        if sleep_for > 0:
            sleep_fn(sleep_for)


def _abort_observation(
    *,
    start: float,
    monotonic_fn: Callable[[], float],
    event_log: list[dict[str, Any]],
    now_fn: Callable[[], datetime],
    condition: str,
    snapshot: dict[str, Any],
    order_id: str,
) -> dict[str, Any]:
    observed_seconds = round(max(0.0, monotonic_fn() - start), 6)
    event_log.append(_event("LONG_OBSERVATION_ABORT_TRIGGERED", now_fn(), order_id=order_id, abort_condition=condition))
    return {
        "observed_seconds": observed_seconds,
        "aborted": True,
        "abort_condition": condition,
        "abort_snapshot": snapshot,
    }


def _abort_from_order_status(row: dict[str, Any]) -> str | None:
    size_matched = _optional_float(row.get("size_matched"))
    if size_matched is not None and size_matched > 0:
        return "UNEXPECTED_FILL_DETECTED"
    status = _status_text(row)
    if status in TERMINAL_ORDER_STATUSES:
        return "ORDER_DISAPPEARED_UNEXPECTEDLY"
    return None


def _open_order_guard_snapshot(
    *,
    client: SingleSideProbeClient,
    token_id: str,
    order_id: str,
    row: dict[str, Any],
    now_fn: Callable[[], datetime],
) -> dict[str, Any]:
    try:
        open_orders = client.get_open_orders(token_id)
        matching = [order for order in open_orders if _raw_field(order, "order_id", "id", "orderID") == order_id]
        return {
            "observed_at_utc": now_fn().isoformat(),
            "open_order_count": len(open_orders),
            "matching_order_count": len(matching),
            "order_id": order_id,
        }
    except LiveClientError as exc:
        return {
            "observed_at_utc": now_fn().isoformat(),
            "order_id": order_id,
            "error": clean_live_error_message(exc),
            "status_row": row,
        }


def _abort_from_open_order_snapshot(snapshot: dict[str, Any] | None) -> str | None:
    if not snapshot:
        return None
    if snapshot.get("error"):
        return "OPEN_ORDER_READ_FAILED"
    if _optional_float(snapshot.get("matching_order_count")) == 0:
        return "ORDER_DISAPPEARED_UNEXPECTEDLY"
    return None


def _abort_from_guard_snapshot(snapshot: dict[str, Any], row: dict[str, Any]) -> str | None:
    heartbeat = _nested(snapshot, "heartbeat", "network", "api_heartbeat") or {}
    api_status = str(heartbeat.get("api_health_status") or "").upper()
    heartbeat_status = str(heartbeat.get("status") or "").upper()
    if api_status == "DISCONNECTED" or heartbeat.get("disconnected") is True:
        return "HEARTBEAT_DISCONNECTED"
    if api_status == "CRITICAL_LATENCY" or heartbeat.get("critical_latency") is True:
        return "HEARTBEAT_CRITICAL_LATENCY"
    if heartbeat_status == "API_HEARTBEAT_BLOCKED":
        blockers = heartbeat.get("blockers") if isinstance(heartbeat.get("blockers"), list) else []
        if "API_HEARTBEAT_DISCONNECTED" in blockers:
            return "HEARTBEAT_DISCONNECTED"
        if "LATENCY_TOO_HIGH_FOR_LIVE_TRADING" in blockers:
            return "HEARTBEAT_CRITICAL_LATENCY"

    toxic = _nested(snapshot, "toxic_flow") or {}
    toxic_blockers = toxic.get("blockers") if isinstance(toxic.get("blockers"), list) else []
    if toxic.get("status") == "TOXIC_FLOW_BLOCKED" or any(
        item in toxic_blockers for item in {"ADVERSE_SELECTION_RISK", "VOLATILITY_LOCK"}
    ):
        return "TOXIC_FLOW_BLOCKED"

    fee = _nested(snapshot, "fee_reconciliation", "fee") or {}
    if fee.get("status") == "FEE_BLOCKER" or fee.get("can_cover_fees") is False:
        return "FEE_RECONCILIATION_FLIPS_NEGATIVE"

    inventory = _nested(snapshot, "inventory_state", "inventory") or {}
    matched = _optional_float(row.get("size_matched")) or 0.0
    balance = _optional_float(inventory.get("token_balance_shares")) or 0.0
    if matched > 0 and balance <= 0:
        return "INVENTORY_BALANCE_MISMATCH"
    if matched <= 0 and balance > 0:
        return "INVENTORY_BALANCE_MISMATCH"

    mutex = _nested(snapshot, "order_mutex", "mutex") or {}
    mutex_state = str(mutex.get("order_mutex_state") or mutex.get("live_order_status") or "").upper()
    if mutex_state and mutex_state not in {"LIVE_ORDER_OPEN", "PLACE_IN_FLIGHT", "CANCEL_IN_FLIGHT"}:
        return "MUTEX_STATE_DRIFT"
    return None


def _compact_guard_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("heartbeat", "network", "api_heartbeat", "toxic_flow", "fee_reconciliation", "fee", "inventory_state", "inventory", "order_mutex", "mutex"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            compact[key] = {
                item_key: value.get(item_key)
                for item_key in (
                    "status",
                    "api_health_status",
                    "latency_ms",
                    "blockers",
                    "can_cover_fees",
                    "token_balance_shares",
                    "open_order_count",
                    "order_mutex_state",
                    "live_order_status",
                )
                if item_key in value
            }
    return compact


def _confirm_order_not_open(
    *,
    client: SingleSideProbeClient,
    token_id: str,
    order_id: str,
    poll_seconds: float,
    sleep_fn: Callable[[float], None],
    now_fn: Callable[[], datetime],
    attempts: int = 3,
) -> bool:
    for attempt in range(max(1, attempts)):
        if attempt:
            sleep_fn(max(0.0, float(poll_seconds)))
        open_orders = client.get_open_orders(token_id)
        matching = [order for order in open_orders if str(getattr(order, "order_id", "")) == order_id]
        if not matching:
            return True
    return False


def _base_report(
    *,
    now: datetime,
    gate: dict[str, Any],
    rehearsal: dict[str, Any],
    token_report: dict[str, Any],
    runtime_token_report: dict[str, Any],
    target: dict[str, Any],
    max_live_risk_usdc: float,
    execute_live_probe: bool,
    consume_token: bool,
    acknowledge_live_risk: bool,
    confirm_single_side_bid_probe: bool,
    hold_seconds: float,
    status_poll_seconds: float,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "execution_mode": "MAKER_SINGLE_SIDE_BID_PROBE",
        "read_only_when_execute_live_probe_false": True,
        "target": target,
        "max_live_risk_usdc": _round(max_live_risk_usdc),
        "hold_seconds": _round(hold_seconds),
        "status_poll_seconds": _round(status_poll_seconds),
        "execute_live_probe_requested": bool(execute_live_probe),
        "consume_token_requested": bool(consume_token),
        "acknowledge_live_risk": bool(acknowledge_live_risk),
        "confirm_single_side_bid_probe": bool(confirm_single_side_bid_probe),
        "gate_summary": {
            "status": gate.get("status"),
            "asserts_passed": gate.get("asserts_passed"),
            "asserts_failed": gate.get("asserts_failed"),
            "blockers": gate.get("blockers") or [],
            "can_submit_order": gate.get("can_submit_order"),
            "live_order_sent": gate.get("live_order_sent"),
        },
        "rehearsal_summary": {
            "status": rehearsal.get("status"),
            "PRELIVE_READY": rehearsal.get("PRELIVE_READY"),
            "EXECUTION_AUTHORIZED": rehearsal.get("EXECUTION_AUTHORIZED"),
            "CAN_SUBMIT_ORDER": rehearsal.get("CAN_SUBMIT_ORDER"),
            "LIVE_ORDER_SENT": rehearsal.get("LIVE_ORDER_SENT"),
        },
        "authorization_summary": {
            "status": runtime_token_report.get("status"),
            "authorization_token_valid": runtime_token_report.get("authorization_token_valid"),
            "execution_release_ready": runtime_token_report.get("execution_release_ready"),
            "token_status": runtime_token_report.get("token_status"),
            "ttl_remaining_seconds": runtime_token_report.get("ttl_remaining_seconds"),
            "blockers": runtime_token_report.get("blockers") or [],
            "caller_report_status": token_report.get("status"),
        },
        "preflight": preflight,
        "submit_result": {},
        "hold_observation": {},
        "cancel_result": {},
        "token_consumed": False,
        "token_consumed_before_submit": False,
        "live_order_sent": False,
        "can_submit_order": False,
        "blockers": preflight.get("blockers") or [],
    }


def _status_row(status: Any, *, observed_at: datetime) -> dict[str, Any]:
    return {
        "observed_at_utc": observed_at.isoformat(),
        "order_id": _raw_field(status, "order_id", "id", "orderID"),
        "status": _raw_field(status, "status"),
        "size_matched": _round(_raw_field(status, "size_matched", "sizeMatched", "matched_size")),
        "size_remaining": _round(_raw_field(status, "size_remaining", "sizeRemaining", "remaining_size")),
        "avg_price": _round(_raw_field(status, "avg_price", "avgPrice")),
    }


def _event(event_type: str, at: datetime, **extra: Any) -> dict[str, Any]:
    return {"event_type": event_type, "timestamp_utc": at.isoformat(), **extra}


def _append_failed(blockers: list[str], checks: dict[str, bool], prefix: str) -> None:
    for name, ok in checks.items():
        if not ok:
            blockers.append(f"{prefix}_{name.upper()}_FAILED")


def _report_fresh(
    report: dict[str, Any],
    *,
    now: datetime,
    max_report_age_minutes: float | None,
) -> bool:
    if max_report_age_minutes is None:
        return True
    ts = _parse_ts(report.get("generated_at_utc") or report.get("generated_ts"))
    if ts is None:
        return False
    return (now - ts).total_seconds() <= max(0.0, float(max_report_age_minutes)) * 60.0


def _parse_ts(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _same_float(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    left_float = _optional_float(left)
    right_float = _optional_float(right)
    if left_float is None or right_float is None:
        return False
    return abs(left_float - right_float) <= tolerance


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_field(obj: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj.get(key)
        value = getattr(obj, key, None)
        if value is not None:
            return value
    return None


def _status_text(row: dict[str, Any]) -> str:
    value = row.get("status")
    if value in {None, ""}:
        return ""
    return str(value).strip().upper()


def _nested(snapshot: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, dict):
            return value
    return None


def _round(value: Any, digits: int = 6) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _string_or_none(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == DRY_RUN_READY:
        return "SINGLE_SIDE_BID_PROBE_DRY_RUN_READY: preflight is valid, but no live order path was entered."
    if status == COMPLETED:
        return "SINGLE_SIDE_BID_PROBE_COMPLETED: one BID probe was submitted and exact order-id cancel was confirmed."
    if status == ABORTED_CANCEL_CONFIRMED:
        return f"SINGLE_SIDE_BID_PROBE_ABORTED_CANCEL_CONFIRMED: exact cancel confirmed after {', '.join(_unique(blockers)) or 'abort'}."
    if status == EMERGENCY_REVIEW:
        return f"SINGLE_SIDE_BID_PROBE_EMERGENCY_REVIEW_REQUIRED: {', '.join(_unique(blockers)) or 'UNKNOWN'}."
    return f"SINGLE_SIDE_BID_PROBE_BLOCKED: {', '.join(_unique(blockers)) or 'UNKNOWN'}."
