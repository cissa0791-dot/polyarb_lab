from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "post_live_probe_audit.v1"
REPORT_TYPE = "post_live_probe_audit"

READY_STATUS = "POST_LIVE_PROBE_AUDIT_READY"
BLOCKED_STATUS = "POST_LIVE_PROBE_AUDIT_BLOCKED"

DEFAULT_AUDIT_ID = "POST_LIVE_PROBE_AUDIT_001"


def load_json_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_post_live_probe_audit(
    *,
    probe: dict[str, Any] | None = None,
    authorization: dict[str, Any] | None = None,
    token: dict[str, Any] | None = None,
    order_reconciliation: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    deposit_wallet: dict[str, Any] | None = None,
    heartbeat: dict[str, Any] | None = None,
    audit_id: str = DEFAULT_AUDIT_ID,
    token_file: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    probe = probe or {}
    authorization = authorization or {}
    token = token or {}
    order_reconciliation = order_reconciliation or {}
    inventory_state = inventory_state or {}
    order_mutex = order_mutex or {}
    gate = gate or {}
    deposit_wallet = deposit_wallet or {}
    heartbeat = heartbeat or {}

    submit = probe.get("submit_result") if isinstance(probe.get("submit_result"), dict) else {}
    cancel = probe.get("cancel_result") if isinstance(probe.get("cancel_result"), dict) else {}
    hold = probe.get("hold_observation") if isinstance(probe.get("hold_observation"), dict) else {}
    polls = hold.get("status_polls") if isinstance(hold.get("status_polls"), list) else []
    order_id = submit.get("order_id") or cancel.get("order_id")
    local_fill = _max_size_matched(submit, polls)

    token_status = token.get("status") or authorization.get("token_status")
    token_hash_prefix = _hash_prefix(token.get("token_hash")) or authorization.get("token_hash_prefix")
    authorization_blockers = authorization.get("blockers") if isinstance(authorization.get("blockers"), list) else []

    local_ledger = {
        "probe_status": probe.get("status"),
        "probe_blockers": probe.get("blockers") if isinstance(probe.get("blockers"), list) else [],
        "probe_abort_condition": probe.get("abort_condition") or hold.get("abort_condition"),
        "order_id": order_id,
        "live_order_sent": probe.get("live_order_sent") is True,
        "top_level_can_submit_order": probe.get("can_submit_order"),
        "submit_latency_ms": _round(submit.get("latency_ms")),
        "cancel_latency_ms": _round(cancel.get("latency_ms")),
        "cancel_request_accepted": cancel.get("cancel_request_accepted") is True,
        "cancel_confirmed_not_open": cancel.get("cancel_confirmed_not_open") is True,
        "max_observed_size_matched": _round(local_fill),
        "zero_fill_observed": _same_float(local_fill, 0.0),
        "hold_observed_seconds": _round(hold.get("observed_seconds")),
    }
    raw_order_audit = {
        "order_reconciliation_status": order_reconciliation.get("status"),
        "order_id": order_reconciliation.get("order_id"),
        "raw_order_status": order_reconciliation.get("raw_order_status"),
        "raw_order_cancelled_zero_fill": _raw_order_cancelled_zero_fill(order_reconciliation, order_id),
        "raw_order_read_only": order_reconciliation.get("read_only") is True,
        "raw_order_blockers": order_reconciliation.get("blockers")
        if isinstance(order_reconciliation.get("blockers"), list)
        else [],
    }
    local_ledger["visibility_drift_abort_confirmed"] = _visibility_drift_abort_confirmed(
        local_ledger,
        raw_order_audit=raw_order_audit,
    )
    token_audit = {
        "token_file": str(token_file) if token_file is not None else None,
        "token_status": token_status,
        "token_hash_prefix": token_hash_prefix,
        "token_expended_at_utc": token.get("expended_at_utc"),
        "authorization_report_status": authorization.get("status"),
        "authorization_token_valid": authorization.get("authorization_token_valid") is True,
        "execution_release_ready": authorization.get("execution_release_ready") is True,
        "authorization_blockers": authorization_blockers,
        "token_consumed_by_probe": probe.get("token_consumed") is True,
        "token_consumed_before_submit": probe.get("token_consumed_before_submit") is True,
        "token_cannot_be_reused": _token_cannot_be_reused(token_status, authorization),
    }
    api_state = {
        "inventory_report_status": inventory_state.get("status"),
        "open_order_count": _int_or_none(inventory_state.get("open_order_count")),
        "token_open_order_count": _int_or_none(inventory_state.get("token_open_order_count")),
        "token_balance_shares": _round(inventory_state.get("token_balance_shares")),
        "non_usdc_position_count": _int_or_none(inventory_state.get("non_usdc_position_count")),
        "order_mutex_status": order_mutex.get("status"),
        "order_mutex_state": order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status"),
        "open_order_clear": _same_float(inventory_state.get("open_order_count"), 0.0)
        and _same_float(inventory_state.get("token_open_order_count"), 0.0),
        "inventory_clear": _same_float(inventory_state.get("token_balance_shares"), 0.0)
        and _same_float(inventory_state.get("non_usdc_position_count"), 0.0),
    }
    onchain_state = {
        "deposit_wallet_status": deposit_wallet.get("status"),
        "deposit_wallet_address_present": bool(deposit_wallet.get("deposit_wallet_address")),
        "available_usdc": _round(deposit_wallet.get("available_usdc")),
        "balance_source": deposit_wallet.get("balance_source"),
        "deposit_wallet_read_only": deposit_wallet.get("read_only") is True,
        "deposit_wallet_ready": deposit_wallet.get("status") == "DEPOSIT_WALLET_READY",
    }
    final_safety = {
        "gate_status": gate.get("status"),
        "gate_asserts_passed": gate.get("asserts_passed"),
        "gate_asserts_failed": gate.get("asserts_failed"),
        "gate_blockers": gate.get("blockers") if isinstance(gate.get("blockers"), list) else [],
        "gate_can_submit_order": gate.get("can_submit_order"),
        "gate_live_order_sent": gate.get("live_order_sent"),
        "heartbeat_status": heartbeat.get("status"),
        "heartbeat_latency_ms": _round(heartbeat.get("latency_ms")),
        "can_submit_order_false": gate.get("can_submit_order") is False and probe.get("can_submit_order") is False,
        "live_order_sent_disabled_after_probe": gate.get("live_order_sent") is False,
    }
    tri_party_consistency = {
        "local_ledger_cancel_confirmed_zero_fill": (
            (
                probe.get("status") == "SINGLE_SIDE_BID_PROBE_COMPLETED"
                or local_ledger["visibility_drift_abort_confirmed"]
            )
            and local_ledger["live_order_sent"]
            and local_ledger["cancel_confirmed_not_open"]
            and local_ledger["zero_fill_observed"]
        ),
        "open_order_visibility_drift_abort_confirmed": local_ledger["visibility_drift_abort_confirmed"],
        "api_reports_show_no_open_order_or_inventory": bool(api_state["open_order_clear"] and api_state["inventory_clear"]),
        "onchain_or_deposit_wallet_readiness_available": onchain_state["deposit_wallet_ready"],
        "final_gate_returned_to_disabled_readiness": bool(
            gate.get("status") == "LIVE_READY_APPROVED"
            and gate.get("can_submit_order") is False
            and gate.get("live_order_sent") is False
        ),
    }
    tri_party_consistency["consistent"] = all(
        [
            tri_party_consistency["local_ledger_cancel_confirmed_zero_fill"],
            tri_party_consistency["api_reports_show_no_open_order_or_inventory"],
            tri_party_consistency["onchain_or_deposit_wallet_readiness_available"],
            tri_party_consistency["final_gate_returned_to_disabled_readiness"],
        ]
    )

    blockers = _blockers(
        local_ledger=local_ledger,
        token_audit=token_audit,
        api_state=api_state,
        onchain_state=onchain_state,
        final_safety=final_safety,
        tri_party_consistency=tri_party_consistency,
    )
    status = READY_STATUS if not blockers else BLOCKED_STATUS
    if status == READY_STATUS and local_ledger["visibility_drift_abort_confirmed"]:
        classification = "OPEN_ORDER_VISIBILITY_DRIFT_ABORT_CONFIRMED"
    elif status == READY_STATUS:
        classification = "ZERO_FILL_EXECUTION_CHAIN_PROVEN"
    else:
        classification = "POST_LIVE_PROBE_AUDIT_INCOMPLETE"

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "audit_id": audit_id,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "classification": classification,
        "target_market_slug": (probe.get("target") or {}).get("market_slug"),
        "order_id": order_id,
        "token_hash_prefix": token_hash_prefix,
        "token_status": token_status,
        "submit_latency_ms": local_ledger["submit_latency_ms"],
        "cancel_latency_ms": local_ledger["cancel_latency_ms"],
        "final_state": {
            "open_order_count": api_state["open_order_count"],
            "token_balance_shares": api_state["token_balance_shares"],
            "can_submit_order": False,
            "live_order_sent": False,
            "cycle_closed": False,
            "profitability_validation_allowed": False,
        },
        "local_ledger": local_ledger,
        "raw_order_audit": raw_order_audit,
        "token_audit": token_audit,
        "api_state": api_state,
        "onchain_state": onchain_state,
        "final_safety": final_safety,
        "tri_party_consistency": tri_party_consistency,
        "pnl_accounting": {
            "realized_cash_pnl_usdc": 0.0,
            "spread_pnl_usdc": 0.0,
            "confirmed_reward_usdc": 0.0,
            "pending_reward_counted_as_confirmed_reward": False,
            "profitability_claimed": False,
            "reason": _pnl_reason(classification),
        },
        "next_probe_requirements": {
            "new_operator_approval_required": True,
            "new_token_required": True,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "reason": "The prior token is expended and this audit does not authorize another execution.",
        },
        "blockers": blockers,
        "can_submit_order": False,
        "live_order_sent": False,
        "one_line_verdict": _one_line_verdict(status, classification, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('audit_id') or DEFAULT_AUDIT_ID}",
        "",
        "## Judgment",
        f"- Status: {report.get('status')}",
        f"- Classification: {report.get('classification')}",
        f"- Order ID: {report.get('order_id')}",
        f"- Token status: {report.get('token_status')}",
        f"- Submit latency ms: {report.get('submit_latency_ms')}",
        f"- Cancel latency ms: {report.get('cancel_latency_ms')}",
        "",
        "## Final State",
    ]
    final_state = report.get("final_state") if isinstance(report.get("final_state"), dict) else {}
    for key in [
        "open_order_count",
        "token_balance_shares",
        "can_submit_order",
        "live_order_sent",
        "cycle_closed",
        "profitability_validation_allowed",
    ]:
        lines.append(f"- {key}: {final_state.get(key)}")
    lines.extend(
        [
            "",
            "## Tri-Party Consistency",
        ]
    )
    consistency = report.get("tri_party_consistency") if isinstance(report.get("tri_party_consistency"), dict) else {}
    for key, value in consistency.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next Probe Boundary"])
    next_probe = report.get("next_probe_requirements") if isinstance(report.get("next_probe_requirements"), dict) else {}
    for key, value in next_probe.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _blockers(
    *,
    local_ledger: dict[str, Any],
    token_audit: dict[str, Any],
    api_state: dict[str, Any],
    onchain_state: dict[str, Any],
    final_safety: dict[str, Any],
    tri_party_consistency: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not local_ledger["live_order_sent"]:
        blockers.append("PROBE_ORDER_NOT_SENT_IN_LOCAL_LEDGER")
    if not local_ledger["cancel_confirmed_not_open"]:
        blockers.append("LOCAL_LEDGER_CANCEL_NOT_CONFIRMED")
    if not local_ledger["zero_fill_observed"]:
        blockers.append("LOCAL_LEDGER_NOT_ZERO_FILL_RECONCILE_SEPARATELY")
    if not token_audit["token_consumed_by_probe"]:
        blockers.append("TOKEN_NOT_CONSUMED_BY_PROBE")
    if not token_audit["token_consumed_before_submit"]:
        blockers.append("TOKEN_NOT_CONSUMED_BEFORE_SUBMIT")
    if not token_audit["token_cannot_be_reused"]:
        blockers.append("TOKEN_REUSE_NOT_BLOCKED")
    if not api_state["open_order_clear"]:
        blockers.append("API_OPEN_ORDER_NOT_CLEAR")
    if not api_state["inventory_clear"]:
        blockers.append("API_INVENTORY_NOT_CLEAR")
    if api_state["order_mutex_state"] != "NO_ORDER":
        blockers.append("ORDER_MUTEX_NOT_CLEAR")
    if not onchain_state["deposit_wallet_ready"]:
        blockers.append("DEPOSIT_WALLET_STATE_NOT_READY")
    if not final_safety["can_submit_order_false"]:
        blockers.append("CAN_SUBMIT_ORDER_NOT_FALSE")
    if not final_safety["live_order_sent_disabled_after_probe"]:
        blockers.append("LIVE_ORDER_SENT_NOT_RESET_IN_GATE")
    if not tri_party_consistency["consistent"]:
        blockers.append("TRI_PARTY_CONSISTENCY_NOT_PROVEN")
    return _unique(blockers)


def _token_cannot_be_reused(token_status: Any, authorization: dict[str, Any]) -> bool:
    blockers = authorization.get("blockers") if isinstance(authorization.get("blockers"), list) else []
    return (
        token_status == "EXPENDED"
        and authorization.get("authorization_token_valid") is False
        and authorization.get("execution_release_ready") is False
        and ("AUTH_TOKEN_ALREADY_EXPENDED" in blockers or authorization.get("token_status") == "EXPENDED")
    )


def _visibility_drift_abort_confirmed(
    local_ledger: dict[str, Any],
    *,
    raw_order_audit: dict[str, Any],
) -> bool:
    blockers = local_ledger.get("probe_blockers") if isinstance(local_ledger.get("probe_blockers"), list) else []
    abort_condition = str(local_ledger.get("probe_abort_condition") or "")
    return (
        local_ledger.get("probe_status") == "SINGLE_SIDE_BID_PROBE_ABORTED_CANCEL_CONFIRMED"
        and (
            abort_condition == "ORDER_DISAPPEARED_UNEXPECTEDLY"
            or "ORDER_DISAPPEARED_UNEXPECTEDLY" in blockers
        )
        and local_ledger.get("live_order_sent") is True
        and local_ledger.get("cancel_confirmed_not_open") is True
        and local_ledger.get("zero_fill_observed") is True
        and raw_order_audit.get("raw_order_cancelled_zero_fill") is True
    )


def _raw_order_cancelled_zero_fill(order_reconciliation: dict[str, Any], order_id: Any) -> bool:
    status = str(order_reconciliation.get("raw_order_status") or "").strip().upper()
    return (
        order_reconciliation.get("status") == "ORDER_STATUS_RECONCILIATION_READY"
        and order_reconciliation.get("read_only") is True
        and str(order_reconciliation.get("order_id") or "") == str(order_id or "")
        and status in {"CANCELED", "CANCELLED"}
        and _same_float(order_reconciliation.get("size_matched"), 0.0)
    )


def _pnl_reason(classification: str) -> str:
    if classification == "OPEN_ORDER_VISIBILITY_DRIFT_ABORT_CONFIRMED":
        return (
            "No fill was observed. The order was cancelled after an open-order "
            "visibility drift abort, so this is safety/consistency evidence only."
        )
    return "No fill was observed, so this is execution-chain evidence only."


def _max_size_matched(submit: dict[str, Any], polls: list[Any]) -> float | None:
    values: list[float] = []
    parsed = _float_or_none(submit.get("size_matched"))
    if parsed is not None:
        values.append(parsed)
    for row in polls:
        if not isinstance(row, dict):
            continue
        parsed = _float_or_none(row.get("size_matched"))
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None
    return max(values)


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return int(parsed)


def _round(value: Any, digits: int = 6) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _same_float(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return False
    return abs(left_float - right_float) <= tolerance


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hash_prefix(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)[:12]


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, classification: str, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return (
            f"{classification}: token expended, one order cancelled by exact id, no open order, "
            "no inventory, and execution is disabled."
        )
    return f"POST_LIVE_PROBE_AUDIT_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
