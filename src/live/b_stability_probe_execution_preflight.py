from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "b_stability_probe_execution_preflight.v1"
REPORT_TYPE = "b_stability_probe_execution_preflight"

READY_STATUS = "B_STABILITY_PROBE_EXECUTION_PREFLIGHT_READY"
BLOCKED_STATUS = "B_STABILITY_PROBE_EXECUTION_PREFLIGHT_BLOCKED"

PREFLIGHT_ID = "B_STABILITY_PROBE_EXECUTION_PREFLIGHT"
PROBE_TYPE = "B_LONG_OBSERVATION_STABILITY"


def build_b_stability_probe_execution_preflight(
    *,
    gate: dict[str, Any] | None,
    approval_package: dict[str, Any] | None,
    token_issuance_review: dict[str, Any] | None,
    authorization: dict[str, Any] | None,
    order_mutex: dict[str, Any] | None,
    inventory_state: dict[str, Any] | None,
    network: dict[str, Any] | None,
    market_microstructure: dict[str, Any] | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Final read-only preflight before a separately approved B probe command.

    This report does not create a token, does not consume a token, does not
    flip can_submit_order, and does not execute a live order.
    """

    now = now or datetime.now(timezone.utc)
    gate = gate or {}
    approval_package = approval_package or {}
    token_issuance_review = token_issuance_review or {}
    authorization = authorization or {}
    order_mutex = order_mutex or {}
    inventory_state = inventory_state or {}
    network = network or {}
    market_microstructure = market_microstructure or {}

    binding = (
        token_issuance_review.get("token_binding_fields")
        if isinstance(token_issuance_review.get("token_binding_fields"), dict)
        else {}
    )
    approval_binding = (
        approval_package.get("token_binding_fields")
        if isinstance(approval_package.get("token_binding_fields"), dict)
        else {}
    )
    gate_asserts = gate.get("asserts_by_id") if isinstance(gate.get("asserts_by_id"), dict) else {}

    checks = {
        "token": _token_checks(authorization, binding=binding),
        "gate": _gate_checks(gate, gate_asserts),
        "approval": _approval_checks(approval_package, token_issuance_review),
        "order_mutex": _order_mutex_checks(order_mutex),
        "inventory": _inventory_checks(inventory_state),
        "heartbeat": _heartbeat_checks(network),
        "market_context": _market_context_checks(
            market_microstructure=market_microstructure,
            binding=binding,
            approval_binding=approval_binding,
            gate=gate,
        ),
        "execution_window_boundary": _execution_window_checks(
            gate=gate,
            approval_package=approval_package,
            token_issuance_review=token_issuance_review,
            authorization=authorization,
        ),
    }
    blockers = _collect_blockers(checks)
    status = READY_STATUS if not blockers else BLOCKED_STATUS

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "preflight_id": PREFLIGHT_ID,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "probe_type": PROBE_TYPE,
        "market_slug": binding.get("market_slug"),
        "selected_side": binding.get("selected_side"),
        "quote_price": _round(binding.get("quote_price")),
        "quote_size": _round(binding.get("quote_size")),
        "hold_seconds": _first_int(binding.get("hold_seconds")),
        "token_ttl_seconds": _first_int(binding.get("token_ttl_seconds")),
        "planner_hash": binding.get("planner_hash"),
        "token_status": authorization.get("token_status"),
        "authorization_token_valid": authorization.get("authorization_token_valid") is True,
        "ttl_remaining_seconds": _round(authorization.get("ttl_remaining_seconds")),
        "asserts_passed": gate.get("asserts_passed"),
        "asserts_failed": gate.get("asserts_failed"),
        "execution_isolation_status": _assertion_reason(gate_asserts, "EXECUTION_ISOLATION_ASSERT"),
        "order_mutex_state": order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status"),
        "open_order_count": _first_int(inventory_state.get("open_order_count"), order_mutex.get("open_order_count")),
        "token_balance_shares": _round(inventory_state.get("token_balance_shares")),
        "heartbeat_status": network.get("status"),
        "heartbeat_latency_ms": _round(network.get("latency_ms") or network.get("api_latency_ms")),
        "market_context": {
            "market_status": market_microstructure.get("status"),
            "market_slug": market_microstructure.get("market_slug") or market_microstructure.get("gamma_market_slug"),
            "token_id_present": bool(market_microstructure.get("token_id")),
            "quote_bid": _round(market_microstructure.get("quote_bid")),
            "quote_ask": _round(market_microstructure.get("quote_ask")),
            "quote_size": _round(market_microstructure.get("quote_size")),
            "tick_size": _round(market_microstructure.get("tick_size") or market_microstructure.get("minimum_tick_size")),
            "checks": market_microstructure.get("checks") if isinstance(market_microstructure.get("checks"), dict) else {},
        },
        "checks": checks,
        "blockers": blockers,
        "execution_window_policy": {
            "preflight_can_submit_order": False,
            "token_consumed_here": False,
            "live_order_sent_here": False,
            "execution_authorized_here": False,
            "can_submit_order_may_be_true_only_inside_runner_atomic_submit": status == READY_STATUS,
            "required_runner_flags": [
                "--execute-live-probe",
                "--consume-token",
                "--acknowledge-live-risk",
                "--confirm-single-side-bid-probe",
                "--enable-long-observation-guards",
            ],
            "max_order_count": 1,
            "auto_retry": False,
            "maker_both_sides_live_allowed": False,
        },
        "token_ready": status == READY_STATUS,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "profitability_claimed": False,
        "pending_reward_counted_as_confirmed_reward": False,
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# B Stability Probe Execution Preflight",
        "",
        "## Boundary",
        f"- Status: {report.get('status')}",
        f"- Token ready: {report.get('token_ready')}",
        f"- Execution authorized: {report.get('execution_authorized')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        "",
        "## Binding",
        f"- Market: {report.get('market_slug')}",
        f"- Side: {report.get('selected_side')}",
        f"- Quote price: {report.get('quote_price')}",
        f"- Quote size: {report.get('quote_size')}",
        f"- Hold seconds: {report.get('hold_seconds')}",
        f"- Planner hash: {report.get('planner_hash')}",
        "",
        "## Blockers",
    ]
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _token_checks(authorization: dict[str, Any], *, binding: dict[str, Any]) -> dict[str, bool]:
    ttl_remaining = _first_float(authorization.get("ttl_remaining_seconds"))
    hold_seconds = _first_float(binding.get("hold_seconds"))
    return {
        "authorization_ready": authorization.get("status") == "SINGLE_SIDE_PROBE_AUTHORIZATION_READY",
        "token_valid": authorization.get("authorization_token_valid") is True,
        "execution_release_ready": authorization.get("execution_release_ready") is True,
        "token_unused": authorization.get("token_status") == "ISSUED_UNUSED",
        "token_not_expired": ttl_remaining is not None and ttl_remaining > 0,
        "token_ttl_covers_hold_window": ttl_remaining is not None
        and hold_seconds is not None
        and ttl_remaining >= hold_seconds,
        "authorization_does_not_enable_submit": authorization.get("can_submit_order") is False
        and authorization.get("live_order_sent") is False
        and authorization.get("execution_authorized") is False,
    }


def _gate_checks(gate: dict[str, Any], gate_asserts: dict[str, Any]) -> dict[str, bool]:
    return {
        "live_ready_approved": gate.get("status") == "LIVE_READY_APPROVED",
        "asserts_12_of_12": gate.get("asserts_passed") == 12 and gate.get("asserts_failed") == 0,
        "no_gate_blockers": not (gate.get("blockers") or []),
        "gate_does_not_enable_submit": gate.get("can_submit_order") is False
        and gate.get("live_order_sent") is False
        and gate.get("execution_enabled") is not True,
        "execution_isolation_passed": _assertion_passed(gate_asserts, "EXECUTION_ISOLATION_ASSERT"),
        "deployment_sync_passed": _assertion_passed(gate_asserts, "DEPLOYMENT_SYNC_ASSERT"),
    }


def _approval_checks(approval_package: dict[str, Any], token_issuance_review: dict[str, Any]) -> dict[str, bool]:
    approval_boundary = (
        approval_package.get("approval_boundary") if isinstance(approval_package.get("approval_boundary"), dict) else {}
    )
    return {
        "approval_package_ready": approval_package.get("status")
        == "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_READY",
        "token_issuance_review_ready": token_issuance_review.get("status")
        == "B_STABILITY_TOKEN_ISSUANCE_REVIEW_READY",
        "probe_type_b_stability": approval_package.get("probe_type") == PROBE_TYPE
        and token_issuance_review.get("probe_type") == PROBE_TYPE,
        "same_token_retry_closed": approval_boundary.get("same_token_retry_allowed") is False,
        "same_approval_retry_closed": approval_boundary.get("same_approval_retry_allowed") is False,
        "approval_does_not_enable_submit": approval_package.get("can_submit_order") is False
        and approval_package.get("live_order_sent") is False
        and approval_package.get("execution_authorized") is False,
        "review_does_not_enable_submit": token_issuance_review.get("can_submit_order") is False
        and token_issuance_review.get("live_order_sent") is False
        and token_issuance_review.get("execution_authorized") is False,
    }


def _order_mutex_checks(order_mutex: dict[str, Any]) -> dict[str, bool]:
    state = order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status")
    return {
        "order_mutex_ready": order_mutex.get("status") == "ORDER_MUTEX_READY",
        "state_no_order": state == "NO_ORDER",
        "open_order_count_zero": _first_int(order_mutex.get("open_order_count")) == 0,
        "mutex_does_not_enable_submit": order_mutex.get("can_submit_order") is False
        and order_mutex.get("live_order_sent") is False,
    }


def _inventory_checks(inventory_state: dict[str, Any]) -> dict[str, bool]:
    return {
        "inventory_clear": inventory_state.get("status") == "INVENTORY_STATE_CLEAR",
        "open_order_count_zero": _first_int(inventory_state.get("open_order_count")) == 0,
        "token_balance_zero": _same_float(inventory_state.get("token_balance_shares"), 0.0),
        "partial_fill_unresolved_false": inventory_state.get("partial_fill_unresolved") is False,
        "inventory_does_not_enable_submit": inventory_state.get("can_submit_order") is False
        and inventory_state.get("live_order_sent") is False,
    }


def _heartbeat_checks(network: dict[str, Any]) -> dict[str, bool]:
    status = str(network.get("status") or "")
    api_status = str(network.get("api_health_status") or "")
    return {
        "heartbeat_ready_or_safe_degraded": status in {"API_HEARTBEAT_READY", "API_HEARTBEAT_DEGRADED", "LIVE_NETWORK_READY"},
        "within_safety_threshold": network.get("is_within_safety_threshold") is True
        or network.get("should_block_live_trading") is False,
        "not_disconnected": network.get("disconnected") is not True and api_status != "DISCONNECTED",
        "not_critical_latency": network.get("critical_latency") is not True and api_status != "CRITICAL_LATENCY",
        "mass_cancel_ready": network.get("mass_cancel_ready") is not False,
        "heartbeat_does_not_enable_submit": network.get("can_submit_order") is False
        and network.get("live_order_sent") is False,
    }


def _market_context_checks(
    *,
    market_microstructure: dict[str, Any],
    binding: dict[str, Any],
    approval_binding: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, bool]:
    market_slug = market_microstructure.get("market_slug") or market_microstructure.get("gamma_market_slug")
    checks = market_microstructure.get("checks") if isinstance(market_microstructure.get("checks"), dict) else {}
    return {
        "market_ready": market_microstructure.get("status") == "MARKET_MICROSTRUCTURE_READY",
        "market_matches_binding": bool(binding.get("market_slug")) and market_slug == binding.get("market_slug"),
        "market_matches_gate": bool(gate.get("target_market_slug")) and market_slug == gate.get("target_market_slug"),
        "token_id_present": bool(market_microstructure.get("token_id")),
        "quote_price_matches_market_quote_bid": _same_float(binding.get("quote_price"), market_microstructure.get("quote_bid")),
        "quote_size_matches_approval": _same_float(binding.get("quote_size"), approval_binding.get("quote_size")),
        "tick_size_valid": checks.get("quote_bid_tick_aligned") is True and checks.get("quote_ask_tick_aligned") is True,
        "price_non_inverted": checks.get("quote_bid_below_quote_ask") is True,
        "market_does_not_enable_submit": market_microstructure.get("can_submit_order") is False
        and market_microstructure.get("live_order_sent") is False,
    }


def _execution_window_checks(
    *,
    gate: dict[str, Any],
    approval_package: dict[str, Any],
    token_issuance_review: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, bool]:
    return {
        "all_report_layers_can_submit_false": all(
            item.get("can_submit_order") is False
            for item in [gate, approval_package, token_issuance_review, authorization]
            if item
        ),
        "all_report_layers_live_order_sent_false": all(
            item.get("live_order_sent") is False
            for item in [gate, approval_package, token_issuance_review, authorization]
            if item
        ),
        "no_report_layer_execution_authorized": all(
            item.get("execution_authorized") is not True
            for item in [gate, approval_package, token_issuance_review, authorization]
            if item
        ),
    }


def _collect_blockers(checks: dict[str, dict[str, bool]]) -> list[str]:
    blockers: list[str] = []
    for group, group_checks in checks.items():
        for name, ok in group_checks.items():
            if not ok:
                blockers.append(f"{group.upper()}_{name.upper()}_FAILED")
    return _unique(blockers)


def _assertion_passed(asserts: dict[str, Any], assert_id: str) -> bool:
    item = asserts.get(assert_id) if isinstance(asserts, dict) else {}
    return isinstance(item, dict) and item.get("passed") is True


def _assertion_reason(asserts: dict[str, Any], assert_id: str) -> str | None:
    item = asserts.get(assert_id) if isinstance(asserts, dict) else {}
    return item.get("reason") if isinstance(item, dict) else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(parsed):
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    parsed = _first_float(*values)
    return int(parsed) if parsed is not None else None


def _same_float(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    parsed_left = _first_float(left)
    parsed_right = _first_float(right)
    if parsed_left is None or parsed_right is None:
        return False
    return abs(parsed_left - parsed_right) <= tolerance


def _round(value: Any, digits: int = 6) -> float | None:
    parsed = _first_float(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return (
            "B_STABILITY_PROBE_EXECUTION_PREFLIGHT_READY: token and live-readiness prerequisites are aligned; "
            "execution remains unauthorized until the separate runner command is invoked."
        )
    return f"B_STABILITY_PROBE_EXECUTION_PREFLIGHT_BLOCKED: {', '.join(_unique(blockers)) or 'UNKNOWN'}."
