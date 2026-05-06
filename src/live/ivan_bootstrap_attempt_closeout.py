from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ATTEMPT_ID = "IVAN_BOOTSTRAP_ATTEMPT_1"
NULL_RESULT_CLASSIFICATION = "NULL_RESULT_BOOTSTRAP_EVENT"


def build_ivan_bootstrap_attempt_closeout(
    *,
    approval: dict[str, Any] | None = None,
    lifecycle: dict[str, Any] | None = None,
    resolution: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    execution_system: dict[str, Any] | None = None,
    attempt_id: str = ATTEMPT_ID,
) -> dict[str, Any]:
    approval = approval or {}
    lifecycle = lifecycle or {}
    resolution = resolution or {}
    validation = validation or {}
    health = health or {}
    execution_system = execution_system or {}

    approval_consumed = approval.get("consumed") is True
    current_order_state = str(resolution.get("current_order_state") or lifecycle.get("resolution_state") or "")
    order_lifecycle = lifecycle.get("order_lifecycle") if isinstance(lifecycle.get("order_lifecycle"), dict) else {}

    cycle_closed = validation.get("cycle_closed") is True or resolution.get("cycle_closed") is True
    evidence_complete = validation.get("evidence_complete") is True or resolution.get("evidence_complete") is True
    token_balance_shares = _float(
        resolution.get("supporting_status_summary", {}).get("token_balance_shares")
        if isinstance(resolution.get("supporting_status_summary"), dict)
        else order_lifecycle.get("token_balance_shares")
    )
    token_open_order_count = _int(order_lifecycle.get("token_open_order_count"))
    open_buy_usdc = _float(order_lifecycle.get("account_open_buy_usdc"))
    health_open_buy_usdc = _nested_float(health, "checks", "account_open_orders", "open_buy_usdc")
    health_open_buy_count = _nested_int(health, "checks", "account_open_orders", "open_buy_order_count")

    no_inventory_remains = token_balance_shares <= 0.0
    no_open_buy_exposure = open_buy_usdc <= 0.0 and health_open_buy_usdc <= 0.0 and health_open_buy_count == 0
    order_produced_fill = token_balance_shares > 0.0 or cycle_closed or evidence_complete
    closed_cycle_exists = cycle_closed and evidence_complete
    profitability_evidence_created = closed_cycle_exists
    retry_allowed_under_consumed_approval = False if approval_consumed else None
    profitability_validation_allowed = False
    execution_disabled = execution_system.get("execution_enabled") is False

    null_result = (
        approval_consumed
        and current_order_state == "MANUALLY_RESOLVED_EXTERNALLY"
        and not order_produced_fill
        and no_inventory_remains
        and no_open_buy_exposure
        and not closed_cycle_exists
        and not profitability_evidence_created
        and retry_allowed_under_consumed_approval is False
    )

    classification = NULL_RESULT_CLASSIFICATION if null_result else "BOOTSTRAP_ATTEMPT_CLOSEOUT_REVIEW_REQUIRED"
    blockers = _blocking_reasons(
        approval_consumed=approval_consumed,
        current_order_state=current_order_state,
        order_produced_fill=order_produced_fill,
        no_inventory_remains=no_inventory_remains,
        no_open_buy_exposure=no_open_buy_exposure,
        closed_cycle_exists=closed_cycle_exists,
        evidence_complete=evidence_complete,
        execution_disabled=execution_disabled,
    )

    return {
        "report_type": "ivan_bootstrap_attempt_closeout",
        "report_schema_version": "ivan_bootstrap_attempt_closeout.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "live_decision_binding": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "attempt_id": attempt_id,
        "attempt_closeout_classification": classification,
        "approval_consumed": approval_consumed,
        "approval_status": approval.get("approval_status"),
        "consumed_at_utc": approval.get("consumed_at_utc"),
        "order_produced_fill": order_produced_fill,
        "current_order_state": current_order_state,
        "no_inventory_remains": no_inventory_remains,
        "token_balance_shares": round(token_balance_shares, 6),
        "open_buy_exposure_present": not no_open_buy_exposure,
        "open_buy_usdc": round(max(open_buy_usdc, health_open_buy_usdc), 6),
        "token_open_order_count": token_open_order_count,
        "closed_cycle_exists": closed_cycle_exists,
        "cycle_closed": cycle_closed,
        "evidence_complete": evidence_complete,
        "profitability_evidence_created": profitability_evidence_created,
        "profitability_success": False,
        "profitability_failure": False,
        "profitability_validation_allowed": profitability_validation_allowed,
        "normal_profitability_validation_allowed": False,
        "profitability_claimed": False,
        "retry_allowed_under_consumed_approval": retry_allowed_under_consumed_approval,
        "retry_allowed": False,
        "evidence_stage_active": True,
        "execution_disabled": execution_disabled,
        "execution_blockers": execution_system.get("execution_blockers") or [],
        "manual_action_required": resolution.get("manual_action_required") is True or lifecycle.get("manual_action_required") is True,
        "supporting_truth": {
            "health_status": health.get("status"),
            "health_open_buy_order_count": health_open_buy_count,
            "health_open_buy_usdc": round(health_open_buy_usdc, 6),
            "lifecycle_resolution_state": lifecycle.get("resolution_state"),
            "resolution_packet_state": resolution.get("current_order_state"),
            "validation_verdict": validation.get("one_line_verdict"),
            "execution_enabled": execution_system.get("execution_enabled"),
        },
        "operator_rules": [
            "Do not retry under the consumed one-time bootstrap approval.",
            "Do not treat cancel-without-fill as a closed cycle.",
            "Do not enter profitability validation without complete closed-cycle evidence.",
            "Any new bootstrap attempt requires a separate owner approval artifact.",
        ],
        "blocking_reasons": blockers,
        "one_line_verdict": (
            "IVAN_BOOTSTRAP_ATTEMPT_1_CLOSED_NULL_RESULT_NO_FILL_NO_PROFITABILITY"
            if null_result
            else "IVAN_BOOTSTRAP_ATTEMPT_1_CLOSEOUT_REVIEW_REQUIRED"
        ),
    }


def _blocking_reasons(
    *,
    approval_consumed: bool,
    current_order_state: str,
    order_produced_fill: bool,
    no_inventory_remains: bool,
    no_open_buy_exposure: bool,
    closed_cycle_exists: bool,
    evidence_complete: bool,
    execution_disabled: bool,
) -> list[str]:
    blockers: list[str] = []
    if not approval_consumed:
        blockers.append("APPROVAL_NOT_CONSUMED")
    if current_order_state != "MANUALLY_RESOLVED_EXTERNALLY":
        blockers.append(f"ORDER_STATE_NOT_MANUALLY_RESOLVED_EXTERNALLY:{current_order_state or 'UNKNOWN'}")
    if order_produced_fill:
        blockers.append("ORDER_PRODUCED_FILL_NOT_NULL_RESULT")
    if not no_inventory_remains:
        blockers.append("INVENTORY_REMAINS")
    if not no_open_buy_exposure:
        blockers.append("OPEN_BUY_EXPOSURE_PRESENT")
    if closed_cycle_exists or evidence_complete:
        blockers.append("CLOSED_CYCLE_OR_EVIDENCE_COMPLETE_NOT_NULL_RESULT")
    if not execution_disabled:
        blockers.append("EXECUTION_NOT_DISABLED")
    return sorted(dict.fromkeys(blockers))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _nested_float(mapping: dict[str, Any], *keys: str) -> float:
    return _float(_nested(mapping, *keys))


def _nested_int(mapping: dict[str, Any], *keys: str) -> int:
    return _int(_nested(mapping, *keys))


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
