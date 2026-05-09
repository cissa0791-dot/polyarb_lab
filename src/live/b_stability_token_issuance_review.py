from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "b_stability_token_issuance_review.v1"
REPORT_TYPE = "b_stability_token_issuance_review"

READY_STATUS = "B_STABILITY_TOKEN_ISSUANCE_REVIEW_READY"
BLOCKED_STATUS = "B_STABILITY_TOKEN_ISSUANCE_REVIEW_BLOCKED"

REVIEW_ID = "B_STABILITY_TOKEN_ISSUANCE_REVIEW"
PROBE_TYPE = "B_LONG_OBSERVATION_STABILITY"
APPROVAL_READY_STATUS = "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_READY"
DEFAULT_EXECUTION_BUFFER_SECONDS = 60


def build_b_stability_token_issuance_review(
    *,
    approval_package: dict[str, Any] | None,
    planner: dict[str, Any] | None,
    authorization_report: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Review whether a B stability probe token may be created later.

    This report is deliberately one layer before token creation. It does not
    create, validate, consume, or write a token, and it does not authorize
    execution.
    """

    now = now or datetime.now(timezone.utc)
    approval_package = approval_package or {}
    planner = planner or {}
    authorization_report = authorization_report or {}

    binding = (
        approval_package.get("token_binding_fields")
        if isinstance(approval_package.get("token_binding_fields"), dict)
        else {}
    )
    boundary = (
        approval_package.get("approval_boundary")
        if isinstance(approval_package.get("approval_boundary"), dict)
        else {}
    )
    recommended = planner.get("recommended_plan") if isinstance(planner.get("recommended_plan"), dict) else {}
    fill_probability_proxy = _first_float(approval_package.get("fill_probability_proxy"))
    stability_max_fill_probability = _first_float(approval_package.get("stability_max_fill_probability"))
    token_ttl_seconds = _first_int(planner.get("token_ttl_seconds"))
    hold_seconds = _first_int(approval_package.get("hold_seconds"), planner.get("hold_seconds"), binding.get("hold_seconds"))
    required_ttl_seconds = _required_ttl_seconds(hold_seconds)

    blockers = _blockers(
        approval_package=approval_package,
        planner=planner,
        authorization_report=authorization_report,
        binding=binding,
        boundary=boundary,
        recommended=recommended,
        fill_probability_proxy=fill_probability_proxy,
        stability_max_fill_probability=stability_max_fill_probability,
        hold_seconds=hold_seconds,
        token_ttl_seconds=token_ttl_seconds,
        required_ttl_seconds=required_ttl_seconds,
    )
    status = READY_STATUS if not blockers else BLOCKED_STATUS

    token_binding_fields = {
        "market_slug": binding.get("market_slug"),
        "selected_side": binding.get("selected_side"),
        "quote_price": _round(binding.get("quote_price")),
        "quote_size": _round(binding.get("quote_size")),
        "max_live_risk_usdc": _round(binding.get("max_live_risk_usdc")),
        "hold_seconds": hold_seconds,
        "token_ttl_seconds": token_ttl_seconds,
        "planner_hash": binding.get("planner_hash"),
    }
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "review_id": REVIEW_ID,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "probe_type": PROBE_TYPE,
        "approval_package_status": approval_package.get("status"),
        "planner_status": planner.get("status"),
        "plan_classification": planner.get("plan_classification")
        or (recommended.get("plan_classification") if isinstance(recommended, dict) else None),
        "market_slug": token_binding_fields["market_slug"],
        "selected_side": token_binding_fields["selected_side"],
        "quote_price": token_binding_fields["quote_price"],
        "quote_size": token_binding_fields["quote_size"],
        "hold_seconds": hold_seconds,
        "token_ttl_seconds": token_ttl_seconds,
        "execution_buffer_seconds": DEFAULT_EXECUTION_BUFFER_SECONDS,
        "required_ttl_seconds": required_ttl_seconds,
        "max_live_risk_usdc": token_binding_fields["max_live_risk_usdc"],
        "fill_probability_proxy": _round(fill_probability_proxy),
        "fill_probability_is_model_estimate": approval_package.get("fill_probability_is_model_estimate") is True,
        "stability_max_fill_probability": _round(stability_max_fill_probability),
        "planner_hash": binding.get("planner_hash") or approval_package.get("planner_hash") or planner.get("planner_hash"),
        "token_binding_required": True,
        "token_binding_fields": token_binding_fields,
        "token_issuance_review_ready": status == READY_STATUS,
        "token_may_be_created_after_explicit_operator_approval": status == READY_STATUS,
        "allowed_next_manual_action": (
            "CREATE_ONE_TIME_TOKEN_AFTER_EXPLICIT_OPERATOR_APPROVAL_ONLY"
            if status == READY_STATUS
            else None
        ),
        "approval_evidence": {
            "approval_package_id": approval_package.get("approval_package_id"),
            "approval_status": approval_package.get("status"),
            "approval_probe_type": approval_package.get("probe_type"),
            "approval_token_created": approval_package.get("token_created"),
            "approval_execution_authorized": approval_package.get("execution_authorized"),
            "approval_can_submit_order": approval_package.get("can_submit_order"),
            "approval_live_order_sent": approval_package.get("live_order_sent"),
            "same_token_retry_allowed": boundary.get("same_token_retry_allowed"),
            "same_approval_retry_allowed": boundary.get("same_approval_retry_allowed"),
        },
        "planner_evidence": {
            "planner_status": planner.get("status"),
            "planner_hash": planner.get("planner_hash"),
            "planner_expires_at": planner.get("planner_expires_at"),
            "planner_snapshot_ts": planner.get("planner_snapshot_ts"),
            "token_ttl_seconds": token_ttl_seconds,
            "requires_new_token": planner.get("requires_new_token"),
            "requires_new_approval": planner.get("requires_new_approval"),
            "can_submit_order": planner.get("can_submit_order"),
            "execution_authorized": planner.get("execution_authorized"),
            "live_order_sent": planner.get("live_order_sent"),
        },
        "active_token_evidence": {
            "authorization_report_present": bool(authorization_report),
            "authorization_status": authorization_report.get("status"),
            "authorization_token_valid": authorization_report.get("authorization_token_valid"),
            "token_status": authorization_report.get("token_status"),
            "ttl_remaining_seconds": _round(authorization_report.get("ttl_remaining_seconds")),
            "active_token_covers_required_ttl": _active_token_covers_required_ttl(
                authorization_report=authorization_report,
                required_ttl_seconds=required_ttl_seconds,
            ),
            "blockers": authorization_report.get("blockers")
            if isinstance(authorization_report.get("blockers"), list)
            else [],
        },
        "approval_boundary": {
            "token_ready": False,
            "token_created": False,
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "same_token_retry_allowed_after_creation": False,
            "same_approval_retry_allowed_after_creation": False,
            "maker_both_sides_live_allowed": False,
            "rule": "This review only permits a future operator decision to create a matching token; it never creates or consumes one.",
        },
        "token_created": False,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "profitability_claimed": False,
        "estimated_reward_counted_as_realized_pnl": False,
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
        "pending_reward_counted_as_confirmed_reward": False,
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# B Stability Token Issuance Review",
        "",
        "## Boundary",
        f"- Status: {report.get('status')}",
        f"- Token issuance review ready: {report.get('token_issuance_review_ready')}",
        f"- Token created: {report.get('token_created')}",
        f"- Execution authorized: {report.get('execution_authorized')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        "",
        "## Token Binding",
    ]
    binding = report.get("token_binding_fields") if isinstance(report.get("token_binding_fields"), dict) else {}
    for key, value in binding.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Evidence",
            f"- Approval package status: {report.get('approval_package_status')}",
            f"- Planner status: {report.get('planner_status')}",
            f"- Plan classification: {report.get('plan_classification')}",
            f"- Fill probability proxy: {report.get('fill_probability_proxy')}",
            f"- Stability max fill probability: {report.get('stability_max_fill_probability')}",
            "",
            "## Blockers",
        ]
    )
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _blockers(
    *,
    approval_package: dict[str, Any],
    planner: dict[str, Any],
    authorization_report: dict[str, Any],
    binding: dict[str, Any],
    boundary: dict[str, Any],
    recommended: dict[str, Any],
    fill_probability_proxy: float | None,
    stability_max_fill_probability: float | None,
    hold_seconds: int | None,
    token_ttl_seconds: int | None,
    required_ttl_seconds: int | None,
) -> list[str]:
    blockers: list[str] = []
    if approval_package.get("status") != APPROVAL_READY_STATUS:
        blockers.append("B_APPROVAL_PACKAGE_NOT_READY")
    if approval_package.get("probe_type") != PROBE_TYPE:
        blockers.append("APPROVAL_PROBE_TYPE_NOT_B_STABILITY")
    if approval_package.get("token_binding_required") is not True:
        blockers.append("APPROVAL_TOKEN_BINDING_NOT_REQUIRED")
    if approval_package.get("token_created") is not False:
        blockers.append("APPROVAL_TOKEN_CREATED_UNEXPECTED")
    if approval_package.get("execution_authorized") is not False:
        blockers.append("APPROVAL_EXECUTION_AUTHORIZED_UNEXPECTED")
    if approval_package.get("can_submit_order") is not False:
        blockers.append("APPROVAL_CAN_SUBMIT_ORDER_UNEXPECTED")
    if approval_package.get("live_order_sent") is not False:
        blockers.append("APPROVAL_LIVE_ORDER_SENT_UNEXPECTED")
    if boundary.get("token_ready") is not False:
        blockers.append("APPROVAL_BOUNDARY_TOKEN_READY_UNEXPECTED")
    if boundary.get("token_created") is not False:
        blockers.append("APPROVAL_BOUNDARY_TOKEN_CREATED_UNEXPECTED")
    if boundary.get("same_token_retry_allowed") is not False:
        blockers.append("SAME_TOKEN_RETRY_NOT_CLOSED")
    if boundary.get("same_approval_retry_allowed") is not False:
        blockers.append("SAME_APPROVAL_RETRY_NOT_CLOSED")
    if planner.get("status") != "LIVE_PROBE_PLAN_READY":
        blockers.append("PLANNER_NOT_READY")
    if (planner.get("plan_classification") or recommended.get("plan_classification")) != PROBE_TYPE:
        blockers.append("PLANNER_CLASSIFICATION_NOT_B_STABILITY")
    if planner.get("requires_new_token") is not True:
        blockers.append("PLANNER_DOES_NOT_REQUIRE_NEW_TOKEN")
    if planner.get("requires_new_approval") is not True:
        blockers.append("PLANNER_DOES_NOT_REQUIRE_NEW_APPROVAL")
    if planner.get("execution_authorized") is not False:
        blockers.append("PLANNER_EXECUTION_AUTHORIZED_UNEXPECTED")
    if planner.get("can_submit_order") is not False:
        blockers.append("PLANNER_CAN_SUBMIT_ORDER_UNEXPECTED")
    if planner.get("live_order_sent") is not False:
        blockers.append("PLANNER_LIVE_ORDER_SENT_UNEXPECTED")
    if not _valid_hash(binding.get("planner_hash")):
        blockers.append("TOKEN_BINDING_PLANNER_HASH_INVALID")
    if planner.get("planner_hash") and binding.get("planner_hash") != planner.get("planner_hash"):
        blockers.append("TOKEN_BINDING_PLANNER_HASH_MISMATCH")
    if not _same_float(binding.get("quote_price"), approval_package.get("quote_price")):
        blockers.append("TOKEN_BINDING_QUOTE_PRICE_MISMATCH")
    if not _same_float(binding.get("quote_size"), approval_package.get("quote_size")):
        blockers.append("TOKEN_BINDING_QUOTE_SIZE_MISMATCH")
    if not _same_float(binding.get("max_live_risk_usdc"), approval_package.get("max_live_risk_usdc")):
        blockers.append("TOKEN_BINDING_MAX_RISK_MISMATCH")
    if hold_seconds is None or hold_seconds <= 0:
        blockers.append("HOLD_SECONDS_MISSING_OR_INVALID")
    elif not _same_float(binding.get("hold_seconds"), hold_seconds):
        blockers.append("TOKEN_BINDING_HOLD_SECONDS_MISMATCH")
    if token_ttl_seconds is None or token_ttl_seconds <= 0:
        blockers.append("TOKEN_TTL_SECONDS_MISSING_OR_INVALID")
    elif required_ttl_seconds is not None and token_ttl_seconds < required_ttl_seconds:
        blockers.append("TOKEN_TTL_SHORTER_THAN_HOLD_PLUS_BUFFER")
    if fill_probability_proxy is None:
        blockers.append("FILL_PROBABILITY_PROXY_MISSING")
    elif stability_max_fill_probability is not None and fill_probability_proxy > stability_max_fill_probability:
        blockers.append("FILL_PROBABILITY_PROXY_TOO_HIGH_FOR_B_STABILITY")
    if approval_package.get("fill_probability_is_model_estimate") is not True:
        blockers.append("FILL_PROBABILITY_NOT_MARKED_AS_PROXY")
    if approval_package.get("pending_reward_counted_as_confirmed_reward") is not False:
        blockers.append("PENDING_REWARD_ACCOUNTING_BOUNDARY_BROKEN")
    if approval_package.get("estimated_net_profit_counted_as_realized_cash_pnl") is not False:
        blockers.append("ESTIMATED_NET_PROFIT_ACCOUNTING_BOUNDARY_BROKEN")
    active_token_covers_required_ttl = _active_token_covers_required_ttl(
        authorization_report=authorization_report,
        required_ttl_seconds=required_ttl_seconds,
    )
    if authorization_report.get("authorization_token_valid") is True and active_token_covers_required_ttl:
        blockers.append("ACTIVE_AUTHORIZATION_TOKEN_ALREADY_VALID")
    if authorization_report.get("execution_release_ready") is True and active_token_covers_required_ttl:
        blockers.append("ACTIVE_EXECUTION_RELEASE_ALREADY_READY")
    return _unique(blockers)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _same_float(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    parsed_left = _first_float(left)
    parsed_right = _first_float(right)
    if parsed_left is None or parsed_right is None:
        return False
    return abs(parsed_left - parsed_right) <= tolerance


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


def _required_ttl_seconds(hold_seconds: int | None) -> int | None:
    if hold_seconds is None:
        return None
    return hold_seconds + DEFAULT_EXECUTION_BUFFER_SECONDS


def _active_token_covers_required_ttl(
    *,
    authorization_report: dict[str, Any],
    required_ttl_seconds: int | None,
) -> bool:
    ttl_remaining = _first_float(authorization_report.get("ttl_remaining_seconds"))
    return (
        authorization_report.get("authorization_token_valid") is True
        and authorization_report.get("execution_release_ready") is True
        and ttl_remaining is not None
        and required_ttl_seconds is not None
        and ttl_remaining >= required_ttl_seconds
    )


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
            "B_STABILITY_TOKEN_ISSUANCE_REVIEW_READY: a matching one-time token may be created only after "
            "separate explicit operator approval; no token or execution is authorized here."
        )
    return f"B_STABILITY_TOKEN_ISSUANCE_REVIEW_BLOCKED: {', '.join(_unique(blockers)) or 'UNKNOWN'}."
