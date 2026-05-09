from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "second_probe_decision_package.v1"
REPORT_TYPE = "second_probe_decision_package"

READY_STATUS = "SECOND_PROBE_DECISION_PACKAGE_READY"
BLOCKED_STATUS = "SECOND_PROBE_DECISION_PACKAGE_BLOCKED"

DEFAULT_DECISION_ID = "SECOND_PROBE_DECISION_PACKAGE"

OPTION_A = "A_BID_ONLY_ZERO_FILL_LIFECYCLE_REPLICATION"
OPTION_B = "B_LONG_OBSERVATION_BID_ONLY_STABILITY_PROBE"
OPTION_C = "C_LOW_RISK_FILL_LIKELIHOOD_PROBE"


def load_json_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_second_probe_decision_package(
    *,
    post_live_audit: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    heartbeat: dict[str, Any] | None = None,
    market_microstructure: dict[str, Any] | None = None,
    decision_id: str = DEFAULT_DECISION_ID,
    suggested_hold_seconds: int = 300,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a governance-only decision package for the second live probe.

    This report intentionally does not create an authorization token and does
    not authorize execution. It only chooses what the next probe should test.
    """

    now = now or datetime.now(timezone.utc)
    post_live_audit = post_live_audit or {}
    gate = gate or {}
    inventory_state = inventory_state or {}
    order_mutex = order_mutex or {}
    heartbeat = heartbeat or {}
    market_microstructure = market_microstructure or {}

    prior = _prior_probe_summary(post_live_audit)
    current = _current_safety_state(
        gate=gate,
        inventory_state=inventory_state,
        order_mutex=order_mutex,
        heartbeat=heartbeat,
        market_microstructure=market_microstructure,
    )
    options = _decision_options(suggested_hold_seconds=suggested_hold_seconds)
    blockers = _blockers(prior=prior, current=current)
    status = READY_STATUS if not blockers else BLOCKED_STATUS

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "decision_id": decision_id,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "target_market_slug": prior.get("target_market_slug") or market_microstructure.get("market_slug") or gate.get("target_market_slug"),
        "prior_probe_summary": prior,
        "current_safety_state": current,
        "option_table": options,
        "recommended_option": OPTION_B,
        "rejected_options": {
            OPTION_A: "Low information gain after the first zero-fill lifecycle already proved place/hold/cancel once.",
            OPTION_C: "Higher risk; defer until a longer open-order stability window proves heartbeat, mutex, cancel, and API state drift behavior.",
        },
        "recommended_probe_blueprint": {
            "mode": "SINGLE_SIDE_BID_LONG_OBSERVATION_REHEARSAL",
            "side": "BID_ONLY",
            "purpose": "Measure open-order stability, heartbeat drift, cancel reliability, stale-order behavior, and accidental fill handling over a longer window.",
            "suggested_hold_seconds": int(suggested_hold_seconds),
            "suggested_hold_window_seconds": [180, 300],
            "quote_price_policy": "Use current best bid at future authorization time; do not chase best ask in this decision package.",
            "quote_size_policy": "Keep size at the existing minimum reward/scoring test size unless a fresh readiness package chooses a lower risk cap.",
            "max_order_count": 1,
            "auto_retry_allowed": False,
            "maker_both_sides_allowed": False,
            "token_created_here": False,
            "execution_authorized_here": False,
        },
        "success_criteria": {
            "primary_success": [
                "One future BID order remains observable as open during the longer hold window unless naturally filled or cancelled.",
                "Heartbeat remains within safety threshold during the observation window.",
                "Order mutex does not drift or unlock incorrectly while the order is open.",
                "Exact order-id cancel is confirmed after the hold window.",
                "Post-probe inventory and open-order reports reconcile to the observed final state.",
            ],
            "secondary_success_if_fill_occurs": [
                "Any fill or partial fill is treated as evidence to reconcile, not as a reason to open another order.",
                "Inventory report captures the filled quantity.",
                "PnL remains unclaimed until a full closed-cycle and cash/reward reconciliation exist.",
            ],
            "failure_conditions": [
                "Cancel is not confirmed.",
                "Open order remains after shutdown.",
                "Inventory appears without a matching fill observation.",
                "Heartbeat enters critical latency or disconnected status.",
                "Mutex remains outside NO_ORDER after cancellation.",
                "Any same-token or same-approval retry is attempted.",
            ],
        },
        "approval_boundary": {
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "new_operator_approval_required": True,
            "new_token_required": True,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "rule": "This package decides the next test shape only. It does not authorize or execute the second probe.",
        },
        "blockers": blockers,
        "can_submit_order": False,
        "live_order_sent": False,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('decision_id') or DEFAULT_DECISION_ID}",
        "",
        "## Decision",
        f"- Status: {report.get('status')}",
        f"- Recommended option: {report.get('recommended_option')}",
        f"- Target market: {report.get('target_market_slug')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        "",
        "## Prior Probe",
    ]
    prior = report.get("prior_probe_summary") if isinstance(report.get("prior_probe_summary"), dict) else {}
    for key in ["status", "classification", "order_id", "token_status", "zero_fill_observed", "open_order_count", "token_balance_shares"]:
        lines.append(f"- {key}: {prior.get(key)}")
    lines.extend(["", "## Option Table"])
    options = report.get("option_table") if isinstance(report.get("option_table"), dict) else {}
    for option_id, row in options.items():
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {option_id}: recommendation={row.get('recommendation')}; "
            f"information_value={row.get('information_value')}; risk={row.get('risk_level')}"
        )
    lines.extend(["", "## Recommended Blueprint"])
    blueprint = report.get("recommended_probe_blueprint") if isinstance(report.get("recommended_probe_blueprint"), dict) else {}
    for key in [
        "mode",
        "side",
        "suggested_hold_seconds",
        "quote_price_policy",
        "max_order_count",
        "auto_retry_allowed",
        "execution_authorized_here",
    ]:
        lines.append(f"- {key}: {blueprint.get(key)}")
    lines.extend(["", "## Approval Boundary"])
    boundary = report.get("approval_boundary") if isinstance(report.get("approval_boundary"), dict) else {}
    for key in [
        "execution_authorized",
        "can_submit_order",
        "live_order_sent",
        "new_operator_approval_required",
        "new_token_required",
        "same_token_retry_allowed",
        "same_approval_retry_allowed",
    ]:
        lines.append(f"- {key}: {boundary.get(key)}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _prior_probe_summary(post_live_audit: dict[str, Any]) -> dict[str, Any]:
    final_state = post_live_audit.get("final_state") if isinstance(post_live_audit.get("final_state"), dict) else {}
    local = post_live_audit.get("local_ledger") if isinstance(post_live_audit.get("local_ledger"), dict) else {}
    next_req = post_live_audit.get("next_probe_requirements") if isinstance(post_live_audit.get("next_probe_requirements"), dict) else {}
    return {
        "audit_status": post_live_audit.get("status"),
        "classification": post_live_audit.get("classification"),
        "target_market_slug": post_live_audit.get("target_market_slug"),
        "order_id": post_live_audit.get("order_id"),
        "token_status": post_live_audit.get("token_status"),
        "submit_latency_ms": post_live_audit.get("submit_latency_ms"),
        "cancel_latency_ms": post_live_audit.get("cancel_latency_ms"),
        "zero_fill_observed": local.get("zero_fill_observed") is True,
        "cancel_confirmed_not_open": local.get("cancel_confirmed_not_open") is True,
        "open_order_count": _int_or_none(final_state.get("open_order_count")),
        "token_balance_shares": _round(final_state.get("token_balance_shares")),
        "can_submit_order": final_state.get("can_submit_order"),
        "live_order_sent": final_state.get("live_order_sent"),
        "new_operator_approval_required": next_req.get("new_operator_approval_required") is True,
        "new_token_required": next_req.get("new_token_required") is True,
        "same_token_retry_allowed": next_req.get("same_token_retry_allowed") is True,
        "same_approval_retry_allowed": next_req.get("same_approval_retry_allowed") is True,
    }


def _current_safety_state(
    *,
    gate: dict[str, Any],
    inventory_state: dict[str, Any],
    order_mutex: dict[str, Any],
    heartbeat: dict[str, Any],
    market_microstructure: dict[str, Any],
) -> dict[str, Any]:
    return {
        "gate_status": gate.get("status"),
        "gate_asserts_passed": gate.get("asserts_passed"),
        "gate_asserts_failed": gate.get("asserts_failed"),
        "gate_blockers": gate.get("blockers") if isinstance(gate.get("blockers"), list) else [],
        "gate_can_submit_order": gate.get("can_submit_order"),
        "gate_live_order_sent": gate.get("live_order_sent"),
        "inventory_status": inventory_state.get("status"),
        "open_order_count": _int_or_none(inventory_state.get("open_order_count")),
        "token_balance_shares": _round(inventory_state.get("token_balance_shares")),
        "order_mutex_status": order_mutex.get("status"),
        "order_mutex_state": order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status"),
        "heartbeat_status": heartbeat.get("status"),
        "heartbeat_latency_ms": _round(heartbeat.get("latency_ms")),
        "market_microstructure_status": market_microstructure.get("status"),
        "best_bid": _round(market_microstructure.get("best_bid")),
        "best_ask": _round(market_microstructure.get("best_ask")),
        "quote_bid": _round(market_microstructure.get("quote_bid")),
        "quote_ask": _round(market_microstructure.get("quote_ask")),
    }


def _decision_options(*, suggested_hold_seconds: int) -> dict[str, dict[str, Any]]:
    return {
        OPTION_A: {
            "purpose": "Repeat the first BID_ONLY zero-fill lifecycle to check reproducibility.",
            "information_value": "LOW",
            "risk_level": "LOW",
            "recommendation": "REJECT",
            "reason": "It mostly repeats evidence already captured by POST_LIVE_PROBE_AUDIT_001.",
        },
        OPTION_B: {
            "purpose": "Run a longer BID_ONLY observation window before cancellation.",
            "information_value": "MEDIUM",
            "risk_level": "LOW_TO_MEDIUM",
            "recommendation": "SELECT",
            "suggested_hold_seconds": int(suggested_hold_seconds),
            "reason": "Adds evidence about open-order stability, heartbeat, mutex, cancel reliability, stale state drift, and accidental fill handling without actively chasing fills.",
        },
        OPTION_C: {
            "purpose": "Design a low-risk fill-likelihood probe to seek real fill or partial-fill evidence.",
            "information_value": "HIGH",
            "risk_level": "HIGHER",
            "recommendation": "DEFER",
            "reason": "Do not increase fill pressure until the longer stability window is proven.",
        },
    }


def _blockers(*, prior: dict[str, Any], current: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if prior.get("audit_status") != "POST_LIVE_PROBE_AUDIT_READY":
        blockers.append("POST_LIVE_PROBE_AUDIT_NOT_READY")
    if prior.get("classification") != "ZERO_FILL_EXECUTION_CHAIN_PROVEN":
        blockers.append("PRIOR_PROBE_NOT_ZERO_FILL_CHAIN_PROVEN")
    if prior.get("token_status") != "EXPENDED":
        blockers.append("PRIOR_TOKEN_NOT_EXPENDED")
    if prior.get("same_token_retry_allowed") is True or prior.get("same_approval_retry_allowed") is True:
        blockers.append("PRIOR_RETRY_BOUNDARY_NOT_CLOSED")
    if prior.get("open_order_count") != 0:
        blockers.append("PRIOR_OPEN_ORDER_NOT_CLEAR")
    if not _same_float(prior.get("token_balance_shares"), 0.0):
        blockers.append("PRIOR_INVENTORY_NOT_CLEAR")
    if prior.get("can_submit_order") is not False or prior.get("live_order_sent") is not False:
        blockers.append("PRIOR_FINAL_STATE_NOT_DISABLED")
    if current.get("gate_can_submit_order") is not False:
        blockers.append("CURRENT_GATE_CAN_SUBMIT_ORDER_NOT_FALSE")
    if current.get("gate_live_order_sent") is not False:
        blockers.append("CURRENT_GATE_LIVE_ORDER_SENT_NOT_FALSE")
    if current.get("open_order_count") not in {0, None}:
        blockers.append("CURRENT_OPEN_ORDER_NOT_CLEAR")
    if not _same_float(current.get("token_balance_shares"), 0.0):
        blockers.append("CURRENT_INVENTORY_NOT_CLEAR")
    if current.get("order_mutex_state") not in {"NO_ORDER", None}:
        blockers.append("CURRENT_ORDER_MUTEX_NOT_CLEAR")
    return _unique(blockers)


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
        return "SECOND_PROBE_DECISION_PACKAGE_READY: recommend B_LONG_OBSERVATION_BID_ONLY; no execution authorized."
    return f"SECOND_PROBE_DECISION_PACKAGE_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
