from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "c_fill_likelihood_probe_approval_package.v1"
REPORT_TYPE = "c_fill_likelihood_probe_approval_package"

READY_STATUS = "C_FILL_LIKELIHOOD_APPROVAL_READY"
BLOCKED_STATUS = "C_FILL_LIKELIHOOD_APPROVAL_BLOCKED"

PROBE_TYPE = "C_FILL_LIKELIHOOD_RECONCILIATION"


def build_c_fill_likelihood_probe_approval_package(
    *,
    gate: dict[str, Any] | None = None,
    planner: dict[str, Any] | None = None,
    fill_readiness: dict[str, Any] | None = None,
    fee_reconciliation: dict[str, Any] | None = None,
    toxic_flow: dict[str, Any] | None = None,
    max_c_probe_size: float = 10.0,
    operator_size_override_approved: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only approval package for a future C fill-likelihood probe.

    The package proves whether the system is ready to ask for explicit operator
    approval for one small fill/reconciliation probe. It never creates tokens,
    never authorizes execution, and never sends/cancels orders.
    """

    now = now or datetime.now(timezone.utc)
    gate = gate or {}
    planner = planner or {}
    fill_readiness = fill_readiness or {}
    fee_reconciliation = fee_reconciliation or {}
    toxic_flow = toxic_flow or {}
    recommended = planner.get("recommended_plan") if isinstance(planner.get("recommended_plan"), dict) else {}

    quote_size = _first_float(recommended.get("quote_size"))
    quote_price = _first_float(recommended.get("quote_price"), recommended.get("quote_bid"), recommended.get("quote_ask"))
    side = str(recommended.get("selected_side") or planner.get("selected_side") or "").upper()
    plan_classification = planner.get("plan_classification") or recommended.get("plan_classification")
    fill_probability = _first_float(recommended.get("fill_probability"), recommended.get("fill_probability_proxy"))
    planner_hash = str(planner.get("planner_hash") or "").strip()
    market_slug = recommended.get("market_slug") or planner.get("target_market_slug") or gate.get("target_market_slug")

    blockers = _blockers(
        gate=gate,
        planner=planner,
        fill_readiness=fill_readiness,
        fee_reconciliation=fee_reconciliation,
        toxic_flow=toxic_flow,
        recommended=recommended,
        side=side,
        quote_size=quote_size,
        quote_price=quote_price,
        plan_classification=plan_classification,
        planner_hash=planner_hash,
        max_c_probe_size=max_c_probe_size,
        operator_size_override_approved=operator_size_override_approved,
    )
    status = READY_STATUS if not blockers else BLOCKED_STATUS

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "probe_type": PROBE_TYPE,
        "market_slug": market_slug,
        "selected_side": side or None,
        "quote_price": _round(quote_price),
        "quote_size": _round(quote_size),
        "fill_probability_proxy": _round(fill_probability),
        "planner_hash": planner_hash or None,
        "max_c_probe_size": _round(max_c_probe_size),
        "operator_size_override_approved": operator_size_override_approved,
        "constraints": {
            "one_side_only": side in {"BID_ONLY", "ASK_ONLY"},
            "one_order_only": True,
            "no_retry": True,
            "both_side_live": False,
            "strict_exact_cancel_of_remainder": True,
            "post_fill_audit_required": True,
        },
        "readiness": {
            "gate_status": gate.get("status"),
            "asserts_passed": gate.get("asserts_passed"),
            "fill_readiness_status": fill_readiness.get("status"),
            "fill_detectable": fill_readiness.get("fill_detectable") is True,
            "partial_fill_detectable": fill_readiness.get("partial_fill_detectable") is True,
            "inventory_update_source_ready": fill_readiness.get("inventory_update_source_ready") is True,
            "cash_delta_source_ready": fill_readiness.get("cash_delta_source_ready") is True,
            "fee_reconciliation_ready": fee_reconciliation.get("status") == "FEE_RECONCILIATION_READY"
            and fee_reconciliation.get("can_cover_fees") is True,
            "toxic_flow_ready": toxic_flow.get("status") == "TOXIC_FLOW_READY",
        },
        "approval_boundary": {
            "token_created": False,
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "rule": "C-class approval package is review-only and cannot create or consume tokens.",
        },
        "success_criteria": {
            "primary": [
                "fill or partial fill is detected from raw order state",
                "any remaining order quantity is cancelled by exact order id",
                "inventory delta reconciles to filled quantity",
                "deposit-wallet cash delta source is present and reconciled",
                "final open_order_count is zero",
                "token is expended and not reusable",
            ],
            "profit_required": False,
            "reward_counted_as_realized_pnl": False,
        },
        "blockers": blockers,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "profitability_claimed": False,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# C Fill-Likelihood Reconciliation Approval Package",
        "",
        f"- Status: {report.get('status')}",
        f"- Probe type: {report.get('probe_type')}",
        f"- Market: {report.get('market_slug')}",
        f"- Side: {report.get('selected_side')}",
        f"- Quote price: {report.get('quote_price')}",
        f"- Quote size: {report.get('quote_size')}",
        f"- Fill probability proxy: {report.get('fill_probability_proxy')}",
        f"- Planner hash: {report.get('planner_hash')}",
        "",
        "## Approval Boundary",
    ]
    boundary = report.get("approval_boundary") if isinstance(report.get("approval_boundary"), dict) else {}
    for key, value in boundary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _blockers(
    *,
    gate: dict[str, Any],
    planner: dict[str, Any],
    fill_readiness: dict[str, Any],
    fee_reconciliation: dict[str, Any],
    toxic_flow: dict[str, Any],
    recommended: dict[str, Any],
    side: str,
    quote_size: float | None,
    quote_price: float | None,
    plan_classification: Any,
    planner_hash: str,
    max_c_probe_size: float,
    operator_size_override_approved: bool,
) -> list[str]:
    blockers: list[str] = []
    if gate.get("status") != "LIVE_READY_APPROVED" or _first_int(gate.get("asserts_passed")) != 12:
        blockers.append("PRELIVE_12_OF_12_NOT_READY")
    if gate.get("can_submit_order") is not False:
        blockers.append("GATE_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if gate.get("live_order_sent") is not False:
        blockers.append("GATE_LIVE_ORDER_SENT_TRUE_UNEXPECTED")
    if planner.get("status") not in {"LIVE_PROBE_PLAN_READY", "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE_FOR_STABILITY"}:
        blockers.append("PLANNER_STATUS_NOT_C_CLASS_COMPATIBLE")
    if plan_classification != PROBE_TYPE:
        blockers.append("PLANNER_CLASSIFICATION_NOT_C_FILL_LIKELIHOOD")
    if not recommended:
        blockers.append("PLANNER_RECOMMENDED_PLAN_MISSING")
    if side not in {"BID_ONLY", "ASK_ONLY"}:
        blockers.append("C_PROBE_SIDE_NOT_SINGLE_SIDE")
    if quote_price is None:
        blockers.append("QUOTE_PRICE_MISSING")
    if quote_size is None:
        blockers.append("QUOTE_SIZE_MISSING")
    elif quote_size > max_c_probe_size and not operator_size_override_approved:
        blockers.append("C_PROBE_SIZE_EXCEEDS_SMALL_SIZE_LIMIT")
    if not planner_hash or len(planner_hash) != 64:
        blockers.append("PLANNER_HASH_MISSING_OR_INVALID")
    if planner.get("can_submit_order") is not False:
        blockers.append("PLANNER_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if planner.get("execution_authorized") is not False:
        blockers.append("PLANNER_EXECUTION_AUTHORIZED_TRUE_UNEXPECTED")
    if fill_readiness.get("status") != "FILL_RECONCILIATION_READINESS_READY":
        blockers.append("FILL_RECONCILIATION_READINESS_NOT_READY")
    for key in [
        "fill_detectable",
        "partial_fill_detectable",
        "inventory_update_source_ready",
        "cash_delta_source_ready",
        "fee_reconciliation_ready",
        "post_fill_audit_required",
    ]:
        if fill_readiness.get(key) is not True:
            blockers.append(f"{key.upper()}_NOT_READY")
    if fee_reconciliation.get("status") != "FEE_RECONCILIATION_READY" or fee_reconciliation.get("can_cover_fees") is not True:
        blockers.append("FEE_RECONCILIATION_NOT_READY")
    if toxic_flow.get("status") != "TOXIC_FLOW_READY":
        blockers.append("TOXIC_FLOW_NOT_READY")
    return _unique(blockers)


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


def _first_int(value: Any) -> int | None:
    parsed = _first_float(value)
    return int(parsed) if parsed is not None else None


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
        return "C_FILL_LIKELIHOOD_APPROVAL_READY: review-only fill reconciliation package is ready; token and execution remain unauthorized."
    return f"C_FILL_LIKELIHOOD_APPROVAL_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
