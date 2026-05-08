from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REPORT_SCHEMA_VERSION = "single_side_live_rehearsal.v1"
REPORT_TYPE = "single_side_live_rehearsal"

READY_STATUS = "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_READY"
BLOCKED_STATUS = "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_BLOCKED"

RECOMMENDED_NEXT_MODE = "MAKER_SINGLE_SIDE_LIVE_REHEARSAL"
REJECTED_NEXT_MODE = "MAKER_BOTH_SIDES_LIVE"
DEFAULT_ORDER_SIDE = "BID_ONLY"


def build_single_side_live_rehearsal_report(
    *,
    gate: dict[str, Any],
    deployment: dict[str, Any],
    execution_system: dict[str, Any],
    deposit_wallet: dict[str, Any],
    market_microstructure: dict[str, Any],
    network: dict[str, Any],
    order_mutex: dict[str, Any],
    fee_reconciliation: dict[str, Any],
    inventory_state: dict[str, Any],
    toxic_flow: dict[str, Any],
    probe_authorization: dict[str, Any] | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    target_market_slug: str | None = None,
    max_live_risk_usdc: float | None = None,
    order_side: str = DEFAULT_ORDER_SIDE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the final switch review packet for a future single-side live probe.

    This report is a governance artifact only. It deliberately keeps execution
    unauthorized even when the live-readiness gate is 12/12.
    """

    now = now or datetime.now(timezone.utc)
    probe_authorization = probe_authorization or {}
    assertions = gate.get("assertions") if isinstance(gate.get("assertions"), list) else []
    asserts_passed = _optional_int(gate.get("asserts_passed"))
    asserts_failed = _optional_int(gate.get("asserts_failed"))
    gate_blockers = gate.get("blockers") if isinstance(gate.get("blockers"), list) else []
    prelive_ready = (
        gate.get("status") == "LIVE_READY_APPROVED"
        and asserts_passed == 12
        and asserts_failed == 0
        and not gate_blockers
    )

    quote_bid = _first_float(market_microstructure.get("quote_bid"), market_microstructure.get("best_bid"))
    quote_ask = _first_float(market_microstructure.get("quote_ask"), market_microstructure.get("best_ask"))
    quote_size = _first_float(market_microstructure.get("quote_size"), market_microstructure.get("planned_quote_size"))
    selected_quote_price = quote_bid if order_side == "BID_ONLY" else quote_ask

    execution_authorized = False
    can_submit_order = False
    live_order_sent = False
    blockers: list[str] = []

    if not prelive_ready:
        blockers.append("PRELIVE_12_OF_12_NOT_READY")
    if deployment.get("unreviewed_changes_present") is True or deployment.get("local_dirty") is True or deployment.get("remote_dirty") is True:
        blockers.append("DIRTY_DEPLOYMENT_FOR_FINAL_SWITCH_REVIEW")
    if deployment.get("head_matches_approved") is not True or deployment.get("critical_checksums_match") is not True:
        blockers.append("DEPLOYMENT_SYNC_NOT_PROVEN_FOR_FINAL_SWITCH_REVIEW")
    if execution_system.get("status") != "EXECUTION_ISOLATION_READY" or execution_system.get("single_writer_ok") is not True:
        blockers.append("EXECUTION_ISOLATION_NOT_READY_FOR_FINAL_SWITCH_REVIEW")
    if _optional_int(execution_system.get("suspicious_process_count")) not in {0, None}:
        blockers.append("SUSPICIOUS_EXECUTION_PROCESS_PRESENT")
    if gate.get("can_submit_order") is not False:
        blockers.append("GATE_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if gate.get("live_order_sent") is not False:
        blockers.append("GATE_LIVE_ORDER_SENT_TRUE_UNEXPECTED")

    status = READY_STATUS if not blockers else BLOCKED_STATUS
    realized_cash_pnl_usdc = 0.0
    confirmed_reward_usdc = 0.0
    estimated_net_profit_usdc = _first_float(fee_reconciliation.get("estimated_net_profit_usdc"))
    pending_reward_usdc = _first_float(
        market_microstructure.get("accrued_pending_yield_usdc"),
        market_microstructure.get("pending_reward_usdc"),
        toxic_flow.get("pending_reward_usdc"),
    )

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "branch": branch,
        "commit_sha": commit_sha,
        "local_vps_sync_status": deployment.get("status"),
        "dirty_clean_status": _dirty_clean_status(deployment),
        "target_market_slug": target_market_slug or gate.get("target_market_slug") or market_microstructure.get("market_slug"),
        "max_live_risk_usdc": _round(max_live_risk_usdc if max_live_risk_usdc is not None else gate.get("max_live_risk_usdc")),
        "available_usdc": _round(deposit_wallet.get("available_usdc")),
        "all_12_readiness_assertions": assertions,
        "readiness_assertion_summary": {
            "gate_status": gate.get("status"),
            "asserts_passed": asserts_passed,
            "asserts_failed": asserts_failed,
            "blockers": gate_blockers,
        },
        "PRELIVE_READY": prelive_ready,
        "EXECUTION_AUTHORIZED": execution_authorized,
        "CAN_SUBMIT_ORDER": can_submit_order,
        "LIVE_ORDER_SENT": live_order_sent,
        "can_submit_order": can_submit_order,
        "live_order_sent": live_order_sent,
        "execution_mode": RECOMMENDED_NEXT_MODE,
        "order_side_selected": order_side,
        "quote_price": _round(selected_quote_price),
        "quote_bid": _round(quote_bid),
        "quote_ask": _round(quote_ask),
        "quote_size": _round(quote_size),
        "tick_size_validation_result": _assertion_result(gate, "TICK_SIZE_PRICE_ASSERT"),
        "reward_scoring_validation_result": _assertion_result(gate, "REWARD_SCORING_ASSERT"),
        "heartbeat_latency_ms": _round(network.get("latency_ms") or network.get("api_latency_ms")),
        "mutex_state": order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status"),
        "cancel_replace_rehearsal_state": "READY_BUT_NOT_AUTHORIZED_NO_REAL_CANCEL_REPLACE_SENT",
        "fee_reconciliation_state": fee_reconciliation.get("status"),
        "inventory_state": inventory_state.get("status"),
        "toxic_flow_adverse_selection_state": toxic_flow.get("status"),
        "estimated_net_profit_usdc": _round(estimated_net_profit_usdc),
        "realized_cash_pnl_usdc": realized_cash_pnl_usdc,
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
        "pending_reward_usdc": _round(pending_reward_usdc),
        "confirmed_reward_usdc": confirmed_reward_usdc,
        "pending_reward_counted_as_confirmed_reward": False,
        "approval_boundary": {
            "prelive_ready_only": prelive_ready,
            "execution_authorized": execution_authorized,
            "can_submit_order": can_submit_order,
            "live_order_sent": live_order_sent,
            "rule": "12/12 permits final review. It does not authorize execution.",
        },
        "one_time_authorization": {
            "report_present": bool(probe_authorization),
            "status": probe_authorization.get("status"),
            "token_valid": probe_authorization.get("authorization_token_valid") is True,
            "execution_release_ready": probe_authorization.get("execution_release_ready") is True,
            "token_status": probe_authorization.get("token_status"),
            "ttl_remaining_seconds": probe_authorization.get("ttl_remaining_seconds"),
            "blockers": probe_authorization.get("blockers") or ["AUTHORIZATION_REPORT_NOT_PROVIDED"],
            "execution_authorized_here": False,
            "can_submit_order_here": False,
        },
        "final_decision": {
            "recommended_next_mode": RECOMMENDED_NEXT_MODE,
            "rejected_next_mode": REJECTED_NEXT_MODE,
            "reason": (
                "First real execution cold-start should isolate one side only; "
                "two-sided live exposes inventory and concurrency risks too early."
            ),
        },
        "future_probe_success_criteria": {
            "primary_success": (
                "One BID order is submitted under manual approval, server acknowledges it, "
                "mutex transitions through in-flight/open state, cancel/replace rehearsal can return to NO_ORDER, "
                "and post-probe reports show no untracked inventory or open-order residue."
            ),
            "primary_success_requires_fill": False,
            "secondary_success": (
                "If a partial or full fill occurs, inventory, fee, and cash/reward separation reports reconcile "
                "the exact filled quantity without counting estimates as realized PnL."
            ),
            "failure_conditions": [
                "order remains open without controlled cancel state",
                "mutex remains stuck outside NO_ORDER after shutdown",
                "inventory_state is not clear or reconciled",
                "fee reconciliation becomes negative or unknown",
                "toxic-flow detector blocks the market before/during probe",
                "any live_order_sent value appears without explicit approval artifact",
            ],
        },
        "status": status,
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, prelive_ready, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Maker Single-Side Live Rehearsal Review",
        "",
        "## Boundary",
        f"- PRELIVE_READY: {report.get('PRELIVE_READY')}",
        f"- EXECUTION_AUTHORIZED: {report.get('EXECUTION_AUTHORIZED')}",
        f"- CAN_SUBMIT_ORDER: {report.get('CAN_SUBMIT_ORDER')}",
        f"- LIVE_ORDER_SENT: {report.get('LIVE_ORDER_SENT')}",
        "",
        "## Target",
        f"- Market: {report.get('target_market_slug')}",
        f"- Mode: {report.get('execution_mode')}",
        f"- Side: {report.get('order_side_selected')}",
        f"- Quote price: {report.get('quote_price')}",
        f"- Quote size: {report.get('quote_size')}",
        "",
        "## Readiness",
        f"- Gate status: {(report.get('readiness_assertion_summary') or {}).get('gate_status')}",
        f"- Assertions: {(report.get('readiness_assertion_summary') or {}).get('asserts_passed')}/12",
        f"- Heartbeat latency ms: {report.get('heartbeat_latency_ms')}",
        f"- Mutex state: {report.get('mutex_state')}",
        f"- Fee state: {report.get('fee_reconciliation_state')}",
        f"- Inventory state: {report.get('inventory_state')}",
        f"- Toxic flow state: {report.get('toxic_flow_adverse_selection_state')}",
        "",
        "## Success Criteria",
        f"- Primary: {(report.get('future_probe_success_criteria') or {}).get('primary_success')}",
        f"- Requires fill: {(report.get('future_probe_success_criteria') or {}).get('primary_success_requires_fill')}",
        f"- Secondary: {(report.get('future_probe_success_criteria') or {}).get('secondary_success')}",
        "",
        "## Final Decision",
        f"- Recommended: {(report.get('final_decision') or {}).get('recommended_next_mode')}",
        f"- Rejected: {(report.get('final_decision') or {}).get('rejected_next_mode')}",
        f"- Reason: {(report.get('final_decision') or {}).get('reason')}",
        "",
        "## One-Time Authorization",
        f"- Status: {(report.get('one_time_authorization') or {}).get('status')}",
        f"- Token valid: {(report.get('one_time_authorization') or {}).get('token_valid')}",
        f"- Execution authorized here: {(report.get('one_time_authorization') or {}).get('execution_authorized_here')}",
        "",
        "## Blockers",
    ]
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _assertion_result(gate: dict[str, Any], assert_id: str) -> dict[str, Any]:
    by_id = gate.get("asserts_by_id") if isinstance(gate.get("asserts_by_id"), dict) else {}
    item = by_id.get(assert_id) if isinstance(by_id.get(assert_id), dict) else {}
    return {
        "assert_id": assert_id,
        "passed": item.get("passed") is True,
        "reason": item.get("reason"),
        "details": item.get("details") if isinstance(item.get("details"), dict) else {},
    }


def _dirty_clean_status(deployment: dict[str, Any]) -> str:
    if deployment.get("unreviewed_changes_present") is True or deployment.get("local_dirty") is True or deployment.get("remote_dirty") is True:
        return "DIRTY"
    if deployment.get("status") == "DEPLOYMENT_SYNC_OK":
        return "CLEAN"
    return "UNKNOWN"


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


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


def _one_line_verdict(status: str, prelive_ready: bool, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_READY: 12/12 prelive only; execution remains unauthorized."
    ready_text = "prelive_ready=true" if prelive_ready else "prelive_ready=false"
    return f"SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_BLOCKED: {ready_text}; {', '.join(blockers) or 'UNKNOWN'}."
