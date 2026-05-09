from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "single_side_continuous_rehearsal.v1"
REPORT_TYPE = "single_side_continuous_rehearsal"

READY_STATUS = "SINGLE_SIDE_CONTINUOUS_REHEARSAL_READY"
STOPPED_STATUS = "SINGLE_SIDE_CONTINUOUS_REHEARSAL_STOPPED"
BLOCKED_STATUS = "SINGLE_SIDE_CONTINUOUS_REHEARSAL_BLOCKED"


def build_single_side_continuous_rehearsal(
    *,
    planner: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    fill_audit: dict[str, Any] | None = None,
    lifecycle_summary: dict[str, Any] | None = None,
    stop_file: str | Path | None = None,
    max_open_order: int = 1,
    max_order_count: int = 3,
    current_order_count: int = 0,
    max_position_shares: float = 10.0,
    session_minutes: int = 15,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a fail-closed controller report for limited single-side rehearsal."""

    now = now or datetime.now(timezone.utc)
    planner = planner or {}
    order_mutex = order_mutex or {}
    inventory_state = inventory_state or {}
    fill_audit = fill_audit or {}
    lifecycle_summary = lifecycle_summary or {}
    blockers = _blockers(
        planner=planner,
        order_mutex=order_mutex,
        inventory_state=inventory_state,
        fill_audit=fill_audit,
        lifecycle_summary=lifecycle_summary,
        stop_file=stop_file,
        max_open_order=max_open_order,
        max_order_count=max_order_count,
        current_order_count=current_order_count,
        max_position_shares=max_position_shares,
    )
    stopped = "STOP_FILE_PRESENT" in blockers or "MAX_ORDER_COUNT_REACHED" in blockers
    status = STOPPED_STATUS if stopped else BLOCKED_STATUS if blockers else READY_STATUS
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "mode": "SINGLE_SIDE_CONTINUOUS_REHEARSAL",
        "session_limits": {
            "one_side_only": True,
            "max_open_order": max_open_order,
            "session_minutes": session_minutes,
            "max_order_count": max_order_count,
            "current_order_count": current_order_count,
            "max_position_shares": max_position_shares,
            "full_live_maker": False,
            "both_side_live": False,
        },
        "flow": [
            "planner",
            "quote",
            "token/preflight only if separately approved",
            "place",
            "monitor",
            "cancel/replace",
            "fill handling",
            "reconcile",
            "next quote",
        ],
        "observed_state": {
            "planner_status": planner.get("status"),
            "order_mutex_state": order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status"),
            "open_order_count": inventory_state.get("open_order_count"),
            "token_open_order_count": inventory_state.get("token_open_order_count"),
            "token_balance_shares": inventory_state.get("token_balance_shares"),
            "fill_audit_status": fill_audit.get("status"),
            "fill_audit_classification": fill_audit.get("classification"),
        },
        "accounting_separation": {
            "estimated_reward_counted_as_realized_cash_pnl": False,
            "pending_reward_counted_as_confirmed_reward": False,
            "lifecycle_summary_profitability_claimed": lifecycle_summary.get("profitability_claimed") is True,
        },
        "next_action": "QUOTE_NEXT" if status == READY_STATUS else "STOP_FAIL_CLOSED",
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "blockers": blockers,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def _blockers(
    *,
    planner: dict[str, Any],
    order_mutex: dict[str, Any],
    inventory_state: dict[str, Any],
    fill_audit: dict[str, Any],
    lifecycle_summary: dict[str, Any],
    stop_file: str | Path | None,
    max_open_order: int,
    max_order_count: int,
    current_order_count: int,
    max_position_shares: float,
) -> list[str]:
    blockers: list[str] = []
    if stop_file and Path(stop_file).exists():
        blockers.append("STOP_FILE_PRESENT")
    if current_order_count >= max_order_count:
        blockers.append("MAX_ORDER_COUNT_REACHED")
    if planner.get("status") not in {"FULL_MARKET_PROBE_PLAN_READY", "LIVE_PROBE_PLAN_READY"}:
        blockers.append("PLANNER_NOT_READY")
    if (order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status")) != "NO_ORDER":
        blockers.append("ORDER_MUTEX_LOCKED")
    open_orders = _float(inventory_state.get("open_order_count"))
    token_open_orders = _float(inventory_state.get("token_open_order_count"))
    if open_orders is None or token_open_orders is None:
        blockers.append("OPEN_ORDER_STATE_MISSING")
    elif open_orders > max_open_order or token_open_orders > max_open_order:
        blockers.append("STALE_OR_DUPLICATE_OPEN_ORDER")
    if open_orders and open_orders > 0:
        blockers.append("UNRESOLVED_OPEN_ORDER_BLOCKS_NEXT_QUOTE")
    shares = _float(inventory_state.get("token_balance_shares"))
    if shares is None:
        blockers.append("INVENTORY_STATE_MISSING")
    elif shares > max_position_shares:
        blockers.append("MAX_POSITION_EXCEEDED")
    if shares and shares > 0 and fill_audit.get("status") != "C_FILL_RECONCILIATION_AUDIT_READY":
        blockers.append("UNRECONCILED_INVENTORY_BLOCKS_NEXT_QUOTE")
    if lifecycle_summary.get("cash_accounting", {}).get("estimated_reward_counted_as_realized_cash_pnl") is True:
        blockers.append("REWARD_PNL_ACCOUNTING_MIXED")
    return _unique(blockers)


def _float(value: Any) -> float | None:
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
        return "SINGLE_SIDE_CONTINUOUS_REHEARSAL_READY: one-side loop may plan the next quote only; live execution remains unauthorized."
    if status == STOPPED_STATUS:
        return f"SINGLE_SIDE_CONTINUOUS_REHEARSAL_STOPPED: {', '.join(blockers)}."
    return f"SINGLE_SIDE_CONTINUOUS_REHEARSAL_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
