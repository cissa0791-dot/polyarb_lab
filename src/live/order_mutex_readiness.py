from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REPORT_SCHEMA_VERSION = "order_mutex_readiness.v1"
REPORT_TYPE = "order_mutex_readiness"

NO_ORDER = "NO_ORDER"
PLACE_IN_FLIGHT = "PLACE_IN_FLIGHT"
LIVE_ORDER_OPEN = "LIVE_ORDER_OPEN"
CANCEL_IN_FLIGHT = "CANCEL_IN_FLIGHT"
UNKNOWN = "UNKNOWN"
VALID_GATE_STATES = {NO_ORDER, PLACE_IN_FLIGHT, LIVE_ORDER_OPEN, CANCEL_IN_FLIGHT}


def build_order_mutex_readiness_report(
    *,
    health: dict[str, Any] | None = None,
    execution_system: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    shadow_latest: dict[str, Any] | None = None,
    explicit_state: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a live-scope order mutex report.

    This report intentionally separates live order mutex truth from shadow
    virtual quotes. Shadow quotes can be open for simulation while live mutex
    remains NO_ORDER when no real order/open/in-flight state exists.
    """

    now = now or datetime.now(timezone.utc)
    health = health or {}
    execution_system = execution_system or {}
    inventory_state = inventory_state or {}
    shadow_latest = shadow_latest or {}
    open_order_count = _open_order_count(health, inventory_state)
    live_order_sent = execution_system.get("live_order_sent") is True or health.get("live_order_sent") is True
    execution_enabled = execution_system.get("execution_enabled") is True
    live_actions_enabled = execution_system.get("live_actions_enabled") is True
    upstream_can_submit_order = execution_system.get("can_submit_order") is True
    source_state = _normalise_state(explicit_state)
    blockers: list[str] = []

    if source_state:
        state = source_state
        state_source = "EXPLICIT_STATE"
    elif open_order_count is not None and open_order_count > 0:
        state = LIVE_ORDER_OPEN
        state_source = "LIVE_OPEN_ORDERS"
    elif live_order_sent or execution_enabled or live_actions_enabled or upstream_can_submit_order:
        state = LIVE_ORDER_OPEN
        state_source = "UPSTREAM_EXECUTION_SYSTEM"
    elif open_order_count == 0:
        state = NO_ORDER
        state_source = "LIVE_ACCOUNT_OPEN_ORDER_COUNT_ZERO"
    else:
        state = UNKNOWN
        state_source = "LIVE_ORDER_MUTEX_SOURCE_MISSING"

    if state == UNKNOWN:
        blockers.append("ORDER_MUTEX_SOURCE_MISSING")
    elif state != NO_ORDER:
        blockers.append("ORDER_MUTEX_NOT_CLEAR")
    if state not in VALID_GATE_STATES:
        blockers.append("ORDER_MUTEX_STATE_INVALID")

    status = "ORDER_MUTEX_READY" if not blockers else "ORDER_MUTEX_BLOCKED"
    shadow_state = _normalise_state(shadow_latest.get("order_mutex_state")) or shadow_latest.get("order_mutex_state")
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "execution_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "status": status,
        "order_mutex_state": state,
        "live_order_status": state,
        "state_source": state_source,
        "scope": "LIVE_ORDER_MUTEX_ONLY",
        "shadow_order_mutex_state": shadow_state,
        "shadow_state_ignored_for_live_gate": bool(shadow_state),
        "open_order_count": open_order_count,
        "upstream_execution_enabled": execution_enabled,
        "upstream_live_actions_enabled": live_actions_enabled,
        "upstream_can_submit_order": upstream_can_submit_order,
        "upstream_live_order_sent": live_order_sent,
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, state, _unique(blockers)),
    }


def _open_order_count(health: dict[str, Any], inventory_state: dict[str, Any] | None = None) -> int | None:
    inventory_state = inventory_state or {}
    checks = health.get("checks") if isinstance(health.get("checks"), dict) else {}
    account_orders = checks.get("account_open_orders") if isinstance(checks.get("account_open_orders"), dict) else {}
    target_state = checks.get("target_account_state") if isinstance(checks.get("target_account_state"), dict) else {}
    for value in (
        inventory_state.get("open_order_count"),
        inventory_state.get("token_open_order_count"),
        account_orders.get("open_order_count"),
        target_state.get("token_open_order_count"),
        health.get("open_order_count"),
    ):
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None


def _normalise_state(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip().upper()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
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


def _one_line_verdict(status: str, state: str, blockers: list[str]) -> str:
    if status == "ORDER_MUTEX_READY":
        return "ORDER_MUTEX_READY: live order mutex is NO_ORDER; can_submit_order=false."
    return f"ORDER_MUTEX_BLOCKED: state={state}; {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
