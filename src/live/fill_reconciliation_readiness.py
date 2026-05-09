from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "fill_reconciliation_readiness.v1"
REPORT_TYPE = "fill_reconciliation_readiness"

READY_STATUS = "FILL_RECONCILIATION_READINESS_READY"
BLOCKED_STATUS = "FILL_RECONCILIATION_READINESS_BLOCKED"


def build_fill_reconciliation_readiness(
    *,
    order_status: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    deposit_wallet: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    fee_reconciliation: dict[str, Any] | None = None,
    previous_probe: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only readiness package for future fill reconciliation.

    This is a pre-C-probe governance artifact. It proves whether the system has
    enough read sources to detect fills, partial fills, remaining open order
    quantity, inventory deltas, cash deltas, and fee state. It never creates a
    token and never authorizes execution.
    """

    now = now or datetime.now(timezone.utc)
    order_status = order_status or {}
    inventory_state = inventory_state or {}
    deposit_wallet = deposit_wallet or {}
    order_mutex = order_mutex or {}
    fee_reconciliation = fee_reconciliation or {}
    previous_probe = previous_probe or {}

    size_matched = _first_float(
        order_status.get("size_matched"),
        order_status.get("matched_size"),
        order_status.get("sizeMatched"),
    )
    size_remaining = _first_float(
        order_status.get("size_remaining"),
        order_status.get("remaining_size"),
        order_status.get("sizeRemaining"),
    )
    original_size = _first_float(order_status.get("original_size"), order_status.get("size"))
    if original_size is None and size_matched is not None and size_remaining is not None:
        original_size = size_matched + size_remaining

    order_status_source_ready = bool(order_status) and (
        order_status.get("status") == "ORDER_STATUS_RECONCILIATION_READY"
        or bool(order_status.get("raw_order_status") or order_status.get("order_status") or order_status.get("status"))
    )
    fill_detectable = order_status_source_ready and size_matched is not None
    partial_fill_detectable = fill_detectable and (
        size_remaining is not None or (original_size is not None and size_matched is not None)
    )
    partial_fill_observed = (
        size_matched is not None
        and size_remaining is not None
        and size_matched > 0.0
        and size_remaining > 0.0
    )
    full_fill_observed = (
        size_matched is not None
        and size_matched > 0.0
        and (
            (size_remaining is not None and size_remaining <= 1e-9)
            or (original_size is not None and abs(size_matched - original_size) <= 1e-9)
        )
    )
    remaining_order_cancel_required = bool(partial_fill_observed or _raw_status(order_status) in {"LIVE", "OPEN", "ACTIVE"})

    inventory_update_source_ready = (
        inventory_state.get("status") in {"INVENTORY_STATE_CLEAR", "INVENTORY_STATE_OPEN", "INVENTORY_STATE_BLOCKED"}
        and _first_float(inventory_state.get("token_balance_shares")) is not None
    )
    cash_delta_source_ready = (
        deposit_wallet.get("status") == "DEPOSIT_WALLET_READY"
        and _first_float(deposit_wallet.get("available_usdc")) is not None
    )
    fee_ready = (
        fee_reconciliation.get("status") == "FEE_RECONCILIATION_READY"
        and fee_reconciliation.get("can_cover_fees") is True
    )
    mutex_observable = bool(order_mutex) and (order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status")) is not None
    previous_probe_available = bool(previous_probe)

    checks = {
        "order_status_source_ready": order_status_source_ready,
        "fill_detectable": fill_detectable,
        "partial_fill_detectable": partial_fill_detectable,
        "inventory_update_source_ready": inventory_update_source_ready,
        "cash_delta_source_ready": cash_delta_source_ready,
        "fee_reconciliation_ready": fee_ready,
        "order_mutex_observable": mutex_observable,
        "previous_probe_available": previous_probe_available,
    }
    blockers: list[str] = []
    if not order_status_source_ready:
        blockers.append("ORDER_STATUS_SOURCE_MISSING")
    if not fill_detectable:
        blockers.append("FILL_DETECTION_FIELDS_MISSING")
    if not partial_fill_detectable:
        blockers.append("PARTIAL_FILL_DETECTION_FIELDS_MISSING")
    if not inventory_update_source_ready:
        blockers.append("INVENTORY_UPDATE_SOURCE_MISSING")
    if not cash_delta_source_ready:
        blockers.append("CASH_DELTA_SOURCE_MISSING")
    if not fee_ready:
        blockers.append("FEE_RECONCILIATION_NOT_READY")
    if not mutex_observable:
        blockers.append("ORDER_MUTEX_SOURCE_MISSING")

    blockers = _unique(blockers)
    status = READY_STATUS if not blockers else BLOCKED_STATUS
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "checks": checks,
        "fill_detectable": fill_detectable,
        "partial_fill_detectable": partial_fill_detectable,
        "remaining_order_cancel_required": remaining_order_cancel_required,
        "inventory_update_source_ready": inventory_update_source_ready,
        "cash_delta_source_ready": cash_delta_source_ready,
        "fee_reconciliation_ready": fee_ready,
        "post_fill_audit_required": True,
        "partial_fill_observed": partial_fill_observed,
        "full_fill_observed": full_fill_observed,
        "order_state": {
            "order_id": order_status.get("order_id"),
            "raw_order_status": order_status.get("raw_order_status") or order_status.get("order_status"),
            "size_matched": _round(size_matched),
            "size_remaining": _round(size_remaining),
            "original_size": _round(original_size),
        },
        "previous_probe_status": previous_probe.get("status"),
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "profitability_claimed": False,
        "estimated_reward_counted_as_realized_pnl": False,
        "blockers": blockers,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def _raw_status(payload: dict[str, Any]) -> str:
    return str(payload.get("raw_order_status") or payload.get("order_status") or payload.get("status") or "").upper()


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(parsed):
            return parsed
    return None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


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
        return "FILL_RECONCILIATION_READINESS_READY: fill, partial-fill, inventory, cash, and fee read sources are available; can_submit_order=false."
    return f"FILL_RECONCILIATION_READINESS_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
