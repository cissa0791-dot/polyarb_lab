from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "c_fill_reconciliation_audit.v1"
REPORT_TYPE = "c_fill_reconciliation_audit"

READY_STATUS = "C_FILL_RECONCILIATION_AUDIT_READY"
BLOCKED_STATUS = "C_FILL_RECONCILIATION_AUDIT_BLOCKED"

ZERO_FILL = "ZERO_FILL"
PARTIAL_FILL_RECONCILED = "PARTIAL_FILL_RECONCILED"
FULL_FILL_RECONCILED = "FULL_FILL_RECONCILED"
FILL_DETECTED_RECONCILIATION_BLOCKED = "FILL_DETECTED_RECONCILIATION_BLOCKED"
CANCEL_REMAINDER_FAILED = "CANCEL_REMAINDER_FAILED"
EMERGENCY_REVIEW_REQUIRED = "EMERGENCY_REVIEW_REQUIRED"


def build_c_fill_reconciliation_audit(
    *,
    probe: dict[str, Any] | None = None,
    order_reconciliation: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    deposit_wallet: dict[str, Any] | None = None,
    fee_reconciliation: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    token: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify the result of a future C-class fill/reconciliation probe.

    This is read-only audit logic. It accepts evidence produced elsewhere and
    proves whether zero fill, partial fill, or full fill is reconciled. It does
    not execute, retry, create tokens, or authorize follow-up orders.
    """

    now = now or datetime.now(timezone.utc)
    probe = probe or {}
    order_reconciliation = order_reconciliation or {}
    inventory_state = inventory_state or {}
    deposit_wallet = deposit_wallet or {}
    fee_reconciliation = fee_reconciliation or {}
    order_mutex = order_mutex or {}
    token = token or {}
    gate = gate or {}

    submit = probe.get("submit_result") if isinstance(probe.get("submit_result"), dict) else {}
    cancel = probe.get("cancel_result") if isinstance(probe.get("cancel_result"), dict) else {}
    hold = probe.get("hold_observation") if isinstance(probe.get("hold_observation"), dict) else {}
    polls = hold.get("status_polls") if isinstance(hold.get("status_polls"), list) else []
    order_id = submit.get("order_id") or cancel.get("order_id") or order_reconciliation.get("order_id")

    local_fill_qty = _max_size_matched(submit, polls)
    raw_fill_qty = _first_float(
        order_reconciliation.get("size_matched"),
        order_reconciliation.get("matched_size"),
        order_reconciliation.get("sizeMatched"),
    )
    original_size = _first_float(
        order_reconciliation.get("original_size"),
        order_reconciliation.get("size"),
        submit.get("quote_size"),
        (probe.get("target") or {}).get("quote_size") if isinstance(probe.get("target"), dict) else None,
    )
    raw_remaining_qty = _first_float(
        order_reconciliation.get("size_remaining"),
        order_reconciliation.get("remaining_size"),
        order_reconciliation.get("sizeRemaining"),
    )
    if raw_remaining_qty is None and original_size is not None and raw_fill_qty is not None:
        raw_remaining_qty = max(0.0, original_size - raw_fill_qty)

    observed_fill_qty = raw_fill_qty if raw_fill_qty is not None else local_fill_qty
    expected_inventory_delta = observed_fill_qty
    inventory_delta = _first_float(
        inventory_state.get("inventory_delta_shares"),
        inventory_state.get("token_balance_delta_shares"),
        inventory_state.get("token_balance_shares"),
    )
    cash_delta = _first_float(
        deposit_wallet.get("cash_delta_usdc"),
        deposit_wallet.get("available_usdc_delta"),
        deposit_wallet.get("deposit_wallet_cash_delta_usdc"),
    )
    expected_cash_delta = _first_float(
        deposit_wallet.get("expected_cash_delta_usdc"),
        probe.get("expected_cash_delta_usdc"),
    )
    token_status = token.get("status")

    checks = {
        "order_id_matched": str(order_reconciliation.get("order_id") or "") == str(order_id or ""),
        "raw_order_fill_quantity_present": raw_fill_qty is not None,
        "local_observed_fill_quantity_present": local_fill_qty is not None,
        "fill_quantities_consistent": _same_float(raw_fill_qty, local_fill_qty),
        "inventory_delta_present": inventory_delta is not None,
        "inventory_delta_consistent": _same_float(inventory_delta, expected_inventory_delta),
        "cash_delta_present": cash_delta is not None,
        "cash_delta_consistent": expected_cash_delta is not None and _same_float(cash_delta, expected_cash_delta, tolerance=1e-6),
        "fee_reconciliation_ready": fee_reconciliation.get("status") == "FEE_RECONCILIATION_READY"
        and fee_reconciliation.get("can_cover_fees") is True,
        "open_order_count_zero": _same_float(inventory_state.get("open_order_count"), 0.0)
        and _same_float(inventory_state.get("token_open_order_count"), 0.0),
        "cancel_confirmed": cancel.get("cancel_confirmed_not_open") is True,
        "order_mutex_clear": (order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status")) == "NO_ORDER",
        "token_expended": token_status == "EXPENDED",
        "token_non_reusable": token_status == "EXPENDED",
        "can_submit_order_false": probe.get("can_submit_order") is False and gate.get("can_submit_order") is False,
        "live_order_sent_disabled_after_probe": gate.get("live_order_sent") is False,
        "single_order_only": _first_int(probe.get("submitted_order_count"), default=1) <= 1,
    }

    blockers = _base_blockers(checks=checks, probe=probe, order_reconciliation=order_reconciliation)
    classification = _classification(
        blockers=blockers,
        observed_fill_qty=observed_fill_qty,
        original_size=original_size,
        raw_remaining_qty=raw_remaining_qty,
        cancel_confirmed=checks["cancel_confirmed"],
        open_order_count_zero=checks["open_order_count_zero"],
    )
    if classification in {FILL_DETECTED_RECONCILIATION_BLOCKED, CANCEL_REMAINDER_FAILED} and classification not in blockers:
        blockers.append(classification)
    status = READY_STATUS if not blockers else BLOCKED_STATUS

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "classification": classification,
        "emergency_review_status": EMERGENCY_REVIEW_REQUIRED
        if classification in {CANCEL_REMAINDER_FAILED, FILL_DETECTED_RECONCILIATION_BLOCKED}
        else None,
        "order_id": order_id,
        "fill_evidence": {
            "raw_order_fill_quantity": _round(raw_fill_qty),
            "local_observed_fill_quantity": _round(local_fill_qty),
            "original_size": _round(original_size),
            "remaining_quantity": _round(raw_remaining_qty),
            "partial_fill": bool(observed_fill_qty and raw_remaining_qty and observed_fill_qty > 0 and raw_remaining_qty > 0),
            "full_fill": bool(
                observed_fill_qty
                and observed_fill_qty > 0
                and (
                    (raw_remaining_qty is not None and raw_remaining_qty <= 1e-9)
                    or (original_size is not None and abs(observed_fill_qty - original_size) <= 1e-9)
                )
            ),
        },
        "reconciliation": {
            "inventory_delta": _round(inventory_delta),
            "expected_inventory_delta": _round(expected_inventory_delta),
            "cash_delta_usdc": _round(cash_delta),
            "expected_cash_delta_usdc": _round(expected_cash_delta),
            "fee_status": fee_reconciliation.get("status"),
            "fee_actual_status": fee_reconciliation.get("actual_fee_status"),
            "open_order_count_zero_after_cancel": checks["open_order_count_zero"],
            "cancel_confirmed": checks["cancel_confirmed"],
        },
        "checks": checks,
        "approval_boundary": {
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "no_second_order_under_same_token": checks["single_order_only"],
        },
        "blockers": _unique(blockers),
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "profitability_claimed": False,
        "one_line_verdict": _one_line_verdict(status, classification, blockers),
    }


def _base_blockers(
    *,
    checks: dict[str, bool],
    probe: dict[str, Any],
    order_reconciliation: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not probe:
        blockers.append("PROBE_REPORT_MISSING")
    if not order_reconciliation:
        blockers.append("ORDER_RECONCILIATION_REPORT_MISSING")
    for key, blocker in {
        "order_id_matched": "ORDER_ID_MISMATCH",
        "raw_order_fill_quantity_present": "RAW_ORDER_FILL_QUANTITY_MISSING",
        "local_observed_fill_quantity_present": "LOCAL_FILL_QUANTITY_MISSING",
        "fill_quantities_consistent": "RAW_LOCAL_FILL_QUANTITY_MISMATCH",
        "inventory_delta_present": "INVENTORY_DELTA_MISSING",
        "inventory_delta_consistent": "INVENTORY_DELTA_MISMATCH",
        "cash_delta_present": "CASH_DELTA_MISSING",
        "cash_delta_consistent": "CASH_DELTA_MISMATCH",
        "fee_reconciliation_ready": "FEE_RECONCILIATION_NOT_READY",
        "open_order_count_zero": "OPEN_ORDER_COUNT_NOT_ZERO",
        "order_mutex_clear": "ORDER_MUTEX_NOT_CLEAR",
        "token_expended": "TOKEN_NOT_EXPENDED",
        "token_non_reusable": "TOKEN_REUSE_NOT_BLOCKED",
        "can_submit_order_false": "CAN_SUBMIT_ORDER_NOT_FALSE",
        "live_order_sent_disabled_after_probe": "LIVE_ORDER_SENT_NOT_RESET_IN_GATE",
        "single_order_only": "SECOND_ORDER_CREATED_UNDER_SAME_TOKEN",
    }.items():
        if not checks.get(key):
            blockers.append(blocker)
    return _unique(blockers)


def _classification(
    *,
    blockers: list[str],
    observed_fill_qty: float | None,
    original_size: float | None,
    raw_remaining_qty: float | None,
    cancel_confirmed: bool,
    open_order_count_zero: bool,
) -> str:
    fill_qty = observed_fill_qty or 0.0
    remaining = raw_remaining_qty
    if fill_qty <= 1e-9 and not blockers:
        return ZERO_FILL
    hard_reconciliation_blockers = {
        "RAW_ORDER_FILL_QUANTITY_MISSING",
        "LOCAL_FILL_QUANTITY_MISSING",
        "RAW_LOCAL_FILL_QUANTITY_MISMATCH",
        "INVENTORY_DELTA_MISSING",
        "INVENTORY_DELTA_MISMATCH",
        "CASH_DELTA_MISSING",
        "CASH_DELTA_MISMATCH",
        "FEE_RECONCILIATION_NOT_READY",
    }
    if fill_qty > 1e-9 and any(blocker in hard_reconciliation_blockers for blocker in blockers):
        return FILL_DETECTED_RECONCILIATION_BLOCKED
    if fill_qty > 1e-9 and (remaining is not None and remaining > 1e-9) and not (cancel_confirmed and open_order_count_zero):
        return CANCEL_REMAINDER_FAILED
    fill_blockers = [
        blocker
        for blocker in blockers
        if blocker
        not in {
            "CAN_SUBMIT_ORDER_NOT_FALSE",
            "LIVE_ORDER_SENT_NOT_RESET_IN_GATE",
        }
    ]
    if fill_qty > 1e-9 and fill_blockers:
        return FILL_DETECTED_RECONCILIATION_BLOCKED
    if fill_qty > 1e-9 and (
        (remaining is not None and remaining > 1e-9)
        or (original_size is not None and fill_qty < original_size - 1e-9)
    ):
        return PARTIAL_FILL_RECONCILED
    if fill_qty > 1e-9:
        return FULL_FILL_RECONCILED
    return FILL_DETECTED_RECONCILIATION_BLOCKED if blockers else ZERO_FILL


def _max_size_matched(submit: dict[str, Any], polls: list[Any]) -> float | None:
    values: list[float] = []
    parsed = _first_float(submit.get("size_matched"))
    if parsed is not None:
        values.append(parsed)
    for row in polls:
        if not isinstance(row, dict):
            continue
        parsed = _first_float(row.get("size_matched"))
        if parsed is not None:
            values.append(parsed)
    return max(values) if values else None


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


def _first_int(value: Any, *, default: int | None = None) -> int | None:
    parsed = _first_float(value)
    return int(parsed) if parsed is not None else default


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


def _one_line_verdict(status: str, classification: str, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return f"{classification}: C-class fill reconciliation evidence is internally consistent; execution remains disabled."
    return f"C_FILL_RECONCILIATION_AUDIT_BLOCKED: {classification}; {', '.join(blockers) or 'UNKNOWN'}."
