from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_c_fill_reconciliation_audit_report import main
from src.live.c_fill_reconciliation_audit import (
    BLOCKED_STATUS,
    FULL_FILL_RECONCILED,
    PARTIAL_FILL_RECONCILED,
    READY_STATUS,
    build_c_fill_reconciliation_audit,
)


NOW = datetime(2026, 5, 9, 13, 30, tzinfo=timezone.utc)
ORDER_ID = "0xabc"


def _probe(size_matched: float = 5.0, **overrides) -> dict:
    payload = {
        "status": "SINGLE_SIDE_BID_PROBE_COMPLETED",
        "submitted_order_count": 1,
        "can_submit_order": False,
        "live_order_sent": True,
        "target": {"quote_size": 10.0},
        "submit_result": {"order_id": ORDER_ID, "size_matched": size_matched},
        "hold_observation": {
            "status_polls": [
                {"order_id": ORDER_ID, "size_matched": size_matched, "size_remaining": max(0.0, 10.0 - size_matched)}
            ]
        },
        "cancel_result": {"order_id": ORDER_ID, "cancel_confirmed_not_open": True},
        "expected_cash_delta_usdc": -2.05,
    }
    payload.update(overrides)
    return payload


def _order(size_matched: float = 5.0, size_remaining: float = 5.0, **overrides) -> dict:
    payload = {
        "status": "ORDER_STATUS_RECONCILIATION_READY",
        "order_id": ORDER_ID,
        "raw_order_status": "CANCELED",
        "size_matched": size_matched,
        "size_remaining": size_remaining,
        "original_size": 10.0,
    }
    payload.update(overrides)
    return payload


def _inventory(delta: float = 5.0, **overrides) -> dict:
    payload = {
        "status": "INVENTORY_STATE_OPEN",
        "open_order_count": 0,
        "token_open_order_count": 0,
        "inventory_delta_shares": delta,
        "token_balance_shares": delta,
    }
    payload.update(overrides)
    return payload


def _deposit(delta: float = -2.05, **overrides) -> dict:
    payload = {
        "status": "DEPOSIT_WALLET_READY",
        "cash_delta_usdc": delta,
        "expected_cash_delta_usdc": delta,
    }
    payload.update(overrides)
    return payload


def _fee(**overrides) -> dict:
    payload = {"status": "FEE_RECONCILIATION_READY", "can_cover_fees": True, "actual_fee_status": "ESTIMATED_ONLY"}
    payload.update(overrides)
    return payload


def _mutex(**overrides) -> dict:
    payload = {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"}
    payload.update(overrides)
    return payload


def _token(**overrides) -> dict:
    payload = {"status": "EXPENDED"}
    payload.update(overrides)
    return payload


def _gate(**overrides) -> dict:
    payload = {"status": "LIVE_READY_APPROVED", "can_submit_order": False, "live_order_sent": False}
    payload.update(overrides)
    return payload


def _audit(**overrides) -> dict:
    payload = {
        "probe": _probe(),
        "order_reconciliation": _order(),
        "inventory_state": _inventory(),
        "deposit_wallet": _deposit(),
        "fee_reconciliation": _fee(),
        "order_mutex": _mutex(),
        "token": _token(),
        "gate": _gate(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_c_fill_reconciliation_audit(**payload)


def test_partial_fill_detected_remainder_cancelled_and_deltas_consistent() -> None:
    report = _audit()

    assert report["status"] == READY_STATUS
    assert report["classification"] == PARTIAL_FILL_RECONCILED
    assert report["fill_evidence"]["raw_order_fill_quantity"] == 5.0
    assert report["reconciliation"]["inventory_delta"] == 5.0
    assert report["reconciliation"]["cash_delta_usdc"] == -2.05
    assert report["approval_boundary"]["same_token_retry_allowed"] is False
    assert report["can_submit_order"] is False


def test_full_fill_detected_without_remainder_is_reconciled() -> None:
    report = _audit(
        probe=_probe(size_matched=10.0, expected_cash_delta_usdc=-4.10),
        order_reconciliation=_order(size_matched=10.0, size_remaining=0.0),
        inventory_state=_inventory(delta=10.0),
        deposit_wallet=_deposit(delta=-4.10),
    )

    assert report["status"] == READY_STATUS
    assert report["classification"] == FULL_FILL_RECONCILED
    assert report["fill_evidence"]["full_fill"] is True


def test_fill_detected_but_inventory_missing_blocks() -> None:
    report = _audit(inventory_state={})

    assert report["status"] == BLOCKED_STATUS
    assert report["classification"] == "FILL_DETECTED_RECONCILIATION_BLOCKED"
    assert "INVENTORY_DELTA_MISSING" in report["blockers"]


def test_fill_detected_but_cash_delta_missing_blocks() -> None:
    report = _audit(deposit_wallet={"status": "DEPOSIT_WALLET_READY"})

    assert report["status"] == BLOCKED_STATUS
    assert report["classification"] == "FILL_DETECTED_RECONCILIATION_BLOCKED"
    assert "CASH_DELTA_MISSING" in report["blockers"]


def test_cancel_remainder_failure_enters_emergency_review() -> None:
    report = _audit(
        probe=_probe(cancel_result={"order_id": ORDER_ID, "cancel_confirmed_not_open": False}),
        inventory_state=_inventory(open_order_count=1, token_open_order_count=1),
    )

    assert report["status"] == BLOCKED_STATUS
    assert report["classification"] == "CANCEL_REMAINDER_FAILED"
    assert "OPEN_ORDER_COUNT_NOT_ZERO" in report["blockers"]


def test_no_second_order_can_be_created_under_same_token() -> None:
    report = _audit(probe=_probe(submitted_order_count=2))

    assert report["status"] == BLOCKED_STATUS
    assert "SECOND_ORDER_CREATED_UNDER_SAME_TOKEN" in report["blockers"]


def test_cli_writes_c_fill_audit(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "single_side_bid_probe_latest.json": _probe(),
        "order_status_reconciliation_latest.json": _order(),
        "inventory_state_latest.json": _inventory(),
        "deposit_wallet_readonly_latest.json": _deposit(),
        "fee_reconciliation_latest.json": _fee(),
        "order_mutex_readiness_latest.json": _mutex(),
        "single_side_probe_authorization_token_latest.json": _token(),
        "live_readiness_gate_latest.json": _gate(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "c_audit.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["classification"] == PARTIAL_FILL_RECONCILED
