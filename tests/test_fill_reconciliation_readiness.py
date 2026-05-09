from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_fill_reconciliation_readiness_report import main
from src.live.fill_reconciliation_readiness import (
    BLOCKED_STATUS,
    READY_STATUS,
    build_fill_reconciliation_readiness,
)


NOW = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
ORDER_ID = "0xabc"


def _order(**overrides) -> dict:
    payload = {
        "status": "ORDER_STATUS_RECONCILIATION_READY",
        "order_id": ORDER_ID,
        "raw_order_status": "LIVE",
        "size_matched": 10.0,
        "size_remaining": 40.0,
        "original_size": 50.0,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _inventory(**overrides) -> dict:
    payload = {
        "status": "INVENTORY_STATE_OPEN",
        "token_balance_shares": 10.0,
        "open_order_count": 1,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _deposit(**overrides) -> dict:
    payload = {
        "status": "DEPOSIT_WALLET_READY",
        "available_usdc": 302.5,
        "read_only": True,
    }
    payload.update(overrides)
    return payload


def _mutex(**overrides) -> dict:
    payload = {
        "status": "ORDER_MUTEX_READY",
        "order_mutex_state": "LIVE_ORDER_OPEN",
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _fee(**overrides) -> dict:
    payload = {
        "status": "FEE_RECONCILIATION_READY",
        "can_cover_fees": True,
        "estimated_net_profit_usdc": 0.12,
    }
    payload.update(overrides)
    return payload


def _probe(**overrides) -> dict:
    payload = {
        "status": "POST_LIVE_PROBE_AUDIT_READY",
        "classification": "PARTIAL_FILL_RECONCILED",
        "order_id": ORDER_ID,
    }
    payload.update(overrides)
    return payload


def _report(**overrides) -> dict:
    payload = {
        "order_status": _order(),
        "inventory_state": _inventory(),
        "deposit_wallet": _deposit(),
        "order_mutex": _mutex(),
        "fee_reconciliation": _fee(),
        "previous_probe": _probe(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_fill_reconciliation_readiness(**payload)


def test_partial_fill_readiness_is_ready_with_all_read_sources() -> None:
    report = _report()

    assert report["status"] == READY_STATUS
    assert report["fill_detectable"] is True
    assert report["partial_fill_detectable"] is True
    assert report["partial_fill_observed"] is True
    assert report["remaining_order_cancel_required"] is True
    assert report["inventory_update_source_ready"] is True
    assert report["cash_delta_source_ready"] is True
    assert report["fee_reconciliation_ready"] is True
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["estimated_reward_counted_as_realized_pnl"] is False


def test_missing_inventory_source_blocks() -> None:
    report = _report(inventory_state={})

    assert report["status"] == BLOCKED_STATUS
    assert "INVENTORY_UPDATE_SOURCE_MISSING" in report["blockers"]
    assert report["execution_authorized"] is False


def test_missing_cash_source_blocks() -> None:
    report = _report(deposit_wallet={"status": "DEPOSIT_WALLET_READY"})

    assert report["status"] == BLOCKED_STATUS
    assert "CASH_DELTA_SOURCE_MISSING" in report["blockers"]


def test_missing_order_status_source_blocks() -> None:
    report = _report(order_status={})

    assert report["status"] == BLOCKED_STATUS
    assert "ORDER_STATUS_SOURCE_MISSING" in report["blockers"]
    assert "FILL_DETECTION_FIELDS_MISSING" in report["blockers"]
    assert "PARTIAL_FILL_DETECTION_FIELDS_MISSING" in report["blockers"]


def test_fee_report_must_explicitly_cover_fees() -> None:
    report = _report(fee_reconciliation=_fee(can_cover_fees=False))

    assert report["status"] == BLOCKED_STATUS
    assert "FEE_RECONCILIATION_NOT_READY" in report["blockers"]


def test_cli_writes_readiness_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "order_status_reconciliation_latest.json": _order(),
        "inventory_state_latest.json": _inventory(),
        "deposit_wallet_readonly_latest.json": _deposit(),
        "order_mutex_readiness_latest.json": _mutex(),
        "fee_reconciliation_latest.json": _fee(),
        "single_side_bid_probe_latest.json": _probe(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "fill_readiness.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["can_submit_order"] is False
