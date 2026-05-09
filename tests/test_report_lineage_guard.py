from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_report_lineage_guard_report import main
from src.live.report_lineage_guard import BLOCKED_STATUS, READY_STATUS, build_report_lineage_guard


NOW = datetime(2026, 5, 9, 16, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
ORDER_ID = "0xabc"


def _probe(**overrides) -> dict:
    payload = {
        "status": "SINGLE_SIDE_BID_PROBE_COMPLETED",
        "generated_at_utc": NOW.isoformat(),
        "target": {"market_slug": MARKET},
        "submit_result": {"order_id": ORDER_ID},
        "cancel_result": {"order_id": ORDER_ID, "cancel_response_at_utc": NOW.isoformat()},
        "can_submit_order": False,
        "live_order_sent": True,
    }
    payload.update(overrides)
    return payload


def _recon(**overrides) -> dict:
    payload = {"status": "ORDER_STATUS_RECONCILIATION_READY", "generated_at_utc": NOW.isoformat(), "order_id": ORDER_ID}
    payload.update(overrides)
    return payload


def _market(**overrides) -> dict:
    payload = {"status": "MARKET_MICROSTRUCTURE_READY", "generated_at_utc": NOW.isoformat(), "market_slug": MARKET}
    payload.update(overrides)
    return payload


def _planner(**overrides) -> dict:
    payload = {
        "status": "LIVE_PROBE_PLAN_READY",
        "planner_snapshot_ts": NOW.isoformat(),
        "recommended_plan": {"market_slug": MARKET},
    }
    payload.update(overrides)
    return payload


def _fee(**overrides) -> dict:
    payload = {"status": "FEE_RECONCILIATION_READY", "generated_at_utc": NOW.isoformat(), "market_slug": MARKET}
    payload.update(overrides)
    return payload


def _toxic(**overrides) -> dict:
    payload = {"status": "TOXIC_FLOW_READY", "generated_at_utc": NOW.isoformat(), "market_slug": MARKET}
    payload.update(overrides)
    return payload


def _generic(**overrides) -> dict:
    payload = {"status": "READY", "generated_at_utc": NOW.isoformat()}
    payload.update(overrides)
    return payload


def _guard(**overrides) -> dict:
    payload = {
        "probe": _probe(),
        "order_reconciliation": _recon(),
        "market_microstructure": _market(),
        "planner": _planner(),
        "fee_reconciliation": _fee(),
        "toxic_flow": _toxic(),
        "inventory_state": _generic(),
        "order_mutex": _generic(),
        "gate": _generic(target_market_slug=MARKET),
        "deposit_wallet": _generic(),
        "heartbeat": _generic(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_report_lineage_guard(**payload)


def test_matching_lineage_is_ready_and_exposes_order_market_and_timestamps() -> None:
    report = _guard()

    assert report["status"] == READY_STATUS
    assert report["lineage_status"] == READY_STATUS
    assert report["lineage_blockers"] == []
    assert report["matched_order_id"] == ORDER_ID
    assert report["matched_market_slug"] == MARKET
    assert report["source_report_timestamps"]["fee_reconciliation"] == NOW.isoformat()
    assert report["can_submit_order"] is False


def test_stale_reconciliation_order_id_blocks() -> None:
    report = _guard(order_reconciliation=_recon(order_id="0xold"))

    assert report["status"] == BLOCKED_STATUS
    assert "LINEAGE_ORDER_ID_MISMATCH" in report["lineage_blockers"]


def test_wrong_market_blocks() -> None:
    report = _guard(market_microstructure=_market(market_slug="wrong-market"))

    assert report["status"] == BLOCKED_STATUS
    assert "LINEAGE_MARKET_MICROSTRUCTURE_MARKET_MISMATCH" in report["lineage_blockers"]


def test_missing_required_report_blocks() -> None:
    report = _guard(toxic_flow={})

    assert report["status"] == BLOCKED_STATUS
    assert "LINEAGE_TOXIC_FLOW_REPORT_MISSING" in report["lineage_blockers"]


def test_stale_latest_json_blocks() -> None:
    old = datetime(2026, 5, 9, 15, 50, tzinfo=timezone.utc).isoformat()
    report = _guard(fee_reconciliation=_fee(generated_at_utc=old))

    assert report["status"] == BLOCKED_STATUS
    assert "LINEAGE_FEE_RECONCILIATION_STALE_BEFORE_PROBE_COMPLETION" in report["lineage_blockers"]


def test_cli_writes_lineage_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "single_side_bid_probe_latest.json": _probe(),
        "order_status_reconciliation_latest.json": _recon(),
        "live_market_microstructure_latest.json": _market(),
        "live_probe_planner_latest.json": _planner(),
        "fee_reconciliation_latest.json": _fee(),
        "toxic_flow_latest.json": _toxic(),
        "inventory_state_latest.json": _generic(),
        "order_mutex_readiness_latest.json": _generic(),
        "live_readiness_gate_latest.json": _generic(target_market_slug=MARKET),
        "deposit_wallet_readonly_latest.json": _generic(),
        "live_network_readiness_latest.json": _generic(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "lineage.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out), "--max-report-age-minutes", "1000"])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["matched_order_id"] == ORDER_ID
