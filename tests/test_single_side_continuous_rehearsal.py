from __future__ import annotations

import json
from pathlib import Path

from scripts.build_single_side_continuous_rehearsal_report import main
from src.live.single_side_continuous_rehearsal import (
    BLOCKED_STATUS,
    READY_STATUS,
    STOPPED_STATUS,
    build_single_side_continuous_rehearsal,
)


def _planner(**overrides) -> dict:
    payload = {"status": "FULL_MARKET_PROBE_PLAN_READY", "selected_candidate": {"market_slug": "m"}}
    payload.update(overrides)
    return payload


def _mutex(**overrides) -> dict:
    payload = {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"}
    payload.update(overrides)
    return payload


def _inventory(**overrides) -> dict:
    payload = {"status": "INVENTORY_STATE_CLEAR", "open_order_count": 0, "token_open_order_count": 0, "token_balance_shares": 0}
    payload.update(overrides)
    return payload


def _fill_audit(**overrides) -> dict:
    payload = {"status": "C_FILL_RECONCILIATION_AUDIT_READY", "classification": "ZERO_FILL"}
    payload.update(overrides)
    return payload


def _lifecycle(**overrides) -> dict:
    payload = {
        "status": "FILL_LIFECYCLE_SUMMARY_READY",
        "cash_accounting": {"estimated_reward_counted_as_realized_cash_pnl": False},
    }
    payload.update(overrides)
    return payload


def _report(**overrides) -> dict:
    payload = {
        "planner": _planner(),
        "order_mutex": _mutex(),
        "inventory_state": _inventory(),
        "fill_audit": _fill_audit(),
        "lifecycle_summary": _lifecycle(),
    }
    payload.update(overrides)
    return build_single_side_continuous_rehearsal(**payload)


def test_ready_rehearsal_never_authorizes_live_execution() -> None:
    report = _report()

    assert report["status"] == READY_STATUS
    assert report["next_action"] == "QUOTE_NEXT"
    assert report["session_limits"]["one_side_only"] is True
    assert report["session_limits"]["max_open_order"] == 1
    assert report["session_limits"]["both_side_live"] is False
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_mutex_locked_blocks_next_quote() -> None:
    report = _report(order_mutex=_mutex(order_mutex_state="LIVE_ORDER_OPEN"))

    assert report["status"] == BLOCKED_STATUS
    assert "ORDER_MUTEX_LOCKED" in report["blockers"]


def test_stale_open_order_blocks_next_quote() -> None:
    report = _report(inventory_state=_inventory(open_order_count=1, token_open_order_count=1))

    assert report["status"] == BLOCKED_STATUS
    assert "UNRESOLVED_OPEN_ORDER_BLOCKS_NEXT_QUOTE" in report["blockers"]


def test_unreconciled_inventory_blocks_next_quote() -> None:
    report = _report(inventory_state=_inventory(token_balance_shares=5), fill_audit={"status": "C_FILL_RECONCILIATION_AUDIT_BLOCKED"})

    assert report["status"] == BLOCKED_STATUS
    assert "UNRECONCILED_INVENTORY_BLOCKS_NEXT_QUOTE" in report["blockers"]


def test_max_order_count_stops_loop() -> None:
    report = _report(current_order_count=3, max_order_count=3)

    assert report["status"] == STOPPED_STATUS
    assert "MAX_ORDER_COUNT_REACHED" in report["blockers"]


def test_stop_file_halts_loop(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP"
    stop_file.write_text("stop", encoding="utf-8")
    report = _report(stop_file=stop_file)

    assert report["status"] == STOPPED_STATUS
    assert "STOP_FILE_PRESENT" in report["blockers"]


def test_reward_pnl_mixing_blocks() -> None:
    report = _report(lifecycle_summary={"cash_accounting": {"estimated_reward_counted_as_realized_cash_pnl": True}})

    assert report["status"] == BLOCKED_STATUS
    assert "REWARD_PNL_ACCOUNTING_MIXED" in report["blockers"]


def test_cli_writes_rehearsal_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "full_market_probe_planner_latest.json": _planner(),
        "order_mutex_readiness_latest.json": _mutex(),
        "inventory_state_latest.json": _inventory(),
        "c_fill_reconciliation_audit_latest.json": _fill_audit(),
        "fill_lifecycle_summary_latest.json": _lifecycle(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "rehearsal.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == READY_STATUS
