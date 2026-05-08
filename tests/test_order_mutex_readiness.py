from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_order_mutex_readiness_report import main
from src.live.order_mutex_readiness import build_order_mutex_readiness_report


NOW = datetime(2026, 5, 8, 1, 0, tzinfo=timezone.utc)


def _health(open_order_count=0):
    return {
        "generated_at_utc": NOW.isoformat(),
        "checks": {
            "account_open_orders": {"open_order_count": open_order_count},
            "target_account_state": {"token_open_order_count": open_order_count},
        },
    }


def _execution(**overrides):
    payload = {
        "generated_at_utc": NOW.isoformat(),
        "can_submit_order": False,
        "execution_enabled": False,
        "live_actions_enabled": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def test_no_live_open_order_reports_no_order_even_with_shadow_quote_open() -> None:
    report = build_order_mutex_readiness_report(
        health=_health(0),
        execution_system=_execution(),
        shadow_latest={"order_mutex_state": "LIVE_ORDER_OPEN"},
        now=NOW,
    )

    assert report["status"] == "ORDER_MUTEX_READY"
    assert report["order_mutex_state"] == "NO_ORDER"
    assert report["shadow_order_mutex_state"] == "LIVE_ORDER_OPEN"
    assert report["shadow_state_ignored_for_live_gate"] is True
    assert report["can_submit_order"] is False


def test_live_open_order_blocks_mutex() -> None:
    report = build_order_mutex_readiness_report(
        health=_health(1),
        execution_system=_execution(),
        now=NOW,
    )

    assert report["status"] == "ORDER_MUTEX_BLOCKED"
    assert report["order_mutex_state"] == "LIVE_ORDER_OPEN"
    assert "ORDER_MUTEX_NOT_CLEAR" in report["blockers"]


def test_upstream_execution_enabled_blocks_mutex() -> None:
    report = build_order_mutex_readiness_report(
        health=_health(0),
        execution_system=_execution(execution_enabled=True),
        now=NOW,
    )

    assert report["order_mutex_state"] == "LIVE_ORDER_OPEN"
    assert "ORDER_MUTEX_NOT_CLEAR" in report["blockers"]


def test_missing_live_sources_blocks_instead_of_guessing_no_order() -> None:
    report = build_order_mutex_readiness_report(now=NOW)

    assert report["status"] == "ORDER_MUTEX_BLOCKED"
    assert report["order_mutex_state"] == "UNKNOWN"
    assert "ORDER_MUTEX_SOURCE_MISSING" in report["blockers"]


def test_cli_writes_ready_report_from_live_account_zero_orders(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "live_api_health_readonly_now.json").write_text(json.dumps(_health(0)), encoding="utf-8")
    (reports / "execution_disabled_auto_trade_system_latest.json").write_text(json.dumps(_execution()), encoding="utf-8")
    (reports / "maker_engine_A_p0_exit_aware_latest.json").write_text(
        json.dumps({"order_mutex_state": "LIVE_ORDER_OPEN"}),
        encoding="utf-8",
    )
    out = tmp_path / "order_mutex.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["order_mutex_state"] == "NO_ORDER"
    assert payload["can_submit_order"] is False
