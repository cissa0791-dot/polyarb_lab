from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_second_probe_decision_package import main
from src.live.second_probe_decision_package import (
    BLOCKED_STATUS,
    OPTION_A,
    OPTION_B,
    OPTION_C,
    READY_STATUS,
    build_second_probe_decision_package,
)


NOW = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
ORDER_ID = "0x36de"


def _post_audit(**overrides) -> dict:
    payload = {
        "status": "POST_LIVE_PROBE_AUDIT_READY",
        "classification": "ZERO_FILL_EXECUTION_CHAIN_PROVEN",
        "target_market_slug": MARKET,
        "order_id": ORDER_ID,
        "token_status": "EXPENDED",
        "submit_latency_ms": 65.49368,
        "cancel_latency_ms": 22.853087,
        "final_state": {
            "open_order_count": 0,
            "token_balance_shares": 0.0,
            "can_submit_order": False,
            "live_order_sent": False,
            "cycle_closed": False,
            "profitability_validation_allowed": False,
        },
        "local_ledger": {
            "zero_fill_observed": True,
            "cancel_confirmed_not_open": True,
        },
        "next_probe_requirements": {
            "new_operator_approval_required": True,
            "new_token_required": True,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
        },
    }
    payload.update(overrides)
    return payload


def _gate(**overrides) -> dict:
    payload = {
        "status": "LIVE_READY_APPROVED",
        "asserts_passed": 12,
        "asserts_failed": 0,
        "blockers": [],
        "can_submit_order": False,
        "live_order_sent": False,
        "target_market_slug": MARKET,
    }
    payload.update(overrides)
    return payload


def _inventory(**overrides) -> dict:
    payload = {
        "status": "INVENTORY_STATE_CLEAR",
        "open_order_count": 0,
        "token_balance_shares": 0.0,
    }
    payload.update(overrides)
    return payload


def _mutex(**overrides) -> dict:
    payload = {
        "status": "ORDER_MUTEX_READY",
        "order_mutex_state": "NO_ORDER",
    }
    payload.update(overrides)
    return payload


def _heartbeat() -> dict:
    return {"status": "API_HEARTBEAT_READY", "latency_ms": 63.306673}


def _micro() -> dict:
    return {
        "status": "MARKET_MICROSTRUCTURE_READY",
        "market_slug": MARKET,
        "best_bid": 0.36,
        "best_ask": 0.37,
        "quote_bid": 0.36,
        "quote_ask": 0.37,
    }


def _report(**overrides) -> dict:
    payload = {
        "post_live_audit": _post_audit(),
        "gate": _gate(),
        "inventory_state": _inventory(),
        "order_mutex": _mutex(),
        "heartbeat": _heartbeat(),
        "market_microstructure": _micro(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_second_probe_decision_package(**payload)


def test_recommends_long_observation_bid_only_without_execution_authorization() -> None:
    report = _report()

    assert report["status"] == READY_STATUS
    assert report["recommended_option"] == OPTION_B
    assert report["option_table"][OPTION_A]["recommendation"] == "REJECT"
    assert report["option_table"][OPTION_B]["recommendation"] == "SELECT"
    assert report["option_table"][OPTION_C]["recommendation"] == "DEFER"
    assert report["approval_boundary"]["execution_authorized"] is False
    assert report["approval_boundary"]["can_submit_order"] is False
    assert report["approval_boundary"]["live_order_sent"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_blueprint_requires_new_approval_and_new_token() -> None:
    report = _report()

    assert report["recommended_probe_blueprint"]["mode"] == "SINGLE_SIDE_BID_LONG_OBSERVATION_REHEARSAL"
    assert report["recommended_probe_blueprint"]["suggested_hold_seconds"] == 300
    assert report["recommended_probe_blueprint"]["max_order_count"] == 1
    assert report["recommended_probe_blueprint"]["auto_retry_allowed"] is False
    assert report["recommended_probe_blueprint"]["token_created_here"] is False
    assert report["approval_boundary"]["new_operator_approval_required"] is True
    assert report["approval_boundary"]["new_token_required"] is True
    assert report["approval_boundary"]["same_token_retry_allowed"] is False
    assert report["approval_boundary"]["same_approval_retry_allowed"] is False


def test_missing_post_audit_blocks_decision_package() -> None:
    report = _report(post_live_audit={})

    assert report["status"] == BLOCKED_STATUS
    assert "POST_LIVE_PROBE_AUDIT_NOT_READY" in report["blockers"]


def test_prior_open_order_or_inventory_blocks_decision_package() -> None:
    report = _report(
        post_live_audit=_post_audit(
            final_state={
                "open_order_count": 1,
                "token_balance_shares": 2.0,
                "can_submit_order": False,
                "live_order_sent": False,
            }
        )
    )

    assert report["status"] == BLOCKED_STATUS
    assert "PRIOR_OPEN_ORDER_NOT_CLEAR" in report["blockers"]
    assert "PRIOR_INVENTORY_NOT_CLEAR" in report["blockers"]


def test_current_gate_submit_flags_block_decision_package() -> None:
    report = _report(gate=_gate(can_submit_order=True, live_order_sent=True))

    assert report["status"] == BLOCKED_STATUS
    assert "CURRENT_GATE_CAN_SUBMIT_ORDER_NOT_FALSE" in report["blockers"]
    assert "CURRENT_GATE_LIVE_ORDER_SENT_NOT_FALSE" in report["blockers"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "post_live_probe_audit_001_latest.json": _post_audit(),
        "live_readiness_gate_latest.json": _gate(),
        "inventory_state_latest.json": _inventory(),
        "order_mutex_readiness_latest.json": _mutex(),
        "live_network_readiness_latest.json": _heartbeat(),
        "live_market_microstructure_latest.json": _micro(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "decision.json"
    md_out = tmp_path / "decision.md"

    rc = main(
        [
            "--reports-dir",
            str(reports),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["recommended_option"] == OPTION_B
    assert payload["can_submit_order"] is False
    assert md_out.exists()
