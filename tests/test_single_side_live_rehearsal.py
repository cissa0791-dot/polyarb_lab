from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_single_side_live_rehearsal_report import main
from src.live.single_side_live_rehearsal import build_single_side_live_rehearsal_report


NOW = datetime(2026, 5, 8, 3, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"


def _assertion(assert_id: str, passed: bool = True) -> dict:
    return {"assert_id": assert_id, "passed": passed, "reason": f"{assert_id}_READY", "details": {"ok": passed}}


def _gate(**overrides) -> dict:
    ids = [
        "EXECUTION_ISOLATION_ASSERT",
        "DEPLOYMENT_SYNC_ASSERT",
        "AUTH_SCOPE_ASSERT",
        "DEPOSIT_WALLET_BALANCE_ASSERT",
        "ORDER_MUTEX_ASSERT",
        "CANCEL_HEARTBEAT_ASSERT",
        "TICK_SIZE_PRICE_ASSERT",
        "REWARD_SCORING_ASSERT",
        "FILL_ADVERSE_SELECTION_ASSERT",
        "INVENTORY_STATE_ASSERT",
        "FEE_RECONCILIATION_ASSERT",
        "FINAL_PHYSICAL_ASSERT",
    ]
    assertions = [_assertion(assert_id) for assert_id in ids]
    payload = {
        "generated_at_utc": NOW.isoformat(),
        "status": "LIVE_READY_APPROVED",
        "asserts_passed": 12,
        "asserts_failed": 0,
        "blockers": [],
        "can_submit_order": False,
        "live_order_sent": False,
        "target_market_slug": MARKET,
        "max_live_risk_usdc": 296.67,
        "assertions": assertions,
        "asserts_by_id": {item["assert_id"]: item for item in assertions},
    }
    payload.update(overrides)
    return payload


def _deployment(**overrides) -> dict:
    payload = {
        "generated_at_utc": NOW.isoformat(),
        "status": "DEPLOYMENT_SYNC_OK",
        "head_matches_approved": True,
        "critical_checksums_match": True,
        "unreviewed_changes_present": False,
        "local_dirty": False,
        "remote_dirty": False,
    }
    payload.update(overrides)
    return payload


def _execution(**overrides) -> dict:
    payload = {
        "generated_at_utc": NOW.isoformat(),
        "status": "EXECUTION_ISOLATION_READY",
        "single_writer_ok": True,
        "suspicious_process_count": 0,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _ready_inputs(**overrides) -> dict:
    payload = {
        "gate": _gate(),
        "deployment": _deployment(),
        "execution_system": _execution(),
        "deposit_wallet": {"status": "DEPOSIT_WALLET_READY", "available_usdc": 306.678811},
        "market_microstructure": {
            "status": "MARKET_MICROSTRUCTURE_READY",
            "market_slug": MARKET,
            "quote_bid": 0.36,
            "quote_ask": 0.37,
            "quote_size": 50,
            "pending_reward_usdc": 0.12,
        },
        "network": {"status": "API_HEARTBEAT_READY", "latency_ms": 98.722425},
        "order_mutex": {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"},
        "fee_reconciliation": {"status": "FEE_RECONCILIATION_READY", "estimated_net_profit_usdc": 0.49},
        "inventory_state": {"status": "INVENTORY_STATE_CLEAR"},
        "toxic_flow": {"status": "TOXIC_FLOW_READY"},
        "branch": "codex/bootstrap-next-action-policy",
        "commit_sha": "abc123",
        "target_market_slug": MARKET,
        "max_live_risk_usdc": 296.67,
        "now": NOW,
    }
    payload.update(overrides)
    return payload


def test_12_of_12_prelive_does_not_authorize_execution() -> None:
    report = build_single_side_live_rehearsal_report(**_ready_inputs())

    assert report["status"] == "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_READY"
    assert report["PRELIVE_READY"] is True
    assert report["EXECUTION_AUTHORIZED"] is False
    assert report["CAN_SUBMIT_ORDER"] is False
    assert report["can_submit_order"] is False
    assert report["LIVE_ORDER_SENT"] is False
    assert report["live_order_sent"] is False
    assert report["final_decision"]["recommended_next_mode"] == "MAKER_SINGLE_SIDE_LIVE_REHEARSAL"
    assert report["final_decision"]["rejected_next_mode"] == "MAKER_BOTH_SIDES_LIVE"


def test_primary_success_does_not_require_a_fill() -> None:
    report = build_single_side_live_rehearsal_report(**_ready_inputs())

    criteria = report["future_probe_success_criteria"]
    assert criteria["primary_success_requires_fill"] is False
    assert "server acknowledges" in criteria["primary_success"]
    assert "partial or full fill" in criteria["secondary_success"]


def test_estimates_and_pending_rewards_are_not_counted_as_realized_cash() -> None:
    report = build_single_side_live_rehearsal_report(**_ready_inputs())

    assert report["estimated_net_profit_usdc"] == 0.49
    assert report["realized_cash_pnl_usdc"] == 0.0
    assert report["estimated_net_profit_counted_as_realized_cash_pnl"] is False
    assert report["pending_reward_usdc"] == 0.12
    assert report["confirmed_reward_usdc"] == 0.0
    assert report["pending_reward_counted_as_confirmed_reward"] is False


def test_dirty_deployment_fails_final_switch_review() -> None:
    report = build_single_side_live_rehearsal_report(
        **_ready_inputs(deployment=_deployment(unreviewed_changes_present=True, local_dirty=True))
    )

    assert report["status"] == "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_BLOCKED"
    assert "DIRTY_DEPLOYMENT_FOR_FINAL_SWITCH_REVIEW" in report["blockers"]


def test_suspicious_old_processes_fail_execution_isolation_review() -> None:
    report = build_single_side_live_rehearsal_report(
        **_ready_inputs(execution_system=_execution(status="EXECUTION_ISOLATION_BLOCKED", single_writer_ok=False, suspicious_process_count=2))
    )

    assert report["status"] == "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_BLOCKED"
    assert "EXECUTION_ISOLATION_NOT_READY_FOR_FINAL_SWITCH_REVIEW" in report["blockers"]
    assert "SUSPICIOUS_EXECUTION_PROCESS_PRESENT" in report["blockers"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "deployment_consistency_latest.json": _deployment(),
        "execution_disabled_auto_trade_system_latest.json": _execution(),
        "deposit_wallet_readonly_latest.json": {"status": "DEPOSIT_WALLET_READY", "available_usdc": 306.678811},
        "live_market_microstructure_latest.json": {
            "status": "MARKET_MICROSTRUCTURE_READY",
            "market_slug": MARKET,
            "quote_bid": 0.36,
            "quote_ask": 0.37,
            "quote_size": 50,
        },
        "live_network_readiness_latest.json": {"status": "API_HEARTBEAT_READY", "latency_ms": 98.722425},
        "order_mutex_readiness_latest.json": {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"},
        "fee_reconciliation_latest.json": {"status": "FEE_RECONCILIATION_READY", "estimated_net_profit_usdc": 0.49},
        "inventory_state_latest.json": {"status": "INVENTORY_STATE_CLEAR"},
        "toxic_flow_latest.json": {"status": "TOXIC_FLOW_READY"},
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "review.json"
    md_out = tmp_path / "review.md"

    rc = main(
        [
            "--reports-dir",
            str(reports),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
            "--target-market-slug",
            MARKET,
            "--max-live-risk-usdc",
            "296.67",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["PRELIVE_READY"] is True
    assert payload["EXECUTION_AUTHORIZED"] is False
    assert md_out.exists()
