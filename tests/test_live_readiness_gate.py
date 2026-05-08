from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_live_readiness_gate_report import main
from src.live.live_readiness_gate import LIVE_NOT_READY, LIVE_READY_APPROVED, build_live_readiness_gate


NOW = datetime(2026, 5, 8, 1, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"


def _ts() -> str:
    return NOW.isoformat()


def _health() -> dict:
    return {
        "generated_at_utc": _ts(),
        "status": "HEALTHY",
        "healthy": True,
        "stale_process_possible": False,
        "target_market": {
            "market_slug": MARKET,
            "best_bid": 0.36,
            "best_ask": 0.38,
            "quote_bid": 0.36,
            "quote_ask": 0.38,
            "quote_size": 60.0,
            "tick_size": 0.01,
            "rewards_min_size": 50.0,
            "rewards_max_spread_cents": 4.5,
            "fill_probability": 0.12,
            "adverse_selection_score": 0.1,
            "toxic_flow_detected": False,
        },
        "checks": {
            "target_account_state": {"ok": True, "token_balance_shares": 0.0, "token_open_order_count": 0},
            "account_open_orders": {"ok": True, "open_order_count": 0},
        },
    }


def _execution_system(**overrides) -> dict:
    payload = {
        "generated_at_utc": _ts(),
        "can_submit_order": False,
        "execution_enabled": False,
        "live_actions_enabled": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _profit_gate() -> dict:
    return {
        "generated_at_utc": _ts(),
        "canary_market_slug": MARKET,
        "current_inventory_status": "FLAT",
        "open_order_status": "NO_OPEN_ORDER",
        "partial_fill_unresolved": False,
    }


def _approval(**overrides) -> dict:
    payload = {
        "generated_at_utc": _ts(),
        "approved_market_slug": MARKET,
        "approved_action_scopes": ["FIRST_CYCLE_BOOTSTRAP_EVIDENCE_GENERATION_ONLY"],
        "consumed": False,
    }
    payload.update(overrides)
    return payload


def _deposit_wallet(**overrides) -> dict:
    payload = {
        "generated_at_utc": _ts(),
        "read_only": True,
        "wallet_type": "DEPOSIT_WALLET",
        "deposit_wallet_address": "0xdeposit",
        "available_usdc": 320.0,
    }
    payload.update(overrides)
    return payload


def _deployment() -> dict:
    return {
        "generated_at_utc": _ts(),
        "head_matches_approved": True,
        "critical_checksums_match": True,
        "unreviewed_changes_present": False,
    }


def _network() -> dict:
    return {
        "generated_at_utc": _ts(),
        "heartbeat_ok": True,
        "mass_cancel_ready": True,
        "http_425_window_active": False,
        "cancel_latency_ms": 250.0,
        "max_cancel_latency_ms": 1000.0,
        "latency_ok": True,
        "market_suspended": False,
        "high_velocity_toxic_flow": False,
    }


def _order_mutex(state: str = "NO_ORDER") -> dict:
    return {"generated_at_utc": _ts(), "order_mutex_state": state}


def _fees() -> dict:
    return {
        "generated_at_utc": _ts(),
        "maker_fee_model_present": True,
        "taker_fee_model_present": True,
        "projected_fee_unknown": False,
        "reward_payout_mismatch": False,
    }


def _kill_switch() -> dict:
    return {"generated_at_utc": _ts(), "kill_switch_active": False}


def _ready_report(**overrides) -> dict:
    payload = {
        "health": _health(),
        "execution_system": _execution_system(),
        "profit_gate": _profit_gate(),
        "approval": _approval(),
        "deposit_wallet": _deposit_wallet(),
        "deployment": _deployment(),
        "network": _network(),
        "order_mutex": _order_mutex(),
        "fee_reconciliation": _fees(),
        "kill_switch": _kill_switch(),
        "now": NOW,
        "max_live_risk_usdc": 300.0,
    }
    payload.update(overrides)
    return build_live_readiness_gate(**payload)


def test_all_asserts_pass_but_report_does_not_enable_orders() -> None:
    report = _ready_report()

    assert report["status"] == LIVE_READY_APPROVED
    assert report["asserts_passed"] == 12
    assert report["asserts_failed"] == 0
    assert report["blockers"] == []
    assert report["can_submit_order"] is False
    assert report["live_actions_enabled"] is False
    assert report["live_order_sent"] is False


def test_missing_deposit_wallet_blocks_with_specific_reason() -> None:
    report = _ready_report(deposit_wallet={})

    assert report["status"] == LIVE_NOT_READY
    assert "DEPOSIT_WALLET_REPORT_MISSING" in report["blockers"]
    assert report["asserts_by_id"]["DEPOSIT_WALLET_BALANCE_ASSERT"]["passed"] is False
    assert report["can_submit_order"] is False


def test_eoa_balance_does_not_satisfy_deposit_wallet_lock() -> None:
    report = _ready_report(deposit_wallet=_deposit_wallet(wallet_type="EOA", deposit_wallet_address=None))

    assert "DEPOSIT_WALLET_SOURCE_NOT_DEPOSIT_WALLET" in report["blockers"]
    details = report["asserts_by_id"]["DEPOSIT_WALLET_BALANCE_ASSERT"]["details"]
    assert details["source_is_deposit_wallet"] is False


def test_eoa_address_does_not_satisfy_deposit_wallet_lock() -> None:
    report = _ready_report(deposit_wallet=_deposit_wallet(wallet_type="EOA", deposit_wallet_address="0xeoa"))

    assert "DEPOSIT_WALLET_SOURCE_NOT_DEPOSIT_WALLET" in report["blockers"]
    details = report["asserts_by_id"]["DEPOSIT_WALLET_BALANCE_ASSERT"]["details"]
    assert details["source_is_deposit_wallet"] is False


def test_deposit_wallet_must_cover_risk_and_buffers() -> None:
    report = _ready_report(deposit_wallet=_deposit_wallet(available_usdc=299.99))

    assert "DEPOSIT_WALLET_BALANCE_BELOW_REQUIRED" in report["blockers"]
    details = report["asserts_by_id"]["DEPOSIT_WALLET_BALANCE_ASSERT"]["details"]
    assert details["required_usdc"] == 310.0
    assert details["available_usdc"] == 299.99


def test_upstream_execution_enabled_is_a_hard_failure() -> None:
    report = _ready_report(execution_system=_execution_system(can_submit_order=True, execution_enabled=True))

    assert "EXECUTION_ISOLATION_NOT_PROVEN" in report["blockers"]
    assert report["asserts_by_id"]["EXECUTION_ISOLATION_ASSERT"]["passed"] is False
    assert report["can_submit_order"] is False


def test_order_mutex_must_be_clear_before_live_readiness() -> None:
    report = _ready_report(order_mutex=_order_mutex("LIVE_ORDER_OPEN"))

    assert "ORDER_MUTEX_NOT_CLEAR" in report["blockers"]
    assert report["asserts_by_id"]["ORDER_MUTEX_ASSERT"]["details"]["order_mutex_state"] == "LIVE_ORDER_OPEN"


def test_builder_writes_report_and_surfaces_missing_physical_inputs(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "live_api_health_readonly_now.json").write_text(json.dumps(_health()), encoding="utf-8")
    (reports / "execution_disabled_auto_trade_system_latest.json").write_text(
        json.dumps(_execution_system()),
        encoding="utf-8",
    )
    (reports / "profit_test_gate_latest.json").write_text(json.dumps(_profit_gate()), encoding="utf-8")
    (reports / "one_time_bootstrap_execution_approval_latest.json").write_text(
        json.dumps(_approval()),
        encoding="utf-8",
    )
    out = tmp_path / "live_readiness.json"
    md_out = tmp_path / "live_readiness.md"

    rc = main(["--reports-dir", str(reports), "--out", str(out), "--md-out", str(md_out)])

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["can_submit_order"] is False
    assert "DEPOSIT_WALLET_REPORT_MISSING" in report["blockers"]
    assert "DEPLOYMENT_SYNC_NOT_PROVEN" in report["blockers"]
    assert md_out.exists()
