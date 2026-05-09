from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_b_stability_probe_execution_preflight import main
from src.live.b_stability_probe_execution_preflight import (
    BLOCKED_STATUS,
    READY_STATUS,
    build_b_stability_probe_execution_preflight,
)


NOW = datetime(2026, 5, 9, 12, 55, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
HASH = "15ab13e972f39c8ab3a563a30b3d6034f9b3c8f2ea48190c314672aca8348d18"
TOKEN_ID = "62374557612691330043510053829591470998236327248924987579815802826988260253261"


def _gate(**overrides) -> dict:
    payload = {
        "status": "LIVE_READY_APPROVED",
        "asserts_passed": 12,
        "asserts_failed": 0,
        "blockers": [],
        "target_market_slug": MARKET,
        "can_submit_order": False,
        "execution_enabled": False,
        "live_order_sent": False,
        "asserts_by_id": {
            "EXECUTION_ISOLATION_ASSERT": {"passed": True, "reason": "EXECUTION_SYSTEM_IS_HARD_DISABLED_AND_ISOLATED"},
            "DEPLOYMENT_SYNC_ASSERT": {"passed": True, "reason": "DEPLOYMENT_SYNC_PROVEN"},
        },
    }
    payload.update(overrides)
    return payload


def _approval(**overrides) -> dict:
    payload = {
        "status": "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_READY",
        "probe_type": "B_LONG_OBSERVATION_STABILITY",
        "can_submit_order": False,
        "execution_authorized": False,
        "live_order_sent": False,
        "approval_boundary": {
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
        },
        "token_binding_fields": {
            "market_slug": MARKET,
            "selected_side": "BID_ONLY",
            "quote_price": 0.38,
            "quote_size": 50.0,
            "max_live_risk_usdc": 296.67,
            "hold_seconds": 300,
            "planner_hash": HASH,
        },
    }
    payload.update(overrides)
    return payload


def _review(**overrides) -> dict:
    payload = {
        "status": "B_STABILITY_TOKEN_ISSUANCE_REVIEW_READY",
        "probe_type": "B_LONG_OBSERVATION_STABILITY",
        "can_submit_order": False,
        "execution_authorized": False,
        "live_order_sent": False,
        "token_binding_fields": {
            "market_slug": MARKET,
            "selected_side": "BID_ONLY",
            "quote_price": 0.38,
            "quote_size": 50.0,
            "max_live_risk_usdc": 296.67,
            "hold_seconds": 300,
            "token_ttl_seconds": 600,
            "planner_hash": HASH,
        },
    }
    payload.update(overrides)
    return payload


def _auth(**overrides) -> dict:
    payload = {
        "status": "SINGLE_SIDE_PROBE_AUTHORIZATION_READY",
        "authorization_token_valid": True,
        "execution_release_ready": True,
        "token_status": "ISSUED_UNUSED",
        "ttl_remaining_seconds": 600,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _mutex(**overrides) -> dict:
    payload = {
        "status": "ORDER_MUTEX_READY",
        "order_mutex_state": "NO_ORDER",
        "open_order_count": 0,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _inventory(**overrides) -> dict:
    payload = {
        "status": "INVENTORY_STATE_CLEAR",
        "open_order_count": 0,
        "token_balance_shares": 0.0,
        "partial_fill_unresolved": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _network(**overrides) -> dict:
    payload = {
        "status": "API_HEARTBEAT_READY",
        "api_health_status": "HEALTHY",
        "latency_ms": 98.7,
        "is_within_safety_threshold": True,
        "should_block_live_trading": False,
        "disconnected": False,
        "critical_latency": False,
        "mass_cancel_ready": True,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _market(**overrides) -> dict:
    payload = {
        "status": "MARKET_MICROSTRUCTURE_READY",
        "market_slug": MARKET,
        "gamma_market_slug": MARKET,
        "token_id": TOKEN_ID,
        "quote_bid": 0.38,
        "quote_ask": 0.42,
        "quote_size": 50.0,
        "tick_size": 0.01,
        "checks": {
            "quote_bid_tick_aligned": True,
            "quote_ask_tick_aligned": True,
            "quote_bid_below_quote_ask": True,
        },
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _payload(**overrides) -> dict:
    payload = {
        "gate": _gate(),
        "approval_package": _approval(),
        "token_issuance_review": _review(),
        "authorization": _auth(),
        "order_mutex": _mutex(),
        "inventory_state": _inventory(),
        "network": _network(),
        "market_microstructure": _market(),
        "now": NOW,
    }
    payload.update(overrides)
    return payload


def test_preflight_ready_keeps_execution_locked() -> None:
    report = build_b_stability_probe_execution_preflight(**_payload())

    assert report["status"] == READY_STATUS
    assert report["token_ready"] is True
    assert report["authorization_token_valid"] is True
    assert report["token_status"] == "ISSUED_UNUSED"
    assert report["quote_price"] == 0.38
    assert report["quote_size"] == 50.0
    assert report["hold_seconds"] == 300
    assert report["execution_buffer_seconds"] == 60
    assert report["required_ttl_remaining_seconds"] == 360
    assert report["planner_hash"] == HASH
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["execution_window_policy"]["can_submit_order_may_be_true_only_inside_runner_atomic_submit"] is True
    assert report["execution_window_policy"]["auto_retry"] is False
    assert report["execution_window_policy"]["maker_both_sides_live_allowed"] is False


def test_preflight_blocks_expired_or_expended_token() -> None:
    expired = build_b_stability_probe_execution_preflight(**_payload(authorization=_auth(ttl_remaining_seconds=0)))
    expended = build_b_stability_probe_execution_preflight(**_payload(authorization=_auth(token_status="EXPENDED")))

    assert expired["status"] == BLOCKED_STATUS
    assert "TOKEN_TOKEN_NOT_EXPIRED_FAILED" in expired["blockers"]
    assert expended["status"] == BLOCKED_STATUS
    assert "TOKEN_TOKEN_UNUSED_FAILED" in expended["blockers"]


def test_preflight_blocks_token_ttl_shorter_than_hold_plus_buffer() -> None:
    report = build_b_stability_probe_execution_preflight(**_payload(authorization=_auth(ttl_remaining_seconds=359)))

    assert report["status"] == BLOCKED_STATUS
    assert report["required_ttl_remaining_seconds"] == 360
    assert "TOKEN_TOKEN_TTL_COVERS_HOLD_PLUS_BUFFER_FAILED" in report["blockers"]


def test_preflight_blocks_if_12_of_12_or_execution_isolation_not_ready() -> None:
    report = build_b_stability_probe_execution_preflight(
        **_payload(
            gate=_gate(
                status="LIVE_NOT_READY",
                asserts_passed=11,
                asserts_failed=1,
                asserts_by_id={
                    "EXECUTION_ISOLATION_ASSERT": {"passed": False},
                    "DEPLOYMENT_SYNC_ASSERT": {"passed": True},
                },
            )
        )
    )

    assert report["status"] == BLOCKED_STATUS
    assert "GATE_LIVE_READY_APPROVED_FAILED" in report["blockers"]
    assert "GATE_ASSERTS_12_OF_12_FAILED" in report["blockers"]
    assert "GATE_EXECUTION_ISOLATION_PASSED_FAILED" in report["blockers"]


def test_preflight_blocks_dirty_order_or_inventory_state() -> None:
    report = build_b_stability_probe_execution_preflight(
        **_payload(
            order_mutex=_mutex(order_mutex_state="LIVE_ORDER_OPEN", open_order_count=1),
            inventory_state=_inventory(open_order_count=1, token_balance_shares=10.0),
        )
    )

    assert report["status"] == BLOCKED_STATUS
    assert "ORDER_MUTEX_STATE_NO_ORDER_FAILED" in report["blockers"]
    assert "ORDER_MUTEX_OPEN_ORDER_COUNT_ZERO_FAILED" in report["blockers"]
    assert "INVENTORY_OPEN_ORDER_COUNT_ZERO_FAILED" in report["blockers"]
    assert "INVENTORY_TOKEN_BALANCE_ZERO_FAILED" in report["blockers"]


def test_preflight_blocks_heartbeat_and_market_context_drift() -> None:
    report = build_b_stability_probe_execution_preflight(
        **_payload(
            network=_network(status="API_HEARTBEAT_BLOCKED", api_health_status="CRITICAL_LATENCY", critical_latency=True),
            market_microstructure=_market(quote_bid=0.39),
        )
    )

    assert report["status"] == BLOCKED_STATUS
    assert "HEARTBEAT_HEARTBEAT_READY_OR_SAFE_DEGRADED_FAILED" in report["blockers"]
    assert "HEARTBEAT_NOT_CRITICAL_LATENCY_FAILED" in report["blockers"]
    assert "MARKET_CONTEXT_QUOTE_PRICE_MATCHES_MARKET_QUOTE_BID_FAILED" in report["blockers"]


def test_cli_writes_preflight_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "b_low_fill_stability_probe_approval_package_latest.json": _approval(),
        "b_stability_token_issuance_review_latest.json": _review(),
        "single_side_probe_authorization_latest.json": _auth(),
        "order_mutex_readiness_latest.json": _mutex(),
        "inventory_state_latest.json": _inventory(),
        "live_network_readiness_latest.json": _network(),
        "live_market_microstructure_latest.json": _market(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "preflight.json"
    md_out = tmp_path / "preflight.md"

    rc = main(["--reports-dir", str(reports), "--out", str(out), "--md-out", str(md_out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["can_submit_order"] is False
    assert "B Stability Probe Execution Preflight" in md_out.read_text(encoding="utf-8")
