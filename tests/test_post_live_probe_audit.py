from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_post_live_probe_audit_report import main
from src.live.one_time_auth_token import create_authorization_token, mark_token_expended
from src.live.post_live_probe_audit import BLOCKED_STATUS, READY_STATUS, build_post_live_probe_audit


NOW = datetime(2026, 5, 9, 7, 30, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
ORDER_ID = "0xabc"


def _probe(**overrides) -> dict:
    payload = {
        "status": "SINGLE_SIDE_BID_PROBE_COMPLETED",
        "generated_at_utc": NOW.isoformat(),
        "target": {"market_slug": MARKET, "quote_price": 0.36, "quote_size": 50},
        "token_consumed": True,
        "token_consumed_before_submit": True,
        "can_submit_order": False,
        "live_order_sent": True,
        "submit_result": {
            "order_id": ORDER_ID,
            "latency_ms": 65.49368,
            "size_matched": 0.0,
            "avg_price": None,
        },
        "hold_observation": {
            "observed_seconds": 30.025015,
            "status_polls": [
                {"order_id": ORDER_ID, "status": "LIVE", "size_matched": 0.0, "size_remaining": 50.0}
            ],
        },
        "cancel_result": {
            "order_id": ORDER_ID,
            "latency_ms": 22.853087,
            "cancel_request_accepted": True,
            "cancel_confirmed_not_open": True,
        },
    }
    payload.update(overrides)
    return payload


def _token() -> dict:
    token = create_authorization_token(
        market_slug=MARKET,
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        now=NOW,
        nonce="fixednonce",
    )
    return mark_token_expended(token, now=NOW)


def _authorization(**overrides) -> dict:
    payload = {
        "status": "SINGLE_SIDE_PROBE_AUTHORIZATION_BLOCKED",
        "token_status": "EXPENDED",
        "authorization_token_valid": False,
        "execution_release_ready": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "blockers": ["AUTH_TOKEN_ALREADY_EXPENDED"],
    }
    payload.update(overrides)
    return payload


def _inventory(**overrides) -> dict:
    payload = {
        "status": "INVENTORY_STATE_CLEAR",
        "open_order_count": 0,
        "token_open_order_count": 0,
        "token_balance_shares": 0,
        "non_usdc_position_count": 0,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _mutex(**overrides) -> dict:
    payload = {
        "status": "ORDER_MUTEX_READY",
        "order_mutex_state": "NO_ORDER",
        "can_submit_order": False,
        "live_order_sent": False,
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
    }
    payload.update(overrides)
    return payload


def _deposit(**overrides) -> dict:
    payload = {
        "status": "DEPOSIT_WALLET_READY",
        "deposit_wallet_address": "0xwallet",
        "available_usdc": 306.678811,
        "balance_source": "CLOB_GET_BALANCE_ALLOWANCE_COLLATERAL",
        "read_only": True,
    }
    payload.update(overrides)
    return payload


def _heartbeat() -> dict:
    return {"status": "API_HEARTBEAT_READY", "latency_ms": 67.793432}


def _order_reconciliation(**overrides) -> dict:
    payload = {
        "status": "ORDER_STATUS_RECONCILIATION_READY",
        "read_only": True,
        "order_id": ORDER_ID,
        "raw_order_status": "CANCELED",
        "size_matched": 0.0,
        "original_size": 50.0,
        "raw_order_cancelled_zero_fill": True,
        "can_submit_order": False,
        "live_order_sent": False,
        "blockers": [],
    }
    payload.update(overrides)
    return payload


def _report(**overrides) -> dict:
    payload = {
        "probe": _probe(),
        "authorization": _authorization(),
        "token": _token(),
        "inventory_state": _inventory(),
        "order_mutex": _mutex(),
        "gate": _gate(),
        "deposit_wallet": _deposit(),
        "heartbeat": _heartbeat(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_post_live_probe_audit(**payload)


def test_zero_fill_probe_audit_freezes_successful_chain_without_profit_claim() -> None:
    report = _report()

    assert report["status"] == READY_STATUS
    assert report["classification"] == "ZERO_FILL_EXECUTION_CHAIN_PROVEN"
    assert report["order_id"] == ORDER_ID
    assert report["submit_latency_ms"] == 65.49368
    assert report["cancel_latency_ms"] == 22.853087
    assert report["tri_party_consistency"]["consistent"] is True
    assert report["pnl_accounting"]["profitability_claimed"] is False
    assert report["pnl_accounting"]["realized_cash_pnl_usdc"] == 0.0


def test_visibility_drift_abort_is_confirmed_as_safe_zero_fill_closeout() -> None:
    probe = _probe(
        status="SINGLE_SIDE_BID_PROBE_ABORTED_CANCEL_CONFIRMED",
        blockers=["ORDER_DISAPPEARED_UNEXPECTEDLY"],
        abort_condition="ORDER_DISAPPEARED_UNEXPECTEDLY",
        hold_observation={
            "observed_seconds": 0.049948,
            "aborted": True,
            "abort_condition": "ORDER_DISAPPEARED_UNEXPECTEDLY",
            "status_polls": [
                {"order_id": ORDER_ID, "status": "unknown", "size_matched": 0.0, "size_remaining": 0.0}
            ],
            "abort_snapshot": {
                "open_order_guard": {
                    "open_order_count": 0,
                    "matching_order_count": 0,
                    "order_id": ORDER_ID,
                }
            },
        },
        cancel_result={
            "order_id": ORDER_ID,
            "latency_ms": 23.933679,
            "cancel_request_accepted": True,
            "cancel_confirmed_not_open": True,
        },
    )
    report = _report(probe=probe, order_reconciliation=_order_reconciliation())

    assert report["status"] == READY_STATUS
    assert report["classification"] == "OPEN_ORDER_VISIBILITY_DRIFT_ABORT_CONFIRMED"
    assert report["blockers"] == []
    assert report["local_ledger"]["visibility_drift_abort_confirmed"] is True
    assert report["tri_party_consistency"]["local_ledger_cancel_confirmed_zero_fill"] is True
    assert report["tri_party_consistency"]["open_order_visibility_drift_abort_confirmed"] is True
    assert report["tri_party_consistency"]["consistent"] is True
    assert report["pnl_accounting"]["profitability_claimed"] is False
    assert report["pnl_accounting"]["realized_cash_pnl_usdc"] == 0.0


def test_visibility_drift_abort_requires_cancel_confirmation() -> None:
    probe = _probe(
        status="SINGLE_SIDE_BID_PROBE_ABORTED_CANCEL_CONFIRMED",
        blockers=["ORDER_DISAPPEARED_UNEXPECTEDLY"],
        abort_condition="ORDER_DISAPPEARED_UNEXPECTEDLY",
        cancel_result={
            "order_id": ORDER_ID,
            "latency_ms": 23.933679,
            "cancel_request_accepted": True,
            "cancel_confirmed_not_open": False,
        },
    )
    report = _report(probe=probe, order_reconciliation=_order_reconciliation())

    assert report["status"] == BLOCKED_STATUS
    assert report["classification"] == "POST_LIVE_PROBE_AUDIT_INCOMPLETE"
    assert report["local_ledger"]["visibility_drift_abort_confirmed"] is False
    assert "LOCAL_LEDGER_CANCEL_NOT_CONFIRMED" in report["blockers"]


def test_visibility_drift_abort_requires_raw_order_cancelled_zero_fill() -> None:
    probe = _probe(
        status="SINGLE_SIDE_BID_PROBE_ABORTED_CANCEL_CONFIRMED",
        blockers=["ORDER_DISAPPEARED_UNEXPECTEDLY"],
        abort_condition="ORDER_DISAPPEARED_UNEXPECTEDLY",
    )
    report = _report(probe=probe, order_reconciliation=_order_reconciliation(raw_order_status="LIVE"))

    assert report["status"] == BLOCKED_STATUS
    assert report["classification"] == "POST_LIVE_PROBE_AUDIT_INCOMPLETE"
    assert report["raw_order_audit"]["raw_order_cancelled_zero_fill"] is False
    assert report["local_ledger"]["visibility_drift_abort_confirmed"] is False
    assert "TRI_PARTY_CONSISTENCY_NOT_PROVEN" in report["blockers"]


def test_expended_token_is_required_and_not_reusable() -> None:
    report = _report()

    assert report["token_audit"]["token_status"] == "EXPENDED"
    assert report["token_audit"]["token_cannot_be_reused"] is True
    assert report["next_probe_requirements"]["new_operator_approval_required"] is True
    assert report["next_probe_requirements"]["new_token_required"] is True
    assert report["next_probe_requirements"]["same_token_retry_allowed"] is False


def test_unused_token_blocks_post_live_audit() -> None:
    unused = create_authorization_token(
        market_slug=MARKET,
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        now=NOW,
        nonce="fixednonce",
    )
    report = _report(
        token=unused,
        authorization=_authorization(
            status="SINGLE_SIDE_PROBE_AUTHORIZATION_READY",
            token_status="ISSUED_UNUSED",
            authorization_token_valid=True,
            execution_release_ready=True,
            blockers=[],
        ),
    )

    assert report["status"] == BLOCKED_STATUS
    assert "TOKEN_REUSE_NOT_BLOCKED" in report["blockers"]


def test_open_order_or_inventory_blocks_audit() -> None:
    report = _report(inventory_state=_inventory(open_order_count=1, token_open_order_count=1, token_balance_shares=10))

    assert report["status"] == BLOCKED_STATUS
    assert "API_OPEN_ORDER_NOT_CLEAR" in report["blockers"]
    assert "API_INVENTORY_NOT_CLEAR" in report["blockers"]


def test_gate_submit_flags_must_return_to_disabled() -> None:
    report = _report(gate=_gate(can_submit_order=True, live_order_sent=True))

    assert report["status"] == BLOCKED_STATUS
    assert "CAN_SUBMIT_ORDER_NOT_FALSE" in report["blockers"]
    assert "LIVE_ORDER_SENT_NOT_RESET_IN_GATE" in report["blockers"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "single_side_bid_probe_latest.json": _probe(),
        "single_side_probe_authorization_latest.json": _authorization(),
        "inventory_state_latest.json": _inventory(),
        "order_mutex_readiness_latest.json": _mutex(),
        "live_readiness_gate_latest.json": _gate(),
        "deposit_wallet_readonly_latest.json": _deposit(),
        "live_network_readiness_latest.json": _heartbeat(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    token_path = tmp_path / "token.json"
    token_path.write_text(json.dumps(_token()), encoding="utf-8")
    out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"

    rc = main(
        [
            "--reports-dir",
            str(reports),
            "--token-file",
            str(token_path),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["final_state"]["open_order_count"] == 0
    assert md_out.exists()
