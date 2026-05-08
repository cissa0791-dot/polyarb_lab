from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_single_side_bid_probe import main as run_probe_main
from src.live.one_time_auth_token import create_authorization_token, load_authorization_token, write_authorization_token
from src.live.single_side_bid_probe import (
    BLOCKED,
    COMPLETED,
    DRY_RUN_READY,
    EMERGENCY_REVIEW,
    build_probe_preflight,
    run_single_side_bid_probe,
)


NOW = datetime(2026, 5, 8, 3, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
TOKEN_ID = "token-ivan"


@dataclass
class _SubmitResult:
    order_id: str | None = "order-1"
    status: str = "live"
    size_matched: float = 0.0
    avg_price: float | None = None


@dataclass
class _OrderStatus:
    order_id: str = "order-1"
    status: str = "live"
    size_matched: float = 0.0
    size_remaining: float = 50.0
    avg_price: float | None = None


@dataclass
class _OpenOrder:
    order_id: str = "order-1"
    side: str = "BUY"
    price: float = 0.36
    size: float = 50.0
    size_matched: float = 0.0
    size_remaining: float = 50.0
    status: str = "open"
    token_id: str = TOKEN_ID


class _FakeClient:
    def __init__(self, *, still_open_after_cancel: bool = False) -> None:
        self.calls: list[tuple] = []
        self.still_open_after_cancel = still_open_after_cancel

    def submit_order(self, token_id, side, price, size, *, neg_risk=False, tick_size=None, fee_rate_bps=0):
        self.calls.append(("submit_order", token_id, side, price, size, neg_risk, tick_size, fee_rate_bps))
        return _SubmitResult()

    def get_order_status(self, order_id):
        self.calls.append(("get_order_status", order_id))
        return _OrderStatus(order_id=order_id)

    def cancel_order(self, order_id):
        self.calls.append(("cancel_order", order_id))
        return True

    def get_open_orders(self, token_id):
        self.calls.append(("get_open_orders", token_id))
        if self.still_open_after_cancel:
            return [_OpenOrder()]
        return []


class _Clock:
    def __init__(self) -> None:
        self.now = NOW
        self.mono = 0.0

    def utcnow(self) -> datetime:
        self.now += timedelta(milliseconds=100)
        return self.now

    def monotonic(self) -> float:
        self.mono += 0.1
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.mono += max(0.0, seconds)
        self.now += timedelta(seconds=max(0.0, seconds))


def _gate(**overrides) -> dict:
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
    }
    payload.update(overrides)
    return payload


def _rehearsal(**overrides) -> dict:
    payload = {
        "generated_at_utc": NOW.isoformat(),
        "status": "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_READY",
        "PRELIVE_READY": True,
        "EXECUTION_AUTHORIZED": False,
        "CAN_SUBMIT_ORDER": False,
        "LIVE_ORDER_SENT": False,
        "target_market_slug": MARKET,
        "order_side_selected": "BID_ONLY",
        "quote_price": 0.36,
        "quote_size": 50,
        "final_decision": {"rejected_next_mode": "MAKER_BOTH_SIDES_LIVE"},
    }
    payload.update(overrides)
    return payload


def _health() -> dict:
    return {
        "target_market": {
            "market_slug": MARKET,
            "token_id": TOKEN_ID,
            "neg_risk": True,
            "tick_size": "0.01",
        }
    }


def _micro() -> dict:
    return {
        "market_slug": MARKET,
        "token_id": TOKEN_ID,
        "quote_bid": 0.36,
        "quote_ask": 0.37,
        "quote_size": 50,
        "tick_size": 0.01,
    }


def _token_file(tmp_path: Path) -> Path:
    token = create_authorization_token(
        market_slug=MARKET,
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        now=NOW,
        nonce="fixednonce",
    )
    path = tmp_path / "token.json"
    write_authorization_token(token, path)
    return path


def test_dry_run_ready_does_not_call_client_or_consume_token(tmp_path: Path) -> None:
    token_file = _token_file(tmp_path)
    fake = _FakeClient()

    report = run_single_side_bid_probe(
        gate=_gate(),
        rehearsal=_rehearsal(),
        token_report={},
        health=_health(),
        market_microstructure=_micro(),
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        token_file=token_file,
        execute_live_probe=False,
        client=fake,
        now_fn=lambda: NOW,
    )
    token, _ = load_authorization_token(token_file)

    assert report["status"] == DRY_RUN_READY
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["token_consumed"] is False
    assert token["status"] == "ISSUED_UNUSED"
    assert fake.calls == []


def test_live_request_without_all_confirmations_blocks_before_client(tmp_path: Path) -> None:
    token_file = _token_file(tmp_path)
    fake = _FakeClient()

    report = run_single_side_bid_probe(
        gate=_gate(),
        rehearsal=_rehearsal(),
        token_report={},
        health=_health(),
        market_microstructure=_micro(),
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        token_file=token_file,
        execute_live_probe=True,
        consume_token=False,
        acknowledge_live_risk=True,
        confirm_single_side_bid_probe=True,
        client=fake,
        now_fn=lambda: NOW,
    )

    assert report["status"] == BLOCKED
    assert "LIVE_FLAG_CONSUME_TOKEN_FLAG_FAILED" in report["blockers"]
    assert report["live_order_sent"] is False
    assert fake.calls == []


def test_live_path_expends_token_submits_once_and_cancels_exact_order(tmp_path: Path) -> None:
    token_file = _token_file(tmp_path)
    fake = _FakeClient()
    clock = _Clock()

    report = run_single_side_bid_probe(
        gate=_gate(),
        rehearsal=_rehearsal(),
        token_report={},
        health=_health(),
        market_microstructure=_micro(),
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        token_file=token_file,
        execute_live_probe=True,
        consume_token=True,
        acknowledge_live_risk=True,
        confirm_single_side_bid_probe=True,
        hold_seconds=0,
        status_poll_seconds=0,
        client=fake,
        now_fn=clock.utcnow,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )
    token, _ = load_authorization_token(token_file)

    assert report["status"] == COMPLETED
    assert report["token_consumed"] is True
    assert report["token_consumed_before_submit"] is True
    assert report["live_order_sent"] is True
    assert report["can_submit_order"] is False
    assert token["status"] == "EXPENDED"
    assert fake.calls[0][0] == "submit_order"
    assert ("cancel_order", "order-1") in fake.calls
    assert report["cancel_result"]["cancel_confirmed_not_open"] is True


def test_unconfirmed_cancel_forces_emergency_review(tmp_path: Path) -> None:
    token_file = _token_file(tmp_path)
    fake = _FakeClient(still_open_after_cancel=True)
    clock = _Clock()

    report = run_single_side_bid_probe(
        gate=_gate(),
        rehearsal=_rehearsal(),
        token_report={},
        health=_health(),
        market_microstructure=_micro(),
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        token_file=token_file,
        execute_live_probe=True,
        consume_token=True,
        acknowledge_live_risk=True,
        confirm_single_side_bid_probe=True,
        hold_seconds=0,
        status_poll_seconds=0,
        client=fake,
        now_fn=clock.utcnow,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )

    assert report["status"] == EMERGENCY_REVIEW
    assert "CANCEL_NOT_CONFIRMED_ORDER_RECONCILIATION_REQUIRED" in report["blockers"]
    assert report["live_order_sent"] is True
    assert report["can_submit_order"] is False


def test_preflight_blocks_when_rehearsal_authorizes_execution() -> None:
    preflight = build_probe_preflight(
        gate=_gate(),
        rehearsal=_rehearsal(EXECUTION_AUTHORIZED=True),
        token_report={
            "status": "SINGLE_SIDE_PROBE_AUTHORIZATION_READY",
            "authorization_token_valid": True,
            "execution_release_ready": True,
            "token_status": "ISSUED_UNUSED",
            "can_submit_order": False,
            "live_order_sent": False,
        },
        target={"market_slug": MARKET, "token_id": TOKEN_ID, "quote_price": 0.36, "quote_size": 50, "order_side_selected": "BID_ONLY", "side": "BUY"},
        max_live_risk_usdc=296.67,
        now=NOW,
    )

    assert preflight["ready"] is False
    assert "REHEARSAL_REHEARSAL_DOES_NOT_AUTHORIZE_EXECUTION_FAILED" in preflight["blockers"]


def test_cli_default_writes_blocked_without_live_execution(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "single_side_live_rehearsal_latest.json": _rehearsal(),
        "single_side_probe_authorization_latest.json": {},
        "live_api_health_readonly_now.json": _health(),
        "live_market_microstructure_latest.json": _micro(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "probe.json"

    rc = run_probe_main(
        [
            "--reports-dir",
            str(reports),
            "--out",
            str(out),
            "--token-file",
            str(tmp_path / "missing-token.json"),
            "--max-live-risk-usdc",
            "296.67",
            "--quote-price",
            "0.36",
            "--quote-size",
            "50",
        ]
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["status"] == BLOCKED
    assert payload["can_submit_order"] is False
    assert payload["live_order_sent"] is False
