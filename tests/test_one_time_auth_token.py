from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.build_single_side_probe_authorization_report import main as build_auth_report_main
from scripts.create_single_side_probe_authorization_token import main as create_token_main
from src.live.one_time_auth_token import (
    build_authorization_report,
    create_authorization_token,
    mark_token_expended,
)


NOW = datetime(2026, 5, 8, 3, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
PLANNER_HASH = "b" * 64


def _token(**overrides) -> dict:
    payload = create_authorization_token(
        market_slug=MARKET,
        max_live_risk_usdc=296.67,
        quote_price=0.36,
        quote_size=50,
        hold_seconds=300,
        planner_hash=PLANNER_HASH,
        now=NOW,
        nonce="fixednonce",
    )
    payload.update(overrides)
    return payload


def _report(token: dict, now: datetime = NOW) -> dict:
    return build_authorization_report(
        token=token,
        expected_market_slug=MARKET,
        expected_max_live_risk_usdc=296.67,
        expected_quote_price=0.36,
        expected_quote_size=50,
        expected_hold_seconds=300,
        expected_planner_hash=PLANNER_HASH,
        now=now,
    )


def test_valid_token_is_release_ready_but_does_not_authorize_execution() -> None:
    report = _report(_token())

    assert report["status"] == "SINGLE_SIDE_PROBE_AUTHORIZATION_READY"
    assert report["authorization_token_valid"] is True
    assert report["execution_release_ready"] is True
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_expired_token_blocks_release() -> None:
    report = _report(_token(), now=NOW + timedelta(seconds=301))

    assert report["status"] == "SINGLE_SIDE_PROBE_AUTHORIZATION_BLOCKED"
    assert "AUTH_TOKEN_EXPIRED" in report["blockers"]


def test_hash_mismatch_blocks_release() -> None:
    token = _token()
    token["quote_price"] = 0.35
    report = _report(token)

    assert "AUTH_TOKEN_HASH_MISMATCH" in report["blockers"]
    assert "AUTH_TOKEN_QUOTE_PRICE_MISMATCH" in report["blockers"]


def test_expended_token_blocks_release() -> None:
    report = _report(mark_token_expended(_token(), now=NOW + timedelta(seconds=1)), now=NOW + timedelta(seconds=2))

    assert "AUTH_TOKEN_ALREADY_EXPENDED" in report["blockers"]


def test_planner_hash_mismatch_blocks_release() -> None:
    report = build_authorization_report(
        token=_token(),
        expected_market_slug=MARKET,
        expected_max_live_risk_usdc=296.67,
        expected_quote_price=0.36,
        expected_quote_size=50,
        expected_hold_seconds=300,
        expected_planner_hash="c" * 64,
        now=NOW,
    )

    assert "AUTH_TOKEN_PLANNER_HASH_MISMATCH" in report["blockers"]


def test_hold_seconds_mismatch_blocks_release() -> None:
    report = build_authorization_report(
        token=_token(),
        expected_market_slug=MARKET,
        expected_max_live_risk_usdc=296.67,
        expected_quote_price=0.36,
        expected_quote_size=50,
        expected_hold_seconds=600,
        expected_planner_hash=PLANNER_HASH,
        now=NOW,
    )

    assert "AUTH_TOKEN_HOLD_SECONDS_MISMATCH" in report["blockers"]


def test_create_token_requires_explicit_confirmation(tmp_path: Path) -> None:
    out = tmp_path / "token.json"

    rc = create_token_main(
        [
            "--market-slug",
            MARKET,
            "--max-live-risk-usdc",
            "296.67",
            "--quote-price",
            "0.36",
            "--quote-size",
            "50",
            "--hold-seconds",
            "300",
            "--out",
            str(out),
        ]
    )

    assert rc == 2
    assert not out.exists()


def test_cli_create_and_validate_token(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    report_path = tmp_path / "auth.json"
    planner_path = tmp_path / "planner.json"
    planner_created = datetime.now(timezone.utc)
    planner_path.write_text(
        json.dumps(
            {
                "status": "LIVE_PROBE_PLAN_READY",
                "planner_hash": PLANNER_HASH,
                "planner_snapshot_ts": planner_created.isoformat(),
                "planner_expires_at": (planner_created + timedelta(minutes=5)).isoformat(),
                "max_live_risk_usdc": 296.67,
                "hold_seconds": 300,
                "recommended_plan": {
                    "market_slug": MARKET,
                    "quote_price": 0.36,
                    "quote_size": 50,
                },
            }
        ),
        encoding="utf-8",
    )

    create_rc = create_token_main(
        [
            "--planner-report",
            str(planner_path),
            "--out",
            str(token_path),
            "--confirm-create-token",
        ]
    )
    validate_rc = build_auth_report_main(
        [
            "--token-file",
            str(token_path),
            "--market-slug",
            MARKET,
            "--max-live-risk-usdc",
            "296.67",
            "--quote-price",
            "0.36",
            "--quote-size",
            "50",
            "--hold-seconds",
            "300",
            "--planner-report",
            str(planner_path),
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert create_rc == 0
    assert validate_rc == 0
    assert report["status"] == "SINGLE_SIDE_PROBE_AUTHORIZATION_READY"
    assert report["can_submit_order"] is False


def test_create_token_rejects_explicit_params_that_do_not_match_planner(tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    planner_created = datetime.now(timezone.utc)
    planner_path = tmp_path / "planner.json"
    planner_path.write_text(
        json.dumps(
            {
                "status": "LIVE_PROBE_PLAN_READY",
                "planner_hash": PLANNER_HASH,
                "planner_snapshot_ts": planner_created.isoformat(),
                "planner_expires_at": (planner_created + timedelta(minutes=5)).isoformat(),
                "max_live_risk_usdc": 296.67,
                "hold_seconds": 300,
                "recommended_plan": {
                    "market_slug": MARKET,
                    "quote_price": 0.36,
                    "quote_size": 50,
                },
            }
        ),
        encoding="utf-8",
    )

    rc = create_token_main(
        [
            "--planner-report",
            str(planner_path),
            "--quote-price",
            "0.37",
            "--out",
            str(token_path),
            "--confirm-create-token",
        ]
    )

    assert rc == 2
    assert not token_path.exists()
