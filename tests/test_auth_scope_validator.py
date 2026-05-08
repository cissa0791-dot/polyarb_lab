from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.build_live_auth_readiness_report import main
from src.live.auth_scope_validator import AuthScopeSnapshot, build_auth_scope_readiness_report
from src.live.live_readiness_gate import build_live_readiness_gate


NOW = datetime(2026, 5, 8, 3, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"


def _snapshot(**overrides) -> AuthScopeSnapshot:
    payload = {
        "configured_api_key": "trading-key",
        "level_1_auth_ok": True,
        "level_2_auth_ok": True,
        "clob_v2_available": True,
        "signer_address": "0xSigner",
        "trading_api_keys": ("trading-key",),
        "readonly_api_keys": (),
        "rate_limit_degraded": False,
        "errors": (),
    }
    payload.update(overrides)
    return AuthScopeSnapshot(**payload)


def _deposit(**overrides) -> dict:
    payload = {
        "read_only": True,
        "wallet_type": "DEPOSIT_WALLET",
        "deposit_wallet_address": "0xDeposit",
        "available_usdc": 320.0,
        "generated_at_utc": NOW.isoformat(),
    }
    payload.update(overrides)
    return payload


def _env(**overrides) -> dict[str, str]:
    payload = {
        "POLYMARKET_API_KEY": "trading-key",
        "POLYMARKET_FUNDER": "0xDeposit",
        "POLYMARKET_SIGNATURE_TYPE": "1",
    }
    payload.update(overrides)
    return payload


def test_ready_auth_scope_report_does_not_expose_secret_key() -> None:
    report = build_auth_scope_readiness_report(
        env=_env(),
        deposit_wallet_report=_deposit(),
        auth_reader=lambda: _snapshot(),
        now=NOW,
    )

    assert report["status"] == "AUTH_SCOPE_READY"
    assert report["is_signing_enabled"] is True
    assert report["configured_key_is_trading_key"] is True
    assert report["configured_key_is_readonly_key"] is False
    assert report["funder_matches_deposit_wallet"] is True
    assert report["deposit_wallet_balance_sufficient"] is True
    assert report["can_submit_order"] is False
    assert "trading-key" not in json.dumps(report)
    assert report["configured_api_key_fingerprint"]["prefix"] == "trad****"


def test_readonly_key_blocks_auth_scope() -> None:
    report = build_auth_scope_readiness_report(
        env=_env(),
        deposit_wallet_report=_deposit(),
        auth_reader=lambda: _snapshot(trading_api_keys=(), readonly_api_keys=("trading-key",)),
        now=NOW,
    )

    assert report["status"] == "AUTH_SCOPE_BLOCKED"
    assert "API_KEY_NOT_REGISTERED_FOR_TRADING" in report["blockers"]
    assert "API_KEY_IS_READONLY" in report["blockers"]
    assert "SIGNING_NOT_ENABLED" in report["blockers"]
    assert "READONLY_API_KEY" in report["abnormal_restrictions"]


def test_funder_must_match_deposit_wallet_report() -> None:
    report = build_auth_scope_readiness_report(
        env=_env(POLYMARKET_FUNDER="0xWrong"),
        deposit_wallet_report=_deposit(),
        auth_reader=lambda: _snapshot(),
        now=NOW,
    )

    assert "FUNDER_DEPOSIT_WALLET_MISMATCH" in report["blockers"]
    assert report["funder_matches_deposit_wallet"] is False


def test_balance_requirement_uses_live_risk_and_buffers() -> None:
    report = build_auth_scope_readiness_report(
        env=_env(),
        deposit_wallet_report=_deposit(available_usdc=306.0),
        auth_reader=lambda: _snapshot(),
        now=NOW,
        max_live_risk_usdc=300.0,
    )

    assert report["required_usdc"] == 310.0
    assert "DEPOSIT_WALLET_BALANCE_BELOW_REQUIRED" in report["blockers"]


def test_auth_report_takes_precedence_over_consumed_bootstrap_approval_in_gate() -> None:
    auth_report = build_auth_scope_readiness_report(
        env=_env(),
        deposit_wallet_report=_deposit(),
        auth_reader=lambda: _snapshot(),
        now=NOW,
    )

    gate = build_live_readiness_gate(
        auth_readiness=auth_report,
        approval={
            "generated_at_utc": NOW.isoformat(),
            "approved_market_slug": MARKET,
            "approved_action_scopes": ["FIRST_CYCLE_BOOTSTRAP_EVIDENCE_GENERATION_ONLY"],
            "consumed": True,
        },
        target_market_slug=MARKET,
        now=NOW,
    )

    assert gate["asserts_by_id"]["AUTH_SCOPE_ASSERT"]["passed"] is True
    assert "AUTH_SCOPE_NOT_PROVEN" not in gate["blockers"]


def test_cli_writes_report_from_injected_deposit_wallet_file(tmp_path, monkeypatch) -> None:
    deposit_path = tmp_path / "deposit.json"
    deposit_path.write_text(json.dumps(_deposit()), encoding="utf-8")
    out = tmp_path / "auth.json"

    monkeypatch.setenv("POLYMARKET_API_KEY", "trading-key")
    monkeypatch.setenv("POLYMARKET_FUNDER", "0xDeposit")
    monkeypatch.setenv("POLYMARKET_SIGNATURE_TYPE", "1")

    import src.live.auth_scope_validator as validator

    monkeypatch.setattr(validator, "_read_live_auth_scope", lambda **_: _snapshot())
    rc = main(["--deposit-wallet-report", str(deposit_path), "--out", str(out), "--max-live-risk-usdc", "300"])

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "AUTH_SCOPE_READY"
    assert report["can_submit_order"] is False
