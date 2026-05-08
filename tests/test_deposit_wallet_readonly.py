from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_deposit_wallet_readonly_report import main
from src.live.deposit_wallet_readonly import build_deposit_wallet_readonly_report
from src.live.live_readiness_gate import build_live_readiness_gate


NOW = datetime(2026, 5, 8, 1, 0, tzinfo=timezone.utc)


def _reader(balance: float = 320_000_000.0):
    return lambda: {"raw_balance": balance, "raw_allowance": balance}


def test_missing_funder_blocks_without_treating_eoa_as_deposit_wallet() -> None:
    report = build_deposit_wallet_readonly_report(
        env={},
        balance_reader=_reader(),
        now=NOW,
    )

    assert report["status"] == "DEPOSIT_WALLET_BLOCKED"
    assert report["wallet_type"] == "EOA_OR_UNKNOWN"
    assert "DEPOSIT_WALLET_ADDRESS_MISSING" in report["blockers"]
    assert report["available_usdc"] is None
    assert report["can_submit_order"] is False


def test_reads_deposit_wallet_collateral_balance_as_usdc() -> None:
    report = build_deposit_wallet_readonly_report(
        env={"POLYMARKET_FUNDER": "0xdeposit", "POLYMARKET_SIGNATURE_TYPE": "1"},
        balance_reader=_reader(320_500_000.0),
        now=NOW,
    )

    assert report["status"] == "DEPOSIT_WALLET_READY"
    assert report["wallet_type"] == "DEPOSIT_WALLET"
    assert report["deposit_wallet_address"] == "0xdeposit"
    assert report["available_usdc"] == 320.5
    assert report["allowance_usdc"] == 320.5
    assert report["balance_source"] == "CLOB_GET_BALANCE_ALLOWANCE_COLLATERAL"
    assert report["can_submit_order"] is False


def test_balance_read_failure_blocks_with_no_secret_output() -> None:
    def fail():
        raise RuntimeError("boom secret-value")

    report = build_deposit_wallet_readonly_report(
        env={"POLYMARKET_FUNDER": "0xdeposit"},
        balance_reader=fail,
        now=NOW,
    )

    assert report["status"] == "DEPOSIT_WALLET_BLOCKED"
    assert "DEPOSIT_WALLET_BALANCE_READ_FAILED" in report["blockers"]
    assert report["available_usdc"] is None
    assert report["can_submit_order"] is False


def test_live_readiness_gate_accepts_ready_deposit_wallet_but_keeps_report_only() -> None:
    deposit = build_deposit_wallet_readonly_report(
        env={"POLYMARKET_FUNDER": "0xdeposit"},
        balance_reader=_reader(320_000_000.0),
        now=NOW,
    )
    report = build_live_readiness_gate(
        deposit_wallet=deposit,
        now=NOW,
        max_live_risk_usdc=300.0,
    )

    assert report["asserts_by_id"]["DEPOSIT_WALLET_BALANCE_ASSERT"]["passed"] is True
    assert report["can_submit_order"] is False


def test_live_readiness_gate_rejects_low_deposit_wallet_balance() -> None:
    deposit = build_deposit_wallet_readonly_report(
        env={"POLYMARKET_FUNDER": "0xdeposit"},
        balance_reader=_reader(299_990_000.0),
        now=NOW,
    )
    report = build_live_readiness_gate(
        deposit_wallet=deposit,
        now=NOW,
        max_live_risk_usdc=300.0,
    )

    assert "DEPOSIT_WALLET_BALANCE_BELOW_REQUIRED" in report["blockers"]
    assert report["can_submit_order"] is False


def test_cli_writes_blocked_report_when_env_missing(tmp_path: Path, monkeypatch) -> None:
    for key in (
        "POLYMARKET_FUNDER",
        "POLYMARKET_PRIVATE_KEY",
        "POLYMARKET_API_KEY",
        "POLYMARKET_API_SECRET",
        "POLYMARKET_API_PASSPHRASE",
        "POLYMARKET_CHAIN_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    out = tmp_path / "deposit_wallet.json"

    rc = main(["--env-file", str(tmp_path / "missing.env"), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["status"] == "DEPOSIT_WALLET_BLOCKED"
    assert "DEPOSIT_WALLET_ADDRESS_MISSING" in payload["blockers"]
    assert payload["can_submit_order"] is False
