from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from scripts.build_fee_reconciliation_report import main
from src.live.fee_auditor import PriorityFeeResult, build_fee_reconciliation_report


NOW = datetime(2026, 5, 8, 1, 0, tzinfo=timezone.utc)


def _market() -> dict:
    return {
        "generated_at_utc": NOW.isoformat(),
        "market_slug": "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election",
        "quote_bid": 0.36,
        "quote_ask": 0.37,
        "quote_size": 50,
        "quote_spread": 0.01,
    }


def test_positive_margin_after_fees_marks_reconciliation_ready() -> None:
    report = build_fee_reconciliation_report(
        market_microstructure=_market(),
        explicit={
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.002,
            "entry_liquidity_role": "maker",
            "exit_liquidity_role": "taker",
            "estimated_gas_costs_usdc": 0.01,
        },
        now=NOW,
    )

    assert report["status"] == "FEE_RECONCILIATION_READY"
    assert report["can_cover_fees"] is True
    assert report["projected_fee_unknown"] is False
    assert report["maker_fee_model_present"] is True
    assert report["taker_fee_model_present"] is True
    assert report["estimated_net_profit_usdc"] == 0.435
    assert report["break_even_spread"] == 0.0013
    assert report["can_submit_order"] is False


def test_non_positive_net_profit_is_fee_blocker() -> None:
    report = build_fee_reconciliation_report(
        market_microstructure=_market(),
        explicit={
            "maker_fee_rate": 0.0,
            "taker_fee_rate": 0.0,
            "estimated_gas_costs_usdc": 0.51,
        },
        now=NOW,
    )

    assert report["status"] == "FEE_BLOCKER"
    assert report["can_cover_fees"] is False
    assert "NON_POSITIVE_NET_PROFIT_AFTER_FEES" in report["blockers"]
    assert report["estimated_net_profit_usdc"] == -0.01


def test_missing_fee_and_gas_inputs_fail_closed() -> None:
    report = build_fee_reconciliation_report(market_microstructure=_market(), explicit={}, env={}, now=NOW)

    assert report["status"] == "FEE_BLOCKER"
    assert report["projected_fee_unknown"] is True
    assert report["can_cover_fees"] is False
    assert "MAKER_FEE_RATE_MISSING" in report["blockers"]
    assert "TAKER_FEE_RATE_MISSING" in report["blockers"]
    assert "ESTIMATED_GAS_COSTS_MISSING" in report["blockers"]


def test_priority_fee_reader_can_estimate_cancel_cost() -> None:
    def reader(_rpc_url: str, _timeout_sec: float) -> PriorityFeeResult:
        return PriorityFeeResult(priority_fee_gwei=Decimal("30"), source="TEST_RPC", latency_ms=12.3)

    report = build_fee_reconciliation_report(
        market_microstructure=_market(),
        explicit={
            "maker_fee_rate": 0.0,
            "taker_fee_rate": 0.0,
            "polygon_rpc_url": "https://example.invalid",
            "gas_units_per_cancel": 100000,
            "gas_asset_usdc": 1.2,
            "cancel_tx_count": 1,
        },
        priority_fee_reader=reader,
        now=NOW,
    )

    assert report["status"] == "FEE_RECONCILIATION_READY"
    assert report["priority_fee_gwei"] == 30.0
    assert report["estimated_gas_costs_usdc"] == 0.0036
    assert report["estimated_net_profit_usdc"] == 0.4964


def test_cli_writes_fee_reconciliation_report(tmp_path: Path) -> None:
    market_report = tmp_path / "market.json"
    out = tmp_path / "fee.json"
    market_report.write_text(json.dumps(_market()), encoding="utf-8")

    rc = main(
        [
            "--market-report",
            str(market_report),
            "--out",
            str(out),
            "--maker-fee-rate",
            "0.001",
            "--taker-fee-rate",
            "0.002",
            "--estimated-gas-costs-usdc",
            "0.01",
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["status"] == "FEE_RECONCILIATION_READY"
    assert report["can_cover_fees"] is True
