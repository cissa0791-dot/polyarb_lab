from __future__ import annotations

import json

from scripts.build_profitability_proof_report import main
from src.live.profitability_proof import (
    CONFIRMED_STATUS,
    FAILED_STATUS,
    INSUFFICIENT_STATUS,
    NO_PROOF_STATUS,
    build_profitability_proof,
)


def _summary(**overrides) -> dict:
    payload = {
        "event_count": 40,
        "cash_accounting": {
            "realized_spread_pnl_usdc": 12.0,
            "confirmed_reward_usdc": 3.0,
            "fees_usdc": 1.0,
            "adverse_selection_loss_usdc": 2.0,
            "inventory_markdown_usdc": 1.0,
            "realized_cash_pnl_usdc": 11.0,
            "drawdown_usdc": 2.0,
            "estimated_reward_counted_as_realized_cash_pnl": False,
        },
        "forecast_accounting": {
            "pending_reward_usdc": 0.5,
            "pending_reward_counted_as_confirmed_reward": False,
        },
    }
    payload.update(overrides)
    return payload


def test_unconfirmed_reward_cannot_make_report_profitable() -> None:
    summary = _summary(
        cash_accounting={"realized_cash_pnl_usdc": 5.0, "estimated_reward_counted_as_realized_cash_pnl": True},
    )
    report = build_profitability_proof(lifecycle_summary=summary, min_sample_count=30)

    assert report["status"] == FAILED_STATUS
    assert "UNCONFIRMED_REWARD_MIXED_INTO_CASH_PNL" in report["blockers"]
    assert report["profitability_claimed"] is False


def test_small_sample_cannot_mark_confirmed() -> None:
    report = build_profitability_proof(lifecycle_summary=_summary(event_count=5), min_sample_count=30)

    assert report["status"] == INSUFFICIENT_STATUS
    assert "SAMPLE_SIZE_TOO_SMALL" in report["blockers"]
    assert report["profitability_claimed"] is False


def test_negative_cash_pnl_with_pending_reward_fails_or_remains_not_profitable() -> None:
    report = build_profitability_proof(
        lifecycle_summary=_summary(cash_accounting={"realized_cash_pnl_usdc": -1.0}, forecast_accounting={"pending_reward_usdc": 10.0}),
        min_sample_count=30,
    )

    assert report["status"] == FAILED_STATUS
    assert "NET_REALIZED_PNL_NOT_POSITIVE" in report["blockers"]
    assert report["pending_reward_counted_as_profit"] is False


def test_only_confirmed_reward_plus_realized_cash_can_confirm_profitability() -> None:
    report = build_profitability_proof(lifecycle_summary=_summary(), min_sample_count=30)

    assert report["status"] == CONFIRMED_STATUS
    assert report["profitability_claimed"] is True
    assert report["confirmed_reward_usdc"] == 3.0


def test_no_events_means_no_profitability_proof_yet() -> None:
    report = build_profitability_proof(lifecycle_summary={}, min_sample_count=30)

    assert report["status"] == NO_PROOF_STATUS
    assert "NO_REAL_EVIDENCE_EVENTS" in report["blockers"]


def test_cli_writes_profitability_proof_report(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "reward_adjusted_ev_lifecycle_summary_latest.json").write_text(json.dumps(_summary()), encoding="utf-8")
    out = tmp_path / "profit.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out), "--min-sample-count", "30"])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == CONFIRMED_STATUS
