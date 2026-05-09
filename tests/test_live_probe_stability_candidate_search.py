from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_live_probe_stability_candidate_search_report import main
from src.live.live_probe_stability_candidate_search import (
    SEARCH_BLOCKED,
    SEARCH_NO_SAFE_CANDIDATE,
    SEARCH_READY,
    build_stability_candidate_search,
)


NOW = datetime(2026, 5, 9, 2, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"


def _market(**overrides) -> dict:
    payload = {
        "status": "MARKET_MICROSTRUCTURE_READY",
        "market_slug": MARKET,
        "token_id": "62374557612691330043510053829591470998236327248924987579815802826988260253261",
        "best_bid": 0.41,
        "best_ask": 0.42,
        "tick_size": 0.01,
        "rewards_min_size": 50.0,
        "rewards_max_spread_cents": 4.5,
    }
    payload.update(overrides)
    return payload


def _wallet(**overrides) -> dict:
    payload = {"status": "DEPOSIT_WALLET_READY", "available_usdc": 306.678811}
    payload.update(overrides)
    return payload


def _toxic(**overrides) -> dict:
    payload = {
        "status": "TOXIC_FLOW_READY",
        "market_slug": MARKET,
        "blockers": [],
        "orderbook_imbalance": -0.056219,
        "adverse_selection_score": 0.2,
    }
    payload.update(overrides)
    return payload


def test_selects_lower_reward_band_bid_for_stability() -> None:
    report = build_stability_candidate_search(
        market_microstructure=_market(),
        deposit_wallet=_wallet(),
        toxic_flow=_toxic(),
        max_live_risk_usdc=296.67,
        now=NOW,
    )

    best = report["best_candidate"]
    assert report["status"] == SEARCH_READY
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["execution_authorized"] is False
    assert report["token_created"] is False
    assert best["quote_bid"] == 0.38
    assert best["quote_price"] == 0.38
    assert best["quote_ask"] == 0.42
    assert best["quote_size"] == 50.0
    assert best["fill_probability"] == 0.222222
    assert best["capital_required_usdc"] == 19.0
    assert best["estimated_net_profit_usdc"] == 1.99
    assert best["checks"]["stability_fill_probability_check"] is True
    assert best["checks"]["spread_inside_reward_band"] is True
    assert best["toxic_flow"]["status"] == "TOXIC_FLOW_READY"
    assert best["fee_reconciliation"]["status"] == "FEE_RECONCILIATION_READY"
    assert report["safe_candidate_count"] == 1
    assert [item["quote_bid"] for item in report["rejected_candidates"]] == [0.41, 0.4, 0.39, 0.37]


def test_rejects_when_no_reward_band_candidate_meets_stability_window() -> None:
    report = build_stability_candidate_search(
        market_microstructure=_market(),
        deposit_wallet=_wallet(),
        toxic_flow=_toxic(),
        max_live_risk_usdc=296.67,
        stability_max_fill_probability=0.2,
        min_fill_probability=0.3,
        now=NOW,
    )

    assert report["status"] == SEARCH_NO_SAFE_CANDIDATE
    assert report["best_candidate"] is None
    assert "NO_LOW_FILL_STABILITY_CANDIDATE" in report["blockers"]


def test_toxic_flow_blocks_search() -> None:
    report = build_stability_candidate_search(
        market_microstructure=_market(),
        deposit_wallet=_wallet(),
        toxic_flow=_toxic(status="TOXIC_FLOW_BLOCKED", blockers=["ADVERSE_SELECTION_RISK"]),
        max_live_risk_usdc=296.67,
        now=NOW,
    )

    assert report["status"] == SEARCH_BLOCKED
    assert "TOXIC_FLOW_NOT_CLEAR" in report["blockers"]
    assert report["candidate_count"] == 0


def test_candidate_capital_cannot_exceed_budget() -> None:
    report = build_stability_candidate_search(
        market_microstructure=_market(),
        deposit_wallet=_wallet(),
        toxic_flow=_toxic(),
        max_live_risk_usdc=18.0,
        now=NOW,
    )

    assert report["status"] == SEARCH_NO_SAFE_CANDIDATE
    assert all("CAPITAL_REQUIRED_EXCEEDS_MAX_LIVE_RISK" in item["blockers"] for item in report["rejected_candidates"])


def test_cli_writes_full_best_and_markdown_reports(tmp_path: Path) -> None:
    market = tmp_path / "market.json"
    wallet = tmp_path / "wallet.json"
    toxic = tmp_path / "toxic.json"
    out = tmp_path / "search.json"
    best = tmp_path / "best.json"
    md = tmp_path / "search.md"
    market.write_text(json.dumps(_market()), encoding="utf-8")
    wallet.write_text(json.dumps(_wallet()), encoding="utf-8")
    toxic.write_text(json.dumps(_toxic()), encoding="utf-8")

    rc = main(
        [
            "--market-report",
            str(market),
            "--deposit-wallet-report",
            str(wallet),
            "--toxic-flow-report",
            str(toxic),
            "--out",
            str(out),
            "--best-out",
            str(best),
            "--md-out",
            str(md),
            "--max-live-risk-usdc",
            "296.67",
        ]
    )

    assert rc == 0
    full_payload = json.loads(out.read_text(encoding="utf-8"))
    best_payload = json.loads(best.read_text(encoding="utf-8"))
    assert full_payload["status"] == SEARCH_READY
    assert best_payload["quote_bid"] == 0.38
    assert "Live Probe Stability Candidate Search" in md.read_text(encoding="utf-8")
