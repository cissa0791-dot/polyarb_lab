from __future__ import annotations

import json

from scripts.build_full_market_probe_planner_report import main
from src.live.full_market_probe_planner import build_full_market_probe_plan


def _candidate(market: str, **overrides) -> dict:
    payload = {
        "market_slug": market,
        "quote_price": 0.38,
        "quote_size": 50,
        "fill_probability": 0.2,
        "capital_required_usdc": 19.0,
        "reward_min_size_check": True,
        "tick_size_check": True,
        "fee_check": True,
        "toxic_flow_status": "TOXIC_FLOW_READY",
        "expected_reward_usdc": 0.2,
        "toxic_flow_score": 0.0,
    }
    payload.update(overrides)
    return payload


def test_rejects_toxic_market() -> None:
    report = build_full_market_probe_plan(candidates=[_candidate("toxic", toxic_unsafe=True)], max_live_risk_usdc=100)

    assert report["status"] == "FULL_MARKET_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert report["rejected_candidates"][0]["rejection_reasons"] == ["TOXIC_FLOW_UNSAFE"]


def test_rejects_fill_probability_too_high_for_b() -> None:
    report = build_full_market_probe_plan(candidates=[_candidate("hot", fill_probability=0.9)], max_live_risk_usdc=100)

    assert "FILL_PROBABILITY_TOO_HIGH_FOR_B" in report["rejection_reasons"]
    assert report["selected_candidate"] is None


def test_rejects_reward_min_size_fail_and_capital_over_budget() -> None:
    report = build_full_market_probe_plan(
        candidates=[
            _candidate("small", reward_min_size_check=False),
            _candidate("expensive", capital_required_usdc=500.0),
        ],
        max_live_risk_usdc=100,
    )

    assert "REWARD_MIN_SIZE_FAIL" in report["rejection_reasons"]
    assert "CAPITAL_OVER_BUDGET" in report["rejection_reasons"]


def test_selects_cleanest_b_candidate_without_execution_authority() -> None:
    report = build_full_market_probe_plan(
        candidates=[
            _candidate("ok-low-reward", expected_reward_usdc=0.1),
            _candidate("ok-better", expected_reward_usdc=0.4),
            _candidate("fee-bad", fee_check=False),
        ],
        max_live_risk_usdc=100,
    )

    assert report["status"] == "FULL_MARKET_PROBE_PLAN_READY"
    assert report["selected_candidate"]["market_slug"] == "ok-better"
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["execution_authorized"] is False


def test_global_optimum_only_when_universe_complete() -> None:
    incomplete = build_full_market_probe_plan(candidates=[_candidate("ok")], max_live_risk_usdc=100)
    complete = build_full_market_probe_plan(
        candidates=[_candidate("ok")],
        max_live_risk_usdc=100,
        scan_scope={"declared_universe": "fixture", "universe_complete": True},
    )

    assert incomplete["is_global_optimum"] is False
    assert complete["is_global_optimum"] is True


def test_cli_writes_full_market_planner_report(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    out = tmp_path / "full_planner.json"
    candidates.write_text(json.dumps([_candidate("ok")]), encoding="utf-8")

    rc = main(["--candidates", str(candidates), "--max-live-risk-usdc", "100", "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == "FULL_MARKET_PROBE_PLAN_READY"
    assert payload["selected_candidate"]["market_slug"] == "ok"
