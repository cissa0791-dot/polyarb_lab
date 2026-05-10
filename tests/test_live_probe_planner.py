from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_live_probe_planner_report import main
from src.live.live_probe_planner import build_live_probe_plan


NOW = datetime(2026, 5, 9, 1, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"


def _assertion(assert_id: str, passed: bool = True) -> dict:
    return {"assert_id": assert_id, "passed": passed, "reason": f"{assert_id}_READY"}


def _gate(**overrides) -> dict:
    ids = [
        "EXECUTION_ISOLATION_ASSERT",
        "DEPLOYMENT_SYNC_ASSERT",
        "AUTH_SCOPE_ASSERT",
        "DEPOSIT_WALLET_BALANCE_ASSERT",
        "ORDER_MUTEX_ASSERT",
        "CANCEL_HEARTBEAT_ASSERT",
        "TICK_SIZE_PRICE_ASSERT",
        "REWARD_SCORING_ASSERT",
        "FILL_ADVERSE_SELECTION_ASSERT",
        "INVENTORY_STATE_ASSERT",
        "FEE_RECONCILIATION_ASSERT",
        "FINAL_PHYSICAL_ASSERT",
    ]
    assertions = [_assertion(assert_id) for assert_id in ids]
    payload = {
        "status": "LIVE_READY_APPROVED",
        "asserts_passed": 12,
        "asserts_failed": 0,
        "blockers": [],
        "can_submit_order": False,
        "live_order_sent": False,
        "target_market_slug": MARKET,
        "max_live_risk_usdc": 296.67,
        "assertions": assertions,
        "asserts_by_id": {item["assert_id"]: item for item in assertions},
    }
    payload.update(overrides)
    return payload


def _market(**overrides) -> dict:
    payload = {
        "status": "MARKET_MICROSTRUCTURE_READY",
        "market_slug": MARKET,
        "best_bid": 0.41,
        "best_ask": 0.42,
        "quote_bid": 0.41,
        "quote_ask": 0.42,
        "quote_size": 50.0,
        "tick_size": 0.01,
        "rewards_min_size": 50.0,
        "rewards_max_spread_cents": 4.5,
        "checks": {
            "quote_bid_tick_aligned": True,
            "quote_ask_tick_aligned": True,
            "price_bounds_ok": True,
        },
    }
    payload.update(overrides)
    return payload


def _toxic(**overrides) -> dict:
    payload = {
        "status": "TOXIC_FLOW_READY",
        "market_slug": MARKET,
        "blockers": [],
        "fill_probability_ok": True,
        "fill_probability": 0.1,
        "best_bid_size": 100.0,
        "best_ask_size": 80.0,
        "orderbook_imbalance": 0.1,
    }
    payload.update(overrides)
    return payload


def _fee(**overrides) -> dict:
    payload = {
        "status": "FEE_RECONCILIATION_READY",
        "market_slug": MARKET,
        "can_cover_fees": True,
        "quote_bid": 0.41,
        "quote_ask": 0.42,
        "quote_size": 50.0,
        "estimated_net_profit_usdc": 0.49,
    }
    payload.update(overrides)
    return payload


def _plan(**overrides) -> dict:
    payload = {
        "gate": _gate(),
        "deposit_wallet": {"status": "DEPOSIT_WALLET_READY", "available_usdc": 306.678811},
        "market_microstructure": _market(),
        "toxic_flow": _toxic(),
        "fee_reconciliation": _fee(),
        "inventory_state": {"status": "INVENTORY_STATE_CLEAR"},
        "order_mutex": {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"},
        "network": {"status": "API_HEARTBEAT_READY", "latency_ms": 98.7},
        "target_market_slug": MARKET,
        "max_live_risk_usdc": 296.67,
        "now": NOW,
    }
    payload.update(overrides)
    return build_live_probe_plan(**payload)


def test_ready_plan_is_read_only_and_token_bound() -> None:
    report = _plan()

    assert report["status"] == "LIVE_PROBE_PLAN_READY"
    assert report["planner_name"] == "NEXT_LIVE_PROBE_PLANNER"
    assert report["recommended_plan"]["quote_price"] == 0.41
    assert report["recommended_plan"]["quote_size"] == 50.0
    assert report["recommended_plan"]["capital_required_usdc"] == 20.5
    assert report["token_binding_required"] is True
    assert "planner_hash" in report["token_binding_fields"]
    assert isinstance(report["planner_hash"], str) and len(report["planner_hash"]) == 64
    assert report["hold_seconds"] == 300
    assert report["token_ttl_seconds"] == 600
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["requires_new_token"] is True
    assert report["same_token_retry_allowed"] is False
    assert report["pending_reward_counted_as_confirmed_reward"] is False


def test_planner_hash_is_stable_for_identical_inputs() -> None:
    first = _plan()
    second = _plan()

    assert first["planner_hash"] == second["planner_hash"]


def test_planner_hash_changes_when_token_bound_fields_change() -> None:
    base = _plan()
    changed_price = _plan(market_microstructure=_market(best_bid=0.42, best_ask=0.43, quote_bid=0.42, quote_ask=0.43), fee_reconciliation=_fee(quote_bid=0.42, quote_ask=0.43))
    changed_size = _plan(market_microstructure=_market(rewards_min_size=60.0, quote_size=60.0), fee_reconciliation=_fee(quote_size=60.0))
    changed_risk = _plan(max_live_risk_usdc=200.0)
    changed_hold = _plan(hold_seconds=600)

    assert changed_price["planner_hash"] != base["planner_hash"]
    assert changed_size["planner_hash"] != base["planner_hash"]
    assert changed_risk["planner_hash"] != base["planner_hash"]
    assert changed_hold["planner_hash"] != base["planner_hash"]


def test_planner_outputs_candidate_and_rejected_plan_lists() -> None:
    bad_market = _market(
        market_slug="unsafe-market",
        quote_bid=0.8,
        quote_ask=0.81,
        best_bid=0.8,
        best_ask=0.81,
        rewards_min_size=500.0,
        fee_reconciliation=_fee(market_slug="unsafe-market", quote_bid=0.8, quote_ask=0.81, quote_size=500.0),
        toxic_flow=_toxic(market_slug="unsafe-market", status="TOXIC_FLOW_BLOCKED", blockers=["ADVERSE_SELECTION_RISK"]),
    )

    report = _plan(candidate_market_microstructures=[bad_market])

    assert report["status"] == "LIVE_PROBE_PLAN_READY"
    assert len(report["candidate_plans"]) == 2
    assert report["recommended_plan"]["market_slug"] == MARKET
    assert report["rejected_plans"][0]["market_slug"] == "unsafe-market"
    assert "TOXIC_FLOW_NOT_CLEAR" in report["rejected_plans"][0]["blockers"]


def test_no_safe_candidate_when_fee_report_not_bound_to_plan() -> None:
    report = _plan(fee_reconciliation=_fee(quote_size=10.0))

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert report["recommended_plan"] is None
    assert "NO_SAFE_CANDIDATE" in report["blockers"]
    assert "FEE_REPORT_NOT_BOUND_TO_PLAN_PRICE_SIZE" in report["rejected_plans"][0]["blockers"]


def test_invalid_tick_size_rejects_candidate() -> None:
    report = _plan(
        market_microstructure=_market(checks={"quote_bid_tick_aligned": False, "quote_ask_tick_aligned": True, "price_bounds_ok": True})
    )

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert "QUOTE_BID_TICK_MISALIGNED" in report["rejected_plans"][0]["blockers"]


def test_reward_min_size_missing_rejects_candidate() -> None:
    market = _market()
    market.pop("rewards_min_size")
    report = _plan(market_microstructure=market)

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert "REWARD_MIN_SIZE_MISSING" in report["rejected_plans"][0]["blockers"]


def test_c_fill_probe_uses_tiny_size_without_reward_min_size_block() -> None:
    market = _market(rewards_min_size=50.0, quote_size=50.0)
    fee = _fee(quote_size=10.0)

    report = _plan(
        market_microstructure=market,
        fee_reconciliation=fee,
        probe_intent="C_FILL_LIKELIHOOD_RECONCILIATION",
        min_probe_size=10.0,
    )

    assert report["status"] == "LIVE_PROBE_PLAN_READY"
    assert report["recommended_plan"]["quote_size"] == 10.0
    assert report["recommended_plan"]["capital_required_usdc"] == 4.1
    assert report["recommended_plan"]["sizing_reason"] == "C_FILL_RECONCILIATION_TINY_SIZE_NOT_REWARD_ELIGIBLE"
    assert report["recommended_plan"]["checks"]["reward_min_size_check"] is False
    assert report["recommended_plan"]["checks"]["reward_min_size_required_for_probe"] is False
    assert "QUOTE_SIZE_BELOW_REWARD_MIN_SIZE" not in report["recommended_plan"]["blockers"]
    assert report["can_submit_order"] is False


def test_c_fill_probe_does_not_require_reward_min_size_metadata() -> None:
    market = _market()
    market.pop("rewards_min_size")

    report = _plan(
        market_microstructure=market,
        fee_reconciliation=_fee(quote_size=10.0),
        probe_intent="C_FILL_LIKELIHOOD_RECONCILIATION",
        min_probe_size=10.0,
    )

    assert report["status"] == "LIVE_PROBE_PLAN_READY"
    assert report["recommended_plan"]["quote_size"] == 10.0
    assert report["recommended_plan"]["reward_min_size"] is None
    assert "REWARD_MIN_SIZE_MISSING" not in report["recommended_plan"]["blockers"]


def test_toxic_unsafe_rejects_candidate() -> None:
    report = _plan(toxic_flow=_toxic(status="TOXIC_FLOW_BLOCKED", blockers=["ADVERSE_SELECTION_RISK"]))

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert "TOXIC_FLOW_NOT_CLEAR" in report["rejected_plans"][0]["blockers"]


def test_high_fill_probability_reclassifies_stability_probe_candidate() -> None:
    report = _plan(toxic_flow=_toxic(fill_probability=0.964083), stability_max_fill_probability=0.3)

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE_FOR_STABILITY"
    assert report["recommended_plan"] is None
    assert "NO_SAFE_CANDIDATE_FOR_STABILITY" in report["blockers"]
    assert report["reclassified_candidates"] == [MARKET]
    rejected = report["rejected_plans"][0]
    assert rejected["plan_classification"] == "PLAN_RECLASSIFIED_TO_FILL_LIKELIHOOD"
    assert rejected["reclassified_probe_intent"] == "C_FILL_LIKELIHOOD_RECONCILIATION"
    assert rejected["fill_probability"] == 0.964083
    assert rejected["stability_max_fill_probability"] == 0.3
    assert "FILL_PROBABILITY_TOO_HIGH_FOR_STABILITY_PROBE" in rejected["blockers"]
    assert rejected["checks"]["stability_fill_probability_check"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_missing_fill_probability_blocks_stability_classification() -> None:
    toxic = _toxic()
    toxic.pop("fill_probability")
    report = _plan(toxic_flow=toxic)

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert "FILL_PROBABILITY_MISSING_FOR_STABILITY_PROBE" in report["rejected_plans"][0]["blockers"]


def test_fee_unsafe_rejects_candidate() -> None:
    report = _plan(fee_reconciliation=_fee(status="FEE_BLOCKER", can_cover_fees=False))

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert "FEE_RECONCILIATION_NOT_READY" in report["rejected_plans"][0]["blockers"]


def test_global_blocker_when_gate_is_not_12_of_12() -> None:
    report = _plan(gate=_gate(status="LIVE_NOT_READY", asserts_passed=11, asserts_failed=1, blockers=["TOXIC_FLOW_NOT_READY"]))

    assert report["status"] == "LIVE_PROBE_PLAN_BLOCKED"
    assert "PRELIVE_12_OF_12_NOT_READY" in report["blockers"]
    assert report["recommended_plan"]["selection_result"] == "SELECTED"
    assert report["can_submit_order"] is False


def test_inventory_not_clear_blocks_ready_even_with_safe_candidate() -> None:
    report = _plan(inventory_state={"status": "INVENTORY_STATE_DIRTY", "token_balance_shares": 5.0})

    assert report["status"] == "LIVE_PROBE_PLAN_BLOCKED"
    assert "INVENTORY_NOT_CLEAR" in report["blockers"]


def test_mutex_not_clear_blocks_ready_even_with_safe_candidate() -> None:
    report = _plan(order_mutex={"status": "ORDER_MUTEX_BLOCKED", "order_mutex_state": "LIVE_ORDER_OPEN"})

    assert report["status"] == "LIVE_PROBE_PLAN_BLOCKED"
    assert "ORDER_MUTEX_NOT_CLEAR" in report["blockers"]


def test_capital_required_over_risk_rejects_candidate() -> None:
    report = _plan(
        market_microstructure=_market(best_bid=0.9, best_ask=0.91, quote_bid=0.9, quote_ask=0.91, rewards_min_size=400.0),
        fee_reconciliation=_fee(quote_bid=0.9, quote_ask=0.91, quote_size=400.0),
    )

    assert report["status"] == "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
    assert "CAPITAL_REQUIRED_EXCEEDS_MAX_LIVE_RISK" in report["rejected_plans"][0]["blockers"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "deposit_wallet_readonly_latest.json": {"status": "DEPOSIT_WALLET_READY", "available_usdc": 306.678811},
        "live_market_microstructure_latest.json": _market(),
        "toxic_flow_latest.json": _toxic(),
        "fee_reconciliation_latest.json": _fee(),
        "inventory_state_latest.json": {"status": "INVENTORY_STATE_CLEAR"},
        "order_mutex_readiness_latest.json": {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"},
        "live_network_readiness_latest.json": {"status": "API_HEARTBEAT_READY", "latency_ms": 98.7},
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "planner.json"
    md_out = tmp_path / "planner.md"

    rc = main(
        [
            "--reports-dir",
            str(reports),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
            "--target-market-slug",
            MARKET,
            "--max-live-risk-usdc",
            "296.67",
            "--planner-valid-seconds",
            "120",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == "LIVE_PROBE_PLAN_READY"
    assert payload["recommended_plan"]["quote_size"] == 50.0
    assert payload["can_submit_order"] is False
    assert md_out.exists()


def test_cli_defaults_c_fill_probe_size_to_ten(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "deposit_wallet_readonly_latest.json": {"status": "DEPOSIT_WALLET_READY", "available_usdc": 306.678811},
        "live_market_microstructure_latest.json": _market(),
        "toxic_flow_latest.json": _toxic(),
        "fee_reconciliation_latest.json": _fee(quote_size=10.0),
        "inventory_state_latest.json": {"status": "INVENTORY_STATE_CLEAR"},
        "order_mutex_readiness_latest.json": {"status": "ORDER_MUTEX_READY", "order_mutex_state": "NO_ORDER"},
        "live_network_readiness_latest.json": {"status": "API_HEARTBEAT_READY", "latency_ms": 98.7},
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "planner.json"
    md_out = tmp_path / "planner.md"

    rc = main(
        [
            "--reports-dir",
            str(reports),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
            "--target-market-slug",
            MARKET,
            "--max-live-risk-usdc",
            "296.67",
            "--probe-intent",
            "C_FILL_LIKELIHOOD_RECONCILIATION",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == "LIVE_PROBE_PLAN_READY"
    assert payload["recommended_plan"]["quote_size"] == 10.0
