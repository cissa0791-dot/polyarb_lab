from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_b_low_fill_stability_probe_approval_package import main
from src.live.b_low_fill_stability_probe_approval_package import (
    BLOCKED_STATUS,
    READY_STATUS,
    build_b_low_fill_stability_probe_approval_package,
)


NOW = datetime(2026, 5, 9, 11, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
HASH = "8c8fa0d0015b7df5f9cba314ca3318ab59bbffc8a6ab8dda9adb269f0b221218"


def _gate(**overrides) -> dict:
    payload = {
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


def _recommended(**overrides) -> dict:
    payload = {
        "market_slug": MARKET,
        "selected_side": "BID_ONLY",
        "plan_classification": "B_LONG_OBSERVATION_STABILITY",
        "quote_price": 0.38,
        "quote_bid": 0.38,
        "quote_ask": 0.42,
        "quote_size": 50.0,
        "capital_required_usdc": 19.0,
        "expected_max_loss_if_filled": 19.0,
        "reward_min_size": 50.0,
        "reward_max_spread_cents": 4.5,
        "fill_probability": 0.222222,
        "stability_max_fill_probability": 0.3,
        "checks": {
            "reward_min_size_check": True,
            "stability_fill_probability_check": True,
            "tick_size_check": True,
            "fee_check": True,
        },
    }
    payload.update(overrides)
    return payload


def _planner(**overrides) -> dict:
    payload = {
        "status": "LIVE_PROBE_PLAN_READY",
        "planner_hash": HASH,
        "requested_probe_intent": "B_LONG_OBSERVATION_STABILITY",
        "plan_classification": "B_LONG_OBSERVATION_STABILITY",
        "recommended_plan": _recommended(),
        "stability_max_fill_probability": 0.3,
        "hold_seconds": 300,
        "max_live_risk_usdc": 296.67,
        "token_binding_required": True,
        "requires_new_token": True,
        "requires_new_approval": True,
        "can_submit_order": False,
        "execution_authorized": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _search(**overrides) -> dict:
    payload = {
        "status": "LOW_FILL_STABILITY_CANDIDATE_SEARCH_READY",
        "safe_candidate_count": 1,
        "best_candidate": _recommended(),
        "can_submit_order": False,
        "execution_authorized": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _decision(**overrides) -> dict:
    payload = {
        "status": "SECOND_PROBE_DECISION_PACKAGE_READY",
        "recommended_option": "B_LONG_OBSERVATION_BID_ONLY_STABILITY_PROBE",
    }
    payload.update(overrides)
    return payload


def _package(**overrides) -> dict:
    payload = {
        "gate": _gate(),
        "planner": _planner(),
        "candidate_search": _search(),
        "second_probe_decision": _decision(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_b_low_fill_stability_probe_approval_package(**payload)


def test_ready_package_freezes_low_fill_b_stability_plan_without_execution() -> None:
    report = _package()

    assert report["status"] == READY_STATUS
    assert report["probe_type"] == "B_LONG_OBSERVATION_STABILITY"
    assert report["quote_price"] == 0.38
    assert report["quote_size"] == 50.0
    assert report["fill_probability_proxy"] == 0.222222
    assert report["fill_probability_is_model_estimate"] is True
    assert report["planner_hash"] == HASH
    assert report["hold_seconds"] == 300
    assert report["token_binding_required"] is True
    assert report["approval_boundary"]["token_ready"] is False
    assert report["approval_boundary"]["token_created"] is False
    assert report["approval_boundary"]["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["profitability_claimed"] is False
    assert report["pending_reward_counted_as_confirmed_reward"] is False


def test_token_binding_fields_include_planner_hash_and_exact_plan() -> None:
    report = _package()

    assert report["token_binding_fields"] == {
        "market_slug": MARKET,
        "selected_side": "BID_ONLY",
        "quote_price": 0.38,
        "quote_size": 50.0,
        "max_live_risk_usdc": 296.67,
        "hold_seconds": 300,
        "planner_hash": HASH,
    }


def test_blocks_when_planner_hash_missing() -> None:
    planner = _planner(planner_hash="")
    report = _package(planner=planner)

    assert report["status"] == BLOCKED_STATUS
    assert "PLANNER_HASH_MISSING_OR_INVALID" in report["blockers"]


def test_blocks_when_planner_reclassified_to_fill_likelihood() -> None:
    planner = _planner(
        status="LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE_FOR_STABILITY",
        plan_classification=None,
        recommended_plan=None,
    )
    report = _package(planner=planner)

    assert report["status"] == BLOCKED_STATUS
    assert "PLANNER_NOT_READY" in report["blockers"]
    assert "PLANNER_CLASSIFICATION_NOT_B_STABILITY" in report["blockers"]


def test_blocks_when_search_candidate_does_not_match_planner() -> None:
    search = _search(best_candidate=_recommended(quote_price=0.39, quote_bid=0.39))
    report = _package(candidate_search=search)

    assert report["status"] == BLOCKED_STATUS
    assert "SEARCH_BEST_CANDIDATE_DOES_NOT_MATCH_PLANNER" in report["blockers"]


def test_blocks_when_gate_or_planner_execution_flags_are_true() -> None:
    report = _package(
        gate=_gate(can_submit_order=True, live_order_sent=True),
        planner=_planner(can_submit_order=True, execution_authorized=True, live_order_sent=True),
    )

    assert report["status"] == BLOCKED_STATUS
    assert "GATE_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED" in report["blockers"]
    assert "GATE_LIVE_ORDER_SENT_TRUE_UNEXPECTED" in report["blockers"]
    assert "PLANNER_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED" in report["blockers"]
    assert "PLANNER_LIVE_ORDER_SENT_TRUE_UNEXPECTED" in report["blockers"]
    assert "PLANNER_EXECUTION_AUTHORIZED_TRUE_UNEXPECTED" in report["blockers"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "live_probe_planner_latest.json": _planner(),
        "live_probe_stability_candidate_search_latest.json": _search(),
        "second_probe_decision_package_latest.json": _decision(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "approval.json"
    md_out = tmp_path / "approval.md"

    rc = main(
        [
            "--reports-dir",
            str(reports),
            "--out",
            str(out),
            "--md-out",
            str(md_out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["quote_price"] == 0.38
    assert payload["approval_boundary"]["token_created"] is False
    assert "B Low-Fill Stability Probe Approval Package" in md_out.read_text(encoding="utf-8")
