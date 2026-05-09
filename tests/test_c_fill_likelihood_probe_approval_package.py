from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_c_fill_likelihood_probe_approval_package import main
from src.live.c_fill_likelihood_probe_approval_package import (
    BLOCKED_STATUS,
    READY_STATUS,
    build_c_fill_likelihood_probe_approval_package,
)


NOW = datetime(2026, 5, 9, 13, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
HASH = "a" * 64


def _gate(**overrides) -> dict:
    payload = {
        "status": "LIVE_READY_APPROVED",
        "asserts_passed": 12,
        "asserts_failed": 0,
        "target_market_slug": MARKET,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _plan(**overrides) -> dict:
    payload = {
        "market_slug": MARKET,
        "selected_side": "BID_ONLY",
        "plan_classification": "C_FILL_LIKELIHOOD_RECONCILIATION",
        "quote_price": 0.41,
        "quote_size": 5.0,
        "fill_probability": 0.91,
    }
    payload.update(overrides)
    return payload


def _planner(**overrides) -> dict:
    payload = {
        "status": "LIVE_PROBE_PLAN_READY",
        "plan_classification": "C_FILL_LIKELIHOOD_RECONCILIATION",
        "recommended_plan": _plan(),
        "planner_hash": HASH,
        "can_submit_order": False,
        "execution_authorized": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _fill_readiness(**overrides) -> dict:
    payload = {
        "status": "FILL_RECONCILIATION_READINESS_READY",
        "fill_detectable": True,
        "partial_fill_detectable": True,
        "inventory_update_source_ready": True,
        "cash_delta_source_ready": True,
        "fee_reconciliation_ready": True,
        "post_fill_audit_required": True,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _fee(**overrides) -> dict:
    payload = {"status": "FEE_RECONCILIATION_READY", "can_cover_fees": True}
    payload.update(overrides)
    return payload


def _toxic(**overrides) -> dict:
    payload = {"status": "TOXIC_FLOW_READY", "blockers": []}
    payload.update(overrides)
    return payload


def _package(**overrides) -> dict:
    payload = {
        "gate": _gate(),
        "planner": _planner(),
        "fill_readiness": _fill_readiness(),
        "fee_reconciliation": _fee(),
        "toxic_flow": _toxic(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_c_fill_likelihood_probe_approval_package(**payload)


def test_ready_package_is_review_only_and_small_one_side() -> None:
    report = _package()

    assert report["status"] == READY_STATUS
    assert report["probe_type"] == "C_FILL_LIKELIHOOD_RECONCILIATION"
    assert report["quote_size"] == 5.0
    assert report["constraints"]["one_side_only"] is True
    assert report["constraints"]["one_order_only"] is True
    assert report["constraints"]["no_retry"] is True
    assert report["constraints"]["both_side_live"] is False
    assert report["constraints"]["strict_exact_cancel_of_remainder"] is True
    assert report["approval_boundary"]["token_created"] is False
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_blocks_when_c_size_is_not_small_without_override() -> None:
    planner = _planner(recommended_plan=_plan(quote_size=50.0))
    report = _package(planner=planner, max_c_probe_size=10.0)

    assert report["status"] == BLOCKED_STATUS
    assert "C_PROBE_SIZE_EXCEEDS_SMALL_SIZE_LIMIT" in report["blockers"]


def test_allows_larger_size_only_with_explicit_override() -> None:
    planner = _planner(recommended_plan=_plan(quote_size=50.0))
    report = _package(planner=planner, max_c_probe_size=10.0, operator_size_override_approved=True)

    assert report["status"] == READY_STATUS
    assert report["operator_size_override_approved"] is True


def test_blocks_if_fill_readiness_missing() -> None:
    report = _package(fill_readiness={})

    assert report["status"] == BLOCKED_STATUS
    assert "FILL_RECONCILIATION_READINESS_NOT_READY" in report["blockers"]


def test_blocks_if_both_side_or_wrong_classification() -> None:
    planner = _planner(
        plan_classification="B_LONG_OBSERVATION_STABILITY",
        recommended_plan=_plan(selected_side="BOTH_SIDES", plan_classification="B_LONG_OBSERVATION_STABILITY"),
    )
    report = _package(planner=planner)

    assert report["status"] == BLOCKED_STATUS
    assert "PLANNER_CLASSIFICATION_NOT_C_FILL_LIKELIHOOD" in report["blockers"]
    assert "C_PROBE_SIDE_NOT_SINGLE_SIDE" in report["blockers"]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "live_readiness_gate_latest.json": _gate(),
        "live_probe_planner_latest.json": _planner(),
        "fill_reconciliation_readiness_latest.json": _fill_readiness(),
        "fee_reconciliation_latest.json": _fee(),
        "toxic_flow_latest.json": _toxic(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "c_package.json"
    md_out = tmp_path / "c_package.md"

    rc = main(["--reports-dir", str(reports), "--out", str(out), "--md-out", str(md_out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert "C Fill-Likelihood" in md_out.read_text(encoding="utf-8")
