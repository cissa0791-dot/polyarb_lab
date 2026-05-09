from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.build_b_stability_token_issuance_review import main
from src.live.b_stability_token_issuance_review import (
    BLOCKED_STATUS,
    READY_STATUS,
    build_b_stability_token_issuance_review,
)


NOW = datetime(2026, 5, 9, 12, 30, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"
HASH = "15ab13e972f39c8ab3a563a30b3d6034f9b3c8f2ea48190c314672aca8348d18"


def _approval(**overrides) -> dict:
    payload = {
        "status": "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_READY",
        "approval_package_id": "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE",
        "probe_type": "B_LONG_OBSERVATION_STABILITY",
        "market_slug": MARKET,
        "selected_side": "BID_ONLY",
        "quote_price": 0.38,
        "quote_size": 50.0,
        "hold_seconds": 300,
        "max_live_risk_usdc": 296.67,
        "fill_probability_proxy": 0.222222,
        "fill_probability_is_model_estimate": True,
        "stability_max_fill_probability": 0.3,
        "planner_hash": HASH,
        "token_binding_required": True,
        "token_binding_fields": {
            "market_slug": MARKET,
            "selected_side": "BID_ONLY",
            "quote_price": 0.38,
            "quote_size": 50.0,
            "max_live_risk_usdc": 296.67,
            "hold_seconds": 300,
            "planner_hash": HASH,
        },
        "approval_boundary": {
            "token_ready": False,
            "token_created": False,
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
        },
        "token_created": False,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "pending_reward_counted_as_confirmed_reward": False,
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
    }
    payload.update(overrides)
    return payload


def _recommended(**overrides) -> dict:
    payload = {
        "market_slug": MARKET,
        "selected_side": "BID_ONLY",
        "plan_classification": "B_LONG_OBSERVATION_STABILITY",
        "quote_price": 0.38,
        "quote_size": 50.0,
        "fill_probability": 0.222222,
    }
    payload.update(overrides)
    return payload


def _planner(**overrides) -> dict:
    payload = {
        "status": "LIVE_PROBE_PLAN_READY",
        "planner_hash": HASH,
        "planner_snapshot_ts": NOW.isoformat(),
        "planner_expires_at": (NOW + timedelta(minutes=2)).isoformat(),
        "plan_classification": "B_LONG_OBSERVATION_STABILITY",
        "recommended_plan": _recommended(),
        "hold_seconds": 300,
        "token_ttl_seconds": 600,
        "max_live_risk_usdc": 296.67,
        "requires_new_token": True,
        "requires_new_approval": True,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload.update(overrides)
    return payload


def _authorization(**overrides) -> dict:
    payload = {
        "status": "SINGLE_SIDE_PROBE_AUTHORIZATION_BLOCKED",
        "authorization_token_valid": False,
        "execution_release_ready": False,
        "token_status": None,
        "blockers": ["TOKEN_FILE_MISSING"],
    }
    payload.update(overrides)
    return payload


def _review(**overrides) -> dict:
    payload = {
        "approval_package": _approval(),
        "planner": _planner(),
        "authorization_report": _authorization(),
        "now": NOW,
    }
    payload.update(overrides)
    return build_b_stability_token_issuance_review(**payload)


def test_ready_review_allows_only_future_token_creation_after_operator_approval() -> None:
    report = _review()

    assert report["status"] == READY_STATUS
    assert report["token_issuance_review_ready"] is True
    assert report["token_may_be_created_after_explicit_operator_approval"] is True
    assert report["allowed_next_manual_action"] == "CREATE_ONE_TIME_TOKEN_AFTER_EXPLICIT_OPERATOR_APPROVAL_ONLY"
    assert report["token_created"] is False
    assert report["execution_authorized"] is False
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["approval_boundary"]["maker_both_sides_live_allowed"] is False
    assert report["approval_boundary"]["same_token_retry_allowed"] is False
    assert report["approval_boundary"]["same_approval_retry_allowed"] is False
    assert report["fill_probability_proxy"] == 0.222222
    assert report["token_binding_fields"] == {
        "market_slug": MARKET,
        "selected_side": "BID_ONLY",
        "quote_price": 0.38,
        "quote_size": 50.0,
        "max_live_risk_usdc": 296.67,
        "hold_seconds": 300,
        "token_ttl_seconds": 600,
        "planner_hash": HASH,
    }


def test_blocks_if_approval_package_is_not_ready() -> None:
    report = _review(approval_package=_approval(status="B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_BLOCKED"))

    assert report["status"] == BLOCKED_STATUS
    assert "B_APPROVAL_PACKAGE_NOT_READY" in report["blockers"]


def test_blocks_if_planner_hash_does_not_match_approval_binding() -> None:
    report = _review(planner=_planner(planner_hash="a" * 64))

    assert report["status"] == BLOCKED_STATUS
    assert "TOKEN_BINDING_PLANNER_HASH_MISMATCH" in report["blockers"]


def test_blocks_if_fill_probability_or_classification_is_not_b_stability() -> None:
    approval = _approval(fill_probability_proxy=0.93)
    planner = _planner(plan_classification="PLAN_RECLASSIFIED_TO_FILL_LIKELIHOOD")
    report = _review(approval_package=approval, planner=planner)

    assert report["status"] == BLOCKED_STATUS
    assert "PLANNER_CLASSIFICATION_NOT_B_STABILITY" in report["blockers"]
    assert "FILL_PROBABILITY_PROXY_TOO_HIGH_FOR_B_STABILITY" in report["blockers"]


def test_blocks_if_token_ttl_is_missing_or_shorter_than_hold() -> None:
    missing = _review(planner=_planner(token_ttl_seconds=None))
    short = _review(planner=_planner(token_ttl_seconds=100))

    assert missing["status"] == BLOCKED_STATUS
    assert "TOKEN_TTL_SECONDS_MISSING_OR_INVALID" in missing["blockers"]
    assert short["status"] == BLOCKED_STATUS
    assert "TOKEN_TTL_SHORTER_THAN_HOLD_WINDOW" in short["blockers"]


def test_blocks_if_active_authorization_token_already_exists() -> None:
    report = _review(
        authorization_report=_authorization(
            status="SINGLE_SIDE_PROBE_AUTHORIZATION_READY",
            authorization_token_valid=True,
            execution_release_ready=True,
            token_status="ISSUED_UNUSED",
            blockers=[],
        )
    )

    assert report["status"] == BLOCKED_STATUS
    assert "ACTIVE_AUTHORIZATION_TOKEN_ALREADY_VALID" in report["blockers"]
    assert "ACTIVE_EXECUTION_RELEASE_ALREADY_READY" in report["blockers"]


def test_blocks_if_any_execution_boundary_is_open() -> None:
    approval = _approval(
        token_created=True,
        execution_authorized=True,
        can_submit_order=True,
        live_order_sent=True,
        approval_boundary={
            "token_ready": True,
            "token_created": True,
            "same_token_retry_allowed": True,
            "same_approval_retry_allowed": True,
        },
    )
    planner = _planner(execution_authorized=True, can_submit_order=True, live_order_sent=True)

    report = _review(approval_package=approval, planner=planner)

    assert report["status"] == BLOCKED_STATUS
    assert "APPROVAL_TOKEN_CREATED_UNEXPECTED" in report["blockers"]
    assert "APPROVAL_EXECUTION_AUTHORIZED_UNEXPECTED" in report["blockers"]
    assert "APPROVAL_CAN_SUBMIT_ORDER_UNEXPECTED" in report["blockers"]
    assert "APPROVAL_LIVE_ORDER_SENT_UNEXPECTED" in report["blockers"]
    assert "APPROVAL_BOUNDARY_TOKEN_READY_UNEXPECTED" in report["blockers"]
    assert "APPROVAL_BOUNDARY_TOKEN_CREATED_UNEXPECTED" in report["blockers"]
    assert "SAME_TOKEN_RETRY_NOT_CLOSED" in report["blockers"]
    assert "SAME_APPROVAL_RETRY_NOT_CLOSED" in report["blockers"]
    assert "PLANNER_EXECUTION_AUTHORIZED_UNEXPECTED" in report["blockers"]
    assert "PLANNER_CAN_SUBMIT_ORDER_UNEXPECTED" in report["blockers"]
    assert "PLANNER_LIVE_ORDER_SENT_UNEXPECTED" in report["blockers"]


def test_cli_writes_review_json_and_markdown(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    files = {
        "b_low_fill_stability_probe_approval_package_latest.json": _approval(),
        "live_probe_planner_latest.json": _planner(),
        "single_side_probe_authorization_latest.json": _authorization(),
    }
    for filename, payload in files.items():
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "review.json"
    md_out = tmp_path / "review.md"

    rc = main(["--reports-dir", str(reports), "--out", str(out), "--md-out", str(md_out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == READY_STATUS
    assert payload["token_created"] is False
    assert payload["execution_authorized"] is False
    assert "B Stability Token Issuance Review" in md_out.read_text(encoding="utf-8")
