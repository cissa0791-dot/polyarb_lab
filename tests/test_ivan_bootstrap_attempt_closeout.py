from __future__ import annotations

from src.live.ivan_bootstrap_attempt_closeout import build_ivan_bootstrap_attempt_closeout


def _approval_consumed() -> dict:
    return {
        "approval_status": "ONE_TIME_BOOTSTRAP_EXECUTION_CONSUMED",
        "consumed": True,
        "consumed_at_utc": "2026-05-06T10:14:07+00:00",
    }


def _lifecycle_manual_resolution() -> dict:
    return {
        "resolution_state": "MANUALLY_RESOLVED_EXTERNALLY",
        "manual_action_required": False,
        "order_lifecycle": {
            "token_balance_shares": 0.0,
            "token_open_order_count": 0,
            "account_open_buy_usdc": 0.0,
        },
    }


def _resolution_manual_external() -> dict:
    return {
        "current_order_state": "MANUALLY_RESOLVED_EXTERNALLY",
        "manual_action_required": False,
        "cycle_closed": False,
        "evidence_complete": False,
        "supporting_status_summary": {
            "token_balance_shares": 0.0,
            "account_open_buy_usdc": 0.0,
            "token_open_order_count": 0,
        },
    }


def _validation_incomplete() -> dict:
    return {
        "cycle_closed": False,
        "evidence_complete": False,
        "profitability_claimed": False,
        "one_line_verdict": "FIRST_CYCLE_EVIDENCE_INCOMPLETE_REMAIN_IN_EVIDENCE_STAGE",
    }


def _healthy_zero_open_buy() -> dict:
    return {
        "status": "HEALTHY",
        "checks": {
            "account_open_orders": {
                "open_buy_order_count": 0,
                "open_buy_usdc": 0.0,
            }
        },
    }


def _execution_disabled() -> dict:
    return {
        "execution_enabled": False,
        "execution_blockers": ["ID_EXECUTION_DISABLED"],
        "can_submit_order": False,
        "live_order_sent": False,
    }


def test_manual_cancel_without_fill_closes_attempt_as_null_result_not_profitability() -> None:
    report = build_ivan_bootstrap_attempt_closeout(
        approval=_approval_consumed(),
        lifecycle=_lifecycle_manual_resolution(),
        resolution=_resolution_manual_external(),
        validation=_validation_incomplete(),
        health=_healthy_zero_open_buy(),
        execution_system=_execution_disabled(),
    )

    assert report["attempt_id"] == "IVAN_BOOTSTRAP_ATTEMPT_1"
    assert report["attempt_closeout_classification"] == "NULL_RESULT_BOOTSTRAP_EVENT"
    assert report["approval_consumed"] is True
    assert report["order_produced_fill"] is False
    assert report["no_inventory_remains"] is True
    assert report["open_buy_exposure_present"] is False
    assert report["closed_cycle_exists"] is False
    assert report["profitability_evidence_created"] is False
    assert report["profitability_success"] is False
    assert report["profitability_failure"] is False
    assert report["profitability_validation_allowed"] is False
    assert report["retry_allowed_under_consumed_approval"] is False
    assert report["evidence_stage_active"] is True
    assert report["execution_disabled"] is True
    assert report["blocking_reasons"] == []


def test_attempt_with_inventory_or_fill_is_not_null_result() -> None:
    lifecycle = _lifecycle_manual_resolution()
    lifecycle["order_lifecycle"]["token_balance_shares"] = 5.0
    resolution = _resolution_manual_external()
    resolution["supporting_status_summary"]["token_balance_shares"] = 5.0

    report = build_ivan_bootstrap_attempt_closeout(
        approval=_approval_consumed(),
        lifecycle=lifecycle,
        resolution=resolution,
        validation=_validation_incomplete(),
        health=_healthy_zero_open_buy(),
        execution_system=_execution_disabled(),
    )

    assert report["attempt_closeout_classification"] == "BOOTSTRAP_ATTEMPT_CLOSEOUT_REVIEW_REQUIRED"
    assert report["order_produced_fill"] is True
    assert "ORDER_PRODUCED_FILL_NOT_NULL_RESULT" in report["blocking_reasons"]
    assert "INVENTORY_REMAINS" in report["blocking_reasons"]


def test_unconsumed_approval_cannot_close_consumed_attempt() -> None:
    approval = {"approval_status": "APPROVED_UNUSED", "consumed": False}

    report = build_ivan_bootstrap_attempt_closeout(
        approval=approval,
        lifecycle=_lifecycle_manual_resolution(),
        resolution=_resolution_manual_external(),
        validation=_validation_incomplete(),
        health=_healthy_zero_open_buy(),
        execution_system=_execution_disabled(),
    )

    assert report["attempt_closeout_classification"] == "BOOTSTRAP_ATTEMPT_CLOSEOUT_REVIEW_REQUIRED"
    assert report["approval_consumed"] is False
    assert report["retry_allowed_under_consumed_approval"] is None
    assert "APPROVAL_NOT_CONSUMED" in report["blocking_reasons"]
