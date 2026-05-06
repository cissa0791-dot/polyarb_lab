from __future__ import annotations

import pytest

from src.live.bootstrap_next_action_policy import build_bootstrap_next_action_policy


@pytest.mark.parametrize(
    ("closeout_class", "action", "line_status", "new_approval", "auto_followup"),
    [
        (
            "NULL_RESULT_BOOTSTRAP_EVENT",
            "REQUEST_NEW_APPROVAL",
            "CLOSED_NULL_RESULT_AWAITING_NEW_APPROVAL",
            True,
            False,
        ),
        (
            "OPEN_ORDER_UNRESOLVED",
            "WAIT_FOR_ORDER_RESOLUTION",
            "OPEN_ORDER_RESOLUTION_PENDING",
            False,
            True,
        ),
        (
            "PROFITABILITY_EVIDENCE_OBTAINED",
            "ENTER_PROFITABILITY_VALIDATION",
            "READY_FOR_PROFITABILITY_VALIDATION",
            False,
            True,
        ),
        (
            "PARTIAL_FILL_UNRESOLVED",
            "HOLD_FOR_MANUAL_REVIEW",
            "PARTIAL_FILL_MANUAL_REVIEW_REQUIRED",
            False,
            False,
        ),
        (
            "PROFITABILITY_FAILURE_CONFIRMED",
            "PARK",
            "PARKED_AFTER_CONFIRMED_PROFITABILITY_FAILURE",
            False,
            False,
        ),
    ],
)
def test_policy_maps_supported_closeout_classes(
    closeout_class: str,
    action: str,
    line_status: str,
    new_approval: bool,
    auto_followup: bool,
) -> None:
    report = build_bootstrap_next_action_policy(
        closeout={"attempt_closeout_classification": closeout_class, "approval_consumed": True},
        evidence_validation={"cycle_closed": False, "evidence_complete": False},
        approval={"approval_status": "ONE_TIME_BOOTSTRAP_EXECUTION_CONSUMED", "consumed": True},
        health={"status": "HEALTHY"},
        execution_system={"execution_enabled": False, "execution_blockers": ["ID_EXECUTION_DISABLED"]},
    )

    assert report["closeout_class"] == closeout_class
    assert report["next_action_recommendation"] == action
    assert report["line_status"] == line_status
    assert report["same_approval_retry_allowed"] is False
    assert report["new_approval_required"] is new_approval
    assert report["auto_followup_allowed"] is auto_followup
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False
    assert report["safety_invariants"]["does_not_execute"] is True
    assert report["safety_invariants"]["does_not_consume_approval"] is True


def test_current_null_result_requests_new_approval_without_retrying_same_approval() -> None:
    report = build_bootstrap_next_action_policy(
        closeout={
            "attempt_id": "IVAN_BOOTSTRAP_ATTEMPT_1",
            "attempt_closeout_classification": "NULL_RESULT_BOOTSTRAP_EVENT",
            "approval_consumed": True,
            "current_order_state": "MANUALLY_RESOLVED_EXTERNALLY",
            "open_buy_exposure_present": False,
            "token_balance_shares": 0.0,
            "cycle_closed": False,
            "evidence_complete": False,
        },
        evidence_validation={"cycle_closed": False, "evidence_complete": False},
        approval={"approval_status": "ONE_TIME_BOOTSTRAP_EXECUTION_CONSUMED", "consumed": True},
        health={"status": "HEALTHY"},
        execution_system={"execution_enabled": False, "execution_blockers": ["ID_EXECUTION_DISABLED"]},
    )

    assert report["next_action_recommendation"] == "REQUEST_NEW_APPROVAL"
    assert report["next_action_reason"].startswith("Bootstrap attempt closed without a fill")
    assert report["same_approval_retry_allowed"] is False
    assert report["new_approval_required"] is True
    assert report["auto_followup_allowed"] is False
    assert report["normal_profitability_validation_allowed"] is False
    assert report["profitability_claimed"] is False


def test_closed_complete_evidence_infers_profitability_evidence_obtained() -> None:
    report = build_bootstrap_next_action_policy(
        closeout={"approval_consumed": True},
        evidence_validation={"cycle_closed": True, "evidence_complete": True},
        approval={"consumed": True},
        execution_system={"execution_enabled": False},
    )

    assert report["closeout_class"] == "PROFITABILITY_EVIDENCE_OBTAINED"
    assert report["next_action_recommendation"] == "ENTER_PROFITABILITY_VALIDATION"
    assert report["normal_profitability_validation_allowed"] is True
    assert report["profitability_claimed"] is False


def test_open_buy_exposure_infers_open_order_unresolved() -> None:
    report = build_bootstrap_next_action_policy(
        closeout={"open_buy_exposure_present": True, "current_order_state": "STILL_OPEN_UNFILLED"},
        evidence_validation={"cycle_closed": False, "evidence_complete": False},
        approval={"consumed": True},
        execution_system={"execution_enabled": False},
    )

    assert report["closeout_class"] == "OPEN_ORDER_UNRESOLVED"
    assert report["next_action_recommendation"] == "WAIT_FOR_ORDER_RESOLUTION"


def test_partial_fill_with_open_exposure_infers_manual_review() -> None:
    report = build_bootstrap_next_action_policy(
        closeout={"open_buy_exposure_present": True, "token_balance_shares": 1.5},
        evidence_validation={"cycle_closed": False, "evidence_complete": False},
        approval={"consumed": True},
        execution_system={"execution_enabled": False},
    )

    assert report["closeout_class"] == "PARTIAL_FILL_UNRESOLVED"
    assert report["next_action_recommendation"] == "HOLD_FOR_MANUAL_REVIEW"
