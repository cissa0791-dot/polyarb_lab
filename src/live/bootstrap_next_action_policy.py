from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


POLICY_MAPPING: dict[str, dict[str, Any]] = {
    "NULL_RESULT_BOOTSTRAP_EVENT": {
        "next_action_recommendation": "REQUEST_NEW_APPROVAL",
        "same_approval_retry_allowed": False,
        "new_approval_required": True,
        "auto_followup_allowed": False,
        "line_status": "CLOSED_NULL_RESULT_AWAITING_NEW_APPROVAL",
        "reason": "Bootstrap attempt closed without a fill or profitability evidence; the one-time approval is consumed.",
    },
    "OPEN_ORDER_UNRESOLVED": {
        "next_action_recommendation": "WAIT_FOR_ORDER_RESOLUTION",
        "same_approval_retry_allowed": False,
        "new_approval_required": False,
        "auto_followup_allowed": True,
        "line_status": "OPEN_ORDER_RESOLUTION_PENDING",
        "reason": "Open order exposure remains unresolved; wait for fill, cancel, expire, or manual resolution evidence.",
    },
    "PROFITABILITY_EVIDENCE_OBTAINED": {
        "next_action_recommendation": "ENTER_PROFITABILITY_VALIDATION",
        "same_approval_retry_allowed": False,
        "new_approval_required": False,
        "auto_followup_allowed": True,
        "line_status": "READY_FOR_PROFITABILITY_VALIDATION",
        "reason": "A closed-cycle evidence set is available; move to normal profitability validation without claiming profitability in this policy report.",
    },
    "PARTIAL_FILL_UNRESOLVED": {
        "next_action_recommendation": "HOLD_FOR_MANUAL_REVIEW",
        "same_approval_retry_allowed": False,
        "new_approval_required": False,
        "auto_followup_allowed": False,
        "line_status": "PARTIAL_FILL_MANUAL_REVIEW_REQUIRED",
        "reason": "Partial fill or inventory with unresolved order state requires manual review before any automated follow-up.",
    },
    "PROFITABILITY_FAILURE_CONFIRMED": {
        "next_action_recommendation": "PARK",
        "same_approval_retry_allowed": False,
        "new_approval_required": False,
        "auto_followup_allowed": False,
        "line_status": "PARKED_AFTER_CONFIRMED_PROFITABILITY_FAILURE",
        "reason": "Profitability failure is confirmed; park the line and do not request retry by default.",
    },
}


def build_bootstrap_next_action_policy(
    *,
    closeout: dict[str, Any] | None = None,
    evidence_validation: dict[str, Any] | None = None,
    approval: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    execution_system: dict[str, Any] | None = None,
) -> dict[str, Any]:
    closeout = closeout or {}
    evidence_validation = evidence_validation or {}
    approval = approval or {}
    health = health or {}
    execution_system = execution_system or {}

    closeout_class = classify_closeout(
        closeout=closeout,
        evidence_validation=evidence_validation,
    )
    decision = POLICY_MAPPING.get(closeout_class, _manual_review_mapping(closeout_class))
    execution_enabled = execution_system.get("execution_enabled") is True
    live_order_sent = any(
        item is True
        for item in (
            closeout.get("live_order_sent"),
            evidence_validation.get("live_order_sent"),
            approval.get("live_order_sent"),
            execution_system.get("live_order_sent"),
        )
    )

    return {
        "report_type": "bootstrap_next_action_policy",
        "report_schema_version": "bootstrap_next_action_policy.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "live_decision_binding": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "policy_scope": "POST_CLOSEOUT_BOOTSTRAP_NEXT_ACTION_ONLY",
        "closeout_class": closeout_class,
        "next_action_recommendation": decision["next_action_recommendation"],
        "next_action_reason": decision["reason"],
        "same_approval_retry_allowed": decision["same_approval_retry_allowed"],
        "new_approval_required": decision["new_approval_required"],
        "auto_followup_allowed": decision["auto_followup_allowed"],
        "line_status": decision["line_status"],
        "normal_profitability_validation_allowed": closeout_class == "PROFITABILITY_EVIDENCE_OBTAINED",
        "profitability_claimed": False,
        "approval_status": approval.get("approval_status") or closeout.get("approval_status"),
        "approval_consumed": approval.get("consumed") is True or closeout.get("approval_consumed") is True,
        "execution_disabled": execution_system.get("execution_enabled") is False,
        "execution_blockers": execution_system.get("execution_blockers") or closeout.get("execution_blockers") or [],
        "health_status": health.get("status"),
        "source_state": {
            "attempt_id": closeout.get("attempt_id"),
            "attempt_closeout_classification": closeout.get("attempt_closeout_classification"),
            "current_order_state": closeout.get("current_order_state"),
            "cycle_closed": evidence_validation.get("cycle_closed", closeout.get("cycle_closed")),
            "evidence_complete": evidence_validation.get("evidence_complete", closeout.get("evidence_complete")),
            "open_buy_exposure_present": closeout.get("open_buy_exposure_present"),
            "token_balance_shares": closeout.get("token_balance_shares"),
            "profitability_failure": closeout.get("profitability_failure"),
        },
        "safety_invariants": {
            "does_not_execute": True,
            "does_not_consume_approval": True,
            "does_not_change_profitability_math": True,
            "same_approval_retry_blocked_when_consumed": not (
                (approval.get("consumed") is True or closeout.get("approval_consumed") is True)
                and decision["same_approval_retry_allowed"] is True
            ),
            "execution_was_not_enabled_by_policy": not execution_enabled,
            "live_order_was_not_sent_by_policy": not live_order_sent,
        },
        "supported_closeout_classes": sorted(POLICY_MAPPING),
        "policy_mapping": POLICY_MAPPING,
        "one_line_verdict": (
            f"{decision['next_action_recommendation']}: {decision['line_status']} "
            f"for {closeout_class}."
        ),
    }


def classify_closeout(
    *,
    closeout: dict[str, Any],
    evidence_validation: dict[str, Any],
) -> str:
    explicit = str(closeout.get("attempt_closeout_classification") or "").strip()
    if explicit in POLICY_MAPPING:
        return explicit

    if closeout.get("profitability_failure") is True or explicit == "PROFITABILITY_FAILURE_CONFIRMED":
        return "PROFITABILITY_FAILURE_CONFIRMED"

    cycle_closed = evidence_validation.get("cycle_closed") is True or closeout.get("cycle_closed") is True
    evidence_complete = evidence_validation.get("evidence_complete") is True or closeout.get("evidence_complete") is True
    if cycle_closed and evidence_complete:
        return "PROFITABILITY_EVIDENCE_OBTAINED"

    current_order_state = str(closeout.get("current_order_state") or "").upper()
    token_balance = _float(closeout.get("token_balance_shares"))
    open_buy_exposure = closeout.get("open_buy_exposure_present") is True

    if current_order_state in {"STILL_OPEN_PARTIAL_FILL", "PARTIALLY_FILLED"} or (
        token_balance > 0.0 and open_buy_exposure
    ):
        return "PARTIAL_FILL_UNRESOLVED"
    if current_order_state in {"STILL_OPEN_UNFILLED", "STILL_OPEN_UNKNOWN_FILL", "OPEN_BUY_ORDER_UNRESOLVED"} or open_buy_exposure:
        return "OPEN_ORDER_UNRESOLVED"

    return explicit or "UNKNOWN_CLOSEOUT_CLASS"


def _manual_review_mapping(closeout_class: str) -> dict[str, Any]:
    return {
        "next_action_recommendation": "HOLD_FOR_MANUAL_REVIEW",
        "same_approval_retry_allowed": False,
        "new_approval_required": False,
        "auto_followup_allowed": False,
        "line_status": "UNKNOWN_CLOSEOUT_CLASS_MANUAL_REVIEW_REQUIRED",
        "reason": f"Unsupported or ambiguous closeout class requires review: {closeout_class}.",
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
