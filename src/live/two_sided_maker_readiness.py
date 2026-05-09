from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REPORT_SCHEMA_VERSION = "two_sided_maker_readiness.v1"
REPORT_TYPE = "two_sided_maker_readiness"

BLOCKED_STATUS = "TWO_SIDED_MAKER_READINESS_BLOCKED"


def build_two_sided_maker_readiness(
    *,
    single_side_rehearsal: dict[str, Any] | None = None,
    inventory_model: dict[str, Any] | None = None,
    split_merge_redeem: dict[str, Any] | None = None,
    simultaneous_fill_risk: dict[str, Any] | None = None,
    two_sided_mutex: dict[str, Any] | None = None,
    post_fill_audit: dict[str, Any] | None = None,
    approval_context: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Produce a preparation-only two-sided maker readiness report.

    This intentionally remains blocked until single-side continuous evidence
    and all two-sided physical prerequisites are proven.
    """

    now = now or datetime.now(timezone.utc)
    single_side_rehearsal = single_side_rehearsal or {}
    inventory_model = inventory_model or {}
    split_merge_redeem = split_merge_redeem or {}
    simultaneous_fill_risk = simultaneous_fill_risk or {}
    two_sided_mutex = two_sided_mutex or {}
    post_fill_audit = post_fill_audit or {}
    approval_context = approval_context or {}
    blockers = _blockers(
        single_side_rehearsal=single_side_rehearsal,
        inventory_model=inventory_model,
        split_merge_redeem=split_merge_redeem,
        simultaneous_fill_risk=simultaneous_fill_risk,
        two_sided_mutex=two_sided_mutex,
        post_fill_audit=post_fill_audit,
        approval_context=approval_context,
    )
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": BLOCKED_STATUS,
        "preparation_only": True,
        "requirements": {
            "yes_no_inventory_model": inventory_model.get("status"),
            "split_merge_redeem_model": split_merge_redeem.get("status"),
            "simultaneous_fill_risk_model": simultaneous_fill_risk.get("status"),
            "two_sided_mutex_cancel_race_rules": two_sided_mutex.get("status"),
            "two_sided_post_fill_audit": post_fill_audit.get("status"),
        },
        "execution_boundary": {
            "two_sided_token_allowed": False,
            "single_side_approval_can_create_two_sided_token": False,
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
        },
        "inventory_skew_requires_reconciliation": True,
        "dual_fill_requires_explicit_reconciliation": True,
        "blockers": blockers,
        "one_line_verdict": f"TWO_SIDED_MAKER_READINESS_BLOCKED: {', '.join(blockers) or 'two-sided live remains intentionally unavailable'}.",
    }


def _blockers(
    *,
    single_side_rehearsal: dict[str, Any],
    inventory_model: dict[str, Any],
    split_merge_redeem: dict[str, Any],
    simultaneous_fill_risk: dict[str, Any],
    two_sided_mutex: dict[str, Any],
    post_fill_audit: dict[str, Any],
    approval_context: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if single_side_rehearsal.get("status") != "SINGLE_SIDE_CONTINUOUS_REHEARSAL_READY":
        blockers.append("SINGLE_SIDE_CONTINUOUS_EVIDENCE_NOT_GREEN")
    for payload, blocker in [
        (inventory_model, "YES_NO_INVENTORY_MODEL_NOT_PROVEN"),
        (split_merge_redeem, "SPLIT_MERGE_REDEEM_REQUIREMENTS_NOT_PROVEN"),
        (simultaneous_fill_risk, "SIMULTANEOUS_FILL_RISK_MODEL_NOT_PROVEN"),
        (two_sided_mutex, "TWO_SIDED_MUTEX_CANCEL_RACE_NOT_PROVEN"),
        (post_fill_audit, "TWO_SIDED_POST_FILL_AUDIT_NOT_PROVEN"),
    ]:
        if payload.get("status") != "READY":
            blockers.append(blocker)
    if approval_context.get("approval_type") == "SINGLE_SIDE":
        blockers.append("SINGLE_SIDE_APPROVAL_CANNOT_AUTHORIZE_TWO_SIDED")
    return blockers
