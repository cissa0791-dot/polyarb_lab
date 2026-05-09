from __future__ import annotations

import json

from scripts.build_two_sided_maker_readiness_report import main
from src.live.two_sided_maker_readiness import BLOCKED_STATUS, build_two_sided_maker_readiness


def _ready_single_side() -> dict:
    return {"status": "SINGLE_SIDE_CONTINUOUS_REHEARSAL_READY"}


def test_two_sided_mode_remains_blocked_without_single_side_green_evidence() -> None:
    report = build_two_sided_maker_readiness(single_side_rehearsal={"status": "BLOCKED"})

    assert report["status"] == BLOCKED_STATUS
    assert "SINGLE_SIDE_CONTINUOUS_EVIDENCE_NOT_GREEN" in report["blockers"]
    assert report["execution_boundary"]["two_sided_token_allowed"] is False


def test_inventory_skew_and_dual_fill_require_explicit_reconciliation() -> None:
    report = build_two_sided_maker_readiness(
        single_side_rehearsal=_ready_single_side(),
        inventory_model={"status": "READY"},
        split_merge_redeem={"status": "READY"},
        simultaneous_fill_risk={"status": "BLOCKED"},
        two_sided_mutex={"status": "READY"},
        post_fill_audit={"status": "BLOCKED"},
    )

    assert report["status"] == BLOCKED_STATUS
    assert report["inventory_skew_requires_reconciliation"] is True
    assert report["dual_fill_requires_explicit_reconciliation"] is True
    assert "SIMULTANEOUS_FILL_RISK_MODEL_NOT_PROVEN" in report["blockers"]
    assert "TWO_SIDED_POST_FILL_AUDIT_NOT_PROVEN" in report["blockers"]


def test_no_two_sided_token_can_be_created_from_single_side_approval() -> None:
    report = build_two_sided_maker_readiness(
        single_side_rehearsal=_ready_single_side(),
        inventory_model={"status": "READY"},
        split_merge_redeem={"status": "READY"},
        simultaneous_fill_risk={"status": "READY"},
        two_sided_mutex={"status": "READY"},
        post_fill_audit={"status": "READY"},
        approval_context={"approval_type": "SINGLE_SIDE"},
    )

    assert report["status"] == BLOCKED_STATUS
    assert "SINGLE_SIDE_APPROVAL_CANNOT_AUTHORIZE_TWO_SIDED" in report["blockers"]
    assert report["execution_boundary"]["single_side_approval_can_create_two_sided_token"] is False


def test_cli_writes_blocked_two_sided_report(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "single_side_continuous_rehearsal_latest.json").write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
    out = tmp_path / "two_sided.json"

    rc = main(["--reports-dir", str(reports), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 2
    assert payload["status"] == BLOCKED_STATUS
