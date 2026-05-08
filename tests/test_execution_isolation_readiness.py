from __future__ import annotations

from datetime import datetime, timezone

from src.live.execution_isolation_readiness import (
    ProcessSnapshot,
    build_execution_isolation_report,
)


NOW = datetime(2026, 5, 8, 2, 0, tzinfo=timezone.utc)


def test_clear_process_table_marks_execution_isolation_ready() -> None:
    report = build_execution_isolation_report(
        root="/root/polyarb_lab",
        process_snapshots=[
            ProcessSnapshot(pid=10, command="python scripts/build_execution_isolation_report.py", cwd="/root/polyarb_lab"),
        ],
        current_pid=10,
        now=NOW,
    )

    assert report["status"] == "EXECUTION_ISOLATION_READY"
    assert report["single_writer_ok"] is True
    assert report["suspicious_process_count"] == 0
    assert report["can_submit_order"] is False


def test_stale_shadow_process_blocks_execution_isolation() -> None:
    report = build_execution_isolation_report(
        root="/root/polyarb_lab",
        process_snapshots=[
            ProcessSnapshot(
                pid=551112,
                command="python scripts/run_1hr_shadow_test.py --market-slug x",
                cwd="/root/polyarb_lab_dirty_archive_20260508T104219Z",
            )
        ],
        current_pid=999,
        now=NOW,
    )

    assert report["status"] == "EXECUTION_ISOLATION_BLOCKED"
    assert "STALE_OR_COMPETING_EXECUTION_PROCESS_PRESENT" in report["blockers"]
    assert report["suspicious_process_count"] == 1
    assert report["suspicious_processes"][0]["cwd_is_archive_or_other"] is True
