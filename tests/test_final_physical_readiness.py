from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.live.final_physical_readiness import (
    CommandResult,
    build_final_physical_readiness_report,
    probe_clock_skew_ms,
)


NOW = datetime(2026, 5, 8, 2, 0, tzinfo=timezone.utc)


def _disk(total: int = 1000, used: int = 100, free: int = 900):
    return shutil._ntuple_diskusage(total, used, free)


def _network() -> dict:
    return {"latency_ok": True, "heartbeat_ok": True, "market_suspended": False, "latency_ms": 99.0}


def test_final_physical_ready_when_disk_clock_network_and_kill_switch_are_clear(tmp_path: Path) -> None:
    report = build_final_physical_readiness_report(
        disk_usage=_disk(),
        clock_skew_ms=12.0,
        clock_source="test",
        network_report=_network(),
        kill_switch_file=tmp_path / "KILL_SWITCH_ACTIVE",
        now=NOW,
    )

    assert report["status"] == "FINAL_PHYSICAL_READY"
    assert report["checks"]["disk_free_pct_ok"] is True
    assert report["checks"]["clock_skew_ok"] is True
    assert report["can_submit_order"] is False


def test_clock_skew_missing_fails_closed() -> None:
    report = build_final_physical_readiness_report(
        disk_usage=_disk(),
        clock_skew_ms=None,
        clock_error="ntp unavailable",
        network_report=_network(),
        now=NOW,
    )

    assert report["status"] == "FINAL_PHYSICAL_BLOCKED"
    assert "CLOCK_SKEW_NOT_PROVEN" in report["blockers"]


def test_kill_switch_file_blocks_final_physical(tmp_path: Path) -> None:
    kill_switch = tmp_path / "KILL_SWITCH_ACTIVE"
    kill_switch.write_text("stop", encoding="utf-8")
    report = build_final_physical_readiness_report(
        disk_usage=_disk(),
        clock_skew_ms=1.0,
        clock_source="test",
        network_report=_network(),
        kill_switch_file=kill_switch,
        now=NOW,
    )

    assert report["status"] == "FINAL_PHYSICAL_BLOCKED"
    assert "KILL_SWITCH_ACTIVE" in report["blockers"]


def test_probe_clock_skew_parses_chronyc_tracking() -> None:
    def runner(command: list[str], timeout_sec: float) -> CommandResult:
        assert command == ["chronyc", "tracking"]
        assert timeout_sec == 5.0
        return CommandResult(command=command, returncode=0, stdout="System time     : 0.000012 seconds slow of NTP time\n", stderr="")

    result = probe_clock_skew_ms(command_runner=runner)

    assert result["clock_skew_ms"] == 0.012
    assert result["clock_source"] == "chronyc tracking"
