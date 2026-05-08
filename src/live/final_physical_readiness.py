from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPORT_SCHEMA_VERSION = "final_physical_readiness.v1"
REPORT_TYPE = "final_physical_readiness"

READY_STATUS = "FINAL_PHYSICAL_READY"
BLOCKED_STATUS = "FINAL_PHYSICAL_BLOCKED"

DEFAULT_MIN_FREE_DISK_PCT = 10.0
DEFAULT_MAX_CLOCK_SKEW_MS = 50.0


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], float], CommandResult]


def build_final_physical_readiness_report(
    *,
    disk_usage: shutil._ntuple_diskusage | None = None,
    clock_skew_ms: float | None = None,
    clock_source: str | None = None,
    clock_error: str | None = None,
    network_report: dict[str, Any] | None = None,
    kill_switch_file: str | Path | None = None,
    min_free_disk_pct: float = DEFAULT_MIN_FREE_DISK_PCT,
    max_clock_skew_ms: float = DEFAULT_MAX_CLOCK_SKEW_MS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a final read-only physical precondition report."""

    now = now or datetime.now(timezone.utc)
    network_report = network_report or {}
    disk_usage = disk_usage or shutil.disk_usage("/")
    total = float(disk_usage.total)
    free = float(disk_usage.free)
    used = float(disk_usage.used)
    free_pct = 0.0 if total <= 0 else (free / total) * 100.0
    kill_switch_active = _kill_switch_active(kill_switch_file)

    checks = {
        "disk_free_pct_ok": free_pct >= min_free_disk_pct,
        "clock_skew_proven": clock_skew_ms is not None,
        "clock_skew_ok": clock_skew_ms is not None and abs(clock_skew_ms) <= max_clock_skew_ms,
        "latency_ok": network_report.get("latency_ok") is True,
        "heartbeat_ok": network_report.get("heartbeat_ok") is True,
        "kill_switch_clear": kill_switch_active is False,
        "market_not_suspended": network_report.get("market_suspended") is False,
    }
    blockers: list[str] = []
    if not checks["disk_free_pct_ok"]:
        blockers.append("DISK_FREE_SPACE_BELOW_THRESHOLD")
    if not checks["clock_skew_proven"]:
        blockers.append("CLOCK_SKEW_NOT_PROVEN")
    elif not checks["clock_skew_ok"]:
        blockers.append("CLOCK_SKEW_ABOVE_THRESHOLD")
    if not checks["latency_ok"] or not checks["heartbeat_ok"]:
        blockers.append("NETWORK_HEARTBEAT_NOT_READY")
    if not checks["kill_switch_clear"]:
        blockers.append("KILL_SWITCH_ACTIVE")
    if not checks["market_not_suspended"]:
        blockers.append("MARKET_SUSPENDED")

    status = READY_STATUS if not blockers else BLOCKED_STATUS
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "execution_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "status": status,
        "checks": checks,
        "disk": {
            "total_bytes": int(total),
            "used_bytes": int(used),
            "free_bytes": int(free),
            "free_pct": round(free_pct, 6),
            "min_free_pct": min_free_disk_pct,
        },
        "clock": {
            "skew_ms": None if clock_skew_ms is None else round(float(clock_skew_ms), 6),
            "source": clock_source,
            "max_skew_ms": max_clock_skew_ms,
            "error": clock_error,
        },
        "network": {
            "latency_ok": network_report.get("latency_ok"),
            "heartbeat_ok": network_report.get("heartbeat_ok"),
            "latency_ms": network_report.get("latency_ms") or network_report.get("api_latency_ms"),
            "market_suspended": network_report.get("market_suspended"),
        },
        "kill_switch_active": kill_switch_active,
        "kill_switch_file": None if kill_switch_file is None else str(kill_switch_file),
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, _unique(blockers)),
    }


def collect_final_physical_readiness_report(
    *,
    network_report: dict[str, Any] | None = None,
    kill_switch_file: str | Path | None = None,
    min_free_disk_pct: float = DEFAULT_MIN_FREE_DISK_PCT,
    max_clock_skew_ms: float = DEFAULT_MAX_CLOCK_SKEW_MS,
    command_runner: CommandRunner | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = probe_clock_skew_ms(command_runner=command_runner)
    return build_final_physical_readiness_report(
        network_report=network_report,
        kill_switch_file=kill_switch_file,
        min_free_disk_pct=min_free_disk_pct,
        max_clock_skew_ms=max_clock_skew_ms,
        clock_skew_ms=clock.get("clock_skew_ms"),
        clock_source=clock.get("clock_source"),
        clock_error=clock.get("clock_error"),
        now=now,
    )


def probe_clock_skew_ms(*, command_runner: CommandRunner | None = None, timeout_sec: float = 5.0) -> dict[str, Any]:
    runner = command_runner or _run_command
    attempts = [
        (["chronyc", "tracking"], _parse_chronyc_tracking, "chronyc tracking"),
        (["timedatectl", "timesync-status"], _parse_timedatectl_timesync, "timedatectl timesync-status"),
        (["ntpdate", "-q", "pool.ntp.org"], _parse_ntpdate_query, "ntpdate -q pool.ntp.org"),
    ]
    errors: list[str] = []
    for command, parser, source in attempts:
        try:
            result = runner(command, timeout_sec)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            continue
        if result.returncode != 0:
            errors.append(f"{source}: rc={result.returncode} {result.stderr.strip()}")
            continue
        skew_ms = parser(result.stdout + "\n" + result.stderr)
        if skew_ms is not None:
            return {"clock_skew_ms": skew_ms, "clock_source": source, "clock_error": None}
        errors.append(f"{source}: offset not found")
    return {"clock_skew_ms": None, "clock_source": None, "clock_error": "; ".join(errors)}


def _run_command(command: list[str], timeout_sec: float) -> CommandResult:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_sec)
    return CommandResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _parse_chronyc_tracking(text: str) -> float | None:
    for line in text.splitlines():
        if "System time" not in line:
            continue
        match = re.search(r":\s*([+-]?\d+(?:\.\d+)?)\s*seconds", line)
        if match:
            return float(match.group(1)) * 1000.0
    return None


def _parse_ntpdate_query(text: str) -> float | None:
    match = re.search(r"offset\s+([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1)) * 1000.0


def _parse_timedatectl_timesync(text: str) -> float | None:
    for line in text.splitlines():
        if not line.strip().lower().startswith("offset:"):
            continue
        value = line.split(":", 1)[1].strip()
        match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(us|µs|ms|s)?", value, flags=re.IGNORECASE)
        if not match:
            return None
        amount = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit in {"us", "µs"}:
            return amount / 1000.0
        if unit == "ms":
            return amount
        return amount * 1000.0


def _kill_switch_active(path: str | Path | None) -> bool:
    if path is None:
        return False
    return Path(path).exists()


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return "FINAL_PHYSICAL_READY: disk, clock, heartbeat, and kill-switch checks passed; can_submit_order=false."
    return f"FINAL_PHYSICAL_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
