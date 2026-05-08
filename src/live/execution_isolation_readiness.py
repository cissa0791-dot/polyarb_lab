from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "execution_isolation_readiness.v1"
REPORT_TYPE = "execution_isolation_readiness"

READY_STATUS = "EXECUTION_ISOLATION_READY"
BLOCKED_STATUS = "EXECUTION_ISOLATION_BLOCKED"

DEFAULT_WATCH_SCRIPT_NAMES = (
    "run_1hr_shadow_test.py",
    "run_auto_trade_profit.py",
    "run_autonomous_live_learning_loop.py",
    "run_execution_disabled_auto_trade_system.py",
    "run_gate_bound_live_canary.py",
    "run_live_order_manager.py",
    "run_micro_live_order_probe.py",
    "run_pm_continuous_learning_loop.py",
    "run_reward_live_mm.py",
)


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    command: str
    cwd: str | None = None


def build_execution_isolation_report(
    *,
    root: str | Path,
    process_snapshots: Iterable[ProcessSnapshot | dict[str, Any]] | None = None,
    current_pid: int | None = None,
    watch_script_names: Iterable[str] = DEFAULT_WATCH_SCRIPT_NAMES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only proof that no competing live/shadow runner is active."""

    now = now or datetime.now(timezone.utc)
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    root_path = Path(root).resolve()
    blockers: list[str] = []
    errors: list[str] = []

    if process_snapshots is None:
        try:
            snapshots = scan_process_table()
        except Exception as exc:
            snapshots = []
            blockers.append("PROCESS_TABLE_SCAN_FAILED")
            errors.append(str(exc))
    else:
        snapshots = [_normalise_process_snapshot(item) for item in process_snapshots]

    watched = _find_watched_processes(
        snapshots=snapshots,
        root=root_path,
        current_pid=current_pid,
        watch_script_names=tuple(watch_script_names),
    )
    if watched:
        blockers.append("STALE_OR_COMPETING_EXECUTION_PROCESS_PRESENT")

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
        "root": str(root_path),
        "current_pid": current_pid,
        "process_scan_supported": "PROCESS_TABLE_SCAN_FAILED" not in blockers,
        "watched_script_names": list(watch_script_names),
        "suspicious_process_count": len(watched),
        "single_writer_ok": len(watched) == 0,
        "suspicious_processes": watched,
        "blockers": _unique(blockers),
        "errors": errors,
        "one_line_verdict": _one_line_verdict(status, _unique(blockers), len(watched)),
    }


def scan_process_table() -> list[ProcessSnapshot]:
    """Return a minimal process snapshot list.

    Linux /proc is the production target for the VPS. Windows falls back to
    PowerShell through a separate code path so local smoke runs fail closed
    only when process enumeration is unavailable.
    """

    proc_root = Path("/proc")
    if proc_root.exists():
        return _scan_linux_proc(proc_root)
    return _scan_windows_processes()


def _scan_linux_proc(proc_root: Path) -> list[ProcessSnapshot]:
    snapshots: list[ProcessSnapshot] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw_cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if not raw_cmd:
            continue
        cwd: str | None
        try:
            cwd = str((entry / "cwd").resolve())
        except OSError:
            cwd = None
        snapshots.append(ProcessSnapshot(pid=pid, command=raw_cmd, cwd=cwd))
    return snapshots


def _scan_windows_processes() -> list[ProcessSnapshot]:
    # Kept intentionally small and dependency-free. Tests inject snapshots; the
    # VPS path uses /proc.
    import json
    import subprocess

    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    payload = json.loads(completed.stdout or "[]")
    rows = payload if isinstance(payload, list) else [payload]
    snapshots: list[ProcessSnapshot] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("CommandLine"):
            continue
        snapshots.append(ProcessSnapshot(pid=int(row.get("ProcessId") or 0), command=str(row["CommandLine"])))
    return snapshots


def _find_watched_processes(
    *,
    snapshots: list[ProcessSnapshot],
    root: Path,
    current_pid: int,
    watch_script_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    watched: list[dict[str, Any]] = []
    root_text = str(root).replace("\\", "/")
    for process in snapshots:
        if process.pid == current_pid:
            continue
        command = process.command.replace("\\", "/")
        matched_scripts = [script for script in watch_script_names if script in command]
        if not matched_scripts:
            continue
        cwd_text = (process.cwd or "").replace("\\", "/")
        cwd_is_current_root = cwd_text == root_text
        cwd_is_archive_or_other = bool(cwd_text and cwd_text != root_text)
        command_mentions_repo = "polyarb_lab" in command or "polyarb_lab" in cwd_text
        if not command_mentions_repo:
            continue
        watched.append(
            {
                "pid": process.pid,
                "cwd": process.cwd,
                "command": process.command,
                "matched_scripts": matched_scripts,
                "cwd_is_current_root": cwd_is_current_root,
                "cwd_is_archive_or_other": cwd_is_archive_or_other,
            }
        )
    return watched


def _normalise_process_snapshot(item: ProcessSnapshot | dict[str, Any]) -> ProcessSnapshot:
    if isinstance(item, ProcessSnapshot):
        return item
    return ProcessSnapshot(
        pid=int(item.get("pid") or item.get("process_id") or 0),
        command=str(item.get("command") or item.get("cmdline") or ""),
        cwd=str(item.get("cwd")) if item.get("cwd") not in {None, ""} else None,
    )


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str], process_count: int) -> str:
    if status == READY_STATUS:
        return "EXECUTION_ISOLATION_READY: no watched live/shadow runner processes found; can_submit_order=false."
    return (
        "EXECUTION_ISOLATION_BLOCKED: "
        f"{process_count} watched process(es) present; {', '.join(blockers) or 'UNKNOWN'}; "
        "can_submit_order=false."
    )
