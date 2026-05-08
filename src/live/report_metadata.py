from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_MODULE_IMPORT_TIME_UTC = datetime.now(timezone.utc)


def attach_writer_metadata(
    payload: dict[str, Any],
    *,
    writer_script: str | Path,
    report_schema_version: str,
    input_reports_used: Iterable[str | Path] = (),
    source_files: Iterable[str | Path] = (),
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Attach runtime provenance to latest-report payloads before writing them."""
    root_path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    writer_path = Path(writer_script).resolve()
    source_paths = [Path(path).resolve() for path in source_files]
    if writer_path not in source_paths:
        source_paths.insert(0, writer_path)

    process_started = _process_start_time_utc()
    source_meta = [_path_meta(path, root=root_path) for path in source_paths]
    latest_source_mtime = _latest_mtime(source_meta)
    stale_possible = bool(latest_source_mtime and process_started and latest_source_mtime > process_started)

    payload.update(
        {
            "writer_script": _display_path(writer_path, root=root_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "process_pid": os.getpid(),
            "process_start_time_utc": process_started.isoformat() if process_started else None,
            "git_commit": _git_commit(root_path),
            "source_file_mtime_utc": latest_source_mtime.isoformat() if latest_source_mtime else None,
            "source_files": source_meta,
            "report_schema_version": report_schema_version,
            "input_reports_used": [_path_meta(Path(path).resolve(), root=root_path) for path in input_reports_used],
            "stale_process_possible": stale_possible,
            "writer_warnings": ["STALE_PROCESS_POSSIBLE"] if stale_possible else [],
        }
    )
    return payload


def _process_start_time_utc() -> datetime | None:
    proc_start = _linux_process_start_time_utc()
    if proc_start is not None:
        return proc_start
    return _MODULE_IMPORT_TIME_UTC


def _linux_process_start_time_utc() -> datetime | None:
    stat_path = Path("/proc/self/stat")
    system_stat_path = Path("/proc/stat")
    if not stat_path.exists() or not system_stat_path.exists():
        return None
    try:
        stat = stat_path.read_text(encoding="utf-8").split()
        start_ticks = int(stat[21])
        boot_time = None
        for line in system_stat_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                boot_time = int(line.split()[1])
                break
        if boot_time is None:
            return None
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        timestamp = boot_time + (start_ticks / ticks_per_second)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return None


def _path_meta(path: Path, *, root: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": _display_path(path, root=root),
        "exists": exists,
        "mtime_utc": _mtime_utc(path).isoformat() if exists else None,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _latest_mtime(path_meta: list[dict[str, Any]]) -> datetime | None:
    latest: datetime | None = None
    for item in path_meta:
        mtime = item.get("mtime_utc")
        if not mtime:
            continue
        parsed = datetime.fromisoformat(str(mtime))
        latest = parsed if latest is None or parsed > latest else latest
    return latest


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None
