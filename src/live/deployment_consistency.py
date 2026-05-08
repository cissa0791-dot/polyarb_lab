from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "deployment_consistency.v1"
REPORT_TYPE = "deployment_consistency"

DEFAULT_CRITICAL_PATHS = [
    "src/live/live_readiness_gate.py",
    "src/live/deposit_wallet_readonly.py",
    "src/live/market_microstructure_readiness.py",
    "src/live/order_mutex_readiness.py",
    "src/live/api_heartbeat_probe.py",
    "src/live/auth_scope_validator.py",
    "src/live/fee_auditor.py",
    "scripts/build_live_readiness_gate_report.py",
    "scripts/build_deposit_wallet_readonly_report.py",
    "scripts/build_live_market_microstructure_report.py",
    "scripts/build_order_mutex_readiness_report.py",
    "scripts/build_api_heartbeat_report.py",
    "scripts/build_live_auth_readiness_report.py",
    "scripts/build_fee_reconciliation_report.py",
]


def build_local_manifest(
    *,
    root: str | Path,
    critical_paths: Iterable[str] = DEFAULT_CRITICAL_PATHS,
    label: str = "local",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    files = [_file_hash(root_path, rel_path) for rel_path in critical_paths]
    status_text = _git_stdout(root_path, ["status", "--porcelain", "--untracked-files=all"]) or ""
    return {
        "label": label,
        "root": str(root_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_stdout(root_path, ["rev-parse", "--short", "HEAD"]),
        "git_status_dirty": bool(status_text.strip()),
        "git_status_entry_count": len([line for line in status_text.splitlines() if line.strip()]),
        "git_status_porcelain": _status_preview(status_text),
        "files": files,
    }


def build_deployment_consistency_report(
    *,
    local_manifest: dict[str, Any],
    remote_manifest: dict[str, Any] | None = None,
    approved_git_commit: str | None = None,
) -> dict[str, Any]:
    remote_manifest = remote_manifest or {}
    local_files = _files_by_path(local_manifest)
    remote_files = _files_by_path(remote_manifest)
    critical_paths = sorted(set(local_files) | set(remote_files))
    comparisons = [_compare_path(path, local_files.get(path), remote_files.get(path)) for path in critical_paths]

    local_commit = str(local_manifest.get("git_commit") or "")
    remote_commit = str(remote_manifest.get("git_commit") or "")
    local_dirty = _manifest_dirty(local_manifest)
    remote_dirty = _manifest_dirty(remote_manifest)
    head_matches_approved = (
        bool(local_commit and remote_commit)
        and local_commit == remote_commit
        and (approved_git_commit in {None, "", local_commit})
    )
    critical_checksums_match = bool(comparisons) and all(item.get("match") is True for item in comparisons)
    unreviewed_changes_present = local_dirty or remote_dirty
    blockers: list[str] = []
    if not remote_manifest:
        blockers.append("REMOTE_DEPLOYMENT_MANIFEST_MISSING")
    if not head_matches_approved:
        blockers.append("DEPLOYMENT_HEAD_MISMATCH")
    if not critical_checksums_match:
        blockers.append("DEPLOYMENT_CHECKSUM_MISMATCH")
    if unreviewed_changes_present:
        blockers.append("UNREVIEWED_DEPLOYMENT_CHANGES_PRESENT")

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "approved_git_commit": approved_git_commit,
        "local_git_commit": local_commit or None,
        "remote_git_commit": remote_commit or None,
        "local_remote_head_match": bool(local_commit and remote_commit and local_commit == remote_commit),
        "head_matches_approved": head_matches_approved,
        "critical_checksums_match": critical_checksums_match,
        "unreviewed_changes_present": unreviewed_changes_present,
        "local_dirty": local_dirty,
        "remote_dirty": remote_dirty,
        "critical_paths": critical_paths,
        "file_comparisons": comparisons,
        "blockers": blockers,
        "status": "DEPLOYMENT_SYNC_OK" if not blockers else "DEPLOYMENT_SYNC_BLOCKED",
        "one_line_verdict": (
            "DEPLOYMENT_SYNC_OK: local and VPS critical file hashes match."
            if not blockers
            else f"DEPLOYMENT_SYNC_BLOCKED: {', '.join(blockers)}."
        ),
    }


def _file_hash(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    if not path.exists() or not path.is_file():
        return {
            "path": rel_path,
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "mtime_utc": None,
        }
    data = path.read_bytes()
    return {
        "path": rel_path,
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def _git_stdout(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _files_by_path(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("files") if isinstance(manifest, dict) else []
    if not isinstance(files, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in files:
        if isinstance(item, dict) and item.get("path"):
            out[str(item["path"])] = item
    return out


def _manifest_dirty(manifest: dict[str, Any]) -> bool:
    if "git_status_dirty" in manifest:
        return bool(manifest.get("git_status_dirty")) or bool(str(manifest.get("git_status_porcelain") or "").strip())
    return bool(str(manifest.get("git_status_porcelain") or "").strip())


def _status_preview(status_text: str, *, max_lines: int = 40) -> str:
    lines = [line for line in status_text.splitlines() if line.strip()]
    preview = lines[:max_lines]
    if len(lines) > max_lines:
        preview.append(f"... truncated {len(lines) - max_lines} additional git status entries")
    return "\n".join(preview)


def _compare_path(
    rel_path: str,
    local_file: dict[str, Any] | None,
    remote_file: dict[str, Any] | None,
) -> dict[str, Any]:
    local_sha = local_file.get("sha256") if isinstance(local_file, dict) else None
    remote_sha = remote_file.get("sha256") if isinstance(remote_file, dict) else None
    local_exists = bool(local_file and local_file.get("exists"))
    remote_exists = bool(remote_file and remote_file.get("exists"))
    return {
        "path": rel_path,
        "local_exists": local_exists,
        "remote_exists": remote_exists,
        "local_sha256": local_sha,
        "remote_sha256": remote_sha,
        "match": bool(local_exists and remote_exists and local_sha and local_sha == remote_sha),
    }
