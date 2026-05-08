from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.deployment_consistency import (  # noqa: E402
    DEFAULT_CRITICAL_PATHS,
    REPORT_SCHEMA_VERSION,
    build_deployment_consistency_report,
    build_local_manifest,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_OUT = ROOT / "data" / "reports" / "deployment_consistency_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local and VPS critical module sha256 hashes.")
    parser.add_argument("--local-root", default=str(ROOT))
    parser.add_argument("--remote-root", default="/root/polyarb_lab")
    parser.add_argument("--critical-path", action="append", dest="critical_paths")
    parser.add_argument("--remote-manifest")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--ssh-password-env")
    parser.add_argument("--ssh-key-file")
    parser.add_argument("--approved-git-commit")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    critical_paths = args.critical_paths or list(DEFAULT_CRITICAL_PATHS)
    local_manifest = build_local_manifest(
        root=args.local_root,
        critical_paths=critical_paths,
        label="local",
    )
    if args.manifest_only:
        return local_manifest

    remote_manifest = _load_remote_manifest(args.remote_manifest)
    if args.remote_host:
        remote_manifest = _build_remote_manifest_via_ssh(
            host=args.remote_host,
            user=args.remote_user,
            remote_root=args.remote_root,
            critical_paths=critical_paths,
            password_env=args.ssh_password_env,
            key_file=args.ssh_key_file,
        )
    report = build_deployment_consistency_report(
        local_manifest=local_manifest,
        remote_manifest=remote_manifest,
        approved_git_commit=args.approved_git_commit,
    )
    report["local_manifest"] = local_manifest
    report["remote_manifest_present"] = bool(remote_manifest)
    if remote_manifest:
        report["remote_manifest"] = {
            "label": remote_manifest.get("label"),
            "root": remote_manifest.get("root"),
            "generated_at_utc": remote_manifest.get("generated_at_utc"),
            "git_commit": remote_manifest.get("git_commit"),
            "git_status_dirty": remote_manifest.get("git_status_dirty"),
            "git_status_entry_count": remote_manifest.get("git_status_entry_count"),
            "git_status_porcelain": remote_manifest.get("git_status_porcelain"),
        }
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        source_files=[Path(__file__), ROOT / "src" / "live" / "deployment_consistency.py"],
        root=ROOT,
    )
    return report


def _load_remote_manifest(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_remote_manifest_via_ssh(
    *,
    host: str,
    user: str,
    remote_root: str,
    critical_paths: list[str],
    password_env: str | None,
    key_file: str | None,
) -> dict[str, Any]:
    try:
        import paramiko  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PARAMIKO_NOT_AVAILABLE_FOR_REMOTE_HASH") from exc

    password = os.environ.get(password_env) if password_env else None
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {"hostname": host, "username": user, "timeout": 20}
    if password:
        connect_kwargs["password"] = password
    if key_file:
        connect_kwargs["key_filename"] = key_file
    ssh.connect(**connect_kwargs)
    try:
        sftp = ssh.open_sftp()
        files: list[dict[str, Any]] = []
        for rel_path in critical_paths:
            remote_path = posixpath.join(remote_root, rel_path.replace("\\", "/"))
            files.append(_remote_file_hash(sftp, rel_path, remote_path))
        git_commit = _remote_git_stdout(ssh, remote_root, "git rev-parse --short HEAD")
        git_status = _remote_git_stdout(ssh, remote_root, "git status --porcelain --untracked-files=all")
        git_status_lines = [line for line in git_status.splitlines() if line.strip()]
        return {
            "label": f"ssh://{user}@{host}",
            "root": remote_root,
            "generated_at_utc": _remote_generated_now(),
            "git_commit": git_commit,
            "git_status_dirty": bool(git_status_lines),
            "git_status_entry_count": len(git_status_lines),
            "git_status_porcelain": _status_preview(git_status),
            "files": files,
        }
    finally:
        try:
            sftp.close()
        except Exception:
            pass
        ssh.close()


def _remote_file_hash(sftp: Any, rel_path: str, remote_path: str) -> dict[str, Any]:
    import hashlib
    from datetime import datetime, timezone

    try:
        stat = sftp.stat(remote_path)
        with sftp.open(remote_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return {"path": rel_path, "exists": False, "sha256": None, "size_bytes": None, "mtime_utc": None}
    return {
        "path": rel_path,
        "exists": True,
        "sha256": digest,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _remote_git_stdout(ssh: Any, remote_root: str, command: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_root} && {command}", timeout=20)
    data = stdout.read().decode("utf-8", errors="replace").strip()
    stderr.read()
    stdout.channel.recv_exit_status()
    return data


def _remote_generated_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _status_preview(status_text: str, *, max_lines: int = 40) -> str:
    lines = [line for line in status_text.splitlines() if line.strip()]
    preview = lines[:max_lines]
    if len(lines) > max_lines:
        preview.append(f"... truncated {len(lines) - max_lines} additional git status entries")
    return "\n".join(preview)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
