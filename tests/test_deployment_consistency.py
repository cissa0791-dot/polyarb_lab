from __future__ import annotations

import json
from pathlib import Path

from scripts.build_deployment_consistency_report import main
from src.live.deployment_consistency import (
    DEFAULT_CRITICAL_PATHS,
    build_deployment_consistency_report,
    build_local_manifest,
)


def _write(root: Path, rel_path: str, text: str) -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_matching_hashes_and_heads_mark_deployment_sync_ok(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    rel = "src/live/live_readiness_gate.py"
    _write(local, rel, "same")
    _write(remote, rel, "same")
    local_manifest = build_local_manifest(root=local, critical_paths=[rel])
    remote_manifest = build_local_manifest(root=remote, critical_paths=[rel], label="remote")
    local_manifest["git_commit"] = remote_manifest["git_commit"] = "abc123"

    report = build_deployment_consistency_report(
        local_manifest=local_manifest,
        remote_manifest=remote_manifest,
    )

    assert report["status"] == "DEPLOYMENT_SYNC_OK"
    assert report["head_matches_approved"] is True
    assert report["critical_checksums_match"] is True
    assert report["unreviewed_changes_present"] is False
    assert report["blockers"] == []
    assert report["can_submit_order"] is False


def test_checksum_mismatch_blocks_deployment_sync(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    rel = "src/live/live_readiness_gate.py"
    _write(local, rel, "local")
    _write(remote, rel, "remote")
    local_manifest = build_local_manifest(root=local, critical_paths=[rel])
    remote_manifest = build_local_manifest(root=remote, critical_paths=[rel], label="remote")
    local_manifest["git_commit"] = remote_manifest["git_commit"] = "abc123"

    report = build_deployment_consistency_report(
        local_manifest=local_manifest,
        remote_manifest=remote_manifest,
    )

    assert report["status"] == "DEPLOYMENT_SYNC_BLOCKED"
    assert "DEPLOYMENT_CHECKSUM_MISMATCH" in report["blockers"]
    assert report["file_comparisons"][0]["match"] is False


def test_dirty_checkout_blocks_even_when_hashes_match(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    rel = "src/live/live_readiness_gate.py"
    _write(local, rel, "same")
    _write(remote, rel, "same")
    local_manifest = build_local_manifest(root=local, critical_paths=[rel])
    remote_manifest = build_local_manifest(root=remote, critical_paths=[rel], label="remote")
    local_manifest["git_commit"] = remote_manifest["git_commit"] = "abc123"
    remote_manifest["git_status_porcelain"] = " M src/live/live_readiness_gate.py"

    report = build_deployment_consistency_report(
        local_manifest=local_manifest,
        remote_manifest=remote_manifest,
    )

    assert "UNREVIEWED_DEPLOYMENT_CHANGES_PRESENT" in report["blockers"]
    assert report["unreviewed_changes_present"] is True


def test_cli_can_compare_against_remote_manifest(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    rel = "src/live/live_readiness_gate.py"
    _write(local, rel, "same")
    _write(remote, rel, "same")
    remote_manifest = build_local_manifest(root=remote, critical_paths=[rel], label="remote")
    remote_manifest["git_commit"] = "abc123"
    remote_manifest_path = tmp_path / "remote_manifest.json"
    remote_manifest_path.write_text(json.dumps(remote_manifest), encoding="utf-8")
    out = tmp_path / "deployment.json"

    rc = main(
        [
            "--local-root",
            str(local),
            "--critical-path",
            rel,
            "--remote-manifest",
            str(remote_manifest_path),
            "--approved-git-commit",
            "abc123",
            "--out",
            str(out),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["can_submit_order"] is False
    assert report["critical_checksums_match"] is True


def test_default_critical_paths_include_deposit_wallet_report_chain() -> None:
    assert "src/live/deposit_wallet_readonly.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_deposit_wallet_readonly_report.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_include_market_microstructure_report_chain() -> None:
    assert "src/live/market_microstructure_readiness.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_live_market_microstructure_report.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_include_order_mutex_report_chain() -> None:
    assert "src/live/order_mutex_readiness.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_order_mutex_readiness_report.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_include_auth_scope_report_chain() -> None:
    assert "src/live/auth_scope_validator.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_live_auth_readiness_report.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_include_fee_reconciliation_report_chain() -> None:
    assert "src/live/fee_auditor.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_fee_reconciliation_report.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_include_toxic_flow_report_chain() -> None:
    assert "src/live/toxic_flow_detector.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_toxic_flow_report.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_include_final_gate_report_chain() -> None:
    assert "src/live/execution_isolation_readiness.py" in DEFAULT_CRITICAL_PATHS
    assert "src/live/inventory_state_readiness.py" in DEFAULT_CRITICAL_PATHS
    assert "src/live/final_physical_readiness.py" in DEFAULT_CRITICAL_PATHS
    assert "src/live/single_side_live_rehearsal.py" in DEFAULT_CRITICAL_PATHS
    assert "src/live/one_time_auth_token.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_execution_isolation_report.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_inventory_state_report.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_final_physical_report.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_single_side_live_rehearsal_report.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/build_single_side_probe_authorization_report.py" in DEFAULT_CRITICAL_PATHS
    assert "scripts/create_single_side_probe_authorization_token.py" in DEFAULT_CRITICAL_PATHS


def test_default_critical_paths_exclude_unreleased_shadow_tools() -> None:
    assert "scripts/run_1hr_shadow_test.py" not in DEFAULT_CRITICAL_PATHS
    assert "scripts/analyze_lifecycle_performance.py" not in DEFAULT_CRITICAL_PATHS
