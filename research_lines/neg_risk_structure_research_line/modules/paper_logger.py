"""
neg_risk_structure_research_line — Module 5: Paper Logger
polyarb_lab / research_line / active

Paper-only logging module. Writes all scan results to disk.
No order submission. No live execution. No connection to mainline data paths.

Output directory: data/research/neg_risk/
  scan_{YYYYMMDDTHHMMSSZ}.json   — full scan result (one per run)
  scan_index.json                 — rolling index of all scans (appended)
  latest_scan.json                — symlink/copy of most recent scan

Read-only except to data/research/neg_risk/.
No mainline imports.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .discovery import NegRiskEventRaw, discovery_summary
from .normalizer import NegRiskEvent
from .structural_check import StructuralCheckResult
from .executable_filter import ExecutionQualityResult

logger = logging.getLogger(__name__)

# Resolve output directory relative to the polyarb_lab root
# Assumes this file is at: research_lines/neg_risk_structure_research_line/modules/paper_logger.py
_MODULE_DIR = Path(__file__).resolve().parent
_LINE_DIR = _MODULE_DIR.parent
_LAB_ROOT = _LINE_DIR.parent.parent
NEG_RISK_DATA_DIR = _LAB_ROOT / "data" / "research" / "neg_risk"


# ---------------------------------------------------------------------------
# Scan result bundle
# ---------------------------------------------------------------------------

class ScanResult:
    """Container for all outputs from one research scan run."""

    def __init__(
        self,
        scan_id: str,
        scan_timestamp: datetime,
        raw_events: list[NegRiskEventRaw],
        normalized_events: list[NegRiskEvent],
        failed_normalization_ids: list[str],
        structural_results: list[StructuralCheckResult],
        execution_results: list[ExecutionQualityResult],
        scan_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.scan_id = scan_id
        self.scan_timestamp = scan_timestamp
        self.raw_events = raw_events
        self.normalized_events = normalized_events
        self.failed_normalization_ids = failed_normalization_ids
        self.structural_results = structural_results
        self.execution_results = execution_results
        self.scan_config = scan_config or {}

    def to_log_dict(self) -> dict[str, Any]:
        """Serialize the full scan result to a JSON-compatible dict."""
        # Discovery summary
        disc_summary = discovery_summary(self.raw_events)

        # Structural summary
        struct_counts: dict[str, int] = {}
        for r in self.structural_results:
            struct_counts[r.constraint_class] = struct_counts.get(r.constraint_class, 0) + 1

        # Execution summary
        exec_counts: dict[str, int] = {}
        for r in self.execution_results:
            exec_counts[r.execution_class] = exec_counts.get(r.execution_class, 0) + 1

        return {
            "scan_id": self.scan_id,
            "scan_timestamp": self.scan_timestamp.isoformat(),
            "scan_config": self.scan_config,
            "summary": {
                "discovery": disc_summary,
                "normalization": {
                    "normalized": len(self.normalized_events),
                    "failed": len(self.failed_normalization_ids),
                    "failed_ids": self.failed_normalization_ids,
                },
                "structural": struct_counts,
                "execution": exec_counts,
            },
            "structural_results": [r.to_log_dict() for r in self.structural_results],
            "execution_results": [r.to_log_dict() for r in self.execution_results],
            # Light normalized event log (no full outcome detail — structural_results covers it)
            "normalized_events_light": [
                {
                    "event_id": e.event_id,
                    "slug": e.slug,
                    "title": e.title,
                    "n_outcomes": e.n_outcomes,
                    "implied_sum": round(e.implied_sum, 6),
                    "constraint_gap": round(e.constraint_gap, 6),
                    "abs_gap": round(e.abs_gap, 6),
                    "has_all_prices": e.has_all_prices,
                    "dominant_price_source": e.dominant_price_source,
                    "end_date_str": e.end_date_str,
                }
                for e in self.normalized_events
            ],
        }


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _scan_filename(scan_timestamp: datetime) -> str:
    ts = scan_timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"scan_{ts}.json"


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _append_to_index(index_path: Path, entry: dict[str, Any]) -> None:
    """Append a scan index entry to scan_index.json (create if missing)."""
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            try:
                index = json.load(f)
            except json.JSONDecodeError:
                index = []
    else:
        index = []

    index.append(entry)
    _write_json(index_path, index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_scan(
    scan_result: ScanResult,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Write a complete scan result to disk.

    Writes:
      {output_dir}/scan_{timestamp}.json   — full scan
      {output_dir}/scan_index.json         — updated rolling index
      {output_dir}/latest_scan.json        — copy of this scan

    Returns:
        Path to the written scan file.
    """
    if output_dir is None:
        output_dir = NEG_RISK_DATA_DIR

    _ensure_output_dir(output_dir)

    # Full scan file
    scan_filename = _scan_filename(scan_result.scan_timestamp)
    scan_path = output_dir / scan_filename
    scan_data = scan_result.to_log_dict()
    _write_json(scan_path, scan_data)
    logger.info("Scan written to %s", scan_path)

    # Latest scan copy
    latest_path = output_dir / "latest_scan.json"
    _write_json(latest_path, scan_data)

    # Update rolling index
    index_entry = {
        "scan_id": scan_result.scan_id,
        "scan_timestamp": scan_result.scan_timestamp.isoformat(),
        "scan_file": scan_filename,
        "summary": scan_data["summary"],
    }
    index_path = output_dir / "scan_index.json"
    _append_to_index(index_path, index_entry)
    logger.info("Scan index updated (%s)", index_path)

    return scan_path


def make_scan_id(timestamp: Optional[datetime] = None) -> str:
    """Generate a unique scan ID from timestamp."""
    ts = timestamp or datetime.now(timezone.utc)
    return f"NEG-SCAN-{ts.strftime('%Y%m%dT%H%M%SZ')}"


def load_scan_index(output_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load the rolling scan index from disk."""
    if output_dir is None:
        output_dir = NEG_RISK_DATA_DIR
    index_path = output_dir / "scan_index.json"
    if not index_path.exists():
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Scan index corrupted at %s — returning empty", index_path)
            return []


def load_scan(scan_file: str, output_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Load a specific scan file by filename."""
    if output_dir is None:
        output_dir = NEG_RISK_DATA_DIR
    path = output_dir / scan_file
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Scan file corrupted: %s", path)
            return None


def load_latest_scan(output_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Load the most recent scan from disk."""
    if output_dir is None:
        output_dir = NEG_RISK_DATA_DIR
    path = output_dir / "latest_scan.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def load_all_scans(output_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load all scan files in order (oldest first)."""
    if output_dir is None:
        output_dir = NEG_RISK_DATA_DIR
    index = load_scan_index(output_dir)
    scans: list[dict[str, Any]] = []
    for entry in index:
        scan = load_scan(entry["scan_file"], output_dir)
        if scan is not None:
            scans.append(scan)
    return scans
