"""
reward_aware_maker_probe — Module 3: Paper Logger
polyarb_lab / research_line / probe-only

Writes probe scan results to disk. No order submission. No mainline imports.

Output directory: data/research/reward_aware_maker_probe/
  probe_{YYYYMMDDTHHMMSSZ}.json  — full probe result (one per run)
  probe_index.json               — rolling index of all probes (appended)
  latest_probe.json              — copy of most recent probe
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .discovery import RawRewardedMarket, discovery_summary
from .ev_model import MarketEVResult, build_ev_summary

logger = logging.getLogger(__name__)

# Resolve output directory relative to repo root.
# File is at: research_lines/reward_aware_maker_probe/modules/paper_logger.py
_MODULE_DIR = Path(__file__).resolve().parent
_LINE_DIR = _MODULE_DIR.parent
_LAB_ROOT = _LINE_DIR.parent.parent
PROBE_DATA_DIR = _LAB_ROOT / "data" / "research" / "reward_aware_maker_probe"


# ---------------------------------------------------------------------------
# Probe result bundle
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Container for all outputs from one probe run."""
    probe_id: str
    probe_timestamp: datetime
    raw_markets: list[RawRewardedMarket]
    ev_results: list[MarketEVResult]
    probe_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        disc = discovery_summary(self.raw_markets)
        ev_summary = build_ev_summary(self.raw_markets, self.ev_results)

        return {
            "probe_id": self.probe_id,
            "probe_timestamp": self.probe_timestamp.isoformat(),
            "probe_version": "v1",
            "probe_config": self.probe_config,
            "summary": {
                **disc,
                **ev_summary,
            },
            "markets": [r.to_dict() for r in self.ev_results],
        }


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _append_index(index_path: Path, entry: dict[str, Any]) -> None:
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index: list = json.load(f)
        except (json.JSONDecodeError, OSError):
            index = []
    else:
        index = []
    index.append(entry)
    _write_json(index_path, index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_probe_id(timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now(timezone.utc)
    return f"RAMM_probe_{ts.strftime('%Y%m%dT%H%M%SZ')}"


def log_probe(
    probe_result: ProbeResult,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Write a complete probe result to disk.

    Writes:
      {output_dir}/probe_{timestamp}.json  — full probe
      {output_dir}/probe_index.json        — updated rolling index
      {output_dir}/latest_probe.json       — copy of this probe

    Returns:
        Path to the written probe file.
    """
    if output_dir is None:
        output_dir = PROBE_DATA_DIR

    _ensure_dir(output_dir)

    ts_str = probe_result.probe_timestamp.strftime("%Y%m%dT%H%M%SZ")
    probe_filename = f"probe_{ts_str}.json"
    probe_path = output_dir / probe_filename
    probe_data = probe_result.to_dict()

    _write_json(probe_path, probe_data)
    logger.info("Probe written to %s", probe_path)

    _write_json(output_dir / "latest_probe.json", probe_data)

    index_entry = {
        "probe_id": probe_result.probe_id,
        "probe_timestamp": probe_result.probe_timestamp.isoformat(),
        "probe_file": probe_filename,
        "summary": probe_data["summary"],
    }
    _append_index(output_dir / "probe_index.json", index_entry)

    return probe_path


def load_probe_index(output_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load the rolling probe index from disk."""
    if output_dir is None:
        output_dir = PROBE_DATA_DIR
    index_path = output_dir / "probe_index.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def load_latest_probe(output_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Load the most recent probe result from disk."""
    if output_dir is None:
        output_dir = PROBE_DATA_DIR
    path = output_dir / "latest_probe.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
