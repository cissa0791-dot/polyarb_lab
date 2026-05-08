from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.final_physical_readiness import (  # noqa: E402
    DEFAULT_MAX_CLOCK_SKEW_MS,
    DEFAULT_MIN_FREE_DISK_PCT,
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    collect_final_physical_readiness_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_NETWORK_REPORT = DEFAULT_REPORTS_DIR / "live_network_readiness_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "final_physical_readiness_latest.json"
DEFAULT_KILL_SWITCH_FILE = ROOT / "data" / "runtime" / "KILL_SWITCH_ACTIVE"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only final physical readiness report.")
    parser.add_argument("--network-report", default=str(DEFAULT_NETWORK_REPORT))
    parser.add_argument("--kill-switch-file", default=str(DEFAULT_KILL_SWITCH_FILE))
    parser.add_argument("--min-free-disk-pct", type=float, default=DEFAULT_MIN_FREE_DISK_PCT)
    parser.add_argument("--max-clock-skew-ms", type=float, default=DEFAULT_MAX_CLOCK_SKEW_MS)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    network_report = _load_json(Path(args.network_report))
    report = collect_final_physical_readiness_report(
        network_report=network_report,
        kill_switch_file=args.kill_switch_file,
        min_free_disk_pct=args.min_free_disk_pct,
        max_clock_skew_ms=args.max_clock_skew_ms,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.network_report)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "final_physical_readiness.py"],
        root=ROOT,
    )
    return report


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
