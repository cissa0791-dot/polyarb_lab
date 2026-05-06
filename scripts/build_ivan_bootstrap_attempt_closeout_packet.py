from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.ivan_bootstrap_attempt_closeout import build_ivan_bootstrap_attempt_closeout
from src.live.report_metadata import attach_writer_metadata


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_APPROVAL = DEFAULT_REPORTS_DIR / "one_time_bootstrap_execution_approval_latest.json"
DEFAULT_LIFECYCLE = DEFAULT_REPORTS_DIR / "ivan_open_buy_order_lifecycle_latest.json"
DEFAULT_RESOLUTION = DEFAULT_REPORTS_DIR / "ivan_open_buy_resolution_packet_latest.json"
DEFAULT_VALIDATION = DEFAULT_REPORTS_DIR / "ivan_first_cycle_evidence_validation_latest.json"
DEFAULT_HEALTH = DEFAULT_REPORTS_DIR / "live_api_health_readonly_now.json"
DEFAULT_EXECUTION_SYSTEM = DEFAULT_REPORTS_DIR / "execution_disabled_auto_trade_system_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "ivan_bootstrap_attempt_1_closeout_latest.json"
SCHEMA_VERSION = "ivan_bootstrap_attempt_closeout.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Ivan bootstrap attempt #1 closeout packet.")
    parser.add_argument("--approval", default=str(DEFAULT_APPROVAL))
    parser.add_argument("--lifecycle", default=str(DEFAULT_LIFECYCLE))
    parser.add_argument("--resolution", default=str(DEFAULT_RESOLUTION))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--health", default=str(DEFAULT_HEALTH))
    parser.add_argument("--execution-system", default=str(DEFAULT_EXECUTION_SYSTEM))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_ivan_bootstrap_attempt_closeout(
        approval=load_json(args.approval),
        lifecycle=load_json(args.lifecycle),
        resolution=load_json(args.resolution),
        validation=load_json(args.validation),
        health=load_json(args.health),
        execution_system=load_json(args.execution_system),
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=SCHEMA_VERSION,
        input_reports_used=[
            Path(args.approval),
            Path(args.lifecycle),
            Path(args.resolution),
            Path(args.validation),
            Path(args.health),
            Path(args.execution_system),
        ],
        source_files=[Path(__file__), ROOT / "src" / "live" / "ivan_bootstrap_attempt_closeout.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("attempt_closeout_classification") == "NULL_RESULT_BOOTSTRAP_EVENT" else 1


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
