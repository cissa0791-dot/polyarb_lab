from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.bootstrap_next_action_policy import build_bootstrap_next_action_policy
from src.live.report_metadata import attach_writer_metadata


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_CLOSEOUT = DEFAULT_REPORTS_DIR / "ivan_bootstrap_attempt_1_closeout_latest.json"
DEFAULT_VALIDATION = DEFAULT_REPORTS_DIR / "ivan_first_cycle_evidence_validation_latest.json"
DEFAULT_APPROVAL = DEFAULT_REPORTS_DIR / "one_time_bootstrap_execution_approval_latest.json"
DEFAULT_HEALTH = DEFAULT_REPORTS_DIR / "live_api_health_readonly_now.json"
DEFAULT_EXECUTION_SYSTEM = DEFAULT_REPORTS_DIR / "execution_disabled_auto_trade_system_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "bootstrap_next_action_latest.json"
SCHEMA_VERSION = "bootstrap_next_action_policy.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-closeout bootstrap next-action policy report.")
    parser.add_argument("--closeout", default=str(DEFAULT_CLOSEOUT))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--approval", default=str(DEFAULT_APPROVAL))
    parser.add_argument("--health", default=str(DEFAULT_HEALTH))
    parser.add_argument("--execution-system", default=str(DEFAULT_EXECUTION_SYSTEM))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_bootstrap_next_action_policy(
        closeout=load_json(args.closeout),
        evidence_validation=load_json(args.validation),
        approval=load_json(args.approval),
        health=load_json(args.health),
        execution_system=load_json(args.execution_system),
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=SCHEMA_VERSION,
        input_reports_used=[
            Path(args.closeout),
            Path(args.validation),
            Path(args.approval),
            Path(args.health),
            Path(args.execution_system),
        ],
        source_files=[Path(__file__), ROOT / "src" / "live" / "bootstrap_next_action_policy.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


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
