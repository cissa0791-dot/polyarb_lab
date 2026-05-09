from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.c_fill_likelihood_probe_approval_package import (  # noqa: E402
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    build_c_fill_likelihood_probe_approval_package,
    markdown_report,
)
from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "c_fill_likelihood_probe_approval_package_latest.json"
DEFAULT_MD_OUT = DEFAULT_REPORTS_DIR / "c_fill_likelihood_probe_approval_package_latest.md"

REPORT_FILES = {
    "gate": "live_readiness_gate_latest.json",
    "planner": "live_probe_planner_latest.json",
    "fill_readiness": "fill_reconciliation_readiness_latest.json",
    "fee_reconciliation": "fee_reconciliation_latest.json",
    "toxic_flow": "toxic_flow_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only C fill-likelihood probe approval package.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--max-c-probe-size", type=float, default=10.0)
    parser.add_argument("--operator-size-override-approved", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    loaded = {key: load_json_report(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    report = build_c_fill_likelihood_probe_approval_package(
        gate=loaded["gate"],
        planner=loaded["planner"],
        fill_readiness=loaded["fill_readiness"],
        fee_reconciliation=loaded["fee_reconciliation"],
        toxic_flow=loaded["toxic_flow"],
        max_c_probe_size=args.max_c_probe_size,
        operator_size_override_approved=args.operator_size_override_approved,
    )
    input_paths = [reports_dir / filename for filename in REPORT_FILES.values()]
    report["source_reports"] = {key: str(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=input_paths,
        source_files=[
            Path(__file__),
            ROOT / "src" / "live" / "c_fill_likelihood_probe_approval_package.py",
        ],
        root=ROOT,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
