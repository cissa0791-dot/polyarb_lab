from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402
from src.live.single_side_continuous_rehearsal import (  # noqa: E402
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    build_single_side_continuous_rehearsal,
)


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "single_side_continuous_rehearsal_latest.json"

REPORT_FILES = {
    "planner": "full_market_probe_planner_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "inventory_state": "inventory_state_latest.json",
    "fill_audit": "c_fill_reconciliation_audit_latest.json",
    "lifecycle_summary": "fill_lifecycle_summary_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build limited single-side continuous rehearsal controller report.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stop-file")
    parser.add_argument("--max-order-count", type=int, default=3)
    parser.add_argument("--current-order-count", type=int, default=0)
    parser.add_argument("--max-position-shares", type=float, default=10.0)
    parser.add_argument("--session-minutes", type=int, default=15)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    loaded = {key: load_json_report(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    report = build_single_side_continuous_rehearsal(
        planner=loaded["planner"],
        order_mutex=loaded["order_mutex"],
        inventory_state=loaded["inventory_state"],
        fill_audit=loaded["fill_audit"],
        lifecycle_summary=loaded["lifecycle_summary"],
        stop_file=args.stop_file,
        max_order_count=args.max_order_count,
        current_order_count=args.current_order_count,
        max_position_shares=args.max_position_shares,
        session_minutes=args.session_minutes,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[reports_dir / filename for filename in REPORT_FILES.values()],
        source_files=[Path(__file__), ROOT / "src" / "live" / "single_side_continuous_rehearsal.py"],
        root=ROOT,
    )
    return report


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
