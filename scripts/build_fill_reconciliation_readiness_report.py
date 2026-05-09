from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.fill_reconciliation_readiness import (  # noqa: E402
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    build_fill_reconciliation_readiness,
)
from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "fill_reconciliation_readiness_latest.json"

REPORT_FILES = {
    "order_status": "order_status_reconciliation_latest.json",
    "inventory_state": "inventory_state_latest.json",
    "deposit_wallet": "deposit_wallet_readonly_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "fee_reconciliation": "fee_reconciliation_latest.json",
    "previous_probe": "single_side_bid_probe_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only fill reconciliation readiness package.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict:
    reports_dir = Path(args.reports_dir)
    loaded = {key: load_json_report(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    report = build_fill_reconciliation_readiness(
        order_status=loaded["order_status"],
        inventory_state=loaded["inventory_state"],
        deposit_wallet=loaded["deposit_wallet"],
        order_mutex=loaded["order_mutex"],
        fee_reconciliation=loaded["fee_reconciliation"],
        previous_probe=loaded["previous_probe"],
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[reports_dir / filename for filename in REPORT_FILES.values()],
        source_files=[Path(__file__), ROOT / "src" / "live" / "fill_reconciliation_readiness.py"],
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
