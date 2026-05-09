from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.one_time_auth_token import DEFAULT_TOKEN_PATH  # noqa: E402
from src.live.post_live_probe_audit import (  # noqa: E402
    DEFAULT_AUDIT_ID,
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    build_post_live_probe_audit,
    load_json_report,
    markdown_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "post_live_probe_audit_001_latest.json"
DEFAULT_MD_OUT = DEFAULT_REPORTS_DIR / "post_live_probe_audit_001_latest.md"


REPORT_FILES = {
    "probe": "single_side_bid_probe_latest.json",
    "authorization": "single_side_probe_authorization_latest.json",
    "inventory_state": "inventory_state_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "gate": "live_readiness_gate_latest.json",
    "deposit_wallet": "deposit_wallet_readonly_latest.json",
    "heartbeat": "live_network_readiness_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build POST_LIVE_PROBE_AUDIT_001 from read-only reports.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--audit-id", default=DEFAULT_AUDIT_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict:
    reports_dir = Path(args.reports_dir)
    loaded = {key: load_json_report(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    token = load_json_report(args.token_file)
    report = build_post_live_probe_audit(
        probe=loaded["probe"],
        authorization=loaded["authorization"],
        token=token,
        inventory_state=loaded["inventory_state"],
        order_mutex=loaded["order_mutex"],
        gate=loaded["gate"],
        deposit_wallet=loaded["deposit_wallet"],
        heartbeat=loaded["heartbeat"],
        audit_id=args.audit_id,
        token_file=args.token_file,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[reports_dir / filename for filename in REPORT_FILES.values()] + [Path(args.token_file)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "post_live_probe_audit.py"],
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
