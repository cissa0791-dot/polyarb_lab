from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402
from src.live.two_sided_maker_readiness import REPORT_SCHEMA_VERSION, build_two_sided_maker_readiness  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "two_sided_maker_readiness_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preparation-only two-sided maker readiness report.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = Path(args.reports_dir)
    report = build_two_sided_maker_readiness(
        single_side_rehearsal=load_json_report(reports_dir / "single_side_continuous_rehearsal_latest.json"),
        inventory_model=load_json_report(reports_dir / "two_sided_inventory_model_latest.json"),
        split_merge_redeem=load_json_report(reports_dir / "split_merge_redeem_readiness_latest.json"),
        simultaneous_fill_risk=load_json_report(reports_dir / "simultaneous_fill_risk_latest.json"),
        two_sided_mutex=load_json_report(reports_dir / "two_sided_mutex_readiness_latest.json"),
        post_fill_audit=load_json_report(reports_dir / "two_sided_post_fill_audit_latest.json"),
        approval_context=load_json_report(reports_dir / "single_side_probe_authorization_latest.json"),
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        source_files=[Path(__file__), ROOT / "src" / "live" / "two_sided_maker_readiness.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
