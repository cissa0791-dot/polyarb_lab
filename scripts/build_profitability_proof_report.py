from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.profitability_proof import CONFIRMED_STATUS, REPORT_SCHEMA_VERSION, build_profitability_proof  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "profitability_proof_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build confirmed profitability proof report.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--summary-file", default="reward_adjusted_ev_lifecycle_summary_latest.json")
    parser.add_argument("--min-sample-count", type=int, default=30)
    parser.add_argument("--max-drawdown-usdc", type=float)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = Path(args.reports_dir)
    summary_path = reports_dir / args.summary_file
    report = build_profitability_proof(
        lifecycle_summary=load_json_report(summary_path),
        min_sample_count=args.min_sample_count,
        max_drawdown_usdc=args.max_drawdown_usdc,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[summary_path],
        source_files=[Path(__file__), ROOT / "src" / "live" / "profitability_proof.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == CONFIRMED_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
