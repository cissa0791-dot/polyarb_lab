from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.full_market_probe_planner import REPORT_SCHEMA_VERSION, build_full_market_probe_plan  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_OUT = ROOT / "data" / "reports" / "full_market_probe_planner_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only full-market probe planner report.")
    parser.add_argument("--candidates", required=True, help="JSON file containing a list of candidate objects.")
    parser.add_argument("--max-live-risk-usdc", type=float, required=True)
    parser.add_argument("--requested-probe-type", default="B_LONG_OBSERVATION_STABILITY")
    parser.add_argument("--stability-max-fill-probability", type=float, default=0.30)
    parser.add_argument("--universe-complete", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates_path = Path(args.candidates)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8-sig"))
    if not isinstance(candidates, list):
        candidates = []
    report = build_full_market_probe_plan(
        candidates=candidates,
        max_live_risk_usdc=args.max_live_risk_usdc,
        requested_probe_type=args.requested_probe_type,
        stability_max_fill_probability=args.stability_max_fill_probability,
        scan_scope={
            "declared_universe": str(candidates_path),
            "universe_complete": args.universe_complete,
        },
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[candidates_path],
        source_files=[Path(__file__), ROOT / "src" / "live" / "full_market_probe_planner.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "FULL_MARKET_PROBE_PLAN_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
