from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.one_time_auth_token import (  # noqa: E402
    DEFAULT_TOKEN_PATH,
    READY_STATUS,
    REPORT_SCHEMA_VERSION,
    build_authorization_report,
    load_authorization_token,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_OUT = ROOT / "data" / "reports" / "single_side_probe_authorization_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one-time single-side BID probe authorization token.")
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--market-slug", required=True)
    parser.add_argument("--max-live-risk-usdc", type=float, required=True)
    parser.add_argument("--quote-price", type=float, required=True)
    parser.add_argument("--quote-size", type=float, required=True)
    parser.add_argument("--hold-seconds", type=float)
    parser.add_argument("--planner-hash")
    parser.add_argument("--planner-report")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token, load_error = load_authorization_token(args.token_file)
    planner = _load_json(Path(args.planner_report)) if args.planner_report else {}
    expected_planner_hash = args.planner_hash or planner.get("planner_hash")
    expected_hold_seconds = args.hold_seconds
    if expected_hold_seconds is None and planner:
        expected_hold_seconds = _optional_float(planner.get("hold_seconds"))
    report = build_authorization_report(
        token=token,
        token_file=args.token_file,
        expected_market_slug=args.market_slug,
        expected_max_live_risk_usdc=args.max_live_risk_usdc,
        expected_quote_price=args.quote_price,
        expected_quote_size=args.quote_size,
        expected_hold_seconds=expected_hold_seconds,
        expected_planner_hash=expected_planner_hash,
        load_error=load_error,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.token_file)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "one_time_auth_token.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY_STATUS else 2


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
