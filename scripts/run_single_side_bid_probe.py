from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.auth import load_live_credentials  # noqa: E402
from src.live.client import LiveWriteClient  # noqa: E402
from src.live.one_time_auth_token import DEFAULT_TOKEN_PATH  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402
from src.live.single_side_bid_probe import (  # noqa: E402
    DEFAULT_HOLD_SECONDS,
    DEFAULT_STATUS_POLL_SECONDS,
    REPORT_SCHEMA_VERSION,
    load_json_report,
    run_single_side_bid_probe,
    write_probe_report,
)


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "single_side_bid_probe_latest.json"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"


REPORT_FILES = {
    "gate": "live_readiness_gate_latest.json",
    "rehearsal": "single_side_live_rehearsal_latest.json",
    "token_report": "single_side_probe_authorization_latest.json",
    "health": "live_api_health_readonly_now.json",
    "market_microstructure": "live_market_microstructure_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-time single-side BID probe controller. Default mode never submits live orders."
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--clob-host", default=DEFAULT_CLOB_HOST)
    parser.add_argument("--max-live-risk-usdc", type=float, required=True)
    parser.add_argument("--quote-price", type=float, required=True)
    parser.add_argument("--quote-size", type=float, required=True)
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument("--status-poll-seconds", type=float, default=DEFAULT_STATUS_POLL_SECONDS)
    parser.add_argument("--max-report-age-minutes", type=float, default=2.0)
    parser.add_argument(
        "--execute-live-probe",
        action="store_true",
        help="Actually submit one live BID probe. Requires all explicit live confirmation flags.",
    )
    parser.add_argument(
        "--consume-token",
        action="store_true",
        help="Mark the one-time token EXPENDED before the live submit call. Required for live execution.",
    )
    parser.add_argument(
        "--acknowledge-live-risk",
        action="store_true",
        help="Required acknowledgement that this can send one real order.",
    )
    parser.add_argument(
        "--confirm-single-side-bid-probe",
        action="store_true",
        help="Required acknowledgement that this is BID_ONLY, one order, no retry, no both-side live.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def _live_client(args: argparse.Namespace) -> LiveWriteClient | None:
    if not args.execute_live_probe:
        return None
    if not (args.consume_token and args.acknowledge_live_risk and args.confirm_single_side_bid_probe):
        return None
    creds = load_live_credentials()
    raw_signature_type = os.environ.get("POLYMARKET_SIGNATURE_TYPE")
    signature_type = int(raw_signature_type) if raw_signature_type not in {None, ""} else None
    funder = os.environ.get("POLYMARKET_FUNDER") or None
    return LiveWriteClient.from_credentials(
        creds,
        host=args.clob_host,
        dry_run=False,
        signature_type=signature_type,
        funder=funder,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = Path(args.reports_dir)
    loaded = {key: load_json_report(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    try:
        client = _live_client(args)
        report = run_single_side_bid_probe(
            gate=loaded["gate"],
            rehearsal=loaded["rehearsal"],
            token_report=loaded["token_report"],
            health=loaded["health"],
            market_microstructure=loaded["market_microstructure"],
            max_live_risk_usdc=args.max_live_risk_usdc,
            quote_price=args.quote_price,
            quote_size=args.quote_size,
            token_file=args.token_file,
            execute_live_probe=args.execute_live_probe,
            consume_token=args.consume_token,
            acknowledge_live_risk=args.acknowledge_live_risk,
            confirm_single_side_bid_probe=args.confirm_single_side_bid_probe,
            hold_seconds=args.hold_seconds,
            status_poll_seconds=args.status_poll_seconds,
            client=client,
            max_report_age_minutes=args.max_report_age_minutes,
        )
    except Exception as exc:
        report = {
            "report_type": "single_side_bid_probe",
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "status": "SINGLE_SIDE_BID_PROBE_BLOCKED",
            "reason": str(exc),
            "can_submit_order": False,
            "live_order_sent": False,
            "blockers": [type(exc).__name__],
            "one_line_verdict": f"SINGLE_SIDE_BID_PROBE_BLOCKED: {type(exc).__name__}.",
        }

    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[reports_dir / filename for filename in REPORT_FILES.values()] + [Path(args.token_file)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "single_side_bid_probe.py"],
        root=ROOT,
    )
    write_probe_report(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") in {"SINGLE_SIDE_BID_PROBE_DRY_RUN_READY", "SINGLE_SIDE_BID_PROBE_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
