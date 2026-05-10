from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.auth import load_live_credentials  # noqa: E402
from src.live.client import LiveWriteClient, clean_live_error_message  # noqa: E402
from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


REPORT_SCHEMA_VERSION = "order_status_reconciliation.v1"
DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_PROBE = DEFAULT_REPORTS_DIR / "single_side_bid_probe_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "order_status_reconciliation_latest.json"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read back one CLOB order status for post-probe reconciliation.")
    parser.add_argument("--probe-report", default=str(DEFAULT_PROBE))
    parser.add_argument("--order-id")
    parser.add_argument("--host", default=DEFAULT_CLOB_HOST)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    probe = load_json_report(args.probe_report)
    order_id = args.order_id or _order_id_from_probe(probe)
    report: dict[str, Any] = {
        "report_type": "order_status_reconciliation",
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "read_only": True,
        "order_id": order_id,
        "can_submit_order": False,
        "live_order_sent": False,
        "blockers": [],
    }
    if not order_id:
        report.update(
            {
                "status": "ORDER_STATUS_RECONCILIATION_BLOCKED",
                "blockers": ["ORDER_ID_MISSING"],
                "one_line_verdict": "ORDER_STATUS_RECONCILIATION_BLOCKED: ORDER_ID_MISSING.",
            }
        )
        return report

    try:
        creds = load_live_credentials()
        raw_signature_type = os.environ.get("POLYMARKET_SIGNATURE_TYPE")
        signature_type = int(raw_signature_type) if raw_signature_type not in {None, ""} else None
        funder = os.environ.get("POLYMARKET_FUNDER") or None
        client = LiveWriteClient.from_credentials(
            creds,
            host=args.host,
            dry_run=False,
            signature_type=signature_type,
            funder=funder,
        )
        raw = client.get_raw_order(str(order_id))
    except Exception as exc:
        reason = clean_live_error_message(exc) or str(exc) or type(exc).__name__
        report.update(
            {
                "status": "ORDER_STATUS_RECONCILIATION_BLOCKED",
                "blockers": [reason],
                "one_line_verdict": f"ORDER_STATUS_RECONCILIATION_BLOCKED: {reason}.",
            }
        )
        return report

    raw_status = str(raw.get("status") or "").strip()
    size_matched = _float_or_none(raw.get("size_matched") or raw.get("sizeMatched"))
    size_remaining = _float_or_none(raw.get("size_remaining") or raw.get("sizeRemaining") or raw.get("remaining_size"))
    original_size = _float_or_none(raw.get("original_size") or raw.get("originalSize") or raw.get("size"))
    field_blockers = _raw_field_blockers(
        raw_status=raw_status,
        size_matched=size_matched,
        size_remaining=size_remaining,
        original_size=original_size,
    )
    if field_blockers:
        report.update(
            {
                "status": "ORDER_STATUS_RECONCILIATION_BLOCKED",
                "raw_order_status": raw_status or None,
                "size_matched": _round(size_matched),
                "size_remaining": _round(size_remaining),
                "original_size": _round(original_size),
                "price": _round(raw.get("price")),
                "side": raw.get("side"),
                "asset_id": raw.get("asset_id") or raw.get("assetId"),
                "market": raw.get("market"),
                "blockers": field_blockers,
                "one_line_verdict": f"ORDER_STATUS_RECONCILIATION_BLOCKED: {', '.join(field_blockers)}.",
            }
        )
        return report
    report.update(
        {
            "status": "ORDER_STATUS_RECONCILIATION_READY",
            "raw_order_status": raw_status,
            "size_matched": _round(size_matched),
            "size_remaining": _round(size_remaining),
            "original_size": _round(original_size),
            "price": _round(raw.get("price")),
            "side": raw.get("side"),
            "asset_id": raw.get("asset_id") or raw.get("assetId"),
            "market": raw.get("market"),
            "raw_order_cancelled_zero_fill": raw_status.upper() in {"CANCELED", "CANCELLED"}
            and _same_float(size_matched, 0.0),
            "one_line_verdict": f"ORDER_STATUS_RECONCILIATION_READY: order {order_id} is {raw_status}.",
        }
    )
    return report


def _order_id_from_probe(probe: dict[str, Any]) -> str | None:
    submit = probe.get("submit_result") if isinstance(probe.get("submit_result"), dict) else {}
    cancel = probe.get("cancel_result") if isinstance(probe.get("cancel_result"), dict) else {}
    value = submit.get("order_id") or cancel.get("order_id")
    return str(value) if value else None


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 6) -> float | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _raw_field_blockers(
    *,
    raw_status: str,
    size_matched: float | None,
    size_remaining: float | None,
    original_size: float | None,
) -> list[str]:
    blockers: list[str] = []
    if not raw_status:
        blockers.append("RAW_ORDER_STATUS_MISSING")
    if size_matched is None:
        blockers.append("SIZE_MATCHED_MISSING")
    if size_remaining is None and original_size is None:
        blockers.append("ORDER_SIZE_FIELDS_MISSING")
    return blockers


def _same_float(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    left_float = _float_or_none(left)
    right_float = _float_or_none(right)
    if left_float is None or right_float is None:
        return False
    return abs(left_float - right_float) <= tolerance


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.probe_report)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "client.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "ORDER_STATUS_RECONCILIATION_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
