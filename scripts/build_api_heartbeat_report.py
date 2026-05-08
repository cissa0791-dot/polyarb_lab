from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.api_heartbeat_probe import (  # noqa: E402
    DEFAULT_CRITICAL_LATENCY_MS,
    DEFAULT_HEALTHY_LATENCY_MS,
    DEFAULT_HEARTBEAT_URL,
    DEFAULT_MASS_CANCEL_FAILURE_THRESHOLD,
    DEFAULT_TIMEOUT_SEC,
    REPORT_SCHEMA_VERSION,
    build_api_heartbeat_report,
    run_http_heartbeat_probe,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "api_heartbeat_latest.json"
DEFAULT_NETWORK_OUT = DEFAULT_REPORTS_DIR / "live_network_readiness_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only API heartbeat and cancel-latency report.")
    parser.add_argument("--url", default=DEFAULT_HEARTBEAT_URL)
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--healthy-ms", type=float, default=DEFAULT_HEALTHY_LATENCY_MS)
    parser.add_argument("--critical-ms", type=float, default=DEFAULT_CRITICAL_LATENCY_MS)
    parser.add_argument("--mass-cancel-failure-threshold", type=int, default=DEFAULT_MASS_CANCEL_FAILURE_THRESHOLD)
    parser.add_argument("--previous-report", default=str(DEFAULT_OUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--network-out", default=str(DEFAULT_NETWORK_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    probe = run_http_heartbeat_probe(url=args.url, timeout_sec=args.timeout_sec)
    previous_report = _load_json(Path(args.previous_report))
    report = build_api_heartbeat_report(
        probe=probe,
        previous_report=previous_report,
        healthy_latency_ms=args.healthy_ms,
        critical_latency_ms=args.critical_ms,
        mass_cancel_failure_threshold=args.mass_cancel_failure_threshold,
        timeout_sec=args.timeout_sec,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.previous_report)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "api_heartbeat_probe.py"],
        root=ROOT,
    )
    return report


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    out.write_text(payload, encoding="utf-8")
    network_out = Path(args.network_out)
    network_out.parent.mkdir(parents=True, exist_ok=True)
    network_out.write_text(payload, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("is_within_safety_threshold") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
