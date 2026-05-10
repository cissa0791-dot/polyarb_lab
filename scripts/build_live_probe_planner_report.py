from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.live_probe_planner import (  # noqa: E402
    DEFAULT_STABILITY_MAX_FILL_PROBABILITY,
    PROBE_INTENT_FILL_LIKELIHOOD,
    PROBE_INTENT_STABILITY,
    REPORT_SCHEMA_VERSION,
    build_live_probe_plan,
    markdown_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "live_probe_planner_latest.json"
DEFAULT_MD_OUT = DEFAULT_REPORTS_DIR / "live_probe_planner_latest.md"

REPORT_FILES = {
    "gate": "live_readiness_gate_latest.json",
    "deposit_wallet": "deposit_wallet_readonly_latest.json",
    "market_microstructure": "live_market_microstructure_latest.json",
    "toxic_flow": "toxic_flow_latest.json",
    "fee_reconciliation": "fee_reconciliation_latest.json",
    "inventory_state": "inventory_state_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "network": "live_network_readiness_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the read-only next live-probe planner report.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--target-market-slug")
    parser.add_argument("--max-live-risk-usdc", type=float)
    parser.add_argument(
        "--min-probe-size",
        type=float,
        default=None,
        help="Minimum probe size. Defaults to 50 for B stability probes and 10 for C fill-reconciliation probes.",
    )
    parser.add_argument("--hold-seconds", type=int, default=300)
    parser.add_argument("--token-ttl-seconds", type=int, default=600)
    parser.add_argument("--planner-valid-seconds", type=int, default=120)
    parser.add_argument("--visibility-grace-period-ms", type=float, default=2000.0)
    parser.add_argument(
        "--probe-intent",
        choices=[PROBE_INTENT_STABILITY, PROBE_INTENT_FILL_LIKELIHOOD],
        default=PROBE_INTENT_STABILITY,
        help="Read-only experiment class the planner is allowed to recommend.",
    )
    parser.add_argument(
        "--stability-max-fill-probability",
        type=float,
        default=DEFAULT_STABILITY_MAX_FILL_PROBABILITY,
        help="Maximum allowed fill probability for a long-observation stability probe.",
    )
    parser.add_argument(
        "--candidate-market-report",
        action="append",
        default=[],
        help="Optional additional market microstructure report path. May be supplied more than once.",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    loaded = {key: _load_json(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    candidate_paths = [Path(path) for path in args.candidate_market_report]
    candidate_reports = [_load_json(path) for path in candidate_paths]
    report = build_live_probe_plan(
        gate=loaded["gate"],
        deposit_wallet=loaded["deposit_wallet"],
        market_microstructure=loaded["market_microstructure"],
        candidate_market_microstructures=candidate_reports,
        toxic_flow=loaded["toxic_flow"],
        fee_reconciliation=loaded["fee_reconciliation"],
        inventory_state=loaded["inventory_state"],
        order_mutex=loaded["order_mutex"],
        network=loaded["network"],
        target_market_slug=args.target_market_slug,
        max_live_risk_usdc=args.max_live_risk_usdc,
        min_probe_size=_default_min_probe_size(args),
        hold_seconds=args.hold_seconds,
        token_ttl_seconds=args.token_ttl_seconds,
        planner_valid_seconds=args.planner_valid_seconds,
        visibility_grace_period_ms=args.visibility_grace_period_ms,
        probe_intent=args.probe_intent,
        stability_max_fill_probability=args.stability_max_fill_probability,
    )
    input_paths = [reports_dir / filename for filename in REPORT_FILES.values()] + candidate_paths
    report["source_reports"] = {key: str(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    if candidate_paths:
        report["additional_candidate_report_paths"] = [str(path) for path in candidate_paths]
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=input_paths,
        source_files=[Path(__file__), ROOT / "src" / "live" / "live_probe_planner.py"],
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


def _default_min_probe_size(args: argparse.Namespace) -> float:
    if args.min_probe_size is not None:
        return float(args.min_probe_size)
    if args.probe_intent == PROBE_INTENT_FILL_LIKELIHOOD:
        return 10.0
    return 50.0


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
    return 0 if report.get("status") == "LIVE_PROBE_PLAN_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
