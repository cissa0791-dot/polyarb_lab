from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.live_readiness_gate import (  # noqa: E402
    DEFAULT_MAX_LIVE_RISK_USDC,
    DEFAULT_MAX_REPORT_AGE_MINUTES,
    REPORT_SCHEMA_VERSION,
    build_live_readiness_gate,
    markdown_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "live_readiness_gate_latest.json"
DEFAULT_MD_OUT = DEFAULT_REPORTS_DIR / "live_readiness_gate_latest.md"

REPORT_FILES = {
    "health": "live_api_health_readonly_now.json",
    "execution_system": "execution_disabled_auto_trade_system_latest.json",
    "profit_gate": "profit_test_gate_latest.json",
    "approval": "one_time_bootstrap_execution_approval_latest.json",
    "auth_readiness": "live_auth_readiness_latest.json",
    "deposit_wallet": "deposit_wallet_readonly_latest.json",
    "deployment": "deployment_consistency_latest.json",
    "network": "live_network_readiness_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "market_microstructure": "live_market_microstructure_latest.json",
    "fee_reconciliation": "fee_reconciliation_latest.json",
    "kill_switch": "kill_switch_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only live-readiness gate report.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--target-market-slug")
    parser.add_argument("--max-live-risk-usdc", type=float, default=DEFAULT_MAX_LIVE_RISK_USDC)
    parser.add_argument("--max-report-age-minutes", type=float, default=DEFAULT_MAX_REPORT_AGE_MINUTES)
    parser.add_argument(
        "--approved-action-scope",
        default="FIRST_CYCLE_BOOTSTRAP_EVIDENCE_GENERATION_ONLY",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def load_json(path: str | Path) -> tuple[dict[str, Any], str | None]:
    source = Path(path)
    if not source.exists():
        return {}, "REPORT_MISSING"
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}, "REPORT_INVALID_JSON"
    if not isinstance(payload, dict):
        return {}, "REPORT_NOT_OBJECT"
    return payload, None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    loaded: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    input_paths = [reports_dir / filename for filename in REPORT_FILES.values()]

    for key, filename in REPORT_FILES.items():
        payload, warning = load_json(reports_dir / filename)
        loaded[key] = payload
        if warning:
            warnings.append(f"{key}:{warning}")
        elif payload.get("stale_process_possible") is True:
            warnings.append(f"{key}:STALE_PROCESS_POSSIBLE")

    report = build_live_readiness_gate(
        health=loaded["health"],
        execution_system=loaded["execution_system"],
        profit_gate=loaded["profit_gate"],
        approval=loaded["approval"],
        auth_readiness=loaded["auth_readiness"],
        deposit_wallet=loaded["deposit_wallet"],
        deployment=loaded["deployment"],
        network=loaded["network"],
        order_mutex=loaded["order_mutex"],
        market_microstructure=loaded["market_microstructure"],
        fee_reconciliation=loaded["fee_reconciliation"],
        kill_switch=loaded["kill_switch"],
        target_market_slug=args.target_market_slug,
        approved_action_scope=args.approved_action_scope,
        max_live_risk_usdc=args.max_live_risk_usdc,
        max_report_age_minutes=args.max_report_age_minutes,
    )
    report.update(
        {
            "source_reports": {key: str(reports_dir / filename) for key, filename in REPORT_FILES.items()},
            "source_report_warnings": warnings,
        }
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=input_paths,
        source_files=[Path(__file__), ROOT / "src" / "live" / "live_readiness_gate.py"],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
