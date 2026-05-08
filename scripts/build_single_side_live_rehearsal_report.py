from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.report_metadata import attach_writer_metadata  # noqa: E402
from src.live.single_side_live_rehearsal import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    build_single_side_live_rehearsal_report,
    markdown_report,
)


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "single_side_live_rehearsal_latest.json"
DEFAULT_MD_OUT = DEFAULT_REPORTS_DIR / "single_side_live_rehearsal_latest.md"

REPORT_FILES = {
    "gate": "live_readiness_gate_latest.json",
    "deployment": "deployment_consistency_latest.json",
    "execution_system": "execution_disabled_auto_trade_system_latest.json",
    "deposit_wallet": "deposit_wallet_readonly_latest.json",
    "market_microstructure": "live_market_microstructure_latest.json",
    "network": "live_network_readiness_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "fee_reconciliation": "fee_reconciliation_latest.json",
    "inventory_state": "inventory_state_latest.json",
    "toxic_flow": "toxic_flow_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final MAKER_SINGLE_SIDE_LIVE_REHEARSAL review packet.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--target-market-slug")
    parser.add_argument("--max-live-risk-usdc", type=float)
    parser.add_argument("--order-side", choices=["BID_ONLY"], default="BID_ONLY")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports_dir = Path(args.reports_dir)
    loaded = {key: _load_json(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    report = build_single_side_live_rehearsal_report(
        gate=loaded["gate"],
        deployment=loaded["deployment"],
        execution_system=loaded["execution_system"],
        deposit_wallet=loaded["deposit_wallet"],
        market_microstructure=loaded["market_microstructure"],
        network=loaded["network"],
        order_mutex=loaded["order_mutex"],
        fee_reconciliation=loaded["fee_reconciliation"],
        inventory_state=loaded["inventory_state"],
        toxic_flow=loaded["toxic_flow"],
        branch=_git_stdout(["branch", "--show-current"]),
        commit_sha=_git_stdout(["rev-parse", "HEAD"]),
        target_market_slug=args.target_market_slug,
        max_live_risk_usdc=args.max_live_risk_usdc,
        order_side=args.order_side,
    )
    report["source_reports"] = {key: str(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[reports_dir / filename for filename in REPORT_FILES.values()],
        source_files=[Path(__file__), ROOT / "src" / "live" / "single_side_live_rehearsal.py"],
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


def _git_stdout(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=str(ROOT), check=False, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip()
    return text or None


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
    return 0 if report.get("status") == "SINGLE_SIDE_LIVE_REHEARSAL_REVIEW_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
