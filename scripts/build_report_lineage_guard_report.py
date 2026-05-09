from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.post_live_probe_audit import load_json_report  # noqa: E402
from src.live.report_lineage_guard import READY_STATUS, REPORT_SCHEMA_VERSION, build_report_lineage_guard  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "report_lineage_guard_latest.json"

REPORT_FILES = {
    "probe": "single_side_bid_probe_latest.json",
    "order_reconciliation": "order_status_reconciliation_latest.json",
    "market_microstructure": "live_market_microstructure_latest.json",
    "planner": "live_probe_planner_latest.json",
    "fee_reconciliation": "fee_reconciliation_latest.json",
    "toxic_flow": "toxic_flow_latest.json",
    "reward_report": "reward_adjusted_ev_lifecycle_summary_latest.json",
    "inventory_state": "inventory_state_latest.json",
    "order_mutex": "order_mutex_readiness_latest.json",
    "gate": "live_readiness_gate_latest.json",
    "deposit_wallet": "deposit_wallet_readonly_latest.json",
    "heartbeat": "live_network_readiness_latest.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-probe report lineage guard.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-report-age-minutes", type=float, default=10.0)
    parser.add_argument("--require-reward-report", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports_dir = Path(args.reports_dir)
    loaded = {key: load_json_report(reports_dir / filename) for key, filename in REPORT_FILES.items()}
    report = build_report_lineage_guard(
        probe=loaded["probe"],
        order_reconciliation=loaded["order_reconciliation"],
        market_microstructure=loaded["market_microstructure"],
        planner=loaded["planner"],
        fee_reconciliation=loaded["fee_reconciliation"],
        toxic_flow=loaded["toxic_flow"],
        reward_report=loaded["reward_report"],
        inventory_state=loaded["inventory_state"],
        order_mutex=loaded["order_mutex"],
        gate=loaded["gate"],
        deposit_wallet=loaded["deposit_wallet"],
        heartbeat=loaded["heartbeat"],
        require_reward_report=args.require_reward_report,
        max_report_age_minutes=args.max_report_age_minutes,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[reports_dir / filename for filename in REPORT_FILES.values()],
        source_files=[Path(__file__), ROOT / "src" / "live" / "report_lineage_guard.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
