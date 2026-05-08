from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.fee_auditor import (  # noqa: E402
    READY,
    REPORT_SCHEMA_VERSION,
    build_fee_reconciliation_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_MARKET_REPORT = DEFAULT_REPORTS_DIR / "live_market_microstructure_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "fee_reconciliation_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only fee and gas reconciliation report.")
    parser.add_argument("--market-report", default=str(DEFAULT_MARKET_REPORT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--market-slug")
    parser.add_argument("--quote-bid", type=float)
    parser.add_argument("--quote-ask", type=float)
    parser.add_argument("--quote-size", type=float)
    parser.add_argument("--quote-spread", type=float)
    parser.add_argument("--maker-fee-rate", type=float)
    parser.add_argument("--taker-fee-rate", type=float)
    parser.add_argument("--entry-liquidity-role", choices=["maker", "taker"], default=None)
    parser.add_argument("--exit-liquidity-role", choices=["maker", "taker"], default=None)
    parser.add_argument("--priority-fee-gwei", type=float)
    parser.add_argument("--polygon-rpc-url")
    parser.add_argument("--rpc-timeout-sec", type=float)
    parser.add_argument("--gas-units-per-cancel", type=float)
    parser.add_argument("--gas-asset-usdc", type=float)
    parser.add_argument("--cancel-tx-count", type=float)
    parser.add_argument("--estimated-gas-costs-usdc", type=float)
    parser.add_argument("--reward-payout-mismatch", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    market_report = _load_json(Path(args.market_report))
    explicit = {
        "market_slug": args.market_slug,
        "quote_bid": args.quote_bid,
        "quote_ask": args.quote_ask,
        "quote_size": args.quote_size,
        "quote_spread": args.quote_spread,
        "maker_fee_rate": args.maker_fee_rate,
        "taker_fee_rate": args.taker_fee_rate,
        "entry_liquidity_role": args.entry_liquidity_role,
        "exit_liquidity_role": args.exit_liquidity_role,
        "priority_fee_gwei": args.priority_fee_gwei,
        "polygon_rpc_url": args.polygon_rpc_url,
        "rpc_timeout_sec": args.rpc_timeout_sec,
        "gas_units_per_cancel": args.gas_units_per_cancel,
        "gas_asset_usdc": args.gas_asset_usdc,
        "cancel_tx_count": args.cancel_tx_count,
        "estimated_gas_costs_usdc": args.estimated_gas_costs_usdc,
        "reward_payout_mismatch": args.reward_payout_mismatch,
    }
    report = build_fee_reconciliation_report(
        market_microstructure=market_report,
        explicit=explicit,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.market_report)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "fee_auditor.py"],
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
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
