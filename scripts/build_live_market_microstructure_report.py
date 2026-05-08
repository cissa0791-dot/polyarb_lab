from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.clob_compat import ClobClient  # noqa: E402
from src.live.market_microstructure_readiness import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    build_market_microstructure_report,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_HEALTH = DEFAULT_REPORTS_DIR / "live_api_health_readonly_now.json"
DEFAULT_CANDIDATE = DEFAULT_REPORTS_DIR / "maker_engine_A_p0_exit_aware_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "live_market_microstructure_latest.json"
DEFAULT_CLOB_HOST = "https://clob.polymarket.com"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only market microstructure quote sanity report.")
    parser.add_argument("--health-report", default=str(DEFAULT_HEALTH))
    parser.add_argument("--candidate-report", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--market-slug")
    parser.add_argument("--token-id")
    parser.add_argument("--quote-bid", type=float)
    parser.add_argument("--quote-ask", type=float)
    parser.add_argument("--quote-size", type=float)
    parser.add_argument("--tick-size", type=float)
    parser.add_argument("--best-bid", type=float)
    parser.add_argument("--best-ask", type=float)
    parser.add_argument("--rewards-min-size", type=float)
    parser.add_argument("--rewards-max-spread-cents", type=float)
    parser.add_argument("--clob-host", default=DEFAULT_CLOB_HOST)
    parser.add_argument("--chain-id", type=int, default=137)
    parser.add_argument("--fetch-tick-size", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    explicit = {
        "market_slug": args.market_slug,
        "token_id": args.token_id,
        "quote_bid": args.quote_bid,
        "quote_ask": args.quote_ask,
        "quote_size": args.quote_size,
        "tick_size": args.tick_size,
        "best_bid": args.best_bid,
        "best_ask": args.best_ask,
        "rewards_min_size": args.rewards_min_size,
        "rewards_max_spread_cents": args.rewards_max_spread_cents,
    }
    tick_reader = _tick_reader(args.clob_host, args.chain_id) if args.fetch_tick_size else None
    report = build_market_microstructure_report(
        candidate_report=_load_json(args.candidate_report),
        health_report=_load_json(args.health_report),
        explicit=explicit,
        tick_size_reader=tick_reader,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[Path(args.candidate_report), Path(args.health_report)],
        source_files=[Path(__file__), ROOT / "src" / "live" / "market_microstructure_readiness.py"],
        root=ROOT,
    )
    return report


def _load_json(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tick_reader(host: str, chain_id: int):
    try:
        client = ClobClient(host, chain_id=chain_id)
    except TypeError:
        client = ClobClient(host)

    def read(token_id: str) -> Any:
        return client.get_tick_size(str(token_id))

    return read


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == "MARKET_MICROSTRUCTURE_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
