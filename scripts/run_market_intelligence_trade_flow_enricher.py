from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.intelligence.trade_flow import (
    build_trade_flow_report,
    collect_market_trade_flow,
    write_trade_flow_report,
)
from src.intelligence.watchlist_ranker import load_latest_intelligence_inputs, load_live_delta_events
from src.monitoring.logger import configure_logging
from src.storage.event_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper-only trade-flow enricher over official Polymarket market websocket data.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory containing latest registry/snapshot/live outputs.")
    parser.add_argument("--duration-sec", type=float, default=15.0, help="Seconds to listen on the official market websocket.")
    parser.add_argument("--delta-limit", type=int, default=500, help="Maximum stored live delta events to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    configure_logging(level=config.monitoring.log_level, json_logs=config.monitoring.json_logs)
    out_dir = Path(args.out_dir)
    registry, _snapshot_report, _live_delta_report = load_latest_intelligence_inputs(out_dir)

    token_ids: list[str] = []
    token_to_market: dict[str, str] = {}
    for event in registry.get("events", []):
        for market in event.get("markets", []):
            market_slug = str(market.get("slug") or "")
            for token_id in (market.get("yes_token_id"), market.get("no_token_id")):
                if not token_id:
                    continue
                token_id = str(token_id)
                token_ids.append(token_id)
                token_to_market[token_id] = market_slug

    store = ResearchStore(config.storage.sqlite_url)
    try:
        trade_events, bbo_events = asyncio.run(
            collect_market_trade_flow(
                token_ids=token_ids,
                token_to_market=token_to_market,
                duration_sec=float(args.duration_sec),
                store=store,
            )
        )
        live_delta_events = load_live_delta_events(store, limit=args.delta_limit)
        report = build_trade_flow_report(
            registry=registry,
            trade_events=trade_events,
            bbo_events=bbo_events,
            live_delta_events=live_delta_events,
        )
        written = write_trade_flow_report(out_dir=out_dir, report=report)
    finally:
        store.close()

    print(
        json.dumps(
            {
                "trade_events_seen": report["summary"]["trade_events_seen"],
                "bbo_events_seen": report["summary"]["bbo_events_seen"],
                "markets_with_trades": report["summary"]["markets_with_trades"],
                "events_with_trades": report["summary"]["events_with_trades"],
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
