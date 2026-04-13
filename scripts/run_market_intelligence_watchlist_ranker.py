from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.intelligence.watchlist_ranker import (
    build_watchlist_report,
    load_latest_intelligence_inputs,
    load_latest_trade_flow_report,
    load_live_delta_events,
    write_watchlist_report,
)
from src.storage.event_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-only market intelligence watchlist report.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory containing latest intelligence outputs.")
    parser.add_argument("--delta-limit", type=int, default=500, help="Maximum number of stored live delta events to rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    out_dir = Path(args.out_dir)
    registry, snapshot_report, live_delta_report = load_latest_intelligence_inputs(out_dir)
    trade_flow_report = load_latest_trade_flow_report(out_dir)
    store = ResearchStore(config.storage.sqlite_url)
    try:
        delta_events = load_live_delta_events(store, limit=args.delta_limit)
    finally:
        store.close()

    report = build_watchlist_report(
        registry=registry,
        snapshot_report=snapshot_report,
        live_delta_report=live_delta_report,
        delta_events=delta_events,
        trade_flow_report=trade_flow_report,
    )
    written = write_watchlist_report(out_dir=out_dir, report=report)
    print(
        json.dumps(
            {
                "live_delta_events": report["summary"]["live_delta_events"],
                "markets_ranked": report["summary"]["markets_ranked"],
                "events_ranked": report["summary"]["events_ranked"],
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
