from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.intelligence.watchlist_ranker import load_latest_intelligence_inputs, load_live_delta_events
from src.intelligence.watchlist_replay_validator import (
    build_watchlist_validation_report,
    write_watchlist_validation_report,
)
from src.storage.event_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate watchlist ranking over stored live delta events.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory containing latest intelligence outputs.")
    parser.add_argument("--delta-limit", type=int, default=500, help="Maximum stored live delta events to replay.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    out_dir = Path(args.out_dir)
    registry, snapshot_report, live_delta_report = load_latest_intelligence_inputs(out_dir)
    watchlist_report = json.loads((out_dir / "market_intelligence_watchlist_latest.json").read_text(encoding="utf-8"))

    store = ResearchStore(config.storage.sqlite_url)
    try:
        delta_events = load_live_delta_events(store, limit=args.delta_limit)
    finally:
        store.close()

    report = build_watchlist_validation_report(
        registry=registry,
        live_delta_report=live_delta_report,
        watchlist_report=watchlist_report,
        delta_events=delta_events,
    )
    written = write_watchlist_validation_report(out_dir=out_dir, report=report)
    print(
        json.dumps(
            {
                "live_delta_events": report["summary"]["live_delta_events"],
                "validation_forward_events": report["summary"]["validation_forward_events"],
                "markets_ranked": report["summary"]["markets_ranked"],
                "events_ranked": report["summary"]["events_ranked"],
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
