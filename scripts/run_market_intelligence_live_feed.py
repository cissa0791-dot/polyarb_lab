from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.clob import ReadOnlyClob
from src.ingest.gamma import fetch_events, fetch_markets
from src.intelligence.live_feed import (
    build_live_delta_report,
    compute_book_delta,
    summarize_book,
    write_live_delta_report,
)
from src.intelligence.market_intelligence import (
    build_event_market_registry,
    collect_registry_token_ids,
)
from src.monitoring.logger import configure_logging, get_logger
from src.storage.event_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal paper-only market intelligence live feed.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--market-limit", type=int, default=None, help="Optional override for active event/market fetch limit.")
    parser.add_argument("--poll-count", type=int, default=2, help="Number of poll rounds, including baseline.")
    parser.add_argument("--poll-interval-sec", type=float, default=2.0, help="Seconds between poll rounds.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory for delta reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    configure_logging(level=config.monitoring.log_level, json_logs=config.monitoring.json_logs)
    logger = get_logger("polyarb.intelligence.live")
    limit = args.market_limit or config.market_data.market_limit
    poll_count = max(1, int(args.poll_count))

    store = ResearchStore(config.storage.sqlite_url)
    clob = ReadOnlyClob(config.market_data.clob_host)
    try:
        snapshot_ts = datetime.now(timezone.utc)
        events = fetch_events(config.market_data.gamma_host, limit=limit)
        markets = fetch_markets(config.market_data.gamma_host, limit=limit)
        registry = build_event_market_registry(events, markets)
        token_ids = collect_registry_token_ids(registry)

        store.save_raw_snapshot("gamma_events", f"live_events:{limit}", events, snapshot_ts)
        store.save_raw_snapshot("gamma_markets", f"live_markets:{limit}", markets, snapshot_ts)

        token_to_market: dict[str, str] = {}
        for event in registry.get("events", []):
            for market in event.get("markets", []):
                for token_id in (market.get("yes_token_id"), market.get("no_token_id")):
                    if token_id:
                        token_to_market[str(token_id)] = str(market.get("slug") or "")

        previous_books: dict[str, dict] = {}
        delta_events: list[dict] = []

        for poll_index in range(poll_count):
            observed_ts = datetime.now(timezone.utc)
            for token_id in token_ids:
                try:
                    book = clob.get_book(token_id)
                except Exception as exc:
                    logger.warning(
                        "book fetch failed",
                        extra={"payload": {"token_id": token_id, "error": str(exc), "poll_index": poll_index}},
                    )
                    current_summary = {
                        "token_id": token_id,
                        "best_bid": None,
                        "best_ask": None,
                        "best_bid_size": None,
                        "best_ask_size": None,
                        "spread": None,
                        "bid_levels": 0,
                        "ask_levels": 0,
                        "complete_top_of_book": False,
                    }
                else:
                    store.save_raw_snapshot("clob_live_book", f"{token_id}:poll:{poll_index}", book.model_dump(mode="json"), observed_ts)
                    current_summary = summarize_book(book)

                delta = compute_book_delta(
                    token_id=token_id,
                    market_slug=token_to_market.get(token_id, ""),
                    previous=previous_books.get(token_id),
                    current=current_summary,
                    observed_ts=observed_ts,
                )
                previous_books[token_id] = current_summary
                if poll_index == 0 or delta is None:
                    continue
                delta_events.append(delta)
                store.save_raw_snapshot("clob_live_delta", f"{token_id}:{observed_ts.isoformat()}", delta, observed_ts)

            if poll_index < poll_count - 1 and args.poll_interval_sec > 0:
                time.sleep(args.poll_interval_sec)

        report = build_live_delta_report(registry=registry, delta_events=delta_events)
        written = write_live_delta_report(out_dir=Path(args.out_dir), report=report)
        print(
            json.dumps(
                {
                    "events_seen": registry["summary"]["events_seen"],
                    "markets_seen": registry["summary"]["markets_seen"],
                    "tracked_tokens": registry["summary"]["tracked_tokens"],
                    "delta_events": len(delta_events),
                    "written_files": {key: str(path) for key, path in written.items()},
                },
                indent=2,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
