from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_events, fetch_markets
from src.ingest.clob import ReadOnlyClob
from src.intelligence.market_intelligence import (
    build_daily_paper_report,
    build_event_market_registry,
    collect_registry_token_ids,
    write_snapshot_outputs,
)
from src.monitoring.logger import configure_logging, get_logger
from src.storage.event_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper-only Polymarket market intelligence snapshot.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--market-limit", type=int, default=None, help="Optional override for active market/event fetch limit.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory for registry/report outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    configure_logging(level=config.monitoring.log_level, json_logs=config.monitoring.json_logs)
    logger = get_logger("polyarb.intelligence")
    limit = args.market_limit or config.market_data.market_limit
    now = datetime.now(timezone.utc)

    store = ResearchStore(config.storage.sqlite_url)
    clob = ReadOnlyClob(config.market_data.clob_host)
    try:
        events = fetch_events(config.market_data.gamma_host, limit=limit)
        markets = fetch_markets(config.market_data.gamma_host, limit=limit)

        store.save_raw_snapshot("gamma_events", f"events:{limit}", events, now)
        store.save_raw_snapshot("gamma_markets", f"markets:{limit}", markets, now)

        registry = build_event_market_registry(events, markets)
        token_ids = collect_registry_token_ids(registry)

        books_by_token: dict[str, object] = {}
        for token_id in token_ids:
            try:
                book = clob.get_book(token_id)
            except Exception as exc:
                logger.warning(
                    "book fetch failed",
                    extra={"payload": {"token_id": token_id, "error": str(exc)}},
                )
                continue
            books_by_token[token_id] = book
            store.save_raw_snapshot("clob_book", token_id, book.model_dump(mode="json"), now)

        report = build_daily_paper_report(registry, books_by_token)
        written = write_snapshot_outputs(
            out_dir=Path(args.out_dir),
            registry=registry,
            report=report,
        )

        print(
            json.dumps(
                {
                    "events_seen": registry["summary"]["events_seen"],
                    "markets_seen": registry["summary"]["markets_seen"],
                    "tracked_tokens": registry["summary"]["tracked_tokens"],
                    "books_fetched": len(books_by_token),
                    "written_files": {key: str(path) for key, path in written.items()},
                },
                indent=2,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
