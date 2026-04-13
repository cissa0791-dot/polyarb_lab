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
    build_daily_paper_report,
    build_event_market_registry,
    collect_registry_token_ids,
    write_snapshot_outputs,
)
from src.intelligence.watchlist_multi_run import (
    build_multi_run_stability_report,
    write_multi_run_stability_report,
)
from src.intelligence.watchlist_ranker import build_watchlist_report, write_watchlist_report
from src.intelligence.watchlist_replay_validator import (
    build_watchlist_validation_report,
    write_watchlist_validation_report,
)
from src.monitoring.logger import configure_logging, get_logger
from src.storage.event_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated paper-only watchlist validation and aggregate stability.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--out-dir", default="data/reports", help="Directory for intelligence reports.")
    parser.add_argument("--market-limit", type=int, default=None, help="Optional override for active market/event fetch limit.")
    parser.add_argument("--run-count", type=int, default=2, help="Number of repeated runs.")
    parser.add_argument("--poll-count", type=int, default=2, help="Number of live-feed polls per run, including baseline.")
    parser.add_argument("--poll-interval-sec", type=float, default=2.0, help="Seconds between live-feed polls.")
    parser.add_argument("--run-interval-sec", type=float, default=0.0, help="Seconds between repeated runs.")
    return parser.parse_args()


def _run_single_cycle(
    *,
    config,
    out_dir: Path,
    limit: int,
    poll_count: int,
    poll_interval_sec: float,
    logger,
) -> dict[str, object]:
    snapshot_ts = datetime.now(timezone.utc)
    store = ResearchStore(config.storage.sqlite_url)
    clob = ReadOnlyClob(config.market_data.clob_host)
    try:
        events = fetch_events(config.market_data.gamma_host, limit=limit)
        markets = fetch_markets(config.market_data.gamma_host, limit=limit)
        store.save_raw_snapshot("gamma_events", f"events:{limit}", events, snapshot_ts)
        store.save_raw_snapshot("gamma_markets", f"markets:{limit}", markets, snapshot_ts)

        registry = build_event_market_registry(events, markets)
        token_ids = collect_registry_token_ids(registry)

        books_by_token: dict[str, object] = {}
        for token_id in token_ids:
            try:
                book = clob.get_book(token_id)
            except Exception as exc:
                logger.warning("book fetch failed", extra={"payload": {"token_id": token_id, "error": str(exc)}})
                continue
            books_by_token[token_id] = book
            store.save_raw_snapshot("clob_book", token_id, book.model_dump(mode="json"), snapshot_ts)

        snapshot_report = build_daily_paper_report(registry, books_by_token)
        write_snapshot_outputs(out_dir=out_dir, registry=registry, report=snapshot_report)

        token_to_market: dict[str, str] = {}
        for event in registry.get("events", []):
            for market in event.get("markets", []):
                for token_id in (market.get("yes_token_id"), market.get("no_token_id")):
                    if token_id:
                        token_to_market[str(token_id)] = str(market.get("slug") or "")

        previous_books: dict[str, dict[str, object]] = {}
        delta_events: list[dict[str, object]] = []
        for poll_index in range(max(1, poll_count)):
            observed_ts = datetime.now(timezone.utc)
            for token_id in token_ids:
                try:
                    book = clob.get_book(token_id)
                except Exception as exc:
                    logger.warning(
                        "live book fetch failed",
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
                    store.save_raw_snapshot(
                        "clob_live_book",
                        f"{token_id}:poll:{poll_index}",
                        book.model_dump(mode="json"),
                        observed_ts,
                    )
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

            if poll_index < max(1, poll_count) - 1 and poll_interval_sec > 0:
                time.sleep(poll_interval_sec)

        live_delta_report = build_live_delta_report(registry=registry, delta_events=delta_events)
        write_live_delta_report(out_dir=out_dir, report=live_delta_report)

        watchlist_report = build_watchlist_report(
            registry=registry,
            snapshot_report=snapshot_report,
            live_delta_report=live_delta_report,
            delta_events=delta_events,
        )
        write_watchlist_report(out_dir=out_dir, report=watchlist_report)

        validation_report = build_watchlist_validation_report(
            registry=registry,
            live_delta_report=live_delta_report,
            watchlist_report=watchlist_report,
            delta_events=delta_events,
        )
        write_watchlist_validation_report(out_dir=out_dir, report=validation_report)
        summary = validation_report["summary"]
        return {
            "run_ts": snapshot_ts.isoformat(),
            "validation_forward_events": summary["validation_forward_events"],
            "markets_ranked": summary["markets_ranked"],
            "events_ranked": summary["events_ranked"],
            "top5_market_avg_forward_signal_score": summary["top5_market_avg_forward_signal_score"],
            "lower_market_avg_forward_signal_score": summary["lower_market_avg_forward_signal_score"],
            "top3_event_avg_forward_signal_score": summary["top3_event_avg_forward_signal_score"],
            "lower_event_avg_forward_signal_score": summary["lower_event_avg_forward_signal_score"],
            "market_signal_gap": round(
                float(summary["top5_market_avg_forward_signal_score"])
                - float(summary["lower_market_avg_forward_signal_score"]),
                6,
            ),
            "event_signal_gap": round(
                float(summary["top3_event_avg_forward_signal_score"])
                - float(summary["lower_event_avg_forward_signal_score"]),
                6,
            ),
        }
    finally:
        store.close()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    configure_logging(level=config.monitoring.log_level, json_logs=config.monitoring.json_logs)
    logger = get_logger("polyarb.intelligence.multi_run")
    out_dir = Path(args.out_dir)
    limit = args.market_limit or config.market_data.market_limit

    run_summaries: list[dict[str, object]] = []
    for run_index in range(max(1, int(args.run_count))):
        run_summaries.append(
            _run_single_cycle(
                config=config,
                out_dir=out_dir,
                limit=limit,
                poll_count=max(1, int(args.poll_count)),
                poll_interval_sec=float(args.poll_interval_sec),
                logger=logger,
            )
        )
        if run_index < max(1, int(args.run_count)) - 1 and float(args.run_interval_sec) > 0:
            time.sleep(float(args.run_interval_sec))

    report = build_multi_run_stability_report(run_summaries)
    written = write_multi_run_stability_report(out_dir=out_dir, report=report)
    print(
        json.dumps(
            {
                "run_count": report["summary"]["run_count"],
                "avg_market_signal_gap": report["summary"]["avg_market_signal_gap"],
                "avg_event_signal_gap": report["summary"]["avg_event_signal_gap"],
                "ranking_edge_stays_positive": report["summary"]["ranking_edge_stays_positive"],
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
