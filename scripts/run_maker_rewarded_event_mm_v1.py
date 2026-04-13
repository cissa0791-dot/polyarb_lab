from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.intelligence.market_intelligence import build_event_market_registry, write_snapshot_outputs
from src.intelligence.watchlist_ranker import build_watchlist_report, load_live_delta_events, write_watchlist_report
from src.ingest.gamma import fetch_events, fetch_markets
from src.scanner.maker_rewarded_mm import (
    build_eligible_rewarded_market_groups,
    load_latest_event_first_inputs,
    write_maker_rewarded_mm_report,
)
from src.storage.event_store import ResearchStore
from src.strategies.opportunity_strategies import MakerRewardedEventMMStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-only maker rewarded event MM v1 simulation.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to runtime settings file.")
    parser.add_argument("--gamma-host", default="https://gamma-api.polymarket.com", help="Gamma API host.")
    parser.add_argument("--market-limit", type=int, default=100, help="Market/event fetch limit.")
    parser.add_argument("--out-dir", default="data/reports", help="Report output directory.")
    parser.add_argument("--top-events", type=int, default=10, help="Maximum event-first selections to consider.")
    parser.add_argument("--delta-limit", type=int, default=500, help="Maximum stored delta events for watchlist rebuild.")
    parser.add_argument("--min-stability-hits", type=int, default=1, help="Minimum stability score for eligible markets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.settings)
    out_dir = Path(args.out_dir)

    events = fetch_events(args.gamma_host, args.market_limit)
    markets = fetch_markets(args.gamma_host, args.market_limit)
    registry = build_event_market_registry(events, markets)
    snapshot_report = {"summary": {"books_fetched": 0}}
    write_snapshot_outputs(out_dir=out_dir, registry=registry, report=snapshot_report)

    store = ResearchStore(config.storage.sqlite_url)
    try:
        delta_events = load_live_delta_events(store, limit=args.delta_limit)
    finally:
        store.close()

    live_delta_report = json.loads((out_dir / "market_intelligence_live_delta_latest.json").read_text(encoding="utf-8"))
    watchlist_report = build_watchlist_report(
        registry=registry,
        snapshot_report=snapshot_report,
        live_delta_report=live_delta_report,
        delta_events=delta_events,
        trade_flow_report=None,
    )
    write_watchlist_report(out_dir=out_dir, report=watchlist_report)

    registry, watchlist_report = load_latest_event_first_inputs(out_dir)
    eligible_groups = build_eligible_rewarded_market_groups(
        registry=registry,
        watchlist_report=watchlist_report,
        max_events=args.top_events,
        min_stability_hits=args.min_stability_hits,
    )

    strategy = MakerRewardedEventMMStrategy()
    candidates = []
    reject_counter: Counter[str] = Counter()
    for event_group in eligible_groups:
        for market in event_group.get("markets", []):
            raw_candidate, audit = strategy.detect_with_audit(event_group, market)
            if raw_candidate is None:
                reject_counter[str((audit or {}).get("failure_reason") or "UNKNOWN")] += 1
                continue
            candidates.append(raw_candidate)

    report = {
        "report_type": "maker_rewarded_event_mm_v1",
        "paper_only": True,
        "maker_only": True,
        "summary": {
            "selected_events": min(args.top_events, len(watchlist_report.get("top_events", []))),
            "eligible_event_groups": len(eligible_groups),
            "eligible_markets": sum(len(group.get("markets", [])) for group in eligible_groups),
            "simulated_quote_plans": len(candidates),
            "top_reject_reasons": dict(reject_counter.most_common(10)),
        },
        "quote_plans": [
            {
                "candidate_id": candidate.candidate_id,
                "event_slug": candidate.metadata.get("event_slug"),
                "market_slug": candidate.market_slugs[0] if candidate.market_slugs else None,
                "quote_bid": candidate.metadata.get("quote_bid"),
                "quote_ask": candidate.metadata.get("quote_ask"),
                "quote_size": candidate.metadata.get("quote_size"),
                "spread_capture_ev": candidate.metadata.get("spread_capture_ev"),
                "liquidity_reward_ev": candidate.metadata.get("liquidity_reward_ev"),
                "adverse_selection_cost_proxy": candidate.metadata.get("adverse_selection_cost_proxy"),
                "inventory_cost_proxy": candidate.metadata.get("inventory_cost_proxy"),
                "cancel_replace_cost_proxy": candidate.metadata.get("cancel_replace_cost_proxy"),
                "total_ev": candidate.metadata.get("total_ev"),
            }
            for candidate in candidates[:25]
        ],
    }
    written = write_maker_rewarded_mm_report(out_dir=out_dir, report=report)
    print(
        json.dumps(
            {
                **report["summary"],
                "written_files": {key: str(path) for key, path in written.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
