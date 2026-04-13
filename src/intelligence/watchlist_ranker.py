from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select

from src.storage.event_store import ResearchStore


def load_latest_intelligence_inputs(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registry = json.loads((out_dir / "market_intelligence_registry_latest.json").read_text(encoding="utf-8"))
    snapshot_report = json.loads((out_dir / "market_intelligence_report_latest.json").read_text(encoding="utf-8"))
    live_delta_report = json.loads((out_dir / "market_intelligence_live_delta_latest.json").read_text(encoding="utf-8"))
    return registry, snapshot_report, live_delta_report


def load_latest_trade_flow_report(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "market_intelligence_trade_flow_latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_live_delta_events(store: ResearchStore, limit: int | None = None) -> list[dict[str, Any]]:
    stmt = select(store.raw_snapshots.c.payload_json).where(store.raw_snapshots.c.source == "clob_live_delta")
    stmt = stmt.order_by(desc(store.raw_snapshots.c.ingest_ts), desc(store.raw_snapshots.c.id))
    if limit is not None:
        stmt = stmt.limit(limit)
    with store.engine.begin() as connection:
        rows = connection.execute(stmt).all()
    events = [json.loads(row[0]) for row in rows]
    events.reverse()
    return events


def build_watchlist_report(
    *,
    registry: dict[str, Any],
    snapshot_report: dict[str, Any],
    live_delta_report: dict[str, Any],
    delta_events: list[dict[str, Any]],
    trade_flow_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    markets_by_slug: dict[str, dict[str, Any]] = {}
    events_by_slug: dict[str, dict[str, Any]] = {}
    for event in registry.get("events", []):
        event_slug = str(event.get("slug") or "")
        events_by_slug[event_slug] = event
        for market in event.get("markets", []):
            market_slug = str(market.get("slug") or "")
            if market_slug:
                markets_by_slug[market_slug] = {
                    "event_slug": event_slug,
                    "event_title": event.get("title"),
                    "question": market.get("question"),
                    "best_bid": market.get("best_bid"),
                    "best_ask": market.get("best_ask"),
                    "liquidity_num": market.get("liquidity_num"),
                    "volume_num": market.get("volume_num"),
                }

    market_rows: dict[str, dict[str, Any]] = {}
    event_rollups: dict[str, dict[str, Any]] = {}
    trade_flow_by_market: dict[str, dict[str, Any]] = {}
    trade_flow_by_event: dict[str, dict[str, Any]] = {}
    if trade_flow_report:
        for row in trade_flow_report.get("top_markets", []):
            market_slug = str(row.get("market_slug") or "")
            if market_slug:
                trade_flow_by_market[market_slug] = row
        for row in trade_flow_report.get("top_events", []):
            event_slug = str(row.get("event_slug") or "")
            if event_slug:
                trade_flow_by_event[event_slug] = row

    for delta in delta_events:
        market_slug = str(delta.get("market_slug") or "")
        if not market_slug:
            continue
        market_meta = markets_by_slug.get(market_slug, {})
        row = market_rows.setdefault(
            market_slug,
            {
                "market_slug": market_slug,
                "event_slug": market_meta.get("event_slug") or "",
                "event_title": market_meta.get("event_title"),
                "question": market_meta.get("question"),
                "activity_events": 0,
                "spread_changes": 0,
                "liquidity_changes": 0,
                "incomplete_book_changes": 0,
                "stability_hits": 0,
            },
        )
        row["activity_events"] += 1
        row["spread_changes"] += int(bool(delta.get("spread_changed")))
        row["liquidity_changes"] += int(bool(delta.get("liquidity_changed")))
        row["incomplete_book_changes"] += int(bool(delta.get("completeness_changed")))
        if not bool(delta.get("completeness_changed")):
            row["stability_hits"] += 1

    for row in market_rows.values():
        trade_flow_market = trade_flow_by_market.get(str(row.get("market_slug") or ""), {})
        row["trade_count"] = int(trade_flow_market.get("trade_count", 0) or 0)
        row["side_imbalance"] = float(trade_flow_market.get("side_imbalance", 0.0) or 0.0)
        row["burst_intensity"] = int(trade_flow_market.get("burst_intensity", 0) or 0)
        row["spread_compression_after_trades"] = int(
            trade_flow_market.get("spread_compression_after_trades", 0) or 0
        )
        row["liquidity_refill_after_trades"] = int(trade_flow_market.get("liquidity_refill_after_trades", 0) or 0)
        row["trade_flow_score"] = round(
            float(trade_flow_market.get("trade_flow_score", 0.0) or 0.0),
            6,
        )
        row["watchlist_score"] = round(
            int(row["activity_events"])
            + int(row["spread_changes"]) * 2
            + int(row["liquidity_changes"])
            + int(row["incomplete_book_changes"]) * 2
            + float(row["trade_flow_score"]),
            6,
        )
        row["stability_score"] = int(row["stability_hits"]) - int(row["incomplete_book_changes"]) * 2
        event_slug = str(row.get("event_slug") or "")
        if not event_slug:
            continue
        event_row = event_rollups.setdefault(
            event_slug,
            {
                "event_slug": event_slug,
                "event_title": row.get("event_title"),
                "markets_ranked": 0,
                "activity_events": 0,
                "spread_changes": 0,
                "liquidity_changes": 0,
                "incomplete_book_changes": 0,
                "watchlist_score": 0,
                "trade_count": 0,
                "trade_flow_score": 0.0,
            },
        )
        event_row["markets_ranked"] += 1
        event_row["activity_events"] += int(row["activity_events"])
        event_row["spread_changes"] += int(row["spread_changes"])
        event_row["liquidity_changes"] += int(row["liquidity_changes"])
        event_row["incomplete_book_changes"] += int(row["incomplete_book_changes"])
        event_row["trade_count"] += int(row["trade_count"])
        event_row["trade_flow_score"] = round(float(event_row["trade_flow_score"]) + float(row["trade_flow_score"]), 6)
        event_row["watchlist_score"] = round(float(event_row["watchlist_score"]) + float(row["watchlist_score"]), 6)

    for event_row in event_rollups.values():
        trade_flow_event = trade_flow_by_event.get(str(event_row.get("event_slug") or ""), {})
        if trade_flow_event:
            event_row["trade_flow_score"] = round(
                max(float(event_row["trade_flow_score"]), float(trade_flow_event.get("trade_flow_score", 0.0) or 0.0)),
                6,
            )

    top_markets = sorted(
        market_rows.values(),
        key=lambda row: (
            -float(row["watchlist_score"]),
            -int(row["spread_changes"]),
            -int(row["liquidity_changes"]),
            str(row["market_slug"]),
        ),
    )[:25]
    top_events = sorted(
        event_rollups.values(),
        key=lambda row: (-float(row["watchlist_score"]), -int(row["markets_ranked"]), str(row["event_slug"])),
    )[:15]

    markets_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in top_markets:
        event_slug = str(row.get("event_slug") or "")
        if event_slug:
            markets_by_event[event_slug].append(row)

    event_watchlist = []
    for event_row in top_events:
        event_watchlist.append(
            {
                **event_row,
                "top_markets": markets_by_event.get(str(event_row.get("event_slug") or ""), [])[:5],
            }
        )

    return {
        "report_type": "market_intelligence_watchlist",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "primary_ranking_level": "event",
        "summary": {
            "events_seen": registry.get("summary", {}).get("events_seen", 0),
            "markets_seen": registry.get("summary", {}).get("markets_seen", 0),
            "tracked_tokens": registry.get("summary", {}).get("tracked_tokens", 0),
            "snapshot_books_fetched": snapshot_report.get("summary", {}).get("books_fetched", 0),
            "live_delta_events": len(delta_events),
            "markets_ranked": len(market_rows),
            "events_ranked": len(event_rollups),
            "markets_updated": live_delta_report.get("summary", {}).get("markets_updated", 0),
            "spread_changes": live_delta_report.get("summary", {}).get("spread_changes", 0),
            "liquidity_changes": live_delta_report.get("summary", {}).get("liquidity_changes", 0),
            "incomplete_or_missing_book_changes": live_delta_report.get("summary", {}).get(
                "incomplete_or_missing_book_changes", 0
            ),
            "trade_flow_markets": len(trade_flow_by_market),
        },
        "top_events": event_watchlist,
        "top_markets": top_markets,
    }


def write_watchlist_report(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"market_intelligence_watchlist_{stamp}.json"
    latest_report_path = out_dir / "market_intelligence_watchlist_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_report_path.write_text(payload, encoding="utf-8")
    return {
        "report_path": report_path,
        "latest_report_path": latest_report_path,
    }
