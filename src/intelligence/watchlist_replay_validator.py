from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _forward_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    activity = len(rows)
    spread = sum(1 for row in rows if row.get("spread_changed"))
    liquidity = sum(1 for row in rows if row.get("liquidity_changed"))
    incomplete = sum(1 for row in rows if row.get("completeness_changed"))
    return {
        "forward_activity_events": activity,
        "forward_spread_changes": spread,
        "forward_liquidity_changes": liquidity,
        "forward_incomplete_book_changes": incomplete,
        "forward_stability_score": activity - (2 * incomplete),
        "forward_signal_score": activity + (2 * spread) + liquidity - (2 * incomplete),
        "forward_active": bool(activity > 0),
    }


def build_watchlist_validation_report(
    *,
    registry: dict[str, Any],
    live_delta_report: dict[str, Any],
    watchlist_report: dict[str, Any],
    delta_events: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_events = sorted(delta_events, key=lambda row: _parse_ts(str(row.get("observed_ts") or "")))
    split_index = max(1, len(ordered_events) // 2) if ordered_events else 0
    forward_events = ordered_events[split_index:]

    event_slug_by_market: dict[str, str] = {}
    event_title_by_slug: dict[str, str | None] = {}
    for event in registry.get("events", []):
        event_slug = str(event.get("slug") or "")
        event_title_by_slug[event_slug] = event.get("title")
        for market in event.get("markets", []):
            market_slug = str(market.get("slug") or "")
            if market_slug:
                event_slug_by_market[market_slug] = event_slug

    forward_by_market: dict[str, list[dict[str, Any]]] = {}
    forward_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in forward_events:
        market_slug = str(row.get("market_slug") or "")
        if not market_slug:
            continue
        forward_by_market.setdefault(market_slug, []).append(row)
        event_slug = event_slug_by_market.get(market_slug, "")
        if event_slug:
            forward_by_event.setdefault(event_slug, []).append(row)

    market_validation_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(watchlist_report.get("top_markets", []), start=1):
        market_slug = str(row.get("market_slug") or "")
        metrics = _forward_metrics(forward_by_market.get(market_slug, []))
        market_validation_rows.append(
            {
                "rank": rank,
                "market_slug": market_slug,
                "event_slug": row.get("event_slug"),
                "watchlist_score": row.get("watchlist_score", 0),
                **metrics,
            }
        )

    event_validation_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(watchlist_report.get("top_events", []), start=1):
        event_slug = str(row.get("event_slug") or "")
        metrics = _forward_metrics(forward_by_event.get(event_slug, []))
        event_validation_rows.append(
            {
                "rank": rank,
                "event_slug": event_slug,
                "event_title": event_title_by_slug.get(event_slug),
                "watchlist_score": row.get("watchlist_score", 0),
                **metrics,
            }
        )

    top_market_slice = market_validation_rows[:5]
    lower_market_slice = market_validation_rows[5:]
    top_event_slice = event_validation_rows[:3]
    lower_event_slice = event_validation_rows[3:]

    def _avg(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 6)

    market_validation_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in market_validation_rows:
        event_slug = str(row.get("event_slug") or "")
        if event_slug:
            market_validation_by_event.setdefault(event_slug, []).append(row)

    event_first_validation_rows = []
    for row in event_validation_rows[:10]:
        event_first_validation_rows.append(
            {
                **row,
                "top_market_drilldown": market_validation_by_event.get(str(row.get("event_slug") or ""), [])[:5],
            }
        )

    return {
        "report_type": "market_intelligence_watchlist_validation",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "primary_ranking_level": "event",
        "summary": {
            "events_seen": registry.get("summary", {}).get("events_seen", 0),
            "markets_seen": registry.get("summary", {}).get("markets_seen", 0),
            "live_delta_events": len(ordered_events),
            "validation_forward_events": len(forward_events),
            "markets_ranked": len(market_validation_rows),
            "events_ranked": len(event_validation_rows),
            "ranked_markets_with_forward_activity": sum(1 for row in market_validation_rows if row["forward_active"]),
            "ranked_events_with_forward_activity": sum(1 for row in event_validation_rows if row["forward_active"]),
            "top5_market_avg_forward_signal_score": _avg(top_market_slice, "forward_signal_score"),
            "lower_market_avg_forward_signal_score": _avg(lower_market_slice, "forward_signal_score"),
            "top3_event_avg_forward_signal_score": _avg(top_event_slice, "forward_signal_score"),
            "lower_event_avg_forward_signal_score": _avg(lower_event_slice, "forward_signal_score"),
            "source_markets_updated": live_delta_report.get("summary", {}).get("markets_updated", 0),
        },
        "top_event_validation": event_first_validation_rows,
        "top_market_validation": market_validation_rows[:10],
    }


def write_watchlist_validation_report(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"market_intelligence_watchlist_validation_{stamp}.json"
    latest_report_path = out_dir / "market_intelligence_watchlist_validation_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_report_path.write_text(payload, encoding="utf-8")
    return {
        "report_path": report_path,
        "latest_report_path": latest_report_path,
    }
