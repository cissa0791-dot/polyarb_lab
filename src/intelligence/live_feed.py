from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_decimal(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def summarize_book(book) -> dict[str, Any]:
    bids = list(getattr(book, "bids", []) or [])
    asks = list(getattr(book, "asks", []) or [])
    best_bid = _normalize_decimal(float(bids[0].price)) if bids else None
    best_ask = _normalize_decimal(float(asks[0].price)) if asks else None
    best_bid_size = _normalize_decimal(float(bids[0].size)) if bids else None
    best_ask_size = _normalize_decimal(float(asks[0].size)) if asks else None
    spread = _normalize_decimal(best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    return {
        "token_id": str(getattr(book, "token_id", "")),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
        "spread": spread,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "complete_top_of_book": bool(bids and asks),
    }


def compute_book_delta(
    *,
    token_id: str,
    market_slug: str,
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    observed_ts: datetime,
) -> dict[str, Any] | None:
    if previous is None:
        return {
            "event_type": "book_initialized",
            "token_id": token_id,
            "market_slug": market_slug,
            "observed_ts": observed_ts.isoformat(),
            "current": current,
        }

    changed_fields = {
        key: {
            "before": previous.get(key),
            "after": current.get(key),
        }
        for key in (
            "best_bid",
            "best_ask",
            "best_bid_size",
            "best_ask_size",
            "spread",
            "complete_top_of_book",
            "bid_levels",
            "ask_levels",
        )
        if previous.get(key) != current.get(key)
    }
    if not changed_fields:
        return None

    liquidity_changed = any(key in changed_fields for key in ("best_bid_size", "best_ask_size", "bid_levels", "ask_levels"))
    spread_changed = "spread" in changed_fields or "best_bid" in changed_fields or "best_ask" in changed_fields
    completeness_changed = "complete_top_of_book" in changed_fields

    return {
        "event_type": "book_delta",
        "token_id": token_id,
        "market_slug": market_slug,
        "observed_ts": observed_ts.isoformat(),
        "spread_changed": spread_changed,
        "liquidity_changed": liquidity_changed,
        "completeness_changed": completeness_changed,
        "changed_fields": changed_fields,
        "previous": previous,
        "current": current,
    }


def build_live_delta_report(
    *,
    registry: dict[str, Any],
    delta_events: list[dict[str, Any]],
) -> dict[str, Any]:
    updated_markets = {str(event.get("market_slug") or "") for event in delta_events if event.get("market_slug")}
    spread_changes = sum(1 for event in delta_events if event.get("spread_changed"))
    liquidity_changes = sum(1 for event in delta_events if event.get("liquidity_changed"))
    incomplete_changes = sum(1 for event in delta_events if event.get("completeness_changed"))
    event_counter: Counter[str] = Counter(event.get("event_type") or "unknown" for event in delta_events)

    by_market: dict[str, dict[str, Any]] = {}
    for event in delta_events:
        market_slug = str(event.get("market_slug") or "")
        if not market_slug:
            continue
        row = by_market.setdefault(
            market_slug,
            {
                "market_slug": market_slug,
                "delta_events": 0,
                "spread_changes": 0,
                "liquidity_changes": 0,
                "incomplete_book_changes": 0,
            },
        )
        row["delta_events"] += 1
        row["spread_changes"] += int(bool(event.get("spread_changed")))
        row["liquidity_changes"] += int(bool(event.get("liquidity_changed")))
        row["incomplete_book_changes"] += int(bool(event.get("completeness_changed")))

    top_markets = sorted(
        by_market.values(),
        key=lambda row: (-int(row["delta_events"]), str(row["market_slug"])),
    )[:10]

    return {
        "report_type": "market_intelligence_live_delta",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "summary": {
            "events_seen": registry.get("summary", {}).get("events_seen", 0),
            "markets_seen": registry.get("summary", {}).get("markets_seen", 0),
            "tracked_tokens": registry.get("summary", {}).get("tracked_tokens", 0),
            "delta_events": len(delta_events),
            "markets_updated": len(updated_markets),
            "spread_changes": spread_changes,
            "liquidity_changes": liquidity_changes,
            "incomplete_or_missing_book_changes": incomplete_changes,
            "delta_event_types": dict(event_counter),
        },
        "top_markets_by_activity": top_markets,
    }


def write_live_delta_report(
    *,
    out_dir: Path,
    report: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"market_intelligence_live_delta_{stamp}.json"
    latest_report_path = out_dir / "market_intelligence_live_delta_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_report_path.write_text(payload, encoding="utf-8")
    return {
        "report_path": report_path,
        "latest_report_path": latest_report_path,
    }
