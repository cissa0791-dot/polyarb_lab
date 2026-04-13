from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _event_id(event: dict[str, Any]) -> str | None:
    raw = event.get("id")
    return str(raw) if raw is not None else None


def _extract_yes_no_tokens(market: dict[str, Any]) -> dict[str, str] | None:
    outcomes = _parse_json_list(market.get("outcomes"))
    token_ids = _parse_json_list(market.get("clobTokenIds"))
    if len(outcomes) != 2 or len(token_ids) != 2:
        return None

    out: dict[str, str] = {}
    for outcome, token_id in zip(outcomes, token_ids):
        key = str(outcome).strip().upper()
        out[key] = str(token_id)
    if "YES" not in out or "NO" not in out:
        return None
    return {"yes_token_id": out["YES"], "no_token_id": out["NO"]}


def build_event_market_registry(events: list[dict[str, Any]], markets: list[dict[str, Any]]) -> dict[str, Any]:
    markets_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for market in markets:
        attached_events = list(market.get("events") or [])
        if not attached_events:
            markets_by_event["unlinked"].append(market)
            continue
        for event_ref in attached_events:
            event_id = event_ref.get("id")
            key = str(event_id) if event_id is not None else "unlinked"
            markets_by_event[key].append(market)

    registry_events: list[dict[str, Any]] = []
    binary_market_count = 0
    orderbook_enabled_count = 0
    token_count = 0
    neg_risk_event_count = 0
    for event in events:
        event_key = _event_id(event)
        linked_markets = list(markets_by_event.get(event_key or "", []))
        normalized_markets: list[dict[str, Any]] = []
        if bool(event.get("negRisk")) or bool(event.get("enableNegRisk")):
            neg_risk_event_count += 1

        for market in linked_markets:
            tokens = _extract_yes_no_tokens(market)
            is_binary = tokens is not None
            if is_binary:
                binary_market_count += 1
                token_count += 2
            if bool(market.get("enableOrderBook")):
                orderbook_enabled_count += 1
            normalized_markets.append(
                {
                    "market_id": str(market.get("id") or ""),
                    "slug": str(market.get("slug") or ""),
                    "question": market.get("question"),
                    "active": bool(market.get("active")),
                    "closed": bool(market.get("closed")),
                    "enable_orderbook": bool(market.get("enableOrderBook")),
                    "best_bid": market.get("bestBid"),
                    "best_ask": market.get("bestAsk"),
                    "volume_num": market.get("volumeNum"),
                    "liquidity_num": market.get("liquidityNum"),
                    "neg_risk": bool(market.get("negRisk")),
                    "neg_risk_other": bool(market.get("negRiskOther")),
                    "group_item_title": market.get("groupItemTitle"),
                    "fees_enabled": bool(market.get("feesEnabled")),
                    "rewards_min_size": market.get("rewardsMinSize"),
                    "rewards_max_spread": market.get("rewardsMaxSpread"),
                    "uma_reward": market.get("umaReward"),
                    "clob_rewards": list(market.get("clobRewards") or []),
                    "is_binary_yes_no": is_binary,
                    "yes_token_id": tokens["yes_token_id"] if tokens else None,
                    "no_token_id": tokens["no_token_id"] if tokens else None,
                }
            )

        registry_events.append(
            {
                "event_id": event_key or "",
                "slug": str(event.get("slug") or ""),
                "title": event.get("title"),
                "active": bool(event.get("active")),
                "closed": bool(event.get("closed")),
                "neg_risk": bool(event.get("negRisk")),
                "enable_neg_risk": bool(event.get("enableNegRisk")),
                "neg_risk_augmented": bool(event.get("negRiskAugmented")),
                "market_count": len(normalized_markets),
                "markets": normalized_markets,
            }
        )

    registry = {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "events_seen": len(events),
            "markets_seen": len(markets),
            "events_with_markets": sum(1 for event in registry_events if event["market_count"] > 0),
            "binary_markets": binary_market_count,
            "orderbook_enabled_markets": orderbook_enabled_count,
            "tracked_tokens": token_count,
            "neg_risk_events": neg_risk_event_count,
        },
        "events": registry_events,
    }
    return registry


def collect_registry_token_ids(registry: dict[str, Any]) -> list[str]:
    ordered: dict[str, None] = {}
    for event in registry.get("events", []):
        for market in event.get("markets", []):
            for token_id in (market.get("yes_token_id"), market.get("no_token_id")):
                if token_id:
                    ordered[str(token_id)] = None
    return list(ordered.keys())


def build_daily_paper_report(registry: dict[str, Any], books_by_token: dict[str, Any]) -> dict[str, Any]:
    missing_books = 0
    top_of_book_complete = 0
    spread_counter: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []

    for event in registry.get("events", []):
        event_top_books = 0
        for market in event.get("markets", []):
            token_id = market.get("yes_token_id")
            if not token_id:
                continue
            book = books_by_token.get(str(token_id))
            if book is None:
                missing_books += 1
                continue
            bids = list(getattr(book, "bids", []) or [])
            asks = list(getattr(book, "asks", []) or [])
            if bids and asks:
                top_of_book_complete += 1
                event_top_books += 1
                spread = float(asks[0].price) - float(bids[0].price)
                if spread <= 0.01:
                    spread_counter["<=0.01"] += 1
                elif spread <= 0.03:
                    spread_counter["<=0.03"] += 1
                else:
                    spread_counter[">0.03"] += 1
            else:
                missing_books += 1

        if event_top_books > 0:
            event_rows.append(
                {
                    "event_slug": event.get("slug"),
                    "event_title": event.get("title"),
                    "tracked_markets": event.get("market_count", 0),
                    "markets_with_top_of_book": event_top_books,
                }
            )

    event_rows.sort(key=lambda row: (-int(row["markets_with_top_of_book"]), str(row["event_slug"])))
    return {
        "report_type": "market_intelligence_snapshot",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "summary": {
            **registry.get("summary", {}),
            "books_fetched": len(books_by_token),
            "markets_with_complete_top_of_book": top_of_book_complete,
            "missing_or_incomplete_books": missing_books,
            "yes_spread_buckets": dict(spread_counter),
        },
        "top_events_by_book_coverage": event_rows[:10],
    }


def write_snapshot_outputs(
    *,
    out_dir: Path,
    registry: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registry_path = out_dir / f"market_intelligence_registry_{stamp}.json"
    report_path = out_dir / f"market_intelligence_report_{stamp}.json"
    latest_registry_path = out_dir / "market_intelligence_registry_latest.json"
    latest_report_path = out_dir / "market_intelligence_report_latest.json"

    registry_json = json.dumps(registry, indent=2, ensure_ascii=False)
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    registry_path.write_text(registry_json, encoding="utf-8")
    report_path.write_text(report_json, encoding="utf-8")
    latest_registry_path.write_text(registry_json, encoding="utf-8")
    latest_report_path.write_text(report_json, encoding="utf-8")
    return {
        "registry_path": registry_path,
        "report_path": report_path,
        "latest_registry_path": latest_registry_path,
        "latest_report_path": latest_report_path,
    }
