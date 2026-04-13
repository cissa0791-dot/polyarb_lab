from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from src.storage.event_store import ResearchStore

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def _normalize_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None


def _parse_event_ts(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    raw = str(value)
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw) / 1000.0, tz=timezone.utc)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def normalize_trade_event(message: dict[str, Any], token_to_market: dict[str, str]) -> dict[str, Any] | None:
    if str(message.get("event_type") or "") != "last_trade_price":
        return None
    token_id = str(message.get("asset_id") or "")
    return {
        "event_type": "last_trade_price",
        "token_id": token_id,
        "market_slug": token_to_market.get(token_id, ""),
        "market": message.get("market"),
        "price": _normalize_float(message.get("price")),
        "size": _normalize_float(message.get("size")),
        "side": str(message.get("side") or "").upper(),
        "fee_rate_bps": _normalize_float(message.get("fee_rate_bps")),
        "transaction_hash": message.get("transaction_hash"),
        "observed_ts": _parse_event_ts(message.get("timestamp")).isoformat(),
    }


def normalize_bbo_event(message: dict[str, Any], token_to_market: dict[str, str]) -> dict[str, Any] | None:
    if str(message.get("event_type") or "") != "best_bid_ask":
        return None
    token_id = str(message.get("asset_id") or "")
    best_bid = _normalize_float(message.get("best_bid"))
    best_ask = _normalize_float(message.get("best_ask"))
    spread = _normalize_float(message.get("spread"))
    if spread is None and best_bid is not None and best_ask is not None:
        spread = round(best_ask - best_bid, 6)
    return {
        "event_type": "best_bid_ask",
        "token_id": token_id,
        "market_slug": token_to_market.get(token_id, ""),
        "market": message.get("market"),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "observed_ts": _parse_event_ts(message.get("timestamp")).isoformat(),
    }


async def collect_market_trade_flow(
    *,
    token_ids: list[str],
    token_to_market: dict[str, str],
    duration_sec: float,
    store: ResearchStore | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trade_events: list[dict[str, Any]] = []
    bbo_events: list[dict[str, Any]] = []
    if not token_ids:
        return trade_events, bbo_events

    async with websockets.connect(MARKET_WS_URL, ping_interval=10, ping_timeout=10, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "assets_ids": token_ids,
                    "type": "market",
                    "custom_feature_enabled": True,
                }
            )
        )
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while time.monotonic() < deadline:
            try:
                raw_message = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
            except TimeoutError:
                break
            except asyncio.TimeoutError:
                break

            payload = json.loads(raw_message)
            messages = payload if isinstance(payload, list) else [payload]
            for message in messages:
                if not isinstance(message, dict):
                    continue
                trade = normalize_trade_event(message, token_to_market)
                if trade is not None:
                    trade_events.append(trade)
                    if store is not None:
                        store.save_raw_snapshot(
                            "clob_live_trade",
                            f"{trade['token_id']}:{trade['observed_ts']}:{len(trade_events)}",
                            trade,
                            _parse_event_ts(trade["observed_ts"]),
                        )
                    continue
                bbo = normalize_bbo_event(message, token_to_market)
                if bbo is not None:
                    bbo_events.append(bbo)
                    if store is not None:
                        store.save_raw_snapshot(
                            "clob_live_bbo",
                            f"{bbo['token_id']}:{bbo['observed_ts']}:{len(bbo_events)}",
                            bbo,
                            _parse_event_ts(bbo["observed_ts"]),
                        )
    return trade_events, bbo_events


def build_trade_flow_report(
    *,
    registry: dict[str, Any],
    trade_events: list[dict[str, Any]],
    bbo_events: list[dict[str, Any]],
    live_delta_events: list[dict[str, Any]],
) -> dict[str, Any]:
    event_slug_by_market: dict[str, str] = {}
    event_title_by_slug: dict[str, str | None] = {}
    for event in registry.get("events", []):
        event_slug = str(event.get("slug") or "")
        event_title_by_slug[event_slug] = event.get("title")
        for market in event.get("markets", []):
            market_slug = str(market.get("slug") or "")
            if market_slug:
                event_slug_by_market[market_slug] = event_slug

    trades_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bbo_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    delta_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trade_events:
        trades_by_market[str(row.get("market_slug") or "")].append(row)
    for row in bbo_events:
        bbo_by_market[str(row.get("market_slug") or "")].append(row)
    for row in live_delta_events:
        delta_by_market[str(row.get("market_slug") or "")].append(row)

    market_rows: list[dict[str, Any]] = []
    event_rollups: dict[str, dict[str, Any]] = {}
    total_spread_compressions = 0
    total_liquidity_refills = 0

    for market_slug, market_trades in trades_by_market.items():
        if not market_slug:
            continue
        ordered_trades = sorted(market_trades, key=lambda row: _parse_event_ts(row.get("observed_ts")))
        ordered_bbo = sorted(bbo_by_market.get(market_slug, []), key=lambda row: _parse_event_ts(row.get("observed_ts")))
        ordered_delta = sorted(delta_by_market.get(market_slug, []), key=lambda row: _parse_event_ts(row.get("observed_ts")))
        buy_size = sum(float(row.get("size") or 0.0) for row in ordered_trades if str(row.get("side") or "") == "BUY")
        sell_size = sum(float(row.get("size") or 0.0) for row in ordered_trades if str(row.get("side") or "") == "SELL")
        total_size = buy_size + sell_size
        side_imbalance = round(((buy_size - sell_size) / total_size), 6) if total_size > 0 else 0.0
        second_buckets = Counter(_parse_event_ts(row.get("observed_ts")).strftime("%Y-%m-%dT%H:%M:%S") for row in ordered_trades)
        burst_intensity = max(second_buckets.values()) if second_buckets else 0

        spread_compression_after_trades = 0
        liquidity_refill_after_trades = 0
        for trade in ordered_trades:
            trade_ts = _parse_event_ts(trade.get("observed_ts"))
            pre_bbo = None
            post_bbo = None
            for row in ordered_bbo:
                row_ts = _parse_event_ts(row.get("observed_ts"))
                if row_ts <= trade_ts:
                    pre_bbo = row
                elif post_bbo is None:
                    post_bbo = row
                    break
            pre_spread = None if pre_bbo is None else _normalize_float(pre_bbo.get("spread"))
            post_spread = None if post_bbo is None else _normalize_float(post_bbo.get("spread"))
            if pre_spread is not None and post_spread is not None and post_spread < pre_spread:
                spread_compression_after_trades += 1

            for row in ordered_delta:
                row_ts = _parse_event_ts(row.get("observed_ts"))
                if row_ts <= trade_ts:
                    continue
                previous = row.get("previous") or {}
                current = row.get("current") or {}
                bid_prev = _normalize_float(previous.get("best_bid_size"))
                ask_prev = _normalize_float(previous.get("best_ask_size"))
                bid_cur = _normalize_float(current.get("best_bid_size"))
                ask_cur = _normalize_float(current.get("best_ask_size"))
                if (
                    (bid_prev is not None and bid_cur is not None and bid_cur > bid_prev)
                    or (ask_prev is not None and ask_cur is not None and ask_cur > ask_prev)
                ):
                    liquidity_refill_after_trades += 1
                    break

        total_spread_compressions += spread_compression_after_trades
        total_liquidity_refills += liquidity_refill_after_trades
        trade_flow_score = round(
            len(ordered_trades)
            + abs(side_imbalance)
            + burst_intensity
            + spread_compression_after_trades
            + liquidity_refill_after_trades,
            6,
        )
        row = {
            "market_slug": market_slug,
            "event_slug": event_slug_by_market.get(market_slug, ""),
            "trade_count": len(ordered_trades),
            "buy_trade_size": round(buy_size, 6),
            "sell_trade_size": round(sell_size, 6),
            "side_imbalance": side_imbalance,
            "burst_intensity": burst_intensity,
            "spread_compression_after_trades": spread_compression_after_trades,
            "liquidity_refill_after_trades": liquidity_refill_after_trades,
            "trade_flow_score": trade_flow_score,
        }
        market_rows.append(row)

        event_slug = row["event_slug"]
        if event_slug:
            event_row = event_rollups.setdefault(
                event_slug,
                {
                    "event_slug": event_slug,
                    "event_title": event_title_by_slug.get(event_slug),
                    "market_count": 0,
                    "trade_count": 0,
                    "spread_compression_after_trades": 0,
                    "liquidity_refill_after_trades": 0,
                    "trade_flow_score": 0.0,
                },
            )
            event_row["market_count"] += 1
            event_row["trade_count"] += int(row["trade_count"])
            event_row["spread_compression_after_trades"] += int(row["spread_compression_after_trades"])
            event_row["liquidity_refill_after_trades"] += int(row["liquidity_refill_after_trades"])
            event_row["trade_flow_score"] = round(float(event_row["trade_flow_score"]) + float(row["trade_flow_score"]), 6)

    top_markets = sorted(market_rows, key=lambda row: (-float(row["trade_flow_score"]), str(row["market_slug"])))[:25]
    top_events = sorted(
        event_rollups.values(),
        key=lambda row: (-float(row["trade_flow_score"]), -int(row["trade_count"]), str(row["event_slug"])),
    )[:15]
    return {
        "report_type": "market_intelligence_trade_flow",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "summary": {
            "events_seen": registry.get("summary", {}).get("events_seen", 0),
            "markets_seen": registry.get("summary", {}).get("markets_seen", 0),
            "trade_events_seen": len(trade_events),
            "bbo_events_seen": len(bbo_events),
            "live_delta_events_seen": len(live_delta_events),
            "markets_with_trades": len(market_rows),
            "events_with_trades": len(event_rollups),
            "spread_compression_after_trades": total_spread_compressions,
            "liquidity_refill_after_trades": total_liquidity_refills,
        },
        "top_markets": top_markets,
        "top_events": top_events,
    }


def write_trade_flow_report(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"market_intelligence_trade_flow_{stamp}.json"
    latest_report_path = out_dir / "market_intelligence_trade_flow_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_report_path.write_text(payload, encoding="utf-8")
    return {
        "report_path": report_path,
        "latest_report_path": latest_report_path,
    }
