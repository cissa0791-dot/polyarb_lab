from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ReplayBookSnapshot:
    ts: str
    market_slug: str
    token_id: str
    best_bid: float
    best_ask: float
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


@dataclass
class ReplayConfig:
    order_size: float = 5.0
    quote_bid_offset: float = 0.0
    quote_ask_offset: float = 0.0
    latency_ms: int = 0
    max_inventory_shares: float = 50.0
    starting_cash_usdc: float = 100.0
    min_spread: float = 0.001


@dataclass
class ReplayOrder:
    side: str
    price: float
    size: float
    created_ts: str
    active_ts: str
    filled_size: float = 0.0

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.size - self.filled_size)


@dataclass
class ReplayFill:
    ts: str
    market_slug: str
    side: str
    price: float
    size: float
    cash_delta: float
    inventory_after: float


@dataclass
class ReplayResult:
    market_slug: str
    token_id: str
    snapshot_count: int
    fill_count: int
    buy_fill_shares: float
    sell_fill_shares: float
    ending_inventory_shares: float
    ending_cash_usdc: float
    realized_pnl_usdc: float
    unrealized_pnl_usdc: float
    net_pnl_usdc: float
    max_inventory_shares: float
    fills: list[ReplayFill]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fills"] = [asdict(fill) for fill in self.fills]
        return payload


def replay_from_saved_books(path: str, config: ReplayConfig | None = None) -> dict[str, Any]:
    snapshots = load_saved_books(path)
    result = replay_quote_strategy(snapshots, config or ReplayConfig())
    return result.to_dict()


def load_saved_books(path: str) -> list[ReplayBookSnapshot]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)
    if source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("snapshots") if isinstance(payload, dict) else payload
    return parse_book_snapshots(rows or [])


def parse_book_snapshots(rows: Iterable[dict[str, Any]]) -> list[ReplayBookSnapshot]:
    snapshots = []
    for row in rows:
        best_bid = _as_float(row.get("best_bid"))
        best_ask = _as_float(row.get("best_ask"))
        if best_bid is None or best_ask is None or best_bid <= 0.0 or best_ask <= 0.0 or best_ask < best_bid:
            continue
        snapshots.append(
            ReplayBookSnapshot(
                ts=str(row.get("ts") or row.get("timestamp") or row.get("time") or ""),
                market_slug=str(row.get("market_slug") or row.get("market") or ""),
                token_id=str(row.get("token_id") or row.get("asset_id") or ""),
                best_bid=round(best_bid, 6),
                best_ask=round(best_ask, 6),
                bids=_parse_levels(row.get("bids")),
                asks=_parse_levels(row.get("asks")),
            )
        )
    snapshots.sort(key=lambda item: _parse_ts(item.ts))
    return snapshots


def replay_quote_strategy(snapshots: list[ReplayBookSnapshot], config: ReplayConfig | None = None) -> ReplayResult:
    cfg = config or ReplayConfig()
    if not snapshots:
        return ReplayResult(
            market_slug="",
            token_id="",
            snapshot_count=0,
            fill_count=0,
            buy_fill_shares=0.0,
            sell_fill_shares=0.0,
            ending_inventory_shares=0.0,
            ending_cash_usdc=round(cfg.starting_cash_usdc, 6),
            realized_pnl_usdc=0.0,
            unrealized_pnl_usdc=0.0,
            net_pnl_usdc=0.0,
            max_inventory_shares=0.0,
            fills=[],
        )

    cash = float(cfg.starting_cash_usdc)
    inventory = 0.0
    avg_cost = 0.0
    realized = 0.0
    max_inventory = 0.0
    fills: list[ReplayFill] = []
    open_bid: ReplayOrder | None = None
    open_ask: ReplayOrder | None = None

    for snapshot in snapshots:
        now = _parse_ts(snapshot.ts)
        open_bid, open_ask, fill_rows, cash, inventory, avg_cost, realized = _apply_fills(
            snapshot,
            now,
            open_bid,
            open_ask,
            cash,
            inventory,
            avg_cost,
            realized,
        )
        fills.extend(fill_rows)
        max_inventory = max(max_inventory, inventory)

        spread = snapshot.best_ask - snapshot.best_bid
        if spread < cfg.min_spread:
            continue

        if open_bid is None and inventory + cfg.order_size <= cfg.max_inventory_shares + 1e-9:
            bid_price = min(snapshot.best_bid + cfg.quote_bid_offset, snapshot.best_ask - 0.000001)
            notional = bid_price * cfg.order_size
            if cash >= notional:
                open_bid = ReplayOrder(
                    side="BUY",
                    price=round(bid_price, 6),
                    size=round(cfg.order_size, 6),
                    created_ts=snapshot.ts,
                    active_ts=_shift_ts(snapshot.ts, cfg.latency_ms),
                )

        sell_size = min(cfg.order_size, inventory)
        if open_ask is None and sell_size > 1e-9:
            ask_price = max(snapshot.best_ask - cfg.quote_ask_offset, snapshot.best_bid + 0.000001)
            open_ask = ReplayOrder(
                side="SELL",
                price=round(ask_price, 6),
                size=round(sell_size, 6),
                created_ts=snapshot.ts,
                active_ts=_shift_ts(snapshot.ts, cfg.latency_ms),
            )

    last = snapshots[-1]
    unrealized = inventory * (last.midpoint - avg_cost) if inventory > 0.0 else 0.0
    net = cash + inventory * last.midpoint - cfg.starting_cash_usdc
    return ReplayResult(
        market_slug=last.market_slug,
        token_id=last.token_id,
        snapshot_count=len(snapshots),
        fill_count=len(fills),
        buy_fill_shares=round(sum(fill.size for fill in fills if fill.side == "BUY"), 6),
        sell_fill_shares=round(sum(fill.size for fill in fills if fill.side == "SELL"), 6),
        ending_inventory_shares=round(inventory, 6),
        ending_cash_usdc=round(cash, 6),
        realized_pnl_usdc=round(realized, 6),
        unrealized_pnl_usdc=round(unrealized, 6),
        net_pnl_usdc=round(net, 6),
        max_inventory_shares=round(max_inventory, 6),
        fills=fills,
    )


def _apply_fills(
    snapshot: ReplayBookSnapshot,
    now: datetime,
    open_bid: ReplayOrder | None,
    open_ask: ReplayOrder | None,
    cash: float,
    inventory: float,
    avg_cost: float,
    realized: float,
) -> tuple[ReplayOrder | None, ReplayOrder | None, list[ReplayFill], float, float, float, float]:
    fills: list[ReplayFill] = []
    if open_bid is not None and now >= _parse_ts(open_bid.active_ts) and snapshot.best_ask <= open_bid.price:
        size = min(open_bid.remaining_size, _top_size(snapshot.asks, open_bid.remaining_size))
        if size > 1e-9:
            cost = size * open_bid.price
            total_cost = inventory * avg_cost + cost
            inventory += size
            cash -= cost
            avg_cost = total_cost / inventory if inventory > 0.0 else 0.0
            open_bid.filled_size += size
            fills.append(
                ReplayFill(
                    ts=snapshot.ts,
                    market_slug=snapshot.market_slug,
                    side="BUY",
                    price=round(open_bid.price, 6),
                    size=round(size, 6),
                    cash_delta=round(-cost, 6),
                    inventory_after=round(inventory, 6),
                )
            )
        if open_bid.remaining_size <= 1e-9:
            open_bid = None

    if open_ask is not None and now >= _parse_ts(open_ask.active_ts) and snapshot.best_bid >= open_ask.price:
        size = min(open_ask.remaining_size, inventory, _top_size(snapshot.bids, open_ask.remaining_size))
        if size > 1e-9:
            proceeds = size * open_ask.price
            realized += size * (open_ask.price - avg_cost)
            inventory -= size
            cash += proceeds
            if inventory <= 1e-9:
                inventory = 0.0
                avg_cost = 0.0
            open_ask.filled_size += size
            fills.append(
                ReplayFill(
                    ts=snapshot.ts,
                    market_slug=snapshot.market_slug,
                    side="SELL",
                    price=round(open_ask.price, 6),
                    size=round(size, 6),
                    cash_delta=round(proceeds, 6),
                    inventory_after=round(inventory, 6),
                )
            )
        if open_ask.remaining_size <= 1e-9 or inventory <= 1e-9:
            open_ask = None

    return open_bid, open_ask, fills, cash, inventory, avg_cost, realized


def _parse_levels(value: Any) -> list[tuple[float, float]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    levels = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                price = _as_float(item.get("price"))
                size = _as_float(item.get("size"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price = _as_float(item[0])
                size = _as_float(item[1])
            else:
                continue
            if price is not None and size is not None and price > 0.0 and size > 0.0:
                levels.append((round(price, 6), round(size, 6)))
    return levels


def _top_size(levels: list[tuple[float, float]], fallback: float) -> float:
    if not levels:
        return fallback
    return max(0.0, float(levels[0][1]))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _shift_ts(value: str, latency_ms: int) -> str:
    base = _parse_ts(value)
    return (base + timedelta(milliseconds=max(0, int(latency_ms)))).isoformat()
