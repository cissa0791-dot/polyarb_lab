from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Iterable, List

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BookParams

from src.core.models import BookLevel, OrderBook


def _normalize_levels(raw_levels: Iterable[Any], descending: bool) -> tuple[list[BookLevel], dict[str, Any]]:
    levels: list[BookLevel] = []
    raw_count = 0
    malformed_count = 0
    non_positive_count = 0
    previous_price: float | None = None
    raw_monotonic = True

    for raw_level in list(raw_levels or []):
        raw_count += 1
        try:
            price = float(getattr(raw_level, "price"))
            size = float(getattr(raw_level, "size"))
        except Exception:
            malformed_count += 1
            continue

        if not isfinite(price) or not isfinite(size):
            malformed_count += 1
            continue

        if previous_price is not None:
            if descending and price > previous_price + 1e-12:
                raw_monotonic = False
            if not descending and price < previous_price - 1e-12:
                raw_monotonic = False
        previous_price = price

        if price <= 0 or size <= 0:
            non_positive_count += 1
            continue

        levels.append(BookLevel(price=price, size=size))

    levels.sort(key=lambda level: level.price, reverse=descending)
    diagnostics = {
        "raw_count": raw_count,
        "normalized_count": len(levels),
        "malformed_count": malformed_count,
        "non_positive_count": non_positive_count,
        "raw_monotonic": raw_monotonic,
    }
    return levels, diagnostics


def _build_orderbook(token_id: str, raw_book: Any) -> OrderBook:
    bids, bid_stats = _normalize_levels(getattr(raw_book, "bids", []), descending=True)
    asks, ask_stats = _normalize_levels(getattr(raw_book, "asks", []), descending=False)
    metadata = {
        "requested_token_id": str(token_id),
        "response_asset_id": str(getattr(raw_book, "asset_id", token_id)),
        "raw_bids_count": bid_stats["raw_count"],
        "raw_asks_count": ask_stats["raw_count"],
        "normalized_bids_count": bid_stats["normalized_count"],
        "normalized_asks_count": ask_stats["normalized_count"],
        "malformed_bid_levels": bid_stats["malformed_count"],
        "malformed_ask_levels": ask_stats["malformed_count"],
        "non_positive_bid_levels": bid_stats["non_positive_count"],
        "non_positive_ask_levels": ask_stats["non_positive_count"],
        "raw_bids_monotonic": bid_stats["raw_monotonic"],
        "raw_asks_monotonic": ask_stats["raw_monotonic"],
    }
    return OrderBook(
        token_id=str(token_id),
        bids=bids,
        asks=asks,
        ts=datetime.now(timezone.utc),
        metadata=metadata,
    )


class ReadOnlyClob:
    def __init__(self, host: str):
        self.client = ClobClient(host)

    def get_book(self, token_id: str) -> OrderBook:
        raw = self.client.get_order_book(token_id)
        return _build_orderbook(token_id, raw)

    def get_books(self, token_ids: List[str]) -> List[OrderBook]:
        raw_books = self.client.get_order_books([BookParams(token_id=token_id) for token_id in token_ids])
        books: List[OrderBook] = []
        for raw in raw_books:
            token_id = str(getattr(raw, "asset_id", ""))
            books.append(_build_orderbook(token_id, raw))
        return books
