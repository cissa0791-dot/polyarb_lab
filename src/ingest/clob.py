from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BookParams

from src.core.models import BookLevel, OrderBook

logger = logging.getLogger(__name__)


def _normalize_scalar_payload(
    payload: Any,
    requested_token_ids: list[str],
    *,
    value_keys: tuple[str, ...],
) -> dict[str, float]:
    results: dict[str, float] = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            payload = data
        elif all(token_id in payload for token_id in requested_token_ids):
            for token_id in requested_token_ids:
                try:
                    results[token_id] = round(float(payload[token_id]), 6)
                except Exception:
                    continue
            return results

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            token_id = str(
                row.get("token_id")
                or row.get("asset_id")
                or row.get("market")
                or ""
            )
            if not token_id:
                continue
            for value_key in value_keys:
                if value_key not in row:
                    continue
                try:
                    results[token_id] = round(float(row[value_key]), 6)
                except Exception:
                    pass
                break
    return results


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
        self.host = host
        self.client = ClobClient(host)
        self.no_orderbook_negative_cache_ttl_sec = 300.0
        self.invalid_token_retry_interval_sec = 900.0
        self._negative_cache: dict[str, dict[str, Any]] = {}
        self._request_stats: Counter[str] = Counter()

    def _ensure_runtime_state(self) -> None:
        if not hasattr(self, "no_orderbook_negative_cache_ttl_sec"):
            self.no_orderbook_negative_cache_ttl_sec = 300.0
        if not hasattr(self, "invalid_token_retry_interval_sec"):
            self.invalid_token_retry_interval_sec = 900.0
        if not hasattr(self, "_negative_cache"):
            self._negative_cache = {}
        if not hasattr(self, "_request_stats"):
            self._request_stats = Counter()

    def configure_negative_cache(self, *, no_orderbook_ttl_sec: float, invalid_token_retry_interval_sec: float) -> None:
        self._ensure_runtime_state()
        self.no_orderbook_negative_cache_ttl_sec = max(0.0, float(no_orderbook_ttl_sec))
        self.invalid_token_retry_interval_sec = max(0.0, float(invalid_token_retry_interval_sec))

    def reset_request_stats(self) -> None:
        self._ensure_runtime_state()
        self._request_stats = Counter()
        self._purge_expired_negative_cache()

    def request_stats_snapshot(self) -> dict[str, int]:
        self._ensure_runtime_state()
        self._purge_expired_negative_cache()
        return {
            "negative_cache_active_count": len(self._negative_cache),
            "negative_cache_hits": int(self._request_stats.get("negative_cache_hits", 0)),
            "negative_cache_expired_rechecks": int(self._request_stats.get("negative_cache_expired_rechecks", 0)),
        }

    def _purge_expired_negative_cache(self) -> None:
        self._ensure_runtime_state()
        now = datetime.now(timezone.utc)
        expired = [token_id for token_id, entry in self._negative_cache.items() if entry["expires_at"] <= now]
        for token_id in expired:
            self._negative_cache.pop(token_id, None)

    def _negative_cache_reason(self, exc: Exception) -> str | None:
        text = str(exc)
        if "No orderbook exists for the requested token id" in text:
            return "no_orderbook"
        if "Invalid token id" in text:
            return "invalid_token"
        lowered = text.lower()
        if "404" in text and ("orderbook" in lowered or "/book" in lowered or "token_id=" in lowered):
            return "no_orderbook"
        if "400" in text and ("token id" in lowered or "token_id" in lowered):
            return "invalid_token"
        return None

    def _negative_cache_ttl(self, reason: str) -> float:
        if reason == "invalid_token":
            return float(self.invalid_token_retry_interval_sec)
        return float(self.no_orderbook_negative_cache_ttl_sec)

    def _mark_negative_cache(self, token_id: str, reason: str) -> None:
        self._ensure_runtime_state()
        ttl_sec = self._negative_cache_ttl(reason)
        if ttl_sec <= 0.0:
            return
        self._negative_cache[str(token_id)] = {
            "reason": reason,
            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_sec),
        }

    def _negative_cache_lookup(self, token_id: str) -> str | None:
        self._ensure_runtime_state()
        entry = self._negative_cache.get(str(token_id))
        if entry is None:
            return None
        if entry["expires_at"] <= datetime.now(timezone.utc):
            self._negative_cache.pop(str(token_id), None)
            self._request_stats["negative_cache_expired_rechecks"] += 1
            return None
        self._request_stats["negative_cache_hits"] += 1
        return str(entry["reason"])

    def negative_cache_reason(self, token_id: str) -> str | None:
        self._ensure_runtime_state()
        entry = self._negative_cache.get(str(token_id))
        if entry is None:
            return None
        if entry["expires_at"] <= datetime.now(timezone.utc):
            return None
        return str(entry["reason"])

    def _eligible_token_ids(self, token_ids: List[str]) -> list[str]:
        self._ensure_runtime_state()
        eligible: list[str] = []
        for token_id in list(dict.fromkeys(token_ids)):
            if self._negative_cache_lookup(token_id) is None:
                eligible.append(token_id)
        return eligible

    def get_book(self, token_id: str) -> OrderBook:
        self._ensure_runtime_state()
        cached_reason = self._negative_cache_lookup(token_id)
        if cached_reason is not None:
            if cached_reason == "invalid_token":
                raise LookupError("Invalid token id (negative cache)")
            raise LookupError("No orderbook exists for the requested token id (negative cache)")
        try:
            raw = self.client.get_order_book(token_id)
        except Exception as exc:
            reason = self._negative_cache_reason(exc)
            if reason is not None:
                self._mark_negative_cache(token_id, reason)
            raise
        return _build_orderbook(token_id, raw)

    def get_books(self, token_ids: List[str]) -> List[OrderBook]:
        self._ensure_runtime_state()
        eligible_ids = self._eligible_token_ids(token_ids)
        if not eligible_ids:
            return []
        raw_books = self.client.get_order_books([BookParams(token_id=token_id) for token_id in eligible_ids])
        books: List[OrderBook] = []
        for raw in raw_books:
            token_id = str(getattr(raw, "asset_id", ""))
            books.append(_build_orderbook(token_id, raw))
        return books

    def prefetch_books(
        self,
        token_ids: List[str],
        max_workers: int = 8,
    ) -> Dict[str, OrderBook]:
        """Fetch multiple books concurrently using a thread pool.

        Returns a dict mapping token_id -> OrderBook for successfully fetched
        books. Tokens that fail are silently omitted from the result (callers
        should treat a missing key as a fetch failure).
        """
        if not token_ids:
            return {}

        unique_ids = self._eligible_token_ids(token_ids)
        results: Dict[str, OrderBook] = {}

        def _fetch_one(tid: str) -> Optional[tuple[str, OrderBook]]:
            try:
                book = self.get_book(tid)
                return tid, book
            except Exception as exc:
                logger.debug("prefetch_books: failed for token %s: %s", tid, exc)
                return None

        with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_ids))) as pool:
            futures = {pool.submit(_fetch_one, tid): tid for tid in unique_ids}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    tid, book = result
                    results[tid] = book

        return results

    def prefetch_books_batch(self, token_ids: List[str]) -> Dict[str, OrderBook]:
        """Fetch multiple books in a single batch API call.

        More network-efficient than prefetch_books but less isolated on error
        (a single bad response can affect the whole batch).
        """
        if not token_ids:
            return {}

        unique_ids = self._eligible_token_ids(token_ids)
        if not unique_ids:
            return {}
        try:
            books = self.get_books(unique_ids)
        except Exception as exc:
            logger.warning("prefetch_books_batch: batch fetch failed: %s", exc)
            return {}

        results: Dict[str, OrderBook] = {}
        for book in books:
            if book.token_id:
                results[book.token_id] = book
        return results

    def fetch_books_batch(self, token_ids: List[str], chunk_size: int = 100) -> Dict[str, OrderBook]:
        if not token_ids:
            return {}

        unique_ids = self._eligible_token_ids(token_ids)
        if not unique_ids:
            return {}
        results: Dict[str, OrderBook] = {}
        for start in range(0, len(unique_ids), max(1, chunk_size)):
            chunk = unique_ids[start: start + max(1, chunk_size)]
            chunk_books = self.prefetch_books_batch(chunk)
            if len(chunk_books) < len(chunk):
                missing = [token_id for token_id in chunk if token_id not in chunk_books]
                if missing:
                    chunk_books.update(self.prefetch_books(missing))
            results.update(chunk_books)
        return results

    def fetch_midpoints_batch(self, token_ids: List[str]) -> Dict[str, float]:
        unique_ids = self._eligible_token_ids(token_ids)
        if not unique_ids:
            return {}
        try:
            payload = self.client.get_midpoints([BookParams(token_id=token_id) for token_id in unique_ids])
        except Exception as exc:
            logger.warning("fetch_midpoints_batch failed: %s", exc)
            return {}
        return _normalize_scalar_payload(
            payload,
            unique_ids,
            value_keys=("midpoint", "mid", "price"),
        )

    def fetch_spreads_batch(self, token_ids: List[str]) -> Dict[str, float]:
        unique_ids = self._eligible_token_ids(token_ids)
        if not unique_ids:
            return {}
        try:
            payload = self.client.get_spreads([BookParams(token_id=token_id) for token_id in unique_ids])
        except Exception as exc:
            logger.warning("fetch_spreads_batch failed: %s", exc)
            return {}
        return _normalize_scalar_payload(
            payload,
            unique_ids,
            value_keys=("spread", "value"),
        )

    def fetch_prices_batch(self, token_ids: List[str], side: str = "BUY") -> Dict[str, float]:
        unique_ids = self._eligible_token_ids(token_ids)
        if not unique_ids:
            return {}
        try:
            payload = self.client.get_prices([BookParams(token_id=token_id, side=side) for token_id in unique_ids])
        except Exception as exc:
            logger.warning("fetch_prices_batch failed: %s", exc)
            return {}
        return _normalize_scalar_payload(
            payload,
            unique_ids,
            value_keys=("price", "value", "market_price"),
        )

    def fetch_simplified_markets(self, limit: int = 1000) -> list[dict]:
        if limit <= 0:
            return []
        try:
            response = httpx.get(f"{self.host.rstrip('/')}/simplified-markets", timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("fetch_simplified_markets failed: %s", exc)
            return []

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return [row for row in data[:limit] if isinstance(row, dict)]
        return []
