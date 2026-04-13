from __future__ import annotations

from datetime import datetime, timedelta, timezone
import httpx
from typing import Any, List


DEFAULT_MARKETS_PAGE_SIZE = 200
DEFAULT_EVENTS_PAGE_SIZE = 100


def _market_identity(market: dict) -> str | None:
    market_id = market.get("id")
    if market_id is not None:
        return f"id:{market_id}"
    slug = market.get("slug")
    if slug:
        return f"slug:{slug}"
    return None


def _fetch_markets_page(host: str, limit: int, offset: int) -> list[dict]:
    url = f"{host.rstrip('/')}/markets"
    params = {
        "limit": limit,
        "offset": offset,
        "closed": "false",
        "active": "true",
    }
    response = httpx.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def _fetch_events_page(host: str, limit: int, offset: int, extra_params: dict[str, Any] | None = None) -> list[dict]:
    url = f"{host.rstrip('/')}/events"
    params = {
        "limit": limit,
        "offset": offset,
        "closed": "false",
        "active": "true",
    }
    if extra_params:
        params.update(extra_params)
    response = httpx.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def _market_liquidity_num(market: dict) -> float | None:
    for field_name in ("liquidityNum", "liquidity", "liquidityClob"):
        value = market.get(field_name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _market_end_datetime(market: dict) -> datetime | None:
    value = market.get("endDate") or market.get("endDateIso")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _market_matches_slice(
    market: dict,
    market_slice: dict,
    *,
    now: datetime,
) -> bool:
    name = str(market_slice.get("name") or "")
    if name not in {"near_term_liquidity_core", "liquidity_core"}:
        raise ValueError(f"Unsupported market slice '{name}'")

    min_liquidity_num = float(market_slice.get("min_liquidity_num", 2500))
    liquidity_num = _market_liquidity_num(market)
    if liquidity_num is None:
        return False
    if name == "liquidity_core":
        return liquidity_num >= min_liquidity_num

    max_days_to_end = int(market_slice.get("max_days_to_end", 90))
    end_dt = _market_end_datetime(market)
    if end_dt is None:
        return False
    return end_dt <= (now + timedelta(days=max_days_to_end)) and liquidity_num >= min_liquidity_num


def fetch_markets(host: str, limit: int = 200) -> List[dict]:
    if limit <= 0:
        return []

    page_size = DEFAULT_MARKETS_PAGE_SIZE
    offset = 0
    ordered_unique_markets: dict[str, dict] = {}

    while len(ordered_unique_markets) < limit:
        page = _fetch_markets_page(host, page_size, offset)
        if not page:
            break

        new_items = 0
        for market in page:
            identity = _market_identity(market)
            if identity is None or identity in ordered_unique_markets:
                continue
            ordered_unique_markets[identity] = market
            new_items += 1
            if len(ordered_unique_markets) >= limit:
                break

        if len(page) < page_size or new_items == 0:
            break
        offset += page_size

    return list(ordered_unique_markets.values())[:limit]


def fetch_markets_with_slice(
    host: str,
    limit: int = 200,
    market_slice: dict | None = None,
    *,
    now: datetime | None = None,
) -> List[dict]:
    if not market_slice:
        return fetch_markets(host, limit)
    if limit <= 0:
        return []

    page_size = DEFAULT_MARKETS_PAGE_SIZE
    offset = 0
    current_time = now or datetime.now(timezone.utc)
    seen_market_ids: set[str] = set()
    accepted_markets: list[dict] = []

    while len(accepted_markets) < limit:
        page = _fetch_markets_page(host, page_size, offset)
        if not page:
            break

        new_items = 0
        for market in page:
            identity = _market_identity(market)
            if identity is None or identity in seen_market_ids:
                continue
            seen_market_ids.add(identity)
            new_items += 1
            if _market_matches_slice(market, market_slice, now=current_time):
                accepted_markets.append(market)
                if len(accepted_markets) >= limit:
                    break

        if len(page) < page_size or new_items == 0:
            break
        offset += page_size

    return accepted_markets[:limit]


def fetch_events(host: str, limit: int = 100) -> List[dict]:
    if limit <= 0:
        return []

    page_size = DEFAULT_EVENTS_PAGE_SIZE
    offset = 0
    ordered_unique_events: dict[str, dict] = {}

    while len(ordered_unique_events) < limit:
        page = _fetch_events_page(host, page_size, offset)
        if not page:
            break

        new_items = 0
        for event in page:
            event_id = event.get("id")
            slug = event.get("slug")
            identity = f"id:{event_id}" if event_id is not None else (f"slug:{slug}" if slug else None)
            if identity is None or identity in ordered_unique_events:
                continue
            ordered_unique_events[identity] = event
            new_items += 1
            if len(ordered_unique_events) >= limit:
                break

        if len(page) < page_size or new_items == 0:
            break
        offset += page_size

    return list(ordered_unique_events.values())[:limit]


def flatten_event_markets(events: List[dict], limit: int | None = None) -> List[dict]:
    ordered_unique_markets: dict[str, dict] = {}
    max_markets = None if limit is None or limit <= 0 else int(limit)

    for event in events:
        markets = event.get("markets") or []
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            identity = _market_identity(market)
            if identity is None or identity in ordered_unique_markets:
                continue
            merged = dict(market)
            merged.setdefault("eventSlug", event.get("slug"))
            merged.setdefault("eventTitle", event.get("title"))
            merged.setdefault("eventId", event.get("id"))
            ordered_unique_markets[identity] = merged
            if max_markets is not None and len(ordered_unique_markets) >= max_markets:
                return list(ordered_unique_markets.values())[:max_markets]
    return list(ordered_unique_markets.values())[:max_markets]


def fetch_markets_from_events(host: str, limit: int = 200) -> List[dict]:
    if limit <= 0:
        return []

    page_size = DEFAULT_EVENTS_PAGE_SIZE
    offset = 0
    ordered_unique_markets: dict[str, dict] = {}

    while len(ordered_unique_markets) < limit:
        page = _fetch_events_page(
            host,
            page_size,
            offset,
            extra_params={"order": "volume_24hr", "ascending": "false"},
        )
        if not page:
            break

        new_items = 0
        for market in flatten_event_markets(page):
            identity = _market_identity(market)
            if identity is None or identity in ordered_unique_markets:
                continue
            ordered_unique_markets[identity] = market
            new_items += 1
            if len(ordered_unique_markets) >= limit:
                break

        if len(page) < page_size or new_items == 0:
            break
        offset += page_size

    return list(ordered_unique_markets.values())[:limit]
