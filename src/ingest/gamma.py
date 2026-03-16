from __future__ import annotations

import httpx
from typing import List


DEFAULT_MARKETS_PAGE_SIZE = 200


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
