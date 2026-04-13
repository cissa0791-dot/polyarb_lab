"""
neg_risk_structure_research_line — Module 1: Discovery
polyarb_lab / research_line / active

Discovers live neg-risk events from the Polymarket Gamma API.

Neg-risk events are multi-outcome markets where:
  - The neg-risk adapter contract is active (negRisk=true in Gamma)
  - Each market in the event represents one possible outcome
  - Exactly one outcome resolves YES at settlement

This module is read-only. No order submission. No mainline imports.

Official references:
  - https://docs.polymarket.com (neg-risk section)
  - https://github.com/Polymarket/neg-risk-ctf-adapter
  - https://github.com/Polymarket/py-clob-client
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

GAMMA_HOST = "https://gamma-api.polymarket.com"
DEFAULT_PAGE_SIZE = 100
MAX_EVENTS = 2000           # safety ceiling — do not paginate beyond this
MIN_OUTCOMES = 2            # minimum outcome count to consider


# ---------------------------------------------------------------------------
# Raw API types (unvalidated dicts from Gamma)
# ---------------------------------------------------------------------------

RawEvent = dict[str, Any]
RawMarket = dict[str, Any]


# ---------------------------------------------------------------------------
# Discovery result
# ---------------------------------------------------------------------------

class NegRiskEventRaw:
    """Container for a single raw neg-risk event and its associated markets."""

    __slots__ = (
        "event_id",
        "slug",
        "title",
        "end_date_str",
        "neg_risk_flag",        # True if Gamma explicitly set negRisk=true
        "markets",              # list of raw market dicts belonging to this event
        "fetched_at",
    )

    def __init__(
        self,
        event_id: str,
        slug: str,
        title: str,
        end_date_str: Optional[str],
        neg_risk_flag: bool,
        markets: list[RawMarket],
        fetched_at: datetime,
    ) -> None:
        self.event_id = event_id
        self.slug = slug
        self.title = title
        self.end_date_str = end_date_str
        self.neg_risk_flag = neg_risk_flag
        self.markets = markets
        self.fetched_at = fetched_at

    def __repr__(self) -> str:
        return (
            f"NegRiskEventRaw(id={self.event_id!r}, slug={self.slug!r}, "
            f"outcomes={len(self.markets)}, neg_risk={self.neg_risk_flag})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bool_field(event: RawEvent, *keys: str) -> bool:
    """Return True if any of the given keys on event is truthy."""
    for key in keys:
        val = event.get(key)
        if val is True or val == "true" or val == 1:
            return True
    return False


def _is_neg_risk_event(event: RawEvent) -> bool:
    """
    Determine if a Gamma event is a neg-risk event.

    Strategy (layered, most reliable first):
    1. Event-level negRisk flag (Gamma >= 2024 API: 'negRisk', 'neg_risk')
    2. Any child market has negRisk flag set
    3. Any child market has a non-null negRiskRequestId or negRiskOther fields
    """
    # Layer 1: event-level flag
    if _bool_field(event, "negRisk", "neg_risk", "negRiskEnabled"):
        return True

    # Layer 2 & 3: inspect child markets
    markets = event.get("markets") or []
    for mkt in markets:
        if _bool_field(mkt, "negRisk", "neg_risk", "negRiskEnabled"):
            return True
        if mkt.get("negRiskRequestId") or mkt.get("negRiskOther"):
            return True

    return False


def _is_active_event(event: RawEvent) -> bool:
    """True if the event appears to be open and active."""
    closed = event.get("closed")
    active = event.get("active")
    # Gamma uses various field naming conventions
    if closed is True or closed == "true":
        return False
    if active is False or active == "false":
        return False
    return True


def _extract_markets(event: RawEvent) -> list[RawMarket]:
    """Extract and return the list of markets from an event dict."""
    markets = event.get("markets") or []
    return [m for m in markets if isinstance(m, dict)]


def _outcome_count(markets: list[RawMarket]) -> int:
    return len(markets)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _fetch_events_page(
    host: str,
    offset: int,
    limit: int,
    neg_risk_filter: bool = True,
    client: Optional[httpx.Client] = None,
) -> list[RawEvent]:
    """Fetch one page of events from Gamma API."""
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "closed": "false",
        "active": "true",
    }
    if neg_risk_filter:
        # Attempt to use server-side neg_risk filter if supported
        params["neg_risk"] = "true"

    url = f"{host.rstrip('/')}/events"
    try:
        if client is not None:
            resp = client.get(url, params=params, timeout=20)
        else:
            resp = httpx.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []
    except Exception as exc:
        logger.warning("Gamma events page fetch failed (offset=%d): %s", offset, exc)
        return []


def _fetch_all_events(
    host: str,
    neg_risk_filter: bool = True,
    max_events: int = MAX_EVENTS,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[RawEvent]:
    """
    Paginate through Gamma API events.

    If neg_risk_filter=True, request neg_risk=true from server.
    Regardless, always apply local _is_neg_risk_event() filter as a safety pass.
    """
    all_events: list[RawEvent] = []
    offset = 0

    with httpx.Client() as client:
        while offset < max_events:
            page = _fetch_events_page(host, offset, page_size, neg_risk_filter, client)
            if not page:
                break
            all_events.extend(page)
            if len(page) < page_size:
                # Last page
                break
            offset += page_size

    logger.info("Fetched %d raw events from Gamma (neg_risk_filter=%s)", len(all_events), neg_risk_filter)
    return all_events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_neg_risk_events(
    gamma_host: str = GAMMA_HOST,
    min_outcomes: int = MIN_OUTCOMES,
    require_end_date: bool = False,
    max_events: int = MAX_EVENTS,
) -> list[NegRiskEventRaw]:
    """
    Discover all active neg-risk events from Gamma API.

    Process:
    1. Fetch all active events (with server-side neg_risk=true filter if supported).
    2. Apply local _is_neg_risk_event() filter (catches events where server filter silently fails).
    3. Filter by minimum outcome count.
    4. Return list of NegRiskEventRaw objects.

    Args:
        gamma_host:       Gamma API base URL.
        min_outcomes:     Minimum number of outcome markets per event.
        require_end_date: If True, skip events with no parseable end_date.
        max_events:       Safety ceiling on total events fetched.

    Returns:
        List of NegRiskEventRaw, empty list on total failure.
    """
    fetched_at = datetime.now(timezone.utc)

    # Primary fetch: server-side neg_risk filter
    raw_events = _fetch_all_events(gamma_host, neg_risk_filter=True, max_events=max_events)

    # If server-side filter returned 0, retry without it (filter may not be supported)
    if not raw_events:
        logger.warning(
            "Server-side neg_risk filter returned 0 events. "
            "Retrying without filter and applying local classification."
        )
        raw_events = _fetch_all_events(gamma_host, neg_risk_filter=False, max_events=max_events)

    results: list[NegRiskEventRaw] = []
    skipped_not_neg_risk = 0
    skipped_too_few_outcomes = 0
    skipped_no_end_date = 0
    skipped_not_active = 0

    for event in raw_events:
        if not isinstance(event, dict):
            continue

        # Must be active
        if not _is_active_event(event):
            skipped_not_active += 1
            continue

        # Must be a neg-risk event (local filter)
        if not _is_neg_risk_event(event):
            skipped_not_neg_risk += 1
            continue

        markets = _extract_markets(event)
        if _outcome_count(markets) < min_outcomes:
            skipped_too_few_outcomes += 1
            continue

        end_date_str = event.get("endDate") or event.get("end_date") or event.get("endDateIso")
        if require_end_date and not end_date_str:
            skipped_no_end_date += 1
            continue

        event_id = str(event.get("id") or event.get("event_id") or "")
        slug = str(event.get("slug") or "")
        title = str(event.get("title") or event.get("question") or slug)
        neg_risk_flag = _bool_field(event, "negRisk", "neg_risk", "negRiskEnabled")

        results.append(NegRiskEventRaw(
            event_id=event_id,
            slug=slug,
            title=title,
            end_date_str=end_date_str,
            neg_risk_flag=neg_risk_flag,
            markets=markets,
            fetched_at=fetched_at,
        ))

    logger.info(
        "Discovery complete: %d neg-risk events found | "
        "skipped: not_active=%d, not_neg_risk=%d, too_few_outcomes=%d, no_end_date=%d",
        len(results),
        skipped_not_active,
        skipped_not_neg_risk,
        skipped_too_few_outcomes,
        skipped_no_end_date,
    )
    return results


def discovery_summary(events: list[NegRiskEventRaw]) -> dict[str, Any]:
    """Return a concise summary dict for logging."""
    if not events:
        return {"total": 0}

    outcome_counts = [len(e.markets) for e in events]
    neg_risk_flagged = sum(1 for e in events if e.neg_risk_flag)

    return {
        "total": len(events),
        "neg_risk_flagged": neg_risk_flagged,
        "locally_classified_only": len(events) - neg_risk_flagged,
        "outcome_count_distribution": {
            "min": min(outcome_counts),
            "max": max(outcome_counts),
            "mean": round(sum(outcome_counts) / len(outcome_counts), 1),
        },
        "events_n2": sum(1 for c in outcome_counts if c == 2),
        "events_n3_to_5": sum(1 for c in outcome_counts if 3 <= c <= 5),
        "events_n6_plus": sum(1 for c in outcome_counts if c >= 6),
        "fetched_at": events[0].fetched_at.isoformat() if events else None,
    }
