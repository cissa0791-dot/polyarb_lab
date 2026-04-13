"""
reward_aware_maker_survivor_persistence_queue_line
polyarb_lab / research_line / audit-only

Per-cycle state tracker for the executable-positive survivors.

For each cycle and each target slug:
  - Checks exec-positive status (still passes all 4 gates?)
  - Fetches a detailed book snapshot (targeted: 1 CLOB /book call per slug)
  - Computes book-realism and queue-pressure flags

Targeted book fetch (4 calls per cycle max) is SEPARATE from the bulk
discovery book fetch (500 calls). This adds ≤4 extra API calls per cycle
only for the surviving slugs — not a broad universe scan.

No order submission. No state. No mainline imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .discovery import RawRewardedMarket, _safe_float, CLOB_BOOK_PATH
from .ev_model import MarketEVResult
from .executable_audit import ExecutableAuditResult, EXEC_POSITIVE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Realism-check thresholds
# ---------------------------------------------------------------------------

# Volume adequacy: daily volume >= min_size × this multiple
VOLUME_ADEQUACY_MULTIPLE = 20.0

# Depth adequacy: top-of-book bid size (shares) >= min_size × this multiple
# Measures whether there is enough taker interest to fill our maker quote
DEPTH_ADEQUACY_MULTIPLE = 2.0

# Queue pressure: if bid levels within the reward spread ≥ this, spread is contested
QUEUE_PRESSURE_LEVEL_THRESHOLD = 3

TICK_SIZE = 0.01  # standard Polymarket tick


# ---------------------------------------------------------------------------
# Targeted book snapshot (per-survivor, not bulk)
# ---------------------------------------------------------------------------

@dataclass
class BookSnapshot:
    """Detailed book snapshot for one token at one point in time."""
    token_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    top_bid_size: Optional[float]       # shares available at best bid
    top_ask_size: Optional[float]       # shares available at best ask
    total_bid_depth_shares: float       # sum of all bid sizes
    total_ask_depth_shares: float       # sum of all ask sizes
    bid_levels_count: int               # total bid price levels
    ask_levels_count: int               # total ask price levels
    bid_levels_in_reward_spread: int    # bid levels within reward_max_spread of best_ask
    ask_levels_in_reward_spread: int    # ask levels within reward_max_spread of best_bid
    fetch_ok: bool


def fetch_book_snapshot(
    host: str,
    token_id: str,
    reward_max_spread: float,
    client: httpx.Client,
) -> BookSnapshot:
    """
    Fetch a detailed book snapshot for one token_id.

    Parses full bid/ask arrays — not just best price.
    reward_max_spread: the spread window within which to count competing levels.
    """
    if not token_id:
        return BookSnapshot(
            token_id=token_id, best_bid=None, best_ask=None,
            top_bid_size=None, top_ask_size=None,
            total_bid_depth_shares=0.0, total_ask_depth_shares=0.0,
            bid_levels_count=0, ask_levels_count=0,
            bid_levels_in_reward_spread=0, ask_levels_in_reward_spread=0,
            fetch_ok=False,
        )
    url = f"{host.rstrip('/')}{CLOB_BOOK_PATH}"
    try:
        resp = client.get(url, params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []

        best_bid: Optional[float] = None
        best_ask: Optional[float] = None
        top_bid_size: Optional[float] = None
        top_ask_size: Optional[float] = None
        total_bid = 0.0
        total_ask = 0.0
        bid_in_spread = 0
        ask_in_spread = 0

        # Bids: sorted descending (best bid first)
        for i, level in enumerate(bids):
            if not isinstance(level, dict):
                continue
            p = _safe_float(level.get("price"), default=None)  # type: ignore[arg-type]
            s = _safe_float(level.get("size"), default=0.0)
            if p is None:
                continue
            if i == 0:
                best_bid = p
                top_bid_size = s
            total_bid += s

        # Asks: sorted ascending (best ask first)
        for i, level in enumerate(asks):
            if not isinstance(level, dict):
                continue
            p = _safe_float(level.get("price"), default=None)  # type: ignore[arg-type]
            s = _safe_float(level.get("size"), default=0.0)
            if p is None:
                continue
            if i == 0:
                best_ask = p
                top_ask_size = s
            total_ask += s

        # Count levels within reward spread window
        # "Within reward spread of best_ask" for bids: bid ≥ best_ask - reward_max_spread
        if best_ask is not None:
            bid_floor = best_ask - reward_max_spread
            for level in bids:
                if not isinstance(level, dict):
                    continue
                p = _safe_float(level.get("price"), default=None)  # type: ignore[arg-type]
                if p is not None and p >= bid_floor:
                    bid_in_spread += 1

        # "Within reward spread of best_bid" for asks: ask ≤ best_bid + reward_max_spread
        if best_bid is not None:
            ask_ceil = best_bid + reward_max_spread
            for level in asks:
                if not isinstance(level, dict):
                    continue
                p = _safe_float(level.get("price"), default=None)  # type: ignore[arg-type]
                if p is not None and p <= ask_ceil:
                    ask_in_spread += 1

        return BookSnapshot(
            token_id=token_id,
            best_bid=best_bid,
            best_ask=best_ask,
            top_bid_size=top_bid_size,
            top_ask_size=top_ask_size,
            total_bid_depth_shares=round(total_bid, 4),
            total_ask_depth_shares=round(total_ask, 4),
            bid_levels_count=len(bids),
            ask_levels_count=len(asks),
            bid_levels_in_reward_spread=bid_in_spread,
            ask_levels_in_reward_spread=ask_in_spread,
            fetch_ok=True,
        )
    except Exception as exc:
        logger.debug("Detailed book fetch failed token=%s: %s", token_id, exc)
        return BookSnapshot(
            token_id=token_id, best_bid=None, best_ask=None,
            top_bid_size=None, top_ask_size=None,
            total_bid_depth_shares=0.0, total_ask_depth_shares=0.0,
            bid_levels_count=0, ask_levels_count=0,
            bid_levels_in_reward_spread=0, ask_levels_in_reward_spread=0,
            fetch_ok=False,
        )


# ---------------------------------------------------------------------------
# Per-cycle slug state
# ---------------------------------------------------------------------------

@dataclass
class CycleSlugState:
    """Full state for one target slug in one audit cycle."""
    cycle: int
    slug: str

    # Presence
    found_in_universe: bool
    full_slug: str                         # actual slug from universe (may differ in casing)

    # Executable audit
    is_exec_positive: bool
    raw_ev: Optional[float]
    reward_rate_daily_usdc: Optional[float]
    modeled_quote_spread: Optional[float]   # EV-layer quote width, not observed live market spread
    midpoint: Optional[float]
    quote_capital_usd: Optional[float]
    ev_roc: Optional[float]
    rejection_codes: list[str] = field(default_factory=list)

    # Book realism (from discovery)
    has_usable_book: bool = False
    liquidity: Optional[float] = None
    volume_24hr: Optional[float] = None
    volume_ok: bool = False               # volume_24hr >= min_size × VOLUME_ADEQUACY_MULTIPLE

    # Book snapshot (targeted fetch)
    book: Optional[BookSnapshot] = None
    depth_ok: bool = False                # top_bid_size >= min_size × DEPTH_ADEQUACY_MULTIPLE
    queue_pressure: bool = False          # bid_levels_in_spread >= QUEUE_PRESSURE_LEVEL_THRESHOLD
    spread_ticks: Optional[int] = None    # modeled_quote_spread / tick_size

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "slug": self.slug,
            "full_slug": self.full_slug,
            "found_in_universe": self.found_in_universe,
            "is_exec_positive": self.is_exec_positive,
            "raw_ev": self.raw_ev,
            "reward_rate_daily_usdc": self.reward_rate_daily_usdc,
            "modeled_quote_spread": self.modeled_quote_spread,
            "midpoint": self.midpoint,
            "quote_capital_usd": self.quote_capital_usd,
            "ev_roc": round(self.ev_roc, 6) if self.ev_roc is not None else None,
            "rejection_codes": self.rejection_codes,
            "has_usable_book": self.has_usable_book,
            "liquidity": self.liquidity,
            "volume_24hr": self.volume_24hr,
            "volume_ok": self.volume_ok,
            "depth_ok": self.depth_ok,
            "queue_pressure": self.queue_pressure,
            "spread_ticks": self.spread_ticks,
            "book": {
                "best_bid": self.book.best_bid,
                "best_ask": self.book.best_ask,
                "top_bid_size": self.book.top_bid_size,
                "top_ask_size": self.book.top_ask_size,
                "total_bid_depth_shares": self.book.total_bid_depth_shares,
                "total_ask_depth_shares": self.book.total_ask_depth_shares,
                "bid_levels_count": self.book.bid_levels_count,
                "ask_levels_count": self.book.ask_levels_count,
                "bid_levels_in_reward_spread": self.book.bid_levels_in_reward_spread,
                "ask_levels_in_reward_spread": self.book.ask_levels_in_reward_spread,
            } if self.book else None,
        }


def _find_market(
    target_slug: str,
    raw_markets: list[RawRewardedMarket],
) -> Optional[RawRewardedMarket]:
    """
    Find a market by slug with prefix matching.

    The display in run_executable_audit.py truncates slugs to 48 chars.
    Target slugs from the user may be truncated. Match on startswith in both
    directions (target starts with market slug, or market slug starts with target).
    """
    # Exact match first
    for m in raw_markets:
        if m.market_slug == target_slug:
            return m
    # Prefix match: market slug starts with target (target is truncated display)
    for m in raw_markets:
        if m.market_slug.startswith(target_slug) or target_slug.startswith(m.market_slug):
            return m
    return None


def build_cycle_state(
    cycle: int,
    target_slug: str,
    raw_markets: list[RawRewardedMarket],
    ev_results: list[MarketEVResult],
    audit_results: list[ExecutableAuditResult],
    clob_host: str,
    http_client: httpx.Client,
) -> CycleSlugState:
    """Build one CycleSlugState for a target slug in a given cycle."""
    ev_by_slug  = {r.market_slug: r for r in ev_results}
    audit_by_slug = {r.market_slug: r for r in audit_results}

    market = _find_market(target_slug, raw_markets)

    if market is None:
        return CycleSlugState(
            cycle=cycle,
            slug=target_slug,
            found_in_universe=False,
            full_slug=target_slug,
            is_exec_positive=False,
            raw_ev=None,
            reward_rate_daily_usdc=None,
            modeled_quote_spread=None,
            midpoint=None,
            quote_capital_usd=None,
            ev_roc=None,
            rejection_codes=["NOT_IN_UNIVERSE"],
        )

    full_slug = market.market_slug
    ev_r    = ev_by_slug.get(full_slug)
    audit_r = audit_by_slug.get(full_slug)

    # Volume check (from discovery data — no extra API call)
    volume_ok = (
        market.volume_24hr is not None
        and market.volume_24hr >= market.rewards_min_size * VOLUME_ADEQUACY_MULTIPLE
    )

    # Quote capital proxy
    quote_capital: Optional[float] = None
    if ev_r and ev_r.midpoint and ev_r.midpoint > 0:
        quote_capital = market.rewards_min_size * ev_r.midpoint

    # Spread in ticks
    spread_ticks = None
    if ev_r and ev_r.quoted_spread is not None:
        spread_ticks = max(1, round(ev_r.quoted_spread / TICK_SIZE))

    # Targeted book snapshot — 1 CLOB /book call for this slug only
    reward_max_spread = market.rewards_max_spread_cents / 100.0
    book = fetch_book_snapshot(
        host=clob_host,
        token_id=market.yes_token_id,
        reward_max_spread=reward_max_spread,
        client=http_client,
    )

    # Depth OK: top-of-book bid size >= min_size × DEPTH_ADEQUACY_MULTIPLE
    depth_ok = (
        book.fetch_ok
        and book.top_bid_size is not None
        and book.top_bid_size >= market.rewards_min_size * DEPTH_ADEQUACY_MULTIPLE
    )

    # Queue pressure: many existing levels competing within the reward spread
    queue_pressure = book.bid_levels_in_reward_spread >= QUEUE_PRESSURE_LEVEL_THRESHOLD

    return CycleSlugState(
        cycle=cycle,
        slug=target_slug,
        found_in_universe=True,
        full_slug=full_slug,
        is_exec_positive=audit_r is not None and audit_r.executable_verdict == EXEC_POSITIVE,
        raw_ev=ev_r.reward_adjusted_raw_ev if ev_r else None,
        reward_rate_daily_usdc=market.reward_daily_rate_usdc,
        modeled_quote_spread=ev_r.quoted_spread if ev_r else None,
        midpoint=ev_r.midpoint if ev_r else None,
        quote_capital_usd=round(quote_capital, 4) if quote_capital else None,
        ev_roc=audit_r.ev_roc if audit_r else None,
        rejection_codes=audit_r.rejection_codes if audit_r else [],
        has_usable_book=market.has_usable_book(),
        liquidity=market.liquidity,
        volume_24hr=market.volume_24hr,
        volume_ok=volume_ok,
        book=book,
        depth_ok=depth_ok,
        queue_pressure=queue_pressure,
        spread_ticks=spread_ticks,
    )
