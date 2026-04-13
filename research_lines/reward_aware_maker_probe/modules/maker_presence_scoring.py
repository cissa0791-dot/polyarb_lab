"""
reward_aware_maker_presence_scoring_line
polyarb_lab / research_line / validation-only

Diagnoses WHY earning_percentage = 0% for found survivors.

earning_percentage = 0% from /rewards/user/markets is account-state truth,
NOT market-thesis falsification.  0% means one of:

  A. NO_ACTIVE_PRESENCE   — account has no maker address / no open orders
  B. OUTSIDE_WINDOW       — orders exist but spread > rewards_max_spread
  C. BELOW_MIN_SIZE       — orders exist but size < rewards_min_size
  D. NO_BILATERAL_QUOTES  — only one side quoted; rewards require both sides
  E. NOT_YET_ACTIVE       — orders placed but current period hasn't settled
  F. SCORING_ZERO         — qualifying orders present but earning 0% (true competitive failure)
  G. NOT_IN_USER_MARKETS  — market not in /user/markets (not registered as maker)

Only case F warrants DOWNGRADE.  All others = CONTINUE (no active participation yet).

Data sources:
  Primary:  /rewards/user/markets entry fields (re-fetched, not cached)
              -> maker_address, spread, earnings, tokens, rewards_max_spread, rewards_min_size
  Secondary: GET /orders?market={condition_id}  (L2 auth)
              -> active open orders for this account
  Context:   GET /book?token_id={token_id}  (public)
              -> current order book state

No order submission.  No mainline contamination.  Read-only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .auth_rewards_truth import (
    CLOBCredentials,
    UserMarketsRawFetch,
    _auth_get,
    extract_earning_percentage,
    lookup_user_market_entry,
)
from .discovery import CLOB_BOOK_PATH, _safe_float, _safe_str

logger = logging.getLogger(__name__)

# Orders endpoint paths (L2 auth required); tried in order
ORDERS_PATH      = "/orders"
ORDERS_DATA_PATH = "/data/orders"

# Ethereum null address — sentinel for "no active maker registered"
# Python bool("0x000...") == True so must check explicitly, not with `not addr`
_NULL_ETH_ADDRESS = "0x0000000000000000000000000000000000000000"

def _is_null_address(addr: Optional[str]) -> bool:
    """Return True for empty, None, or the Ethereum zero address."""
    if not addr:
        return True
    return addr.strip().lower() == _NULL_ETH_ADDRESS.lower()


# ---------------------------------------------------------------------------
# Diagnosis verdict constants
# ---------------------------------------------------------------------------
# Five explicit states — no catch-all UNKNOWN that could leak into CONTINUE

DIAG_NO_ACTIVE_PRESENCE   = "NO_ACTIVE_PRESENCE"   # null/zero address OR zero open orders
DIAG_IDENTITY_MISMATCH    = "IDENTITY_MISMATCH"    # entry address ≠ account wallet
DIAG_NON_SCORING_PRESENCE = "NON_SCORING_PRESENCE" # orders exist but fail qualification
DIAG_NOT_IN_USER_MARKETS  = "NOT_IN_USER_MARKETS"  # not registered as maker
DIAG_SCORING_ZERO         = "SCORING_ZERO"         # qualifying orders, still 0% → DOWNGRADE
DIAG_UNVERIFIABLE         = "UNVERIFIABLE"         # order-state unavailable (405, no creds)

# Deprecated aliases kept for any callers that reference old codes
DIAG_OUTSIDE_WINDOW      = "NON_SCORING_PRESENCE"
DIAG_BELOW_MIN_SIZE      = "NON_SCORING_PRESENCE"
DIAG_NO_BILATERAL_QUOTES = "NON_SCORING_PRESENCE"
DIAG_UNKNOWN             = "UNVERIFIABLE"

# Judgment constants (per-survivor, separate from line verdict)
JUDGMENT_CONTINUE      = "CONTINUE"
JUDGMENT_DOWNGRADE     = "DOWNGRADE"
JUDGMENT_UNVERIFIABLE  = "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# Order data (from /orders endpoint)
# ---------------------------------------------------------------------------

@dataclass
class OpenOrderSummary:
    """Summary of this account's open orders on one market."""
    orders_http_status: int
    fetch_ok: bool
    total_orders: int = 0
    bid_orders: int = 0
    ask_orders: int = 0
    # Orders within reward window (spread <= rewards_max_spread)
    bid_orders_in_window: int = 0
    ask_orders_in_window: int = 0
    # Largest bid/ask sizes within window
    max_bid_size_in_window: Optional[float] = None
    max_ask_size_in_window: Optional[float] = None
    # Raw order list (first 10)
    raw_orders_sample: list[dict[str, Any]] = field(default_factory=list)
    error_detail: str = ""
    # Human-readable explanation of why fetch failed (e.g. "HTTP_405_ALL_VARIANTS")
    orders_fetch_blocker: str = ""
    # All status codes observed across endpoint variants tried
    variant_statuses: list[tuple[str, int]] = field(default_factory=list)


def fetch_open_orders(
    host: str,
    condition_id: str,
    token_id: str,
    creds: CLOBCredentials,
    client: httpx.Client,
    rewards_max_spread_fraction: float,
    rewards_min_size: float,
) -> OpenOrderSummary:
    """
    Fetch this account's open orders for a market via L2 auth.

    Tries all known endpoint + param variants in order.  The Polymarket CLOB
    REST API shape can differ across deployments; we probe until one returns 200.

    Variants tried (path, params):
      1. GET /orders  {}                          — all open orders, no filter
      2. GET /orders  {asset_id: token_id}        — token as asset_id
      3. GET /orders  {market: condition_id}      — condition_id as market
      4. GET /orders  {token_id: token_id}        — token_id directly
      5. GET /orders  {market_id: condition_id}   — condition_id as market_id
      6. GET /data/orders  {market: condition_id} — gamma-style data endpoint

    If ALL variants return 405 or non-200, `fetch_ok=False` and
    `orders_fetch_blocker` records the reason.  Caller must treat this as
    UNVERIFIABLE — do not infer order state from a 405.
    """
    variants: list[tuple[str, dict]] = [
        (ORDERS_PATH, {}),
        (ORDERS_PATH, {"asset_id": token_id}),
        (ORDERS_PATH, {"market": condition_id}),
        (ORDERS_PATH, {"token_id": token_id}),
        (ORDERS_PATH, {"market_id": condition_id}),
        (ORDERS_DATA_PATH, {"market": condition_id}),
    ]

    best_status = -1
    best_body: dict[str, Any] = {}
    variant_statuses: list[tuple[str, int]] = []
    all_405 = True

    for path, params in variants:
        # Skip variants where a required param is empty — server returns 400 for empty values
        if any(isinstance(v, str) and v == "" for v in params.values()):
            continue
        label = f"{path}?{'&'.join(f'{k}={v[:8] if isinstance(v,str) else v}...' for k,v in params.items()) or '(none)'}"
        s, b, _raw = _auth_get(host, path, params, creds, client)
        variant_statuses.append((label, s))
        if s == 200:
            best_status, best_body = s, b
            all_405 = False
            break
        if s != 405:
            all_405 = False
        if s != -1 and best_status == -1:
            best_status, best_body = s, b

    summary = OpenOrderSummary(
        orders_http_status=best_status,
        fetch_ok=(best_status == 200),
        variant_statuses=variant_statuses,
    )

    if best_status != 200:
        if all_405:
            summary.orders_fetch_blocker = "HTTP_405_ALL_VARIANTS"
            summary.error_detail = (
                f"HTTP 405 on all {len(variants)} endpoint variants. "
                "Method Not Allowed — endpoint does not accept GET with these params. "
                "Order-state inference is unavailable; scoring verdict must be UNVERIFIABLE."
            )
        else:
            summary.orders_fetch_blocker = f"HTTP_{best_status}"
            summary.error_detail = f"HTTP {best_status} (best variant)"
        return summary

    data = best_body.get("data") or best_body
    orders: list[dict[str, Any]] = []
    if isinstance(data, list):
        orders = [o for o in data if isinstance(o, dict)]
    elif isinstance(data, dict):
        # Some endpoints return {orders: [...]}
        orders = [o for o in (data.get("orders") or []) if isinstance(o, dict)]

    summary.total_orders = len(orders)
    summary.raw_orders_sample = orders[:10]

    bid_in = 0
    ask_in = 0
    max_bid: Optional[float] = None
    max_ask: Optional[float] = None

    for order in orders:
        side = str(order.get("side") or order.get("type") or "").upper()
        price = _safe_float(order.get("price"), default=None)  # type: ignore[arg-type]
        size  = _safe_float(order.get("size") or order.get("original_size"), default=0.0)
        status_str = str(order.get("status") or order.get("order_status") or "").upper()

        # Only count LIVE/OPEN orders
        if status_str in ("CANCELLED", "FILLED", "MATCHED", "EXPIRED"):
            continue

        if side in ("BUY", "BID"):
            summary.bid_orders += 1
            if price is not None and price >= (1.0 - rewards_max_spread_fraction):
                bid_in += 1
                if size >= rewards_min_size:
                    max_bid = max(max_bid or 0.0, size)
        elif side in ("SELL", "ASK"):
            summary.ask_orders += 1
            if price is not None and price <= rewards_max_spread_fraction:
                ask_in += 1
                if size >= rewards_min_size:
                    max_ask = max(max_ask or 0.0, size)

    summary.bid_orders_in_window = bid_in
    summary.ask_orders_in_window = ask_in
    summary.max_bid_size_in_window = max_bid
    summary.max_ask_size_in_window = max_ask
    return summary


# ---------------------------------------------------------------------------
# Book context (public — no auth)
# ---------------------------------------------------------------------------

@dataclass
class BookContext:
    """Public book snapshot for reward-window context."""
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None       # best_ask - best_bid
    bid_levels_in_window: int = 0
    ask_levels_in_window: int = 0
    fetch_ok: bool = False


def fetch_book_context(
    host: str,
    token_id: str,
    client: httpx.Client,
    rewards_max_spread_fraction: float,
) -> BookContext:
    """Fetch public book and compute reward-window context."""
    ctx = BookContext()
    try:
        url = f"{host.rstrip('/')}{CLOB_BOOK_PATH}"
        resp = client.get(url, params={"token_id": token_id}, timeout=8)
        resp.raise_for_status()
        pl = resp.json()
        bids = pl.get("bids") or []
        asks = pl.get("asks") or []

        if bids:
            bp = _safe_float(bids[0].get("price"), default=None)  # type: ignore[arg-type]
            ctx.best_bid = bp
        if asks:
            ap = _safe_float(asks[0].get("price"), default=None)  # type: ignore[arg-type]
            ctx.best_ask = ap

        if ctx.best_bid is not None and ctx.best_ask is not None:
            ctx.spread = round(ctx.best_ask - ctx.best_bid, 4)

        # Count levels within reward window
        if ctx.best_ask is not None:
            floor = ctx.best_ask - rewards_max_spread_fraction
            ctx.bid_levels_in_window = sum(
                1 for b in bids
                if isinstance(b, dict) and (_safe_float(b.get("price"), default=0.0) or 0.0) >= floor  # type: ignore[arg-type]
            )
        if ctx.best_bid is not None:
            ceil = ctx.best_bid + rewards_max_spread_fraction
            ctx.ask_levels_in_window = sum(
                1 for a in asks
                if isinstance(a, dict) and (_safe_float(a.get("price"), default=1.0) or 1.0) <= ceil  # type: ignore[arg-type]
            )
        ctx.fetch_ok = True
    except Exception as exc:
        logger.debug("book context fetch failed token=%s: %s", token_id[:16], exc)
    return ctx


# ---------------------------------------------------------------------------
# Per-survivor presence diagnosis
# ---------------------------------------------------------------------------

@dataclass
class PresenceDiagnosis:
    """
    Full presence diagnosis for one survivor.

    Separates account-state (0% earning now) from market-potential
    (is there a path to earning?).
    """
    slug: str
    condition_id: str
    token_id: str
    rewards_max_spread_fraction: float  # in price units (not cents)
    rewards_min_size: float

    # From /user/markets entry
    found_in_user_markets: bool = False
    current_earning_pct: float = 0.0
    maker_address: Optional[str] = None     # account's quoting address
    quoted_spread: Optional[float] = None   # spread field from /user/markets entry
    cumulative_earnings: Optional[float] = None  # earnings field (USDC)
    tokens_raw: Any = None
    market_competitiveness: Optional[float] = None

    # Open orders (L2 auth)
    orders: Optional[OpenOrderSummary] = None

    # Book context (public)
    book: Optional[BookContext] = None

    # Account wallet (from credentials — used for IDENTITY_MISMATCH check)
    wallet_address: Optional[str] = None

    # Diagnosis result
    diagnosis_code: str = DIAG_UNVERIFIABLE   # default: unknown until proven otherwise
    diagnosis_detail: str = ""
    has_active_presence: bool = False
    scoring_status: str = "UNKNOWN"   # SCORING / NOT_SCORING / NO_ORDERS / UNKNOWN
    judgment: str = JUDGMENT_UNVERIFIABLE

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "condition_id": self.condition_id,
            "wallet_address": self.wallet_address,
            "found_in_user_markets": self.found_in_user_markets,
            "current_earning_pct": self.current_earning_pct,
            "maker_address": self.maker_address,
            "maker_address_is_null": _is_null_address(self.maker_address),
            "quoted_spread": self.quoted_spread,
            "rewards_max_spread_fraction": self.rewards_max_spread_fraction,
            "spread_in_window": (
                self.quoted_spread <= self.rewards_max_spread_fraction
            ) if self.quoted_spread is not None else None,
            "cumulative_earnings": self.cumulative_earnings,
            "market_competitiveness": self.market_competitiveness,
            "orders_http_status": self.orders.orders_http_status if self.orders else None,
            "orders_fetch_blocker": self.orders.orders_fetch_blocker if self.orders else None,
            "orders_variant_statuses": self.orders.variant_statuses if self.orders else None,
            "open_bid_orders": self.orders.bid_orders if self.orders else None,
            "open_ask_orders": self.orders.ask_orders if self.orders else None,
            "bid_orders_in_window": self.orders.bid_orders_in_window if self.orders else None,
            "ask_orders_in_window": self.orders.ask_orders_in_window if self.orders else None,
            "book_best_bid": self.book.best_bid if self.book else None,
            "book_best_ask": self.book.best_ask if self.book else None,
            "book_spread": self.book.spread if self.book else None,
            "book_bid_levels_in_window": self.book.bid_levels_in_window if self.book else None,
            "book_ask_levels_in_window": self.book.ask_levels_in_window if self.book else None,
            "diagnosis_code": self.diagnosis_code,
            "diagnosis_detail": self.diagnosis_detail,
            "has_active_presence": self.has_active_presence,
            "scoring_status": self.scoring_status,
            "judgment": self.judgment,
        }


def _extract_entry_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Extract maker-presence fields from a /user/markets entry.

    Handles multiple possible field name variants defensively.
    Returns dict with canonical keys.
    """
    maker_addr = _safe_str(entry.get("maker_address") or entry.get("makerAddress") or "")
    # spread: may be in fractional price units (0.04 = 4 cents) or raw cents
    spread_raw = _safe_float(entry.get("spread"), default=None)  # type: ignore[arg-type]
    earnings = _safe_float(
        entry.get("earnings") or entry.get("total_earnings"), default=None  # type: ignore[arg-type]
    )
    earn_pct = _safe_float(
        entry.get("earning_percentage") or entry.get("earnings_percentage"), default=None  # type: ignore[arg-type]
    )
    if earn_pct is not None and earn_pct > 1.0:
        earn_pct = earn_pct / 100.0
    competitiveness = _safe_float(
        entry.get("market_competitiveness"), default=None  # type: ignore[arg-type]
    )
    return {
        "maker_address": maker_addr,
        "spread": spread_raw,
        "earnings": earnings,
        "earning_percentage": earn_pct,
        "tokens": entry.get("tokens"),
        "market_competitiveness": competitiveness,
    }


def build_presence_diagnosis(
    slug: str,
    condition_id: str,
    token_id: str,
    rewards_max_spread_cents: float,
    rewards_min_size: float,
    user_markets_fetch: Optional[UserMarketsRawFetch],
    host: str,
    creds: Optional[CLOBCredentials],
    http_client: httpx.Client,
    wallet_address: Optional[str] = None,
) -> PresenceDiagnosis:
    """
    Build full presence diagnosis for one survivor.

    Step 1: Locate entry in /user/markets → extract maker_address, spread, earnings
    Step 2: Fetch open orders via /orders (L2 auth)
    Step 3: Fetch public book for context
    Step 4: Synthesize diagnosis code + judgment
    """
    # Reward window in fractional price units (e.g. 4 cents → 0.04)
    spread_frac = rewards_max_spread_cents / 100.0

    diag = PresenceDiagnosis(
        slug=slug,
        condition_id=condition_id,
        token_id=token_id,
        rewards_max_spread_fraction=spread_frac,
        rewards_min_size=rewards_min_size,
        wallet_address=wallet_address,
    )

    # ── Step 1: /user/markets entry ──────────────────────────────────────
    if user_markets_fetch and user_markets_fetch.fetch_ok:
        from .auth_rewards_truth import lookup_user_market_entry
        entry, _match_key = lookup_user_market_entry(
            raw_fetch=user_markets_fetch,
            condition_id=condition_id,
            token_id=token_id,
            slug=slug,
        )
        if entry:
            diag.found_in_user_markets = True
            fields = _extract_entry_fields(entry)
            diag.maker_address         = fields["maker_address"] or None
            diag.quoted_spread         = fields["spread"]
            diag.cumulative_earnings   = fields["earnings"]
            diag.current_earning_pct   = fields["earning_percentage"] or 0.0
            diag.tokens_raw            = fields["tokens"]
            diag.market_competitiveness = fields["market_competitiveness"]

    # ── Step 2: Open orders (L2 auth) ────────────────────────────────────
    if creds is not None:
        diag.orders = fetch_open_orders(
            host=host,
            condition_id=condition_id,
            token_id=token_id,
            creds=creds,
            client=http_client,
            rewards_max_spread_fraction=spread_frac,
            rewards_min_size=rewards_min_size,
        )

    # ── Step 3: Public book context ──────────────────────────────────────
    diag.book = fetch_book_context(
        host=host,
        token_id=token_id,
        client=http_client,
        rewards_max_spread_fraction=spread_frac,
    )

    # ── Step 4: Synthesis ────────────────────────────────────────────────
    _apply_diagnosis(diag)
    return diag


def _apply_diagnosis(diag: PresenceDiagnosis) -> None:
    """
    Fill diagnosis_code, scoring_status, judgment from collected evidence.

    Five explicit states — no catch-all that can leak into CONTINUE:

      NOT_IN_USER_MARKETS   → CONTINUE  (not registered; 0% expected)
      NO_ACTIVE_PRESENCE    → CONTINUE  (null/zero address or zero orders; 0% expected)
      IDENTITY_MISMATCH     → UNVERIFIABLE (entry address ≠ wallet; can't infer scoring)
      UNVERIFIABLE          → UNVERIFIABLE (orders unavailable: 405, no creds)
      NON_SCORING_PRESENCE  → CONTINUE  (orders exist but fail qualification; 0% expected)
      SCORING_ZERO          → DOWNGRADE  (qualifying orders present but still 0%)

    Critical: UNKNOWN is not a valid explained-non-participation state.
    If order-state is unavailable, verdict must be UNVERIFIABLE.
    """
    # ── Case 1: not registered as maker ──────────────────────────────────────
    if not diag.found_in_user_markets:
        diag.diagnosis_code      = DIAG_NOT_IN_USER_MARKETS
        diag.scoring_status      = "NO_ORDERS"
        diag.has_active_presence = False
        diag.judgment            = JUDGMENT_CONTINUE
        diag.diagnosis_detail    = (
            "Market not found in /rewards/user/markets. "
            "Account has not registered as a maker on this market. "
            "0% earning is expected — no participation."
        )
        return

    # ── Case 2: null / zero Ethereum address ─────────────────────────────────
    # NOTE: Python `not addr` does NOT catch "0x000...000" because it is
    # a non-empty string and evaluates True. Must use _is_null_address().
    if _is_null_address(diag.maker_address):
        diag.diagnosis_code      = DIAG_NO_ACTIVE_PRESENCE
        diag.scoring_status      = "NO_ORDERS"
        diag.has_active_presence = False
        diag.judgment            = JUDGMENT_CONTINUE
        diag.diagnosis_detail    = (
            f"maker_address={diag.maker_address!r} is the Ethereum null/zero address. "
            "0x000...000 is a sentinel for 'no active maker registered on this market'. "
            "Account appears in /user/markets but has never placed qualifying maker orders. "
            "0% earning is expected — no active maker identity."
        )
        return

    # ── Case 3: identity mismatch (non-null address ≠ account wallet) ────────
    if (
        diag.wallet_address
        and diag.maker_address
        and diag.maker_address.lower() != diag.wallet_address.lower()
    ):
        diag.diagnosis_code      = DIAG_IDENTITY_MISMATCH
        diag.scoring_status      = "UNKNOWN"
        diag.has_active_presence = True
        diag.judgment            = JUDGMENT_UNVERIFIABLE
        diag.diagnosis_detail    = (
            f"maker_address={diag.maker_address} does not match "
            f"account wallet={diag.wallet_address}. "
            "Cannot confirm this account controls the registered maker address. "
            "Order-state inference is not reliable under identity mismatch. "
            "Requires manual investigation."
        )
        return

    # ── Non-null, non-zero address present (and matches wallet if known) ──────

    # ── Case 4: order-state unavailable ──────────────────────────────────────
    orders = diag.orders
    if orders is None or not orders.fetch_ok:
        blocker = ""
        if orders is not None:
            blocker = (
                f" orders_http_status={orders.orders_http_status}"
                f" blocker={orders.orders_fetch_blocker!r}"
            )
        diag.diagnosis_code      = DIAG_UNVERIFIABLE
        diag.scoring_status      = "UNKNOWN"
        diag.has_active_presence = True   # maker_address non-null; orders unknown
        diag.judgment            = JUDGMENT_UNVERIFIABLE
        diag.diagnosis_detail    = (
            f"maker_address is non-null non-zero ({diag.maker_address}). "
            f"Open-order state unavailable.{blocker} "
            "Cannot confirm whether orders are active, in-window, or qualifying. "
            "UNVERIFIABLE — do NOT infer non-participation from unavailable orders."
        )
        return

    # ── Orders fetch succeeded (HTTP 200) ─────────────────────────────────────
    has_bids = orders.bid_orders > 0
    has_asks = orders.ask_orders > 0
    has_any  = orders.total_orders > 0

    # ── Case 5: no open orders ────────────────────────────────────────────────
    if not has_any:
        diag.diagnosis_code      = DIAG_NO_ACTIVE_PRESENCE
        diag.scoring_status      = "NO_ORDERS"
        diag.has_active_presence = False
        diag.judgment            = JUDGMENT_CONTINUE
        diag.diagnosis_detail    = (
            f"maker_address present ({diag.maker_address}) "
            "but /orders returned 0 open orders. "
            "Account has no active maker orders. "
            "0% earning is expected — no live participation."
        )
        return

    diag.has_active_presence = True

    bids_in     = orders.bid_orders_in_window
    asks_in     = orders.ask_orders_in_window
    min_size    = diag.rewards_min_size
    spread_frac = diag.rewards_max_spread_fraction

    # ── Case 6: NON_SCORING_PRESENCE — orders exist but fail qualification ────
    # Sub-reasons reported in detail but all map to same diagnosis code.

    if not has_bids or not has_asks:
        missing_side = "ask" if not has_asks else "bid"
        diag.diagnosis_code   = DIAG_NON_SCORING_PRESENCE
        diag.scoring_status   = "NOT_SCORING"
        diag.judgment         = JUDGMENT_CONTINUE
        diag.diagnosis_detail = (
            f"Open orders: {orders.bid_orders} bids, {orders.ask_orders} asks. "
            "Maker rewards require simultaneous bid + ask. "
            f"Missing {missing_side} side → not scoring. 0% earning is expected."
        )
        return

    if bids_in == 0 and asks_in == 0:
        diag.diagnosis_code   = DIAG_NON_SCORING_PRESENCE
        diag.scoring_status   = "NOT_SCORING"
        diag.judgment         = JUDGMENT_CONTINUE
        diag.diagnosis_detail = (
            f"{orders.total_orders} open orders, "
            f"none within reward spread window (±{spread_frac:.4f} price units). "
            "0% earning is expected — must move quotes into reward window."
        )
        return

    bid_qualifying = (
        orders.max_bid_size_in_window is not None
        and orders.max_bid_size_in_window >= min_size
    )
    ask_qualifying = (
        orders.max_ask_size_in_window is not None
        and orders.max_ask_size_in_window >= min_size
    )

    if not bid_qualifying or not ask_qualifying:
        diag.diagnosis_code   = DIAG_NON_SCORING_PRESENCE
        diag.scoring_status   = "NOT_SCORING"
        diag.judgment         = JUDGMENT_CONTINUE
        diag.diagnosis_detail = (
            f"Orders in reward window: {bids_in} bids, {asks_in} asks. "
            f"Max size below rewards_min_size={min_size:.0f}sh. "
            f"bid_max={orders.max_bid_size_in_window}sh "
            f"ask_max={orders.max_ask_size_in_window}sh. "
            "0% earning expected — must increase order size to qualify."
        )
        return

    # ── Case 7: SCORING_ZERO — qualifying orders present, still 0% ───────────
    diag.diagnosis_code   = DIAG_SCORING_ZERO
    diag.scoring_status   = "SCORING"
    diag.judgment         = JUDGMENT_DOWNGRADE
    diag.diagnosis_detail = (
        f"Qualifying maker orders: "
        f"{bids_in} bids / {asks_in} asks in reward window, "
        f"size >= {min_size:.0f}sh. "
        f"Yet earning_percentage = {diag.current_earning_pct * 100:.4f}%. "
        "Bilateral qualifying presence confirmed but earning is zero. "
        "Suggests extreme competition or scoring failure. "
        "DOWNGRADE — this is the only case warranting reduced confidence."
    )
