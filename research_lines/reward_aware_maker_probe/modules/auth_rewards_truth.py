"""
reward_aware_authenticated_rewards_truth_line
polyarb_lab / research_line / validation-only

Authenticated user-level reward truth for the 4 confirmed survivors.

Data sources (CLOB L2 auth required):
  GET /rewards/user/percentages
    -> user's actual reward percentage across all eligible markets
  GET /rewards/user/markets
    -> user's reward-eligible markets with per-market share breakdown

Auth mechanism: Polymarket CLOB L2 (API key + HMAC-SHA256 signature)
  Headers: CLOB-API-KEY, CLOB-SIGNATURE, CLOB-TIMESTAMP, CLOB-PASSPHRASE
  Signature: base64(HMAC-SHA256(api_secret, timestamp + method + path + body))
  IMPORTANT: `path` in signature = URL path only, NO query string.
  L2 is distinct from L1 (L1 uses EIP-712 Ethereum wallet signing).

Credential sources (env vars only — no file reads, no hardcoding):
  POLY_API_KEY       — CLOB API key
  POLY_API_SECRET    — CLOB API secret (used for HMAC signing)
  POLY_PASSPHRASE    — CLOB passphrase
  POLY_WALLET_ADDRESS — Polymarket wallet address (for user-scoped queries)

Three-way comparison per survivor:
  Column 1: model_share_fraction (5%, probe constant)
  Column 2: implied_share_official (from market_competitiveness proxy)
  Column 3: auth_reward_pct (authenticated user-level reward %)

Final judgment: CONTINUE / DOWNGRADE / UNVERIFIABLE

STRICT RULES:
  - No order submission.
  - No mainline contamination.
  - market_competitiveness-derived share is supporting evidence only.
  - Authenticated % is user-level truth (not guaranteed future rate).
  - PARK/DOWNGRADE verdicts must be reachable.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .discovery import _safe_float, _safe_str
from .ev_model import REWARD_POOL_SHARE_FRACTION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLOB L2 auth constants
# ---------------------------------------------------------------------------

CLOB_AUTH_KEY_HEADER        = "CLOB-API-KEY"
CLOB_AUTH_SIGNATURE_HEADER  = "CLOB-SIGNATURE"
CLOB_AUTH_TIMESTAMP_HEADER  = "CLOB-TIMESTAMP"
CLOB_AUTH_PASSPHRASE_HEADER = "CLOB-PASSPHRASE"

# Official documented reward endpoints (CLOB L2 auth required)
# Source: Polymarket CLOB API docs
AUTH_USER_PERCENTAGES_PATH = "/rewards/user/percentages"   # user's reward % per market
AUTH_USER_MARKETS_PATH     = "/rewards/user/markets"       # user's reward-eligible markets

# Verdict constants
VERDICT_CONTINUE      = "CONTINUE"
VERDICT_DOWNGRADE     = "DOWNGRADE"
VERDICT_UNVERIFIABLE  = "UNVERIFIABLE"

# Share comparison thresholds
# If authenticated share is < model assumption × this factor → DOWNGRADE signal
DOWNGRADE_THRESHOLD_FACTOR = 0.50   # auth share < 50% of model assumption → warning
# If authenticated share confirms model is within ±40% → CONTINUE
CONTINUE_TOLERANCE = 0.40


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

@dataclass
class CLOBCredentials:
    """
    CLOB API credentials for L2 auth (HMAC-SHA256).

    L2 auth uses api_key + api_secret + passphrase.
    L1 auth (EIP-712 Ethereum wallet signing) is a separate mechanism not used here.
    Loaded from environment variables only — never from files or literals.
    """
    api_key: str
    api_secret: str
    passphrase: str
    wallet_address: str

    @property
    def is_complete(self) -> bool:
        """True only if all four credential fields are non-empty."""
        return bool(self.api_key and self.api_secret and self.passphrase)


def load_credentials_from_env() -> Optional[CLOBCredentials]:
    """
    Load CLOB credentials from environment variables.

    Returns None if any required credential is missing.
    Does NOT raise — callers must handle the None case as UNVERIFIABLE.

    Required env vars:
      POLY_API_KEY        — CLOB API key
      POLY_API_SECRET     — CLOB API secret (used for HMAC signing)
      POLY_PASSPHRASE     — CLOB passphrase

    Optional:
      POLY_WALLET_ADDRESS — Wallet address (needed for user-scoped endpoints)
    """
    api_key    = os.environ.get("POLY_API_KEY", "").strip()
    api_secret = os.environ.get("POLY_API_SECRET", "").strip()
    passphrase = os.environ.get("POLY_PASSPHRASE", "").strip()
    wallet     = os.environ.get("POLY_WALLET_ADDRESS", "").strip()

    if not (api_key and api_secret and passphrase):
        logger.debug(
            "CLOB credentials incomplete: key=%s secret=%s passphrase=%s",
            bool(api_key), bool(api_secret), bool(passphrase),
        )
        return None

    return CLOBCredentials(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        wallet_address=wallet,
    )


def get_missing_env_vars() -> list[str]:
    """Return list of missing required env var names for user guidance."""
    missing = []
    for var in ("POLY_API_KEY", "POLY_API_SECRET", "POLY_PASSPHRASE"):
        if not os.environ.get(var, "").strip():
            missing.append(var)
    if not os.environ.get("POLY_WALLET_ADDRESS", "").strip():
        missing.append("POLY_WALLET_ADDRESS (optional but needed for user-scoped queries)")
    return missing


# ---------------------------------------------------------------------------
# L2 auth header generation
# ---------------------------------------------------------------------------

def _build_l2_signature(
    api_secret: str,
    timestamp: str,
    method: str,
    path: str,
    body: str = "",
) -> str:
    """
    Generate Polymarket CLOB L2 HMAC-SHA256 signature.

    Canonical format (matches py-clob-client v0.x):
      message   = timestamp_ms + method.upper() + path + body
      signature = HMAC-SHA256(api_secret, message).hexdigest()

    Notes:
      - timestamp must be milliseconds since epoch (13-digit string)
      - path = URL path ONLY — do NOT include query string
      - body = "" for GET requests
      - hexdigest (lowercase hex, 64 chars) — NOT base64

    History: previous implementation used base64(digest()) with second-granularity
    timestamps.  Aligned to py-clob-client canonical (hexdigest + ms) 2026-03-25.
    """
    message = timestamp + method.upper() + path + body
    mac = hmac.new(
        api_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    )
    return mac.hexdigest()


def get_l2_auth_headers(
    creds: CLOBCredentials,
    method: str,
    path: str,
    body: str = "",
) -> dict[str, str]:
    """
    Build CLOB L2 auth headers for one request.

    Args:
        creds:  CLOBCredentials (must have api_key, api_secret, passphrase)
        method: HTTP method string ("GET", "POST", etc.)
        path:   URL path WITHOUT query string
        body:   Request body as string (empty string for GET)

    Timestamp is milliseconds since epoch (13 digits), matching py-clob-client.
    Signature is HMAC-SHA256 hexdigest — 64 lowercase hex chars.
    """
    timestamp = str(int(time.time() * 1000))   # milliseconds — matches py-clob-client
    signature = _build_l2_signature(
        api_secret=creds.api_secret,
        timestamp=timestamp,
        method=method,
        path=path,
        body=body,
    )
    return {
        CLOB_AUTH_KEY_HEADER:        creds.api_key,
        CLOB_AUTH_SIGNATURE_HEADER:  signature,
        CLOB_AUTH_TIMESTAMP_HEADER:  timestamp,
        CLOB_AUTH_PASSPHRASE_HEADER: creds.passphrase,
    }


# ---------------------------------------------------------------------------
# Authenticated endpoint responses
# ---------------------------------------------------------------------------

@dataclass
class AuthPercentagesResponse:
    """Response from GET /rewards/percentages for one token."""
    token_id: str
    fetch_ok: bool
    http_status: int
    fetched_at: datetime
    # Raw fields from response
    raw_payload: dict[str, Any] = field(default_factory=dict)
    # Parsed reward percentage (0.0–1.0 scale, or None if not parseable)
    user_reward_pct: Optional[float] = None          # user's share fraction (0..1)
    user_reward_pct_display: Optional[str] = None    # display string e.g. "16.48%"
    # Secondary fields (if endpoint returns them)
    bid_reward_pct: Optional[float] = None
    ask_reward_pct: Optional[float] = None
    error_detail: str = ""


@dataclass
class AuthUserSharesResponse:
    """Per-survivor parsed share data (populated from UserMarketsRawFetch lookup)."""
    token_id: str
    fetch_ok: bool
    http_status: int
    fetched_at: datetime
    raw_payload: dict[str, Any] = field(default_factory=dict)
    # Parsed fields
    user_bid_share: Optional[float] = None
    user_ask_share: Optional[float] = None
    user_avg_share: Optional[float] = None
    error_detail: str = ""


@dataclass
class UserMarketsRawFetch:
    """
    Full result of GET /rewards/user/markets (fetched once, shared across all survivors).

    Holds all paginated entries so each survivor lookup does not re-fetch.
    Diagnostic fields (first_entry_keys, pages_fetched) support field discovery
    when the response shape is not yet confirmed.
    """
    http_status: int
    fetch_ok: bool
    total_entries: int
    raw_entries: list[dict[str, Any]] = field(default_factory=list)
    error_detail: str = ""
    raw_error_text: str = ""    # full raw response body when fetch_ok=False
    first_entry_keys: list[str] = field(default_factory=list)   # keys in first entry
    pages_fetched: int = 0
    fetched_at: Optional[datetime] = None



# ---------------------------------------------------------------------------
# Authenticated fetch functions
# ---------------------------------------------------------------------------

def _auth_get(
    host: str,
    path: str,
    params: dict[str, str],
    creds: CLOBCredentials,
    client: httpx.Client,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any], str]:
    """
    Execute an authenticated GET request with CLOB L2 headers.

    Returns (http_status, response_body_dict, raw_text).
    raw_text is the full response body as a string (for non-200 diagnosis).
    Body dict is {} on parse failure.

    L2 auth signs the path only — query params are passed to the server but
    are NOT included in the HMAC signature message.
    """
    # L2 auth: sign the path only, NOT the query string
    headers = get_l2_auth_headers(creds=creds, method="GET", path=path)
    url = f"{host.rstrip('/')}{path}"
    try:
        resp = client.get(url, params=params, headers=headers, timeout=timeout)
        raw_text = resp.text
        try:
            body = resp.json()
        except Exception:
            body = {}
        if resp.status_code != 200:
            logger.warning(
                "auth_get HTTP %d  path=%s  params=%s  body=%s",
                resp.status_code, path,
                {k: v[:8] + "..." if isinstance(v, str) and len(v) > 8 else v for k, v in params.items()},
                raw_text[:400],
            )
        return resp.status_code, body, raw_text
    except Exception as exc:
        logger.debug("auth_get failed path=%s: %s", path, exc)
        return -1, {}, str(exc)



def extract_earning_percentage(
    entry: dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract (avg_pct, bid_pct, ask_pct) from a /user/markets entry.

    Returns fractions in 0.0–1.0 range.  Normalises values > 1.0 by ÷100.

    Tries Polymarket-style field name variants:
      avg: earning_percentage, earnings_percentage, percentage, avg_percentage,
           earning, earnings, share, pct, reward_percentage
      bid: bid_earning_percentage, bid_percentage, bid_share
      ask: ask_earning_percentage, ask_percentage, ask_share
    """
    def _norm(raw: Any) -> Optional[float]:
        v = _safe_float(raw, default=None)  # type: ignore[arg-type]
        if v is None:
            return None
        return round(v / 100.0 if v > 1.0 else v, 6)

    avg_fields = (
        "earning_percentage", "earnings_percentage", "percentage",
        "avg_percentage", "earning", "earnings", "share", "pct",
        "reward_percentage", "rewardPercentage", "earningPercentage",
    )
    bid_fields = (
        "bid_earning_percentage", "bid_percentage", "bid_share",
        "bidPercentage", "bidEarningPercentage",
    )
    ask_fields = (
        "ask_earning_percentage", "ask_percentage", "ask_share",
        "askPercentage", "askEarningPercentage",
    )

    avg_pct = next(
        (_norm(entry.get(f)) for f in avg_fields if entry.get(f) is not None), None
    )
    bid_pct = next(
        (_norm(entry.get(f)) for f in bid_fields if entry.get(f) is not None), None
    )
    ask_pct = next(
        (_norm(entry.get(f)) for f in ask_fields if entry.get(f) is not None), None
    )
    return avg_pct, bid_pct, ask_pct


def fetch_all_user_markets(
    host: str,
    creds: CLOBCredentials,
    client: httpx.Client,
) -> UserMarketsRawFetch:
    """
    GET /rewards/user/markets  — fetched ONCE, not per-survivor.

    Handles pagination via next_cursor.  All pages are merged into raw_entries.
    The caller passes the returned UserMarketsRawFetch to each survivor lookup
    so the endpoint is not called repeatedly.

    Safety limit: max 20 pages (avoids unbounded loops on broken cursors).
    """
    fetched_at = datetime.now(timezone.utc)
    all_entries: list[dict[str, Any]] = []
    pages = 0

    status, body, raw_text = _auth_get(host, AUTH_USER_MARKETS_PATH, {}, creds, client)
    if status != 200:
        return UserMarketsRawFetch(
            http_status=status,
            fetch_ok=False,
            total_entries=0,
            error_detail=f"HTTP {status}",
            raw_error_text=raw_text[:800],
            fetched_at=fetched_at,
        )

    def _extend(b: dict) -> str:
        """Extract data list from body, return next_cursor."""
        data = b.get("data", b)
        if isinstance(data, list):
            all_entries.extend(e for e in data if isinstance(e, dict))
        elif isinstance(data, dict):
            all_entries.append(data)
        return b.get("next_cursor") or ""

    cursor = _extend(body)
    pages = 1

    while cursor and cursor not in ("", "LTE=") and pages < 20:
        status2, body2, _ = _auth_get(
            host, AUTH_USER_MARKETS_PATH, {"next_cursor": cursor}, creds, client
        )
        if status2 != 200:
            logger.debug("user/markets pagination failed page=%d status=%d", pages, status2)
            break
        cursor = _extend(body2)
        pages += 1

    first_keys = list(all_entries[0].keys()) if all_entries else []
    return UserMarketsRawFetch(
        http_status=status,
        fetch_ok=True,
        total_entries=len(all_entries),
        raw_entries=all_entries,
        first_entry_keys=first_keys,
        pages_fetched=pages,
        fetched_at=fetched_at,
    )


def lookup_user_market_entry(
    raw_fetch: UserMarketsRawFetch,
    condition_id: str,
    token_id: str,
    slug: str,
) -> tuple[Optional[dict[str, Any]], str]:
    """
    Find the /user/markets entry for one survivor.

    Match priority:
      1. condition_id  (most reliable — canonical market identifier)
      2. token_id      (YES-token specific — may appear as nested field)
      3. market_slug   (prefix match — handles truncated display slugs)

    Returns (entry_dict, match_key_used) or (None, "").
    """
    entries = raw_fetch.raw_entries

    # 1. condition_id — check top-level and nested fields
    for entry in entries:
        cid = (
            str(entry.get("condition_id") or "")
            or str(entry.get("conditionId") or "")
            or str(entry.get("market_id") or "")
        )
        if cid and cid == str(condition_id):
            return entry, "condition_id"

    # 2. token_id — may appear at top level or in a tokens[] array
    for entry in entries:
        # top-level token_id
        if str(entry.get("token_id") or "") == str(token_id):
            return entry, "token_id"
        # tokens[] array (some endpoints wrap per-outcome token data)
        for tok in entry.get("tokens") or []:
            if isinstance(tok, dict) and str(tok.get("token_id") or "") == str(token_id):
                return entry, "token_id_nested"

    # 3. market_slug — prefix/suffix match
    if slug:
        for entry in entries:
            entry_slug = str(
                entry.get("market_slug") or entry.get("slug") or entry.get("marketSlug") or ""
            )
            if not entry_slug:
                continue
            if (
                entry_slug == slug
                or entry_slug.startswith(slug)
                or slug.startswith(entry_slug)
            ):
                return entry, "market_slug"

    return None, ""


def extract_earning_percentage(
    entry: dict[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract (avg_pct, bid_pct, ask_pct) from a /user/markets entry.

    Returns fractions in [0, 1] range. If the API returns e.g. 16.48 (not 0.1648),
    we normalise by dividing by 100.

    Tries many possible field names since the response shape is not publicly specced.
    """
    def _norm(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return round(v / 100.0 if v > 1.0 else v, 6)

    # avg / overall
    avg: Optional[float] = None
    for f in ("earning_percentage", "earnings_percentage", "percentage",
              "avg_percentage", "reward_percentage", "pct", "share",
              "earnings", "earning"):
        v = _safe_float(entry.get(f), default=None)  # type: ignore[arg-type]
        if v is not None:
            avg = _norm(v)
            break

    # bid
    bid: Optional[float] = None
    for f in ("bid_earning_percentage", "bid_percentage", "bid_pct", "bid_share"):
        v = _safe_float(entry.get(f), default=None)  # type: ignore[arg-type]
        if v is not None:
            bid = _norm(v)
            break

    # ask
    ask: Optional[float] = None
    for f in ("ask_earning_percentage", "ask_percentage", "ask_pct", "ask_share"):
        v = _safe_float(entry.get(f), default=None)  # type: ignore[arg-type]
        if v is not None:
            ask = _norm(v)
            break

    return avg, bid, ask


def fetch_auth_percentages(
    host: str,
    token_id: str,
    creds: CLOBCredentials,
    client: httpx.Client,
) -> AuthPercentagesResponse:
    """
    GET /rewards/user/percentages

    Official documented endpoint (CLOB L2 auth required).
    Returns the authenticated user's reward percentages for their eligible markets.

    Called without params first (returns all user markets); if response is a list,
    scans for the entry matching token_id. If token_id param is accepted, also tried.

    Response shapes attempted (defensive — actual spec not published):
      {"data": [{"token_id": ..., "percentage": 0.1648, ...}]}
      {"data": [{"condition_id": ..., "percentage": 0.1648, ...}]}
      {"data": {"percentage": 0.1648}}   (flat, user-level aggregate)
    """
    fetched_at = datetime.now(timezone.utc)
    # Try without params first (most likely shape for /user/percentages)
    status, body, _ = _auth_get(host, AUTH_USER_PERCENTAGES_PATH, {}, creds, client)
    if status != 200:
        # Retry with token_id param in case endpoint filters by it
        status2, body2, _ = _auth_get(
            host, AUTH_USER_PERCENTAGES_PATH, {"token_id": token_id}, creds, client
        )
        if status2 == 200:
            status, body = status2, body2

    resp = AuthPercentagesResponse(
        token_id=token_id,
        fetch_ok=status == 200,
        http_status=status,
        fetched_at=fetched_at,
        raw_payload=body,
    )

    if status != 200:
        resp.error_detail = f"HTTP {status}"
        return resp

    # Parse the response.  /rewards/user/percentages may return:
    #   {"data": [{"token_id": "...", "percentage": 0.16, ...}, ...]}  ← list, scan for match
    #   {"data": {"token_id": "...", "percentage": 0.16, ...}}         ← single dict
    #   {"percentage": 0.16, ...}                                      ← flat
    data = body.get("data", body)
    item: dict = {}

    if isinstance(data, list):
        # Scan list for entry matching our token_id (exact match first)
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("token_id", "")) == str(token_id):
                item = entry
                break
        # Fallback: take first item if no token_id match (user may have only one market)
        if not item and data and isinstance(data[0], dict):
            item = data[0]
    elif isinstance(data, dict):
        item = data
    else:
        item = body if isinstance(body, dict) else {}

    if not item:
        resp.error_detail = (
            f"HTTP 200 but no entry matching token_id={token_id[:16]}... "
            f"in response. data type={type(data).__name__} "
            f"len={len(data) if isinstance(data, list) else 'N/A'}. "
            f"raw_keys={list(body.keys())[:8]}"
        )
        return resp

    # Extract reward percentage — try multiple field names
    for field_name in ("percentage", "avg_percentage", "reward_percentage", "share", "pct"):
        val = _safe_float(item.get(field_name), default=None)  # type: ignore[arg-type]
        if val is not None:
            # Normalize: API may return 16.48 (percent) or 0.1648 (fraction)
            if val > 1.0:
                val = val / 100.0
            resp.user_reward_pct = round(val, 6)
            resp.user_reward_pct_display = f"{val:.2%}"
            break

    # Optional bid/ask breakdown
    for field_name, attr in (
        ("bid_percentage", "bid_reward_pct"),
        ("ask_percentage", "ask_reward_pct"),
        ("bid_pct", "bid_reward_pct"),
        ("ask_pct", "ask_reward_pct"),
    ):
        val = _safe_float(item.get(field_name), default=None)  # type: ignore[arg-type]
        if val is not None:
            if val > 1.0:
                val = val / 100.0
            setattr(resp, attr, round(val, 6))

    if resp.user_reward_pct is None:
        resp.error_detail = (
            f"HTTP 200, entry found but no percentage field. "
            f"item keys: {list(item.keys())[:12]}"
        )

    return resp


def fetch_auth_user_markets(
    host: str,
    token_id: str,
    creds: CLOBCredentials,
    client: httpx.Client,
) -> AuthUserSharesResponse:
    """
    GET /rewards/user/markets

    Official documented endpoint (CLOB L2 auth required).
    Returns the authenticated user's reward-eligible markets with per-market
    share/percentage breakdown.

    Scans the returned list for the entry matching token_id.
    If no token_id match, stores first entry as fallback.
    """
    fetched_at = datetime.now(timezone.utc)
    # Fetch without params first; retry with token_id if endpoint supports filtering
    status, body, _ = _auth_get(host, AUTH_USER_MARKETS_PATH, {}, creds, client)
    if status != 200:
        status2, body2, _ = _auth_get(
            host, AUTH_USER_MARKETS_PATH, {"token_id": token_id}, creds, client
        )
        if status2 == 200:
            status, body = status2, body2

    resp = AuthUserSharesResponse(
        token_id=token_id,
        fetch_ok=status == 200,
        http_status=status,
        fetched_at=fetched_at,
        raw_payload=body,
    )

    if status != 200:
        resp.error_detail = f"HTTP {status}"
        return resp

    data = body.get("data", body)
    item: dict = {}

    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("token_id", "")) == str(token_id):
                item = entry
                break
        if not item and data and isinstance(data[0], dict):
            item = data[0]
    elif isinstance(data, dict):
        item = data

    if not isinstance(item, dict) or not item:
        resp.error_detail = (
            f"HTTP 200 but no entry for token_id={token_id[:16]}... "
            f"data type={type(data).__name__} "
            f"len={len(data) if isinstance(data, list) else 'N/A'}"
        )
        return resp

    for field_name, attr in (
        ("bid_share", "user_bid_share"),
        ("ask_share", "user_ask_share"),
        ("avg_share", "user_avg_share"),
        ("bid_percentage", "user_bid_share"),
        ("ask_percentage", "user_ask_share"),
        ("percentage", "user_avg_share"),
        ("share", "user_avg_share"),
    ):
        val = _safe_float(item.get(field_name), default=None)  # type: ignore[arg-type]
        if val is not None:
            if val > 1.0:
                val = val / 100.0
            if getattr(resp, attr) is None:  # don't overwrite
                setattr(resp, attr, round(val, 6))

    return resp


# ---------------------------------------------------------------------------
# Three-way comparison result
# ---------------------------------------------------------------------------

@dataclass
class AuthRewardTruth:
    """
    Three-way reward share comparison for one survivor.

    Columns:
      1. model_share_fraction      — probe constant (5%)
      2. implied_share_official    — from market_competitiveness (proxy, not ground truth)
      3. auth_reward_pct           — authenticated user-level % (ground truth if available)

    auth_reward_pct is only populated when CLOB credentials are available and
    the /rewards/user/percentages or /rewards/user/markets endpoint returns 200.

    Verdict: CONTINUE / DOWNGRADE / UNVERIFIABLE
    """
    slug: str
    token_id: str
    condition_id: str

    # Column 1: model assumption
    model_share_fraction: float = REWARD_POOL_SHARE_FRACTION
    model_reward_contribution: float = 0.0       # rate × model_share_fraction

    # Column 2: official competitiveness proxy (from prior official_rewards_truth run)
    implied_share_official: Optional[float] = None
    implied_reward_contribution_official: Optional[float] = None

    # Column 3: authenticated user-level truth
    auth_reward_pct: Optional[float] = None       # None = not available (no auth or fetch fail)
    auth_reward_contribution: Optional[float] = None
    auth_bid_share: Optional[float] = None
    auth_ask_share: Optional[float] = None

    # /user/markets identity mapping results
    mapping_status: str = "NO_AUTH"          # FOUND / NOT_FOUND / NO_AUTH / FETCH_FAIL
    mapping_key: str = ""                    # which field matched (condition_id / token_id / market_slug)
    matched_entry_keys: list[str] = field(default_factory=list)   # diagnostic: keys in matched entry
    markets_http_status: Optional[int] = None
    markets_total_entries: Optional[int] = None
    pct_http_status: Optional[int] = None

    # Raw /user/percentages response (secondary path, 401 expected)
    pct_response: Optional[AuthPercentagesResponse] = None

    # Comparison deltas
    auth_vs_model_delta: Optional[float] = None      # auth_pct - model_fraction (+= conservative)
    auth_vs_proxy_delta: Optional[float] = None      # auth_pct - implied_official

    # Per-survivor verdict
    verdict: str = VERDICT_UNVERIFIABLE
    verdict_detail: str = ""
    credentials_present: bool = False

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "token_id": self.token_id,
            "condition_id": self.condition_id,
            "model_share_fraction": self.model_share_fraction,
            "model_reward_contribution": self.model_reward_contribution,
            "implied_share_official": self.implied_share_official,
            "implied_reward_contribution_official": self.implied_reward_contribution_official,
            "auth_reward_pct": self.auth_reward_pct,
            "auth_reward_contribution": self.auth_reward_contribution,
            "auth_bid_share": self.auth_bid_share,
            "auth_ask_share": self.auth_ask_share,
            "auth_vs_model_delta": self.auth_vs_model_delta,
            "auth_vs_proxy_delta": self.auth_vs_proxy_delta,
            "mapping_status": self.mapping_status,
            "mapping_key": self.mapping_key,
            "matched_entry_keys": self.matched_entry_keys,
            "markets_http_status": self.markets_http_status,
            "markets_total_entries": self.markets_total_entries,
            "pct_http_status": self.pct_http_status,
            "verdict": self.verdict,
            "verdict_detail": self.verdict_detail,
            "credentials_present": self.credentials_present,
        }


def build_auth_reward_truth(
    slug: str,
    token_id: str,
    condition_id: str,
    rate_per_day: float,
    implied_share_official: Optional[float],
    creds: Optional[CLOBCredentials],
    user_markets_fetch: Optional[UserMarketsRawFetch],
    pct_response: Optional[AuthPercentagesResponse],
) -> AuthRewardTruth:
    """
    Build three-way reward truth for one survivor.

    Primary data source: user_markets_fetch (pre-fetched /user/markets, all pages).
    Secondary:           pct_response (/user/percentages — 401 expected, kept for diagnostics).

    Lookup uses condition_id → token_id → market_slug priority.
    Does NOT make any network calls — all data is passed in by the caller.
    """
    truth = AuthRewardTruth(
        slug=slug,
        token_id=token_id,
        condition_id=condition_id,
        model_share_fraction=REWARD_POOL_SHARE_FRACTION,
        model_reward_contribution=round(rate_per_day * REWARD_POOL_SHARE_FRACTION, 4),
        implied_share_official=implied_share_official,
        implied_reward_contribution_official=(
            round(rate_per_day * implied_share_official, 4)
            if implied_share_official is not None else None
        ),
        credentials_present=creds is not None,
        pct_response=pct_response,
        pct_http_status=pct_response.http_status if pct_response else None,
    )

    if creds is None:
        truth.mapping_status = "NO_AUTH"
        truth.verdict = VERDICT_UNVERIFIABLE
        truth.verdict_detail = (
            "CLOB credentials not available. "
            "Set POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE in environment."
        )
        return truth

    # /user/markets fetch result
    if user_markets_fetch is None or not user_markets_fetch.fetch_ok:
        truth.mapping_status = "FETCH_FAIL"
        truth.markets_http_status = user_markets_fetch.http_status if user_markets_fetch else None
        truth.verdict = VERDICT_UNVERIFIABLE
        truth.verdict_detail = (
            f"/rewards/user/markets fetch failed "
            f"(HTTP {truth.markets_http_status}). "
            f"Cannot map survivor to authenticated reward data."
        )
        return truth

    truth.markets_http_status   = user_markets_fetch.http_status
    truth.markets_total_entries = user_markets_fetch.total_entries

    # Lookup this survivor in the pre-fetched list
    entry, match_key = lookup_user_market_entry(
        raw_fetch=user_markets_fetch,
        condition_id=condition_id,
        token_id=token_id,
        slug=slug,
    )

    if entry is None:
        truth.mapping_status = "NOT_FOUND"
        truth.verdict = VERDICT_UNVERIFIABLE
        truth.verdict_detail = (
            f"Survivor not found in /rewards/user/markets "
            f"({user_markets_fetch.total_entries} entries, "
            f"{user_markets_fetch.pages_fetched} page(s)). "
            f"Market may not be in user's active reward pool, or "
            f"condition_id={condition_id[:20]}... not present."
        )
        return truth

    # Entry found — extract earning percentages
    truth.mapping_status     = "FOUND"
    truth.mapping_key        = match_key
    truth.matched_entry_keys = list(entry.keys())

    avg_pct, bid_pct, ask_pct = extract_earning_percentage(entry)

    auth_pct: Optional[float] = avg_pct
    truth.auth_reward_pct = auth_pct
    truth.auth_bid_share  = bid_pct
    truth.auth_ask_share  = ask_pct

    if auth_pct is not None:
        truth.auth_reward_contribution = round(rate_per_day * auth_pct, 4)
        truth.auth_vs_model_delta      = round(auth_pct - REWARD_POOL_SHARE_FRACTION, 6)
        if implied_share_official is not None:
            truth.auth_vs_proxy_delta  = round(auth_pct - implied_share_official, 6)

    # Verdict
    if auth_pct is None:
        # Entry was found but contained no parseable earning % field
        truth.verdict = VERDICT_UNVERIFIABLE
        truth.verdict_detail = (
            f"Entry matched via {match_key} but no earning % field found. "
            f"Entry keys: {list(entry.keys())[:15]}. "
            "Response shape may differ from expected — needs field inspection."
        )
    else:
        model_share = REWARD_POOL_SHARE_FRACTION
        downgrade_threshold = model_share * DOWNGRADE_THRESHOLD_FACTOR

        if auth_pct < downgrade_threshold:
            truth.verdict = VERDICT_DOWNGRADE
            truth.verdict_detail = (
                f"Authenticated earning % = {auth_pct:.2%} via {match_key}. "
                f"BELOW {DOWNGRADE_THRESHOLD_FACTOR:.0%} of model assumption ({model_share:.2%}). "
                f"Auth contribution ${truth.auth_reward_contribution:.4f}/day vs "
                f"model ${truth.model_reward_contribution:.4f}/day."
            )
        else:
            truth.verdict = VERDICT_CONTINUE
            direction = "above" if auth_pct >= model_share else "below"
            truth.verdict_detail = (
                f"Authenticated earning % = {auth_pct:.2%} via {match_key}, "
                f"{direction} model 5%. "
                f"Auth contribution: ${truth.auth_reward_contribution:.4f}/day. "
                f"Model is {'CONSERVATIVE' if auth_pct >= model_share else 'SLIGHTLY OPTIMISTIC'}."
            )

    return truth


# ---------------------------------------------------------------------------
# Credential guidance (for UNVERIFIABLE path output)
# ---------------------------------------------------------------------------

CREDENTIAL_GUIDANCE = """
To enable authenticated reward truth:

  1. Log in to https://polymarket.com
  2. Go to Profile → Settings → API Keys
  3. Generate a new CLOB API key — note the Key, Secret, and Passphrase
  4. Set environment variables before running this script:

     Windows PowerShell:
       $env:POLY_API_KEY       = "your-api-key"
       $env:POLY_API_SECRET    = "your-api-secret"
       $env:POLY_PASSPHRASE    = "your-passphrase"
       $env:POLY_WALLET_ADDRESS = "your-wallet-address"  # optional

     Or persist in your profile / .env file (do not commit to git).

  5. Re-run:
       py -3 research_lines/reward_aware_maker_probe/run_authenticated_rewards_truth.py

IMPORTANT:
  - The authenticated % reflects your account's current reward share.
  - This is a point-in-time snapshot, not a guaranteed future rate.
  - Do NOT treat auth_reward_pct as profitability proof.
  - Model robustness analysis still applies even with auth data.
"""
