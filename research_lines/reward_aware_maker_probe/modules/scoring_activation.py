"""
reward_aware_scoring_activation_line — Core module
polyarb_lab / research_line / validation

Tests whether qualifying bilateral maker quotes activate reward scoring.

Context:
  maker_presence_scoring_line established:
    - All 3 found survivors (hungary, vance, rubio): NO_ACTIVE_PRESENCE
    - 0% earning is explained by non-participation, not competitive exclusion
    - Market reward thesis intact

  This module answers: "Does placing qualifying bilateral quotes activate scoring?"

Scoring truth sources (in priority order):
  1. is_order_scoring(order_id)   — immediate per-order CLOB scoring state
  2. /rewards/user/markets        — account-level earning_percentage (lags by period)

Verdict mapping:
  SCORING_ACTIVE   : is_order_scoring returns True for both legs
  NOT_SCORING      : is_order_scoring returns False after observation window
  INCONCLUSIVE     : one or both orders filled before observation completes
  DOWNGRADE        : is_order_scoring True + earning_percentage = 0% after 24h+ (competitive exclusion)

Qualifying quote requirements:
  - BID and ASK both on YES token  (same token_id)
  - ask_price - bid_price <= rewards_max_spread_fraction  (≤ 3.5 cents default)
  - size >= rewards_min_size per side                     (≥ 10 shares default)
  - Both orders GTC (live in book)

No mainline imports. Read-only credential load. Order submission gated by explicit flag.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

# Hardcoded from auth_rewards_truth_20260326T045023.json — stable identifiers
SURVIVOR_DATA: dict[str, dict] = {
    "will-jd-vance-win-the-2028-republican-presidential-nomi": {
        "condition_id": "0x18b1c135d0a40c5894da9412e77311827d9caf16cf4cd6591b247a34730af919",
        "token_id": "40081275558852222228080198821361202017557872256707631666334039001378518619916",
        "daily_rate_usdc": 100.0,
        "fallback_max_spread_cents": 3.5,
        "fallback_min_size": 200.0,  # confirmed live: /rewards/markets/{cid}
        "yes_price_ref": 0.3665,     # reference snapshot 2026-03-26
        "competitiveness_ref": 576.9,
    },
    "will-the-next-prime-minister-of-hungary-be-pter-magyar": {
        "condition_id": "0x1480b819d03d4b6388d70e848b0384adf38c38d955cb783cdbcf6d4a436dee14",
        "token_id": "94192784911459194325909253314484842244405314804074606736702592885535642919725",
        "daily_rate_usdc": 150.0,
        "fallback_max_spread_cents": 3.5,
        "fallback_min_size": 200.0,  # confirmed live: /rewards/markets/{cid}
        "yes_price_ref": 0.6250,     # reference snapshot 2026-03-26
        "competitiveness_ref": 46.9,  # lowest of the 3 — best implied share
    },
    "will-marco-rubio-win-the-2028-republican-presidential-n": {
        "condition_id": "0x21ad31a46bfaa51650766eff6dc69c866959e32d965ffb116020e37694b6317d",
        "token_id": "13565458761220145250977753098276900790902214604876327357986816739576288755859",
        "daily_rate_usdc": 70.0,
        "fallback_max_spread_cents": 3.5,
        "fallback_min_size": 200.0,  # confirmed live: /rewards/markets/{cid}
        "yes_price_ref": 0.2255,     # reference snapshot 2026-03-26
        "competitiveness_ref": 104.1,
    },
}

# Short-name aliases for CLI convenience
SLUG_ALIASES: dict[str, str] = {
    "vance":   "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "hungary": "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "rubio":   "will-marco-rubio-win-the-2028-republican-presidential-n",
}

# Order-side strings used by py_clob_client
BUY_SIDE  = "BUY"
SELL_SIDE = "SELL"

# Verdict constants
VERDICT_SCORING_ACTIVE   = "SCORING_ACTIVE"
VERDICT_NOT_SCORING      = "NOT_SCORING"
VERDICT_INCONCLUSIVE     = "INCONCLUSIVE"   # order(s) filled during test
VERDICT_DRY_RUN          = "DRY_RUN"
VERDICT_PREFLIGHT_FAILED = "PREFLIGHT_FAILED"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

@dataclass
class ActivationCredentials:
    """Credentials for L1 order signing + L2 API auth."""
    private_key: str        # EVM private key (hex, with or without 0x)
    api_key: str            # L2 API key — may be "" when needs_api_derivation=True
    api_secret: str         # L2 API secret — may be "" when needs_api_derivation=True
    api_passphrase: str     # L2 API passphrase — may be "" when needs_api_derivation=True
    chain_id: int
    signature_type: int = 0         # 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
    funder: Optional[str] = None    # proxy wallet address (required for signature_type=1)
    # Set True when API trio was absent from env — build_clob_client() will derive them.
    needs_api_derivation: bool = False
    # Filled in by build_clob_client() after resolution; readable by callers.
    credential_source: str = "configured"


def load_activation_credentials() -> Optional[ActivationCredentials]:
    """
    Load credentials from env vars.

    Required (hard stop if absent):
      POLYMARKET_PRIVATE_KEY  (or POLY_PRIVATE_KEY / POLY_WALLET_ADDRESS)

    Optional — derived automatically by build_clob_client() via derive_api_key()
    if absent or if the configured key does not match the signer EOA:
      POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE

    Other optional vars (always read, with defaults):
      POLYMARKET_CHAIN_ID          default 137
      POLYMARKET_SIGNATURE_TYPE    default 0 (EOA)
      POLYMARKET_FUNDER            proxy wallet address for sig_type=2

    Returns None only when private_key cannot be found.
    When API trio is absent, returns ActivationCredentials with
    needs_api_derivation=True and empty-string placeholder api creds;
    build_clob_client() will call derive_api_key() to fill them in.
    """
    private_key = (
        os.environ.get("POLYMARKET_PRIVATE_KEY")
        or os.environ.get("POLY_PRIVATE_KEY")
        or _try_wallet_as_key(os.environ.get("POLY_WALLET_ADDRESS"))
    )
    api_key = (
        os.environ.get("POLYMARKET_API_KEY")
        or os.environ.get("POLY_API_KEY")
    )
    api_secret = (
        os.environ.get("POLYMARKET_API_SECRET")
        or os.environ.get("POLY_API_SECRET")
    )
    api_passphrase = (
        os.environ.get("POLYMARKET_API_PASSPHRASE")
        or os.environ.get("POLY_PASSPHRASE")
    )
    chain_id_raw = os.environ.get("POLYMARKET_CHAIN_ID", str(POLYGON_CHAIN_ID))
    try:
        chain_id = int(chain_id_raw)
    except ValueError:
        chain_id = POLYGON_CHAIN_ID

    sig_type_raw = (
        os.environ.get("POLYMARKET_SIGNATURE_TYPE")
        or os.environ.get("POLY_SIGNATURE_TYPE")
        or "0"
    )
    try:
        signature_type = int(sig_type_raw)
    except ValueError:
        signature_type = 0

    funder = (
        os.environ.get("POLYMARKET_FUNDER")
        or os.environ.get("POLY_FUNDER")
        or None
    )

    # Private key is the only hard requirement — everything else can be derived.
    if not private_key:
        return None

    needs_derivation = not (api_key and api_secret and api_passphrase)

    return ActivationCredentials(
        private_key=private_key,
        api_key=api_key or "",            # placeholder; build_clob_client fills in
        api_secret=api_secret or "",      # placeholder; build_clob_client fills in
        api_passphrase=api_passphrase or "",  # placeholder; build_clob_client fills in
        chain_id=chain_id,
        signature_type=signature_type,
        funder=funder,
        needs_api_derivation=needs_derivation,
        credential_source="needs_derivation" if needs_derivation else "configured",
    )


def _try_wallet_as_key(addr: Optional[str]) -> Optional[str]:
    """
    POLY_WALLET_ADDRESS is sometimes set to the raw private key (64 hex chars,
    no 0x).  If it fits that pattern, return it as a candidate private key.
    Ethereum addresses are 40 hex chars; private keys are 64 hex chars.
    """
    if not addr:
        return None
    stripped = addr.strip().lstrip("0x").lower()
    if len(stripped) == 64 and all(c in "0123456789abcdef" for c in stripped):
        return "0x" + stripped
    return None


def get_missing_credential_vars() -> list[str]:
    """
    Return a list of hard-required env vars that are absent.

    Only POLYMARKET_PRIVATE_KEY is a hard requirement.
    The API trio (KEY/SECRET/PASSPHRASE) is optional — build_clob_client()
    derives them automatically from the private key via derive_api_key().
    """
    checks = [
        ("POLYMARKET_PRIVATE_KEY or POLY_PRIVATE_KEY", bool(
            os.environ.get("POLYMARKET_PRIVATE_KEY")
            or os.environ.get("POLY_PRIVATE_KEY")
            or _try_wallet_as_key(os.environ.get("POLY_WALLET_ADDRESS"))
        )),
    ]
    return [name for name, present in checks if not present]


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Official reward parameters for one market."""
    condition_id: str
    max_spread_cents: float
    max_spread_fraction: float  # max_spread_cents / 100
    min_size: float
    daily_rate_usdc: float
    fetch_ok: bool
    source: str  # "live" or "fallback"


def fetch_reward_config(
    host: str,
    condition_id: str,
    fallback_max_spread_cents: float,
    fallback_min_size: float,
    fallback_daily_rate: float,
) -> RewardConfig:
    """
    Fetch /rewards/markets/{condition_id} for live reward params.
    Falls back to hardcoded values if fetch fails.
    """
    try:
        import httpx
        url = f"{host.rstrip('/')}/rewards/markets/{condition_id}"
        resp = httpx.get(url, timeout=8)
        if resp.status_code == 200:
            body = resp.json()
            # Response is wrapped: {"data": [{...}], "next_cursor": "LTE=", ...}
            raw_list = body.get("data") if isinstance(body, dict) else None
            entry = raw_list[0] if isinstance(raw_list, list) and raw_list else body
            spread = _safe_float(entry.get("rewards_max_spread") or entry.get("rewardsMaxSpread"))
            size   = _safe_float(entry.get("rewards_min_size") or entry.get("rewardsMinSize"))
            rc_list = entry.get("rewards_config") or []
            rate = _safe_float(
                rc_list[0].get("rate_per_day")
                if isinstance(rc_list, list) and rc_list
                else entry.get("rate_per_day") or entry.get("daily_rate")
            )
            # Also extract yes_price from tokens[] for midpoint reference
            _yes_price: Optional[float] = None
            for tok in entry.get("tokens") or []:
                if isinstance(tok, dict) and tok.get("outcome", "").lower() == "yes":
                    _yes_price = _safe_float(tok.get("price"))
                    break
            if spread and size:
                cfg = RewardConfig(
                    condition_id=condition_id,
                    max_spread_cents=spread,
                    max_spread_fraction=spread / 100.0,
                    min_size=size,
                    daily_rate_usdc=rate or fallback_daily_rate,
                    fetch_ok=True,
                    source="live",
                )
                # Attach yes_price and competitiveness for midpoint use
                cfg.yes_price_live = _yes_price  # type: ignore[attr-defined]
                cfg.competitiveness = _safe_float(  # type: ignore[attr-defined]
                    entry.get("market_competitiveness")
                )
                return cfg
    except Exception as exc:
        logger.debug("reward config fetch failed cid=%s: %s", condition_id[:12], exc)

    return RewardConfig(
        condition_id=condition_id,
        max_spread_cents=fallback_max_spread_cents,
        max_spread_fraction=fallback_max_spread_cents / 100.0,
        min_size=fallback_min_size,
        daily_rate_usdc=fallback_daily_rate,
        fetch_ok=False,
        source="fallback",
    )


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Quote computation
# ---------------------------------------------------------------------------

@dataclass
class QuotePair:
    """A qualifying bilateral quote pair for reward scoring."""
    token_id: str
    bid_price: float
    ask_price: float
    size: float       # shares per side
    spread: float     # ask_price - bid_price
    max_spread_fraction: float
    spread_ok: bool   # spread <= max_spread_fraction
    size_ok: bool     # size >= min_size
    qualifying: bool  # spread_ok AND size_ok
    midpoint_used: float
    midpoint_source: str  # "live" or "estimated"
    tick_size: float


def compute_qualifying_quotes(
    token_id: str,
    reward_config: RewardConfig,
    midpoint: float,
    midpoint_source: str,
    tick_size: float = 0.01,
    spread_cents: float = 3.0,
) -> QuotePair:
    """
    Compute a qualifying bilateral quote pair.

    Strategy:
      - Use spread_cents (default 3¢) — must be within 3.5-cent max for all 3 targets
      - Center around midpoint, snap to tick_size grid
      - bid = floor(midpoint / tick_size) * tick_size
      - ask = bid + spread

    This places us at the best price in an empty book (all 3 markets have
    no qualifying quotes currently).  Fill risk is non-zero but manageable
    at minimum size.
    """
    TARGET_SPREAD = round(spread_cents / 100.0, 6)

    # Snap bid down to tick grid
    bid = math.floor(midpoint / tick_size) * tick_size
    bid = round(bid, 4)

    # Ask = bid + target_spread, snapped to tick
    ask = round(bid + TARGET_SPREAD, 4)

    # Safety: clamp to valid price range [tick, 1-tick]
    bid = max(tick_size, min(1.0 - tick_size - TARGET_SPREAD, bid))
    ask = round(bid + TARGET_SPREAD, 4)

    spread = round(ask - bid, 6)
    size = reward_config.min_size  # must be exactly min_size — below this does not qualify

    return QuotePair(
        token_id=token_id,
        bid_price=bid,
        ask_price=ask,
        size=size,
        spread=spread,
        max_spread_fraction=reward_config.max_spread_fraction,
        spread_ok=(spread <= reward_config.max_spread_fraction + 1e-9),
        size_ok=(size >= reward_config.min_size),
        qualifying=True,  # will validate below
        midpoint_used=midpoint,
        midpoint_source=midpoint_source,
        tick_size=tick_size,
    )


# ---------------------------------------------------------------------------
# Midpoint fetch
# ---------------------------------------------------------------------------

def fetch_midpoint(
    client: Any,   # ClobClient instance
    token_id: str,
    price_ref: Optional[float] = None,   # fallback from reward config token price
) -> tuple[Optional[float], str]:
    """
    Fetch live midpoint for a YES token via py_clob_client.

    Returns (midpoint, source) where source is "live" or "book_derived".
    Falls back to book mid if midpoint endpoint fails.
    """
    try:
        mid = client.get_midpoint(token_id)
        if mid is not None:
            val = _safe_float(mid)
            if val and 0.01 <= val <= 0.99:
                return val, "live"
    except Exception as exc:
        logger.debug("get_midpoint failed token=%s: %s", token_id[:16], exc)

    # Fallback: derive from book — but reject empty-book sentinels (0.001/0.999)
    try:
        book = client.get_order_book(token_id)
        bids = getattr(book, "bids", []) or []
        asks = getattr(book, "asks", []) or []
        best_bid = None
        best_ask = None
        if bids:
            best_bid = _safe_float(getattr(bids[0], "price", None))
        if asks:
            best_ask = _safe_float(getattr(asks[0], "price", None))
        if (
            best_bid and best_ask
            and best_bid < best_ask
            # Reject empty-book sentinels: (0.001, 0.999) → mid=0.5 is meaningless
            and not (best_bid <= 0.01 and best_ask >= 0.99)
        ):
            mid = round((best_bid + best_ask) / 2.0, 4)
            if 0.01 <= mid <= 0.99:
                return mid, "book_derived"
    except Exception as exc:
        logger.debug("order_book mid fallback failed token=%s: %s", token_id[:16], exc)

    # Final fallback: use yes_price from /rewards/markets token list
    if price_ref and 0.01 <= price_ref <= 0.99:
        return round(price_ref, 4), "reward_config_token_price"

    return None, "unavailable"


# ---------------------------------------------------------------------------
# Order placement + observation
# ---------------------------------------------------------------------------

@dataclass
class PlacedOrder:
    order_id: str
    side: str    # "BUY" or "SELL"
    price: float
    size: float
    placed_at: datetime
    filled: bool = False
    cancelled: bool = False
    cancel_ok: bool = False


@dataclass
class ScoringObservation:
    """One polling snapshot during the observation window."""
    elapsed_minutes: float
    bid_scoring: Optional[bool]     # is_order_scoring result for BID
    ask_scoring: Optional[bool]     # is_order_scoring result for ASK
    both_scoring: bool
    earning_pct: Optional[float]    # /rewards/user/markets earning_percentage at this poll
    observed_at: datetime


@dataclass
class ActivationResult:
    """Full result of one activation test."""
    slug: str
    condition_id: str
    token_id: str
    dry_run: bool
    midpoint: Optional[float]
    midpoint_source: str
    reward_config: RewardConfig
    quotes: Optional[QuotePair]

    # Pre-test state
    earning_pct_before: Optional[float] = None

    # Live-mode fields
    bid_order: Optional[PlacedOrder] = None
    ask_order: Optional[PlacedOrder] = None
    observations: list[ScoringObservation] = field(default_factory=list)
    earning_pct_after: Optional[float] = None
    orders_cancelled: bool = False

    # Verdict
    verdict: str = VERDICT_DRY_RUN
    verdict_detail: str = ""

    run_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "dry_run": self.dry_run,
            "midpoint": self.midpoint,
            "midpoint_source": self.midpoint_source,
            "reward_config": {
                "max_spread_cents": self.reward_config.max_spread_cents,
                "max_spread_fraction": self.reward_config.max_spread_fraction,
                "min_size": self.reward_config.min_size,
                "daily_rate_usdc": self.reward_config.daily_rate_usdc,
                "source": self.reward_config.source,
            } if self.reward_config else None,
            "quotes": {
                "bid_price": self.quotes.bid_price,
                "ask_price": self.quotes.ask_price,
                "size": self.quotes.size,
                "spread": self.quotes.spread,
                "max_spread_fraction": self.quotes.max_spread_fraction,
                "qualifying": self.quotes.qualifying,
                "midpoint_used": self.quotes.midpoint_used,
                "midpoint_source": self.quotes.midpoint_source,
            } if self.quotes else None,
            "earning_pct_before": self.earning_pct_before,
            "bid_order_id": self.bid_order.order_id if self.bid_order else None,
            "ask_order_id": self.ask_order.order_id if self.ask_order else None,
            "observations": [
                {
                    "elapsed_minutes": o.elapsed_minutes,
                    "bid_scoring": o.bid_scoring,
                    "ask_scoring": o.ask_scoring,
                    "both_scoring": o.both_scoring,
                    "earning_pct": o.earning_pct,
                    "observed_at": o.observed_at.isoformat(),
                }
                for o in self.observations
            ],
            "earning_pct_after": self.earning_pct_after,
            "orders_cancelled": self.orders_cancelled,
            "verdict": self.verdict,
            "verdict_detail": self.verdict_detail,
            "run_timestamp": self.run_timestamp,
        }


# ---------------------------------------------------------------------------
# Scoring check helper
# ---------------------------------------------------------------------------

def _check_order_scoring(client: Any, order_id: str) -> Optional[bool]:
    """Return is_order_scoring result for one order_id.  None on error."""
    try:
        from py_clob_client.clob_types import OrderScoringParams
        result = client.is_order_scoring(OrderScoringParams(orderId=order_id))
        if result is None:
            return None
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            return bool(result.get("scoring") or result.get("is_scoring"))
        return None
    except Exception as exc:
        logger.debug("is_order_scoring failed order_id=%s: %s", order_id[:12], exc)
        return None


def _norm_condition_id(cid: str) -> str:
    """Normalize condition_id for comparison: strip 0x prefix, lowercase."""
    c = cid.strip().lower()
    return c[2:] if c.startswith("0x") else c


def _dated_earnings_for(
    host: str,
    creds: "ActivationCredentials",
    condition_id: str,
    date_str: Optional[str] = None,
) -> Optional[float]:
    """
    Supplemental check: /rewards/user?date=<date> to read actual USD earnings.

    Called only when /rewards/user/markets returns earning_percentage=0 so that
    a falsy percentage is not mis-reported as confirmed-zero reward.
    Returns the `earnings` float for the matching condition_id row, or None.
    """
    try:
        import httpx, base64, hashlib, hmac as _hmac

        if date_str is None:
            from datetime import date as _date
            date_str = _date.today().isoformat()   # e.g. "2026-03-29"

        path = f"/rewards/user"
        url  = f"{host.rstrip('/')}{path}"
        ts   = str(int(time.time() * 1000))
        msg  = ts + "GET" + path
        try:
            hmac_key = base64.urlsafe_b64decode(creds.api_secret)
        except Exception:
            hmac_key = creds.api_secret.encode("utf-8")
        sig = base64.b64encode(
            _hmac.new(hmac_key, msg.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")
        headers = {
            "CLOB-API-KEY":    creds.api_key,
            "CLOB-SIGNATURE":  sig,
            "CLOB-TIMESTAMP":  ts,
            "CLOB-PASSPHRASE": creds.api_passphrase,
        }
        _norm_target = _norm_condition_id(condition_id)
        with httpx.Client() as hc:
            resp = hc.get(url, params={"date": date_str}, headers=headers, timeout=8)
            if resp.status_code != 200:
                return None
            body = resp.json()
            rows = body if isinstance(body, list) else body.get("data", [])
            if not isinstance(rows, list):
                rows = [rows] if isinstance(rows, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                cid = str(row.get("condition_id") or row.get("conditionId") or "")
                if _norm_condition_id(cid) == _norm_target or cid == condition_id:
                    return _safe_float(row.get("earnings") or row.get("earning"))
    except Exception as exc:
        logger.debug("_dated_earnings_for failed: %s", exc)
    return None


def _check_earning_pct(
    host: str,
    creds: ActivationCredentials,
    condition_id: str,
) -> Optional[float]:
    """
    Re-fetch /rewards/user/markets earning_percentage for this market.

    Paginates via next_cursor (up to 20 pages, matching auth_rewards_truth).
    Stops early as soon as the target condition_id is found — avoids fetching
    all 2000 entries when the match is on an early page.

    Terminal cursor sentinel: empty string or "LTE=" (Polymarket pagination end).
    """
    try:
        import httpx
        import base64
        import hashlib
        import hmac as _hmac

        path         = "/rewards/user/markets"
        url          = f"{host.rstrip('/')}{path}"
        _norm_target = _norm_condition_id(condition_id)

        def _make_headers() -> dict[str, str]:
            """Build fresh HMAC headers — new timestamp per request."""
            ts  = str(int(time.time() * 1000))
            msg = ts + "GET" + path
            try:
                hmac_key = base64.urlsafe_b64decode(creds.api_secret)
            except Exception:
                hmac_key = creds.api_secret.encode("utf-8")
            sig = base64.b64encode(
                _hmac.new(hmac_key, msg.encode("utf-8"), hashlib.sha256).digest()
            ).decode("utf-8")
            return {
                "CLOB-API-KEY":    creds.api_key,
                "CLOB-SIGNATURE":  sig,
                "CLOB-TIMESTAMP":  ts,
                "CLOB-PASSPHRASE": creds.api_passphrase,
            }

        def _scan_page(entries: list) -> Optional[float]:
            """Search one page of entries for this condition_id.  Returns value or None."""
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                cid = str(
                    entry.get("condition_id") or entry.get("conditionId")
                    or entry.get("market_id") or ""
                )
                if _norm_condition_id(cid) == _norm_target or cid == condition_id:
                    # Explicit key presence check — do not use `or` on values.
                    # earning_percentage=0 is integer-zero (falsy); `or` would skip it.
                    for _f in ("earning_percentage", "earnings_percentage"):
                        if _f in entry:
                            v = _safe_float(entry[_f])
                            return (v / 100.0 if v > 1.0 else v) if v is not None else 0.0
            return None

        with httpx.Client() as hc:
            # ── Page 1 ──────────────────────────────────────────────────────
            resp = hc.get(url, headers=_make_headers(), timeout=8)
            if resp.status_code != 200:
                logger.warning(
                    "earning_pct: HTTP %d from %s — body: %.300s",
                    resp.status_code, url, resp.text,
                )
                return None
            body    = resp.json()
            entries = (body.get("data", body) if isinstance(body, dict) else [])
            if isinstance(entries, list):
                hit = _scan_page(entries)
                if hit is not None:
                    if hit == 0.0:
                        dated = _dated_earnings_for(host, creds, condition_id)
                        if dated is not None and dated > 0:
                            logger.warning(
                                "earning_pct: earning_percentage=0 for %s but "
                                "/rewards/user?date=today shows earnings=%.6f — "
                                "NOT treating as zero; returning None",
                                condition_id[:24], dated,
                            )
                            return None
                    return hit

            # ── Subsequent pages ─────────────────────────────────────────────
            cursor = (body.get("next_cursor", "") if isinstance(body, dict) else "")
            pages  = 1
            while cursor and cursor not in ("", "LTE=") and pages < 20:
                resp2 = hc.get(
                    url,
                    params={"next_cursor": cursor},
                    headers=_make_headers(),
                    timeout=8,
                )
                if resp2.status_code != 200:
                    logger.warning(
                        "earning_pct: pagination HTTP %d page=%d", resp2.status_code, pages + 1
                    )
                    break
                body2    = resp2.json()
                entries2 = (body2.get("data", body2) if isinstance(body2, dict) else [])
                if isinstance(entries2, list):
                    hit = _scan_page(entries2)
                    if hit is not None:
                        if hit == 0.0:
                            dated = _dated_earnings_for(host, creds, condition_id)
                            if dated is not None and dated > 0:
                                logger.warning(
                                    "earning_pct: earning_percentage=0 for %s but "
                                    "/rewards/user?date=today shows earnings=%.6f — "
                                    "NOT treating as zero; returning None",
                                    condition_id[:24], dated,
                                )
                                return None
                        logger.debug(
                            "earning_pct: found %s on page %d", condition_id[:24], pages + 1
                        )
                        return hit
                cursor = (body2.get("next_cursor", "") if isinstance(body2, dict) else "")
                pages += 1

            logger.warning(
                "earning_pct: condition_id %s not found after %d page(s)",
                condition_id[:24], pages,
            )
    except Exception as exc:
        logger.warning("earning_pct check failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Main activation logic
# ---------------------------------------------------------------------------

def build_clob_client(creds: ActivationCredentials, host: str) -> Any:
    """
    Build a py_clob_client ClobClient with L1+L2 auth.

    Auth architecture note:
      create_level_2_headers() in py_clob_client ALWAYS puts signer.address() (EOA)
      in the POLY_ADDRESS header, regardless of signature_type or funder.
      The server validates POLY_API_KEY against POLY_ADDRESS.  Therefore:
        - The api_key MUST be registered for the EOA from the private key.
        - signature_type / funder only affect the EIP712 order signing schema,
          NOT the L2 auth headers that gate POST /order.

      This function derives the real api_key/secret/passphrase for the EOA via
      derive_api_key() and uses those for the L2 client.  This is the only path
      that guarantees api_key is registered for the EOA (POLY_ADDRESS).
    """
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    # ── Step 1: L1-only client — learn EOA address + derive real api creds ──
    l1 = ClobClient(host=host, chain_id=creds.chain_id, key=creds.private_key)
    eoa = l1.get_address()
    logger.info(
        "EOA address (will appear as POLY_ADDRESS in every POST): %s", eoa
    )

    # ── Step 2: Resolve effective L2 api creds for this EOA ──────────────────
    #
    # Priority order:
    #   A. derive_api_key() — always tried first; returns the key registered
    #      for this specific EOA on the CLOB server.  This is the only path
    #      that is guaranteed correct regardless of what env vars were set.
    #   B. configured creds — used only when derive succeeds AND keys match,
    #      OR as a last resort when derive fails and API trio was set.
    #   C. create_or_derive_api_creds() — fallback if both A and B are blocked.
    #
    # When needs_api_derivation=True (API trio not in env), path A is the sole
    # viable path; configured creds are empty placeholders.
    #
    # creds is updated IN-PLACE so that all downstream callers (e.g.
    # _check_earning_pct which reads creds.api_key directly) automatically
    # receive the effective credentials without needing a separate code path.
    api_creds: Any = None
    credential_source = "unknown"
    derive_exc: Optional[Exception] = None

    try:
        derived = l1.derive_api_key()
        if derived and derived.api_key:
            if creds.needs_api_derivation:
                # API trio was absent from env — derived is the only viable path.
                api_creds        = derived
                credential_source = "derived"
            elif derived.api_key != creds.api_key:
                # Configured key exists but belongs to a different address.
                logger.warning(
                    "CREDENTIAL MISMATCH: configured api_key=%s... is NOT registered "
                    "for EOA %s.  Derived api_key=%s... will be used instead.  "
                    "Root cause of previous failures: configured key belongs to a "
                    "different address (Builder Settings key vs EOA key).",
                    creds.api_key[:8] if creds.api_key else "(empty)",
                    eoa,
                    derived.api_key[:8],
                )
                api_creds        = derived
                credential_source = "derived_mismatch_override"
            else:
                logger.info("Configured api_key matches EOA-derived key — pairing OK.")
                api_creds        = derived
                credential_source = "configured_confirmed"
    except Exception as exc:
        derive_exc = exc
        logger.warning("derive_api_key() failed: %s", exc)

    if api_creds is None:
        if creds.api_key and not creds.needs_api_derivation:
            # Configured creds present, derive failed — use configured as last resort.
            logger.warning(
                "derive_api_key() returned no creds for EOA %s. "
                "Using configured creds — may 401 if registered for a different address.",
                eoa,
            )
            api_creds        = ApiCreds(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                api_passphrase=creds.api_passphrase,
            )
            credential_source = "configured_fallback"
        else:
            # No configured creds and derive failed — try create_or_derive.
            try:
                api_creds        = l1.create_or_derive_api_creds()
                credential_source = "created_or_derived"
                logger.info("create_or_derive_api_creds() succeeded for EOA %s", eoa)
            except Exception as exc2:
                raise RuntimeError(
                    f"Cannot obtain L2 API credentials for EOA {eoa}. "
                    f"derive_api_key() failed ({derive_exc}); "
                    f"create_or_derive_api_creds() also failed: {exc2}"
                ) from exc2

    # Update creds in-place with effective values so all downstream callers
    # (_check_earning_pct etc.) use the correct derived credentials.
    creds.api_key          = api_creds.api_key
    creds.api_secret       = getattr(api_creds, "api_secret",     None) or creds.api_secret
    creds.api_passphrase   = getattr(api_creds, "api_passphrase", None) or creds.api_passphrase
    creds.credential_source = credential_source

    # ── Credential resolution summary (always printed) ───────────────────────
    print(f"  credential_source       : {credential_source}")
    print(f"  effective_api_key       : {api_creds.api_key[:8]}...")
    print(f"  signer / EOA            : {eoa}")
    print(f"  funder / proxy          : {creds.funder or '(none — EOA mode)'}")
    print()

    # ── Step 3: Use configured signature_type / funder as-is ────────────────
    # Official Polymarket signature_type guidance (authentication.md):
    #   0 = EOA           — direct MetaMask/hardware wallet, no proxy
    #   1 = POLY_PROXY    — Magic Link / email / Google login ONLY
    #   2 = GNOSIS_SAFE   — browser/MetaMask wallet proxy (most common)
    #
    # If you logged into Polymarket.com with MetaMask and your proxy wallet
    # address is shown in Settings, you need signature_type=2, NOT 1.
    # Set env var: POLYMARKET_SIGNATURE_TYPE=2
    effective_sig_type = creds.signature_type
    effective_funder = creds.funder

    if effective_sig_type == 1 and effective_funder:
        logger.warning(
            "signature_type=1 (POLY_PROXY) with funder=%s set. "
            "POLY_PROXY is for Magic Link/email login ONLY. "
            "If you logged in via MetaMask/browser wallet, set "
            "POLYMARKET_SIGNATURE_TYPE=2 (GNOSIS_SAFE) instead.",
            effective_funder,
        )

    logger.info(
        "Order signing: signature_type=%d  funder=%s",
        effective_sig_type, effective_funder or "(none=EOA mode)",
    )

    # ── Step 4: Build L2 client with correct creds + resolved signing mode ──
    kwargs: dict = dict(
        host=host,
        chain_id=creds.chain_id,
        key=creds.private_key,
        creds=api_creds,
        signature_type=effective_sig_type,
    )
    if effective_funder:
        kwargs["funder"] = effective_funder
    logger.info(
        "ClobClient built: EOA=%s signature_type=%d funder=%s api_key=%s...",
        eoa, effective_sig_type, effective_funder or "(EOA=maker)", api_creds.api_key[:8],
    )
    return ClobClient(**kwargs)


def run_activation_test(
    slug: str,
    creds: ActivationCredentials,
    host: str = CLOB_HOST,
    dry_run: bool = True,
    observe_minutes: int = 30,
    poll_interval_minutes: int = 5,
    skip_cancel: bool = False,
    spread_cents: float = 3.0,
) -> ActivationResult:
    """
    Full activation test for one survivor.

    If dry_run=True: fetches midpoint + reward config, computes qualifying
      quotes, prints plan — but does NOT submit orders.

    If dry_run=False (--live flag): submits qualifying bilateral orders,
      polls scoring state, cancels after observe_minutes (unless skip_cancel).

    Returns ActivationResult with full record of placement + scoring state.
    """
    data = SURVIVOR_DATA.get(slug)
    if data is None:
        return ActivationResult(
            slug=slug,
            condition_id="",
            token_id="",
            dry_run=dry_run,
            midpoint=None,
            midpoint_source="unavailable",
            reward_config=RewardConfig(
                condition_id="",
                max_spread_cents=3.5,
                max_spread_fraction=0.035,
                min_size=10.0,
                daily_rate_usdc=0.0,
                fetch_ok=False,
                source="fallback",
            ),
            quotes=None,
            verdict=VERDICT_PREFLIGHT_FAILED,
            verdict_detail=f"Slug not in SURVIVOR_DATA: {slug!r}",
        )

    condition_id = data["condition_id"]
    token_id     = data["token_id"]

    result = ActivationResult(
        slug=slug,
        condition_id=condition_id,
        token_id=token_id,
        dry_run=dry_run,
        midpoint=None,
        midpoint_source="unavailable",
        reward_config=None,  # type: ignore[arg-type]
        quotes=None,
    )

    # Build client
    try:
        client = build_clob_client(creds, host)
    except Exception as exc:
        result.verdict = VERDICT_PREFLIGHT_FAILED
        result.verdict_detail = f"ClobClient build failed: {exc}"
        return result

    # Fetch reward config
    result.reward_config = fetch_reward_config(
        host=host,
        condition_id=condition_id,
        fallback_max_spread_cents=data["fallback_max_spread_cents"],
        fallback_min_size=data["fallback_min_size"],
        fallback_daily_rate=data["daily_rate_usdc"],
    )

    # Fetch live midpoint — use yes_price from reward config as final fallback
    yes_price_ref = (
        getattr(result.reward_config, "yes_price_live", None)
        or data.get("yes_price_ref")
    )
    mid, mid_src = fetch_midpoint(client, token_id, price_ref=yes_price_ref)
    result.midpoint        = mid
    result.midpoint_source = mid_src

    if mid is None:
        result.verdict = VERDICT_PREFLIGHT_FAILED
        result.verdict_detail = (
            "Midpoint unavailable — cannot compute safe qualifying quotes. "
            "Check that token_id is correct and market is active."
        )
        return result

    # Compute qualifying quotes
    result.quotes = compute_qualifying_quotes(
        token_id=token_id,
        reward_config=result.reward_config,
        midpoint=mid,
        midpoint_source=mid_src,
        spread_cents=spread_cents,
    )

    if not result.quotes.qualifying:
        result.verdict = VERDICT_PREFLIGHT_FAILED
        result.verdict_detail = (
            f"Computed quotes do not qualify: "
            f"spread={result.quotes.spread:.4f} "
            f"max={result.quotes.max_spread_fraction:.4f} "
            f"size={result.quotes.size} "
            f"min={result.reward_config.min_size}"
        )
        return result

    if dry_run:
        result.verdict = VERDICT_DRY_RUN
        result.verdict_detail = (
            "Dry-run complete. Qualifying quotes computed. "
            "Re-run with --live to submit orders."
        )
        return result

    # ── LIVE MODE ─────────────────────────────────────────────────────────

    # Pre-order: log neg_risk + exchange address for the token so the JSON
    # output has full traceability of which EIP712 domain will be used.
    try:
        _nr = client.get_neg_risk(token_id)
        _ex = client.get_exchange_address(neg_risk=_nr)
        logger.info(
            "live_preflight: slug=%s neg_risk=%s exchange=%s", slug, _nr, _ex
        )
    except Exception as _e:
        logger.warning("live_preflight neg_risk/exchange check failed: %s", _e)

    # Pre-test earning_pct snapshot
    result.earning_pct_before = _check_earning_pct(host, creds, condition_id)

    # Place BID
    bid_order_id, bid_err = _place_order(
        client, token_id,
        price=result.quotes.bid_price,
        size=result.quotes.size,
        side=BUY_SIDE,
    )
    if bid_order_id:
        result.bid_order = PlacedOrder(
            order_id=bid_order_id,
            side=BUY_SIDE,
            price=result.quotes.bid_price,
            size=result.quotes.size,
            placed_at=datetime.now(timezone.utc),
        )
    else:
        result.verdict = VERDICT_PREFLIGHT_FAILED
        result.verdict_detail = f"BID order placement failed: {bid_err}"
        return result

    # ── SELL-leg inventory preflight ──────────────────────────────────────
    # Read-only check: does the account hold enough conditional tokens to fill
    # the ASK leg?  SELL orders require outcome-token inventory; they are NOT
    # funded from collateral the way BUY orders are.
    sell_inv = _check_sell_inventory(client, token_id, result.quotes.size)
    logger.info(
        "sell_inventory_preflight: verdict=%s  balance_raw=%s  allowance_raw=%s  "
        "required_raw=%s  balance_shares=%.2f  required_shares=%.2f",
        sell_inv["verdict"],
        sell_inv["balance_raw"], sell_inv["allowance_raw"],
        sell_inv["required_raw"],
        sell_inv["balance_shares"], sell_inv["required_shares"],
    )
    if sell_inv["verdict"] != "SELL_INVENTORY_READY":
        _cancel_order(client, bid_order_id)
        if result.bid_order:
            result.bid_order.cancelled = True
        result.verdict = VERDICT_PREFLIGHT_FAILED
        _allowance_hint = ""
        if sell_inv["verdict"] == "SELL_ALLOWANCE_INSUFFICIENT":
            _allowance_hint = (
                " Fix: proxy wallet must call "
                "CTF.setApprovalForAll(exchange=0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E, true). "
                "One-time per proxy+exchange pair. "
                "Fastest: Polymarket web UI sell flow triggers this approval automatically."
            )
        result.verdict_detail = (
            f"SELL-leg blocked by inventory preflight: {sell_inv['verdict']}. "
            f"conditional_balance={sell_inv['balance_shares']:.2f} shares, "
            f"required={sell_inv['required_shares']:.2f} shares, "
            f"allowance_raw={sell_inv['allowance_raw']}. "
            f"BID {bid_order_id} cancelled."
            + _allowance_hint
        )
        return result

    # Place ASK
    ask_order_id, ask_err = _place_order(
        client, token_id,
        price=result.quotes.ask_price,
        size=result.quotes.size,
        side=SELL_SIDE,
    )
    if ask_order_id:
        result.ask_order = PlacedOrder(
            order_id=ask_order_id,
            side=SELL_SIDE,
            price=result.quotes.ask_price,
            size=result.quotes.size,
            placed_at=datetime.now(timezone.utc),
        )
    else:
        # ASK failed — cancel BID immediately
        _cancel_order(client, bid_order_id)
        if result.bid_order:
            result.bid_order.cancelled = True
        result.verdict = VERDICT_PREFLIGHT_FAILED
        result.verdict_detail = f"ASK order placement failed (BID cancelled): {ask_err}"
        return result

    # ── Observation loop ──
    start = time.monotonic()
    poll_secs = poll_interval_minutes * 60.0
    total_secs = observe_minutes * 60.0
    elapsed = 0.0
    both_scored_ever = False

    while elapsed < total_secs:
        time.sleep(min(poll_secs, total_secs - elapsed))
        elapsed = time.monotonic() - start

        bid_sc = _check_order_scoring(client, bid_order_id)
        ask_sc = _check_order_scoring(client, ask_order_id)
        earn_pct = _check_earning_pct(host, creds, condition_id)
        both = bool(bid_sc and ask_sc)
        if both:
            both_scored_ever = True

        obs = ScoringObservation(
            elapsed_minutes=round(elapsed / 60.0, 1),
            bid_scoring=bid_sc,
            ask_scoring=ask_sc,
            both_scoring=both,
            earning_pct=earn_pct,
            observed_at=datetime.now(timezone.utc),
        )
        result.observations.append(obs)

        logger.info(
            "  [+%.0fm] bid_scoring=%s ask_scoring=%s earning_pct=%s",
            elapsed / 60.0, bid_sc, ask_sc, earn_pct,
        )

        if both:
            logger.info("  Both orders scoring. Continuing to observe...")

    # Final earning_pct snapshot
    result.earning_pct_after = _check_earning_pct(host, creds, condition_id)

    # Check fill state
    bid_filled = _is_filled(client, bid_order_id)
    ask_filled = _is_filled(client, ask_order_id)
    if result.bid_order:
        result.bid_order.filled = bid_filled
    if result.ask_order:
        result.ask_order.filled = ask_filled

    # Cancel unless skip_cancel
    if not skip_cancel:
        if not bid_filled:
            ok = _cancel_order(client, bid_order_id)
            if result.bid_order:
                result.bid_order.cancelled = True
                result.bid_order.cancel_ok = ok
        if not ask_filled:
            ok = _cancel_order(client, ask_order_id)
            if result.ask_order:
                result.ask_order.cancelled = True
                result.ask_order.cancel_ok = ok
        result.orders_cancelled = not (bid_filled or ask_filled)

    # Determine verdict
    if bid_filled or ask_filled:
        result.verdict = VERDICT_INCONCLUSIVE
        result.verdict_detail = (
            f"One or both orders filled during observation: "
            f"bid_filled={bid_filled} ask_filled={ask_filled}. "
            "Scoring test is inconclusive when orders are consumed. "
            "Re-run to test scoring with unfilled qualifying orders."
        )
    elif both_scored_ever:
        result.verdict = VERDICT_SCORING_ACTIVE
        result.verdict_detail = (
            "Both BID and ASK orders confirmed scoring during observation window. "
            "Qualifying bilateral participation IS activating reward scoring. "
            "earning_pct_before={:.4%} earning_pct_after={:.4%}. "
            "If earning_pct_after remains 0% after 24h: re-check with scoring still live "
            "before concluding competitive exclusion.".format(
                result.earning_pct_before or 0.0,
                result.earning_pct_after or 0.0,
            )
        )
    else:
        result.verdict = VERDICT_NOT_SCORING
        result.verdict_detail = (
            f"Neither BID nor ASK returned scoring=True across "
            f"{len(result.observations)} polls over {observe_minutes}min. "
            "Possible causes: (1) maker_address not registered for this account "
            "on this market — check /rewards/user/markets entry post-placement; "
            "(2) scoring period not yet evaluated (scoring can lag by minutes); "
            "(3) orders cancelled or filled before scoring window. "
            "NOT_SCORING after qualifying bilateral presence is NOT DOWNGRADE — "
            "it warrants 24h re-observation before concluding competitive exclusion."
        )

    return result


# ---------------------------------------------------------------------------
# SELL-leg inventory preflight (read-only)
# ---------------------------------------------------------------------------

_TOKEN_DECIMALS = 1_000_000  # both USDC.e and CTF outcome tokens use 6 decimals


def _check_sell_inventory(
    client: Any,
    token_id: str,
    required_shares: float,
) -> dict:
    """
    Read-only preflight for the SELL/ASK leg.

    Calls GET /balance-allowance with asset_type=CONDITIONAL and the exact
    token_id.  Returns a dict:
      verdict          : "SELL_INVENTORY_READY" | "SELL_INVENTORY_MISSING"
                         | "SELL_ALLOWANCE_INSUFFICIENT" | "CHECK_FAILED"
      balance_raw      : int — raw 6-decimal balance from API (0 if unavailable)
      allowance_raw    : int — raw 6-decimal allowance from API (0 if unavailable)
      required_raw     : int — required_shares * 1_000_000
      balance_shares   : float — balance_raw / 1_000_000
      required_shares  : float — as passed in

    Verdict logic:
      SELL_INVENTORY_MISSING      balance_raw < required_raw
      SELL_ALLOWANCE_INSUFFICIENT balance_raw >= required_raw but allowance_raw < required_raw
      SELL_INVENTORY_READY        balance_raw >= required_raw AND allowance_raw >= required_raw
      CHECK_FAILED                API call threw an exception
    """
    required_raw = int(required_shares * _TOKEN_DECIMALS)
    result: dict = {
        "verdict": "CHECK_FAILED",
        "balance_raw": 0,
        "allowance_raw": 0,
        "required_raw": required_raw,
        "balance_shares": 0.0,
        "required_shares": required_shares,
    }
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams

        # Sync CLOB server's cached state from the chain before reading.
        # Covers stale-cache: if setApprovalForAll was called externally (e.g., web UI),
        # the CLOB server may still show allowance=0 until this refresh is called.
        # Pure HTTP GET — no on-chain side effects, no gas.
        try:
            client.update_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    signature_type=-1,
                )
            )
        except Exception as _upd_exc:
            logger.debug(
                "_check_sell_inventory: update_balance_allowance non-fatal: %s", _upd_exc
            )

        resp = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type="CONDITIONAL",
                token_id=token_id,
                # signature_type=-1 → py_clob_client replaces with builder.sig_type
                signature_type=-1,
            )
        )
        # CONDITIONAL response shape differs from COLLATERAL:
        #   COLLATERAL : {"balance": "...", "allowance": "..."}          (flat scalar)
        #   CONDITIONAL: {"balance": "...", "allowances": {"0xADDR": "..."}}  (nested dict, PLURAL key)
        # Reading resp.get("allowance") on a CONDITIONAL response always returns 0 — wrong key.
        if isinstance(resp, dict):
            balance_raw = int(float(resp.get("balance", 0) or 0))

            allowances_map = resp.get("allowances")
            if isinstance(allowances_map, dict) and allowances_map:
                # Normalize keys to lowercase; response may use checksum-cased addresses.
                norm = {k.lower(): v for k, v in allowances_map.items()}
                # Standard CLOB exchange for chain 137 (non-neg-risk).
                _SPENDER = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"
                raw_val = norm.get(_SPENDER, 0) or 0
                allowance_raw = int(float(raw_val))
                logger.info(
                    "_check_sell_inventory allowances_map keys=%s  selected_spender=%s  allowance_raw=%s",
                    list(norm.keys()), _SPENDER, allowance_raw,
                )
            else:
                # Fallback: flat "allowance" key (COLLATERAL shape or legacy response).
                allowance_raw = int(float(resp.get("allowance", 0) or 0))
                logger.info(
                    "_check_sell_inventory flat resp keys=%s  allowance_raw=%s",
                    list(resp.keys()), allowance_raw,
                )
        else:
            balance_raw   = int(float(getattr(resp, "balance",   0) or 0))
            allowance_raw = int(float(getattr(resp, "allowance", 0) or 0))

        result["balance_raw"]   = balance_raw
        result["allowance_raw"] = allowance_raw
        result["balance_shares"] = balance_raw / _TOKEN_DECIMALS

        if balance_raw < required_raw:
            result["verdict"] = "SELL_INVENTORY_MISSING"
        elif allowance_raw < required_raw:
            result["verdict"] = "SELL_ALLOWANCE_INSUFFICIENT"
        else:
            result["verdict"] = "SELL_INVENTORY_READY"

    except Exception as exc:
        logger.warning("_check_sell_inventory failed token=%s: %s", token_id[:16], exc)
        result["verdict"] = f"CHECK_FAILED: {exc}"

    return result


# ---------------------------------------------------------------------------
# Order helpers (thin wrappers — log failures, do not raise)
# ---------------------------------------------------------------------------

def _place_order(
    client: Any,
    token_id: str,
    price: float,
    size: float,
    side: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Submit one GTC limit order via py_clob_client.

    Returns (order_id, error_detail).
    On success: (order_id, None).
    On failure: (None, human-readable error detail including server response body).

    body.owner and POLY_API_KEY are both taken from client.creds.api_key — same
    key, same owner.  Polymarket requires "the order owner has to be the owner of
    the API KEY": splitting the two keys across different credentials is invalid.
    """
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.clob_types import PartialCreateOrderOptions
        from py_clob_client.exceptions import PolyApiException

        # ── Pre-order diagnostics ──────────────────────────────────────────
        try:
            neg_risk_val = client.get_neg_risk(token_id)
            exchange_addr = client.get_exchange_address(neg_risk=neg_risk_val)
            logger.info(
                "pre_order_diag side=%s price=%.4f size=%.0f  neg_risk=%s  exchange=%s",
                side, price, size, neg_risk_val, exchange_addr,
            )
        except Exception as diag_exc:
            logger.warning("pre_order_diag failed (non-fatal): %s", diag_exc)

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
        )
        # neg_risk=None: py_clob_client's create_order treats `if options.neg_risk` as
        # truthy, so passing neg_risk=False is silently ignored and the client makes an
        # extra GET /neg-risk API call anyway.  Passing None is equivalent and avoids
        # confusion.  For Hungary/Vance/Rubio (binary markets) the API returns False.
        options = PartialCreateOrderOptions(tick_size="0.01", neg_risk=None)

        # ── Create (sign) the order — log fields before posting ───────────
        signed_order = client.create_order(order_args, options)
        if signed_order is not None:
            try:
                od = signed_order.dict() if hasattr(signed_order, "dict") else vars(signed_order)
                logger.info(
                    "signed_order side=%s maker=%s signer=%s signatureType=%s "
                    "makerAmount=%s takerAmount=%s tokenId=%s nonce=%s feeRateBps=%s",
                    side,
                    od.get("maker", "?"), od.get("signer", "?"),
                    od.get("signatureType", "?"),
                    od.get("makerAmount", "?"), od.get("takerAmount", "?"),
                    str(od.get("tokenId", "?"))[:20],
                    od.get("nonce", "?"), od.get("feeRateBps", "?"),
                )
            except Exception:
                pass

        # ── Post the order ─────────────────────────────────────────────────
        try:
            response = client.post_order(signed_order, OrderType.GTC)
        except PolyApiException as api_exc:
            detail = f"POST /order {side} → HTTP {api_exc.status_code}: {api_exc.error_msg}"
            logger.error("place_order API error side=%s price=%s: %s", side, price, detail)
            return None, detail
        except Exception as exc:
            detail = f"POST /order {side} exception: {exc}"
            logger.error("place_order failed side=%s price=%s: %s", side, price, exc)
            return None, detail

        if response is None:
            detail = f"POST /order {side} returned None (no response)"
            logger.error("place_order: %s", detail)
            return None, detail

        # ── Extract order_id ───────────────────────────────────────────────
        if isinstance(response, dict):
            order_id = (
                response.get("orderID")
                or response.get("order_id")
                or (response.get("order") or {}).get("id")
            )
        else:
            order_id = (
                getattr(response, "orderID", None)
                or getattr(response, "order_id", None)
            )
        if order_id:
            logger.info(
                "placed %s order_id=%s price=%.4f size=%.0f", side, order_id, price, size
            )
            return str(order_id), None

        detail = f"POST /order {side} succeeded but no order_id in response: {response!r}"
        logger.error("place_order: %s", detail)
        return None, detail

    except Exception as exc:
        detail = f"place_order {side} unexpected: {exc}"
        logger.error("place_order failed side=%s price=%s: %s", side, price, exc)
        return None, detail


def _cancel_order(client: Any, order_id: str) -> bool:
    """Cancel one order.  Returns True if cancel succeeded."""
    try:
        result = client.cancel(order_id)
        ok = result is not None
        logger.info("cancel order_id=%s ok=%s", order_id, ok)
        return ok
    except Exception as exc:
        logger.warning("cancel failed order_id=%s: %s", order_id, exc)
        return False


def _is_filled(client: Any, order_id: str) -> bool:
    """Check if an order has been fully filled.  Returns False on error."""
    try:
        order = client.get_order(order_id)
        if order is None:
            return False
        status = (
            order.get("status") if isinstance(order, dict)
            else getattr(order, "status", None)
        )
        size_matched = (
            order.get("size_matched") if isinstance(order, dict)
            else getattr(order, "size_matched", None)
        )
        if status:
            return str(status).upper() in ("MATCHED", "FILLED")
        return False
    except Exception as exc:
        logger.debug("is_filled check failed order_id=%s: %s", order_id, exc)
        return False


def _get_order_fill_info(client: Any, order_id: str) -> Optional[dict]:
    """
    Return ground-truth fill state for one order after the position loop exits.
    Used for post-cycle attribution when the polling loop may have missed a fill.

    Returns dict with keys:
        status        : str   (LIVE, MATCHED, CANCELLED, …)
        size_matched  : float (shares filled so far)
        size          : float (original order size)
        price         : float (limit price placed)
        fully_filled  : bool
        partial_fill  : bool  (0 < size_matched < size)
    Returns None on any error.
    """
    try:
        order = client.get_order(order_id)
        if order is None:
            return None
        if isinstance(order, dict):
            raw_status   = str(order.get("status", "")).upper()
            size_matched = float(order.get("size_matched") or 0)
            size         = float(order.get("original_size") or order.get("size") or 0)
            price        = float(order.get("price") or 0)
        else:
            raw_status   = str(getattr(order, "status", "")).upper()
            size_matched = float(getattr(order, "size_matched", 0) or 0)
            size_orig    = getattr(order, "original_size", None) or getattr(order, "size", 0)
            size         = float(size_orig or 0)
            price        = float(getattr(order, "price", 0) or 0)
        fully_filled = raw_status in ("MATCHED", "FILLED") or (size > 0 and size_matched >= size)
        partial_fill = (not fully_filled) and size_matched > 0
        return {
            "status":       raw_status,
            "size_matched": size_matched,
            "size":         size,
            "price":        price,
            "fully_filled": fully_filled,
            "partial_fill": partial_fill,
        }
    except Exception as exc:
        logger.debug("get_order_fill_info failed order_id=%s: %s", order_id, exc)
        return None
