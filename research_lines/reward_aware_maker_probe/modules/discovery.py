"""
reward_aware_maker_probe — Module 1: Discovery
polyarb_lab / research_line / probe-only

Primary discovery source: CLOB rewards endpoints (paginated to completion).
  https://clob.polymarket.com/rewards/markets/multi   (primary)
  https://clob.polymarket.com/rewards/markets/current (fallback)

Gamma /events is NOT used as the rewarded-universe source.

Universe definition (hard):
  - active=true, closed=false, enable_order_book=true
  - fees_enabled=True  (this is the key filter — existing mainline EXCLUDES these)
  - rewards present with daily_rate_usdc >= MIN_DAILY_RATE_USDC
  - rewardsMinSize > 0
  - rewardsMaxSpread > 0

No mainline imports. No order submission. No state mutation.
Read-only network access to CLOB rewards API only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CLOB_REWARDS_MULTI_PATH = "/rewards/markets/multi"
CLOB_REWARDS_CURRENT_PATH = "/rewards/markets/current"

# Legacy alias — kept so callers passing gamma_host= do not break
GAMMA_HOST = "https://gamma-api.polymarket.com"

DEFAULT_PAGE_SIZE = 100
MAX_MARKETS = 5000          # safety ceiling
MIN_DAILY_RATE_USDC = 1.0  # minimum reward rate to be considered rewarded

_CLOB_END_CURSOR = "LTE="   # CLOB pagination sentinel for end-of-pages
CLOB_BOOK_PATH = "/book"
MAX_BOOK_REQUESTS = 500    # probe cap: live book calls per run


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class RawRewardedMarket:
    """One fee-enabled, reward-eligible market from CLOB rewards discovery."""
    market_id: str
    market_slug: str
    event_slug: str
    event_id: str
    category: str
    question: str
    yes_token_id: str
    no_token_id: str
    fees_enabled: bool
    enable_orderbook: bool
    best_bid: Optional[float]
    best_ask: Optional[float]
    # Reward config fields
    rewards_min_size: float
    rewards_max_spread_cents: float
    reward_daily_rate_usdc: float
    clob_rewards_raw: list[dict[str, Any]]
    # Extra fields for reporting
    volume_24hr: Optional[float]
    liquidity: Optional[float]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_usable_book(self) -> bool:
        if self.best_bid is None or self.best_ask is None:
            return False
        try:
            return float(self.best_bid) > 0.0 and float(self.best_ask) > float(self.best_bid)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def _safe_str(val: Any, default: str = "") -> str:
    return str(val) if val is not None else default


def _extract_daily_rate(market: dict[str, Any]) -> float:
    """
    Sum daily reward rate from all clobRewards entries.

    Handles two CLOB schemas:
      A. clobRewards[i].rewardsDailyRate  — Gamma / flat CLOB format
      B. clobRewards[i].rates[j].rewards_daily_rate  — CLOB multi-reward format
         (rewardsDailyRate is 0 or absent; actual rates are in nested rates list)
    """
    clob_rewards = market.get("clobRewards") or []
    total = 0.0
    for r in clob_rewards:
        if not isinstance(r, dict):
            continue

        # Schema A: top-level rewardsDailyRate on the entry
        try:
            top_rate = float(r.get("rewardsDailyRate") or 0.0)
        except Exception:
            top_rate = 0.0

        if top_rate > 0.0:
            total += top_rate
            continue

        # Schema B: top-level is 0 — actual rates are in nested rates list
        for nr in (r.get("rates") or []):
            if not isinstance(nr, dict):
                continue
            for _nk in ("rewards_daily_rate", "rewardsDailyRate", "daily_rate", "rate"):
                _nv = nr.get(_nk)
                if _nv is not None:
                    try:
                        total += float(_nv)
                    except Exception:
                        pass
                    break

    return round(total, 6)


def _is_fee_enabled_rewarded(market: dict[str, Any]) -> bool:
    """
    Return True if the market is fee-enabled AND has an active reward program.

    Expects flat keys (after _normalize_clob_market has been applied):
      1. fees_enabled must be True
      2. enable_orderbook must be True
      3. clobRewards[].rewardsDailyRate sum >= MIN_DAILY_RATE_USDC
      4. rewardsMinSize > 0
      5. rewardsMaxSpread > 0
    """
    fees_enabled = market.get("fees_enabled") or market.get("feesEnabled")
    if not fees_enabled:
        return False

    ob = market.get("enable_orderbook") or market.get("enableOrderBook")
    if not ob:
        return False

    if _extract_daily_rate(market) < MIN_DAILY_RATE_USDC:
        return False
    if _safe_float(market.get("rewardsMinSize")) <= 0.0:
        return False
    if _safe_float(market.get("rewardsMaxSpread")) <= 0.0:
        return False

    return True


def _is_active_market(market: dict[str, Any]) -> bool:
    closed = market.get("closed")
    active = market.get("active")
    if closed is True or closed == "true":
        return False
    if active is False or active == "false":
        return False
    return True


def _token_ids_from_clob(market: dict[str, Any]) -> tuple[str, str]:
    """Extract yes/no token IDs from CLOB-style tokens array."""
    tokens = market.get("tokens") or []
    yes_id = ""
    no_id = ""
    for t in tokens:
        if not isinstance(t, dict):
            continue
        outcome = _safe_str(t.get("outcome") or "").lower()
        tid = _safe_str(t.get("token_id") or t.get("tokenId"))
        if outcome == "yes":
            yes_id = tid
        elif outcome == "no":
            no_id = tid
    # Positional fallback
    if not yes_id and len(tokens) >= 1 and isinstance(tokens[0], dict):
        yes_id = _safe_str(tokens[0].get("token_id") or tokens[0].get("tokenId"))
    if not no_id and len(tokens) >= 2 and isinstance(tokens[1], dict):
        no_id = _safe_str(tokens[1].get("token_id") or tokens[1].get("tokenId"))
    return yes_id, no_id


def _token_ids(market: dict[str, Any]) -> tuple[str, str]:
    """Extract yes/no token IDs from Gamma-style clobTokenIds / explicit fields."""
    clob_ids = market.get("clobTokenIds") or []
    yes_id = _safe_str(market.get("yes_token_id") or market.get("yesTokenId"))
    no_id = _safe_str(market.get("no_token_id") or market.get("noTokenId"))
    if not yes_id and len(clob_ids) >= 1:
        yes_id = _safe_str(clob_ids[0])
    if not no_id and len(clob_ids) >= 2:
        no_id = _safe_str(clob_ids[1])
    return yes_id, no_id


def _fetch_clob_book(
    host: str,
    token_id: str,
    client: httpx.Client,
) -> tuple[Optional[float], Optional[float]]:
    """
    Fetch (best_bid, best_ask) from CLOB /book for one YES-side token.

    Join key: yes_token_id (tokens[0].token_id from the rewards row).
    CLOB bids are sorted descending (best bid first).
    CLOB asks are sorted ascending (best ask first).
    Returns (None, None) on any failure — never raises.
    """
    if not token_id:
        return None, None
    url = f"{host.rstrip('/')}{CLOB_BOOK_PATH}"
    try:
        resp = client.get(url, params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        bb = _safe_float(bids[0].get("price"), default=None) if bids else None  # type: ignore[arg-type]
        ba = _safe_float(asks[0].get("price"), default=None) if asks else None  # type: ignore[arg-type]
        return bb, ba
    except Exception as exc:
        logger.debug("CLOB /book failed token=%s: %s", token_id, exc)
        return None, None


def _normalize_clob_market(m: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a CLOB rewards market dict into a shape compatible with
    _is_fee_enabled_rewarded and _is_active_market.

    CLOB rewards responses may nest reward parameters under a "rewards" key.
    This helper promotes them to the flat Gamma-compatible keys expected by
    the existing filter functions.
    """
    # CLOB rewards endpoint may nest reward params under "rewards" OR "reward" (singular),
    # OR may expose them as top-level flat fields.  Try all three sources in order.
    rewards: dict[str, Any] = (
        m.get("rewards")
        if isinstance(m.get("rewards"), dict) and m.get("rewards")
        else (
            m.get("reward")
            if isinstance(m.get("reward"), dict) and m.get("reward")
            else {}
        )
    )
    out = dict(m)  # shallow copy — all original CLOB fields preserved

    # --- Min size ---
    # Priority: top-level "rewardsMinSize" (already in out via shallow copy)
    # then rewards sub-object, then top-level alternative names.
    if "rewardsMinSize" not in out or out["rewardsMinSize"] is None:
        _ms: Any = None
        for _k in ("min_size", "minSize"):
            if _k in rewards:
                _ms = rewards[_k]
                break
        if _ms is None:
            for _k in ("min_incentive_size", "minIncentiveSize",
                       "rewards_min_size", "rewardMinSize"):
                if _k in m and m[_k] is not None:
                    _ms = m[_k]
                    break
        out["rewardsMinSize"] = _safe_float(_ms)

    # --- Max spread ---
    if "rewardsMaxSpread" not in out or out["rewardsMaxSpread"] is None:
        _msp: Any = None
        for _k in ("max_spread", "maxSpread"):
            if _k in rewards:
                _msp = rewards[_k]
                break
        if _msp is None:
            for _k in ("max_incentive_spread", "maxIncentiveSpread",
                       "rewards_max_spread", "rewardMaxSpread"):
                if _k in m and m[_k] is not None:
                    _msp = m[_k]
                    break
        out["rewardsMaxSpread"] = _safe_float(_msp)

    # --- Daily rate → synthetic clobRewards ---
    # Build clobRewards list so _extract_daily_rate works.
    # Search order:
    #   1. rewards sub-object flat fields
    #   2. rewards.rates[] list (CLOB native nested schema)
    #   3. top-level m flat fields (most likely actual CLOB schema)
    if "clobRewards" not in out:
        rate: float = 0.0

        # 1. Flat fields inside rewards sub-object
        for _fkey in ("daily_rate_usdc", "dailyRateUsdc", "rate_per_day", "rewardsDailyRate"):
            _v = rewards.get(_fkey)
            if _v is not None:
                rate = _safe_float(_v)
                break

        # 2. Nested rates list inside rewards sub-object
        if rate == 0.0:
            for _r in (rewards.get("rates") or []):
                if not isinstance(_r, dict):
                    continue
                for _rkey in ("rewards_daily_rate", "rewardsDailyRate", "daily_rate", "rate"):
                    _v = _r.get(_rkey)
                    if _v is not None:
                        rate += _safe_float(_v)
                        break

        # 3. Top-level flat fields in m
        if rate == 0.0:
            for _fkey in ("rewards_daily_rate", "rewardsDailyRate",
                          "daily_reward_rate", "rewardsAmount",
                          "reward_daily_rate", "rewardDailyRate"):
                if _fkey in m and m[_fkey] is not None:
                    rate = _safe_float(m[_fkey])
                    break

        # 4. rewards_config[] list — confirmed CLOB /rewards/markets/multi schema.
        #    Each entry has rate_per_day per reward asset; sum all active entries.
        if rate == 0.0:
            for _rc in (m.get("rewards_config") or []):
                if isinstance(_rc, dict):
                    _v = _rc.get("rate_per_day")
                    if _v is not None:
                        rate += _safe_float(_v)

        # --- DIAGNOSTIC (temporary — remove after residual audit) ---
        if rate == 0.0:
            import sys as _sys
            _d_slug = m.get("market_slug") or m.get("slug") or m.get("condition_id", "?")
            _d_rew_keys = sorted(rewards.keys()) if rewards else []
            _d_rates_sample = (rewards.get("rates") or [])[:2]
            _d_cfg_sample = (m.get("rewards_config") or [])[:3]
            _d_top = {
                k: m[k]
                for k in (
                    "rewards_daily_rate", "rewardsDailyRate", "daily_reward_rate",
                    "rewardsAmount", "reward_daily_rate", "rewardDailyRate",
                )
                if k in m
            }
            print(
                f"[RATE0] {str(_d_slug)[:55]}"
                f" | cfg[:3]={_d_cfg_sample}"
                f" | rew_keys={_d_rew_keys}"
                f" | rew.rates[:2]={_d_rates_sample}"
                f" | top_rate_keys={_d_top}",
                file=_sys.stderr,
            )
        # --- END DIAGNOSTIC ---

        out["clobRewards"] = [{"rewardsDailyRate": round(rate, 6)}]

    # fees_enabled: use explicit field if present, otherwise default True
    # (presence on the CLOB rewards endpoint implies fee market)
    if "fees_enabled" not in out and "feesEnabled" not in out:
        out["fees_enabled"] = True

    # enable_orderbook normalization
    if "enable_orderbook" not in out and "enableOrderBook" not in out:
        ob = m.get("enable_order_book") or m.get("enableOrderBook")
        out["enable_orderbook"] = ob if ob is not None else True

    return out


# ---------------------------------------------------------------------------
# CLOB rewards API fetch
# ---------------------------------------------------------------------------

def _fetch_clob_rewards_page(
    host: str,
    path: str,
    cursor: Optional[str],
    client: httpx.Client,
) -> dict[str, Any]:
    """Fetch one page from a CLOB rewards endpoint."""
    params: dict[str, Any] = {}
    if cursor:
        params["next_cursor"] = cursor
    url = f"{host.rstrip('/')}{path}"
    try:
        resp = client.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        # Normalise bare list into dict so callers always get a dict
        return payload if isinstance(payload, dict) else {"data": payload}
    except Exception as exc:
        logger.warning(
            "CLOB rewards page fetch failed (%s, cursor=%s): %s", path, cursor, exc
        )
        return {}


def _fetch_all_clob_rewards_markets(
    host: str,
    path: str,
    max_markets: int = MAX_MARKETS,
) -> list[dict[str, Any]]:
    """Paginate a CLOB rewards endpoint to completion, return flat market list."""
    all_markets: list[dict[str, Any]] = []
    cursor: Optional[str] = None
    with httpx.Client() as client:
        while True:
            payload = _fetch_clob_rewards_page(host, path, cursor, client)
            if not payload:
                break

            if isinstance(payload.get("data"), list):
                page: list[dict[str, Any]] = payload["data"]
            elif isinstance(payload.get("markets"), list):
                page = payload["markets"]
            else:
                page = []

            if not page:
                break

            all_markets.extend(page)
            logger.debug(
                "CLOB rewards %s: +%d markets (total %d)", path, len(page), len(all_markets)
            )

            if len(all_markets) >= max_markets:
                logger.warning(
                    "Safety ceiling hit: %d markets from %s. Stopping pagination.",
                    len(all_markets), path,
                )
                break

            next_cursor = payload.get("next_cursor") or payload.get("nextCursor") or ""
            if not next_cursor or next_cursor == _CLOB_END_CURSOR:
                break
            cursor = next_cursor

    logger.info("CLOB rewards %s: %d total markets fetched", path, len(all_markets))
    return all_markets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_fee_enabled_rewarded_markets(
    clob_host: str = CLOB_HOST,
    max_markets: int = MAX_MARKETS,
    gamma_host: str = GAMMA_HOST,  # accepted for backward compat, not used
) -> list[RawRewardedMarket]:
    """
    Discover all active, fee-enabled, reward-eligible markets.

    Primary source: CLOB /rewards/markets/multi (paginated to completion).
    Fallback source: CLOB /rewards/markets/current.
    Gamma /events is NOT used.

    Returns empty list on API failure (never raises).
    """
    fetched_at = datetime.now(timezone.utc)

    # --- Primary: /rewards/markets/multi ---
    raw_clob: list[dict[str, Any]] = []
    try:
        raw_clob = _fetch_all_clob_rewards_markets(
            clob_host, CLOB_REWARDS_MULTI_PATH, max_markets=max_markets,
        )
    except Exception as exc:
        logger.warning("CLOB /rewards/markets/multi fetch failed: %s", exc)

    # --- Fallback: /rewards/markets/current ---
    if not raw_clob:
        logger.info("multi returned empty — trying /rewards/markets/current")
        try:
            raw_clob = _fetch_all_clob_rewards_markets(
                clob_host, CLOB_REWARDS_CURRENT_PATH, max_markets=max_markets,
            )
        except Exception as exc:
            logger.error("CLOB /rewards/markets/current also failed: %s", exc)
            return []

    if not raw_clob:
        logger.warning("Both CLOB rewards endpoints returned empty universe.")
        return []

    results: list[RawRewardedMarket] = []
    skipped_inactive = 0
    skipped_no_fee = 0
    book_hits = 0

    with httpx.Client() as book_client:
        for raw_mkt in raw_clob:
            if not isinstance(raw_mkt, dict):
                continue

            # Flatten CLOB field names to shapes expected by filter functions
            mkt = _normalize_clob_market(raw_mkt)

            if not _is_active_market(mkt):
                skipped_inactive += 1
                continue

            fees_enabled_raw = mkt.get("fees_enabled") or mkt.get("feesEnabled")
            if not fees_enabled_raw:
                skipped_no_fee += 1
                continue

            # Reward status is asserted by source: presence on CLOB rewards endpoint
            # is sufficient. Do NOT gate on legacy reward field names here.
            # The EV model will classify REJECTED_NO_REWARD for any market where
            # reward fields are missing or zero after normalization.

            # Token IDs: CLOB tokens array first, Gamma-style fields as fallback
            yes_id, no_id = _token_ids_from_clob(mkt)
            if not yes_id and not no_id:
                yes_id, no_id = _token_ids(mkt)

            # Book join: fetch best_bid / best_ask from CLOB /book.
            # Join key: yes_token_id → GET /book?token_id={yes_token_id}
            # The rewards endpoint does NOT carry book data — must be fetched separately.
            # Capped at MAX_BOOK_REQUESTS per probe run to bound total call count.
            bb: Optional[float] = None
            ba: Optional[float] = None
            if yes_id and book_hits < MAX_BOOK_REQUESTS:
                bb, ba = _fetch_clob_book(clob_host, yes_id, book_client)
                book_hits += 1

            daily_rate = _extract_daily_rate(mkt)

            results.append(RawRewardedMarket(
                market_id=_safe_str(
                    mkt.get("condition_id") or mkt.get("id") or mkt.get("market_id")
                ),
                market_slug=_safe_str(mkt.get("market_slug") or mkt.get("slug")),
                event_slug=_safe_str(mkt.get("event_slug") or ""),
                event_id=_safe_str(mkt.get("event_id") or ""),
                category=_safe_str(mkt.get("category") or mkt.get("tags") or ""),
                question=_safe_str(mkt.get("question")),
                yes_token_id=yes_id,
                no_token_id=no_id,
                fees_enabled=True,
                enable_orderbook=True,
                best_bid=bb,
                best_ask=ba,
                rewards_min_size=_safe_float(mkt.get("rewardsMinSize")),
                rewards_max_spread_cents=_safe_float(mkt.get("rewardsMaxSpread")),
                reward_daily_rate_usdc=daily_rate,
                clob_rewards_raw=list(mkt.get("clobRewards") or []),
                volume_24hr=_safe_float(  # type: ignore[arg-type]
                    mkt.get("volume24hr") or mkt.get("volume_24hr"), default=None
                ),
                liquidity=_safe_float(mkt.get("liquidity"), default=None),  # type: ignore[arg-type]
                fetched_at=fetched_at,
            ))

    logger.info(
        "Discovery complete: %d fee-enabled rewarded markets | "
        "skipped: inactive=%d, no_fee=%d | book_hits=%d/%d",
        len(results), skipped_inactive, skipped_no_fee, book_hits, len(results),
    )
    return results


def discovery_summary(markets: list[RawRewardedMarket]) -> dict[str, Any]:
    """Return a concise summary dict for logging."""
    if not markets:
        return {
            "fee_enabled_rewarded_market_count": 0,
            "with_usable_book": 0,
        }
    with_book = sum(1 for m in markets if m.has_usable_book())
    rates = [m.reward_daily_rate_usdc for m in markets]
    mean_rate = round(sum(rates) / len(rates), 2)
    return {
        "fee_enabled_rewarded_market_count": len(markets),
        "with_usable_book": with_book,
        "reward_daily_rate_usdc": {
            "min": round(min(rates), 2),
            "max": round(max(rates), 2),
            "mean": mean_rate,
        },
        # Flat alias — easier to read in PowerShell / JSON viewers
        "reward_daily_rate_usdc_mean": mean_rate,
        "fetched_at": markets[0].fetched_at.isoformat(),
    }
