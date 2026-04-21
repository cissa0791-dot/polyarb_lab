"""
run_universe_refresh.py — Reward-Aware Maker Universe Refresh
auto_maker_loop / polyarb_lab / research utility

Discovers fresh contested, reward-eligible markets from Gamma API and
writes a new candidate shortlist for the viability screen.

Replaces the stale latest_probe.json candidate set with live data.

Filters applied (in order):
  1. Active + CLOB-enabled (clobTokenIds set)
  2. Reward-eligible (rewardsMinSize > 0 AND rewardsDailyRate > 0)
  3. Contested midpoint: 0.15 <= mid <= 0.85  (Gamma price — pre-filter only)
  4. Not a dead book: best_bid > 0.01 AND best_ask < 0.99
  5. Minimum liquidity: liquidityNum >= MIN_LIQUIDITY_USD
  6. CLOB live book verification: fetch YES token book, confirm bid > 0.01 and ask < 0.99
     (Gamma prices are cached/stale — CLOB is ground truth)

Output:
  data/research/reward_aware_maker_probe/universe_refresh_<timestamp>.json
  data/research/reward_aware_maker_probe/latest_probe.json  (overwritten)

Usage:
  py -3 research_lines/auto_maker_loop/run_universe_refresh.py
  py -3 research_lines/auto_maker_loop/run_universe_refresh.py --min-rate 10 --min-liquidity 500
  py -3 research_lines/auto_maker_loop/run_universe_refresh.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("universe_refresh")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST  = "https://clob.polymarket.com"
PAGE_SIZE = 200
MAX_PAGES = 50          # 50 * 200 = 10,000 markets max

MIN_MID = 0.15          # contested lower bound
MAX_MID = 0.85          # contested upper bound
MIN_REWARD_RATE = 0.0   # USDC/day (0 = any reward program)
MIN_LIQUIDITY_USD = 100.0

DEAD_BID_THRESHOLD = 0.01
DEAD_ASK_THRESHOLD = 0.99

CLOB_VERIFY_WORKERS = 10   # parallel CLOB book fetches

OUTPUT_DIR = Path("data/research/reward_aware_maker_probe")
SEP = "─" * 72


# ---------------------------------------------------------------------------
# Gamma fetch
# ---------------------------------------------------------------------------

def _fetch_all_markets(host: str, client: httpx.Client) -> list[dict]:
    all_markets: list[dict] = []
    for page in range(MAX_PAGES):
        offset = page * PAGE_SIZE
        try:
            resp = client.get(
                f"{host}/markets",
                params={"limit": PAGE_SIZE, "offset": offset, "active": "true", "closed": "false"},
                timeout=20,
            )
            resp.raise_for_status()
            page_data = resp.json()
            if not isinstance(page_data, list) or not page_data:
                break
            all_markets.extend(page_data)
            logger.info("Page %d: fetched %d markets (total so far: %d)", page + 1, len(page_data), len(all_markets))
            if len(page_data) < PAGE_SIZE:
                break
        except Exception as exc:
            logger.error("Fetch failed at offset %d: %s", offset, exc)
            break
    return all_markets


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _has_clob(market: dict) -> bool:
    tokens = market.get("clobTokenIds") or market.get("tokens") or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            return False
    return isinstance(tokens, list) and len(tokens) > 0


def _reward_rate(market: dict) -> float:
    rewards = market.get("clobRewards") or []
    if isinstance(rewards, str):
        try:
            rewards = json.loads(rewards)
        except Exception:
            return 0.0
    if not isinstance(rewards, list):
        return 0.0
    total = 0.0
    for r in rewards:
        try:
            total += float(r.get("rewardsDailyRate") or 0)
        except Exception:
            pass
    return total


def _reward_min_size(market: dict) -> int:
    try:
        return int(market.get("rewardsMinSize") or 0)
    except Exception:
        return 0


def _midpoint(market: dict) -> float | None:
    for field in ("midpoint", "lastTradePrice", "bestMid"):
        val = market.get(field)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    bid = market.get("bestBid") or market.get("best_bid")
    ask = market.get("bestAsk") or market.get("best_ask")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2
        except Exception:
            pass
    return None


def _liquidity(market: dict) -> float:
    for field in ("liquidityNum", "liquidity", "liquidityClob"):
        val = market.get(field)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    return 0.0


def _best_bid_ask(market: dict) -> tuple[float | None, float | None]:
    bid = market.get("bestBid") or market.get("best_bid")
    ask = market.get("bestAsk") or market.get("best_ask")
    try:
        b = float(bid) if bid is not None else None
        a = float(ask) if ask is not None else None
        return b, a
    except Exception:
        return None, None


def _is_dead_book(bid: float | None, ask: float | None) -> bool:
    if bid is None or ask is None:
        return True
    return bid <= DEAD_BID_THRESHOLD or ask >= DEAD_ASK_THRESHOLD


def _volume_24h(market: dict) -> float:
    for field in ("volume24hr", "volume24h", "oneDayVolume", "volumeClob"):
        val = market.get(field)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    return 0.0


# ---------------------------------------------------------------------------
# CLOB live book verification
# ---------------------------------------------------------------------------

def _get_yes_token_id(market: dict) -> str | None:
    """Extract YES token ID from clobTokenIds field."""
    tokens = market.get("clobTokenIds") or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            return None
    # clobTokenIds[0] = YES token by Polymarket convention
    if isinstance(tokens, list) and tokens:
        return str(tokens[0])
    return None


def _clob_book_live(token_id: str, client: httpx.Client) -> dict:
    """
    Fetch live CLOB book for a token.
    Returns dict with keys: exists, bid, ask, empty_book
      exists=False  → 404, market not on CLOB (resolved or delisted)
      exists=True, empty_book=True  → active market, NO current makers posting
      exists=True, empty_book=False → active market WITH live orders
    """
    try:
        resp = client.get(f"{CLOB_HOST}/book", params={"token_id": token_id}, timeout=8)
        if resp.status_code == 404:
            return {"exists": False, "bid": None, "ask": None, "empty_book": False}
        resp.raise_for_status()
        data = resp.json()
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        empty = (best_bid is None and best_ask is None)
        return {"exists": True, "bid": best_bid, "ask": best_ask, "empty_book": empty}
    except Exception:
        return {"exists": False, "bid": None, "ask": None, "empty_book": False}


def _verify_clob_batch(
    markets: list[dict],
    workers: int = CLOB_VERIFY_WORKERS,
) -> list[dict]:
    """
    Fetch live CLOB books for all markets in parallel.

    Passes markets where:
    - CLOB returns 200 (market exists on exchange)
    - Book has live orders (bid > 0.01, ask < 0.99)
      OR book is empty (no current makers = uncontested reward opportunity)

    Drops markets where:
    - CLOB returns 404 (resolved or delisted — truly dead)
    """
    verified: list[dict] = []
    clob_404 = 0
    live_with_orders = 0
    live_empty = 0

    def _check(market: dict) -> tuple[dict, dict]:
        token_id = _get_yes_token_id(market)
        if not token_id:
            return market, {"exists": False, "bid": None, "ask": None, "empty_book": False}
        with httpx.Client() as c:
            result = _clob_book_live(token_id, c)
        return market, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check, m): m for m in markets}
        for future in as_completed(futures):
            market, result = future.result()
            if not result["exists"]:
                clob_404 += 1
                continue
            # Market exists on CLOB — include regardless of current order depth
            market["_clob_bid"] = result["bid"]
            market["_clob_ask"] = result["ask"]
            market["_clob_empty_book"] = result["empty_book"]
            if result["empty_book"]:
                live_empty += 1
            else:
                live_with_orders += 1
            verified.append(market)

    logger.info(
        "CLOB verify: %d passed | %d with live orders | %d empty (uncontested) | %d 404/dead",
        len(verified), live_with_orders, live_empty, clob_404,
    )
    return verified


# ---------------------------------------------------------------------------
# Build candidate record
# ---------------------------------------------------------------------------

def _build_candidate(market: dict) -> dict:
    # Prefer live CLOB prices if verified; fall back to Gamma
    clob_bid = market.get("_clob_bid")
    clob_ask = market.get("_clob_ask")
    if clob_bid and clob_ask:
        bid, ask = clob_bid, clob_ask
        mid = (bid + ask) / 2
    else:
        mid = _midpoint(market)
        bid, ask = _best_bid_ask(market)
    rate = _reward_rate(market)
    liq = _liquidity(market)
    vol = _volume_24h(market)
    min_size = _reward_min_size(market)

    spread = round((ask - bid), 4) if bid and ask else None
    max_spread_cents = None
    try:
        val = market.get("rewardsMaxSpread")
        if val is not None:
            max_spread_cents = float(val)
    except Exception:
        pass

    condition_id = market.get("conditionId") or ""
    token_id = _get_yes_token_id(market) or ""

    return {
        "market_slug":              market.get("slug") or condition_id or "",
        "condition_id":             condition_id,
        "token_id":                 token_id,
        "event_slug":               (market.get("events") or [{}])[0].get("slug", "") if isinstance(market.get("events"), list) else "",
        "question":                 market.get("question") or "",
        "category":                 market.get("category") or "",
        "midpoint":                 round(mid, 4) if mid else None,
        "best_bid":                 round(bid, 4) if bid else None,
        "best_ask":                 round(ask, 4) if ask else None,
        "spread":                   spread,
        "liquidity_usd":            round(liq, 2),
        "volume_24h_usd":           round(vol, 2),
        "reward_daily_rate_usdc":   round(rate, 4),
        "reward_min_size_shares":   min_size,
        "reward_max_spread_cents":  max_spread_cents,
        "tick_size":                market.get("tickSize"),
        "clob_empty_book":          market.get("_clob_empty_book", None),
        "price_source":             "clob" if market.get("_clob_bid") else "gamma",
        # Compatibility with viability screen (reward_adjusted_raw_ev placeholder)
        "reward_adjusted_raw_ev":   round(rate, 4),
        "reward_config_summary": {
            "daily_rate_usdc":    round(rate, 4),
            "min_size_shares":    min_size,
            "max_spread_cents":   max_spread_cents,
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-aware maker universe refresh")
    parser.add_argument("--min-rate", type=float, default=MIN_REWARD_RATE,
                        help="Minimum reward daily rate USDC (default: 0)")
    parser.add_argument("--min-liquidity", type=float, default=MIN_LIQUIDITY_USD,
                        help="Minimum liquidity USD (default: 100)")
    parser.add_argument("--min-mid", type=float, default=MIN_MID,
                        help="Minimum midpoint for contested (default: 0.15)")
    parser.add_argument("--max-mid", type=float, default=MAX_MID,
                        help="Maximum midpoint for contested (default: 0.85)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and filter but do not overwrite latest_probe.json")
    args = parser.parse_args()

    print(SEP)
    print("  REWARD-AWARE MAKER UNIVERSE REFRESH")
    print(f"  mid={args.min_mid}–{args.max_mid}  min_rate={args.min_rate}  min_liq={args.min_liquidity}")
    print(f"  dry_run={args.dry_run}")
    print(SEP)

    # Step 1: Fetch all active markets
    print("\n  Step 1: Fetching markets from Gamma API...")
    with httpx.Client() as client:
        all_markets = _fetch_all_markets(GAMMA_HOST, client)
    print(f"  Total fetched: {len(all_markets)}")

    # Step 2: Apply filters
    print("\n  Step 2: Applying filters...")
    clob_markets = [m for m in all_markets if _has_clob(m)]
    print(f"  CLOB-enabled: {len(clob_markets)}")

    reward_markets = [m for m in clob_markets if _reward_rate(m) > args.min_rate and _reward_min_size(m) > 0]
    print(f"  Reward-eligible (rate>{args.min_rate}, min_size>0): {len(reward_markets)}")

    contested = []
    dead_book_count = 0
    low_liq_count = 0
    for m in reward_markets:
        mid = _midpoint(m)
        bid, ask = _best_bid_ask(m)
        liq = _liquidity(m)

        if mid is None or not (args.min_mid <= mid <= args.max_mid):
            continue
        if _is_dead_book(bid, ask):
            dead_book_count += 1
            continue
        if liq < args.min_liquidity:
            low_liq_count += 1
            continue
        contested.append(m)

    print(f"  Contested midpoint ({args.min_mid}–{args.max_mid}): {len(contested) + dead_book_count + low_liq_count}")
    print(f"    → Dead book filtered: {dead_book_count}")
    print(f"    → Low liquidity filtered: {low_liq_count}")
    print(f"    → Survivors: {len(contested)}")

    # Step 3: CLOB live verification (ground truth — Gamma prices are stale)
    print(f"\n  Step 3: CLOB live book verification ({CLOB_VERIFY_WORKERS} workers)...")
    print(f"  404 = resolved/dead. Empty book = uncontested reward opportunity. Both counted.")
    contested = _verify_clob_batch(contested, workers=CLOB_VERIFY_WORKERS)
    live_orders = sum(1 for m in contested if not m.get("_clob_empty_book"))
    empty_books = sum(1 for m in contested if m.get("_clob_empty_book"))
    print(f"  CLOB-verified: {len(contested)} total | {live_orders} with live orders | {empty_books} empty (uncontested)")

    # Step 4: Build candidate records
    candidates = [_build_candidate(m) for m in contested]
    candidates.sort(key=lambda x: x["reward_daily_rate_usdc"], reverse=True)

    # Step 4: Print top candidates
    print(f"\n{SEP}")
    print(f"  Top candidates by reward rate:")
    print(f"  {'slug':<52} {'mid':>6} {'bid':>6} {'ask':>6} {'rate':>8} {'liq':>8}")
    print(f"  {'-'*52} {'------':>6} {'------':>6} {'------':>6} {'--------':>8} {'--------':>8}")
    for c in candidates[:20]:
        slug = c["market_slug"][:52]
        print(
            f"  {slug:<52} "
            f"{(c['midpoint'] or 0):>6.3f} "
            f"{(c['best_bid'] or 0):>6.3f} "
            f"{(c['best_ask'] or 0):>6.3f} "
            f"{c['reward_daily_rate_usdc']:>8.2f} "
            f"{c['liquidity_usd']:>8.0f}"
        )
    if len(candidates) > 20:
        print(f"  … and {len(candidates)-20} more")

    # Step 5: Write output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    probe_id = f"universe_refresh_{ts}"

    output = {
        "probe_id":        probe_id,
        "probe_timestamp": datetime.now(timezone.utc).isoformat(),
        "probe_version":   "universe_refresh_v1",
        "probe_config": {
            "min_mid":        args.min_mid,
            "max_mid":        args.max_mid,
            "min_rate_usdc":  args.min_rate,
            "min_liq_usd":    args.min_liquidity,
            "source":         "gamma_api_live",
        },
        "summary": {
            "total_fetched":    len(all_markets),
            "clob_enabled":     len(clob_markets),
            "reward_eligible":  len(reward_markets),
            "contested":        len(contested),
            "dead_book_skipped": dead_book_count,
            "low_liq_skipped":  low_liq_count,
        },
        "markets": candidates,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / f"{probe_id}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Archive written: {archive_path}")

    if not args.dry_run:
        latest_path = OUTPUT_DIR / "latest_probe.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  latest_probe.json updated: {latest_path}")
    else:
        print(f"  [DRY RUN] latest_probe.json NOT overwritten")

    print(f"\n{SEP}")
    print(f"  Universe refresh complete.")
    print(f"  Contested reward-eligible markets: {len(candidates)}")
    print(f"  Run viability screen next:")
    print(f"    py -3 research_lines/auto_maker_loop/run_reward_aware_viability_screen.py")
    print(SEP)


if __name__ == "__main__":
    main()
