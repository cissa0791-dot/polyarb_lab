"""
run_hungary_alignment_test
polyarb_lab / research_lines / auto_maker_loop

Dual-stream alignment test for Hungary (ASK side only, no BID).

Purpose
-------
Determine whether our ASK order reaches top-of-book and whether no-fill
evidence points to queue outrank, sparse flow, or a local attribution gap.

What it does
------------
1.  Start market WS (public)  +  user WS (auth)  simultaneously.
2.  Wait 3s for stream stabilisation; snapshot initial best_bid / best_ask.
3.  Locate an existing open SELL order on Hungary, or place a minimal new one.
4.  Wait 2s for the book to reflect our order; snapshot at-placement state.
5.  Compute queue position: our_ask_price − market_best_ask (in cents).
6.  Observe for --window-minutes, logging every 5s.
7.  After window: parse user WS JSONL for order_update / trade events tied to
    our order.  Query exchange for fill ground-truth.
8.  Emit structured diagnosis:
        FILL_CONFIRMED      order was filled during the window
        ATTRIBUTION_GAP     exchange says filled; user WS missed the event
        QUEUE_OUTRANK       market traded; our order was live; no fill
        SPARSE_FLOW         0 market trades in the window
        QUOTE_UNCOMPETITIVE our ask was >3¢ above best_ask; never competitive
        INCONCLUSIVE        insufficient signal
9.  Cancel the ASK if *we* placed it.
10. Save summary JSON.

Queue position buckets (our_ask − market_best_ask in cents):
    ≤ 0.0¢   AT_TOP_OR_AHEAD   we are best ask or have already improved on it
     0–1¢    NEAR_TOP          competitive
     1–3¢    INSIDE_SPREAD     within typical reward zone but not leading
    > 3¢     BEHIND            not at top; fill unlikely even with real flow

Usage (PowerShell from repo root)
----------------------------------
    $env:POLYMARKET_PRIVATE_KEY = "<key>"

    # Place a fresh minimal ASK and observe 10 min
    py -3 research_lines/auto_maker_loop/run_hungary_alignment_test.py --live

    # Use an existing open order (no placement)
    py -3 research_lines/auto_maker_loop/run_hungary_alignment_test.py --live ^
        --ask-order-id 0xABC...

    # Shorter window
    py -3 research_lines/auto_maker_loop/run_hungary_alignment_test.py --live ^
        --window-minutes 5

    # Manual ask price override
    py -3 research_lines/auto_maker_loop/run_hungary_alignment_test.py --live ^
        --ask-price 0.635

    # Stream-only (no orders, no auth required)
    py -3 research_lines/auto_maker_loop/run_hungary_alignment_test.py

Required env vars
-----------------
    POLYMARKET_PRIVATE_KEY   EVM private key (only needed with --live)
"""
from __future__ import annotations

import argparse
import json
import logging
import requests
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hungary_alignment_test")

DATA_DIR = Path("data/research/auto_maker_loop")
DEFAULT_TARGET_SLUG = "will-the-next-prime-minister-of-hungary-be-pter-magyar"
TARGET_ALIASES = {
    "hungary": DEFAULT_TARGET_SLUG,
    "netanyahu": "netanyahu-out-by-june-30-383-244-575",
}
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
BRIDGED_GAMMA_METADATA_SLUGS = {
    "will-the-next-prime-minister-of-hungary-be-viktor-orbn",
}
AUTH_TRUTH_DIR = Path(
    "data/research/reward_aware_maker_probe/authenticated_rewards_truth"
)
LATEST_PROBE_PATH = Path("data/research/reward_aware_maker_probe/latest_probe.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def _queue_label(delta_cents: Optional[float]) -> str:
    if delta_cents is None:
        return "UNKNOWN"
    if delta_cents <= 0.0:
        return "AT_TOP_OR_AHEAD"
    if delta_cents <= 1.0:
        return "NEAR_TOP"
    if delta_cents <= 3.0:
        return "INSIDE_SPREAD"
    return "BEHIND"


def _file_tag_for_slug(slug: str) -> str:
    reverse_aliases = {v: k for k, v in TARGET_ALIASES.items()}
    base = reverse_aliases.get(slug, slug)
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in base.lower()
    ).strip("_")
    return safe[:64] or "target"


def _latest_auth_truth_path() -> Optional[Path]:
    if not AUTH_TRUTH_DIR.exists():
        return None
    paths = sorted(AUTH_TRUTH_DIR.glob("auth_rewards_truth_*.json"))
    return paths[-1] if paths else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_slug_entry(obj: Any, slug: str) -> Optional[dict[str, Any]]:
    if isinstance(obj, dict):
        if obj.get("slug") == slug or obj.get("market_slug") == slug or obj.get("full_slug") == slug:
            return obj
        for value in obj.values():
            hit = _find_slug_entry(value, slug)
            if hit:
                return hit
    elif isinstance(obj, list):
        for value in obj:
            hit = _find_slug_entry(value, slug)
            if hit:
                return hit
    return None


def _fetch_gamma_market_by_slug(slug: str) -> Optional[dict[str, Any]]:
    response = requests.get(
        GAMMA_MARKETS_URL,
        params={"slug": slug, "limit": 5},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and row.get("slug") == slug:
                return row
        return payload[0] if payload and isinstance(payload[0], dict) else None
    if isinstance(payload, dict) and payload.get("slug") == slug:
        return payload
    return None


def _extract_yes_token_id(gamma_row: dict[str, Any]) -> str:
    raw_token_ids = gamma_row.get("clobTokenIds") or "[]"
    raw_outcomes = gamma_row.get("outcomes") or "[]"
    token_ids = json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids
    outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    if isinstance(token_ids, list) and isinstance(outcomes, list):
        for outcome, token_id in zip(outcomes, token_ids):
            if str(outcome).strip().lower() == "yes":
                return str(token_id)
    if isinstance(token_ids, list) and token_ids:
        return str(token_ids[0])
    return ""


def _resolve_target_metadata(
    requested_target: str,
    survivor_data: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any], str]:
    slug = TARGET_ALIASES.get(requested_target, requested_target)
    if slug in survivor_data:
        return slug, dict(survivor_data[slug]), "SURVIVOR_DATA"

    auth_path = _latest_auth_truth_path()
    if auth_path is None:
        raise SystemExit(
            "ERROR: target is not in SURVIVOR_DATA and no authenticated rewards truth file exists."
        )

    auth_obj = _load_json(auth_path)
    auth_row = (auth_obj.get("auth_truths") or {}).get(slug)
    metadata_source = f"auth_truth:{auth_path.name}"
    token_id = ""
    condition_id = ""
    if isinstance(auth_row, dict):
        token_id = str(auth_row.get("token_id") or "")
        condition_id = str(auth_row.get("condition_id") or "")

    if (not token_id or not condition_id) and slug in BRIDGED_GAMMA_METADATA_SLUGS:
        gamma_row = _fetch_gamma_market_by_slug(slug)
        if not isinstance(gamma_row, dict):
            raise SystemExit(
                f"ERROR: target '{requested_target}' resolved to slug '{slug}', "
                f"but it is not present in {auth_path} and gamma metadata lookup failed."
            )
        token_id = _extract_yes_token_id(gamma_row)
        condition_id = str(gamma_row.get("conditionId") or gamma_row.get("condition_id") or "")
        metadata_source = "gamma_markets:slug_bridge"

    if not token_id or not condition_id:
        raise SystemExit(
            f"ERROR: target '{requested_target}' resolved to slug '{slug}', "
            f"but usable token_id/condition_id metadata was not found in {auth_path}."
        )

    probe_row: dict[str, Any] = {}
    if LATEST_PROBE_PATH.exists():
        try:
            probe_obj = _load_json(LATEST_PROBE_PATH)
            probe_row = _find_slug_entry(probe_obj, slug) or {}
        except Exception as exc:
            logger.warning("latest_probe load failed for slug=%s: %s", slug, exc)

    reward_summary = (
        probe_row.get("reward_config_summary")
        if isinstance(probe_row, dict)
        else None
    ) or {}
    price_ref = probe_row.get("midpoint") if isinstance(probe_row, dict) else None
    if price_ref is not None:
        try:
            price_ref = float(price_ref)
        except (TypeError, ValueError):
            price_ref = None
    if price_ref is not None and not (0.01 <= price_ref <= 0.99):
        price_ref = None

    min_size = reward_summary.get("rewards_min_size_shares", 200.0)
    daily_rate = reward_summary.get("reward_daily_rate_usdc", 0.0)
    max_spread = reward_summary.get("rewards_max_spread_cents", 3.5)

    return slug, {
        "condition_id": condition_id,
        "token_id": token_id,
        "daily_rate_usdc": float(daily_rate),
        "fallback_max_spread_cents": float(max_spread),
        "fallback_min_size": float(min_size),
        "yes_price_ref": price_ref,
    }, metadata_source


def _parse_user_log(log_path: Path, our_order_id: Optional[str]) -> dict:
    """
    Parse the user WS JSONL log.

    Returns breakdown of event types.  Filters for our specific order_id
    to separate our events from any background account activity.
    """
    result = {
        "total_events":        0,
        "order_update_events": 0,
        "trade_events":        0,
        "status_events":       0,
        "our_order_updates":   0,   # order_update with id == our_order_id
        "our_fill_events":     0,   # trade where maker or taker == our_order_id
        "any_fill_events":     0,   # any trade event (account-wide)
    }
    if not log_path.exists():
        return result

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            result["total_events"] += 1
            etype = str(event.get("_type") or event.get("type") or "")

            if etype in ("order_update", "order"):
                result["order_update_events"] += 1
                if our_order_id:
                    eid = str(
                        event.get("id") or event.get("order_id")
                        or event.get("orderId") or ""
                    )
                    if eid == our_order_id:
                        result["our_order_updates"] += 1

            elif etype == "trade":
                result["trade_events"]    += 1
                result["any_fill_events"] += 1
                if our_order_id:
                    maker = str(
                        event.get("maker_order_id") or event.get("makerOrderId")
                        or event.get("maker_id") or ""
                    )
                    taker = str(
                        event.get("taker_order_id") or event.get("takerOrderId")
                        or event.get("taker_id") or ""
                    )
                    if our_order_id in (maker, taker):
                        result["our_fill_events"] += 1

            elif etype == "status":
                result["status_events"] += 1

    return result


def _diagnose(
    queue_lbl: str,
    market_trades: int,
    user_events: dict,
    exchange_filled: bool,
    window_min: float,
) -> tuple[str, str]:
    """
    Classify the dominant cause of no-fill (or confirm a fill).

    Returns (diagnosis_code, one-sentence explanation).
    """
    our_filled = user_events["our_fill_events"] > 0

    # ── Fill path ────────────────────────────────────────────────────────────
    if exchange_filled or our_filled:
        if exchange_filled and not our_filled:
            return (
                "ATTRIBUTION_GAP",
                "Exchange confirms ASK fill but user WS did not capture the trade event — "
                "poll-loop attribution lag or missed event.",
            )
        return (
            "FILL_CONFIRMED",
            "ASK filled during the observation window — fill path is working.",
        )

    # ── No fill path ─────────────────────────────────────────────────────────
    if market_trades == 0:
        return (
            "SPARSE_FLOW",
            f"0 market trades in {window_min:.1f} min — flow was lower than the "
            f"HIGH_COMPETITIVENESS_POSSIBLE verdict implied; market may be intermittent.",
        )

    if queue_lbl == "BEHIND":
        return (
            "QUOTE_UNCOMPETITIVE",
            f"Market had {market_trades} trade(s) but our ask was >3¢ above best_ask "
            f"at placement — never competitive; price must be tightened.",
        )

    if market_trades > 0:
        return (
            "QUEUE_OUTRANK",
            f"Market had {market_trades} trade(s) in {window_min:.1f} min; our order "
            f"was {queue_lbl} but was not matched — earlier queue entrants filled first.",
        )

    return ("INCONCLUSIVE", "Insufficient signal to classify dominant cause.")


def _fetch_existing_sell(client: Any, token_id: str) -> Optional[dict]:
    """
    Return the first open SELL order for this token on the account.
    Returns dict(order_id, price, size) or None.
    """
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw = client.get_orders(OpenOrderParams())
        orders: list = (
            raw if isinstance(raw, list)
            else (raw.get("data") or [] if isinstance(raw, dict) else [])
        )
        for o in orders:
            if not isinstance(o, dict):
                continue
            asset = str(
                o.get("asset_id") or o.get("token_id") or o.get("market") or ""
            )
            if token_id not in asset and asset != token_id:
                continue
            if str(o.get("side") or "").upper() != "SELL":
                continue
            oid = str(o.get("id") or o.get("order_id") or "")
            if not oid:
                continue
            return {
                "order_id": oid,
                "price":    float(o.get("price") or 0.0),
                "size":     float(
                    o.get("size") or o.get("original_size") or o.get("size_remaining") or 0.0
                ),
            }
    except Exception as exc:
        logger.warning("fetch_existing_sell: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hungary dual-stream alignment test")
    parser.add_argument(
        "--target", default="hungary",
        help="Target alias or full slug (default: hungary)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode: place real orders + connect authenticated user WS",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Resolve target metadata and exit before WS/order actions",
    )
    parser.add_argument(
        "--ask-order-id", default="",
        help="Existing ASK order ID to observe (skips placement)",
    )
    parser.add_argument(
        "--ask-price", type=float, default=None,
        help="Override ask price for newly placed order (default: midpoint + 1.5¢)",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=10.0,
        help="Observation window in minutes (default: 10)",
    )
    args = parser.parse_args()

    if args.live and not args.ask_order_id and args.ask_price is None:
        print(
            "ERROR: --live requires either --ask-order-id <id> or --ask-price <price>.\n"
            "       Refusing to silently derive ask price from midpoint formula."
        )
        sys.exit(1)

    # ── Imports ───────────────────────────────────────────────────────────────
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        SURVIVOR_DATA,
        load_activation_credentials,
        build_clob_client,
        fetch_midpoint,
        _place_order,
        _cancel_order,
        _is_filled,
        CLOB_HOST,
    )
    from research_lines.auto_maker_loop.modules.market_ws_client import MarketWsClient
    from research_lines.auto_maker_loop.modules.user_ws_client   import UserWsClient

    target_slug, data, metadata_source = _resolve_target_metadata(
        args.target.strip(),
        SURVIVOR_DATA,
    )
    target_tag = _file_tag_for_slug(target_slug)
    target_name = target_tag.upper()
    token_id = data["token_id"]
    condition_id = str(data.get("condition_id") or "")
    price_ref = data.get("yes_price_ref")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    mkt_log = DATA_DIR / f"alignment_market_{target_tag}_{ts_label}.jsonl"
    usr_log = DATA_DIR / f"alignment_user_{target_tag}_{ts_label}.jsonl"
    sum_path = DATA_DIR / f"alignment_summary_{target_tag}_{ts_label}.json"

    _section(f"ALIGNMENT TEST — {target_name}")
    print(f"  target    : {args.target}")
    print(f"  slug      : {target_slug[:55]}")
    print(f"  token_id  : {token_id[:28]}...")
    print(f"  condition : {condition_id[:28] + '...' if condition_id else 'N/A'}")
    print(f"  price_ref : {price_ref:.4f}" if price_ref is not None else "  price_ref : N/A")
    print(f"  metadata  : {metadata_source}")
    print(f"  window    : {args.window_minutes:.0f} min")
    print(f"  mode      : {'LIVE' if args.live else 'STREAM-ONLY (no orders)'}")
    print(f"  mkt_log   : {mkt_log}")
    print(f"  usr_log   : {usr_log}")

    if args.metadata_only:
        print("\n  metadata-only       : True  (skipping WS + order actions)")
        return

    # ── Credentials + client ──────────────────────────────────────────────────
    creds  = None
    client = None
    if args.live:
        creds = load_activation_credentials()
        if creds is None:
            print("\n  ERROR: Could not load credentials. Set POLYMARKET_PRIVATE_KEY.")
            sys.exit(1)
        client = build_clob_client(creds, CLOB_HOST)
        print(f"\n  credentials loaded  : OK")

    # ── Start streams ─────────────────────────────────────────────────────────
    _section("Stream Startup")
    mkt_ws = MarketWsClient(token_ids=[token_id], log_path=mkt_log)
    mkt_ws.start()
    print("  market WS           : started  (public, no auth)")

    usr_ws: Optional[UserWsClient] = None
    if args.live and creds:
        usr_ws = UserWsClient(
            api_key        = creds.api_key,
            api_secret     = creds.api_secret,
            api_passphrase = creds.api_passphrase,
            log_path       = usr_log,
        )
        usr_ws.start()
        print("  user WS             : started  (authenticated)")
    else:
        print("  user WS             : SKIPPED  (need --live flag)")

    print("\n  Waiting 3s for stream stabilisation...")
    time.sleep(3.0)

    # ── Initial book snapshot ─────────────────────────────────────────────────
    init_bid = mkt_ws.last_best_bid
    init_ask = mkt_ws.last_best_ask
    init_spread = (
        round((init_ask - init_bid) * 100, 2)
        if init_bid and init_ask else None
    )
    _section("Initial Book State  (market WS — 3s after connect)")
    print(f"  last_best_bid       : {init_bid}")
    print(f"  last_best_ask       : {init_ask}")
    print(f"  spread_cents        : {init_spread}")
    print(f"  mkt_events_so_far   : {mkt_ws.events_received}")
    if usr_ws:
        print(f"  user_events_so_far  : {usr_ws.events_received}")

    # ── Order setup ───────────────────────────────────────────────────────────
    _section("Order Setup")
    ask_order_id: Optional[str] = None
    ask_price:    Optional[float] = None
    ask_size:     Optional[float] = None
    we_placed = False

    if args.ask_order_id:
        # Caller supplied an existing order ID
        ask_order_id = args.ask_order_id.strip()
        print(f"  using provided order: {ask_order_id[:24]}...")
        # Try to fetch its price from exchange for queue position calc
        if client:
            try:
                raw_order = client.get_order(ask_order_id)
                if raw_order:
                    ask_price = float(
                        (raw_order.get("price") if isinstance(raw_order, dict)
                         else getattr(raw_order, "price", None)) or 0.0
                    ) or None
                    ask_size = float(
                        (raw_order.get("size") or raw_order.get("original_size")
                         if isinstance(raw_order, dict)
                         else getattr(raw_order, "size", None)) or 0.0
                    ) or None
                    print(f"  fetched price       : {ask_price}")
                    print(f"  fetched size        : {ask_size}")
            except Exception as exc:
                logger.warning("could not fetch provided order: %s", exc)

    elif args.live and client:
        # Check for an existing open SELL order first
        existing = _fetch_existing_sell(client, token_id)
        if existing:
            ask_order_id = existing["order_id"]
            ask_price    = existing["price"]
            ask_size     = existing["size"]
            print(f"  found existing SELL : {ask_order_id[:24]}...")
            print(f"  price               : {ask_price}")
            print(f"  size                : {ask_size}")
            print(f"  we_placed           : False  (not our order to cancel)")
        else:
            # No existing order — place a minimal qualifying ASK
            midpoint, mid_src = fetch_midpoint(client, token_id, price_ref)
            if midpoint is None and args.ask_price is None:
                print("  ERROR: midpoint unavailable — cannot place order")
                mkt_ws.stop()
                if usr_ws:
                    usr_ws.stop()
                sys.exit(1)
            if midpoint is not None:
                print(f"  midpoint            : {midpoint:.4f}  ({mid_src})")
            else:
                print("  midpoint            : unavailable  (continuing with explicit --ask-price)")
            ask_price = (
                args.ask_price
                if args.ask_price is not None
                else round(midpoint + 0.015, 4)   # midpoint + half of 3¢ reward spread
            )
            ask_size = float(data["fallback_min_size"])   # 200 shares (minimum qualifying)
            print(f"  placing ASK         : {ask_size:.0f} shares @ {ask_price:.4f}")
            placed_id, place_err = _place_order(client, token_id, ask_price, ask_size, "SELL")
            if not placed_id:
                print(f"  ASK PLACEMENT FAILED: {place_err}")
                mkt_ws.stop()
                if usr_ws:
                    usr_ws.stop()
                sys.exit(1)
            ask_order_id = placed_id
            we_placed    = True
            print(f"  ASK placed          : {ask_order_id[:24]}...")
            print(f"  we_placed           : True  (will cancel at end if unfilled)")
    else:
        print("  no order — stream-only observe mode (no queue position assessment)")

    # ── At-placement book snapshot ────────────────────────────────────────────
    if ask_order_id:
        print(f"\n  Waiting 2s for book update after order placement...")
        time.sleep(2.0)

    placement_bid   = mkt_ws.last_best_bid
    placement_ask   = mkt_ws.last_best_ask
    queue_delta: Optional[float] = None
    queue_lbl   = "UNKNOWN"

    if ask_price is not None and placement_ask is not None:
        queue_delta = round((ask_price - placement_ask) * 100, 2)
        queue_lbl   = _queue_label(queue_delta)

    _section("At-Placement Book State  (market WS — 2s after order ack)")
    print(f"  our_ask_price       : {ask_price}")
    print(f"  market_best_ask     : {placement_ask}")
    print(f"  market_best_bid     : {placement_bid}")
    if queue_delta is not None:
        sign = "+" if queue_delta >= 0 else ""
        print(f"  delta_cents         : {sign}{queue_delta:.2f}¢  (our_ask − mkt_best_ask)")
    print(f"  queue_position      : {queue_lbl}")
    print()
    if queue_lbl == "AT_TOP_OR_AHEAD":
        print("  ✓ order is at or better than best ask — competing for top-of-book fills")
    elif queue_lbl == "NEAR_TOP":
        print("  ~ order is within 1¢ of best ask — competitive but not leading")
    elif queue_lbl == "INSIDE_SPREAD":
        print("  ~ order is 1–3¢ above best ask — in reward zone but behind leader")
    elif queue_lbl == "BEHIND":
        print("  ✗ order is >3¢ above best ask — not competitive; fills very unlikely")

    # ── Observation window ────────────────────────────────────────────────────
    _section(f"Observing — {args.window_minutes:.0f} min window  (print every 5s)")
    window_sec = args.window_minutes * 60.0
    start_ts   = time.monotonic()

    try:
        while (time.monotonic() - start_ts) < window_sec:
            elapsed    = time.monotonic() - start_ts
            usr_count  = usr_ws.events_received if usr_ws else "n/a"
            print(
                f"  [{elapsed / 60:.1f}/{args.window_minutes:.0f}min]  "
                f"mkt_events={mkt_ws.events_received}  "
                f"mkt_trades={mkt_ws.trade_count}  "
                f"bid={mkt_ws.last_best_bid}  "
                f"ask={mkt_ws.last_best_ask}  "
                f"user_events={usr_count}",
                end="\r",
            )
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("\n  [interrupted early]")

    actual_min = (time.monotonic() - start_ts) / 60.0
    print(f"\n  window complete     : {actual_min:.1f} min")

    # ── Stop streams ──────────────────────────────────────────────────────────
    mkt_ws.stop()
    if usr_ws:
        usr_ws.stop()

    # ── Final market WS metrics ───────────────────────────────────────────────
    final_bid    = mkt_ws.last_best_bid
    final_ask    = mkt_ws.last_best_ask
    trade_count  = mkt_ws.trade_count
    total_events = mkt_ws.events_received

    # ── Parse user WS log ────────────────────────────────────────────────────
    user_events = _parse_user_log(usr_log, ask_order_id)

    # ── Exchange fill ground-truth ────────────────────────────────────────────
    exchange_filled = False
    if client and ask_order_id:
        try:
            exchange_filled = _is_filled(client, ask_order_id)
        except Exception as exc:
            logger.warning("exchange fill check failed: %s", exc)

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    diagnosis, explanation = _diagnose(
        queue_lbl       = queue_lbl,
        market_trades   = trade_count,
        user_events     = user_events,
        exchange_filled = exchange_filled,
        window_min      = actual_min,
    )

    # ── Final report ──────────────────────────────────────────────────────────
    _section(f"ALIGNMENT RESULT — {target_name}")

    print(f"  observation_min      : {actual_min:.1f}")
    print()
    print(f"  ── STREAM HEALTH ───────────────────────────────────────────")
    print(f"  mkt_ws_connected     : {total_events > 0}  (events={total_events})")
    print(f"  user_ws_connected    : {usr_ws is not None}")
    print(f"  user_events_total    : {user_events['total_events']}")
    print(f"  user_order_updates   : {user_events['order_update_events']}")
    print(f"  user_trade_events    : {user_events['trade_events']}")
    print()
    print(f"  ── QUEUE POSITION ──────────────────────────────────────────")
    print(f"  our_ask_price        : {ask_price}")
    print(f"  at_placement_ask     : {placement_ask}")
    delta_str = (
        f"{'+' if queue_delta >= 0 else ''}{queue_delta:.2f}¢"
        if queue_delta is not None else "N/A"
    )
    print(f"  delta_cents          : {delta_str}  (our_ask − mkt_best_ask)")
    print(f"  queue_position       : {queue_lbl}")
    print()
    print(f"  ── MARKET FLOW ─────────────────────────────────────────────")
    print(f"  market_trade_count   : {trade_count}")
    print(f"  final_best_bid       : {final_bid}")
    print(f"  final_best_ask       : {final_ask}")
    print()
    print(f"  ── OUR ORDER EVENTS ────────────────────────────────────────")
    print(f"  our_order_updates    : {user_events['our_order_updates']}")
    print(f"  our_fill_events      : {user_events['our_fill_events']}")
    print(f"  exchange_filled      : {exchange_filled}")
    print()
    print(f"  ── DIAGNOSIS ───────────────────────────────────────────────")
    print(f"  VERDICT              : {diagnosis}")
    print(f"  explanation          : {explanation}")

    # ── Clean up our order ────────────────────────────────────────────────────
    if we_placed and client and ask_order_id and not exchange_filled:
        _section("Cleanup — Cancelling placed ASK")
        ok = _cancel_order(client, ask_order_id)
        print(f"  cancel result        : {'OK' if ok else 'FAILED  (check for open order manually)'}")

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {
        "ts":                    datetime.now(timezone.utc).isoformat(),
        "target":                target_tag,
        "slug":                  target_slug,
        "token_id":              token_id,
        "condition_id":          condition_id,
        "metadata_source":       metadata_source,
        "window_min":            actual_min,
        "mode":                  "live" if args.live else "stream_only",
        # order
        "ask_order_id":          ask_order_id,
        "ask_price":             ask_price,
        "ask_size":              ask_size,
        "we_placed":             we_placed,
        # book snapshots
        "initial_best_bid":      init_bid,
        "initial_best_ask":      init_ask,
        "initial_spread_cents":  init_spread,
        "placement_best_bid":    placement_bid,
        "placement_best_ask":    placement_ask,
        # queue position
        "queue_delta_cents":     queue_delta,
        "queue_label":           queue_lbl,
        # market WS
        "market_events_total":   total_events,
        "market_trade_count":    trade_count,
        "final_best_bid":        final_bid,
        "final_best_ask":        final_ask,
        # user WS
        "user_events":           user_events,
        "exchange_filled":       exchange_filled,
        # diagnosis
        "diagnosis":             diagnosis,
        "explanation":           explanation,
        # log paths
        "mkt_log":               str(mkt_log),
        "usr_log":               str(usr_log),
    }
    with sum_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  summary saved → {sum_path}\n")


if __name__ == "__main__":
    main()
