"""
auto_maker_loop — run_ws_truth_test
polyarb_lab / research_lines / auto_maker_loop

Narrow live websocket truth test.

Question:
    For a no-fill or weak-fill case, is the dominant blocker:
    a) real zero market flow (nobody is trading this market at all)
    b) high competitiveness / queue crowding (market is active but we are outranked)
    c) quote not competitive enough (our price is too far from where trades happen)
    d) local attribution missing truth (poll loop missed fills that WS would have caught)

Method:
    - Connects to Polymarket market WS for the target token (public, no auth)
    - Optionally connects to user WS (requires POLYMARKET_PRIVATE_KEY) to observe order events
    - Monitors for --window-minutes (default 5)
    - Reports: trade count, price range, bid/ask spread, and blocker verdict

Usage (PowerShell from repo root):
    # Market data only (no auth needed)
    py -3 research_lines/auto_maker_loop/run_ws_truth_test.py --target rubio

    # Market + user stream (full truth, requires live key)
    py -3 research_lines/auto_maker_loop/run_ws_truth_test.py --target rubio --with-user-ws

    # Custom observation window
    py -3 research_lines/auto_maker_loop/run_ws_truth_test.py --target rubio --window-minutes 10

Output:
    Printed to stdout.  Events persisted to data/research/auto_maker_loop/ws_truth_test_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ws_truth_test")

DATA_DIR = Path("data/research/auto_maker_loop")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    SLUG_ALIASES,
    SURVIVOR_DATA,
)
from research_lines.auto_maker_loop.modules.market_ws_client import MarketWsClient
from research_lines.auto_maker_loop.modules.user_ws_client   import UserWsClient


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _verdict(
    trade_count: int,
    window_min: float,
    last_bid: Optional[float],
    last_ask: Optional[float],
    our_ask: Optional[float],
    user_events: int,
) -> str:
    """
    Classify the dominant blocker from observed WS data.

    Returns one of:
        REAL_ZERO_MARKET_FLOW
        HIGH_COMPETITIVENESS_POSSIBLE
        QUOTE_UNCOMPETITIVE
        LOCAL_ATTRIBUTION_MISSING
        INCONCLUSIVE
    """
    trades_per_min = trade_count / max(window_min, 0.1)

    # (a) Real zero market flow: no trades at all in the window
    if trade_count == 0:
        return "REAL_ZERO_MARKET_FLOW"

    # (b) Market is active but we can estimate where trades happened vs our price
    if last_ask is not None and our_ask is not None:
        spread_cents = round((last_ask - (last_bid or 0)) * 100, 2) if last_bid else None
        ask_distance_cents = round((our_ask - last_ask) * 100, 2)
        if ask_distance_cents > 2.0:
            return "QUOTE_UNCOMPETITIVE"
        if trade_count > 0 and ask_distance_cents <= 1.0:
            return "HIGH_COMPETITIVENESS_POSSIBLE"

    # (c) User WS captured order events but poll loop did not show fills
    if user_events > 0:
        return "LOCAL_ATTRIBUTION_MISSING"

    if trade_count > 0:
        return "HIGH_COMPETITIVENESS_POSSIBLE"

    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Narrow WS truth test for one target market")
    parser.add_argument("--target", default="rubio",
                        help="Market target: rubio, hungary, vance (default: rubio)")
    parser.add_argument("--window-minutes", type=float, default=5.0,
                        help="Observation window in minutes (default: 5)")
    parser.add_argument("--with-user-ws", action="store_true",
                        help="Also connect user WS (requires POLYMARKET credentials in env)")
    args = parser.parse_args()

    slug = SLUG_ALIASES.get(args.target, args.target)
    if slug not in SURVIVOR_DATA:
        print(f"ERROR: unknown target '{args.target}'. Valid: {list(SLUG_ALIASES)}")
        sys.exit(1)

    data      = SURVIVOR_DATA[slug]
    token_id  = data["token_id"]
    price_ref = data.get("yes_price_ref", 0.50)

    window_sec = args.window_minutes * 60.0
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ts_label     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    mkt_log_path = DATA_DIR / f"ws_truth_market_{args.target}_{ts_label}.jsonl"
    usr_log_path = DATA_DIR / f"ws_truth_user_{args.target}_{ts_label}.jsonl"

    print(f"\n{'='*60}")
    print(f"  WS TRUTH TEST — {args.target.upper()}")
    print(f"{'='*60}")
    print(f"  token_id   : {token_id[:24]}...")
    print(f"  price_ref  : {price_ref:.4f}")
    print(f"  window     : {args.window_minutes:.0f} min")
    print(f"  mkt_log    : {mkt_log_path}")

    # ── Start market WS ───────────────────────────────────────────────────
    mkt_ws = MarketWsClient(token_ids=[token_id], log_path=mkt_log_path)
    mkt_ws.start()

    # ── Optionally start user WS ──────────────────────────────────────────
    usr_ws: Optional[UserWsClient] = None
    if args.with_user_ws:
        api_key      = os.environ.get("CLOB_API_KEY", "")
        api_secret   = os.environ.get("CLOB_API_SECRET", "")
        api_passphrase = os.environ.get("CLOB_PASSPHRASE", "")
        if not api_key:
            # Try loading from scoring_activation creds
            try:
                from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
                    load_activation_credentials,
                )
                creds      = load_activation_credentials()
                api_key        = creds.api_key
                api_secret     = creds.api_secret
                api_passphrase = creds.api_passphrase
            except Exception as exc:
                logger.warning("user WS: could not load credentials: %s", exc)
        if api_key:
            usr_ws = UserWsClient(api_key, api_secret, api_passphrase, usr_log_path)
            usr_ws.start()
            print(f"  user_log   : {usr_log_path}")
        else:
            print("  user WS    : SKIPPED (no credentials found)")

    # ── Observation window ────────────────────────────────────────────────
    print(f"\n  Observing for {args.window_minutes:.0f} min... (Ctrl+C to abort early)\n")
    start_ts = time.monotonic()
    try:
        while (time.monotonic() - start_ts) < window_sec:
            elapsed = time.monotonic() - start_ts
            print(
                f"  [{elapsed/60:.1f}/{args.window_minutes:.0f} min]  "
                f"mkt_events={mkt_ws.events_received}  "
                f"trades={mkt_ws.trade_count}  "
                f"bid={mkt_ws.last_best_bid}  "
                f"ask={mkt_ws.last_best_ask}  "
                f"user_events={usr_ws.events_received if usr_ws else 'n/a'}",
                end="\r",
            )
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("\n  [interrupted]")

    actual_window_min = (time.monotonic() - start_ts) / 60.0

    # ── Stop clients ──────────────────────────────────────────────────────
    mkt_ws.stop()
    if usr_ws:
        usr_ws.stop()

    # ── Final report ──────────────────────────────────────────────────────
    verdict = _verdict(
        trade_count=mkt_ws.trade_count,
        window_min=actual_window_min,
        last_bid=mkt_ws.last_best_bid,
        last_ask=mkt_ws.last_best_ask,
        our_ask=price_ref + 0.015,    # estimate: price_ref + half spread
        user_events=usr_ws.events_received if usr_ws else 0,
    )

    spread_cents = (
        round((mkt_ws.last_best_ask - mkt_ws.last_best_bid) * 100, 2)
        if mkt_ws.last_best_bid and mkt_ws.last_best_ask
        else None
    )

    print(f"\n{'='*60}")
    print(f"  WS TRUTH RESULT — {args.target.upper()}")
    print(f"{'='*60}")
    print(f"  observation_min      : {actual_window_min:.1f}")
    print(f"  market_events_total  : {mkt_ws.events_received}")
    print(f"  trade_count          : {mkt_ws.trade_count}  (last_trade_price events)")
    print(f"  trades_per_min       : {mkt_ws.trade_count / max(actual_window_min, 0.1):.1f}")
    print(f"  last_best_bid        : {mkt_ws.last_best_bid}")
    print(f"  last_best_ask        : {mkt_ws.last_best_ask}")
    print(f"  spread_cents         : {spread_cents}")
    print(f"  user_events          : {usr_ws.events_received if usr_ws else 'not captured'}")
    print(f"\n  DOMINANT_BLOCKER     : {verdict}")
    print(f"\n  Interpretation:")
    _print_interpretation(verdict, mkt_ws.trade_count, actual_window_min, spread_cents)
    print(f"{'='*60}\n")

    # Persist summary
    summary = {
        "ts":              datetime.now(timezone.utc).isoformat(),
        "target":          args.target,
        "token_id":        token_id,
        "window_min":      actual_window_min,
        "market_events":   mkt_ws.events_received,
        "trade_count":     mkt_ws.trade_count,
        "last_best_bid":   mkt_ws.last_best_bid,
        "last_best_ask":   mkt_ws.last_best_ask,
        "spread_cents":    spread_cents,
        "user_events":     usr_ws.events_received if usr_ws else None,
        "dominant_blocker": verdict,
        "mkt_log":         str(mkt_log_path),
        "usr_log":         str(usr_log_path) if usr_ws else None,
    }
    summary_path = DATA_DIR / f"ws_truth_summary_{args.target}_{ts_label}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  summary saved → {summary_path}\n")


def _print_interpretation(
    verdict: str,
    trade_count: int,
    window_min: float,
    spread_cents: Optional[float],
) -> None:
    msgs = {
        "REAL_ZERO_MARKET_FLOW": (
            f"    0 trades in {window_min:.1f} min. No market participants are trading at any price.\n"
            f"    Quote competitiveness is irrelevant — there are no counterparties.\n"
            f"    Action: check if Rubio market has meaningful open interest or consider switching target."
        ),
        "HIGH_COMPETITIVENESS_POSSIBLE": (
            f"    {trade_count} trades in {window_min:.1f} min. Market IS flowing.\n"
            f"    Our quotes were in the right zone but likely outranked by earlier queue entrants.\n"
            f"    Action: confirm our order arrived at top-of-book vs best_ask; consider tighter spread."
        ),
        "QUOTE_UNCOMPETITIVE": (
            f"    {trade_count} trades in {window_min:.1f} min but our ask price > best_ask by >2¢.\n"
            f"    Our ASK was never the best available sell offer.\n"
            f"    Action: tighten ask placement or verify midpoint is computed correctly."
        ),
        "LOCAL_ATTRIBUTION_MISSING": (
            f"    User WS captured order events that the poll loop may have missed.\n"
            f"    Possible: fills occurred between poll intervals and were not recorded.\n"
            f"    Action: compare user WS event log vs position_manager poll timestamps."
        ),
        "INCONCLUSIVE": (
            f"    Insufficient signal to isolate dominant cause.\n"
            f"    Extend window or re-run with --with-user-ws for order-level truth."
        ),
    }
    print(msgs.get(verdict, f"    verdict={verdict}"))


if __name__ == "__main__":
    main()
