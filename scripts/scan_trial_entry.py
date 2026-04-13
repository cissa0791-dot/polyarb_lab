"""Trial entry scanner — read-only, no DB writes, no live orders.

Scans Polymarket binary markets for single_market_mispricing entries that
satisfy the locked trial spec:

  YES_ask + NO_ask <= 1.0 - min_edge

When both sides can be bought for less than $1/pair combined, the position
is guaranteed to pay $1 at resolution regardless of outcome.  The edge is
the guaranteed profit per share pair before fees.

Usage:
    python scripts/scan_trial_entry.py
    python scripts/scan_trial_entry.py --limit 200 --min-edge 0.03
    python scripts/scan_trial_entry.py --loop --interval-sec 30
    python scripts/scan_trial_entry.py --loop --interval-sec 60 --min-edge 0.02

Locked trial spec defaults:
    --min-edge     0.03   (3 cents)
    --target-usd   10.00  ($10/pair)
    --limit        100    (top markets by 24hr volume)
    --interval-sec 30     (rescan every 30s in loop mode)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import httpx

_GAMMA = "https://gamma-api.polymarket.com"
_CLOB  = "https://clob.polymarket.com"

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fetch_markets(limit: int) -> list[dict]:
    resp = httpx.get(
        f"{_GAMMA}/markets",
        params={
            "limit": limit,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else data.get("markets", [])


def _scan_once(limit: int, min_edge: float, target_usd: float) -> list[dict]:
    """Scan markets and return list of qualifying hit dicts."""
    markets = _fetch_markets(limit)
    total = len(markets)
    hits: list[dict] = []
    scanned = 0

    for i, m in enumerate(markets):
        raw_ids = m.get("clobTokenIds") or []
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except Exception:
                continue
        if len(raw_ids) < 2:
            continue

        slug     = str(m.get("slug") or m.get("id") or "unknown")
        min_size = float(m.get("orderMinSize") or 5)
        yes_id   = str(raw_ids[0])
        no_id    = str(raw_ids[1])

        # Progress line — overwrite in place
        short_slug = slug[:52]
        print(f"\r  [{i+1:>3}/{total}] {short_slug:<52}", end="", flush=True)

        try:
            yb = httpx.get(f"{_CLOB}/book", params={"token_id": yes_id}, timeout=5).json()
            nb = httpx.get(f"{_CLOB}/book", params={"token_id": no_id},  timeout=5).json()
        except Exception:
            scanned += 1
            continue

        ya = yb.get("asks") or []
        na = nb.get("asks") or []
        if not ya or not na:
            scanned += 1
            continue

        yes_ask = float(ya[0]["price"])
        no_ask  = float(na[0]["price"])

        # Skip stub books (both sides at placeholder 0.98+)
        if yes_ask >= 0.98 and no_ask >= 0.98:
            scanned += 1
            continue

        edge = 1.0 - yes_ask - no_ask
        scanned += 1

        if edge >= min_edge:
            buy_side  = "YES" if yes_ask <= no_ask else "NO"
            buy_token = yes_id if buy_side == "YES" else no_id
            buy_ask   = yes_ask if buy_side == "YES" else no_ask
            shares    = max(int(min_size), int(target_usd / buy_ask))
            cost      = round(shares * buy_ask, 4)

            hits.append(dict(
                slug=slug,
                edge=edge,
                yes_ask=yes_ask,
                no_ask=no_ask,
                buy_side=buy_side,
                buy_token=buy_token,
                buy_ask=buy_ask,
                shares=shares,
                cost=cost,
                min_size=min_size,
            ))

    # Clear the progress line
    print(f"\r  Scanned {scanned} markets.{' ' * 60}", flush=True)
    return hits


def _print_hits(hits: list[dict]) -> None:
    for h in hits:
        edge_c = h["edge"] * 100
        print(
            f"  {_GREEN}{_BOLD}HIT{_RESET}  "
            f"{h['slug'][:55]}  "
            f"YES={h['yes_ask']}  NO={h['no_ask']}  "
            f"edge={edge_c:.2f}c  "
            f"buy={h['buy_side']}@{h['buy_ask']}  "
            f"{h['shares']}sh=${h['cost']:.2f}"
        )

    best = sorted(hits, key=lambda x: -x["edge"])[0]
    print()
    print(f"  {_BOLD}BEST ENTRY:{_RESET}")
    print(f"    market-slug : {best['slug']}")
    print(f"    buy side    : {best['buy_side']}")
    print(f"    token       : {best['buy_token']}")
    print(f"    ask         : {best['buy_ask']}")
    print(f"    min-size    : {best['shares']} shares")
    print(f"    est. cost   : ${best['cost']:.2f}")
    print(f"    edge        : {best['edge']*100:.2f}c")
    print()
    print(f"  {_CYAN}Run entry command:{_RESET}")
    print(
        f"    python scripts/run_controlled_live.py "
        f"--token {best['buy_token']} "
        f"--ask {best['buy_ask']} "
        f"--min-size {best['shares']} "
        f"--market-slug \"{best['slug']}\" "
        f"--max-open-positions 1"
    )


def run(args: argparse.Namespace) -> None:
    round_num = 0

    while True:
        round_num += 1
        if args.loop:
            print(f"\n{_BOLD}[{_ts()}]  Scan round {round_num}{_RESET}")
        else:
            print(f"\n{_BOLD}[{_ts()}]{_RESET}")

        print(f"  Scanning top {args.limit} markets  "
              f"(min_edge={args.min_edge*100:.1f}c  target=${args.target_usd:.2f}) ...")

        hits = _scan_once(args.limit, args.min_edge, args.target_usd)

        if hits:
            _print_hits(hits)
        else:
            print(f"  {_YELLOW}No qualifying entry found.{_RESET}")

        if not args.loop:
            break

        print(f"  Next scan in {args.interval_sec}s  (Ctrl+C to stop)")
        try:
            time.sleep(args.interval_sec)
        except KeyboardInterrupt:
            print(f"\n{_YELLOW}Scan stopped by user.{_RESET}")
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trial entry scanner — read-only, no orders, no DB writes."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of markets to scan per round, sorted by 24hr volume (default: 100)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge in dollars for a qualifying entry (default: 0.03 = 3 cents)",
    )
    parser.add_argument(
        "--target-usd",
        type=float,
        default=10.0,
        help="Target notional per buy leg in USD, used to size shares (default: 10.0)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep rescanning on --interval-sec until Ctrl+C",
    )
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=30,
        help="Seconds between scans in loop mode (default: 30)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
