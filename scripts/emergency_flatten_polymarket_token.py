"""Cancel open orders for a token, then place one limit SELL.

This is for emergency inventory cleanup when local bot state is out of sync.
Safe default is dry-run. Add --live to mutate real CLOB orders.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.auth import load_live_credentials
from src.live.client import LiveWriteClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cancel all open orders for a Polymarket token, then place one limit SELL."
    )
    parser.add_argument("--token-id", required=True, help="Outcome token ID to flatten")
    parser.add_argument("--size", required=True, help="Shares to sell, or 'auto' for token balance minus dust")
    parser.add_argument("--price", type=float, required=True, help="Limit sell price, e.g. 0.463")
    parser.add_argument("--neg-risk", action="store_true", help="Force neg-risk order option")
    parser.add_argument("--tick-size", default=None, help="Optional tick size override, e.g. 0.001")
    parser.add_argument("--skip-cancel", action="store_true", help="Do not cancel open token orders first")
    parser.add_argument("--live", action="store_true", help="Actually cancel and submit live orders")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.price < 1:
        raise SystemExit("--price must be between 0 and 1")

    raw_sig = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type = int(raw_sig) if raw_sig else None
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None
    creds = load_live_credentials()

    dry_run = not args.live
    client = LiveWriteClient.from_credentials(
        creds,
        "https://clob.polymarket.com",
        dry_run=dry_run,
        signature_type=signature_type,
        funder=funder,
    )
    if str(args.size).lower() == "auto":
        balance = client.get_token_balance(args.token_id)
        size = max(0.0, int(balance * 1_000_000 - 10) / 1_000_000.0)
    else:
        size = float(args.size)
    if size <= 0:
        raise SystemExit("--size must be positive")

    print("POLYMARKET EMERGENCY FLATTEN")
    print(f"Mode:    {'LIVE' if args.live else 'DRY_RUN'}")
    print(f"Token:   {args.token_id}")
    print(f"Sell:    {size:.6f} @ {args.price:.6f}")

    if not args.skip_cancel:
        if dry_run:
            print("Cancel:  dry-run, would cancel all open orders for token")
        else:
            cancel_result = client.cancel_market_orders(args.token_id)
            print(f"Cancel:  {cancel_result}")

    result = client.submit_order(
        token_id=args.token_id,
        side="SELL",
        price=args.price,
        size=size,
        neg_risk=args.neg_risk,
        tick_size=args.tick_size,
    )

    print(f"Order:   {result.order_id}")
    print(f"Status:  {result.status}")
    print(f"Matched: {result.size_matched:.6f}")
    if result.avg_price is not None:
        print(f"Avg:     {result.avg_price:.6f}")
    if dry_run:
        print("Dry-run only. Add --live to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
