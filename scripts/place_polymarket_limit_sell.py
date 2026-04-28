"""Place a Polymarket CLOB limit SELL order.

Safe default is dry-run. Add --live to submit a real order.

Example:
    python scripts/place_polymarket_limit_sell.py \
        --token-id 123 \
        --size 21.79 \
        --price 0.464 \
        --live
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.auth import load_live_credentials
from src.live.client import LiveWriteClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place a Polymarket limit SELL order.")
    parser.add_argument("--token-id", required=True, help="Outcome token ID to sell")
    parser.add_argument("--size", type=float, required=True, help="Shares to sell")
    parser.add_argument("--price", type=float, required=True, help="Limit sell price, e.g. 0.464")
    parser.add_argument("--neg-risk", action="store_true", help="Force neg-risk order option")
    parser.add_argument("--tick-size", default=None, help="Optional tick size override, e.g. 0.001")
    parser.add_argument("--live", action="store_true", help="Actually submit the order")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if not 0 < args.price < 1:
        raise SystemExit("--price must be between 0 and 1")

    raw_sig = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type = int(raw_sig) if raw_sig else None
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None
    creds = load_live_credentials()
    client = LiveWriteClient.from_credentials(
        creds,
        "https://clob.polymarket.com",
        dry_run=not args.live,
        signature_type=signature_type,
        funder=funder,
    )

    result = client.submit_order(
        token_id=args.token_id,
        side="SELL",
        price=args.price,
        size=args.size,
        neg_risk=args.neg_risk,
        tick_size=args.tick_size,
    )

    mode = "LIVE" if args.live else "DRY_RUN"
    print("POLYMARKET LIMIT SELL")
    print(f"Mode:         {mode}")
    print(f"Token:        {args.token_id}")
    print(f"Size:         {args.size:.6f}")
    print(f"Price:        {args.price:.6f}")
    print(f"Order ID:     {result.order_id}")
    print(f"Status:       {result.status}")
    print(f"Matched size: {result.size_matched:.6f}")
    if result.avg_price is not None:
        print(f"Avg price:    {result.avg_price:.6f}")
    if not args.live:
        print("Dry-run only. Add --live to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
