"""Print live Polymarket token balance and open orders."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.auth import load_live_credentials
from src.live.client import LiveWriteClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect live Polymarket token state.")
    parser.add_argument("--token-id", required=True, help="Outcome token ID")
    args = parser.parse_args()

    raw_sig = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type = int(raw_sig) if raw_sig else None
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None
    creds = load_live_credentials()
    client = LiveWriteClient.from_credentials(
        creds,
        "https://clob.polymarket.com",
        dry_run=False,
        signature_type=signature_type,
        funder=funder,
    )

    balance = client.get_token_balance(args.token_id)
    orders = client.get_open_orders(args.token_id)
    buys = [order for order in orders if order.side == "BUY"]
    sells = [order for order in orders if order.side == "SELL"]

    print("POLYMARKET TOKEN STATE")
    print(f"Token:      {args.token_id}")
    print(f"Balance:    {balance:.6f}")
    print(f"Open buys:  {len(buys)}")
    for order in buys:
        print(f"  BUY  {order.size_remaining:.6f} @ {order.price:.6f} id={order.order_id}")
    print(f"Open sells: {len(sells)}")
    for order in sells:
        print(f"  SELL {order.size_remaining:.6f} @ {order.price:.6f} id={order.order_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
