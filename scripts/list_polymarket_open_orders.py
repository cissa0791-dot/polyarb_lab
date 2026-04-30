"""Print all live Polymarket open orders for the authenticated account."""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.auth import load_live_credentials
from src.live.client import LiveWriteClient


def main() -> int:
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

    orders = client.get_all_open_orders()
    grouped: dict[str, list] = defaultdict(list)
    for order in orders:
        grouped[order.token_id or "UNKNOWN"].append(order)

    print("POLYMARKET OPEN ORDERS")
    print(f"Total: {len(orders)}")
    print(f"Buys:  {sum(1 for order in orders if order.side == 'BUY')}")
    print(f"Sells: {sum(1 for order in orders if order.side == 'SELL')}")
    for token_id, token_orders in sorted(grouped.items()):
        print(f"\nToken: {token_id}")
        for order in sorted(token_orders, key=lambda item: (item.side, item.price)):
            notional = order.price * order.size_remaining
            print(
                f"  {order.side:<4} {order.size_remaining:.6f} @ {order.price:.6f} "
                f"notional=${notional:.4f} id={order.order_id}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
