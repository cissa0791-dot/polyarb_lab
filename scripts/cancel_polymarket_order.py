"""Cancel a Polymarket CLOB order by order ID.

Usage:
    python scripts/cancel_polymarket_order.py 0x...
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.auth import build_authenticated_client, load_live_credentials


def main() -> int:
    parser = argparse.ArgumentParser(description="Cancel a Polymarket CLOB order.")
    parser.add_argument("order_id", help="CLOB order ID to cancel")
    args = parser.parse_args()

    creds = load_live_credentials()
    raw_sig = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type = int(raw_sig) if raw_sig else None
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None

    client = build_authenticated_client(
        creds,
        "https://clob.polymarket.com",
        signature_type=signature_type,
        funder=funder,
    )
    result = client.cancel(args.order_id)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
