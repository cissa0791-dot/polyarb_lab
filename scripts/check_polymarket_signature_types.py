"""Test Polymarket Level-2 auth across signature types without placing orders."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.clob_compat import ApiCreds, ClobClient
from src.live.auth import load_live_credentials


CLOB_HOST = "https://clob.polymarket.com"


def main() -> int:
    creds = load_live_credentials()
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None

    print("POLYMARKET SIGNATURE TYPE CHECK")
    print(f"Funder: {funder or '(not set)'}")

    ok_types: list[int] = []
    for signature_type in (0, 1, 2):
        client = ClobClient(
            host=CLOB_HOST,
            chain_id=creds.chain_id,
            key=creds.private_key,
            creds=ApiCreds(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                api_passphrase=creds.api_passphrase,
            ),
            signature_type=signature_type,
            funder=funder,
        )
        try:
            client.get_api_keys()
        except Exception as exc:
            print(f"type {signature_type}: FAIL {exc}")
            continue
        print(f"type {signature_type}: OK")
        ok_types.append(signature_type)

    if not ok_types:
        print("No signature type passed Level-2 auth.")
        return 1

    print("Use:")
    print(f"export POLYMARKET_SIGNATURE_TYPE={ok_types[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
