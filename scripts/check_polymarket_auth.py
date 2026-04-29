"""Check Polymarket CLOB auth setup without placing orders.

This script is intentionally short to run from a VPS web console:

    python scripts/check_polymarket_auth.py

It never prints private keys, API secrets, or passphrases.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.clob_compat import ApiCreds, ClobClient
from src.live.auth import CredentialError, assert_clob_v2_available, load_live_credentials


CLOB_HOST = "https://clob.polymarket.com"
PUBLIC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return value[:4] + "..."
    return value[:8] + "..." + value[-4:]


def _fetch_json(path: str) -> object:
    req = Request(CLOB_HOST + path, headers=PUBLIC_HEADERS)
    with urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def main() -> int:
    print("POLYMARKET AUTH CHECK")

    try:
        assert_clob_v2_available()
        print("CLOB SDK:           V2 OK")
    except CredentialError as exc:
        print("CLOB SDK:           FAIL")
        print(str(exc))
        return 2

    try:
        server_time = _fetch_json("/time")
        print(f"Server time:        {server_time}")
    except Exception as exc:
        print(f"Server time:        FAIL ({exc})")

    local_ts = int(datetime.now(timezone.utc).timestamp())
    print(f"Local UTC time:     {local_ts}")

    try:
        creds = load_live_credentials()
    except CredentialError as exc:
        print("Credentials:        FAIL")
        print(str(exc))
        return 2

    sig_type_raw = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type = int(sig_type_raw) if sig_type_raw else None
    funder = os.environ.get("POLYMARKET_FUNDER", "").strip() or None

    print("Credentials:        OK")
    print(f"Chain ID:           {creds.chain_id}")
    print(f"API key env:        {_mask(creds.api_key)}")
    print(f"Signature type:     {signature_type if signature_type is not None else 'default/0'}")
    print(f"Funder:             {funder or '(not set)'}")

    base_client = ClobClient(
        host=CLOB_HOST,
        chain_id=creds.chain_id,
        key=creds.private_key,
    )
    print(f"Signer address:     {base_client.signer.address()}")

    configured_client = ClobClient(
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
    print(f"Client address:     {configured_client.get_address()}")

    if funder and configured_client.get_address().lower() == base_client.signer.address().lower():
        print("Address note:       Type 2 still signs with the EOA; funder is the proxy/safe wallet.")

    try:
        derived = base_client.derive_api_key()
    except Exception as exc:
        print(f"Derived API key:    FAIL ({exc})")
        return 3

    if derived is None:
        print("Derived API key:    FAIL (no key returned)")
        return 4

    print(f"Derived API key:    {_mask(derived.api_key)}")
    if derived.api_key != creds.api_key:
        print("API key pairing:    FAIL")
        print("Fix: use the API key/secret/passphrase created by this same private key wallet.")
        return 5

    print("API key pairing:    OK")

    try:
        keys = configured_client.get_api_keys()
    except Exception as exc:
        print(f"Level-2 auth:       FAIL ({exc})")
        print("Fix: test POLYMARKET_SIGNATURE_TYPE=0, 1, and 2 with the same funder, then use the one where Level-2 auth is OK.")
        return 6

    key_count = len(keys) if isinstance(keys, list) else "OK"
    print(f"Level-2 auth:       OK ({key_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
