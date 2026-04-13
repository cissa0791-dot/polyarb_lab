"""
reward_aware_order_auth_diagnosis — CLI
polyarb_lab / research_line / diagnostic

Diagnoses the 401 Unauthorized root cause for POST /order.

Steps:
  1. Load credentials from env vars (POLYMARKET_* or POLY_*)
  2. Build L1-only ClobClient (private key only, no api_creds)
  3. Call derive_api_key() — returns the API key registered for the signing address
  4. Compare derived key with configured POLYMARKET_API_KEY
  5. Report: EOA address, key match/mismatch, signature_type + funder env state

Root causes for POST /order 401:
  A. Key mismatch — configured API key is registered for a different address
     (proxy wallet address ≠ EOA from private key)
     Fix: set POLYMARKET_SIGNATURE_TYPE=1 and POLYMARKET_FUNDER=<proxy_addr>

  B. Wrong funder — signature_type=1 but POLYMARKET_FUNDER is wrong or missing
     Fix: set POLYMARKET_FUNDER to the exact proxy wallet address

  C. API key scope — key exists but was derived for a different chain or host
     Fix: re-derive API key against the correct host/chain

Usage (from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_order_auth_diagnosis.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    load_activation_credentials,
    get_missing_credential_vars,
    POLYGON_CHAIN_ID,
)

CLOB_HOST = "https://clob.polymarket.com"


def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


def main() -> None:
    print()
    _section("reward_aware_order_auth_diagnosis")
    print("  Diagnoses POST /order 401 root cause.")
    print()

    # ── Step 1: Load credentials ──────────────────────────────────────────
    _section("Step 1: Credential Load")
    creds = load_activation_credentials()
    missing = get_missing_credential_vars()

    if creds is None:
        print("  STATUS: CREDENTIALS NOT AVAILABLE")
        if missing:
            print(f"  Missing vars: {missing}")
        print()
        print("  Required env vars:")
        print("    POLYMARKET_PRIVATE_KEY   — EVM private key (hex)")
        print("    POLYMARKET_API_KEY       — CLOB L2 API key")
        print("    POLYMARKET_API_SECRET    — CLOB L2 API secret")
        print("    POLYMARKET_API_PASSPHRASE — CLOB L2 passphrase")
        print()
        sys.exit(1)

    print(f"  private_key     : SET (len={len(creds.private_key)})")
    print(f"  api_key         : {creds.api_key[:8]}... (len={len(creds.api_key)})")
    print(f"  chain_id        : {creds.chain_id}")
    print(f"  signature_type  : {creds.signature_type}  "
          f"({'EOA' if creds.signature_type == 0 else 'POLY_PROXY' if creds.signature_type == 1 else 'POLY_GNOSIS_SAFE'})")
    print(f"  funder          : {creds.funder!r}")
    print()

    # ── Step 2: Derive EOA address from private key ───────────────────────
    _section("Step 2: EOA Address from Private Key")
    try:
        from eth_account import Account
        acct = Account.from_key(creds.private_key)
        eoa_address = acct.address
        print(f"  EOA address     : {eoa_address}")
        print()
    except Exception as exc:
        print(f"  ERROR: could not derive EOA address: {exc}")
        print("  Check that POLYMARKET_PRIVATE_KEY is a valid EVM private key (64 hex chars).")
        print()
        sys.exit(1)

    # ── Step 3: L1-only ClobClient + derive_api_key ───────────────────────
    _section("Step 3: derive_api_key() — API Key Registered for EOA")
    try:
        from py_clob_client.client import ClobClient
        l1_client = ClobClient(
            host=CLOB_HOST,
            chain_id=creds.chain_id,
            key=creds.private_key,
        )
        derived = l1_client.derive_api_key()
        if isinstance(derived, dict):
            derived_key = (
                derived.get("apiKey")
                or derived.get("api_key")
                or derived.get("key")
                or str(derived)
            )
        else:
            derived_key = str(derived) if derived else None

        if derived_key:
            print(f"  derived api_key : {derived_key[:8]}... (len={len(derived_key)})")
        else:
            print(f"  derived api_key : None (no key registered for {eoa_address})")
        print()
    except Exception as exc:
        print(f"  ERROR: derive_api_key() failed: {exc}")
        derived_key = None
        print()

    # ── Step 4: Compare ───────────────────────────────────────────────────
    _section("Step 4: Key Pairing Check")
    configured_key = creds.api_key

    if derived_key is None:
        print("  RESULT: CANNOT COMPARE — derive_api_key() returned None or failed.")
        print()
        print("  Possible causes:")
        print("    - No API key has been registered for this EOA address")
        print("    - Private key is wrong / does not match expected wallet")
        print("    - Network error reaching CLOB host")
        match = None
    elif derived_key == configured_key:
        print("  RESULT: MATCH — configured POLYMARKET_API_KEY matches EOA-derived key.")
        print("  The 401 is NOT caused by a key/address mismatch.")
        print("  Check: POLYMARKET_FUNDER if using proxy wallet, or try --dry-run to verify L2 reads.")
        match = True
    else:
        print("  RESULT: MISMATCH — configured API key does NOT match EOA-derived key.")
        print()
        print(f"  Configured : {configured_key[:12]}...")
        print(f"  Derived    : {derived_key[:12]}...")
        print()
        print("  ROOT CAUSE: Your POLYMARKET_API_KEY is registered for a proxy wallet,")
        print("  not the EOA address from your private key. This is the standard setup")
        print("  for Polymarket web-app users (POLY_PROXY wallet).")
        print()
        print("  FIX:")
        print("    1. Find your proxy wallet address in the Polymarket web app")
        print("       (Settings → Wallet → Contract address / proxy address)")
        print("    2. Set env vars:")
        print("         $env:POLYMARKET_SIGNATURE_TYPE = '1'")
        print("         $env:POLYMARKET_FUNDER = '<your_proxy_wallet_address>'")
        print("    3. Re-run: py -3 research_lines/reward_aware_maker_probe/run_scoring_activation.py --live")
        match = False
    print()

    # ── Step 5: Env var state for signature_type / funder ────────────────
    _section("Step 5: Signature Type + Funder Env State")
    sig_type_env = (
        os.environ.get("POLYMARKET_SIGNATURE_TYPE")
        or os.environ.get("POLY_SIGNATURE_TYPE")
    )
    funder_env = (
        os.environ.get("POLYMARKET_FUNDER")
        or os.environ.get("POLY_FUNDER")
    )
    print(f"  POLYMARKET_SIGNATURE_TYPE : {sig_type_env!r}  (effective={creds.signature_type})")
    print(f"  POLYMARKET_FUNDER         : {funder_env!r}")
    print()

    if creds.signature_type == 1 and not creds.funder:
        print("  WARNING: signature_type=1 (POLY_PROXY) but POLYMARKET_FUNDER is not set.")
        print("  The POST /order POLY_ADDRESS header will use EOA address — will cause 401.")
        print("  Set POLYMARKET_FUNDER to your proxy wallet address.")
        print()
    elif creds.signature_type == 0 and match is False:
        print("  RECOMMENDATION: Set POLYMARKET_SIGNATURE_TYPE=1 and POLYMARKET_FUNDER=<proxy_addr>.")
        print()

    _sep("=")
    if match is True:
        print("  DIAGNOSIS: Auth credentials appear correctly paired.")
        print("  If 401 persists: check POLYMARKET_FUNDER or try re-deriving API key.")
    elif match is False:
        print("  DIAGNOSIS: Key mismatch — POLY_PROXY wallet fix required (see Step 4).")
    else:
        print("  DIAGNOSIS: Inconclusive — derive_api_key() did not return a key.")
    _sep("=")
    print()
    print("reward_aware_maker_probe")


if __name__ == "__main__":
    main()
