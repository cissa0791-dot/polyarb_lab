"""End-to-end API integration verification script.

Runs 6 verification steps against the real Polymarket API using your live
credentials.  Default mode: dry_run=True (no real orders submitted).

Prerequisites:
    Set these environment variables before running:
        POLYMARKET_PRIVATE_KEY      Your EVM wallet private key (hex)
        POLYMARKET_API_KEY          Polymarket CLOB Level-2 API key
        POLYMARKET_API_SECRET       Polymarket CLOB Level-2 API secret
        POLYMARKET_API_PASSPHRASE   Polymarket CLOB Level-2 API passphrase
        POLYMARKET_CHAIN_ID         137  (Polygon mainnet)

Usage:
    # Dry-run (default — safe, no real orders):
    python scripts/verify_api_integration.py

    # Live order test (submits a tiny real order!):
    python scripts/verify_api_integration.py --live

    # Target testnet instead:
    python scripts/verify_api_integration.py --chain-id 80002
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any

# Ensure src/ is importable from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from src.live.auth import CredentialError, LiveCredentials, load_live_credentials
from src.live.client import LiveClientError, LiveWriteClient
from src.live.rewards import RewardClient, RewardClientError
from src.ingest.clob import ReadOnlyClob
from src.ingest.gamma import fetch_markets

_CLOB_HOST  = "https://clob.polymarket.com"
_GAMMA_HOST = "https://gamma-api.polymarket.com"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓ PASS{_RESET}  {msg}")

def _fail(msg: str) -> None:
    print(f"  {_RED}✗ FAIL{_RESET}  {msg}")

def _warn(msg: str) -> None:
    print(f"  {_YELLOW}! WARN{_RESET}  {msg}")

def _step(n: int | str, title: str) -> None:
    print(f"\n{_BOLD}[Step {n}] {title}{_RESET}")

def _detail(key: str, value: Any) -> None:
    print(f"         {key}: {value}")


# ---------------------------------------------------------------------------
# Dynamic market discovery
# ---------------------------------------------------------------------------

def _discover_live_token() -> tuple[str, str, bool, float] | None:
    """Return (token_id, tick_size, neg_risk, min_order_size) for a currently active market.

    Queries the Gamma API for active non-closed markets, then confirms the
    chosen token has a non-empty CLOB order book.  Returns None if no viable
    token is found.
    """
    try:
        resp = httpx.get(
            f"{_GAMMA_HOST}/markets",
            params={"limit": 20, "active": "true", "closed": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        markets = resp.json()
        if not isinstance(markets, list):
            markets = markets.get("markets", [])
    except Exception as exc:
        print(f"         [discover] Gamma fetch failed: {exc}")
        return None

    for market in markets:
        raw_ids = market.get("clobTokenIds") or market.get("clob_token_ids") or []
        if isinstance(raw_ids, str):
            try:
                import json as _j
                raw_ids = _j.loads(raw_ids)
            except Exception:
                continue
        if not raw_ids:
            continue

        token_id = str(raw_ids[0])

        # Confirm non-empty book on the CLOB.
        try:
            br = httpx.get(
                f"{_CLOB_HOST}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            if br.status_code != 200:
                continue
            bk = br.json()
            if not bk.get("bids") and not bk.get("asks"):
                continue
        except Exception:
            continue

        # Fetch tick size and neg_risk from the CLOB.
        try:
            tr = httpx.get(f"{_CLOB_HOST}/tick-size", params={"token_id": token_id}, timeout=8)
            tick_size = str(tr.json().get("minimum_tick_size", "0.01")) if tr.status_code == 200 else "0.01"
            nr = httpx.get(f"{_CLOB_HOST}/neg-risk", params={"token_id": token_id}, timeout=8)
            neg_risk = bool(nr.json().get("neg_risk", False)) if nr.status_code == 200 else False
        except Exception:
            tick_size, neg_risk = "0.01", False

        min_order_size = float(market.get("orderMinSize") or 5)

        question = str(market.get("question", ""))[:60]
        print(f"         [discover] Selected market: {question!r}")
        print(f"         [discover] token_id={token_id[:20]}...  tick={tick_size}  neg_risk={neg_risk}  min_size={min_order_size}")
        return token_id, tick_size, neg_risk, min_order_size

    return None


# ---------------------------------------------------------------------------
# Verification steps
# ---------------------------------------------------------------------------

def step1_credentials() -> LiveCredentials | None:
    """Load and validate credentials from environment variables."""
    _step(1, "API Authentication — credential loading")
    try:
        creds = load_live_credentials()
        _ok(f"All 5 env vars present  {creds!r}")
        return creds
    except CredentialError as exc:
        _fail(str(exc))
        return None


def step1b_validate_api_key_pairing(creds: LiveCredentials) -> bool:
    """Derive the canonical API key for this private key and compare with env var.

    Uses derive_api_key() — a true Level-1 call (private key only, no ApiCreds).
    Hits GET /auth/derive-api-key with an L1-signed header and returns the
    api_key/secret/passphrase registered for this private key's address.

    This definitively separates credential-pairing mismatch from any other cause:
      PASS — the derived apiKey matches POLYMARKET_API_KEY → pairing confirmed
      FAIL — mismatch or no key exists → must regenerate credentials
    """
    _step("1b", "API key pairing — derive canonical key for this private key (L1)")
    try:
        from src.live.clob_compat import ClobClient as _ClobClient
        # Level-1 client: private key only, no ApiCreds.
        l1_client = _ClobClient(
            host=_CLOB_HOST,
            chain_id=creds.chain_id,
            key=creds.private_key,
        )
        wallet_address = l1_client.signer.address()
        _detail("wallet address (derived from POLYMARKET_PRIVATE_KEY)", wallet_address)

        derived = l1_client.derive_api_key()

        if derived is None:
            _fail("derive_api_key() returned None — no API key exists for this private key.")
            _warn("Go to https://polymarket.com → Profile → API Keys and create a key,")
            _warn("then copy the key/secret/passphrase into your env vars.")
            return False

        derived_key = derived.api_key
        configured_key = creds.api_key

        _detail("derived apiKey  (from private key)", f"{derived_key[:8]}...{derived_key[-4:]}")
        _detail("configured key  (POLYMARKET_API_KEY)", f"{configured_key[:8]}...{configured_key[-4:]}")

        if derived_key == configured_key:
            _ok("Derived apiKey matches POLYMARKET_API_KEY — credential pairing confirmed")
            _warn("If order placement still returns 401, check POLYMARKET_SIGNATURE_TYPE / POLYMARKET_FUNDER")
            return True
        else:
            _fail("Derived apiKey does NOT match POLYMARKET_API_KEY.")
            _fail("The API key/secret/passphrase in env are from a different credential set.")
            _warn("Fix: set env vars to the values derived here, or re-derive via the Polymarket UI.")
            _detail("correct key to use", derived_key)
            _detail("correct secret",     "(re-run derive_api_key() to retrieve — not shown here)")
            return False

    except Exception as exc:
        _fail(f"step1b failed: {exc}")
        traceback.print_exc()
        return False


def step2_clob_connectivity(creds: LiveCredentials) -> bool:
    """Fetch one order book to confirm CLOB read access."""
    _step(2, "CLOB connectivity — read order book")
    try:
        clob = ReadOnlyClob(host=_CLOB_HOST)
        # Use a well-known liquid token (Trump 2024 winner YES token).
        # This is a public read — no auth required.
        TOKEN_ID = (
            "21742633143463906290569050155826241533067272736897614950488156847919"
            "9340557"
        )
        books = clob.get_books([TOKEN_ID])
        if books:
            book = books[0]
            _ok(f"Order book fetched  bids={len(book.bids)}  asks={len(book.asks)}")
            _detail("token_id (truncated)", TOKEN_ID[:20] + "...")
        else:
            _warn("Book returned empty — token may be resolved.  CLOB is reachable.")
        return True
    except Exception as exc:
        _fail(f"CLOB read failed: {exc}")
        return False


def step3_order_placement_dryrun(creds: LiveCredentials) -> bool:
    """Submit a dry-run order — validates auth headers without real submission."""
    _step(3, "Order placement — dry_run=True (no real order)")
    try:
        write_client = LiveWriteClient.from_credentials(
            creds, host=_CLOB_HOST, dry_run=True
        )
        TOKEN_ID = (
            "21742633143463906290569050155826241533067272736897614950488156847919"
            "9340557"
        )
        result = write_client.submit_order(
            token_id=TOKEN_ID,
            side="BUY",
            price=0.01,    # Far from touch — would never fill if real
            size=1.0,
        )
        _ok(f"submit_order returned  status={result.status!r}  dry_run={result.dry_run}")
        return True
    except LiveClientError as exc:
        _fail(f"submit_order failed: {exc}")
        return False
    except Exception as exc:
        _fail(f"Unexpected error: {exc}")
        traceback.print_exc()
        return False


def step3b_order_placement_live(creds: LiveCredentials) -> str | None:
    """Submit a REAL tiny order.  Only called when --live flag is set."""
    _step(3, "Order placement — LIVE (submitting real order!)")
    print(f"  {_YELLOW}WARNING: This will submit a real order to Polymarket.{_RESET}")

    discovered = _discover_live_token()
    if discovered is None:
        _fail("Could not discover an active liquid market — aborting live order step")
        return None
    token_id, tick_size, neg_risk, min_order_size = discovered

    # Read optional proxy-wallet overrides from env.
    # POLYMARKET_SIGNATURE_TYPE: 0=EOA (default), 1=POLY_PROXY (most web users), 2=POLY_GNOSIS_SAFE
    # POLYMARKET_FUNDER: proxy wallet address shown on polymarket.com (if not set, defaults to EOA)
    raw_sig_type = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    signature_type: int | None = int(raw_sig_type) if raw_sig_type else None
    funder: str | None = os.environ.get("POLYMARKET_FUNDER", "").strip() or None

    _detail("signature_type", signature_type if signature_type is not None else "None → EOA(0)")
    _detail("funder",         funder or "None → EOA address of private key")

    try:
        write_client = LiveWriteClient.from_credentials(
            creds, host=_CLOB_HOST, dry_run=False,
            signature_type=signature_type,
            funder=funder,
        )
        result = write_client.submit_order(
            token_id=token_id,
            side="BUY",
            price=0.01,            # Deep out-of-money — will rest as maker order
            size=min_order_size,   # Live minimum from Gamma orderMinSize
            neg_risk=neg_risk,
            tick_size=tick_size,
        )
        if result.order_id:
            _ok(f"Real order submitted  order_id={result.order_id}  status={result.status!r}")
            return result.order_id
        else:
            _warn(f"Order submitted but no order_id returned  status={result.status!r}")
            return None
    except LiveClientError as exc:
        _fail(f"Live submit_order failed: {exc}")
        return None


def step4_order_status(creds: LiveCredentials, order_id: str) -> bool:
    """Check the status of a previously submitted order."""
    _step(4, f"Order status tracking — order_id={order_id[:12]}...")
    try:
        write_client = LiveWriteClient.from_credentials(
            creds, host=_CLOB_HOST, dry_run=False
        )
        status = write_client.get_order_status(order_id)
        _ok(
            f"status={status.status!r}  "
            f"matched={status.size_matched}  "
            f"remaining={status.size_remaining}"
        )
        return True
    except LiveClientError as exc:
        _fail(f"get_order_status failed: {exc}")
        return False


def step5_cancel_order(creds: LiveCredentials, order_id: str) -> bool:
    """Cancel the test order placed in step 3b."""
    _step(5, f"Order cancellation — order_id={order_id[:12]}...")
    try:
        write_client = LiveWriteClient.from_credentials(
            creds, host=_CLOB_HOST, dry_run=False
        )
        ok = write_client.cancel_order(order_id)
        if ok:
            _ok("Cancellation request accepted by CLOB")
        else:
            _warn("cancel_order returned False (order may already be terminal)")
        return True
    except LiveClientError as exc:
        _fail(f"cancel_order failed: {exc}")
        return False


def step6_rewards(creds: LiveCredentials) -> bool | None:
    """Fetch the current epoch info and user reward stats.

    Returns:
        True  — rewards data retrieved successfully.
        None  — endpoint unavailable (405 / no confirmed path); non-blocking.
        False — unexpected failure.
    """
    _step(6, "Reward retrieval — current epoch + user stats")
    try:
        reward_client = RewardClient.from_credentials(creds, dry_run=False)
        _detail("wallet address", reward_client.address)

        summary = reward_client.get_rewards_summary()

        _ok(f"Epoch {summary['epoch_id']}  "
            f"{summary['epoch_start']} → {summary['epoch_end']}")
        _detail("epoch total pool", f"${summary['epoch_total_pool_usd']:,.2f}")
        _detail("your earned",      f"${summary['user_earned_usd']:,.4f}")
        _detail("your maker vol",   f"${summary['user_maker_vol_usd']:,.2f}")
        _detail("your share",       f"{summary['user_share_pct']:.4f}%")
        return True
    except RewardClientError as exc:
        msg = str(exc)
        if "405" in msg:
            _warn("REWARDS_API_UNVERIFIED — endpoint returned 405 (Method Not Allowed)")
            _warn("No confirmed public rewards endpoint available on clob.polymarket.com")
            _warn("Core order lifecycle (steps 1–5) is unaffected by this.")
            return None   # non-blocking
        _fail(f"Reward retrieval failed: {exc}")
        return False
    except Exception as exc:
        _fail(f"Unexpected error in reward retrieval: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket API integration verifier")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Submit a real tiny order (step 3b) and cancel it (step 5). "
             "Without this flag, only dry-run tests are performed.",
    )
    parser.add_argument(
        "--chain-id",
        type=int,
        default=None,
        help="Override POLYMARKET_CHAIN_ID (e.g. 80002 for Amoy testnet).",
    )
    args = parser.parse_args()

    if args.chain_id is not None:
        os.environ["POLYMARKET_CHAIN_ID"] = str(args.chain_id)
        print(f"Override: POLYMARKET_CHAIN_ID={args.chain_id}")

    print(f"\n{_BOLD}Polymarket API Integration Verification{_RESET}")
    print("=" * 50)
    if args.live:
        print(f"{_YELLOW}Mode: LIVE — real orders will be submitted and cancelled{_RESET}")
    else:
        print("Mode: DRY-RUN — no real orders will be submitted")

    results: dict[str, bool] = {}

    # Step 1 — always required.
    creds = step1_credentials()
    results["credentials"] = creds is not None
    if creds is None:
        _print_final(results)
        return

    # Step 1b — API key pairing check (L1 auth, no ApiCreds needed).
    results["api_key_pairing"] = step1b_validate_api_key_pairing(creds)

    # Step 2 — CLOB read (no auth needed).
    results["clob_read"] = step2_clob_connectivity(creds)

    live_order_id: str | None = None

    if args.live:
        # Steps 3b, 4, 5 — real order lifecycle.
        live_order_id = step3b_order_placement_live(creds)
        results["order_place_live"] = live_order_id is not None

        if live_order_id:
            time.sleep(1)   # Give CLOB a moment to register the order.
            results["order_status"] = step4_order_status(creds, live_order_id)
            time.sleep(1)
            results["order_cancel"] = step5_cancel_order(creds, live_order_id)
        else:
            results["order_status"] = False
            results["order_cancel"] = False
    else:
        # Step 3 — dry-run order.
        results["order_place_dryrun"] = step3_order_placement_dryrun(creds)
        print("\n  (Steps 4 & 5 skipped in dry-run mode — run with --live to test them)")

    # Step 6 — rewards (always attempted).
    results["rewards"] = step6_rewards(creds)

    _print_final(results)


def _print_final(results: dict[str, bool | None]) -> None:
    passed   = sum(1 for v in results.values() if v is True)
    unverif  = sum(1 for v in results.values() if v is None)
    total    = len(results)
    print(f"\n{_BOLD}{'=' * 50}")
    print(f"Result: {passed}/{total} checks passed  ({unverif} unverified / non-blocking){_RESET}")

    for name, ok in results.items():
        if ok is True:
            icon = f"{_GREEN}✓{_RESET}"
            label = ""
        elif ok is None:
            icon = f"{_YELLOW}~{_RESET}"
            label = "  [UNVERIFIED — non-blocking]"
        else:
            icon = f"{_RED}✗{_RESET}"
            label = ""
        print(f"  {icon}  {name}{label}")

    failed = [k for k, v in results.items() if v is False]
    if not failed:
        suffix = " (rewards unverified — see above)" if unverif else ""
        print(f"\n{_GREEN}{_BOLD}Core checks passed — API integration is working.{_RESET}{suffix}")
    else:
        print(f"\n{_RED}Failed checks: {', '.join(failed)}{_RESET}")
        print("Fix the issues above, then re-run this script.")

    print()


if __name__ == "__main__":
    main()
