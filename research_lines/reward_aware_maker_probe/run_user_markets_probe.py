"""
reward_aware_user_markets_probe
polyarb_lab / diagnostic-only / read-only

Tests all 4 signing combinations against /rewards/user/markets to identify
which format the Polymarket CLOB API accepts:

  Matrix:
    encoding  | timestamp
    ----------+----------
    hexdigest | seconds     (variant A)
    hexdigest | ms          (variant B) ← py-clob-client canonical
    base64    | seconds     (variant C) ← previous impl that WAS working
    base64    | ms          (variant D)

Prints raw response body for every attempt so root cause is visible.
Does NOT modify any state. Read-only.

Usage (Windows PowerShell from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_user_markets_probe.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import logging
import os
import sys
import time
from pathlib import Path

import httpx

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

CLOB_HOST  = "https://clob.polymarket.com"
USER_MKT   = "/rewards/user/markets"
USER_PCT   = "/rewards/user/percentages"


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _sep(c: str = "-", w: int = 72) -> None:
    print(c * w)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


# ---------------------------------------------------------------------------
# Four signature variants
# ---------------------------------------------------------------------------

def _sig_hex(secret: str, ts: str, method: str, path: str) -> str:
    """HMAC-SHA256 hexdigest (py-clob-client canonical)."""
    msg = ts + method.upper() + path
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _sig_b64(secret: str, ts: str, method: str, path: str) -> str:
    """HMAC-SHA256 standard base64 (previous impl)."""
    msg = ts + method.upper() + path
    mac = hmac.new(secret.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _sig_b64url(secret: str, ts: str, method: str, path: str) -> str:
    """HMAC-SHA256 URL-safe base64 (no padding)."""
    msg = ts + method.upper() + path
    mac = hmac.new(secret.encode(), msg.encode(), hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).decode().rstrip("=")


def _make_headers(key: str, secret: str, passphrase: str, path: str,
                  sig_fn, use_ms: bool) -> dict:
    ts = str(int(time.time() * 1000) if use_ms else int(time.time()))
    sig = sig_fn(secret, ts, "GET", path)
    return {
        "CLOB-API-KEY":    key,
        "CLOB-SIGNATURE":  sig,
        "CLOB-TIMESTAMP":  ts,
        "CLOB-PASSPHRASE": passphrase,
    }


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------

def _probe(client: httpx.Client, label: str, url: str, headers: dict,
           params: dict | None = None) -> int:
    """Make one raw request, print full diagnostic, return HTTP status."""
    try:
        resp = client.get(url, headers=headers, params=params or {}, timeout=12)
        st   = resp.status_code
        raw  = resp.text
        print(f"  [{label}]  HTTP {st}")
        if st == 200:
            try:
                d    = resp.json()
                data = d.get("data", []) if isinstance(d, dict) else d
                cnt  = len(data) if isinstance(data, list) else "?"
                keys = list(d.keys()) if isinstance(d, dict) else "list"
                print(f"    OK  keys={keys}  data_count={cnt}")
            except Exception:
                print(f"    OK  text[:200]={raw[:200]}")
        else:
            # Show full diagnostic
            sig_preview = headers.get("CLOB-SIGNATURE", "")[:12] + "..."
            ts_val      = headers.get("CLOB-TIMESTAMP", "")
            ts_len      = len(ts_val)
            print(f"    CLOB-TIMESTAMP={ts_val}  (len={ts_len}, {'ms' if ts_len == 13 else 's' if ts_len == 10 else '?'})")
            print(f"    CLOB-SIGNATURE[:12]={sig_preview}  (len={len(headers.get('CLOB-SIGNATURE',''))})")
            if raw:
                print(f"    raw_response: {raw[:500]}")
            else:
                print(f"    raw_response: (empty)")
            # Also check response headers for clues
            www_auth = resp.headers.get("www-authenticate", "")
            if www_auth:
                print(f"    www-authenticate: {www_auth}")
        return st
    except Exception as exc:
        print(f"  [{label}]  ERROR: {exc}")
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L2 auth diagnostic — tests all signing variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    _configure_logging(args.log_level)

    api_key    = os.environ.get("POLY_API_KEY", "").strip()
    api_secret = os.environ.get("POLY_API_SECRET", "").strip()
    passphrase = os.environ.get("POLY_PASSPHRASE", "").strip()
    wallet     = os.environ.get("POLY_WALLET_ADDRESS", "").strip()

    host = args.clob_host

    print()
    _section("reward_aware_user_markets_probe — L2 auth regression diagnostic")
    print()
    print("  Credentials (presence + length — values NOT shown):")
    print(f"    POLY_API_KEY        : {'SET' if api_key    else '*** MISSING ***'}  len={len(api_key)}")
    print(f"    POLY_API_SECRET     : {'SET' if api_secret else '*** MISSING ***'}  len={len(api_secret)}")
    print(f"    POLY_PASSPHRASE     : {'SET' if passphrase else '*** MISSING ***'}  len={len(passphrase)}")
    print(f"    POLY_WALLET_ADDRESS : {'SET' if wallet     else 'NOT SET (optional)'}  len={len(wallet)}")
    print()
    print(f"    api_key  first 8 chars : {api_key[:8]}...")
    print(f"    api_key  last  4 chars : ...{api_key[-4:]}")
    print(f"    passphrase first 4    : {passphrase[:4]}...")
    print()

    if not (api_key and api_secret and passphrase):
        print("  *** ABORT: required credentials missing ***")
        print("  Set POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE and re-run.")
        return

    url_mkt = f"{host.rstrip('/')}{USER_MKT}"
    url_pct = f"{host.rstrip('/')}{USER_PCT}"

    # -----------------------------------------------------------------------
    # Step 1: Connectivity — unauthenticated baseline
    # -----------------------------------------------------------------------
    _section("Step 1: Unauthenticated connectivity baseline")
    with httpx.Client(follow_redirects=True) as client:
        _probe(client, "GET /markets (unauth)", f"{host}/markets",
               headers={"accept": "application/json"}, params={"limit": "1"})
        _probe(client, "GET /rewards/markets (unauth)",
               f"{host}/rewards/markets", headers={},
               params={"condition_id": "0x1234"})
    print()

    # -----------------------------------------------------------------------
    # Step 2: /rewards/user/markets — 4-variant signing matrix
    # -----------------------------------------------------------------------
    _section("Step 2: /rewards/user/markets — 4-variant L2 signing matrix")
    print("  Testing hexdigest (py-clob-client canonical) vs base64 (previous impl)")
    print("  x seconds vs milliseconds timestamps")
    print()

    results: dict[str, int] = {}

    with httpx.Client(follow_redirects=True) as client:
        variants = [
            ("A: hexdigest + ms  [py-clob-client canonical]",
             _make_headers(api_key, api_secret, passphrase, USER_MKT, _sig_hex, True)),
            ("B: hexdigest + sec [alternative]",
             _make_headers(api_key, api_secret, passphrase, USER_MKT, _sig_hex, False)),
            ("C: base64    + sec [previous impl that was working]",
             _make_headers(api_key, api_secret, passphrase, USER_MKT, _sig_b64, False)),
            ("D: base64    + ms  [tested before → 400]",
             _make_headers(api_key, api_secret, passphrase, USER_MKT, _sig_b64, True)),
            ("E: base64url + ms  [URL-safe base64]",
             _make_headers(api_key, api_secret, passphrase, USER_MKT, _sig_b64url, True)),
        ]

        for label, hdrs in variants:
            st = _probe(client, label, url_mkt, hdrs)
            results[label[0]] = st   # store result by variant letter
            print()

    # -----------------------------------------------------------------------
    # Step 3: /rewards/user/percentages — same variant that worked (if any)
    # -----------------------------------------------------------------------
    _section("Step 3: /rewards/user/percentages — best variant from Step 2")
    winning = [k for k, v in results.items() if v == 200]
    if winning:
        best_label, best_hdrs = next(
            (lbl, hdrs) for lbl, hdrs in variants if lbl[0] in winning
        )
        with httpx.Client(follow_redirects=True) as client:
            _probe(client, f"GET /user/percentages [{best_label}]",
                   url_pct, best_hdrs)
    else:
        print("  No variant returned 200 in Step 2 — all formats rejected.")
        print("  This means the credentials themselves are invalid/expired.")
        print("  Action required: regenerate API key on Polymarket dashboard.")
        with httpx.Client(follow_redirects=True) as client:
            # Still try the canonical variant for completeness
            hdrs_canon = _make_headers(api_key, api_secret, passphrase,
                                       USER_PCT, _sig_hex, True)
            _probe(client, "GET /user/percentages [hexdigest+ms]",
                   url_pct, hdrs_canon)
    print()

    # -----------------------------------------------------------------------
    # Step 4: Summary and root cause diagnosis
    # -----------------------------------------------------------------------
    _section("Step 4: Summary and root-cause diagnosis")
    print(f"  {'Variant':<50}  Status")
    print(f"  {'-'*50}  ------")
    for lbl, hdrs in variants:
        st = results.get(lbl[0], "?")
        ok = "*** 200 OK ***" if st == 200 else f"HTTP {st}"
        print(f"  {lbl:<50}  {ok}")
    print()

    if any(v == 200 for v in results.values()):
        winner = [k for k, v in results.items() if v == 200]
        print(f"  ROOT CAUSE: signature/timestamp format mismatch.")
        print(f"  Working variant(s): {winner}")
        print(f"  Fix: update _build_l2_signature / get_l2_auth_headers to use")
        print(f"       the working variant's encoding and timestamp scale.")
    else:
        print("  ALL VARIANTS FAILED (no 200).")
        print()
        print("  Root cause: credentials are invalid or expired.")
        print("  The 400 response code indicates the server cannot process the")
        print("  request at all — the auth format is recognized but the key/secret")
        print("  pair is rejected before any resource-level check occurs.")
        print()
        print("  Required action:")
        print("  1. Log into Polymarket and go to Account > API Keys")
        print("  2. Delete the current API key")
        print("  3. Create a new API key and copy key/secret/passphrase")
        print("  4. In PowerShell, set the new values:")
        print('     $env:POLY_API_KEY     = "new-key-here"')
        print('     $env:POLY_API_SECRET  = "new-secret-here"')
        print('     $env:POLY_PASSPHRASE  = "new-passphrase-here"')
        print("  5. Re-run this probe to confirm 200")
        print("  6. Then re-run run_maker_presence_scoring.py (without --skip-discovery)")
    print()
    print("reward_aware_user_markets_probe")


if __name__ == "__main__":
    main()
