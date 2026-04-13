"""
run_ui_reconciliation — UI-source / holder-source reconciliation audit
polyarb_lab / research_lines / auto_maker_loop

Answers why balanceOf(funder, token_id) = 0 even though UI shows 200 YES shares.

Hypothesis being tested
-----------------------
When a maker SELL order is placed on Polymarket, YES tokens are transferred
from the maker's wallet INTO the CTF Exchange contract as collateral for the
open order.  The tokens remain there until the order fills or is cancelled.
Therefore balanceOf(any_user_address, token_id) = 0 while the SELL is open,
because the contract holds the collateral, not the user.

Sources checked
---------------
1. Full raw order JSON — all fields dumped, no truncation
2. Polymarket data-API positions (public, no auth):
     https://data-api.polymarket.com/positions?user=<address>
   Checked for: EOA, FUNDER
3. Polygon RPC balanceOf for the Exchange contract as holder:
     holder = 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E  (CTF Exchange)
   If the exchange holds the token, hypothesis is confirmed.
4. py_clob_client get_positions() / get_portfolio() if available

Does NOT place, cancel, or modify any orders.  Pure read / audit.

Usage
-----
  $env:POLYMARKET_PRIVATE_KEY    = "<key>"
  $env:POLYMARKET_SIGNATURE_TYPE = "2"
  $env:POLYMARKET_FUNDER         = "0x8E5C2ABc4387cC0c5d06AE34B6d49a1AE68c65C1"

  py -3 research_lines/auto_maker_loop/run_ui_reconciliation.py `
       --ask-order-id 0x82a479c351ea91f2904f8e145101626b3657c5d845e615ac761a3cf905f98e15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ui_recon")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOB_HOST = "https://clob.polymarket.com"

SURVIVOR_TOKEN_ID = (
    "94192784911459194325909253314484842244405314804074606736702592885535642919725"
)
HUNGARY_CONDITION_ID = (
    "0x1480b819d03d4b6388d70e848b0384adf38c38d955cb783cdbcf6d4a436dee14"
)

CANDIDATE_EOA    = "0x1B0E607f96f0987A58Ad2245C06d0be452172125"
CANDIDATE_FUNDER = "0x8E5C2ABc4387cC0c5d06AE34B6d49a1AE68c65C1"

# Polymarket CTF Exchange contract (Polygon) — holds tokens for open orders
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
# CTF ERC-1155 contract (Polygon)
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

CTF_DECIMALS = 1_000_000
POLYGON_RPC  = "https://polygon-rpc.com"
BAL_SELECTOR = "0x00fdd58e"

DATA_API_BASE = "https://data-api.polymarket.com"

import requests as _req


# ---------------------------------------------------------------------------
# Credential + client build
# ---------------------------------------------------------------------------

def _load_and_build():
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        load_activation_credentials,
        build_clob_client,
    )
    creds  = load_activation_credentials()
    client = build_clob_client(creds, CLOB_HOST)
    return creds, client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for method in ("model_dump", "dict"):
        if hasattr(obj, method):
            try:
                return getattr(obj, method)()
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"raw": repr(obj)}


def _balanceof_rpc(holder: str, token_id: str, label: str = "") -> tuple[int, float]:
    addr_padded     = holder.lower().replace("0x", "").zfill(64)
    token_id_padded = hex(int(token_id))[2:].zfill(64)
    call_data       = BAL_SELECTOR + addr_padded + token_id_padded
    payload = {
        "jsonrpc": "2.0", "method": "eth_call",
        "params":  [{"to": CTF_CONTRACT, "data": call_data}, "latest"],
        "id": 1,
    }
    try:
        resp = _req.post(POLYGON_RPC, json=payload, timeout=10)
        body = resp.json()
        result_hex  = body.get("result", "0x0") or "0x0"
        raw         = int(result_hex, 16) if result_hex not in ("0x", "0x0", "") else 0
        shares      = round(raw / CTF_DECIMALS, 4)
        return raw, shares
    except Exception as exc:
        logger.warning("RPC balanceOf failed %s: %s", label, exc)
        return -1, -1.0


def _data_api_positions(address: str) -> list[dict]:
    """
    Fetch positions from Polymarket data API for a given address.
    Public endpoint — no auth required.
    Returns list of position dicts, or [] on failure.
    """
    url = f"{DATA_API_BASE}/positions"
    params = {"user": address, "sizeThreshold": "0.01"}
    try:
        resp = _req.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            body = resp.json()
            if isinstance(body, list):
                return body
            if isinstance(body, dict):
                return body.get("data") or body.get("positions") or []
        logger.warning("data-api positions HTTP %d for %s", resp.status_code, address[:12])
        return []
    except Exception as exc:
        logger.warning("data-api positions failed for %s: %s", address[:12], exc)
        return []


def _clob_client_positions(client: Any) -> list[dict]:
    """Try any portfolio/positions method on the CLOB client."""
    results = {}
    for method_name in ("get_positions", "get_portfolio", "get_holdings"):
        if hasattr(client, method_name):
            try:
                raw = getattr(client, method_name)()
                results[method_name] = _to_dict(raw) if not isinstance(raw, list) else raw
            except Exception as exc:
                results[method_name] = f"ERROR: {exc}"
        else:
            results[method_name] = "NOT_PRESENT"
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_reconciliation(ask_order_id: Optional[str]) -> None:
    print("\n" + "=" * 70)
    print("  UI / HOLDER RECONCILIATION AUDIT")
    print("=" * 70)

    # -- Build client -----------------------------------------------------------
    try:
        creds, client = _load_and_build()
    except Exception as exc:
        print(f"\n  FATAL: client build failed — {exc}")
        return

    sig_type = getattr(creds, "signature_type", "?")
    funder   = getattr(creds, "funder", None)
    try:
        signer = client.signer.address()
    except Exception:
        signer = "unavailable"

    print(f"\n  signature_type : {sig_type}")
    print(f"  signer (EOA)   : {signer}")
    print(f"  creds.funder   : {funder or '(none)'}")

    # ==========================================================================
    # SECTION 1 — Full raw order JSON dump
    # ==========================================================================
    print("\n" + "-" * 70)
    print("  SECTION 1 — Full raw order JSON (all fields, no truncation)")
    print("-" * 70)

    if not ask_order_id:
        print("  No --ask-order-id provided — skipping direct order fetch.")
    else:
        print(f"\n  Order ID: {ask_order_id}")
        try:
            raw = client.get_order(ask_order_id)
            if raw is None:
                print("  get_order() returned None — order may be gone (filled/cancelled)")
            else:
                d = _to_dict(raw)
                print("\n  --- BEGIN ORDER JSON ---")
                print(json.dumps(d, indent=4, default=str))
                print("  --- END ORDER JSON ---")

                # Key fields summary
                print("\n  Key field extraction:")
                for key in ("id", "order_id", "asset_id", "token_id", "market",
                            "side", "price", "original_price", "size", "size_matched",
                            "status", "maker_address", "maker", "owner",
                            "outcome", "condition_id", "question_id"):
                    val = d.get(key)
                    if val is not None:
                        print(f"    {key:<20} : {val}")
        except Exception as exc:
            print(f"  ERROR fetching order: {exc}")

    # ==========================================================================
    # SECTION 2 — Polymarket data-API positions (public)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("  SECTION 2 — Polymarket data-API positions")
    print(f"  URL: {DATA_API_BASE}/positions?user=<address>")
    print("-" * 70)

    for label, addr in [("EOA   ", CANDIDATE_EOA), ("FUNDER", CANDIDATE_FUNDER)]:
        print(f"\n  [{label}] {addr}")
        positions = _data_api_positions(addr)
        if not positions:
            print("    No positions returned (empty list or API error)")
        else:
            print(f"    {len(positions)} position(s) returned:")
            for p in positions:
                cid    = p.get("conditionId") or p.get("condition_id") or p.get("market") or "?"
                asset  = p.get("asset") or p.get("token_id") or p.get("asset_id") or "?"
                size   = p.get("size") or p.get("shares") or p.get("balance") or "?"
                price  = p.get("avgPrice") or p.get("avg_price") or p.get("price") or "?"
                title  = str(p.get("title") or p.get("slug") or p.get("market_slug") or "")[:50]
                print(f"    condition : {cid}")
                print(f"    asset     : {asset}")
                print(f"    size      : {size}")
                print(f"    avg_price : {price}")
                print(f"    title     : {title}")
                print(f"    raw keys  : {list(p.keys())}")
                print()

    # ==========================================================================
    # SECTION 3 — balanceOf(CTF_EXCHANGE, token_id) via RPC
    # ==========================================================================
    print("\n" + "-" * 70)
    print("  SECTION 3 — Polygon RPC balanceOf for CTF Exchange contract")
    print("  Hypothesis: tokens locked in exchange as SELL order collateral")
    print("-" * 70)

    print(f"\n  CTF Exchange : {CTF_EXCHANGE}")
    print(f"  token_id     : {SURVIVOR_TOKEN_ID[:30]}…")
    print()

    # Check exchange contract holds the YES tokens
    raw, shares = _balanceof_rpc(CTF_EXCHANGE, SURVIVOR_TOKEN_ID, "exchange/survivor_token")
    flag = "  ← HYPOTHESIS CONFIRMED" if shares >= 190 else ("  ← has balance" if shares > 0 else "")
    print(f"  balanceOf(exchange, SURVIVOR_TOKEN)  raw={raw:>14}  shares={shares:.4f}{flag}")

    # Also complement
    try:
        complement = str((2**256 - 1) - int(SURVIVOR_TOKEN_ID))
        raw_c, shares_c = _balanceof_rpc(CTF_EXCHANGE, complement, "exchange/complement")
        flag_c = "  ← complement has balance" if shares_c > 0 else ""
        print(f"  balanceOf(exchange, COMPLEMENT)      raw={raw_c:>14}  shares={shares_c:.4f}{flag_c}")
    except Exception as exc:
        print(f"  complement check failed: {exc}")

    # Also re-confirm the prior checks inline for reference
    print()
    for hlabel, haddr in [("EOA   ", CANDIDATE_EOA), ("FUNDER", CANDIDATE_FUNDER)]:
        raw, shares = _balanceof_rpc(haddr, SURVIVOR_TOKEN_ID, hlabel)
        print(f"  balanceOf({hlabel}, SURVIVOR_TOKEN)      raw={raw:>14}  shares={shares:.4f}  (prior audit — should = 0)")

    # ==========================================================================
    # SECTION 4 — py_clob_client portfolio methods
    # ==========================================================================
    print("\n" + "-" * 70)
    print("  SECTION 4 — py_clob_client portfolio/position methods")
    print("-" * 70)
    results = _clob_client_positions(client)
    for method, val in results.items():
        print(f"\n  {method}():")
        if isinstance(val, str):
            print(f"    {val}")
        elif isinstance(val, list):
            print(f"    {len(val)} item(s)")
            for item in val[:5]:
                print(f"      {item}")
        else:
            print(f"    {json.dumps(val, indent=6, default=str)[:500]}")

    # ==========================================================================
    # CONCLUSION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("  AUDIT CONCLUSION TEMPLATE")
    print("=" * 70)
    print()
    print("  Exchange balanceOf:")
    print(f"    balanceOf(exchange, survivor_token) = {shares:.4f} shares")
    if shares >= 190:
        print()
        print("  CONFIRMED: tokens are held by CTF Exchange contract.")
        print("  The 200 YES shares are locked as collateral for the open SELL order.")
        print("  balanceOf(user_address, token_id) = 0 is EXPECTED and CORRECT.")
        print()
        print("  Impact on run_bounded_observation.py:")
        print("  - pos=0sh is correct on-chain representation while SELL is open")
        print("  - To track position meaningfully: use data-API positions OR")
        print("    use order status (SELL order LIVE = 200 YES still committed)")
        print("  - The SELL fill IS the correct exit signal — no inventory tracking needed")
    else:
        print()
        print("  Hypothesis NOT confirmed at exchange address.")
        print("  Check Section 1 order JSON and Section 2 data-API results above")
        print("  for alternative explanations.")
    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="UI/holder reconciliation audit")
    p.add_argument("--ask-order-id", default=None,
                   help="Order ID of the live 65c SELL order")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_reconciliation(ask_order_id=args.ask_order_id)
