"""
run_holder_audit — narrow token-holder audit
polyarb_lab / research_lines / auto_maker_loop

Answers exactly four questions:
  1. What token_id / asset_id does the live 65c Hungary SELL order use?
  2. What token_id does SURVIVOR_DATA say the Hungary market uses?
  3. For each candidate holder × each candidate token_id:
       balanceOf(holder, token_id) via Polygon RPC — raw + shares
  4. Which (holder, token_id) pair holds the 200 shares?

Does NOT place, cancel, or modify any orders.
Does NOT change strategy or observation logic.
Pure read / audit.

Usage
-----
  $env:POLYMARKET_PRIVATE_KEY   = "<key>"
  $env:POLYMARKET_SIGNATURE_TYPE = "2"
  $env:POLYMARKET_FUNDER         = "0x8E5C2ABc4387cC0c5d06AE34B6d49a1AE68c65C1"

  py -3 research_lines/auto_maker_loop/run_holder_audit.py ^
       --ask-order-id 0xYourAskOrderId
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("holder_audit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# From SURVIVOR_DATA in scoring_activation.py
SURVIVOR_TOKEN_ID = (
    "94192784911459194325909253314484842244405314804074606736702592885535642919725"
)

# Candidate holders
CANDIDATE_HOLDER_EOA    = "0x1B0E607f96f0987A58Ad2245C06d0be452172125"
CANDIDATE_HOLDER_FUNDER = "0x8E5C2ABc4387cC0c5d06AE34B6d49a1AE68c65C1"

# Polygon RPC
POLYGON_RPC     = "https://polygon-rpc.com"
# balanceOf(address,uint256) selector
BAL_SELECTOR    = "0x00fdd58e"
# CTF ERC-1155 contract on Polygon
CTF_CONTRACT    = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
CTF_DECIMALS    = 1_000_000   # 6 decimal places; 1 share = 1_000_000 raw units

# ---------------------------------------------------------------------------
# HTTP — use requests only (no httpx dependency)
# ---------------------------------------------------------------------------

import requests as _req_lib

_session = _req_lib.Session()
_session.headers.update({"User-Agent": "polyarb_lab/holder_audit"})


# ---------------------------------------------------------------------------
# Credential + client build (mirrors scoring_activation.py pattern)
# ---------------------------------------------------------------------------

def _load_creds():
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        load_activation_credentials,
    )
    return load_activation_credentials()


CLOB_HOST = "https://clob.polymarket.com"


def _build_client(creds):
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        build_clob_client,
    )
    return build_clob_client(creds, CLOB_HOST)


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


def _balanceof_rpc(holder: str, token_id: str) -> tuple[int, float]:
    """
    Returns (balance_raw, balance_shares) via Polygon JSON-RPC eth_call.
    balanceOf(address,uint256) selector = 0x00fdd58e
    """
    addr_padded     = holder.lower().replace("0x", "").zfill(64)
    token_id_padded = hex(int(token_id))[2:].zfill(64)
    call_data       = BAL_SELECTOR + addr_padded + token_id_padded

    payload = {
        "jsonrpc": "2.0",
        "method":  "eth_call",
        "params":  [{"to": CTF_CONTRACT, "data": call_data}, "latest"],
        "id":      1,
    }
    try:
        resp = _session.post(POLYGON_RPC, json=payload, timeout=10)
        body = resp.json()
        result_hex  = body.get("result", "0x0") or "0x0"
        balance_raw = int(result_hex, 16) if result_hex not in ("0x", "0x0", "") else 0
        balance_shares = round(balance_raw / CTF_DECIMALS, 4)
        return balance_raw, balance_shares
    except Exception as exc:
        logger.warning("RPC balanceOf failed holder=%s token=%s…: %s",
                       holder[:10], token_id[:20], exc)
        return -1, -1.0


def _fetch_order(client, order_id: str) -> dict:
    try:
        raw = client.get_order(order_id)
        return _to_dict(raw)
    except Exception as exc:
        logger.warning("get_order(%s…) failed: %s", order_id[:16], exc)
        return {}


def _discover_sell_orders(client) -> list[dict]:
    """Fetch all open orders (no market filter) and return SELL orders."""
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw = client.get_orders(OpenOrderParams())
        if isinstance(raw, list):
            orders = [_to_dict(o) for o in raw]
        elif isinstance(raw, dict) and "data" in raw:
            orders = [_to_dict(o) for o in (raw.get("data") or [])]
        else:
            orders = []
        sells = [o for o in orders if str(o.get("side", "")).upper() == "SELL"]
        return sells
    except Exception as exc:
        logger.warning("get_orders() failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(ask_order_id: Optional[str]) -> None:
    # ------------------------------------------------------------------
    # 1. Build client
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  HUNGARY TOKEN-HOLDER AUDIT")
    print("=" * 70)

    try:
        creds  = _load_creds()
        client = _build_client(creds)
    except Exception as exc:
        print(f"\n  FATAL: could not build client — {exc}")
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

    # ------------------------------------------------------------------
    # 2. Retrieve token_id from live SELL order
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SECTION 1 — Order token_id inspection")
    print("-" * 70)

    order_token_id   = None
    order_maker_addr = None
    order_price      = None
    order_keys_shown = False

    if ask_order_id:
        print(f"\n  Fetching order: {ask_order_id}")
        odict = _fetch_order(client, ask_order_id)
        if odict:
            print(f"  order keys: {list(odict.keys())}")
            order_keys_shown = True
            # Extract token_id / asset_id
            order_token_id = (
                odict.get("asset_id")
                or odict.get("token_id")
                or odict.get("market")
            )
            order_maker_addr = (
                odict.get("maker_address")
                or odict.get("maker")
                or odict.get("owner")
            )
            order_price = odict.get("price") or odict.get("original_price")
            print(f"  order asset_id / token_id : {order_token_id}")
            print(f"  order maker_address        : {order_maker_addr}")
            print(f"  order price                : {order_price}")
            print(f"  order status               : {odict.get('status', 'N/A')}")
        else:
            print("  WARNING: get_order returned empty — order may have filled or ID is wrong")
    else:
        print("  No --ask-order-id provided; skipping direct order fetch.")

    # Also scan all open SELL orders for any SELL near 0.65
    print(f"\n  Scanning all open SELL orders...")
    sells = _discover_sell_orders(client)
    print(f"  Total open SELL orders found: {len(sells)}")
    for o in sells:
        price = o.get("price") or o.get("original_price")
        asset = o.get("asset_id") or o.get("token_id") or o.get("market")
        oid   = o.get("id") or o.get("order_id")
        maker = o.get("maker_address") or o.get("maker") or o.get("owner")
        try:
            pf = float(price) if price else 0.0
        except Exception:
            pf = 0.0
        near_target = abs(pf - 0.65) < 0.01
        marker = " ← 65c SELL" if near_target else ""
        print(f"    id={str(oid)[:20]}…  price={price}  asset={str(asset)[:30]}…  maker={str(maker)[:20]}{marker}")
        if near_target and not order_token_id:
            order_token_id   = asset
            order_maker_addr = maker

    # ------------------------------------------------------------------
    # 3. Build candidate (holder, token_id) matrix
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SECTION 2 — Candidate token IDs")
    print("-" * 70)

    token_ids: dict[str, str] = {
        "SURVIVOR_DATA": SURVIVOR_TOKEN_ID,
    }
    if order_token_id and str(order_token_id) != SURVIVOR_TOKEN_ID:
        token_ids["ORDER_ASSET"] = str(order_token_id)
        print(f"  SURVIVOR_DATA token_id : {SURVIVOR_TOKEN_ID}")
        print(f"  ORDER_ASSET  token_id  : {order_token_id}")
        print(f"  *** MISMATCH — will check both ***")
    elif order_token_id:
        print(f"  SURVIVOR_DATA token_id : {SURVIVOR_TOKEN_ID}")
        print(f"  ORDER_ASSET  token_id  : {order_token_id}  (same as SURVIVOR_DATA)")
    else:
        print(f"  SURVIVOR_DATA token_id : {SURVIVOR_TOKEN_ID}")
        print(f"  ORDER_ASSET  token_id  : (not retrieved)")

    # Candidate holders
    holders: dict[str, str] = {
        "EOA   (signer)": CANDIDATE_HOLDER_EOA,
        "FUNDER(proxy) ": CANDIDATE_HOLDER_FUNDER,
    }
    if order_maker_addr and order_maker_addr.lower() not in (
        CANDIDATE_HOLDER_EOA.lower(), CANDIDATE_HOLDER_FUNDER.lower()
    ):
        holders["ORDER_MAKER     "] = order_maker_addr
        print(f"  ORDER_MAKER (novel)   : {order_maker_addr}")

    # ------------------------------------------------------------------
    # 4. balanceOf matrix
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SECTION 3 — Polygon RPC balanceOf matrix")
    print("-" * 70)
    print(f"  CTF contract : {CTF_CONTRACT}")
    print(f"  RPC endpoint : {POLYGON_RPC}")
    print()

    winner_holder  = None
    winner_token   = None
    winner_shares  = 0.0

    # Header
    col_w = 18
    print(f"  {'holder_label':<20}  {'token_label':<14}  {'raw_balance':>14}  {'shares':>10}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*10}")

    for hlabel, haddr in holders.items():
        for tlabel, tid in token_ids.items():
            raw, shares = _balanceof_rpc(haddr, tid)
            flag = ""
            if shares > 0:
                flag = "  ← HAS BALANCE"
            if shares >= 190:
                flag = "  ← *** 200-SHARE CANDIDATE ***"
                if shares > winner_shares:
                    winner_holder = f"{hlabel.strip()} ({haddr})"
                    winner_token  = f"{tlabel} ({tid[:30]}…)"
                    winner_shares = shares
            print(f"  {hlabel:<20}  {tlabel:<14}  {raw:>14}  {shares:>10.4f}{flag}")

    # ------------------------------------------------------------------
    # 5. Conclusion
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AUDIT CONCLUSION")
    print("=" * 70)

    if winner_holder:
        print(f"\n  *** WINNER FOUND ***")
        print(f"  holder : {winner_holder}")
        print(f"  token  : {winner_token}")
        print(f"  shares : {winner_shares:.4f}")
        print()
        print("  ACTION: update _inv_shares in run_bounded_observation.py")
        print("          to use the winning (holder, token_id) pair.")
    else:
        print()
        print("  NO holder/token combination found with >= 190 shares.")
        print()
        print("  Possibilities to investigate:")
        print("  1. Shares held under a DIFFERENT proxy not listed here.")
        print("  2. Token ID is the NO token, not the YES token.")
        print("     Check: complement token_id = 2^256 - 1 - current_token_id")
        print("  3. Shares were sold/transferred since the UI snapshot.")
        print("  4. The position is in a different condition entirely.")
        print()
        # Print complement token_id for YES→NO swap check
        try:
            complement = (2**256 - 1) - int(SURVIVOR_TOKEN_ID)
            print(f"  Complement of SURVIVOR_DATA token_id:")
            print(f"    {complement}")
            print()
            print("  Running balanceOf with complement token_id for both holders...")
            for hlabel, haddr in holders.items():
                raw, shares = _balanceof_rpc(haddr, str(complement))
                flag = "  ← HAS BALANCE" if shares > 0 else ""
                if shares >= 190:
                    flag = "  ← *** COMPLEMENT 200-SHARE CANDIDATE ***"
                print(f"    {hlabel}  complement  raw={raw}  shares={shares:.4f}{flag}")
        except Exception as exc:
            print(f"  (complement check failed: {exc})")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Narrow token-holder audit for Hungary YES position"
    )
    p.add_argument(
        "--ask-order-id",
        default=None,
        help="Order ID of the live 65c SELL order (optional; enables direct order fetch)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_audit(ask_order_id=args.ask_order_id)
