"""
reward_aware_inventory_seed — feasibility probe
polyarb_lab / research_line / diagnostic

Read-only feasibility check for conditional token inventory seeding.

Prints:
  1. USDC.e collateral balance + CLOB exchange allowance (via CLOB API)
  2. Conditional token balance + CLOB allowance for Hungary YES token (via CLOB API)
  3. USDC.e → CTF approval from proxy wallet (raw eth_call, not covered by CLOB API)
  4. splitPosition calldata preview for the required seed (no broadcast)
  5. Relayer availability probe
  6. Verdict: RELAYER_PATH_AVAILABLE | CTF_APPROVAL_MISSING | RAW_SAFE_TX_REQUIRED

Verdict logic:
  RELAYER_PATH_AVAILABLE : Polymarket relayer client installed + CTF approval present
  CTF_APPROVAL_MISSING   : Relayer installed but USDC.e → CTF allowance is zero
  RAW_SAFE_TX_REQUIRED   : No relayer client; must use raw Safe execTransaction

Note on CLOB API balance calls:
  get_balance_allowance(COLLATERAL)          — USDC.e balance + CLOB exchange allowance
  get_balance_allowance(CONDITIONAL,token_id) — conditional token balance + CLOB exchange allowance
  Neither call returns the USDC.e → CTF approval; that requires a direct eth_call.

Current environment:
  py_builder_signing_sdk=0.0.2 (HMAC signing only, no gasless/relayer capability)
  web3.py: not installed
  Expected verdict: RAW_SAFE_TX_REQUIRED

Usage (from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_inventory_seed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    load_activation_credentials,
    get_missing_credential_vars,
    build_clob_client,
    POLYGON_CHAIN_ID,
)

CLOB_HOST = "https://clob.polymarket.com"

# Hungary market — primary scoring activation target ($150/day reward)
HUNGARY_CONDITION_ID = "0x1480b819d03d4b6388d70e848b0384adf38c38d955cb783cdbcf6d4a436dee14"
HUNGARY_TOKEN_ID = "94192784911459194325909253314484842244405314804074606736702592885535642919725"
SEED_SHARES = 200.0  # rewards_min_size (SELL leg requires this many YES tokens)

# Polygon contract addresses (chain 137) — from py_clob_client/config.py
CTF_ADDRESS   = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # ConditionalTokens ERC1155
USDC_ADDRESS  = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (collateral)
EXCHANGE_ADDR = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # CLOB exchange

# Public Polygon JSON-RPC (no API key required)
POLYGON_RPC = "https://polygon-rpc.com"

# Token decimals
USDC_DECIMALS = 1_000_000  # 6 decimals
CTF_DECIMALS  = 1_000_000  # 6 decimals (Polymarket conditional tokens)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


# ---------------------------------------------------------------------------
# Raw JSON-RPC helpers (httpx only, no web3)
# ---------------------------------------------------------------------------

def _eth_call(rpc_url: str, to: str, data: str) -> str:
    """
    Execute a read-only eth_call via JSON-RPC.
    Returns the hex result string, or raises on network/RPC error.
    """
    import httpx
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
    }
    r = httpx.post(rpc_url, json=payload, timeout=10.0)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"eth_call RPC error: {body['error']}")
    return body.get("result", "0x")


def _encode_allowance_call(owner: str, spender: str) -> str:
    """
    Encode ERC20.allowance(address owner, address spender) calldata.
    Selector: keccak256("allowance(address,address)")[:4] = 0xdd62ed3e
    """
    from eth_abi import encode as abi_encode
    selector = bytes.fromhex("dd62ed3e")
    args = abi_encode(["address", "address"], [owner, spender])
    return "0x" + selector.hex() + args.hex()


def _decode_uint256_result(hex_result: str) -> int:
    """Decode a single uint256 from an eth_call hex result."""
    from eth_abi import decode as abi_decode
    if not hex_result or hex_result in ("0x", "0x0"):
        return 0
    padded = hex_result.lstrip("0x").zfill(64)
    (val,) = abi_decode(["uint256"], bytes.fromhex(padded))
    return int(val)


def _encode_split_position_calldata(
    collateral_token: str,
    condition_id_hex: str,
    amount_raw: int,
) -> str:
    """
    Encode CTF.splitPosition calldata for a top-level YES/NO split.

    Solidity signature:
      splitPosition(address collateralToken, bytes32 parentCollectionId,
                    bytes32 conditionId, uint256[] partition, uint256 amount)

    For top-level YES/NO:
      parentCollectionId = bytes32(0x00...00)
      partition = [1, 2]  (YES = bit 0, NO = bit 1 → bitmask values 1 and 2)
      amount = SEED_SHARES in 6-decimal units
    """
    from eth_abi import encode as abi_encode
    from eth_utils import keccak

    sig = "splitPosition(address,bytes32,bytes32,uint256[],uint256)"
    selector = keccak(text=sig)[:4]

    parent_collection_id = b"\x00" * 32
    condition_id_bytes   = bytes.fromhex(condition_id_hex.lstrip("0x"))
    partition            = [1, 2]

    args = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]", "uint256"],
        [
            collateral_token,
            parent_collection_id,
            condition_id_bytes,
            partition,
            amount_raw,
        ],
    )
    return "0x" + selector.hex() + args.hex()


# ---------------------------------------------------------------------------
# Relayer availability probe
# ---------------------------------------------------------------------------

def _check_relayer_available() -> dict:
    """
    Probe for installed Polymarket relayer clients.
    Returns {"available": bool, "method": str, "detail": str}
    """
    for mod_name in ("polymarket_relayer", "relayer_client", "poly_relayer"):
        try:
            __import__(mod_name)
            return {
                "available": True,
                "method": mod_name,
                "detail": f"import {mod_name!r} succeeded",
            }
        except ImportError:
            pass

    # web3.py present — raw tx is feasible but still requires Safe execution
    try:
        import web3  # noqa: F401
        return {
            "available": False,
            "method": "web3_raw_tx",
            "detail": "web3.py present but is not a relayer — Safe execTransaction still required",
        }
    except ImportError:
        pass

    return {
        "available": False,
        "method": "none",
        "detail": (
            "py_builder_signing_sdk=0.0.2 (HMAC signing only). "
            "web3.py not installed. "
            "No gasless split path available."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    _section("reward_aware_inventory_seed — feasibility probe")
    print("  Read-only. No transactions submitted. Dry-run only.")
    print()

    # ── Step 1: Credentials ───────────────────────────────────────────────
    _section("Step 1: Credentials")
    creds = load_activation_credentials()
    if creds is None:
        missing = get_missing_credential_vars()
        print("  STATUS: CREDENTIALS NOT AVAILABLE")
        if missing:
            print(f"  Missing vars: {missing}")
        sys.exit(1)

    print(f"  api_key        : {creds.api_key[:8]}...")
    print(f"  signature_type : {creds.signature_type}")
    print(f"  funder (proxy) : {creds.funder!r}")
    print()

    proxy_addr = creds.funder
    if not proxy_addr:
        print("  ERROR: POLYMARKET_FUNDER not set.")
        print("  Proxy wallet address is required to check on-chain balances.")
        print("  Set: $env:POLYMARKET_FUNDER = '<your_proxy_wallet_address>'")
        sys.exit(1)

    # ── Step 2: Build CLOB client ─────────────────────────────────────────
    _section("Step 2: CLOB Client")
    try:
        client = build_clob_client(creds, CLOB_HOST)
        print("  client: ok")
        print()
    except Exception as exc:
        print(f"  ERROR: build_clob_client failed: {exc}")
        sys.exit(1)

    # Import once, reuse in steps 3 and 4
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams
    except ImportError as exc:
        print(f"  ERROR: cannot import BalanceAllowanceParams: {exc}")
        sys.exit(1)

    # ── Step 3: Collateral balance (USDC.e, via CLOB API) ─────────────────
    _section("Step 3: Collateral Balance — USDC.e (CLOB API)")
    collateral_balance_raw: int = -1
    try:
        resp = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=-1)
        )
        collateral_balance_raw   = int(float(resp.get("balance",   0) or 0))
        collateral_allowance_raw = int(float(resp.get("allowance", 0) or 0))

        print(f"  balance_raw       : {collateral_balance_raw}")
        print(f"  balance_usdc      : {collateral_balance_raw / USDC_DECIMALS:.6f}")
        print(f"  allowance_raw     : {collateral_allowance_raw}  (CLOB exchange)")
        print(f"  allowance_usdc    : {collateral_allowance_raw / USDC_DECIMALS:.6f}")
        print(f"  seed_cost_usdc    : {SEED_SHARES:.2f}  (need this much for splitPosition)")
        collateral_sufficient = collateral_balance_raw >= int(SEED_SHARES * USDC_DECIMALS)
        print(f"  collateral_ok     : {collateral_sufficient}")
        print()
    except Exception as exc:
        print(f"  ERROR: get_balance_allowance(COLLATERAL) failed: {exc}")
        print()

    # ── Step 4: Conditional balance for Hungary YES token (CLOB API) ──────
    _section("Step 4: Conditional Balance — Hungary YES token (CLOB API)")
    print(f"  condition_id : {HUNGARY_CONDITION_ID}")
    print(f"  token_id     : {HUNGARY_TOKEN_ID[:24]}...")
    conditional_balance_raw: int  = -1
    conditional_allowance_raw: int = -1
    try:
        resp2 = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type="CONDITIONAL",
                token_id=HUNGARY_TOKEN_ID,
                signature_type=-1,
            )
        )
        conditional_balance_raw   = int(float(resp2.get("balance",   0) or 0))
        conditional_allowance_raw = int(float(resp2.get("allowance", 0) or 0))
        required_raw = int(SEED_SHARES * CTF_DECIMALS)

        print(f"  balance_raw      : {conditional_balance_raw}")
        print(f"  balance_shares   : {conditional_balance_raw / CTF_DECIMALS:.2f}")
        print(f"  allowance_raw    : {conditional_allowance_raw}  (CLOB exchange)")
        print(f"  required_raw     : {required_raw}  ({SEED_SHARES:.0f} shares for SELL leg)")
        inventory_ok = conditional_balance_raw >= required_raw
        print(f"  inventory_ok     : {inventory_ok}")
        print()
    except Exception as exc:
        print(f"  ERROR: get_balance_allowance(CONDITIONAL) failed: {exc}")
        print()

    # ── Step 5: USDC.e → CTF approval (raw eth_call, not in CLOB API) ─────
    _section("Step 5: USDC.e → CTF Approval (raw eth_call)")
    print(f"  owner   (proxy) : {proxy_addr}")
    print(f"  spender (CTF)   : {CTF_ADDRESS}")
    print(f"  rpc             : {POLYGON_RPC}")
    ctf_allowance_raw: int = -1
    try:
        calldata = _encode_allowance_call(proxy_addr, CTF_ADDRESS)
        hex_result = _eth_call(POLYGON_RPC, USDC_ADDRESS, calldata)
        ctf_allowance_raw = _decode_uint256_result(hex_result)
        seed_amount_raw = int(SEED_SHARES * USDC_DECIMALS)

        print(f"  allowance_raw   : {ctf_allowance_raw}")
        print(f"  allowance_usdc  : {ctf_allowance_raw / USDC_DECIMALS:.4f}")
        print(f"  seed_need_raw   : {seed_amount_raw}  ({SEED_SHARES:.0f} USDC.e)")
        ctf_approval_ok = ctf_allowance_raw >= seed_amount_raw
        print(f"  approval_ok     : {ctf_approval_ok}")
        print()
    except Exception as exc:
        print(f"  ERROR: eth_call allowance check failed: {exc}")
        print()

    # ── Step 6: splitPosition calldata preview ────────────────────────────
    _section("Step 6: splitPosition calldata preview (no broadcast)")
    seed_amount_raw = int(SEED_SHARES * USDC_DECIMALS)
    calldata_hex: str | None = None
    try:
        calldata_hex = _encode_split_position_calldata(
            collateral_token=USDC_ADDRESS,
            condition_id_hex=HUNGARY_CONDITION_ID,
            amount_raw=seed_amount_raw,
        )
        print(f"  to               : {CTF_ADDRESS}  (ConditionalTokens)")
        print(f"  collateralToken  : {USDC_ADDRESS}  (USDC.e)")
        print(f"  parentCollection : 0x{'0' * 64}  (top-level split)")
        print(f"  conditionId      : {HUNGARY_CONDITION_ID}")
        print(f"  partition        : [1, 2]  (YES=bit0 → value 1, NO=bit1 → value 2)")
        print(f"  amount           : {seed_amount_raw}  ({SEED_SHARES:.0f} USDC.e → {SEED_SHARES:.0f} YES + {SEED_SHARES:.0f} NO)")
        print(f"  calldata_len     : {len(calldata_hex)} chars  ({(len(calldata_hex) - 2) // 2} bytes)")
        print(f"  calldata_preview : {calldata_hex[:66]}...")
        print()
    except Exception as exc:
        print(f"  ERROR: splitPosition encoding failed: {exc}")
        print()

    # ── Step 7: Relayer probe ─────────────────────────────────────────────
    _section("Step 7: Relayer Availability")
    relayer = _check_relayer_available()
    print(f"  relayer_available : {relayer['available']}")
    print(f"  method            : {relayer['method']}")
    print(f"  detail            : {relayer['detail']}")
    print()

    # ── Step 8: Verdict ───────────────────────────────────────────────────
    _sep("=")
    print()

    if relayer["available"]:
        if ctf_allowance_raw == 0:
            verdict = "CTF_APPROVAL_MISSING"
            print(f"  VERDICT: {verdict}")
            print()
            print("  Relayer is present but USDC.e → CTF allowance is zero.")
            print("  Before splitPosition can execute, the proxy wallet must approve")
            print(f"  the CTF contract to spend USDC.e:")
            print(f"    ERC20.approve(spender={CTF_ADDRESS}, amount=<sufficient>)")
            print("  Execute this approval via the proxy wallet, then re-run.")
        else:
            verdict = "RELAYER_PATH_AVAILABLE"
            print(f"  VERDICT: {verdict}")
            print()
            print("  Relayer client installed and CTF approval is present.")
            print("  Gasless splitPosition is executable via the relayer.")
    else:
        verdict = "RAW_SAFE_TX_REQUIRED"
        print(f"  VERDICT: {verdict}")
        print()
        print("  No Polymarket relayer client installed.")
        print("  py_builder_signing_sdk provides HMAC builder signing only.")
        print("  web3.py not installed.")
        print()
        print("  To seed 200 YES token inventory for bilateral quoting:")
        print()

        if ctf_allowance_raw == 0:
            print("  Step A — Approve CTF to spend USDC.e (allowance currently ZERO):")
            print(f"    ERC20({USDC_ADDRESS}).approve(")
            print(f"      {CTF_ADDRESS},  // CTF spender")
            print(f"      {seed_amount_raw}   // {SEED_SHARES:.0f} USDC.e in 6-decimal units")
            print("    ) via proxy wallet")
            print()
            print("  Step B — Split USDC.e into YES + NO conditional tokens:")
        else:
            print(f"  CTF approval already present (allowance={ctf_allowance_raw / USDC_DECIMALS:.4f} USDC.e).")
            print("  Only splitPosition is needed:")
            print()
            print("  Step A — Split USDC.e into YES + NO conditional tokens:")

        print(f"    CTF({CTF_ADDRESS}).splitPosition(")
        print(f"      collateralToken  = {USDC_ADDRESS},")
        print(f"      parentCollection = bytes32(0),")
        print(f"      conditionId      = {HUNGARY_CONDITION_ID},")
        print(f"      partition        = [1, 2],")
        print(f"      amount           = {seed_amount_raw}   // {SEED_SHARES:.0f} USDC.e")
        print(f"    ) via proxy wallet Safe execTransaction")
        print()
        print("  Fastest unblock: use Polymarket web UI")
        print("  (Add liquidity → Split → Hungary YES) — web UI calls the relayer internally.")

    print()
    _sep("=")
    print()
    print("reward_aware_maker_probe")


if __name__ == "__main__":
    main()
