"""
Diagnose COLLATERAL_INSUFFICIENT by finding the correct proxy wallet address.

Run this from the polyarb_lab root:
    python tmp_find_proxy_wallet.py

It will show your EOA, check balance at different signature types,
and print the env vars you need to add.
"""
import os, sys, json, requests
sys.path.insert(0, ".")

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

private_key = (
    os.environ.get("POLYMARKET_PRIVATE_KEY")
    or os.environ.get("POLY_PRIVATE_KEY")
    or ""
)
if not private_key:
    print("ERROR: POLYMARKET_PRIVATE_KEY not set in environment")
    sys.exit(1)

# ── 1. Derive EOA address ────────────────────────────────────────────────────
from py_clob_client.client import ClobClient
l1 = ClobClient(host=CLOB_HOST, chain_id=137, key=private_key)
eoa = l1.get_address()
print(f"\n=== Wallet Diagnosis ===")
print(f"EOA (signer)  : {eoa}")

# ── 2. Try to get L2 API credentials ────────────────────────────────────────
try:
    api_creds = l1.derive_api_key()
    print(f"API key (EOA) : {api_creds.api_key[:8]}...")
except Exception as e:
    print(f"WARNING: derive_api_key failed: {e}")
    api_creds = None

# ── 3. Check collateral balance at sig_type=0 (EOA) ─────────────────────────
if api_creds:
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams
    client_eoa = ClobClient(
        host=CLOB_HOST, chain_id=137, key=private_key,
        creds=api_creds, signature_type=0,
    )
    try:
        r = client_eoa.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=0)
        )
        bal_eoa = float(r.get("balance", 0) or 0) / 1_000_000
        print(f"\nBalance at EOA (sig_type=0)      : ${bal_eoa:.4f} USDC")
    except Exception as e:
        print(f"Balance at EOA check failed: {e}")
        bal_eoa = 0.0

    # sig_type=1 (POLY_PROXY — for Magic Link / email users)
    try:
        r1 = client_eoa.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=1)
        )
        bal_proxy1 = float(r1.get("balance", 0) or 0) / 1_000_000
        print(f"Balance at POLY_PROXY (sig_type=1): ${bal_proxy1:.4f} USDC")
    except Exception as e:
        print(f"Balance at POLY_PROXY check failed: {e}")
        bal_proxy1 = 0.0

# ── 4. Query Polymarket user API for proxy wallet address ───────────────────
print(f"\n=== Polymarket User Profile ===")
proxy_addr = None
try:
    resp = requests.get(
        f"https://polymarket.com/api/profile/address/{eoa}",
        timeout=8,
        headers={"User-Agent": "polyarb_lab/diagnostic"}
    )
    if resp.ok:
        data = resp.json()
        proxy_addr = data.get("proxyWallet") or data.get("proxy_wallet") or data.get("address")
        print(f"Profile response: {json.dumps(data, indent=2)[:500]}")
    else:
        print(f"Profile API status: {resp.status_code}")
except Exception as e:
    print(f"Profile API failed: {e}")

# Try alternate endpoint
if not proxy_addr:
    try:
        resp2 = requests.get(
            f"https://clob.polymarket.com/user?address={eoa}",
            timeout=8,
        )
        if resp2.ok:
            data2 = resp2.json()
            proxy_addr = data2.get("proxyWallet") or data2.get("proxy_wallet")
            print(f"CLOB user response: {json.dumps(data2, indent=2)[:500]}")
        else:
            print(f"CLOB user endpoint status: {resp2.status_code} — {resp2.text[:200]}")
    except Exception as e2:
        print(f"CLOB user endpoint failed: {e2}")

# Try Polymarket auth API
if not proxy_addr:
    try:
        resp3 = requests.get(
            f"https://clob.polymarket.com/profile?address={eoa}",
            timeout=8,
        )
        if resp3.ok:
            data3 = resp3.json()
            proxy_addr = data3.get("proxyWallet") or data3.get("proxy_wallet")
            print(f"Profile endpoint: {json.dumps(data3, indent=2)[:500]}")
        else:
            print(f"Profile endpoint status: {resp3.status_code}")
    except Exception as e3:
        print(f"Profile endpoint failed: {e3}")

# ── 5. If proxy found, check its balance (sig_type=2) ───────────────────────
if proxy_addr and api_creds:
    print(f"\nFound proxy wallet: {proxy_addr}")
    try:
        client_proxy = ClobClient(
            host=CLOB_HOST, chain_id=137, key=private_key,
            creds=api_creds, signature_type=2, funder=proxy_addr,
        )
        r2 = client_proxy.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=2)
        )
        bal_gnosis = float(r2.get("balance", 0) or 0) / 1_000_000
        print(f"Balance at Gnosis proxy (sig_type=2): ${bal_gnosis:.4f} USDC")
    except Exception as e:
        print(f"Balance at proxy check failed: {e}")
        bal_gnosis = 0.0
else:
    bal_gnosis = 0.0

# ── 6. Print recommendation ──────────────────────────────────────────────────
print(f"\n=== DIAGNOSIS ===")
if bal_proxy1 > 0.5:
    print(f"✓ Your $60 is in the POLY_PROXY wallet (sig_type=1)")
    print(f"  You need to set: POLYMARKET_SIGNATURE_TYPE=1")
    print(f"  (no POLYMARKET_FUNDER needed for sig_type=1)")
elif bal_gnosis > 0.5:
    print(f"✓ Your $60 is in the Gnosis Safe proxy (sig_type=2)")
    print(f"  You need to set:")
    print(f"    POLYMARKET_SIGNATURE_TYPE=2")
    print(f"    POLYMARKET_FUNDER={proxy_addr}")
elif proxy_addr:
    print(f"Proxy wallet found: {proxy_addr}")
    print(f"But balance not confirmed via API check.")
    print(f"Try setting:")
    print(f"  POLYMARKET_SIGNATURE_TYPE=2")
    print(f"  POLYMARKET_FUNDER={proxy_addr}")
else:
    print(f"Could not auto-detect proxy wallet address.")
    print(f"Manual steps:")
    print(f"  1. Go to https://polymarket.com")
    print(f"  2. Click your profile → Settings")
    print(f"  3. Find 'Wallet address' or 'Polymarket address'")
    print(f"     (it should be different from {eoa})")
    print(f"  4. Add to your .env file:")
    print(f"     POLYMARKET_SIGNATURE_TYPE=2")
    print(f"     POLYMARKET_FUNDER=<that address>")
    if bal_proxy1 == 0.0 and bal_eoa == 0.0:
        print(f"\nWARNING: Balance is $0 at all checked sig types.")
        print(f"Please confirm the $60 deposit is visible at polymarket.com")
        print(f"(may need a few minutes to settle)")
