"""
One-shot sell script — liquidate all YES share inventory for a given market.

Usage (PowerShell, with env vars already set):
    python tmp_sell_inventory.py --slug will-tom-steyer-win-the-california-governor-election-in-2026
    python tmp_sell_inventory.py --slug will-tom-steyer-...  --dry-run
    python tmp_sell_inventory.py --slug will-tom-steyer-...  --shares 80

Requires env vars:
    POLYMARKET_PRIVATE_KEY
    POLYMARKET_SIGNATURE_TYPE=2
    POLYMARKET_FUNDER=0x...proxy...
    POLYMARKET_CHAIN_ID=137
"""
import argparse, os, sys, time
sys.path.insert(0, ".")

CLOB_HOST = "https://clob.polymarket.com"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--shares", type=float, default=None,
                    help="Shares to sell (default: all you have)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # ── 1. Load creds ────────────────────────────────────────────────────────
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        load_activation_credentials, build_clob_client,
    )
    creds = load_activation_credentials()
    if not creds:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set"); sys.exit(1)

    print(f"sig_type : {creds.signature_type}")
    print(f"funder   : {creds.funder or '(EOA mode)'}")

    client = build_clob_client(creds, CLOB_HOST)

    # ── 2. Look up token_id for the slug ─────────────────────────────────────
    from research_lines.auto_maker_loop.run_auto_maker import _load_dynamic_survivor_data
    survivor_data = _load_dynamic_survivor_data(min_daily_rate_usdc=0, max_markets=20)
    data = survivor_data.get(args.slug)
    if not data:
        # Try partial match
        matches = [s for s in survivor_data if args.slug in s]
        if len(matches) == 1:
            data = survivor_data[matches[0]]
            print(f"Matched slug: {matches[0]}")
        else:
            print(f"ERROR: slug {args.slug!r} not found in survivor data")
            print("Available:", list(survivor_data.keys()))
            sys.exit(1)

    token_id = data["token_id"]
    print(f"token_id : {token_id[:24]}…")

    # ── 3. Check current YES balance ─────────────────────────────────────────
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        _check_sell_inventory,
    )
    inv = _check_sell_inventory(client, token_id, required_shares=1)
    balance = inv.get("balance_shares", 0.0)
    print(f"YES balance : {balance:.4f} shares")

    if balance <= 0:
        print("Nothing to sell — balance is 0."); return

    sell_size = args.shares if args.shares is not None else balance
    sell_size = min(sell_size, balance)
    if sell_size <= 0:
        print(f"--shares={args.shares} but balance={balance:.2f}; nothing to sell"); return

    # ── 4. Get current best bid ───────────────────────────────────────────────
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        fetch_midpoint,
    )
    mid, src = fetch_midpoint(client, token_id, None)
    if mid is None:
        print("ERROR: could not fetch midpoint — cannot price SELL"); sys.exit(1)

    # Sell at (mid - 0.01) — one tick below mid, aggressive enough to fill quickly
    sell_price = round(max(0.01, mid - 0.01), 2)
    print(f"midpoint   : {mid:.4f}  (src={src})")
    print(f"sell_price : {sell_price:.4f}  (mid - 1 tick)")
    print(f"sell_size  : {sell_size:.1f} shares")
    print(f"est. value : ${sell_price * sell_size:.2f}")

    if args.dry_run:
        print("\nDRY-RUN: would place SELL order. Add --no-dry-run or remove --dry-run to execute.")
        return

    # ── 5. Place SELL limit order ─────────────────────────────────────────────
    print(f"\nPlacing SELL order…")
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import _place_order
    order_id, err = _place_order(client, token_id, sell_price, sell_size, "SELL")

    if not order_id:
        print(f"SELL FAILED: {err}"); sys.exit(1)

    print(f"SELL order placed  order_id={order_id[:20]}…")

    # ── 6. Poll for fill (up to 2 minutes) ───────────────────────────────────
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import _is_filled
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        time.sleep(5)
        if _is_filled(client, order_id):
            print(f"FILLED ✓  sold {sell_size:.1f} shares @ {sell_price:.4f}")
            print(f"Proceeds ≈ ${sell_price * sell_size:.2f}")
            return
        print("  …waiting for fill")

    print(f"Order not yet filled after 2 min — it remains as a GTC SELL on the book.")
    print(f"order_id = {order_id}")
    print("Check polymarket.com or re-run to monitor.")

if __name__ == "__main__":
    main()
