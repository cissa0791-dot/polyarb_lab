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

def _sell_all(args):
    """
    Scan every probe file for token_ids we've ever seen, check balance,
    and sell any non-zero position. Used to clean up orphan positions
    spread across multiple markets.
    """
    import glob, json
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        load_activation_credentials, build_clob_client,
        _check_sell_inventory, fetch_midpoint, _place_order, _is_filled,
    )

    creds = load_activation_credentials()
    if not creds:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set"); sys.exit(1)
    print(f"sig_type : {creds.signature_type}")
    print(f"funder   : {creds.funder or '(EOA mode)'}")

    client = build_clob_client(creds, CLOB_HOST)

    # Collect every (slug, token_id, yes_price_ref) from probe files
    seen: dict = {}  # token_id → (slug, yes_price_ref)
    for f in sorted(glob.glob("data/research/reward_aware_maker_probe/*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for m in data.get("markets", []):
                tok = m.get("token_id")
                if tok and tok not in seen:
                    seen[tok] = (m.get("slug", "?"), m.get("yes_price_ref"))
        except Exception:
            pass

    print(f"\nScanning {len(seen)} unique token_ids from probe history …\n")

    positions = []
    for tok, (slug, price_ref) in seen.items():
        try:
            inv = _check_sell_inventory(client, tok, required_shares=1)
            bal = inv.get("balance_shares", 0.0)
            if bal > 0.5:
                positions.append((slug, tok, bal, price_ref))
                print(f"  FOUND  {slug[:45]:<45}  balance={bal:.1f}")
        except Exception as e:
            print(f"  skipped {slug[:30]}: {e}")

    if not positions:
        print("\nNo non-zero positions found. Clean!"); return

    print(f"\n{len(positions)} market(s) have inventory. ", end="")
    if args.dry_run:
        print("DRY-RUN — would sell each one."); return
    print("Selling each …\n")

    for slug, tok, bal, price_ref in positions:
        print(f"\n── Selling {bal:.1f} of {slug[:50]} ─────")
        mid, src = fetch_midpoint(client, tok, price_ref)
        if mid is None:
            print(f"  SKIP: no midpoint for {slug[:40]} (price_ref={price_ref})")
            continue
        sell_price = round(max(0.01, mid - 0.01), 2)
        print(f"  mid={mid:.4f} src={src} → sell_price={sell_price:.4f}")
        order_id, err = _place_order(client, tok, sell_price, bal, "SELL")
        if not order_id:
            print(f"  FAILED: {err}"); continue
        print(f"  SELL placed order_id={order_id[:20]}…")
        deadline = time.monotonic() + 60
        filled = False
        while time.monotonic() < deadline:
            time.sleep(5)
            if _is_filled(client, order_id):
                print(f"  FILLED ✓  ~${sell_price * bal:.2f}")
                filled = True
                break
        if not filled:
            print(f"  not yet filled — GTC remains on book")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None,
                    help="Slug to sell (omit when using --all)")
    ap.add_argument("--all", action="store_true",
                    help="Scan every market in ALL probe files and sell any non-zero balance")
    ap.add_argument("--shares", type=float, default=None,
                    help="Shares to sell (default: all you have)")
    ap.add_argument("--sell-price", type=float, default=None,
                    help="Override sell price (e.g. 0.34). Use when midpoint unavailable.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.slug and not args.all:
        ap.error("must pass --slug OR --all")

    if args.all:
        _sell_all(args); return

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
    price_ref = data.get("yes_price_ref") or data.get("yes_price_clob")
    mid, src = fetch_midpoint(client, token_id, price_ref)
    if mid is None:
        print(f"ERROR: could not fetch midpoint (price_ref={price_ref}) — cannot price SELL")
        print("Tip: set a manual price with  --sell-price 0.35")
        sys.exit(1)

    # Sell at (mid - 0.01) — one tick below mid, aggressive enough to fill quickly
    if args.sell_price is not None:
        sell_price = round(max(0.01, min(0.99, args.sell_price)), 2)
        print(f"midpoint   : {mid:.4f}  (src={src})")
        print(f"sell_price : {sell_price:.4f}  (manual override)")
    else:
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
