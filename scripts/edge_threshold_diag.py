"""
Diagnostic-only threshold study.
Read-only. No orders. No DB writes.
Classifies all hits in a single API pass across thresholds [0.01, 0.02, 0.03, 0.04].
"""

import datetime as dt
import json
import sys

import httpx

THRESHOLDS = [0.01, 0.02, 0.03, 0.04]
LIMIT = 50
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# Heuristics for "stub / illiquid / non-actionable"
STUB_KEYWORDS = [
    "jesus", "christ", "lebron", "2028-presidential", "2028-us-pres",
    "win-the-2028", "win-the-202", "hillary", "tim-walz",
    "will-algeria", "will-iran", "iranian-regime",
]
STUB_MIN_SIZE_THRESHOLD = 50   # orderMinSize >= 50 → likely illiquid / wide spread
STUB_EDGE_CAP = 0.98           # if yes_ask or no_ask >= 0.98, already filtered


def is_stub(slug: str, min_size: float, yes_ask: float, no_ask: float) -> bool:
    slug_l = slug.lower()
    if any(kw in slug_l for kw in STUB_KEYWORDS):
        return True
    if min_size >= STUB_MIN_SIZE_THRESHOLD:
        return True
    # If one side is nearly certain (>0.95), the "edge" is likely a stale quote, not arb
    if yes_ask > 0.95 or no_ask > 0.95:
        return True
    return False


def main():
    print(f"\n{'='*70}")
    print(f"  EDGE THRESHOLD DIAGNOSTIC  |  {dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"  limit={LIMIT}  thresholds={THRESHOLDS}")
    print(f"{'='*70}\n")

    print(f"Fetching top {LIMIT} active markets by 24h volume...", flush=True)
    try:
        resp = httpx.get(
            f"{GAMMA}/markets",
            params={
                "limit": LIMIT,
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
            },
            timeout=20,
        )
        markets = resp.json()
        if not isinstance(markets, list):
            markets = markets.get("markets", [])
    except Exception as e:
        print(f"ERROR fetching markets: {e}")
        sys.exit(1)

    print(f"Received {len(markets)} markets. Fetching order books...\n", flush=True)

    # Collect all raw market data in one pass
    records = []   # {slug, edge, yes_ask, no_ask, min_size, stub}
    skipped_no_pair = 0
    skipped_no_book = 0
    skipped_both_98 = 0

    for i, m in enumerate(markets, 1):
        raw_ids = m.get("clobTokenIds") or []
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except Exception:
                skipped_no_pair += 1
                continue

        if len(raw_ids) < 2:
            skipped_no_pair += 1
            continue

        slug = str(m.get("slug") or "?")
        min_size = float(m.get("orderMinSize") or 5)
        yes_id, no_id = str(raw_ids[0]), str(raw_ids[1])

        print(f"  [{i:2d}/{len(markets)}] {slug[:60]}", flush=True)

        try:
            yb = httpx.get(f"{CLOB}/book", params={"token_id": yes_id}, timeout=5).json()
            nb = httpx.get(f"{CLOB}/book", params={"token_id": no_id}, timeout=5).json()
        except Exception:
            skipped_no_book += 1
            continue

        ya = yb.get("asks") or []
        na = nb.get("asks") or []
        if not ya or not na:
            skipped_no_book += 1
            continue

        yes_ask = float(ya[0]["price"])
        no_ask = float(na[0]["price"])

        if yes_ask >= 0.98 and no_ask >= 0.98:
            skipped_both_98 += 1
            continue

        edge = 1.0 - yes_ask - no_ask
        stub = is_stub(slug, min_size, yes_ask, no_ask)

        records.append({
            "slug": slug,
            "edge": edge,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "min_size": min_size,
            "stub": stub,
        })

    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Markets fetched   : {len(markets)}")
    print(f"  No binary pair    : {skipped_no_pair}")
    print(f"  No order book     : {skipped_no_book}")
    print(f"  Both asks ≥ 0.98  : {skipped_both_98}")
    print(f"  Viable (computed) : {len(records)}")
    print()

    all_edges = sorted([r["edge"] for r in records], reverse=True)
    if all_edges:
        print(f"  Edge distribution across all viable markets:")
        print(f"    max   = {max(all_edges)*100:.2f}¢")
        print(f"    median= {sorted(all_edges)[len(all_edges)//2]*100:.2f}¢")
        print(f"    min   = {min(all_edges)*100:.2f}¢")
        bins = [(0.00, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 1.0)]
        for lo, hi in bins:
            count = sum(1 for e in all_edges if lo <= e < hi)
            label = f"[{lo:.2f}, {hi:.2f})"
            bar = "#" * count
            print(f"    {label}: {count:3d}  {bar}")
    print()

    # Per-threshold breakdown
    print(f"  {'Threshold':<12} {'Total Hits':<12} {'Actionable':<12} {'Stubs':<8}  Sample slugs")
    print(f"  {'-'*9:<12} {'-'*10:<12} {'-'*10:<12} {'-'*5:<8}  ------")
    for t in THRESHOLDS:
        hits = [r for r in records if r["edge"] >= t]
        stubs = [r for r in hits if r["stub"]]
        actionable = [r for r in hits if not r["stub"]]
        samples = ", ".join(h["slug"][:35] for h in sorted(hits, key=lambda x: x["edge"], reverse=True)[:3])
        print(f"  {t:<12.2f} {len(hits):<12} {len(actionable):<12} {len(stubs):<8}  {samples}")

    print()

    # Detailed hit list at 0.01 (catches everything)
    hits_001 = sorted([r for r in records if r["edge"] >= 0.01], key=lambda x: x["edge"], reverse=True)
    if hits_001:
        print(f"  All hits at min_edge=0.01 (sorted by edge desc):")
        print(f"  {'edge':>7}  {'yes_ask':>8}  {'no_ask':>8}  {'stub':>5}  slug")
        for r in hits_001:
            stub_flag = "STUB" if r["stub"] else "ok"
            print(f"  {r['edge']*100:>6.2f}¢  {r['yes_ask']:>8.4f}  {r['no_ask']:>8.4f}  {stub_flag:>5}  {r['slug'][:55]}")
    else:
        print("  No hits even at min_edge=0.01 — market is currently very efficient or all stale.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
