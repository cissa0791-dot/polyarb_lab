"""
Research-only probe: Senate mutex execution edge.

For each US state with both a D and R senate race market in 2026:
  relation check: P(D_YES) + P(R_YES) > 1.0
  execution:      buy D_NO + buy R_NO
  edge:           1.0 - (D_NO_ask + R_NO_ask)
                = P(D_YES) + P(R_YES) - 1.0

Uses analyze_political_mutex_pair logic directly (no full runner needed).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_markets
from src.ingest.clob import ReadOnlyClob
from src.core.normalize import build_yes_no_pairs

SETTINGS = ROOT / "config" / "settings.yaml"

# Pattern: will-the-{party}-win-the-{state}-senate-race-in-{year}
_SENATE_RE = re.compile(
    r"^will-the-(?P<party>democrats|republicans)-win-the-(?P<state>[a-z-]+)-senate-race-in-(?P<year>20\d{2})$"
)

def main():
    cfg = load_runtime_config(str(SETTINGS))
    gamma_host = cfg.market_data.gamma_host
    clob_host = cfg.market_data.clob_host

    print("Fetching 3000 markets...", flush=True)
    markets = fetch_markets(gamma_host, limit=3000)
    slugs_to_market = {str(m.get("slug", "")).strip().lower(): m for m in markets if m.get("slug")}

    # Find senate pairs by state+year
    pairs_by_state: dict[str, dict[str, str]] = defaultdict(dict)  # state -> {party: slug}
    for slug in slugs_to_market:
        m = _SENATE_RE.match(slug)
        if m:
            state = m.group("state")
            party = m.group("party")
            year = m.group("year")
            key = f"{state}_{year}"
            pairs_by_state[key][party] = slug

    complete_pairs = {
        k: v for k, v in pairs_by_state.items()
        if "democrats" in v and "republicans" in v
    }
    print(f"\nFound {len(complete_pairs)} complete D+R state senate pairs.\n")

    if not complete_pairs:
        # Try alternate slug form
        print("Trying alternate senate slug patterns...")
        _ALT_RE = re.compile(r".*senate.*(?P<state>[a-z-]+).*(?:democratic|republican).*|.*(?:democratic|republican).*senate.*")
        for slug in slugs_to_market:
            if "senate" in slug:
                print(f"  {slug}")
        return

    # Build YES/NO token map from market data
    pairs_list = build_yes_no_pairs(markets)
    pair_by_slug = {p.market_slug: p for p in pairs_list}

    # Collect all token IDs needed
    token_ids = []
    for state_year, party_slugs in complete_pairs.items():
        for slug in party_slugs.values():
            p = pair_by_slug.get(slug)
            if p:
                token_ids.extend([p.yes_token_id, p.no_token_id])
    token_ids = list(set(t for t in token_ids if t))

    print(f"Fetching CLOB books for {len(token_ids)} tokens...", flush=True)
    clob = ReadOnlyClob(clob_host)
    books: dict[str, object] = {}
    try:
        for book in clob.get_books(token_ids):
            books[str(book.token_id)] = book
    except Exception as e:
        print(f"Batch fetch failed ({e}), falling back to individual...", flush=True)
        for tid in token_ids:
            try:
                books[tid] = clob.get_book(tid)
            except Exception:
                pass
    print(f"Got {len(books)} books.\n")

    def best_ask(tok):
        b = books.get(tok)
        if b is None or not getattr(b, "asks", []):
            return None
        return float(b.asks[0].price)

    def best_bid(tok):
        b = books.get(tok)
        if b is None or not getattr(b, "bids", []):
            return None
        return float(b.bids[0].price)

    # Score each pair
    results = []
    for state_year, party_slugs in sorted(complete_pairs.items()):
        d_slug = party_slugs["democrats"]
        r_slug = party_slugs["republicans"]
        dp = pair_by_slug.get(d_slug)
        rp = pair_by_slug.get(r_slug)
        if dp is None or rp is None:
            continue

        d_yes_ask = best_ask(dp.yes_token_id)
        r_yes_ask = best_ask(rp.yes_token_id)
        d_no_ask = best_ask(dp.no_token_id)
        r_no_ask = best_ask(rp.no_token_id)

        if d_yes_ask is None or r_yes_ask is None:
            continue

        relation_gap = (d_yes_ask + r_yes_ask) - 1.0  # >0 means mutex violated
        exec_edge = None
        if d_no_ask is not None and r_no_ask is not None:
            exec_edge = 1.0 - (d_no_ask + r_no_ask)   # = P(D) + P(R) - 1.0 after bid-ask adjustment

        results.append({
            "state_year": state_year,
            "d_slug": d_slug,
            "r_slug": r_slug,
            "d_yes_ask": d_yes_ask,
            "r_yes_ask": r_yes_ask,
            "yes_sum": round(d_yes_ask + r_yes_ask, 6),
            "relation_gap": round(relation_gap, 6),
            "d_no_ask": d_no_ask,
            "r_no_ask": r_no_ask,
            "exec_edge": round(exec_edge, 6) if exec_edge is not None else None,
        })

    # Sort by relation_gap descending
    results.sort(key=lambda x: x["relation_gap"], reverse=True)

    print(f"{'State/Year':<30} {'D_YES':>7} {'R_YES':>7} {'Sum':>7} {'RelGap':>8} {'ExecEdge':>10}")
    print("-" * 80)
    for r in results:
        edge_s = f"{r['exec_edge']:>10.4f}" if r['exec_edge'] is not None else f"{'None':>10}"
        flag = " <-- POSITIVE" if r['relation_gap'] > 0 else ""
        print(f"{r['state_year']:<30} {r['d_yes_ask']:>7.4f} {r['r_yes_ask']:>7.4f} {r['yes_sum']:>7.4f} {r['relation_gap']:>8.4f} {edge_s}{flag}")

    n_positive = sum(1 for r in results if r['relation_gap'] > 0)
    n_exec_positive = sum(1 for r in results if r['exec_edge'] is not None and r['exec_edge'] > 0)
    print(f"\nTotal pairs: {len(results)}")
    print(f"Relation gap > 0 (P(D)+P(R) > 1.0): {n_positive}")
    print(f"Exec edge > 0: {n_exec_positive}")
    if n_exec_positive > 0:
        best = max((r for r in results if r['exec_edge'] is not None and r['exec_edge'] > 0), key=lambda x: x['exec_edge'])
        print(f"Best exec edge: {best['exec_edge']:.4f}¢ ({best['state_year']})")
    else:
        best_exec = max(results, key=lambda x: x['exec_edge'] if x['exec_edge'] is not None else -99)
        print(f"Best exec edge (all negative): {best_exec['exec_edge']:.4f}¢ ({best_exec['state_year']})")

if __name__ == "__main__":
    main()
