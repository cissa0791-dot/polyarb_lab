"""
Research-only probe: negRisk basket bid edge.
For each live negRisk event group, computes basket_bid_sum and edge = 1.0 - basket_bid_sum.
Reports top events by bid edge.
Uses analyze_neg_risk_rebalancing_event directly.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_runtime.loader import load_runtime_config
from src.ingest.gamma import fetch_markets
from src.ingest.clob import ReadOnlyClob
from src.scanner.neg_risk import build_eligible_neg_risk_event_groups, analyze_neg_risk_rebalancing_event

SETTINGS = ROOT / "config" / "settings.yaml"

def _fetch_events(gamma_host: str, limit: int) -> list[dict]:
    """Fetch markets and reconstruct event groups."""
    import requests
    from datetime import datetime, timezone
    events = []
    page_size = 200
    offset = 0
    seen_ids: set[str] = set()
    while len(events) < limit:
        try:
            resp = requests.get(
                f"{gamma_host}/events",
                params={"active": "true", "closed": "false", "limit": page_size, "offset": offset},
                timeout=30,
            )
            if not resp.ok:
                break
            page = resp.json()
            if not isinstance(page, list):
                # Some APIs return {"data": [...]}
                page = page.get("data", []) if isinstance(page, dict) else []
            if not page:
                break
            added = 0
            for ev in page:
                eid = str(ev.get("id") or "")
                if eid and eid not in seen_ids:
                    seen_ids.add(eid)
                    events.append(ev)
                    added += 1
            if added == 0 or len(page) < page_size:
                break
            offset += page_size
        except Exception as e:
            print(f"Event fetch error at offset {offset}: {e}", flush=True)
            break
    return events[:limit]


def main():
    cfg = load_runtime_config(str(SETTINGS))
    gamma_host = cfg.market_data.gamma_host
    clob_host = cfg.market_data.clob_host

    print("Fetching events from Gamma API...", flush=True)
    events = _fetch_events(gamma_host, limit=2000)
    print(f"Fetched {len(events)} events.", flush=True)

    groups = build_eligible_neg_risk_event_groups(events)
    print(f"Eligible negRisk event groups: {len(groups)}", flush=True)

    if not groups:
        print("No eligible negRisk groups found.")
        return

    # Collect all YES token IDs
    token_ids = list({
        m["yes_token_id"]
        for g in groups
        for m in g.get("markets", [])
        if m.get("yes_token_id")
    })
    print(f"Fetching {len(token_ids)} YES token books...", flush=True)

    clob = ReadOnlyClob(clob_host)
    books: dict[str, object] = {}
    try:
        for book in clob.get_books(token_ids):
            books[str(book.token_id)] = book
    except Exception as e:
        print(f"Batch fetch failed ({e}), falling back individually...", flush=True)
        for tid in token_ids:
            try:
                books[tid] = clob.get_book(tid)
            except Exception:
                pass
    print(f"Got {len(books)} books.\n", flush=True)

    results = []
    for g in groups:
        analysis = analyze_neg_risk_rebalancing_event(g, books, max_notional=100.0)
        opp = analysis["opportunity"]
        audit = analysis.get("audit", {})
        markets = g.get("markets", [])
        n_markets = len(markets)
        event_slug = g.get("event_slug", "?")

        if opp is not None:
            edge = float(opp.edge_cents)
            basket_sum = float(opp.details.get("basket_bid_sum", 0.0))
            results.append({"event_slug": event_slug, "n_markets": n_markets, "basket_bid_sum": basket_sum, "edge": edge, "status": "PASS"})
        else:
            reason = (audit or {}).get("failure_reason", "?")
            # Try to get partial info
            basket_sum = (audit or {}).get("basket_bid_sum", None)
            results.append({"event_slug": event_slug, "n_markets": n_markets, "basket_bid_sum": basket_sum, "edge": None, "status": reason})

    results.sort(key=lambda x: x["edge"] if x["edge"] is not None else -99, reverse=True)

    print(f"{'Event Slug':<50} {'N':>4} {'BaskSum':>9} {'Edge¢':>7}  Status")
    print("-" * 90)
    for r in results[:30]:
        bs = f"{r['basket_bid_sum']:>9.4f}" if r['basket_bid_sum'] is not None else f"{'None':>9}"
        ed = f"{r['edge']:>7.4f}" if r['edge'] is not None else f"{'None':>7}"
        print(f"{r['event_slug']:<50} {r['n_markets']:>4} {bs} {ed}  {r['status']}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_basket_non_pos = sum(1 for r in results if r["status"] == "NEG_RISK_BASKET_EDGE_NON_POSITIVE")
    print(f"\nTotal groups: {len(results)}")
    print(f"PASS (edge > 0): {n_pass}")
    print(f"NEG_RISK_BASKET_EDGE_NON_POSITIVE: {n_basket_non_pos}")
    print(f"Other failures: {len(results) - n_pass - n_basket_non_pos}")

if __name__ == "__main__":
    main()
