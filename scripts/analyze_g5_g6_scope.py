"""
Narrow empirical analysis of G5/G6 scope across the full maker-MM eligible population.

Read-only: fetches Gamma market metadata, computes EV, produces statistics.
No paper execution, no DB writes, no config changes.

G5: Does the default min_edge_cents=0.03 threshold trigger across the family?
    Metric: per_share_edge = total_ev / quote_size
    G5 triggered if per_share_edge < 0.03

G6: Does the default max_order_notional_usd=100 cap trigger across the family?
    Metric: rewards_min_size (determines minimum quote_size)
    G6 triggered if rewards_min_size > 100

Usage:
    py -3 scripts/analyze_g5_g6_scope.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.gamma import fetch_events, fetch_markets
from src.intelligence.market_intelligence import build_event_market_registry
from src.scanner.wide_scan_maker_mm import compute_wide_scan_ev

GAMMA_HOST = "https://gamma-api.polymarket.com"
DEFAULT_MIN_EDGE_CENTS = 0.03
DEFAULT_MAX_NOTIONAL = 100.0


def main() -> None:
    print("=" * 70)
    print("G5/G6 FAMILY-SCOPE ANALYSIS — maker-MM eligible population")
    print("=" * 70)

    # Read-only Gamma fetch
    print("\n[1] Fetching Gamma registry (read-only)...")
    events  = fetch_events(GAMMA_HOST, limit=500)
    markets = fetch_markets(GAMMA_HOST, limit=500)
    registry = build_event_market_registry(events, markets)
    print(f"    events: {len(events)}, markets: {len(markets)}")

    # Collect all reward-eligible markets (same criteria as runner integration)
    raw_eligible: list[dict] = []
    for event in registry.get("events", []):
        for m in event.get("markets", []):
            if not m.get("is_binary_yes_no") or not m.get("enable_orderbook"):
                continue
            rewards     = m.get("clob_rewards") or []
            reward_rate = sum(float(r.get("rewardsDailyRate", 0) or 0) for r in rewards)
            if reward_rate <= 0:
                continue
            min_size   = float(m.get("rewards_min_size") or 0)
            max_spread = float(m.get("rewards_max_spread") or 0)
            if min_size <= 0 or max_spread <= 0:
                continue
            best_bid = float(m.get("best_bid") or 0)
            best_ask = float(m.get("best_ask") or 0)
            if best_bid <= 0 or best_ask <= best_bid:
                continue
            if not m.get("yes_token_id") or not m.get("no_token_id"):
                continue

            raw_eligible.append({
                "event_slug":         str(event.get("slug") or ""),
                "market_slug":        str(m.get("slug") or ""),
                "neg_risk":           bool(m.get("neg_risk")),
                "best_bid":           best_bid,
                "best_ask":           best_ask,
                "rewards_min_size":   min_size,
                "rewards_max_spread": max_spread,
                "reward_daily_rate":  reward_rate,
                "volume_num":         float(m.get("volume_num") or 0),
            })

    print(f"    reward_eligible_markets: {len(raw_eligible)}")

    # Compute EV and G5/G6 classification for each market
    results: list[dict] = []
    for m in raw_eligible:
        ev = compute_wide_scan_ev(m)
        quote_size       = ev["quote_size"]
        total_ev         = ev["total_ev"]
        per_share_edge   = total_ev / max(quote_size, 1e-9)
        g5_triggered     = per_share_edge < DEFAULT_MIN_EDGE_CENTS
        g6_triggered     = m["rewards_min_size"] > DEFAULT_MAX_NOTIONAL
        results.append({
            **m,
            "quote_size":        quote_size,
            "total_ev":          total_ev,
            "reward_ev":         ev["reward_ev"],
            "spread_capture_ev": ev["spread_capture_ev"],
            "per_share_edge":    per_share_edge,
            "g5_triggered":      g5_triggered,
            "g6_triggered":      g6_triggered,
        })

    results.sort(key=lambda x: x["total_ev"], reverse=True)
    n = len(results)

    # ---------- population statistics ----------
    g5_count      = sum(1 for r in results if r["g5_triggered"])
    g6_count      = sum(1 for r in results if r["g6_triggered"])
    both_count    = sum(1 for r in results if r["g5_triggered"] and r["g6_triggered"])
    neither_count = sum(1 for r in results if not r["g5_triggered"] and not r["g6_triggered"])
    pos_ev_count  = sum(1 for r in results if r["total_ev"] > 0)
    min_sizes     = [r["rewards_min_size"] for r in results]
    ps_edges      = [r["per_share_edge"] for r in results]
    ps_sorted     = sorted(ps_edges)

    # ---------- print top-20 sample ----------
    print(f"\n[2] Top 20 markets by total_ev (of {n} eligible with positive book):")
    hdr = f"{'event_slug':<44} {'min_sz':>6} {'q_sz':>6} {'tot_ev':>7} {'ps_edge':>8}  G5? G6?"
    print(hdr)
    print("-" * 84)
    for r in results[:20]:
        slug = r["event_slug"][:43]
        print(
            f"{slug:<44} {r['rewards_min_size']:>6.0f} {r['quote_size']:>6.0f}"
            f" {r['total_ev']:>7.4f} {r['per_share_edge']:>8.5f}"
            f"   {'Y' if r['g5_triggered'] else 'N'}   {'Y' if r['g6_triggered'] else 'N'}"
        )

    # ---------- population summary ----------
    print(f"\n[3] Population summary (n={n} reward-eligible markets with valid book):")
    print(f"    G5 triggered (ps_edge < {DEFAULT_MIN_EDGE_CENTS:.3f}):   {g5_count:>3}/{n} = {100*g5_count/max(n,1):5.1f}%")
    print(f"    G6 triggered (min_size > {DEFAULT_MAX_NOTIONAL:.0f}): {g6_count:>3}/{n} = {100*g6_count/max(n,1):5.1f}%")
    print(f"    Both G5+G6:                         {both_count:>3}/{n} = {100*both_count/max(n,1):5.1f}%")
    print(f"    Neither (fits default config):       {neither_count:>3}/{n} = {100*neither_count/max(n,1):5.1f}%")
    print(f"    Positive total_ev:                  {pos_ev_count:>3}/{n} = {100*pos_ev_count/max(n,1):5.1f}%")
    print(f"\n    rewards_min_size range: {min(min_sizes):.0f} – {max(min_sizes):.0f}")
    print(f"    per_share_edge range:   {min(ps_sorted):.5f} – {max(ps_sorted):.5f}")
    print(f"    per_share_edge p25:     {ps_sorted[n//4]:.5f}")
    print(f"    per_share_edge median:  {ps_sorted[n//2]:.5f}")
    print(f"    per_share_edge p75:     {ps_sorted[3*n//4]:.5f}")

    # ---------- G5 by reward_daily_rate tier ----------
    print(f"\n[4] G5 by reward_daily_rate tier:")
    print(f"    {'rate':>6}  {'n':>4}  {'G5':>4}  {'G5%':>6}  {'avg_ps_edge':>12}  {'avg_min_size':>12}")
    all_rates = sorted({r["reward_daily_rate"] for r in results}, reverse=True)
    for rate in all_rates:
        grp = [r for r in results if r["reward_daily_rate"] == rate]
        g5  = sum(1 for r in grp if r["g5_triggered"])
        avg_pse  = sum(r["per_share_edge"] for r in grp) / len(grp)
        avg_msz  = sum(r["rewards_min_size"] for r in grp) / len(grp)
        print(f"    {rate:>6.0f}  {len(grp):>4}  {g5:>4}  {100*g5/len(grp):>5.0f}%  {avg_pse:>12.5f}  {avg_msz:>12.0f}")

    # ---------- G6 by min_size band ----------
    print(f"\n[5] G6 by rewards_min_size band:")
    bands = [(1, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 9999)]
    for lo, hi in bands:
        grp = [r for r in results if lo <= r["rewards_min_size"] <= hi]
        if not grp:
            continue
        g6 = sum(1 for r in grp if r["g6_triggered"])
        avg_pse = sum(r["per_share_edge"] for r in grp) / len(grp)
        print(f"    min_size {lo:>5}–{hi if hi < 9999 else '∞':>5}: n={len(grp):>3}  G6={g6:>3} ({100*g6/len(grp):>4.0f}%)  avg_ps_edge={avg_pse:.5f}")

    # ---------- per_share_edge CDF ----------
    print(f"\n[6] per_share_edge cumulative distribution:")
    for t in [0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100]:
        below = sum(1 for r in results if r["per_share_edge"] < t)
        print(f"    ps_edge < {t:.3f}: {below:>3}/{n} = {100*below/max(n,1):5.1f}%")

    # ---------- Hungary position in the distribution ----------
    hun = [r for r in results if r["event_slug"] == "next-prime-minister-of-hungary"]
    if hun:
        print(f"\n[7] Hungary position in population:")
        for r in hun:
            rank = sorted(results, key=lambda x: x["per_share_edge"]).index(r) + 1
            print(f"    {r['market_slug'][:60]}")
            print(f"      ps_edge={r['per_share_edge']:.5f}  min_size={r['rewards_min_size']:.0f}  rank_by_ps_edge={rank}/{n} (1=lowest)")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

    # Save to file
    out_path = ROOT / "data" / "reports" / "g5_g6_scope_analysis_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "generated_ts":     datetime.now(timezone.utc).isoformat(),
            "n_eligible":       n,
            "g5_count":         g5_count,
            "g6_count":         g6_count,
            "both_count":       both_count,
            "neither_count":    neither_count,
            "positive_ev_count": pos_ev_count,
            "markets":          results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
