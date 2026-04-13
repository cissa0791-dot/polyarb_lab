"""
Check SUM(YES ask) < 1.0 for SUM_BELOW_ONE candidates.
BUY YES arb: buy all legs at ask, one pays $1 at resolution.
Profit = 1.0 - SUM(YES ask) - fees
"""
import json
from pathlib import Path

scan = json.loads(Path("data/research/neg_risk/scan_20260401T081739Z.json").read_text())
struct = scan["structural_results"]

TAKER_FEE = 0.01  # 1% per leg

candidates = [
    s for s in struct
    if s.get("gap_direction") == "SUM_BELOW_ONE"
    and s.get("abs_gap", 0) >= 0.03
    and s.get("has_all_prices")
]

print(f"{'slug':<52} {'n':>3} {'sum_mid':>8} {'sum_ask':>8} {'fee_est':>8} {'net_profit':>10} {'viable':>6}")
print("-" * 100)

for s in sorted(candidates, key=lambda x: x["abs_gap"], reverse=True):
    outcomes = s.get("outcome_details", [])
    if not outcomes:
        print(f"  {s['slug']:<50} — no outcome_details")
        continue

    yes_asks = []
    yes_mids = []
    for o in outcomes:
        mid = o.get("yes_mid")
        if mid is None:
            break
        # Estimate ask: outcome_details has yes_mid and no_mid but not yes_ask directly
        # Check if yes_ask is stored
        ask = o.get("yes_ask") or o.get("cost_yes_direct")
        yes_mids.append(mid)
        yes_asks.append(ask)

    sum_mid = sum(yes_mids) if yes_mids else None
    has_asks = all(a is not None for a in yes_asks)
    sum_ask = sum(yes_asks) if has_asks else None

    n = s["n_outcomes"]
    fee_est = TAKER_FEE * (sum_ask or sum_mid or 0) if (sum_ask or sum_mid) else None
    net = (1.0 - sum_ask - (fee_est or 0)) if sum_ask is not None else None
    viable = "YES" if (net is not None and net > 0) else ("NO" if net is not None else "?")

    sum_mid_str = f"{sum_mid:.4f}" if sum_mid else "?"
    sum_ask_str = f"{sum_ask:.4f}" if sum_ask else f"~{(sum_mid or 0) + 0.01*n:.4f}*"
    fee_str = f"{fee_est:.4f}" if fee_est else "?"
    net_str = f"{net:.4f}" if net is not None else "?"

    print(f"  {s['slug']:<50} {n:>3} {sum_mid_str:>8} {sum_ask_str:>8} {fee_str:>8} {net_str:>10} {viable:>6}")

print()
print("* = estimated ask (mid + 0.01 per leg) where yes_ask not stored")
print("Viable = net_profit > 0 after 1% taker fee")
