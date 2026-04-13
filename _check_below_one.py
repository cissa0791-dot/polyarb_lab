import json
from pathlib import Path

scan = json.loads(Path("data/research/neg_risk/scan_20260401T081739Z.json").read_text())
struct = scan["structural_results"]

sig = [
    s for s in struct
    if s.get("gap_direction") == "SUM_BELOW_ONE"
    and s.get("abs_gap", 0) >= 0.03
    and s.get("has_all_prices")
]

print(f"SUM_BELOW_ONE with prices: {len(sig)}")
for s in sorted(sig, key=lambda x: x["abs_gap"], reverse=True)[:10]:
    print(f"  {s['slug']} | gap={s['abs_gap']:.4f} | n={s['n_outcomes']}")
