"""Read G5/G6 analysis JSON and print event-level summary for sweep planning."""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

data = json.loads((ROOT / "data/reports/g5_g6_scope_analysis_latest.json").read_text(encoding="utf-8"))
markets = data["markets"]

events: dict[str, dict] = defaultdict(lambda: {
    "n_markets": 0, "pos_ev": 0, "total_ev": 0.0,
    "min_sizes": set(), "neg_risks": set(), "rates": set(),
})
for m in markets:
    slug = m["event_slug"]
    events[slug]["n_markets"] += 1
    if m["total_ev"] > 0:
        events[slug]["pos_ev"] += 1
    events[slug]["total_ev"] += m["total_ev"]
    events[slug]["min_sizes"].add(m["rewards_min_size"])
    events[slug]["neg_risks"].add(m["neg_risk"])
    events[slug]["rates"].add(m["reward_daily_rate"])

ranked = sorted(events.items(), key=lambda x: x[1]["total_ev"], reverse=True)

print(f"Total distinct events: {len(ranked)}")
print()
hdr = f"{'event_slug':<52} {'n':>2} {'pos':>3} {'tot_ev':>7} {'min_sz':>12} {'neg_risk':>10} {'rates':>20}"
print(hdr)
print("-" * 115)
for slug, v in ranked:
    print(
        f"{slug[:51]:<52} {v['n_markets']:>2} {v['pos_ev']:>3} {v['total_ev']:>7.4f}"
        f"  {str(sorted(v['min_sizes'])):>12}"
        f"  {str(sorted(v['neg_risks'])):>10}"
        f"  {str(sorted(v['rates'])):>20}"
    )
