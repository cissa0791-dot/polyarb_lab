"""
Extract and print all nba_finals_implies_playoffs pairs with execution edge.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

with open(ROOT / "data/reports/discovered_constraints_nba_playoffs_20260322.yaml", encoding="utf-8") as f:
    data = yaml.safe_load(f)

pairs = [c for c in data if isinstance(c, dict) and c.get("discovery_rule") == "nba_finals_implies_playoffs"]

rows = []
for p in pairs:
    meta = p.get("metadata", {})
    rows.append({
        "name": p["name"],
        "rank": meta.get("current_execution_rank"),
        "edge": meta.get("execution_best_ask_edge_cents"),
        "relation_gap": meta.get("relation_gap"),
        "lhs_ask": meta.get("lhs_relation_ask"),
        "rhs_ask": meta.get("rhs_relation_ask"),
        "lhs_no_ask": meta.get("lhs_execution_ask"),
        "rhs_yes_ask": meta.get("rhs_execution_ask"),
    })

rows.sort(key=lambda x: x["rank"] if x["rank"] else 9999)
print(f"{'Rank':>5} {'Edge¢':>8} {'RelGap':>8} {'LHS_YES':>8} {'RHS_YES':>8} {'LHS_NO':>8}  name")
for r in rows:
    print(f"{r['rank']:>5} {r['edge']:>8.4f} {r['relation_gap']:>8.4f} {r['lhs_ask']:>8.3f} {r['rhs_ask']:>8.3f} {r['lhs_no_ask']:>8.3f}  {r['name']}")
