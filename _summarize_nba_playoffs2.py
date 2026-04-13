"""
Extract and print all nba_finals_implies_playoffs pairs with execution edge.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Parse the YAML manually — look for blocks by discovery_rule
content = (ROOT / "data/reports/discovered_constraints_nba_playoffs_20260322.yaml").read_text(encoding="utf-8")

import re

# Extract all blocks for nba_finals_implies_playoffs
# Each block starts with "- name:" and ends before the next "- name:" or EOF
blocks = re.split(r'\n(?=- name:)', content)

rows = []
for block in blocks:
    if 'nba_finals_implies_playoffs' not in block:
        continue
    name_m = re.search(r'name:\s*(.+)', block)
    rank_m = re.search(r'current_execution_rank:\s*(\d+)', block)
    edge_m = re.search(r'execution_best_ask_edge_cents:\s*([-\d.]+)', block)
    rgap_m = re.search(r'relation_gap:\s*([-\d.]+)', block)
    lhs_yes_m = re.search(r'lhs_relation_ask:\s*([-\d.]+)', block)
    rhs_yes_m = re.search(r'rhs_relation_ask:\s*([-\d.]+)', block)
    lhs_no_m = re.search(r'lhs_execution_ask:\s*([-\d.]+)', block)
    rhs_ex_m = re.search(r'rhs_execution_ask:\s*([-\d.]+)', block)
    rows.append({
        "name": name_m.group(1).strip() if name_m else "?",
        "rank": int(rank_m.group(1)) if rank_m else 9999,
        "edge": float(edge_m.group(1)) if edge_m else None,
        "rgap": float(rgap_m.group(1)) if rgap_m else None,
        "lhs_yes": float(lhs_yes_m.group(1)) if lhs_yes_m else None,
        "rhs_yes": float(rhs_yes_m.group(1)) if rhs_yes_m else None,
        "lhs_no": float(lhs_no_m.group(1)) if lhs_no_m else None,
        "rhs_ex": float(rhs_ex_m.group(1)) if rhs_ex_m else None,
    })

# Deduplicate by name
seen = set()
uniq = []
for r in rows:
    if r["name"] not in seen:
        seen.add(r["name"])
        uniq.append(r)

uniq.sort(key=lambda x: x["rank"])

print(f"{'Rank':>5} {'Edge¢':>8} {'RGap':>8} {'LHS_YES':>8} {'RHS_YES':>8} {'LHS_NO':>8} {'RHS_NO':>8}  name")
for r in uniq:
    def fmt(v): return f"{v:>8.4f}" if v is not None else "    None"
    print(f"{r['rank']:>5} {fmt(r['edge'])} {fmt(r['rgap'])} {fmt(r['lhs_yes'])} {fmt(r['rhs_yes'])} {fmt(r['lhs_no'])} {fmt(r['rhs_ex'])}  {r['name']}")
