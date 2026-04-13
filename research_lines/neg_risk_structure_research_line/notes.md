# neg_risk_structure_research_line — running notes
# polyarb_lab / research_line

---

## 2026-03-21 — Line created

### Context

Reward/rebate-aware market making tested locally and found unprofitable. That line downgraded.
New priority: Polymarket neg-risk structure research.

Neg risk is an official Polymarket market structure (not a community claim):
- Official adapter contract: Polymarket/neg-risk-ctf-adapter
- Official Python client: Polymarket/py-clob-client
- Official docs: docs.polymarket.com

### Hypothesis registry at creation

7 hypotheses registered (NEG-001 through NEG-007):
- 7 pending, 0 park, 0 reject, 0 escalate

**Priority test order:**

1. NEG-001 (sum_constraint) — foundational
   - Determines if any structural gap exists at all.
   - If gaps are consistently near zero, park the entire line.
   - Run: `py -3 research_lines/neg_risk_structure_research_line/run_research_scan.py --no-clob`
   - Use `--no-clob` for first scans to confirm gap structure before burning CLOB API calls.

2. NEG-005 (execution_quality) — gate check
   - Once NEG-001 shows abs_gap > 0.01 in >= 3 events, run CLOB fetches.
   - Run: `py -3 research_lines/neg_risk_structure_research_line/run_research_scan.py`
   - If THIN_ALL dominates, classify line as RESEARCH_VALUE_ONLY. Do not escalate.

3. NEG-004 (directional_bias) — analytical
   - Once >= 10 events with abs_gap > 0.01 accumulated, run table analysis.
   - Run: `py -3 research_lines/neg_risk_structure_research_line/outputs/tables.py --table useful_vs_noise`

4. NEG-002 (conversion_arbitrage) — requires CLOB data
   - Depends on NEG-001 + NEG-005 confirmation.
   - Requires CLOB bid/ask for equivalent_gap computation.

5. NEG-006 (persistence) + NEG-003 (temporal) + NEG-007 (comparative)
   - Purely analytical — run after >= 5 scan sessions logged.
   - Run: `py -3 research_lines/neg_risk_structure_research_line/outputs/tables.py --table all`

### Module structure

```
modules/
  discovery.py        — Gamma API event discovery (negRisk filter)
  normalizer.py       — multi-outcome → NegRiskEvent with implied_sum
  structural_check.py — constraint gap classification + equivalent_gap
  executable_filter.py — CLOB depth / spread / slippage gates
  paper_logger.py     — JSON logging to data/research/neg_risk/

outputs/
  tables.py           — all 6 required research tables

run_research_scan.py  — top-level orchestrator (CLI)
```

### Isolation verified

- No imports from src/ in any module
- All data written to data/research/neg_risk/ only
- No order submission in any module
- No connection to mainline paper.db or data/processed/

---

## Iteration log

(append entries below as scan sessions are run)

### First scan command
```powershell
# From D:\Issac\polyarb_lab
py -3 research_lines/neg_risk_structure_research_line/run_research_scan.py --no-clob --verbose
```

### Decision gate after first scan
- If total_discovered = 0: Gamma API neg_risk filter not supported or no active events.
  Action: inspect raw Gamma response, adjust discovery._is_neg_risk_event() heuristics.

- If n_violation = 0: All events AT_CONSTRAINT or BOUNDARY.
  Action: run 2 more scans. If still 0, park NEG-001 and re-evaluate line.

- If n_violation > 0: Foundation confirmed.
  Action: enable CLOB fetches, run full scan, observe execution quality.

