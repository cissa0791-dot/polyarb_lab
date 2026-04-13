"""
Runtime-closure test for negRisk maker pipeline.

Runs TWO cycles of ResearchRunner (same instance, persisted ledger) targeted at
neg_risk_rebalancing only.

Cycle 1: opens positions, no exits expected (positions are new).
Cycle 2: re-marks open positions; edge-decay (EDGE_DECAY) or age (MAX_HOLDING_AGE)
          triggers close, producing realized_pnl and closed_positions > 0.

Prints per-cycle: raw candidates, qualified, risk accepted/rejected,
orders/fills, open/closed positions, realized_pnl, unrealized_pnl, system errors.

No live trading. No order submission to live broker.
Read-only market data. Paper-only candidate store writes.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.runner import ResearchRunner

SETTINGS = ROOT / "config" / "settings.yaml"

EXPERIMENT = {
    "experiment_label": "neg_risk_runtime_closure_20260322",
    "campaign_target_strategy_families": ["neg_risk_rebalancing"],
}


def _print_cycle(label: str, summary) -> None:
    meta = summary.metadata
    raw_by_family = meta.get("raw_candidates_by_family", {})
    qual_by_family = meta.get("qualified_candidates_by_family", {})

    print(f"\n=== {label} ===")
    print(f"  raw_candidates   : {raw_by_family.get('neg_risk_rebalancing', 0)}")
    print(f"  qualified        : {qual_by_family.get('neg_risk_rebalancing', 0)}")
    print(f"  risk_accepted    : {summary.risk_accepted}")
    print(f"  risk_rejected    : {summary.risk_rejected}")
    print(f"  orders_created   : {summary.paper_orders_created}")
    print(f"  fills            : {summary.fills}")
    print(f"  open_positions   : {summary.open_positions}")
    print(f"  closed_positions : {summary.closed_positions}")
    print(f"  realized_pnl     : {summary.realized_pnl:.6f}")
    print(f"  unrealized_pnl   : {summary.unrealized_pnl:.6f}")
    print(f"  system_errors    : {summary.system_errors}")
    print(f"  elapsed          : {(summary.ended_ts - summary.started_ts).total_seconds():.1f}s")

    if summary.rejection_reason_counts:
        print("  rejection_counts :", dict(sorted(
            summary.rejection_reason_counts.items(), key=lambda x: -x[1]
        )))


def main():
    print("Initialising ResearchRunner...", flush=True)
    runner = ResearchRunner(settings_path=str(SETTINGS))

    print("\nCycle 1 — opens negRisk positions...", flush=True)
    summary1 = runner.run_once(experiment_context=EXPERIMENT)
    _print_cycle("Cycle 1 — OPEN", summary1)

    print("\nCycle 2 — re-marks positions; exits expected (EDGE_DECAY / MAX_HOLDING_AGE)...", flush=True)
    summary2 = runner.run_once(experiment_context=EXPERIMENT)
    _print_cycle("Cycle 2 — EXIT", summary2)

    closed = summary2.closed_positions
    rpnl = summary2.realized_pnl
    print(f"\n--- Exit-path verdict ---")
    print(f"  Positions closed in cycle 2 : {closed}")
    print(f"  realized_pnl after cycle 2  : {rpnl:.6f}")
    if closed > 0:
        print("  PASS — exit path is live; realized_pnl is now populated.")
    else:
        print("  WARN — no positions closed; check max_holding_sec / edge_decay_bid_delta config.")


if __name__ == "__main__":
    main()
