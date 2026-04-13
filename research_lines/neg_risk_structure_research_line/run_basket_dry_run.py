"""
run_basket_dry_run.py — Neg-Risk Basket Executor Dry-Run Test
neg_risk_structure_research_line / polyarb_lab / research utility

Fetches top SIGNIFICANT_VIOLATION events from a fresh discovery pass,
runs them through the structural check, then passes the top N events
through BasketExecutor in dry_run=True mode.

No orders are submitted. No mainline state is touched.
Results written to data/research/neg_risk/baskets/

Usage:
    py -3 research_lines/neg_risk_structure_research_line/run_basket_dry_run.py
    py -3 research_lines/neg_risk_structure_research_line/run_basket_dry_run.py --top 5
    py -3 research_lines/neg_risk_structure_research_line/run_basket_dry_run.py --min-gap 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.neg_risk_structure_research_line.modules.discovery import discover_neg_risk_events
from research_lines.neg_risk_structure_research_line.modules.normalizer import normalize_batch
from research_lines.neg_risk_structure_research_line.modules.structural_check import check_batch
from research_lines.neg_risk_structure_research_line.basket_executor import BasketExecutor, BasketConfig
from src.domain.models import OrderMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP = "─" * 72


def main() -> None:
    parser = argparse.ArgumentParser(description="Neg-risk basket executor dry run")
    parser.add_argument("--top", type=int, default=3, help="Number of top events to test (default: 3)")
    parser.add_argument("--min-gap", type=float, default=0.030, help="Minimum abs_gap to attempt (default: 0.030)")
    parser.add_argument("--shares", type=float, default=10.0, help="Paper shares per leg (default: 10)")
    args = parser.parse_args()

    print(SEP)
    print("  NEG-RISK BASKET EXECUTOR — DRY RUN")
    print(f"  top={args.top}  min_gap={args.min_gap}  shares_per_leg={args.shares}")
    print(f"  mode=DRY_RUN (no orders submitted)")
    print(SEP)

    # Step 1: Discovery
    print("\n  Step 1: Discovery")
    raw_events = discover_neg_risk_events()
    print(f"  Found {len(raw_events)} neg-risk events")

    # Step 2: Normalize
    print("\n  Step 2: Normalization")
    normalized, failed_ids = normalize_batch(raw_events)
    print(f"  Normalized: {len(normalized)}  Failed: {len(failed_ids)}")

    # Step 3: Structural check
    print("\n  Step 3: Structural Check")
    structural = check_batch(normalized)

    # Filter to SIGNIFICANT_VIOLATION with sufficient gap and all prices available
    sig = [
        (norm, struct)
        for norm, struct in zip(normalized, structural)
        if struct.constraint_class == "SIGNIFICANT_VIOLATION"
        and norm.abs_gap >= args.min_gap
        and norm.has_all_prices
    ]
    sig.sort(key=lambda x: x[0].abs_gap, reverse=True)
    print(f"  SIGNIFICANT_VIOLATION (has_all_prices, gap>={args.min_gap}): {len(sig)}")

    if not sig:
        print("\n  No events meet dry-run criteria. Try lowering --min-gap.")
        return

    # Step 4: Basket executor dry run on top N
    print(f"\n{SEP}")
    print(f"  Step 4: Basket Executor Dry Run (top {min(args.top, len(sig))} events)")
    print(SEP)

    config = BasketConfig(
        shares_per_leg=args.shares,
        min_abs_gap=args.min_gap,
        dry_run=True,
    )
    executor = BasketExecutor(broker=None, config=config)

    for i, (event, _struct) in enumerate(sig[: args.top]):
        print(f"\n  [{i+1}] {event.slug}")
        print(f"      abs_gap={event.abs_gap:.4f}  implied_sum={event.implied_sum:.4f}  n_outcomes={len(event.outcomes)}")

        report = executor.execute(event, mode=OrderMode.PAPER)

        n_with_bid = sum(1 for o in event.outcomes if (o.yes_bid or 0.0) >= args.min_gap / 10)
        print(f"      basket_id  : {report.basket_id}")
        print(f"      status     : {report.status}")
        print(f"      legs_with_bid / total : {n_with_bid} / {len(event.outcomes)}")
        print(f"      net_collected (paper): ${report.net_collected_usd:.4f}")
        print(f"      fee_estimate (paper) : ${report.fee_estimate_usd:.4f}")
        print(f"      implied_gross_profit : ${report.implied_gross_profit_usd:.4f}")

        if report.rollback_note:
            print(f"      rollback   : {report.rollback_note}")

        if report.leg_results:
            print(f"      legs submitted: {len(report.leg_results)}")
            for lr in report.leg_results[:3]:
                q_short = lr.outcome_question[:50] + "…" if len(lr.outcome_question) > 50 else lr.outcome_question
                print(f"        leg{lr.leg_index}: bid={lr.yes_bid_used:.4f}  shares={lr.shares}  status={lr.status}  [{q_short}]")
            if len(report.leg_results) > 3:
                print(f"        … and {len(report.leg_results)-3} more legs")

    print(f"\n{SEP}")
    print(f"  Dry-run complete. Reports written to data/research/neg_risk/baskets/")
    print(SEP)


if __name__ == "__main__":
    main()
