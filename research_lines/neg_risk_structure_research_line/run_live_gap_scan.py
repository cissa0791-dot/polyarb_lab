"""
neg_risk_structure_research_line — Live Gap Scanner CLI
polyarb_lab / research_line / active

Standalone runner for live_gap_scanner.py.
Discovers neg-risk events, normalises them, then fetches all YES-leg bids
in parallel and computes:

  live_gap = 1 - sum(best_bid_i for all legs, parallel CLOB fetch)

If live_gap > FEE_HURDLE (0.020): gap may survive round-trip fees.

Usage:
    py -3 research_lines/neg_risk_structure_research_line/run_live_gap_scan.py
    py -3 research_lines/neg_risk_structure_research_line/run_live_gap_scan.py --cycles 5
    py -3 research_lines/neg_risk_structure_research_line/run_live_gap_scan.py --cycles 3 --delay 5.0 --verbose

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/neg_risk/ only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.neg_risk_structure_research_line.modules.discovery import (
    discover_neg_risk_events,
)
from research_lines.neg_risk_structure_research_line.modules.normalizer import (
    normalize_batch,
)
from research_lines.neg_risk_structure_research_line.modules.live_gap_scanner import (
    scan_with_persistence,
    scan_summary,
    FEE_HURDLE,
    DEFAULT_CYCLES,
    DEFAULT_CYCLE_DELAY_SEC,
)

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
NEG_RISK_DATA_DIR = Path("data/research/neg_risk")


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live gap scanner — parallel CLOB bid fetch for neg-risk baskets.\n"
            "Paper-only. No order submission. No live execution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gamma-host", default=GAMMA_HOST)
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help=f"Number of scan cycles for persistence check (default: {DEFAULT_CYCLES})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_CYCLE_DELAY_SEC,
        help=f"Seconds between cycles (default: {DEFAULT_CYCLE_DELAY_SEC})",
    )
    parser.add_argument(
        "--min-outcomes",
        type=int,
        default=2,
        help="Minimum outcomes per event (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this file (default: data/research/neg_risk/live_gap_<ts>.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all events, not just LIVE_EDGE",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    print()
    _section("Live Gap Scanner — neg_risk_structure_research_line")
    print(f"  Timestamp  : {datetime.now(timezone.utc).isoformat()}")
    print(f"  CLOB host  : {args.clob_host}")
    print(f"  Cycles     : {args.cycles}  (delay={args.delay}s between cycles)")
    print(f"  Fee hurdle : {FEE_HURDLE}  (live_gap must exceed this to classify LIVE_EDGE)")
    print()

    # Step 1: Discovery
    _section("Step 1: Discovery")
    raw_events = discover_neg_risk_events(
        gamma_host=args.gamma_host,
        min_outcomes=args.min_outcomes,
    )
    print(f"  Discovered : {len(raw_events)} neg-risk events")
    print()

    if not raw_events:
        print("  WARNING: No events discovered. Check Gamma API connectivity.")
        return

    # Step 2: Normalise
    _section("Step 2: Normalise")
    normalized, failed = normalize_batch(raw_events)
    print(f"  Normalised : {len(normalized)}  |  Failed: {len(failed)}")
    print()

    if not normalized:
        print("  WARNING: No events normalised.")
        return

    # Step 3: Live gap scan
    _section("Step 3: Live Gap Scan (parallel CLOB bids)")
    results = scan_with_persistence(
        normalized,
        clob_host=args.clob_host,
        cycles=args.cycles,
        cycle_delay_sec=args.delay,
    )

    summary = scan_summary(results)
    print(f"  Total events in universe : {summary['total_events']}")
    print(f"  Actually scanned         : {summary.get('scanned_count', '?')}  (legs attempted)")
    print(f"  Not scanned (cap/filter) : {summary.get('not_scanned_count', '?')}")
    print(f"  LIVE_EDGE                : {summary['live_edge_count']}")
    print(f"  ASK_ARB (real buy arb)   : {summary.get('ask_arb_count', 0)}")
    arb_profit = summary.get("ask_arb_profit_summary", {})
    if arb_profit.get("total_net_profit_usd") is not None:
        print(f"  Total net profit est.    : ${arb_profit['total_net_profit_usd']:.4f}  (top-of-book, pre-execution)")
    print(f"  SUBTHRESHOLD             : {summary['subthreshold_count']}")
    print(f"  NO_EDGE                  : {summary['no_edge_count']}")
    print(f"  PARTIAL_BOOK             : {summary['partial_book_count']}")
    print(f"  Avg fetch latency        : {summary.get('avg_fetch_latency_ms', '?')} ms")

    gap_stats = summary.get("live_gap_stats", {})
    if gap_stats.get("min") is not None:
        print(
            f"  live_gap (min/max/mean)  : "
            f"{gap_stats['min']} / {gap_stats['max']} / {gap_stats['mean']}"
        )

    drift_stats = summary.get("gap_drift_stats", {})
    if drift_stats.get("mean") is not None:
        print(
            f"  gap_drift mean           : {drift_stats['mean']}"
            f"  (+ve = live bids show more gap than stale Gamma prices)"
        )
    print()

    # Print LIVE_EDGE results
    live_edge_events = summary.get("live_edge_events", [])
    result_by_slug = {r.slug: r for r in results}
    if live_edge_events:
        _section("LIVE_EDGE Events")
        for ev in live_edge_events:
            ask_gap_val = ev.get("ask_gap")
            if ask_gap_val is None:
                ask_tag = "ask=N/A"
            elif ask_gap_val > FEE_HURDLE:
                ask_tag = f"ask_gap={ask_gap_val:.4f} *** REAL ARB"
            elif ask_gap_val > 0:
                ask_tag = f"ask_gap={ask_gap_val:.4f} (sub-fee)"
            else:
                ask_tag = f"ask_gap={ask_gap_val:.4f} (spread only)"
            bid_dep = ev.get("total_bid_depth_usd")
            ask_dep = ev.get("total_ask_depth_usd")
            depth_tag = (
                f"bid_depth=${bid_dep:.2f} ask_depth=${ask_dep:.2f}"
                if bid_dep is not None and ask_dep is not None
                else ""
            )
            print(
                f"  [{ev['n_outcomes']}L] live_gap={ev['live_gap']:.4f}"
                f" | {ask_tag}"
                f" | persist={ev['fee_threshold_cycles']}/{ev['cycle_count']}"
                f" | {ev['slug'][:45]}"
            )
            if depth_tag:
                print(f"         {depth_tag}")
            # Constrained profit detail for real arb events
            if ask_gap_val is not None and ask_gap_val > FEE_HURDLE:
                r = result_by_slug.get(ev["slug"])
                if r and r.constrained_net_profit_usd is not None:
                    print(
                        f"         DEPTH: shares={r.constrained_shares:.2f}"
                        f" | cost=${r.constrained_cost_usd:.4f}"
                        f" | gross=${r.constrained_gross_profit_usd:.4f}"
                        f" | net=${r.constrained_net_profit_usd:.4f}"
                        f" (after {FEE_HURDLE*100:.0f}% fee est.)"
                        f" | binding depth=${r.binding_leg_depth_usd:.2f}"
                        f" @ask={r.binding_leg_ask:.4f}"
                    )
                if r and r.sweep_profile:
                    sweep_str = "  ".join(
                        f"{k}sh→{'N/A' if v is None else ('+' if v > 0 else '') + f'{v:.4f}'}"
                        for k, v in r.sweep_profile.items()
                        if v is not None or int(k) <= 50
                    )
                    print(f"         SWEEP: {sweep_str}")
                    if r.max_profitable_shares is not None:
                        print(f"         MAX FILL: {r.max_profitable_shares:.0f} shares before gap collapses")
        print()
    elif args.verbose:
        sub = sorted(
            [r for r in results if r.live_gap is not None],
            key=lambda r: r.live_gap or -999,
            reverse=True,
        )[:15]
        if sub:
            _section("Top Events by live_gap (no LIVE_EDGE found)")
            for r in sub:
                print(
                    f"  [{r.n_outcomes}L] {r.gap_class:<12}"
                    f" live_gap={r.live_gap:.4f}"
                    f" gamma_gap={r.gamma_gap:.4f}"
                    f" | {r.slug[:50]}"
                )
            print()

    # Verdict
    _section("Verdict")
    if summary["live_edge_count"] == 0:
        print("  No LIVE_EDGE found in current scan.")
        print("  All gaps are subthreshold, negative, or books are incomplete.")
        print("  Do not escalate. Continue monitoring.")
    else:
        print(f"  {summary['live_edge_count']} LIVE_EDGE basket(s) found.")
        if args.cycles > 1:
            persistent = [
                ev for ev in live_edge_events
                if ev["fee_threshold_cycles"] == args.cycles
            ]
            print(
                f"  Persistent across all {args.cycles} cycles: {len(persistent)}"
            )
        print()
        print("  NOTE: Raw gap only. No fill guarantee. No profitability claim.")
        print("  Action: Run >= 5 cycles before any escalation discussion.")

    print()
    print("  Paper-only research. No orders submitted.")
    _sep()
    print()

    # Write results to disk
    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        NEG_RISK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = NEG_RISK_DATA_DIR / f"live_gap_{ts}.json"

    output_payload: dict[str, Any] = {
        "scan_timestamp": datetime.now(timezone.utc).isoformat(),
        "cycles": args.cycles,
        "cycle_delay_sec": args.delay,
        "clob_host": args.clob_host,
        "fee_hurdle": FEE_HURDLE,
        "summary": summary,
        "results": [r.to_log_dict() for r in results],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"  Results written : {out_path}")


if __name__ == "__main__":
    main()
