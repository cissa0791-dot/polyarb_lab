"""
reward_aware_maker_probe — CLI Entry Point
polyarb_lab / research_line / probe-only

Answers one question:
  Does the fee-enabled rewarded-market universe produce a positive
  reward-adjusted raw maker EV pool?

Universe: active, fee-enabled, reward-eligible markets from Gamma API only.
This is NOT tied to neg-risk family logic. It is a broader single-market
rewarded-maker probe on fee-enabled rewarded markets.

Usage (Windows PowerShell from repo root):
    py -3 research_lines/reward_aware_maker_probe/run_maker_probe.py
    py -3 research_lines/reward_aware_maker_probe/run_maker_probe.py --verbose
    py -3 research_lines/reward_aware_maker_probe/run_maker_probe.py --gamma-host https://gamma-api.polymarket.com

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/reward_aware_maker_probe/ only.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from this file's directory
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.discovery import (
    discover_fee_enabled_rewarded_markets,
    discovery_summary,
)
from research_lines.reward_aware_maker_probe.modules.ev_model import (
    evaluate_batch,
    build_ev_summary,
    ECON_POSITIVE_RAW_EV,
    ECON_NEGATIVE_RAW_EV,
    ECON_REJECTED_NO_BOOK,
    ECON_REJECTED_NO_REWARD,
    ECON_REJECTED_SPREAD_TOO_WIDE,
)
from research_lines.reward_aware_maker_probe.modules.paper_logger import (
    ProbeResult,
    log_probe,
    make_probe_id,
    load_probe_index,
    PROBE_DATA_DIR,
)

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"  # legacy alias, not used for discovery


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


def _print_discovery_summary(summary: dict) -> None:
    print(f"  fee_enabled_rewarded_markets : {summary.get('fee_enabled_rewarded_market_count', 0)}")
    print(f"  with_usable_book             : {summary.get('with_usable_book', 0)}")
    rates = summary.get("reward_daily_rate_usdc", {})
    if rates:
        print(f"  daily_rate_usdc (min/max/mean): "
              f"{rates.get('min', '?')}/{rates.get('max', '?')}/{rates.get('mean', '?')}")


def _print_ev_summary(results: list, verbose: bool = False) -> None:
    from collections import Counter
    counts = Counter(r.economics_class for r in results)
    print(f"  POSITIVE_RAW_EV              : {counts.get(ECON_POSITIVE_RAW_EV, 0)}")
    print(f"  NEGATIVE_RAW_EV              : {counts.get(ECON_NEGATIVE_RAW_EV, 0)}")
    print(f"  REJECTED_NO_BOOK             : {counts.get(ECON_REJECTED_NO_BOOK, 0)}")
    print(f"  REJECTED_NO_REWARD           : {counts.get(ECON_REJECTED_NO_REWARD, 0)}")
    print(f"  REJECTED_SPREAD_TOO_WIDE     : {counts.get(ECON_REJECTED_SPREAD_TOO_WIDE, 0)}")

    if verbose:
        pos = [r for r in results if r.economics_class == ECON_POSITIVE_RAW_EV]
        if pos:
            print()
            print("  Top positive raw EV candidates:")
            top = sorted(pos, key=lambda r: r.reward_adjusted_raw_ev, reverse=True)[:10]
            for r in top:
                print(
                    f"    [{r.economics_class[:4]}] ev={r.reward_adjusted_raw_ev:.6f} | "
                    f"spread={r.quoted_spread} | rate={r.reward_config_summary.get('reward_daily_rate_usdc', '?'):.1f}$/d | "
                    f"{r.market_slug[:50]}"
                )


def _print_final_verdict(n_positive: int) -> None:
    _section("Probe Verdict")
    if n_positive == 0:
        print("  RAW POOL: EMPTY")
        print("  No positive reward-adjusted raw EV found in fee-enabled rewarded universe.")
        print("  Decision: do not promote this line.")
        print("  Action: park as parked_research. Do not escalate.")
    elif n_positive <= 3:
        print(f"  RAW POOL: FRAGILE ({n_positive} positive candidates)")
        print("  Small positive raw pool found. Fragility unknown without multi-cycle check.")
        print("  Decision: classify as research/supporting line only.")
        print("  Action: run multiple cycles before escalation discussion.")
    else:
        print(f"  RAW POOL: NON-EMPTY ({n_positive} positive candidates)")
        print("  Positive reward-adjusted raw EV pool exists.")
        print("  Decision: classify as research candidate. Requires multi-cycle confirmation.")
        print("  Action: run >= 5 probe cycles. Do NOT claim profitability yet.")
    print()
    print("  NOTE: This is paper-only research. No orders submitted. No live execution.")
    _sep()


# ---------------------------------------------------------------------------
# Core probe logic
# ---------------------------------------------------------------------------

def run_probe(
    clob_host: str = CLOB_HOST,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> ProbeResult:
    """
    Run one full reward-aware maker probe scan.

    Paper-only. No order submission anywhere in this pipeline.
    """
    probe_timestamp = datetime.now(timezone.utc)
    probe_id = make_probe_id(probe_timestamp)
    probe_config = {
        "clob_host": clob_host,
        "output_dir": str(output_dir or PROBE_DATA_DIR),
        "probe_version": "v2",
    }

    print()
    _section("reward_aware_single_market_maker_probe — Probe Scan")
    print(f"  Probe ID   : {probe_id}")
    print(f"  Timestamp  : {probe_timestamp.isoformat()}")
    print(f"  CLOB host  : {clob_host}")
    print()

    # -------------------------------------------------------------------
    # Module 1: Discovery
    # -------------------------------------------------------------------
    _section("Module 1: Discovery — fee-enabled rewarded markets")
    raw_markets = discover_fee_enabled_rewarded_markets(clob_host=clob_host)
    disc_summary = discovery_summary(raw_markets)
    _print_discovery_summary(disc_summary)
    print()

    if not raw_markets:
        print("  WARNING: No fee-enabled rewarded markets discovered.")
        print("  Check Gamma API connectivity or universe definition.")

    # -------------------------------------------------------------------
    # Module 2: EV Model
    # -------------------------------------------------------------------
    _section("Module 2: EV Model — reward-adjusted raw EV decomposition")
    ev_results = evaluate_batch(raw_markets)
    _print_ev_summary(ev_results, verbose=verbose)
    print()

    # -------------------------------------------------------------------
    # Module 3: Paper Logger
    # -------------------------------------------------------------------
    _section("Module 3: Paper Logger")
    probe_result = ProbeResult(
        probe_id=probe_id,
        probe_timestamp=probe_timestamp,
        raw_markets=raw_markets,
        ev_results=ev_results,
        probe_config=probe_config,
    )
    probe_path = log_probe(probe_result, output_dir=output_dir)
    print(f"  Probe written : {probe_path}")

    index = load_probe_index(output_dir)
    print(f"  Total probes in index : {len(index)}")
    print()

    # -------------------------------------------------------------------
    # Summary + Verdict
    # -------------------------------------------------------------------
    ev_summary = build_ev_summary(raw_markets, ev_results)
    n_positive = ev_summary["positive_raw_maker_candidates"]
    best_raw = ev_summary.get("best_raw_candidate")

    _section("Probe Summary")
    print(f"  rewarded_market_count              : {ev_summary['rewarded_market_count']}")
    print(f"  fee_enabled_rewarded_market_count  : {ev_summary['fee_enabled_rewarded_market_count']}")
    print(f"  positive_raw_maker_candidates      : {n_positive}")
    print(f"  best_raw_candidate                 : {best_raw or 'none'}")
    print(f"  best_net_candidate                 : {ev_summary.get('best_net_candidate') or 'none'}")
    print()

    _print_final_verdict(n_positive)

    return probe_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "reward_aware_single_market_maker_probe — Probe Scanner\n"
            "Probe-only. No order submission. No live execution.\n"
            "Answers: does the fee-enabled rewarded-market universe have a "
            "non-empty positive reward-adjusted raw maker EV pool?"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clob-host",
        default=CLOB_HOST,
        help=f"CLOB API host (default: {CLOB_HOST})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Probe output directory (default: {PROBE_DATA_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-market EV details for positive candidates",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of probe cycles to run for persistence confirmation (default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=30.0,
        help="Seconds between cycles (default: 30.0)",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    if args.cycles == 1:
        run_probe(
            clob_host=args.clob_host,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        return

    # --- Multi-cycle confirmation mode ---
    cycle_positive_sets: list[set[str]] = []

    for cycle_num in range(1, args.cycles + 1):
        print(f"\n{'='*72}")
        print(f"  CYCLE {cycle_num} / {args.cycles}")
        print(f"{'='*72}")
        result = run_probe(
            clob_host=args.clob_host,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        pos_slugs = {
            r.market_slug
            for r in result.ev_results
            if r.economics_class == ECON_POSITIVE_RAW_EV
        }
        cycle_positive_sets.append(pos_slugs)

        if cycle_num < args.cycles:
            print(f"\n  Waiting {args.delay}s before next cycle...")
            time.sleep(args.delay)

    # --- Persistence summary ---
    _sep("=")
    print("  MULTI-CYCLE CONFIRMATION SUMMARY")
    _sep("=")
    print(f"  Cycles run : {args.cycles}")
    print()

    all_seen: set[str] = set().union(*cycle_positive_sets)
    print(f"  Unique positive slugs across all cycles : {len(all_seen)}")
    print()
    print(f"  {'Slug':<55} {'Cycles present':>14}  Persistent?")
    print(f"  {'-'*55} {'-'*14}  ----------")
    persistent: list[str] = []
    for slug in sorted(all_seen):
        count = sum(1 for s in cycle_positive_sets if slug in s)
        flag = "YES ***" if count == args.cycles else f"partial ({count}/{args.cycles})"
        print(f"  {slug[:55]:<55} {count:>7}/{args.cycles:<6}  {flag}")
        if count == args.cycles:
            persistent.append(slug)
    print()
    print(f"  Persistent across ALL {args.cycles} cycles : {len(persistent)}")
    print()

    # Final judgment
    _sep("=")
    print("  CONFIRMATION JUDGMENT")
    _sep("=")
    if len(persistent) >= 5:
        print(f"  CONTINUE — {len(persistent)} candidates persistent across all {args.cycles} cycles.")
        print("  Positive raw EV pool is stable. Eligible for next confirmation layer.")
        print("  Do NOT claim profitability. Do NOT submit orders.")
    elif len(persistent) >= 1:
        print(f"  BORDERLINE — {len(persistent)} candidate(s) persistent across all {args.cycles} cycles.")
        print("  Pool exists but thin. Run >= 5 cycles before any escalation.")
    else:
        print(f"  PARK — 0 candidates persistent across all {args.cycles} cycles.")
        print("  Positive raw EV candidates are transient. Do not escalate this line.")
    print()
    _sep("=")


if __name__ == "__main__":
    main()
