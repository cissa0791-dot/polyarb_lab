"""
neg_risk_structure_research_line — Top-Level Research Scanner
polyarb_lab / research_line / active

Orchestrates all 5 modules in sequence:
  1. Discovery   — fetch neg-risk events from Gamma API
  2. Normalizer  — convert to unified NegRiskEvent representation
  3. Structural  — detect pricing inconsistencies
  4. Filter      — assess execution quality (CLOB book fetch)
  5. Logger      — write paper-only scan results to disk

Then generates all 6 required output tables.

Usage:
    py -3 research_lines/neg_risk_structure_research_line/run_research_scan.py
    py -3 research_lines/neg_risk_structure_research_line/run_research_scan.py --no-clob
    py -3 research_lines/neg_risk_structure_research_line/run_research_scan.py --tables-only

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
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from this file's directory
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.neg_risk_structure_research_line.modules.discovery import (
    discover_neg_risk_events, discovery_summary
)
from research_lines.neg_risk_structure_research_line.modules.normalizer import (
    normalize_batch
)
from research_lines.neg_risk_structure_research_line.modules.structural_check import (
    check_batch
)
from research_lines.neg_risk_structure_research_line.modules.executable_filter import (
    filter_batch
)
from research_lines.neg_risk_structure_research_line.modules.paper_logger import (
    ScanResult, log_scan, make_scan_id, NEG_RISK_DATA_DIR, load_scan_index
)
from research_lines.neg_risk_structure_research_line.outputs.tables import (
    generate_all
)

# ---------------------------------------------------------------------------
# Defaults (read-only fetches; no order submission possible)
# ---------------------------------------------------------------------------

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
MIN_OUTCOMES = 2


# ---------------------------------------------------------------------------
# Logging setup
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

def _print_separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _print_section(title: str) -> None:
    _print_separator()
    print(f"  {title}")
    _print_separator()


def _print_discovery_summary(summary: dict[str, Any]) -> None:
    print(f"  Total neg-risk events found : {summary.get('total', 0)}")
    print(f"  negRisk flag set (Gamma)     : {summary.get('neg_risk_flagged', 0)}")
    print(f"  Locally classified only      : {summary.get('locally_classified_only', 0)}")
    dist = summary.get("outcome_count_distribution", {})
    print(f"  Outcome count (min/max/mean) : "
          f"{dist.get('min', '?')}/{dist.get('max', '?')}/{dist.get('mean', '?')}")
    print(f"  Binary (N=2)                 : {summary.get('events_n2', 0)}")
    print(f"  Small multi (N=3–5)          : {summary.get('events_n3_to_5', 0)}")
    print(f"  Large multi (N>=6)           : {summary.get('events_n6_plus', 0)}")


def _print_structural_summary(results: list) -> None:
    from collections import Counter
    counts = Counter(r.constraint_class for r in results)
    print(f"  SIGNIFICANT_VIOLATION        : {counts.get('SIGNIFICANT_VIOLATION', 0)}")
    print(f"  CONSTRAINT_VIOLATION         : {counts.get('CONSTRAINT_VIOLATION', 0)}")
    print(f"  BOUNDARY                     : {counts.get('BOUNDARY', 0)}")
    print(f"  AT_CONSTRAINT                : {counts.get('AT_CONSTRAINT', 0)}")
    print(f"  UNKNOWN (missing prices)     : {counts.get('UNKNOWN', 0)}")
    gaps = [r.abs_gap for r in results if r.abs_gap is not None]
    if gaps:
        print(f"  Mean abs_gap                 : {sum(gaps)/len(gaps):.4f}")
        print(f"  Max abs_gap                  : {max(gaps):.4f}")


def _print_execution_summary(results: list) -> None:
    from collections import Counter
    counts = Counter(r.execution_class for r in results)
    print(f"  EXECUTABLE                   : {counts.get('EXECUTABLE', 0)}")
    print(f"  RESEARCH_VALUE_ONLY          : {counts.get('RESEARCH_VALUE_ONLY', 0)}")
    print(f"  NOISE                        : {counts.get('NOISE', 0)}")
    print(f"  CLOB_UNAVAILABLE             : {counts.get('CLOB_UNAVAILABLE', 0)}")


def _print_top_violations(structural_results: list, n: int = 10) -> None:
    sorted_r = sorted(
        [r for r in structural_results if r.abs_gap >= 0.005],
        key=lambda r: r.abs_gap,
        reverse=True,
    )[:n]

    if not sorted_r:
        print("  No events above rounding floor (abs_gap >= 0.005)")
        return

    for r in sorted_r:
        print(
            f"  [{r.constraint_class[:4]}] {r.abs_gap:.4f} | "
            f"{r.gap_direction} | N={r.n_outcomes} | "
            f"{r.slug[:45] if r.slug else r.event_id[:45]}"
        )


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

def run_scan(
    gamma_host: str = GAMMA_HOST,
    clob_host: str = CLOB_HOST,
    min_outcomes: int = MIN_OUTCOMES,
    fetch_clob: bool = True,
    output_dir: Optional[Path] = None,
    generate_tables: bool = True,
    verbose: bool = False,
) -> ScanResult:
    """
    Run one full research scan and return the ScanResult.

    Paper-only: no order submission anywhere in this pipeline.
    """
    scan_timestamp = datetime.now(timezone.utc)
    scan_id = make_scan_id(scan_timestamp)
    scan_config = {
        "gamma_host": gamma_host,
        "clob_host": clob_host,
        "min_outcomes": min_outcomes,
        "fetch_clob": fetch_clob,
        "output_dir": str(output_dir or NEG_RISK_DATA_DIR),
    }

    print()
    _print_section(f"neg_risk_structure_research_line — Research Scan")
    print(f"  Scan ID    : {scan_id}")
    print(f"  Timestamp  : {scan_timestamp.isoformat()}")
    print(f"  CLOB fetch : {'enabled' if fetch_clob else 'disabled'}")
    print()

    # -----------------------------------------------------------------------
    # Module 1: Discovery
    # -----------------------------------------------------------------------
    _print_section("Module 1: Discovery")
    raw_events = discover_neg_risk_events(
        gamma_host=gamma_host,
        min_outcomes=min_outcomes,
    )
    summary = discovery_summary(raw_events)
    _print_discovery_summary(summary)
    print()

    if not raw_events:
        print("  WARNING: No neg-risk events discovered. Check Gamma API connectivity.")
        print("  Scan will complete with empty results.")

    # -----------------------------------------------------------------------
    # Module 2: Normalization
    # -----------------------------------------------------------------------
    _print_section("Module 2: Normalization")
    normalized_events, failed_ids = normalize_batch(raw_events)
    print(f"  Normalized : {len(normalized_events)}")
    print(f"  Failed     : {len(failed_ids)}")
    if failed_ids and verbose:
        print(f"  Failed IDs : {failed_ids}")
    print()

    # -----------------------------------------------------------------------
    # Module 3: Structural Check
    # -----------------------------------------------------------------------
    _print_section("Module 3: Structural Check")
    structural_results = check_batch(normalized_events)
    _print_structural_summary(structural_results)
    print()

    if verbose and structural_results:
        print("  Top violations by abs_gap:")
        _print_top_violations(structural_results, n=15)
        print()

    # -----------------------------------------------------------------------
    # Module 4: Executable Filter
    # -----------------------------------------------------------------------
    _print_section("Module 4: Executable Filter")
    execution_results = filter_batch(
        events=normalized_events,
        check_results=structural_results,
        clob_host=clob_host,
        fetch_clob=fetch_clob,
        only_gap_threshold=True,  # skip CLOB fetch for events below gap threshold
    )
    _print_execution_summary(execution_results)
    print()

    # -----------------------------------------------------------------------
    # Module 5: Paper Logger
    # -----------------------------------------------------------------------
    _print_section("Module 5: Paper Logger")
    scan_result = ScanResult(
        scan_id=scan_id,
        scan_timestamp=scan_timestamp,
        raw_events=raw_events,
        normalized_events=normalized_events,
        failed_normalization_ids=failed_ids,
        structural_results=structural_results,
        execution_results=execution_results,
        scan_config=scan_config,
    )
    scan_path = log_scan(scan_result, output_dir=output_dir)
    print(f"  Scan written : {scan_path}")

    scan_index = load_scan_index(output_dir)
    print(f"  Total scans in index : {len(scan_index)}")
    print()

    # -----------------------------------------------------------------------
    # Output Tables
    # -----------------------------------------------------------------------
    if generate_tables:
        _print_section("Output Tables")
        written = generate_all(output_dir=output_dir or NEG_RISK_DATA_DIR)
        for name, path in written.items():
            print(f"  {name}: {path.name}")
        print()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    _print_section("Scan Complete — Research Summary")
    n_violation = sum(
        1 for r in structural_results
        if r.constraint_class in ("CONSTRAINT_VIOLATION", "SIGNIFICANT_VIOLATION")
    )
    n_executable = sum(
        1 for r in execution_results if r.execution_class == "EXECUTABLE"
    )
    n_research_only = sum(
        1 for r in execution_results if r.execution_class == "RESEARCH_VALUE_ONLY"
    )

    print(f"  Events discovered         : {len(raw_events)}")
    print(f"  Events normalized         : {len(normalized_events)}")
    print(f"  Constraint violations     : {n_violation}")
    print(f"  EXECUTABLE classification : {n_executable}")
    print(f"  RESEARCH_VALUE_ONLY       : {n_research_only}")
    print()

    if n_executable == 0 and n_violation == 0:
        print("  RESEARCH VERDICT: No structural inconsistencies detected this scan.")
        print("  Continue monitoring. Do not escalate. Do not claim edge.")
    elif n_executable == 0 and n_violation > 0:
        print(f"  RESEARCH VERDICT: {n_violation} structural gap(s) detected.")
        print("  Execution quality insufficient for paper position at current depth/spread.")
        print("  Classified as RESEARCH_VALUE_ONLY. Continue monitoring.")
    else:
        print(f"  RESEARCH VERDICT: {n_executable} EXECUTABLE event(s) detected.")
        print("  Structural gap + execution quality gates passed.")
        print("  Requires >= 5 scans with consistent EXECUTABLE classification before escalation.")
        print("  Do NOT claim profitability. Do NOT escalate yet.")

    print()
    print("  NOTE: This is paper-only research. No orders submitted. No live execution.")
    _print_separator()

    return scan_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "neg_risk_structure_research_line — Research Scanner\n"
            "Paper-only. No order submission. No live execution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gamma-host",
        default=GAMMA_HOST,
        help=f"Gamma API host (default: {GAMMA_HOST})",
    )
    parser.add_argument(
        "--clob-host",
        default=CLOB_HOST,
        help=f"CLOB API host (default: {CLOB_HOST})",
    )
    parser.add_argument(
        "--min-outcomes",
        type=int,
        default=MIN_OUTCOMES,
        help=f"Minimum number of outcomes per event (default: {MIN_OUTCOMES})",
    )
    parser.add_argument(
        "--no-clob",
        action="store_true",
        help="Skip CLOB book fetches (faster, lower API load — use Gamma fallback only)",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip generating output tables at end of scan",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate output tables from existing scan data (no new scan)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Scan output directory (default: {NEG_RISK_DATA_DIR})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-event structural violation details",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    if args.tables_only:
        _print_section("neg_risk_structure_research_line — Table Generation Only")
        written = generate_all(output_dir=args.output_dir)
        for name, path in written.items():
            print(f"  {name}: {path}")
        print()
        return

    run_scan(
        gamma_host=args.gamma_host,
        clob_host=args.clob_host,
        min_outcomes=args.min_outcomes,
        fetch_clob=not args.no_clob,
        output_dir=args.output_dir,
        generate_tables=not args.no_tables,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
