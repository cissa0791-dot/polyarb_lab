"""
reward_aware_single_market_maker_executable_conversion_line — CLI
polyarb_lab / research_line / audit-only

Runs the full pipeline (discovery → EV model → executable audit) and
reports which of the 13 confirmed POSITIVE_RAW_EV candidates survive
four executable conversion gates.

Usage (Windows PowerShell from repo root):
    py -3 research_lines/reward_aware_maker_probe/run_executable_audit.py
    py -3 research_lines/reward_aware_maker_probe/run_executable_audit.py --verbose

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/reward_aware_maker_probe/executable_audit/ only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.discovery import (
    discover_fee_enabled_rewarded_markets,
)
from research_lines.reward_aware_maker_probe.modules.ev_model import (
    ECON_POSITIVE_RAW_EV,
    evaluate_batch,
)
from research_lines.reward_aware_maker_probe.modules.executable_audit import (
    CAPITAL_THRESHOLD_USD,
    EV_ROC_MIN,
    EXEC_POSITIVE,
    EXEC_REJECTED,
    NEAR_RESOLUTION_HIGH,
    NEAR_RESOLUTION_LOW,
    REWARD_FLOOR_USDC,
    REWARD_ROC_DAILY_INFO,
    audit_batch,
    build_audit_summary,
)

CLOB_HOST = "https://clob.polymarket.com"
AUDIT_DATA_DIR = Path("data/research/reward_aware_maker_probe/executable_audit")


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
            "Executable conversion audit — reward_aware_single_market_maker\n"
            "Paper-only. No order submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write JSON results here "
            "(default: data/research/reward_aware_maker_probe/executable_audit/audit_<ts>.json)"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full per-slug table including rejected candidates",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    print()
    _section("reward_aware_single_market_maker_executable_conversion_line")
    print(f"  Timestamp  : {datetime.now(timezone.utc).isoformat()}")
    print(f"  CLOB host  : {args.clob_host}")
    print()
    print("  Executable gates applied to each POSITIVE_RAW_EV candidate:")
    print(f"    E1  EC_CAPITAL_INTENSIVE : quote_capital > ${CAPITAL_THRESHOLD_USD:.0f}  [redesigned: was $50]")
    print(f"    E2  EC_NEAR_RESOLUTION   : midpoint < {NEAR_RESOLUTION_LOW} or > {NEAR_RESOLUTION_HIGH}")
    print(f"    E3  EC_ROC_TOO_LOW       : raw_ev / capital < {EV_ROC_MIN} ({EV_ROC_MIN * 100:.1f}%)  [redesigned: was 0.5%]")
    print(f"    E4  EC_REWARD_TOO_LOW    : reward_contribution < ${REWARD_FLOOR_USDC:.2f}/day")
    print(f"    [i] reward_roc_daily     : reward_contribution / capital (informational, floor {REWARD_ROC_DAILY_INFO * 100:.1f}%/day)")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discovery
    # -----------------------------------------------------------------------
    _section("Step 1: Discovery")
    raw_markets = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
    with_book = sum(1 for m in raw_markets if m.has_usable_book())
    print(f"  Discovered : {len(raw_markets)} fee-enabled rewarded markets")
    print(f"  With book  : {with_book}")
    print()

    if not raw_markets:
        print("  WARNING: 0 markets discovered. Check CLOB API connectivity.")
        return

    # -----------------------------------------------------------------------
    # Step 2: EV model
    # -----------------------------------------------------------------------
    _section("Step 2: EV Model")
    ev_results = evaluate_batch(raw_markets)
    pos_raw = [r for r in ev_results if r.economics_class == ECON_POSITIVE_RAW_EV]
    print(f"  POSITIVE_RAW_EV : {len(pos_raw)}")
    print(f"  Other           : {len(ev_results) - len(pos_raw)}")
    print()

    if not pos_raw:
        _section("Result")
        print("  0 positive raw EV candidates found in this scan.")
        print("  PARK — raw pool appears empty. Re-run probe to confirm.")
        return

    # -----------------------------------------------------------------------
    # Step 3: Executable audit
    # -----------------------------------------------------------------------
    _section("Step 3: Executable Audit")
    audit_results = audit_batch(ev_results, raw_markets)
    summary = build_audit_summary(audit_results)

    survivors = [r for r in audit_results if r.executable_verdict == EXEC_POSITIVE]
    rejected_list = [r for r in audit_results if r.executable_verdict == EXEC_REJECTED]

    print(f"  Raw-positive candidates audited : {len(audit_results)}")
    print(f"  Executable-positive survivors   : {len(survivors)}")
    print(f"  Rejected by executable gates    : {len(rejected_list)}")
    print()

    # Per-slug table (survivors always shown; rejected shown with --verbose)
    col_slug    = 48
    col_verdict = 14
    header = (
        f"  {'Slug':<{col_slug}} {'Verdict':<{col_verdict}}"
        f"  {'raw_ev':>9}  {'roc%':>6}  {'rew_roc%':>8}  {'capital':>8}  {'mid':>5}  {'rate$/d':>7}  Codes"
    )
    divider = (
        f"  {'-'*col_slug} {'-'*col_verdict}"
        f"  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*7}  -----"
    )

    if survivors or args.verbose:
        print(header)
        print(divider)

    def _print_row(r) -> None:
        roc_str     = f"{r.ev_roc * 100:.2f}" if r.ev_roc is not None else "N/A"
        rew_roc_str = f"{r.reward_roc_daily * 100:.2f}" if r.reward_roc_daily is not None else "N/A"
        cap_str     = f"${r.quote_capital_usd:.2f}" if r.quote_capital_usd is not None else "N/A"
        mid_str     = f"{r.midpoint:.3f}" if r.midpoint is not None else "N/A"
        codes       = ",".join(r.rejection_codes) if r.rejection_codes else "—"
        tag         = "EXEC_POS ***" if r.executable_verdict == EXEC_POSITIVE else "EXEC_REJ"
        print(
            f"  {r.market_slug[:col_slug]:<{col_slug}} {tag:<{col_verdict}}"
            f"  {r.raw_ev:>9.6f}  {roc_str:>6}  {rew_roc_str:>8}  {cap_str:>8}  {mid_str:>5}"
            f"  {r.reward_rate_daily_usdc:>7.1f}  {codes}"
        )

    for r in sorted(survivors, key=lambda x: x.raw_ev, reverse=True):
        _print_row(r)

    if args.verbose:
        for r in sorted(rejected_list, key=lambda x: x.raw_ev, reverse=True):
            _print_row(r)

    print()

    # Rejection reason breakdown
    code_counts = summary.get("rejection_code_counts", {})
    if code_counts:
        _section("Rejection Breakdown")
        for code, count in sorted(code_counts.items(), key=lambda x: -x[1]):
            print(f"  {code:<30} : {count} candidate(s)")
        print()

    # Rejected slug detail
    if rejected_list and not args.verbose:
        _section("Rejected Candidates (summary)")
        for r in sorted(rejected_list, key=lambda x: x.raw_ev, reverse=True):
            cap_str = f"${r.quote_capital_usd:.2f}" if r.quote_capital_usd is not None else "N/A"
            roc_str = f"{r.ev_roc * 100:.2f}%" if r.ev_roc is not None else "N/A"
            mid_str = f"{r.midpoint:.3f}" if r.midpoint is not None else "N/A"
            codes   = ",".join(r.rejection_codes)
            print(
                f"  {r.market_slug[:52]:<52}"
                f"  raw_ev={r.raw_ev:.6f}  cap={cap_str}  roc={roc_str}  mid={mid_str}"
                f"  → {codes}"
            )
        print()

    # -----------------------------------------------------------------------
    # Final judgment
    # -----------------------------------------------------------------------
    n_survivors = len(survivors)
    _sep("=")
    print("  EXECUTABLE CONVERSION JUDGMENT")
    _sep("=")
    print(f"  persistent_raw_positive_count        : {summary['persistent_raw_positive_count']}")
    print(f"  persistent_executable_positive_count : {summary['persistent_executable_positive_count']}")
    print()

    if n_survivors > 0:
        print(f"  CONTINUE — {n_survivors} executable-positive survivor(s).")
        print("  Survivors passed all 4 execution gates.")
        print("  Eligible for next layer: reward-rate persistence + queue-depth study.")
        print("  Do NOT claim profitability. Do NOT submit orders.")
        print()
        print("  Top survivors:")
        for r in survivors:
            roc_str = f"{r.ev_roc * 100:.2f}%" if r.ev_roc is not None else "N/A"
            cap_str = f"${r.quote_capital_usd:.2f}" if r.quote_capital_usd is not None else "N/A"
            print(
                f"    {r.market_slug[:55]}"
                f"  ev={r.raw_ev:.6f}  roc={roc_str}  capital={cap_str}"
                f"  rate=${r.reward_rate_daily_usdc:.1f}/d"
            )
    else:
        print("  PARK — 0 executable-positive survivors.")
        print("  All raw candidates fail at least one execution constraint.")
        print("  Do not escalate reward_aware_single_market_maker_executable_conversion_line.")
        print()
        print("  Review rejection_code_counts above.")
        print("  If a single gate dominates, recalibrate that gate threshold first.")
        print("  Do NOT lower multiple gates simultaneously without new evidence.")

    print()
    _sep("=")
    print()

    # -----------------------------------------------------------------------
    # Write JSON output
    # -----------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        AUDIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = AUDIT_DATA_DIR / f"audit_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "audit_timestamp": datetime.now(timezone.utc).isoformat(),
                "audit_gates": {
                    "E1_CAPITAL_THRESHOLD_USD": CAPITAL_THRESHOLD_USD,
                    "E2_NEAR_RESOLUTION_LOW": NEAR_RESOLUTION_LOW,
                    "E2_NEAR_RESOLUTION_HIGH": NEAR_RESOLUTION_HIGH,
                    "E3_EV_ROC_MIN": EV_ROC_MIN,
                    "E4_REWARD_FLOOR_USDC": REWARD_FLOOR_USDC,
                },
                "summary": summary,
                "per_slug": [
                    r.to_dict()
                    for r in sorted(audit_results, key=lambda x: x.raw_ev, reverse=True)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Results written : {out_path}")
    print()


if __name__ == "__main__":
    main()
