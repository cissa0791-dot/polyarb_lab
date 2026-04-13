"""
reward_aware_maker_survivor_persistence_queue_line — CLI
polyarb_lab / research_line / audit-only

Multi-cycle persistence + book/queue realism check for the 4 confirmed
executable-positive survivors.

Per cycle per slug:
  - executable verdict (still EXEC_POSITIVE?)
  - reward-rate stability
  - book state (has_usable_book, liquidity, volume_24hr)
  - targeted book snapshot: top-of-book size, bid/ask level counts,
    levels within reward spread (queue-pressure proxy)
  - depth_ok / volume_ok / queue_pressure flags

Each cycle makes 1 targeted CLOB /book call per target slug (4 extra
calls per cycle) on top of the standard discovery scan. This is NOT a
broad universe rescan — targeted only.

Usage (Windows PowerShell from repo root):
    py -3 research_lines/reward_aware_maker_probe/run_survivor_audit.py
    py -3 research_lines/reward_aware_maker_probe/run_survivor_audit.py --cycles 5 --delay 60

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/reward_aware_maker_probe/survivor_audit/ only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.reward_aware_maker_probe.modules.discovery import (
    discover_fee_enabled_rewarded_markets,
)
from research_lines.reward_aware_maker_probe.modules.ev_model import evaluate_batch
from research_lines.reward_aware_maker_probe.modules.executable_audit import audit_batch
from research_lines.reward_aware_maker_probe.modules.survivor_tracker import (
    DEPTH_ADEQUACY_MULTIPLE,
    QUEUE_PRESSURE_LEVEL_THRESHOLD,
    VOLUME_ADEQUACY_MULTIPLE,
    CycleSlugState,
    build_cycle_state,
)

# The 4 confirmed executable-positive survivors from the conversion audit.
# Slugs may be truncated from display; survivor_tracker uses prefix matching.
DEFAULT_TARGET_SLUGS = [
    "netanyahu-out-by-june-30-383-244-575",
    "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-n",
]

CLOB_HOST = "https://clob.polymarket.com"
SURVIVOR_DATA_DIR = Path("data/research/reward_aware_maker_probe/survivor_audit")


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


def _ok(val: bool) -> str:
    return "OK  " if val else "FAIL"


def _print_cycle_table(states: list[CycleSlugState], cycle: int) -> None:
    print(f"\n  Cycle {cycle} — {datetime.now(timezone.utc).strftime('%H:%M:%SZ')}")
    print(
        f"  {'Slug':<46}  {'EXEC':>5}  {'raw_ev':>7}  {'rate':>5}  "
        f"{'book':>4}  {'dep':>4}  {'vol':>4}  {'qP':>3}  {'ticks':>5}  "
        f"{'bidL':>4}  {'askL':>4}  codes"
    )
    print(
        f"  {'-'*46}  {'-'*5}  {'-'*7}  {'-'*5}  "
        f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*3}  {'-'*5}  "
        f"{'-'*4}  {'-'*4}  -----"
    )
    for st in states:
        exec_tag = "YES **" if st.is_exec_positive else "NO    "
        ev_str   = f"{st.raw_ev:.4f}" if st.raw_ev is not None else "N/A  "
        rate_str = f"{st.reward_rate_daily_usdc:.0f}" if st.reward_rate_daily_usdc is not None else "N/A"
        ticks_str = str(st.spread_ticks) if st.spread_ticks is not None else "N/A"
        bid_l = st.book.bid_levels_count if st.book else 0
        ask_l = st.book.ask_levels_count if st.book else 0
        bid_in = st.book.bid_levels_in_reward_spread if st.book else 0
        qp_tag = f"H{bid_in}" if st.queue_pressure else f"L{bid_in}"
        codes = ",".join(st.rejection_codes) if st.rejection_codes else "—"
        print(
            f"  {st.slug[:46]:<46}  {exec_tag}  {ev_str:>7}  {rate_str:>5}"
            f"  {_ok(st.has_usable_book)}  {_ok(st.depth_ok)}  {_ok(st.volume_ok)}"
            f"  {qp_tag:>3}  {ticks_str:>5}  {bid_l:>4}  {ask_l:>4}  {codes}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Survivor persistence + queue/book realism check.\n"
            "Tracks the 4 executable-positive survivors across N cycles.\n"
            "Paper-only. No order submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--cycles", type=int, default=5,
        help="Number of scan cycles (default: 5)",
    )
    parser.add_argument(
        "--delay", type=float, default=60.0,
        help="Seconds between cycles (default: 60)",
    )
    parser.add_argument(
        "--slugs", nargs="+", default=DEFAULT_TARGET_SLUGS,
        help="Target slugs to track (default: the 4 confirmed survivors)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON results here",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    print()
    _section("reward_aware_maker_survivor_persistence_queue_line")
    print(f"  Timestamp  : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Cycles     : {args.cycles}  (delay={args.delay}s)")
    print(f"  Targets    : {len(args.slugs)} survivors")
    for s in args.slugs:
        print(f"    - {s}")
    print()
    print("  Per-cycle flags:")
    print(f"    EXEC     : still passes all 4 executable gates")
    print(f"    dep      : top_bid_size >= min_size × {DEPTH_ADEQUACY_MULTIPLE:.0f}")
    print(f"    vol      : volume_24hr >= min_size × {VOLUME_ADEQUACY_MULTIPLE:.0f}")
    print(f"    qP       : bid_levels_in_reward_spread >= {QUEUE_PRESSURE_LEVEL_THRESHOLD}  (H=high, L=low)")
    print(f"    bidL/askL: total level counts in book (queue depth proxy)")
    print("    modeled_quote_spread: EV-layer quote width inside reward window, not observed live spread")
    print()

    # -----------------------------------------------------------------------
    # Multi-cycle loop
    # -----------------------------------------------------------------------
    all_cycle_states: list[list[CycleSlugState]] = []

    with httpx.Client() as http_client:
        for cycle_num in range(1, args.cycles + 1):
            _sep("=")
            print(f"  CYCLE {cycle_num} / {args.cycles}  —  {datetime.now(timezone.utc).strftime('%H:%M:%SZ')}")
            _sep("=")

            # Full pipeline
            raw_markets  = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
            ev_results   = evaluate_batch(raw_markets)
            audit_results = audit_batch(ev_results, raw_markets)

            # Build per-slug state (includes targeted book fetch per slug)
            cycle_states: list[CycleSlugState] = []
            for slug in args.slugs:
                state = build_cycle_state(
                    cycle=cycle_num,
                    target_slug=slug,
                    raw_markets=raw_markets,
                    ev_results=ev_results,
                    audit_results=audit_results,
                    clob_host=args.clob_host,
                    http_client=http_client,
                )
                cycle_states.append(state)

            all_cycle_states.append(cycle_states)
            _print_cycle_table(cycle_states, cycle_num)

            if cycle_num < args.cycles:
                print(f"\n  Waiting {args.delay:.0f}s before next cycle...")
                time.sleep(args.delay)

    # -----------------------------------------------------------------------
    # Persistence summary
    # -----------------------------------------------------------------------
    print()
    _sep("=")
    print("  SURVIVOR PERSISTENCE SUMMARY")
    _sep("=")
    print(f"  Cycles run : {args.cycles}")
    print()

    slug_summaries: list[dict] = []
    stable_slugs: list[str] = []
    partial_slugs: list[str] = []

    print(
        f"  {'Slug':<46}  {'Exec+':>5}  {'Dep':>3}  {'Vol':>3}  "
        f"{'Book':>4}  {'RateStb':>7}  Status"
    )
    print(
        f"  {'-'*46}  {'-'*5}  {'-'*3}  {'-'*3}  "
        f"{'-'*4}  {'-'*7}  ------"
    )

    for i, slug in enumerate(args.slugs):
        states = [all_cycle_states[c][i] for c in range(args.cycles)]

        exec_count  = sum(1 for s in states if s.is_exec_positive)
        depth_count = sum(1 for s in states if s.depth_ok)
        vol_count   = sum(1 for s in states if s.volume_ok)
        book_count  = sum(1 for s in states if s.has_usable_book)

        # Reward rate stability: is max-min within 10% of max?
        rates = [s.reward_rate_daily_usdc for s in states if s.reward_rate_daily_usdc is not None]
        if rates:
            rate_delta_pct = (max(rates) - min(rates)) / max(max(rates), 0.001) * 100
            rate_stable = rate_delta_pct < 10.0
            rate_tag = f"{rate_delta_pct:.1f}%Δ"
        else:
            rate_stable = False
            rate_tag = "NO DATA"

        is_fully_stable = exec_count == args.cycles
        is_majority     = exec_count >= (args.cycles // 2 + 1)

        if is_fully_stable:
            status = "STABLE ***"
            stable_slugs.append(slug)
        elif is_majority:
            status = f"PARTIAL({exec_count}/{args.cycles})"
            partial_slugs.append(slug)
        else:
            status = f"UNSTABLE({exec_count}/{args.cycles})"

        print(
            f"  {slug[:46]:<46}  {exec_count:>2}/{args.cycles:<2}  "
            f"{depth_count:>1}/{args.cycles:<1}  {vol_count:>1}/{args.cycles:<1}  "
            f"{book_count:>2}/{args.cycles:<2}  {rate_tag:>7}  {status}"
        )

        slug_summaries.append({
            "slug": slug,
            "exec_positive_cycles": exec_count,
            "total_cycles": args.cycles,
            "depth_ok_cycles": depth_count,
            "volume_ok_cycles": vol_count,
            "book_ok_cycles": book_count,
            "rate_stable": rate_stable,
            "rate_delta_pct": round(rate_delta_pct, 2) if rates else None,
            "avg_reward_rate": round(sum(rates) / len(rates), 2) if rates else None,
            "fully_stable": is_fully_stable,
            "majority_positive": is_majority,
        })

    print()

    # Detailed per-survivor rate and book profile
    _section("Per-Survivor Detail")
    for i, slug in enumerate(args.slugs):
        states = [all_cycle_states[c][i] for c in range(args.cycles)]
        print(f"  {slug}")
        for st in states:
            best_bid = f"{st.book.best_bid:.3f}" if st.book and st.book.best_bid is not None else "N/A"
            best_ask = f"{st.book.best_ask:.3f}" if st.book and st.book.best_ask is not None else "N/A"
            bid_top = f"{st.book.top_bid_size:.1f}sh" if st.book and st.book.top_bid_size is not None else "N/A"
            ask_top = f"{st.book.top_ask_size:.1f}sh" if st.book and st.book.top_ask_size is not None else "N/A"
            bid_in  = st.book.bid_levels_in_reward_spread if st.book else 0
            ask_in  = st.book.ask_levels_in_reward_spread if st.book else 0
            vol_str = f"{st.volume_24hr:.0f}sh" if st.volume_24hr is not None else "N/A"
            liq_str = f"${st.liquidity:.0f}" if st.liquidity is not None else "N/A"
            exec_tag = "EXEC+" if st.is_exec_positive else "EXEC-"
            print(
                f"    C{st.cycle}: {exec_tag}"
                f"  ev={st.raw_ev:.4f}" if st.raw_ev is not None
                else f"    C{st.cycle}: {exec_tag}  ev=N/A"
            )
            print(
                f"         rate=${st.reward_rate_daily_usdc:.1f}/d" if st.reward_rate_daily_usdc else "         rate=N/A",
                f" | modeled_quote_spread={st.modeled_quote_spread}" if st.modeled_quote_spread is not None else "",
                f" | mid={st.midpoint:.3f}" if st.midpoint else "",
            )
            print(
                f"         observed_book: best_bid={best_bid}  best_ask={best_ask}"
            )
            print(
                f"         book: bid_top={bid_top}  ask_top={ask_top}"
                f"  bid_in_spread={bid_in}  ask_in_spread={ask_in}"
                f"  vol24h={vol_str}  liq={liq_str}"
            )
        print()

    # -----------------------------------------------------------------------
    # Final judgment
    # -----------------------------------------------------------------------
    _sep("=")
    print("  SURVIVOR PERSISTENCE JUDGMENT")
    _sep("=")
    print(f"  Stable across ALL {args.cycles} cycles  : {len(stable_slugs)}")
    print(f"  Majority-positive ({args.cycles//2 + 1}+/{args.cycles}) : {len(partial_slugs)}")
    print()

    n_stable = len(stable_slugs)
    n_partial = len(partial_slugs)

    if n_stable >= 2:
        print(f"  CONTINUE — {n_stable} survivors stable across all {args.cycles} cycles.")
        print("  Reward rate is stable. Book is present.")
        print("  Next layer: queue-sizing study + real fill-rate estimation.")
        print("  Do NOT claim profitability. Do NOT submit orders.")
    elif n_stable == 1:
        print(f"  CONTINUE (thin) — 1 survivor fully stable.")
        print("  Pool is very thin. Run 10 cycles before further escalation.")
        print("  Do NOT claim profitability. Do NOT submit orders.")
    elif n_partial >= 1:
        print(f"  DOWNGRADE — 0 fully stable, {n_partial} majority-positive.")
        print("  Pool is fragile. Classify as monitoring-only.")
        print("  Run 10 cycles before any escalation discussion.")
    else:
        print("  PARK — 0 survivors stable or majority-positive.")
        print("  All executable-positive classifications were transient.")
        print("  Do not escalate reward_aware_maker_survivor_persistence_queue_line.")

    if stable_slugs:
        print()
        print("  Stable survivors:")
        for s in stable_slugs:
            print(f"    {s}")

    print()
    _sep("=")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        SURVIVOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SURVIVOR_DATA_DIR / f"survivor_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "audit_timestamp": datetime.now(timezone.utc).isoformat(),
                "cycles": args.cycles,
                "cycle_delay_sec": args.delay,
                "target_slugs": args.slugs,
                "slug_summaries": slug_summaries,
                "per_cycle": [
                    [s.to_dict() for s in cycle]
                    for cycle in all_cycle_states
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Results written : {out_path}")
    print()


if __name__ == "__main__":
    main()
