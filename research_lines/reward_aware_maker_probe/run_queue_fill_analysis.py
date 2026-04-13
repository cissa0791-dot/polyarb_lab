"""
reward_aware_maker_queue_sizing_fillrate_line — CLI
polyarb_lab / research_line / analysis-only

Queue-sizing, fill-rate, and stale-snapshot analysis for the 4 confirmed
executable-positive survivors.

Approach:
  1. Run one full discovery + EV + audit pass to get current market state
  2. Fetch 3 rapid book rounds for the 4 survivors only (12 extra CLOB calls,
     15s apart) to check book freshness and get stable queue estimates
  3. Build queue-fill model for each survivor
  4. Output:
       - Queue state table
       - Freshness / stale-snapshot check
       - EV decomposition: original model view vs counterfactual (side-by-side)
       - Size sensitivity (two views per size point)
       - Per-survivor counterfactual caveats
       - Final judgment: COUNTERFACTUAL SUPPORTED / NOT SUPPORTED / STALE / UNVERIFIABLE

Usage (Windows PowerShell from repo root):
    py -3 research_lines/reward_aware_maker_probe/run_queue_fill_analysis.py
    py -3 research_lines/reward_aware_maker_probe/run_queue_fill_analysis.py --rounds 3 --round-delay 15

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/reward_aware_maker_probe/queue_fill/ only.
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
    _find_market,
    fetch_book_snapshot,
)
from research_lines.reward_aware_maker_probe.modules.queue_fill_model import (
    VOLUME_AT_BEST_FRACTION,
    SIZE_LADDER,
    QueueFillAnalysis,
    build_queue_fill_analysis,
    rank_survivors,
)

# The 4 confirmed survivors
DEFAULT_TARGET_SLUGS = [
    "netanyahu-out-by-june-30-383-244-575",
    "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-n",
]

CLOB_HOST = "https://clob.polymarket.com"
QUEUE_FILL_DATA_DIR = Path("data/research/reward_aware_maker_probe/queue_fill")


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


def _na(val, fmt: str = ".4f") -> str:
    if val is None:
        return "N/A"
    return format(val, fmt)


def _print_queue_table(analyses: list[QueueFillAnalysis]) -> None:
    print(
        f"  {'Slug':<46}  {'mid':>5}  {'qAheadB':>7}  {'qAheadA':>7}"
        f"  {'bidL':>4}  {'vol24h':>8}  {'volLvl':>8}  {'p_fill':>6}  {'waitH':>6}"
    )
    print(
        f"  {'-'*46}  {'-'*5}  {'-'*7}  {'-'*7}"
        f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}"
    )
    for a in analyses:
        print(
            f"  {a.slug[:46]:<46}"
            f"  {_na(a.midpoint, '.3f'):>5}"
            f"  {_na(a.queue_ahead_bid, '.1f'):>7}"
            f"  {_na(a.queue_ahead_ask, '.1f'):>7}"
            f"  {a.bid_levels:>4}"
            f"  {_na(a.volume_24hr, '.0f'):>8}"
            f"  {_na(a.vol_at_bid_level, '.0f'):>8}"
            f"  {_na(a.p_fill_min_size, '.4f'):>6}"
            f"  {_na(a.fill_wait_hours, '.1f'):>6}"
        )


def _print_ev_decomp_table(analyses: list[QueueFillAnalysis]) -> None:
    """
    Side-by-side EV table: original model view vs counterfactual view.

    original_model_ev  — source-of-truth from ev_model.py
    cf_reward_only_ev  — COUNTERFACTUAL/DIAGNOSTIC: reward_contribution alone
    cf_delta           — cf_reward_only_ev - original_model_ev (how much the
                         counterfactual adds above the model; positive means
                         model penalises more than counterfactual assumes)
    """
    print(
        f"  {'Slug':<46}  {'reward$':>7}  {'fillNet':>8}  {'orig_ev':>8}"
        f"  {'cf_ev':>7}  {'cf_delta':>8}  {'p_fill_mdl':>10}  rec_size"
    )
    print(
        f"  {'-'*46}  {'-'*7}  {'-'*8}  {'-'*8}"
        f"  {'-'*7}  {'-'*8}  {'-'*10}  --------"
    )
    for a in analyses:
        fill_net_str = _na(a.model_fill_net_at_min_size, '.4f')
        print(
            f"  {a.slug[:46]:<46}"
            f"  {a.reward_contribution:>7.4f}"
            f"  {fill_net_str:>8}"
            f"  {a.original_model_ev:>8.4f}"
            f"  {a.cf_reward_only_ev:>7.4f}"
            f"  {a.cf_delta_vs_model:>+8.4f}"
            f"  {a.model_fill_prob_assumed:>10.3f}"
            f"  {a.recommended_quote_size:.0f}sh"
        )


def _print_size_sensitivity(a: QueueFillAnalysis) -> None:
    print(f"\n  {a.slug[:60]}")
    print(
        f"    {'size':>5}  {'p_fill':>7}  {'exp_fill/d':>10}  {'wait_h':>7}"
        f"  {'model_fill_net':>14}  {'orig_model_ev':>13}  {'cf_ev':>7}"
    )
    print(
        f"    {'-'*5}  {'-'*7}  {'-'*10}  {'-'*7}"
        f"  {'-'*14}  {'-'*13}  {'-'*7}"
    )
    for p in a.size_sensitivity:
        if p.quote_size < a.min_quote_size:
            marker = "(sub-min)"
        elif p.quote_size == a.min_quote_size:
            marker = "<- MIN"
        else:
            marker = ""
        print(
            f"    {p.quote_size:>5.0f}"
            f"  {_na(p.p_fill, '.4f'):>7}"
            f"  {_na(p.expected_fill_shares_daily, '.2f'):>10}"
            f"  {_na(p.fill_wait_hours, '.1f'):>7}"
            f"  {_na(p.model_fill_net, '.4f'):>14}"
            f"  {_na(p.original_model_ev, '.4f'):>13}"
            f"  {p.cf_reward_only_ev:>7.4f}  {marker}"
        )


def _print_freshness_table(analyses: list[QueueFillAnalysis]) -> None:
    print(
        f"  {'Slug':<46}  {'Rounds':>6}  {'Verdict':<16}  Notes"
    )
    print(
        f"  {'-'*46}  {'-'*6}  {'-'*16}  -----"
    )
    for a in analyses:
        tag = "*** STALE ***" if a.freshness_verdict == "STALE_SUSPECT" else a.freshness_verdict
        notes_short = a.freshness_notes[:55] if a.freshness_notes else ""
        print(
            f"  {a.slug[:46]:<46}  {a.n_rounds_checked:>6}  {tag:<16}  {notes_short}"
        )


def _print_side_by_side(analyses: list[QueueFillAnalysis]) -> None:
    """
    Per-survivor side-by-side: original model interpretation vs counterfactual.
    Also shows cf_caveats — per-survivor reasons the counterfactual may be too strong.
    """
    for a in analyses:
        _sep("─", 72)
        print(f"  {a.slug}")
        print(f"  Freshness  : {a.freshness_verdict}  ({a.freshness_notes[:60] if a.freshness_notes else 'N/A'})")
        print()

        # Original model
        print("  ORIGINAL MODEL VIEW (source-of-truth from ev_model.py):")
        print(f"    reward_contribution      = {a.reward_contribution:+.6f}")
        print(f"    model_fill_net_at_min    = {_na(a.model_fill_net_at_min_size, '+.6f')}")
        print(f"    original_model_ev (raw_ev) = {a.original_model_ev:+.6f}")
        print(f"    model_fill_prob_assumed  = {a.model_fill_prob_assumed:.3f}  (ev_model base)")
        print(f"    queue_fill_model p_fill  = {_na(a.p_fill_min_size, '.4f')}  (observed queue depth)")
        print()

        # Counterfactual
        print("  COUNTERFACTUAL / DIAGNOSTIC (NOT source-of-truth):")
        print(f"    cf_reward_only_ev        = {a.cf_reward_only_ev:+.6f}  (reward_contribution only)")
        print(f"    cf_delta_vs_model        = {a.cf_delta_vs_model:+.6f}  (cf - original)")
        if a.cf_delta_vs_model > 0:
            print(f"    Interpretation: model's fill penalties reduce EV by ${a.cf_delta_vs_model:.4f}")
            print(f"    Counterfactual claims these penalties vanish at low fill rate.")
        else:
            print(f"    Counterfactual does NOT raise EV above original model.")
        print()

        # Recommended size
        print(f"  Recommended size: {a.recommended_quote_size:.0f}sh")
        print(f"  Reasoning: {a.recommended_reasoning[:120]}")
        print()

        # Caveats
        print("  Counterfactual caveats (reasons it may be too strong):")
        for i, c in enumerate(a.cf_caveats, 1):
            # Word-wrap at 70 chars
            words = c.split()
            lines: list[str] = []
            current = ""
            for w in words:
                if len(current) + len(w) + 1 > 70:
                    lines.append(current)
                    current = w
                else:
                    current = (current + " " + w).strip()
            if current:
                lines.append(current)
            print(f"    [{i}] {lines[0]}")
            for line in lines[1:]:
                print(f"        {line}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Queue-sizing and fill-rate analysis for 4 executable-positive survivors.\n"
            "Paper-only. No order submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of rapid book re-fetch rounds for stale detection (default: 3)",
    )
    parser.add_argument(
        "--round-delay", type=float, default=15.0,
        help="Seconds between rapid book rounds (default: 15)",
    )
    parser.add_argument(
        "--slugs", nargs="+", default=DEFAULT_TARGET_SLUGS,
        help="Target slugs to analyse",
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
    _section("reward_aware_maker_queue_sizing_fillrate_line")
    print(f"  Timestamp   : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Targets     : {len(args.slugs)} survivors")
    print(f"  Book rounds : {args.rounds}  (delay={args.round_delay}s, for stale detection)")
    print(f"  Vol@best    : {VOLUME_AT_BEST_FRACTION * 100:.0f}% of volume_24hr assumed at best-bid level")
    print(f"  Size ladder : {SIZE_LADDER}")
    print()
    print("  EV FRAMEWORK:")
    print("    original_model_ev  — raw_ev from ev_model.py  [SOURCE-OF-TRUTH]")
    print("    cf_reward_only_ev  — reward_contribution only  [COUNTERFACTUAL/DIAGNOSTIC]")
    print("    cf_delta           — cf - original  (positive = model penalises more than CF)")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discovery + EV + audit (one pass)
    # -----------------------------------------------------------------------
    _section("Step 1: Discovery + EV + Executable Audit")
    raw_markets  = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
    ev_results   = evaluate_batch(raw_markets)
    audit_results = audit_batch(ev_results, raw_markets)

    market_by_slug = {}
    ev_by_slug     = {r.market_slug: r for r in ev_results}

    for slug in args.slugs:
        m = _find_market(slug, raw_markets)
        if m:
            market_by_slug[slug] = m
            print(f"  Found : {slug[:60]}  ->  full_slug={m.market_slug[:55]}")
        else:
            print(f"  MISSING : {slug}  -- not in current universe")
    print()

    if not market_by_slug:
        print("  No target slugs found. Aborting.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Rapid multi-round book fetch (stale detection)
    # -----------------------------------------------------------------------
    _section("Step 2: Rapid Book Re-fetch (Stale Detection)")
    print(f"  Fetching {args.rounds} rounds x {len(market_by_slug)} survivors = {args.rounds * len(market_by_slug)} CLOB calls")
    print()

    # snapshots_by_slug: slug -> [snap_round_0, snap_round_1, ...]
    snapshots_by_slug: dict[str, list] = {slug: [] for slug in market_by_slug}

    with httpx.Client() as client:
        for rnd in range(1, args.rounds + 1):
            ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            print(f"  Round {rnd}/{args.rounds}  {ts}")
            for slug, market in market_by_slug.items():
                reward_max_spread = market.rewards_max_spread_cents / 100.0
                snap = fetch_book_snapshot(
                    host=args.clob_host,
                    token_id=market.yes_token_id,
                    reward_max_spread=reward_max_spread,
                    client=client,
                )
                snapshots_by_slug[slug].append(snap)
                status = "ok" if snap.fetch_ok else "FAIL"
                bid_s  = f"{snap.top_bid_size:.1f}" if snap.top_bid_size is not None else "N/A"
                ask_s  = f"{snap.top_ask_size:.1f}" if snap.top_ask_size is not None else "N/A"
                print(
                    f"    {slug[:40]:<40}  [{status}]"
                    f"  bid={snap.best_bid}  ask={snap.best_ask}"
                    f"  top_bid_sz={bid_s}  top_ask_sz={ask_s}"
                    f"  bidLevels={snap.bid_levels_count}"
                )
            if rnd < args.rounds:
                print(f"  Waiting {args.round_delay}s...")
                time.sleep(args.round_delay)

    print()

    # -----------------------------------------------------------------------
    # Step 3: Build queue-fill analysis
    # -----------------------------------------------------------------------
    _section("Step 3: Queue-Fill Model")
    analyses: list[QueueFillAnalysis] = []

    for slug, market in market_by_slug.items():
        ev_r = ev_by_slug.get(market.market_slug)
        if ev_r is None:
            print(f"  WARNING: no EV result for {slug} -- skipping")
            continue
        snaps = snapshots_by_slug.get(slug, [])
        analysis = build_queue_fill_analysis(
            slug=slug,
            market=market,
            ev_result=ev_r,
            snapshots=snaps,
        )
        analyses.append(analysis)

    ranked = rank_survivors(analyses)

    # -----------------------------------------------------------------------
    # Freshness table
    # -----------------------------------------------------------------------
    _section("Freshness / Stale-Snapshot Check")
    _print_freshness_table(ranked)
    print()
    for a in ranked:
        if a.freshness_notes:
            print(f"  {a.slug[:50]}: {a.freshness_notes}")
    print()

    any_stale = any(a.freshness_verdict == "STALE_SUSPECT" for a in ranked)
    all_stale = all(a.freshness_verdict == "STALE_SUSPECT" for a in ranked)
    if any_stale:
        print(
            "  *** STALE_SUSPECT raised for one or more survivors. ***\n"
            "  If book data is cached, fill probability estimates are unreliable.\n"
            "  Reward signal (from CLOB rewards endpoint) is NOT book-cached and\n"
            "  remains valid, but fill model results should be treated as UNVERIFIABLE."
        )
    else:
        print("  No stale signals detected. Book data shows variation across rapid rounds.")

    # -----------------------------------------------------------------------
    # Queue state table
    # -----------------------------------------------------------------------
    print()
    _section("Queue State  (queue_ahead = shares at best bid/ask before our order)")
    _print_queue_table(ranked)
    print()
    print(
        f"  Columns: mid=midpoint | qAheadB/A=queue ahead bid/ask (shares)"
        f" | bidL=bid level count | vol24h=daily volume"
        f" | volLvl=est. vol at best level ({VOLUME_AT_BEST_FRACTION*100:.0f}% of vol24h)"
        f" | p_fill=fill prob at min_size | waitH=hours to fill"
    )

    # -----------------------------------------------------------------------
    # EV decomposition: side-by-side original vs counterfactual
    # -----------------------------------------------------------------------
    print()
    _section("EV Decomposition  (original model vs counterfactual — side-by-side)")
    _print_ev_decomp_table(ranked)
    print()
    print("  reward$     : estimated daily reward share contribution (constant, size-independent)")
    print("  fillNet     : model's fill term net (spread_capture - adverse_sel - inventory)")
    print("                [negative by current calibration — not a universal truth]")
    print("  orig_ev     : raw_ev from ev_model.py  [SOURCE-OF-TRUTH]")
    print("  cf_ev       : cf_reward_only_ev = reward$ only  [COUNTERFACTUAL/DIAGNOSTIC]")
    print("  cf_delta    : cf_ev - orig_ev  (positive = model's fill drag exceeds CF)")
    print("  p_fill_mdl  : fill_prob assumed by ev_model (0.20 base, 0.30 wide market)")
    print("  rec_size    : recommended quote size")

    # -----------------------------------------------------------------------
    # Per-survivor side-by-side detail + caveats
    # -----------------------------------------------------------------------
    print()
    _section("Per-Survivor: Original Model vs Counterfactual + Caveats")
    _print_side_by_side(ranked)

    # -----------------------------------------------------------------------
    # Size sensitivity per survivor
    # -----------------------------------------------------------------------
    _section("Size Sensitivity Tables  (two views per size point)")
    print("  orig_model_ev = reward$ + model_fill_net  [source-of-truth]")
    print("  cf_ev         = reward$ only              [counterfactual, constant across sizes]")
    print()
    for a in ranked:
        _print_size_sensitivity(a)

    # -----------------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------------
    print()
    _section("Survivor Ranking")
    print(f"  Ranked by: reward_contribution x (1 + p_fill_bonus)")
    print(f"  {'Rank':<5}  {'Slug':<46}  {'reward$':>7}  {'orig_ev':>7}  {'cf_ev':>6}  {'p_fill':>7}  {'stale':>8}  {'score':>7}")
    print(f"  {'-'*5}  {'-'*46}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*7}")
    for i, a in enumerate(ranked, 1):
        stale_tag = "STALE?" if a.freshness_verdict == "STALE_SUSPECT" else "live"
        print(
            f"  #{i:<4}  {a.slug[:46]:<46}"
            f"  {a.reward_contribution:>7.4f}"
            f"  {a.original_model_ev:>7.4f}"
            f"  {a.cf_reward_only_ev:>6.4f}"
            f"  {_na(a.p_fill_min_size, '.4f'):>7}"
            f"  {stale_tag:>8}"
            f"  {a.ranking_score:>7.4f}"
        )
    print()

    # -----------------------------------------------------------------------
    # Final judgment: three-state
    # -----------------------------------------------------------------------
    _sep("=")
    print("  COUNTERFACTUAL HYPOTHESIS JUDGMENT")
    _sep("=")

    n_with_positive_reward = sum(1 for a in ranked if a.reward_contribution > 0)
    n_stale = sum(1 for a in ranked if a.freshness_verdict == "STALE_SUSPECT")
    n_live  = sum(1 for a in ranked if a.freshness_verdict == "LIVE")
    n_cf_above_model = sum(1 for a in ranked if a.cf_delta_vs_model > 0)
    best    = ranked[0] if ranked else None

    print(f"  Survivors with positive reward_contribution : {n_with_positive_reward}/{len(ranked)}")
    print(f"  Survivors with cf_ev > orig_ev (model penalises fills) : {n_cf_above_model}/{len(ranked)}")
    print(f"  Survivors with LIVE book signal             : {n_live}/{len(ranked)}")
    print(f"  Survivors with STALE_SUSPECT book signal    : {n_stale}/{len(ranked)}")
    print()

    # Determine verdict
    if all_stale:
        verdict = "STALE / UNVERIFIABLE"
        print(f"  VERDICT: {verdict}")
        print()
        print("  All book snapshots flagged STALE_SUSPECT across all rapid rounds.")
        print("  Fill probability estimates are unreliable — p_fill and queue_ahead cannot")
        print("  be validated with current data. Reward signal remains valid but the")
        print("  counterfactual (which depends on fill rate being low) is unverifiable.")
        print()
        print("  Recommended action: run extended monitoring (>5 min intervals) to confirm")
        print("  whether book data is truly stale or activity is simply low.")
    elif n_with_positive_reward == 0:
        verdict = "COUNTERFACTUAL NOT SUPPORTED"
        print(f"  VERDICT: {verdict}")
        print()
        print("  reward_contribution has collapsed to zero for all survivors.")
        print("  The counterfactual (reward-only EV) provides no positive signal.")
        print("  PARK reward_aware_maker_queue_sizing_fillrate_line.")
    elif n_cf_above_model > 0 and n_with_positive_reward >= 2 and n_live >= 1:
        verdict = "COUNTERFACTUAL SUPPORTED (DIAGNOSTIC)"
        print(f"  VERDICT: {verdict}")
        print()
        print("  The counterfactual observation is structurally coherent:")
        print(f"    {n_cf_above_model}/{len(ranked)} survivors show cf_ev > orig_ev.")
        print(f"    This means the original model's fill penalties reduce EV below")
        print(f"    reward_contribution alone.")
        print()
        print("  This does NOT confirm that 'realistic EV = reward_contribution'.")
        print("  It confirms: the fill-net term is negative by current calibration,")
        print("  and the counterfactual bound is above the model's conservative estimate.")
        print()
        print("  The counterfactual remains diagnostic because:")
        print("    - ADVERSE_SELECTION_FACTOR and INVENTORY_FACTOR have not been replaced")
        print("    - Rare fills may carry higher conditional adverse selection")
        print("    - REWARD_POOL_SHARE_FRACTION is itself a proxy")
        print()
        print("  Size recommendation: quote at minimum size (rewards_min_size) always.")
        print("    Larger sizes increase model fill drag without confirmed proportional reward.")
        print()
        if best:
            print(f"  Strongest survivor (by reward contribution):")
            print(f"    {best.slug}")
            print(f"    reward_contribution = ${best.reward_contribution:.4f}/day")
            print(f"    original_model_ev   = ${best.original_model_ev:.4f}/day  [source-of-truth]")
            print(f"    cf_reward_only_ev   = ${best.cf_reward_only_ev:.4f}/day  [counterfactual ceiling]")
            print(f"    recommended_size    = {best.recommended_quote_size:.0f} shares")
        print()
        print("  Do NOT claim profitability. Do NOT submit orders.")
        print("  Next: confirm reward pool share fraction with direct rewards endpoint monitoring.")
    elif n_with_positive_reward >= 1:
        verdict = "COUNTERFACTUAL NOT SUPPORTED"
        print(f"  VERDICT: {verdict}")
        print()
        print(f"  {n_with_positive_reward} survivor(s) have positive reward contribution but")
        if n_stale > 0:
            print(f"  {n_stale} have STALE_SUSPECT book data — fill estimates unreliable.")
        if n_cf_above_model == 0:
            print("  cf_ev does not exceed orig_ev for any survivor — fill penalties")
            print("  do not exceed counterfactual correction.")
        print()
        print("  Reward signal exists but counterfactual framing is not clearly supported.")
        print("  Run with fresh book data before escalating.")
    else:
        verdict = "COUNTERFACTUAL NOT SUPPORTED"
        print(f"  VERDICT: {verdict}")
        print("  Insufficient data to evaluate the counterfactual.")

    print()
    _sep("=")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        QUEUE_FILL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = QUEUE_FILL_DATA_DIR / f"queue_fill_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "clob_host": args.clob_host,
                "rounds": args.rounds,
                "round_delay_sec": args.round_delay,
                "volume_at_best_fraction": VOLUME_AT_BEST_FRACTION,
                "target_slugs": args.slugs,
                "counterfactual_verdict": verdict,
                "freshness_summary": {
                    "n_live": n_live,
                    "n_stale": n_stale,
                    "n_single_snapshot": len(ranked) - n_live - n_stale,
                },
                "ranked_analyses": [a.to_dict() for a in ranked],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Results written : {out_path}")
    print()


if __name__ == "__main__":
    main()
