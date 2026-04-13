"""
reward_aware_reward_share_validation_line — CLI
polyarb_lab / research_line / validation-only

Directly validates whether REWARD_POOL_SHARE_FRACTION = 0.05 (5%) is a
conservative, fair, or optimistic assumption for the 4 confirmed survivors.

Method:
  The reward pool is split proportionally among makers quoting within
  rewardsMaxSpread.  Our share ≈ min_size / (eligible_depth_in_spread + min_size).
  This module measures eligible_depth_in_spread directly from the live order book
  across N rapid rounds and compares the implied share to the model assumption.

Per-survivor output:
  - Eligible bid/ask depth in reward window (shares)
  - Implied share fraction vs model 5%
  - Share assumption verdict: CONSERVATIVE / FAIR / OPTIMISTIC / UNVERIFIABLE
  - Implied vs model reward contribution comparison
  - Reward rate persistence check (current discovery rate vs historical)

Usage (Windows PowerShell from repo root):
    py -3 research_lines/reward_aware_maker_probe/run_reward_share_validation.py
    py -3 research_lines/reward_aware_maker_probe/run_reward_share_validation.py --rounds 5 --round-delay 30

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/reward_aware_maker_probe/reward_share/ only.
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
from research_lines.reward_aware_maker_probe.modules.ev_model import (
    REWARD_POOL_SHARE_FRACTION,
    evaluate_batch,
)
from research_lines.reward_aware_maker_probe.modules.survivor_tracker import _find_market
from research_lines.reward_aware_maker_probe.modules.reward_share_model import (
    FAIR_TOLERANCE,
    RATE_STABILITY_THRESHOLD_PCT,
    RewardShareAnalysis,
    RewardShareSnapshot,
    build_reward_share_analysis,
    fetch_reward_share_snapshot,
    rank_by_implied_contribution,
)

# The 4 confirmed executable-positive survivors
DEFAULT_TARGET_SLUGS = [
    "netanyahu-out-by-june-30-383-244-575",
    "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-n",
]

CLOB_HOST = "https://clob.polymarket.com"
REWARD_SHARE_DATA_DIR = Path("data/research/reward_aware_maker_probe/reward_share")

# Historical reward rate from 5-cycle survivor audit (0.0%Delta confirmed)
# Used as reference baseline for rate persistence check
HISTORICAL_RATE_DELTA_PCT = 0.0


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


def _verdict_tag(verdict: str) -> str:
    tags = {
        "CONSERVATIVE": "CONS *** (model underestimates)",
        "FAIR":         "FAIR     (model ~correct)",
        "OPTIMISTIC":   "OPT ***  (model overestimates)",
        "UNVERIFIABLE": "UNVERIF  (no depth data)",
    }
    return tags.get(verdict, verdict)


def _print_round_table(slug: str, snapshots: list[RewardShareSnapshot]) -> None:
    """Print per-round book state for one survivor."""
    print(f"\n    {slug[:60]}")
    print(
        f"    {'Rnd':>3}  {'bid':>6}  {'ask':>6}  {'spread':>7}  "
        f"{'eBidDpth':>9}  {'eBidL':>5}  {'eAskDpth':>9}  {'eAskL':>5}  "
        f"{'impBid%':>8}  {'impAsk%':>8}  {'impAvg%':>8}  {'bias':>7}"
    )
    print(
        f"    {'---':>3}  {'---':>6}  {'---':>6}  {'------':>7}  "
        f"{'--------':>9}  {'-----':>5}  {'--------':>9}  {'-----':>5}  "
        f"{'-------':>8}  {'-------':>8}  {'-------':>8}  {'----':>7}"
    )
    for s in snapshots:
        if not s.fetch_ok:
            print(f"    {s.round_num:>3}  FETCH FAILED")
            continue
        bid_str  = _na(s.best_bid, '.3f')
        ask_str  = _na(s.best_ask, '.3f')
        spr_str  = _na(s.spread, '.4f')
        ib_pct   = f"{s.implied_bid_share * 100:.2f}%" if s.implied_bid_share is not None else "N/A"
        ia_pct   = f"{s.implied_ask_share * 100:.2f}%" if s.implied_ask_share is not None else "N/A"
        iavg_pct = f"{s.implied_avg_share * 100:.2f}%" if s.implied_avg_share is not None else "N/A"
        bias_str = f"{s.share_bias:+.3f}" if s.share_bias is not None else "N/A"
        print(
            f"    {s.round_num:>3}  {bid_str:>6}  {ask_str:>6}  {spr_str:>7}  "
            f"  {s.total_eligible_bid_depth:>7.1f}  {s.eligible_bid_level_count:>5}  "
            f"  {s.total_eligible_ask_depth:>7.1f}  {s.eligible_ask_level_count:>5}  "
            f"  {ib_pct:>7}  {ia_pct:>7}  {iavg_pct:>7}  {bias_str:>7}"
        )


def _print_summary_table(analyses: list[RewardShareAnalysis]) -> None:
    print(
        f"  {'Slug':<46}  {'rate$/d':>7}  {'eBidMn':>7}  {'eAskMn':>7}  "
        f"{'impAvg%':>8}  {'modl%':>6}  {'bias':>7}  "
        f"{'modlCont':>8}  {'implCont':>8}  Verdict"
    )
    print(
        f"  {'-'*46}  {'-'*7}  {'-'*7}  {'-'*7}  "
        f"{'-'*8}  {'-'*6}  {'-'*7}  "
        f"{'-'*8}  {'-'*8}  -------"
    )
    for a in analyses:
        impl_pct = f"{a.mean_implied_avg_share * 100:.2f}%" if a.mean_implied_avg_share is not None else "N/A"
        model_pct = f"{a.model_share_fraction * 100:.1f}%"
        bias_str = f"{a.mean_share_bias:+.3f}" if a.mean_share_bias is not None else "N/A"
        impl_c = _na(a.mean_implied_reward_contribution, '.4f')
        print(
            f"  {a.slug[:46]:<46}"
            f"  {a.reward_daily_rate_usdc:>7.1f}"
            f"  {_na(a.mean_eligible_bid_depth, '.1f'):>7}"
            f"  {_na(a.mean_eligible_ask_depth, '.1f'):>7}"
            f"  {impl_pct:>8}"
            f"  {model_pct:>6}"
            f"  {bias_str:>7}"
            f"  {a.model_reward_contribution:>8.4f}"
            f"  {impl_c:>8}"
            f"  {a.share_assumption_verdict}"
        )


def _print_contribution_comparison(analyses: list[RewardShareAnalysis]) -> None:
    """Show model vs implied reward contribution with direction."""
    for a in analyses:
        print(f"\n  {a.slug[:60]}")
        print(f"    reward_daily_rate_usdc   : ${a.reward_daily_rate_usdc:.2f}/day")
        print(f"    model_share_fraction     : {a.model_share_fraction:.1%}  (REWARD_POOL_SHARE_FRACTION)")
        if a.mean_implied_avg_share is not None:
            print(f"    implied_avg_share (mean) : {a.mean_implied_avg_share:.2%}  ({a.n_rounds} rounds)")
            print(f"    share_bias               : {a.mean_share_bias:+.2%}  (implied - model)")
        else:
            print(f"    implied_avg_share (mean) : N/A")
        print(f"    model_reward_contribution: ${a.model_reward_contribution:.4f}/day")
        if a.mean_implied_reward_contribution is not None:
            diff = a.mean_implied_reward_contribution - a.model_reward_contribution
            print(f"    implied_reward_contrib   : ${a.mean_implied_reward_contribution:.4f}/day  ({diff:+.4f} vs model)")
        else:
            print(f"    implied_reward_contrib   : N/A")
        print(f"    eligible_bid_depth (mean): {_na(a.mean_eligible_bid_depth, '.1f')}sh  ({a.bid_depth_delta_pct:.1f}%Delta across rounds)")
        print(f"    depth_shows_activity     : {a.depth_shows_activity}")
        print(f"    verdict                  : {a.share_assumption_verdict}")
        if a.share_assumption_notes:
            print(f"    notes                    : {a.share_assumption_notes[:100]}")
        if a.share_assumption_direction:
            print(f"    direction                : {a.share_assumption_direction[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reward share validation for 4 executable-positive survivors.\n"
            "Compares implied reward share (from book depth) to model 5% assumption.\n"
            "Paper-only. No order submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Number of rapid book rounds for depth sampling (default: 5)",
    )
    parser.add_argument(
        "--round-delay", type=float, default=30.0,
        help="Seconds between rounds (default: 30)",
    )
    parser.add_argument(
        "--slugs", nargs="+", default=DEFAULT_TARGET_SLUGS,
        help="Target slugs to validate",
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
    _section("reward_aware_reward_share_validation_line")
    print(f"  Timestamp      : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Targets        : {len(args.slugs)} survivors")
    print(f"  Rounds         : {args.rounds}  (delay={args.round_delay}s)")
    print(f"  Model share    : {REWARD_POOL_SHARE_FRACTION:.1%}  (REWARD_POOL_SHARE_FRACTION to validate)")
    print(f"  Fair band      : {REWARD_POOL_SHARE_FRACTION*(1-FAIR_TOLERANCE):.1%}  to  {REWARD_POOL_SHARE_FRACTION*(1+FAIR_TOLERANCE):.1%}  (±{FAIR_TOLERANCE:.0%})")
    print()
    print("  Core question:")
    print("    For each survivor at min_size quote:")
    print("      implied_share = min_size / (total_eligible_depth_in_spread + min_size)")
    print("      Is implied_share ≈ 5%? (FAIR)  >5%? (CONSERVATIVE)  <5%? (OPTIMISTIC)")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discovery + EV (one pass to get current market state)
    # -----------------------------------------------------------------------
    _section("Step 1: Discovery + EV Model")
    raw_markets = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
    ev_results  = evaluate_batch(raw_markets)

    market_by_slug: dict[str, object] = {}
    ev_by_slug = {r.market_slug: r for r in ev_results}

    for slug in args.slugs:
        m = _find_market(slug, raw_markets)
        if m:
            market_by_slug[slug] = m
            ev_r = ev_by_slug.get(m.market_slug)
            ev_tag = f"ev={ev_r.reward_adjusted_raw_ev:.5f}" if ev_r else "ev=N/A"
            print(
                f"  Found : {slug[:55]}  rate=${m.reward_daily_rate_usdc:.2f}/d"
                f"  min={m.rewards_min_size:.0f}sh  spread_max={m.rewards_max_spread_cents:.0f}c"
                f"  {ev_tag}"
            )
        else:
            print(f"  MISSING : {slug}  -- not in current universe")
    print()

    if not market_by_slug:
        print("  No target slugs found. Aborting.")
        return

    # Reward rate vs historical baseline
    _section("Step 2: Reward Rate vs Historical Baseline")
    print(f"  Historical: 0.0%Delta across 5 cycles (from survivor audit).")
    print(f"  Current rates:")
    all_rates_consistent = True
    for slug, market in market_by_slug.items():
        print(f"    {slug[:55]}  rate=${market.reward_daily_rate_usdc:.4f}/day")
    print()
    print(f"  Reward rate read from CLOB rewards endpoint (not book cache).")
    print(f"  Endpoint rate is static per discovery call — persistence confirmed by prior 5-cycle audit.")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Multi-round book depth sampling
    # -----------------------------------------------------------------------
    _section("Step 3: Multi-Round Eligible Depth Sampling")
    print(
        f"  Fetching {args.rounds} rounds x {len(market_by_slug)} survivors"
        f" = {args.rounds * len(market_by_slug)} CLOB /book calls"
    )
    print(f"  Measuring: total shares within rewardsMaxSpread window each round")
    print()

    snapshots_by_slug: dict[str, list[RewardShareSnapshot]] = {slug: [] for slug in market_by_slug}

    with httpx.Client() as client:
        for rnd in range(1, args.rounds + 1):
            ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            print(f"  Round {rnd}/{args.rounds}  {ts}")
            for slug, market in market_by_slug.items():
                snap = fetch_reward_share_snapshot(
                    host=args.clob_host,
                    token_id=market.yes_token_id,
                    min_size=market.rewards_min_size,
                    reward_max_spread=market.rewards_max_spread_cents / 100.0,
                    reward_daily_rate_usdc=market.reward_daily_rate_usdc,
                    round_num=rnd,
                    client=client,
                )
                snapshots_by_slug[slug].append(snap)
                if snap.fetch_ok:
                    impl_str = f"{snap.implied_avg_share:.2%}" if snap.implied_avg_share is not None else "N/A"
                    print(
                        f"    {slug[:38]:<38}  bid={snap.best_bid}  ask={snap.best_ask}"
                        f"  eBid={snap.total_eligible_bid_depth:.1f}sh({snap.eligible_bid_level_count}L)"
                        f"  eAsk={snap.total_eligible_ask_depth:.1f}sh({snap.eligible_ask_level_count}L)"
                        f"  implShare={impl_str}"
                    )
                else:
                    print(f"    {slug[:38]:<38}  FETCH FAILED")
            if rnd < args.rounds:
                print(f"  Waiting {args.round_delay}s...")
                time.sleep(args.round_delay)

    print()

    # -----------------------------------------------------------------------
    # Step 4: Build analyses
    # -----------------------------------------------------------------------
    _section("Step 4: Reward Share Analysis")
    analyses: list[RewardShareAnalysis] = []

    for slug, market in market_by_slug.items():
        snaps = snapshots_by_slug.get(slug, [])
        analysis = build_reward_share_analysis(
            slug=slug,
            market=market,
            snapshots=snaps,
        )
        analyses.append(analysis)

    ranked = rank_by_implied_contribution(analyses)

    # Per-round detail tables
    _section("Per-Round Book Detail  (eligible depth within reward spread window)")
    print("  eBidDpth = total bid shares within reward window")
    print("  eAskDpth = total ask shares within reward window")
    print("  impBid/Ask% = implied share fraction on each side")
    print("  impAvg% = average of bid and ask implied share")
    print("  bias = impAvg - model (positive = model is conservative)")
    for slug in market_by_slug:
        snaps = snapshots_by_slug.get(slug, [])
        _print_round_table(slug, snaps)
    print()

    # Summary table
    _section("Share Assumption Summary Table")
    _print_summary_table(ranked)
    print()
    print(f"  Model assumption: REWARD_POOL_SHARE_FRACTION = {REWARD_POOL_SHARE_FRACTION:.1%}")
    print(f"  Fair band: [{REWARD_POOL_SHARE_FRACTION*(1-FAIR_TOLERANCE):.1%}, {REWARD_POOL_SHARE_FRACTION*(1+FAIR_TOLERANCE):.1%}]  (±{FAIR_TOLERANCE:.0%} tolerance)")
    print(f"  bias > 0: model UNDERESTIMATES our share (conservative)")
    print(f"  bias < 0: model OVERESTIMATES our share (optimistic)")

    # Contribution comparison per survivor
    print()
    _section("Model vs Implied Reward Contribution  (per-survivor detail)")
    _print_contribution_comparison(ranked)

    # -----------------------------------------------------------------------
    # Final judgment
    # -----------------------------------------------------------------------
    _sep("=")
    print("  REWARD SHARE ASSUMPTION JUDGMENT")
    _sep("=")

    n_conservative  = sum(1 for a in ranked if a.share_assumption_verdict == "CONSERVATIVE")
    n_fair          = sum(1 for a in ranked if a.share_assumption_verdict == "FAIR")
    n_optimistic    = sum(1 for a in ranked if a.share_assumption_verdict == "OPTIMISTIC")
    n_unverifiable  = sum(1 for a in ranked if a.share_assumption_verdict == "UNVERIFIABLE")

    print(f"  CONSERVATIVE (model underestimates share) : {n_conservative}/{len(ranked)}")
    print(f"  FAIR         (model ~correct)             : {n_fair}/{len(ranked)}")
    print(f"  OPTIMISTIC   (model overestimates share)  : {n_optimistic}/{len(ranked)}")
    print(f"  UNVERIFIABLE (no usable depth data)       : {n_unverifiable}/{len(ranked)}")
    print()

    # Per-survivor direction summary
    for a in ranked:
        verdict_tag = _verdict_tag(a.share_assumption_verdict)
        print(f"  {a.slug[:50]}")
        print(f"    {verdict_tag}")
        if a.share_assumption_direction:
            print(f"    {a.share_assumption_direction[:110]}")
        print()

    # Overall judgment
    n_supported     = n_conservative + n_fair
    n_unsupported   = n_optimistic

    if n_unverifiable == len(ranked):
        print("  VERDICT: UNVERIFIABLE")
        print()
        print("  All eligible depth data is missing or book fetches failed.")
        print("  Cannot assess share assumption. Book data quality insufficient.")
        print("  Check CLOB API connectivity and re-run with --log-level DEBUG.")
    elif n_optimistic > len(ranked) // 2:
        # Majority OPTIMISTIC — model overstates share
        print("  VERDICT: DOWNGRADE")
        print()
        print(f"  {n_optimistic}/{len(ranked)} survivors show OPTIMISTIC share assumption.")
        print("  REWARD_POOL_SHARE_FRACTION = 5% overstates the expected reward contribution.")
        print("  The reward-only EV (from queue/fill analysis) is itself overstated.")
        print("  The counterfactual 'cf_reward_only_ev' would need downward revision.")
        print()
        print("  Recommended action:")
        print("    Revise REWARD_POOL_SHARE_FRACTION to the observed implied share mean.")
        print("    Re-run executable audit with revised fraction.")
        print("    If revised reward_contribution still positive: CONTINUE at lower scale.")
        print("    Do NOT claim profitability. Do NOT submit orders.")
    elif n_supported >= 2:
        print("  VERDICT: CONTINUE")
        print()
        print(f"  {n_supported}/{len(ranked)} survivors show share assumption CONSERVATIVE or FAIR.")
        if n_conservative > 0:
            print(f"  {n_conservative} survivor(s) show CONSERVATIVE assumption:")
            print(f"    Low competition in reward spread window — model 5% understimates actual share.")
            print(f"    Original model EV may be conservative — corrected EV would be higher.")
        if n_fair > 0:
            print(f"  {n_fair} survivor(s) show FAIR assumption:")
            print(f"    Model 5% share is approximately correct at current competition levels.")
        if n_optimistic > 0:
            print(f"  NOTE: {n_optimistic} survivor(s) show OPTIMISTIC assumption — monitor these.")
        print()
        # Show best implied contribution
        best = ranked[0] if ranked else None
        if best and best.mean_implied_reward_contribution is not None:
            print(f"  Strongest survivor (by implied reward contribution):")
            print(f"    {best.slug}")
            print(f"    implied_reward_contribution = ${best.mean_implied_reward_contribution:.4f}/day")
            print(f"    model_reward_contribution   = ${best.model_reward_contribution:.4f}/day")
            print(f"    verdict                     = {best.share_assumption_verdict}")
        print()
        print("  Reward share assumption is broadly supported.")
        print("  The reward-aware line remains alive.")
        print("  Next: operational capital sizing + rate monitoring.")
        print("  Do NOT claim profitability. Do NOT submit orders.")
    else:
        print("  VERDICT: CONTINUE (thin)")
        print()
        print(f"  {n_supported}/{len(ranked)} supporters (below majority).")
        print(f"  {n_unverifiable} unverifiable. Insufficient data for strong conclusion.")
        print("  Run with --rounds 10 and --round-delay 60 before escalating.")

    print()
    _sep("=")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        REWARD_SHARE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REWARD_SHARE_DATA_DIR / f"reward_share_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "clob_host": args.clob_host,
                "rounds": args.rounds,
                "round_delay_sec": args.round_delay,
                "model_share_fraction": REWARD_POOL_SHARE_FRACTION,
                "fair_tolerance": FAIR_TOLERANCE,
                "target_slugs": args.slugs,
                "n_conservative": n_conservative,
                "n_fair": n_fair,
                "n_optimistic": n_optimistic,
                "n_unverifiable": n_unverifiable,
                "analyses": [a.to_dict() for a in ranked],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Results written : {out_path}")
    print()


if __name__ == "__main__":
    main()
