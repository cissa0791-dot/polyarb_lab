"""
reward_aware_official_rewards_truth_line — CLI
polyarb_lab / research_line / validation-only

Replaces proxy-based reward share estimation with official CLOB data
for the 4 confirmed executable-positive survivors.

Official data used (public, no auth):
  GET /rewards/markets/{condition_id}
    -> market_competitiveness  (official competition metric)
    -> rewards_config[]        (rate_per_day, start_date, end_date, total_days)
    -> rewards_min_size, rewards_max_spread

Auth-required endpoints (probed, results reported, not used):
  /rewards/percentages, /rewards/user-shares, /rewards/market-scores, etc.
  Returns 405 without CLOB L1/L2 auth credentials.

Key computation:
  implied_share_official = min_size / (market_competitiveness + min_size)
  NOTE: market_competitiveness unit interpretation is INFERRED from endpoint
  semantics, not a confirmed public spec.  Use as best-available official
  signal, not confirmed ground truth.

Per-survivor output:
  - Official reward config (rate, spread, size, start/end dates)
  - market_competitiveness value + implied share
  - Break-even share required for positive EV
  - Discrepancy vs discovery proxy
  - Identity check (is the matched market the same as the survivor?)
  - Share assumption verdict: CONSERVATIVE / FAIR / OPTIMISTIC / UNVERIFIABLE /
    NO_ACTIVE_REWARDS / UNCONTESTED_OFFICIAL

Usage (Windows PowerShell from repo root):
    py -3 research_lines/reward_aware_maker_probe/run_official_rewards_truth.py
    py -3 research_lines/reward_aware_maker_probe/run_official_rewards_truth.py --verbose

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - Results go to data/research/reward_aware_maker_probe/official_rewards_truth/ only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
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
from research_lines.reward_aware_maker_probe.modules.official_rewards_truth import (
    FAIR_TOLERANCE,
    NO_REWARDS_LABEL,
    UNCONTESTED_LABEL,
    UNVERIFIABLE_LABEL,
    OfficialTruthAnalysis,
    build_official_truth_analysis,
    fetch_official_reward_config,
    probe_auth_endpoints,
)

DEFAULT_TARGET_SLUGS = [
    "netanyahu-out-by-june-30-383-244-575",
    "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-n",
]

CLOB_HOST = "https://clob.polymarket.com"
OFFICIAL_TRUTH_DATA_DIR = Path(
    "data/research/reward_aware_maker_probe/official_rewards_truth"
)


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _sep(char: str = "-", width: int = 72) -> None:
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
        "CONSERVATIVE":          "CONS ***  (model underestimates share)",
        "FAIR":                  "FAIR      (model ~correct)",
        "OPTIMISTIC":            "OPT ***   (model overestimates share)",
        UNVERIFIABLE_LABEL:      "UNVERIF   (data unavailable)",
        NO_REWARDS_LABEL:        "NO_REWARDS (market has no active reward program)",
        UNCONTESTED_LABEL:       "UNCONTESTED_OFFICIAL (competitiveness=0)",
    }
    return tags.get(verdict, verdict)


def _print_official_config_table(analyses: list[OfficialTruthAnalysis]) -> None:
    print(
        f"  {'Slug':<46}  {'offRate':>7}  {'compScore':>10}  "
        f"{'minSz':>6}  {'maxSpr':>6}  {'active':>6}  {'startDate'}"
    )
    print(
        f"  {'-'*46}  {'-'*7}  {'-'*10}  "
        f"{'-'*6}  {'-'*6}  {'-'*6}  ----------"
    )
    for a in analyses:
        off = a.official
        rate_str  = _na(off.official_rate_per_day, '.1f') if off and off.fetch_ok else "FAIL"
        comp_str  = _na(off.market_competitiveness, '.2f') if off and off.fetch_ok else "N/A"
        msz_str   = _na(off.official_rewards_min_size, '.0f') if off and off.fetch_ok else "N/A"
        msp_str   = _na(off.official_rewards_max_spread, '.1f') if off and off.fetch_ok else "N/A"
        active    = "YES" if (off and off.has_active_reward_config) else "NO"
        start     = (off.reward_start_date or "N/A") if off else "N/A"
        mismatch_flag = " [ID!]" if a.identity_mismatch else ""
        print(
            f"  {(a.slug[:46] + mismatch_flag):<46}  {rate_str:>7}  {comp_str:>10}  "
            f"{msz_str:>6}  {msp_str:>6}  {active:>6}  {start}"
        )


def _print_share_analysis_table(analyses: list[OfficialTruthAnalysis]) -> None:
    print(
        f"  {'Slug':<46}  {'impl%':>7}  {'mdl%':>6}  {'bias':>7}  "
        f"{'bkEvShr':>8}  {'robust':>6}  {'mdlCont':>8}  {'offCont':>8}  Verdict"
    )
    print(
        f"  {'-'*46}  {'-'*7}  {'-'*6}  {'-'*7}  "
        f"{'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  -------"
    )
    for a in analyses:
        impl_str = (
            f"{a.implied_share_official * 100:.2f}%"
            if a.implied_share_official is not None else "N/A"
        )
        mdl_pct   = f"{a.model_share_fraction * 100:.1f}%"
        bias_str  = (
            f"{a.share_bias_official:+.3f}"
            if a.share_bias_official is not None else "N/A"
        )
        bke_str   = (
            f"{a.break_even_share * 100:.2f}%"
            if a.break_even_share is not None else "N/A"
        )
        rob_str   = (
            "YES" if a.model_is_robust else
            "NO " if a.model_is_robust is False else "N/A"
        )
        mdl_c     = _na(a.model_reward_contribution, '.4f')
        off_c     = _na(a.implied_reward_contribution_official, '.4f')
        print(
            f"  {a.slug[:46]:<46}  {impl_str:>7}  {mdl_pct:>6}  {bias_str:>7}  "
            f"{bke_str:>8}  {rob_str:>6}  {mdl_c:>8}  {off_c:>8}  {a.share_assumption_verdict}"
        )


def _print_per_survivor_detail(a: OfficialTruthAnalysis, verbose: bool) -> None:
    _sep("-", 68)
    print(f"  {a.slug}")
    print(f"  full_slug  : {a.full_slug}")
    print(f"  condition_id: {a.condition_id}")
    if a.identity_mismatch:
        print(f"  *** IDENTITY MISMATCH: official slug = {a.official.market_slug if a.official else 'N/A'} ***")
    print()

    off = a.official
    if off and off.fetch_ok:
        print("  OFFICIAL DATA (from /rewards/markets/{condition_id}):")
        print(f"    has_active_rewards       : {off.has_active_reward_config}")
        print(f"    official_rate_per_day    : ${_na(off.official_rate_per_day, '.2f')}/day")
        print(f"    market_competitiveness   : {_na(off.market_competitiveness, '.4f')}")
        print(f"      (interpretation: total eligible quoting score in market;")
        print(f"       unit semantics INFERRED — not confirmed spec)")
        print(f"    official_rewards_min_size: {_na(off.official_rewards_min_size, '.0f')}sh")
        print(f"    official_max_spread_cents: {_na(off.official_rewards_max_spread, '.1f')}c")
        if off.reward_start_date:
            print(f"    reward_start_date        : {off.reward_start_date}")
        if off.reward_end_date:
            print(f"    reward_end_date          : {off.reward_end_date}")
        if off.reward_total_days:
            print(f"    reward_total_days        : {off.reward_total_days}")
        print()
    else:
        print("  OFFICIAL DATA: fetch failed")
        print()

    print("  PROXY vs OFFICIAL DISCREPANCIES:")
    print(f"    discovery_rate    : ${a.discovery_rate_per_day:.2f}/day  |  "
          f"official: ${_na(off.official_rate_per_day if off else None, '.2f')}/day  |  "
          f"delta: {_na(a.rate_discrepancy_pct, '.1f')}%")
    print(f"    discovery_min_sz  : {a.discovery_min_size:.0f}sh  |  "
          f"official: {_na(off.official_rewards_min_size if off else None, '.0f')}sh  |  "
          f"delta: {_na(a.size_discrepancy_pct, '.1f')}%")
    print(f"    discovery_max_spr : {a.discovery_max_spread_cents:.1f}c  |  "
          f"official: {_na(off.official_rewards_max_spread if off else None, '.1f')}c  |  "
          f"delta: {_na(a.spread_discrepancy_pct, '.1f')}%")
    print()

    print("  SHARE ANALYSIS:")
    print(f"    model_share_fraction         : {a.model_share_fraction:.1%}  (REWARD_POOL_SHARE_FRACTION)")
    if a.implied_share_official is not None:
        print(f"    implied_share_official       : {a.implied_share_official:.2%}")
        print(f"    share_bias (implied - model) : {a.share_bias_official:+.2%}")
    else:
        print(f"    implied_share_official       : N/A")
    print(f"    model_reward_contribution    : ${a.model_reward_contribution:.4f}/day")
    if a.implied_reward_contribution_official is not None:
        diff = a.implied_reward_contribution_official - a.model_reward_contribution
        print(f"    implied_reward_contribution  : ${a.implied_reward_contribution_official:.4f}/day  ({diff:+.4f} vs model)")
    print()

    print("  BREAK-EVEN ANALYSIS:")
    if a.break_even_share is not None:
        print(f"    break_even_share             : {a.break_even_share:.2%}")
        print(f"      (minimum share to cover fill penalties at probe calibration)")
        print(f"    model_share >= break_even?   : {'YES — model robust' if a.model_is_robust else 'NO — fill penalties exceed model share contribution'}")
        if a.implied_share_official is not None and a.implied_share_official > 0:
            robust_vs_official = a.implied_share_official >= a.break_even_share
            print(f"    official_share >= break_even?: {'YES' if robust_vs_official else 'NO'}")
    else:
        print(f"    break_even_share             : N/A (no EV result)")
    print()

    print(f"  VERDICT: {_verdict_tag(a.share_assumption_verdict)}")
    if a.share_assumption_notes:
        print(f"  Notes: {a.share_assumption_notes[:140]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Official rewards truth validation for 4 executable-positive survivors.\n"
            "Uses CLOB /rewards/markets/{condition_id} as primary data source.\n"
            "Paper-only. No order submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--slugs", nargs="+", default=DEFAULT_TARGET_SLUGS,
        help="Target slugs to validate",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON results here",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full per-survivor detail",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    print()
    _section("reward_aware_official_rewards_truth_line")
    print(f"  Timestamp   : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Targets     : {len(args.slugs)} survivors")
    print(f"  Model share : {REWARD_POOL_SHARE_FRACTION:.1%}  (to be compared vs official)")
    print(f"  Fair band   : [{REWARD_POOL_SHARE_FRACTION*(1-FAIR_TOLERANCE):.1%}, "
          f"{REWARD_POOL_SHARE_FRACTION*(1+FAIR_TOLERANCE):.1%}]  (+-{FAIR_TOLERANCE:.0%})")
    print()
    print("  Official data source: GET /rewards/markets/{condition_id}")
    print("  key field: market_competitiveness (official CLOB competition metric)")
    print("  implied_share = min_size / (market_competitiveness + min_size)")
    print("  [NOTE: market_competitiveness unit INFERRED — not confirmed spec]")
    print()
    print("  PRIOR RESULT WITHDRAWN:")
    print("  reward_share_validation (book-depth proxy) returned CONSERVATIVE / CONTINUE")
    print("  for all 4 survivors based on empty eligible depth -> implied_share=100%.")
    print("  That result is INVALID: eligible_depth=0 is EMPTY_REWARD_WINDOW, not CONSERVATIVE.")
    print("  This run supersedes it with official market_competitiveness data.")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discovery + EV
    # -----------------------------------------------------------------------
    _section("Step 1: Discovery + EV Model")
    raw_markets = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
    ev_results  = evaluate_batch(raw_markets)
    ev_by_full_slug = {r.market_slug: r for r in ev_results}

    market_by_slug: dict[str, object] = {}
    for slug in args.slugs:
        m = _find_market(slug, raw_markets)
        if m:
            market_by_slug[slug] = m
            ev_r = ev_by_full_slug.get(m.market_slug)
            ev_tag = f"ev={ev_r.reward_adjusted_raw_ev:.5f}" if ev_r else "ev=N/A"
            print(
                f"  Found : {m.market_slug[:55]}  cid={m.market_id[:20]}..."
                f"  rate=${m.reward_daily_rate_usdc:.0f}/d  {ev_tag}"
            )
        else:
            print(f"  MISSING : {slug}  -- not in current universe")
    print()

    if not market_by_slug:
        print("  No target slugs found. Aborting.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Auth endpoint probe (one-time, any market)
    # -----------------------------------------------------------------------
    _section("Step 2: Auth Endpoint Probe")
    print("  Probing auth-required endpoints (expect 401/403/405 without credentials)...")
    auth_results: dict[str, int] = {}
    with httpx.Client() as client:
        auth_results = probe_auth_endpoints(args.clob_host, client)
    for path, code in auth_results.items():
        status = "PUBLIC" if code == 200 else f"AUTH_REQUIRED ({code})" if code in (401, 403, 405) else f"NOT_FOUND ({code})" if code == 404 else f"ERROR ({code})"
        print(f"  {path:<40}  {status}")
    public_paths = [p for p, c in auth_results.items() if c == 200]
    auth_paths   = [p for p, c in auth_results.items() if c in (401, 403, 405)]
    print()
    if public_paths:
        print(f"  Public endpoints found: {public_paths}")
    print(f"  Auth-required: {len(auth_paths)} endpoints need CLOB credentials (wallet + L1/L2 sig)")
    print(f"  Without auth: market_competitiveness is the best available official signal")
    print()

    # -----------------------------------------------------------------------
    # Step 3: Fetch official configs per survivor
    # -----------------------------------------------------------------------
    _section("Step 3: Official Reward Config Fetch")
    print(f"  Fetching {len(market_by_slug)} targeted /rewards/markets/{{condition_id}} calls...")
    print()

    analyses: list[OfficialTruthAnalysis] = []

    with httpx.Client() as client:
        for slug, market in market_by_slug.items():
            cid = market.market_id
            official = fetch_official_reward_config(
                host=args.clob_host,
                condition_id=cid,
                client=client,
            )
            official.auth_probe_results = auth_results

            ev_r = ev_by_full_slug.get(market.market_slug)
            analysis = build_official_truth_analysis(
                slug=slug,
                market=market,
                ev_result=ev_r,
                official=official,
            )
            analyses.append(analysis)

            fetch_tag = "ok" if official.fetch_ok else "FAIL"
            comp_str = f"{official.market_competitiveness:.2f}" if official.market_competitiveness is not None else "N/A"
            active_tag = "active" if official.has_active_reward_config else "NO_REWARDS"
            print(
                f"  {slug[:50]:<50}  [{fetch_tag}]"
                f"  comp={comp_str}  rate=${_na(official.official_rate_per_day, '.0f')}/d"
                f"  {active_tag}"
            )
    print()

    # -----------------------------------------------------------------------
    # Results tables
    # -----------------------------------------------------------------------
    _section("Official Reward Config Table")
    _print_official_config_table(analyses)
    print()
    print("  offRate    : official rate_per_day from rewards_config (USDC/day)")
    print("  compScore  : market_competitiveness from official endpoint")
    print("  active     : rewards_config non-empty with rate > 0")
    print("  [ID!]      : official market_slug differs from survivor's full_slug (identity mismatch)")

    print()
    _section("Share Analysis Table  (official market_competitiveness)")
    _print_share_analysis_table(analyses)
    print()
    print(f"  impl%    : implied_share = min_size / (market_competitiveness + min_size)")
    print(f"  mdl%     : model 5% assumption (REWARD_POOL_SHARE_FRACTION)")
    print(f"  bias     : implied - model  (positive = model conservative, negative = optimistic)")
    print(f"  bkEvShr  : break-even share  (min share to cover fill penalties)")
    print(f"  robust   : model share >= break_even (EV model is not fragile to share compression)")
    print(f"  mdlCont  : model_reward_contribution = rate x 5%")
    print(f"  offCont  : implied_reward_contribution = rate x implied_share")
    print()
    print(f"  GOVERNANCE NOTE: market_competitiveness unit is INFERRED from endpoint semantics.")
    print(f"  implied_share is best-available official signal, NOT confirmed realized reward %.")
    print(f"  Ground truth would require auth credentials for /rewards/percentages endpoint.")

    # Per-survivor detail
    print()
    _section("Per-Survivor Official Truth Detail")
    for a in analyses:
        _print_per_survivor_detail(a, verbose=args.verbose)

    # -----------------------------------------------------------------------
    # Final judgment
    # -----------------------------------------------------------------------
    _sep("=")
    print("  OFFICIAL REWARD TRUTH JUDGMENT")
    _sep("=")

    n_conservative   = sum(1 for a in analyses if a.share_assumption_verdict == "CONSERVATIVE")
    n_fair           = sum(1 for a in analyses if a.share_assumption_verdict == "FAIR")
    n_optimistic     = sum(1 for a in analyses if a.share_assumption_verdict == "OPTIMISTIC")
    n_no_rewards     = sum(1 for a in analyses if a.share_assumption_verdict == NO_REWARDS_LABEL)
    n_uncontested    = sum(1 for a in analyses if a.share_assumption_verdict == UNCONTESTED_LABEL)
    n_unverifiable   = sum(1 for a in analyses if a.share_assumption_verdict == UNVERIFIABLE_LABEL)
    n_identity_miss  = sum(1 for a in analyses if a.identity_mismatch)

    print(f"  CONSERVATIVE (model < implied)         : {n_conservative}/{len(analyses)}")
    print(f"  FAIR         (model ~= implied)        : {n_fair}/{len(analyses)}")
    print(f"  OPTIMISTIC   (model > implied)         : {n_optimistic}/{len(analyses)}")
    print(f"  NO_ACTIVE_REWARDS                      : {n_no_rewards}/{len(analyses)}")
    print(f"  UNCONTESTED_OFFICIAL (comp=0)          : {n_uncontested}/{len(analyses)}")
    print(f"  UNVERIFIABLE                           : {n_unverifiable}/{len(analyses)}")
    print(f"  Identity mismatches                    : {n_identity_miss}/{len(analyses)}")
    print()

    # Report identity issues first
    if n_identity_miss > 0:
        print(f"  IDENTITY WARNINGS:")
        for a in analyses:
            if a.identity_mismatch:
                off_slug = a.official.market_slug if a.official else "N/A"
                print(f"    Survivor: {a.slug}")
                print(f"    Official: {off_slug}")
                print(f"    These may be different markets. Verify the condition_id mapping.")
        print()

    if n_no_rewards > 0:
        print(f"  ACTIVE REWARD FAILURES:")
        for a in analyses:
            if a.share_assumption_verdict == NO_REWARDS_LABEL:
                print(f"    {a.slug}  ->  rewards_config empty on official endpoint")
                print(f"    This survivor has NO ACTIVE REWARD PROGRAM.")
                print(f"    Remove from active survivor pool immediately.")
        print()

    # Main judgment
    n_supported   = n_conservative + n_fair
    n_actionable  = n_supported + n_uncontested

    if n_no_rewards == len(analyses):
        print("  VERDICT: PARK")
        print()
        print("  ALL survivors have no active reward program on official endpoint.")
        print("  The reward-aware line has no valid basis. Park this line.")
    elif n_no_rewards > 0 and n_supported == 0:
        print("  VERDICT: DOWNGRADE")
        print()
        print(f"  {n_no_rewards} survivor(s) have no active rewards.")
        print(f"  {n_unverifiable + n_uncontested} are unverifiable or uncontested.")
        print(f"  {n_optimistic} show OPTIMISTIC assumption.")
        print("  Remove no-reward survivors. Re-run with reduced pool.")
    elif n_optimistic > len(analyses) // 2:
        print("  VERDICT: DOWNGRADE")
        print()
        print(f"  {n_optimistic}/{len(analyses)} survivors show OPTIMISTIC share assumption.")
        print("  REWARD_POOL_SHARE_FRACTION = 5% materially overstates expected share.")
        print("  Re-run executable audit with revised fraction before escalating.")
    elif n_supported >= 2:
        print("  VERDICT: CONTINUE")
        print()
        print(f"  {n_supported}/{len(analyses)} survivors show CONSERVATIVE or FAIR share assumption.")
        if n_conservative > 0:
            print(f"  {n_conservative} survivor(s) are CONSERVATIVE: model underestimates reward contribution.")
        if n_fair > 0:
            print(f"  {n_fair} survivor(s) are FAIR: model approximately correct.")
        if n_no_rewards > 0:
            print(f"  IMPORTANT: {n_no_rewards} survivor(s) have no active rewards — remove from pool.")
        if n_uncontested > 0:
            print(f"  NOTE: {n_uncontested} survivor(s) show market_competitiveness=0 (uncontested)")
            print(f"  — treat as unverified pending more data, not as confirmed reward capture.")
        print()
        # Best survivor by implied contribution
        ranked_by_implied = sorted(
            [a for a in analyses if a.implied_reward_contribution_official is not None],
            key=lambda x: x.implied_reward_contribution_official or 0,
            reverse=True,
        )
        if ranked_by_implied:
            best = ranked_by_implied[0]
            print(f"  Strongest official-supported survivor:")
            print(f"    {best.slug}")
            print(f"    market_competitiveness           = {_na(best.official.market_competitiveness if best.official else None, '.2f')}")
            print(f"    implied_share_official           = {_na(best.implied_share_official, '.2%')}")
            print(f"    implied_reward_contribution      = ${_na(best.implied_reward_contribution_official, '.4f')}/day")
            print(f"    model_reward_contribution        = ${best.model_reward_contribution:.4f}/day")
            print(f"    break_even_share                 = {_na(best.break_even_share, '.2%')}")
            print(f"    model_robust_vs_break_even       = {best.model_is_robust}")
        print()
        print("  Reward share assumptions broadly supported by official CLOB data.")
        print("  NOTE: implied_share is a proxy from market_competitiveness — not realized reward %.")
        print("  Auth credentials (/rewards/percentages) would provide ground-truth user reward %.")
        print("  Do NOT claim profitability. Do NOT submit orders.")
    elif n_actionable >= 1:
        print("  VERDICT: CONTINUE (thin)")
        print()
        print(f"  {n_actionable} survivor(s) have non-negative official verdict.")
        print("  Pool is thin. Run again with more context before escalating.")
    else:
        print("  VERDICT: UNVERIFIABLE")
        print()
        print("  Insufficient official data to assess share assumption.")
        print("  Check API connectivity and condition_id mapping.")

    print()
    _sep("=")

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        OFFICIAL_TRUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OFFICIAL_TRUTH_DATA_DIR / f"official_truth_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "clob_host": args.clob_host,
                "model_share_fraction": REWARD_POOL_SHARE_FRACTION,
                "fair_tolerance": FAIR_TOLERANCE,
                "target_slugs": args.slugs,
                "auth_probe_results": auth_results,
                "n_conservative": n_conservative,
                "n_fair": n_fair,
                "n_optimistic": n_optimistic,
                "n_no_rewards": n_no_rewards,
                "n_uncontested": n_uncontested,
                "n_unverifiable": n_unverifiable,
                "n_identity_mismatch": n_identity_miss,
                "analyses": [a.to_dict() for a in analyses],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  Results written : {out_path}")
    print()


if __name__ == "__main__":
    main()
