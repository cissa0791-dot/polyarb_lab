"""
reward_aware_authenticated_rewards_truth_line — CLI
polyarb_lab / research_line / validation-only

Authenticated user-level reward truth for the 4 confirmed survivors.

Extends official_rewards_truth_line with CLOB L2 auth:
  - GET /rewards/user/percentages  -> user's actual reward % per market
  - GET /rewards/user/markets      -> user's reward-eligible markets + share breakdown

Three-way comparison per survivor:
  Column 1: model_share_fraction (5%, probe constant)
  Column 2: implied_share_official (market_competitiveness proxy — supporting evidence only)
  Column 3: auth_reward_pct (authenticated user-level %, ground truth if available)

If credentials are absent: UNVERIFIABLE path with detailed setup guidance.
Final judgment: CONTINUE / DOWNGRADE / UNVERIFIABLE

Credential env vars (set before running):
  POLY_API_KEY        — CLOB API key
  POLY_API_SECRET     — CLOB API secret
  POLY_PASSPHRASE     — CLOB passphrase
  POLY_WALLET_ADDRESS — Wallet address (optional)

Usage (Windows PowerShell from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_authenticated_rewards_truth.py
  py -3 research_lines/reward_aware_maker_probe/run_authenticated_rewards_truth.py --verbose
  py -3 research_lines/reward_aware_maker_probe/run_authenticated_rewards_truth.py --skip-discovery

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - market_competitiveness-derived share is supporting evidence only — not final truth.
  - CONTINUE verdict does not mean execution approval.
  - Results go to data/research/reward_aware_maker_probe/authenticated_rewards_truth/ only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
    OfficialTruthAnalysis,
    build_official_truth_analysis,
    fetch_official_reward_config,
)
from research_lines.reward_aware_maker_probe.modules.auth_rewards_truth import (
    CREDENTIAL_GUIDANCE,
    VERDICT_CONTINUE,
    VERDICT_DOWNGRADE,
    VERDICT_UNVERIFIABLE,
    AuthRewardTruth,
    UserMarketsRawFetch,
    build_auth_reward_truth,
    fetch_all_user_markets,
    fetch_auth_percentages,
    get_missing_env_vars,
    load_credentials_from_env,
)

DEFAULT_TARGET_SLUGS = [
    "netanyahu-out-by-june-30-383-244-575",
    "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-n",
]

CLOB_HOST = "https://clob.polymarket.com"
AUTH_TRUTH_DATA_DIR = Path(
    "data/research/reward_aware_maker_probe/authenticated_rewards_truth"
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


def _pct(val: Optional[float]) -> str:
    if val is None:
        return "  N/A "
    return f"{val * 100:6.2f}%"


def _delta_str(val: Optional[float]) -> str:
    if val is None:
        return "     N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.2f}%"


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_three_way_table(
    official_analyses: dict[str, OfficialTruthAnalysis],
    auth_truths: dict[str, AuthRewardTruth],
    slugs: list[str],
) -> None:
    """
    Print three-way comparison table: model 5% | proxy | authenticated.
    """
    print(
        f"  {'Slug':<46}  {'model%':>7}  {'proxy%':>7}  {'auth%':>7}  "
        f"{'authCont':>9}  {'a-m delta':>10}  Verdict"
    )
    print(
        f"  {'-'*46}  {'-'*7}  {'-'*7}  {'-'*7}  "
        f"{'-'*9}  {'-'*10}  -------"
    )
    for slug in slugs:
        off = official_analyses.get(slug)
        auth = auth_truths.get(slug)

        proxy_pct    = off.implied_share_official if off else None
        auth_pct     = auth.auth_reward_pct if auth else None
        auth_cont    = auth.auth_reward_contribution if auth else None
        delta_am     = auth.auth_vs_model_delta if auth else None
        verdict      = auth.verdict if auth else VERDICT_UNVERIFIABLE

        # Short slug for display
        slug_display = slug[:46]
        print(
            f"  {slug_display:<46}  {_pct(REWARD_POOL_SHARE_FRACTION):>7}  "
            f"{_pct(proxy_pct):>7}  {_pct(auth_pct):>7}  "
            f"{'$'+_na(auth_cont,'.4f') if auth_cont is not None else '     N/A':>9}  "
            f"{_delta_str(delta_am):>10}  {verdict}"
        )


def _print_per_survivor_auth_detail(
    slug: str,
    off: Optional[OfficialTruthAnalysis],
    auth: AuthRewardTruth,
    verbose: bool,
) -> None:
    _sep("-", 68)
    print(f"  {slug}")
    print(f"  token_id    : {auth.token_id or 'N/A'}")
    print(f"  condition_id: {auth.condition_id or 'N/A'}")
    print()

    # Column 1: Model
    print(f"  [1] MODEL ASSUMPTION (probe constant)")
    print(f"      model_share_fraction   : {REWARD_POOL_SHARE_FRACTION:.1%}")
    print(f"      model_reward_contrib   : ${auth.model_reward_contribution:.4f}/day")
    print()

    # Column 2: Official proxy
    print(f"  [2] OFFICIAL COMPETITIVENESS PROXY (market_competitiveness, not confirmed ground truth)")
    if off and off.implied_share_official is not None:
        print(f"      implied_share_official : {off.implied_share_official:.2%}")
        print(f"      implied_contrib        : ${_na(off.implied_reward_contribution_official, '.4f')}/day")
        print(f"      market_competitiveness : {_na(off.official.market_competitiveness if off.official else None, '.2f')}")
        print(f"      NOTE: unit semantics INFERRED from endpoint — not a confirmed spec")
    else:
        print(f"      implied_share_official : N/A (official config not available)")
    print()

    # Column 3: Authenticated
    print(f"  [3] AUTHENTICATED USER-LEVEL TRUTH (/rewards/user/markets)")
    print(f"      credentials_present    : {auth.credentials_present}")
    if auth.credentials_present:
        print(f"      /user/percentages HTTP : {auth.pct_http_status or 'N/A'}  "
              f"(401 expected — secondary path)")
        print(f"      /user/markets HTTP     : {auth.markets_http_status or 'N/A'}")
        print(f"      /user/markets entries  : {auth.markets_total_entries or 'N/A'}")
        print(f"      mapping_status         : {auth.mapping_status}")
        if auth.mapping_key:
            print(f"      mapping_key            : {auth.mapping_key}")
        if auth.matched_entry_keys:
            print(f"      matched_entry_keys     : {auth.matched_entry_keys[:15]}")
        print()
        if auth.auth_reward_pct is not None:
            print(f"      auth_reward_pct        : {auth.auth_reward_pct:.2%}  *** AUTHENTICATED ***")
            print(f"      auth_reward_contrib    : ${auth.auth_reward_contribution:.4f}/day")
            if auth.auth_bid_share is not None:
                print(f"      auth_bid_share         : {auth.auth_bid_share:.2%}")
            if auth.auth_ask_share is not None:
                print(f"      auth_ask_share         : {auth.auth_ask_share:.2%}")
            print()
            print(f"      auth vs model delta    : {_delta_str(auth.auth_vs_model_delta)}")
            if auth.auth_vs_proxy_delta is not None:
                print(f"      auth vs proxy delta    : {_delta_str(auth.auth_vs_proxy_delta)}")
        else:
            print(f"      auth_reward_pct        : N/A ({auth.mapping_status})")
    else:
        print(f"      No credentials — set POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE")
    print()

    print(f"  VERDICT: {auth.verdict}")
    if auth.verdict_detail:
        # Word-wrap at 70 chars
        words = auth.verdict_detail.split()
        line = "  Notes: "
        for w in words:
            if len(line) + len(w) + 1 > 78:
                print(line)
                line = "         " + w
            else:
                line += (" " if line.rstrip() else "") + w
        if line.strip():
            print(line)
    print()


def _print_final_judgment(
    auth_truths: list[AuthRewardTruth],
    credentials_present: bool,
) -> None:
    _section("FINAL JUDGMENT: AUTHENTICATED REWARDS TRUTH")

    if not credentials_present:
        print("  STATUS: UNVERIFIABLE — no CLOB credentials available")
        print()
        print("  Three-way comparison status:")
        print("    Column 1 (model 5%)           : AVAILABLE")
        print("    Column 2 (competitiveness proxy): AVAILABLE (see official_rewards_truth)")
        print("    Column 3 (authenticated %)     : NOT AVAILABLE (no credentials)")
        print()
        print("  GOVERNANCE POSITION:")
        print("  The official competitiveness proxy supports CONSERVATIVE classification")
        print("  for all 4 survivors (implied_share 16-63% vs model 5%). However this")
        print("  proxy's unit semantics are not confirmed. Authenticated % would provide")
        print("  ground truth for the user's actual reward share.")
        print()
        print("  To unlock authenticated truth, set credentials and re-run.")
        print("  See credential setup guidance below.")
        return

    continue_count   = sum(1 for t in auth_truths if t.verdict == VERDICT_CONTINUE)
    downgrade_count  = sum(1 for t in auth_truths if t.verdict == VERDICT_DOWNGRADE)
    unverif_count    = sum(1 for t in auth_truths if t.verdict == VERDICT_UNVERIFIABLE)
    total = len(auth_truths)

    print(f"  Survivors evaluated : {total}")
    print(f"  CONTINUE            : {continue_count}")
    print(f"  DOWNGRADE           : {downgrade_count}")
    print(f"  UNVERIFIABLE        : {unverif_count}")
    print()

    # Determine line-level verdict
    if unverif_count == total:
        line_verdict = VERDICT_UNVERIFIABLE
        line_reason  = (
            "Credentials present but all authenticated calls returned unusable results. "
            "Check key validity and confirm L2 HMAC-SHA256 auth format is accepted. "
            "If HTTP 405 persists, review raw payload for server error detail."
        )
    elif downgrade_count > 0:
        line_verdict = VERDICT_DOWNGRADE
        downs = [t.slug for t in auth_truths if t.verdict == VERDICT_DOWNGRADE]
        line_reason = (
            f"{downgrade_count}/{total} survivor(s) have authenticated reward % "
            f"below {50:.0f}% of model assumption: {downs}. "
            "Model overstates reward contribution for these markets. "
            "Reduce model_share_fraction or remove affected survivors from active pool."
        )
    elif continue_count == total:
        line_verdict = VERDICT_CONTINUE
        line_reason  = (
            f"All {total} survivors confirmed at or above model threshold via authenticated data. "
            "Three-way comparison consistent. Model assumption validated."
        )
    elif continue_count > 0:
        line_verdict = VERDICT_CONTINUE
        line_reason  = (
            f"{continue_count}/{total} authenticated, {unverif_count} UNVERIFIABLE. "
            "No DOWNGRADE signals. Continue with confirmed survivors; "
            "treat unverifiable as pending."
        )
    else:
        line_verdict = VERDICT_UNVERIFIABLE
        line_reason  = "No definitive authenticated verdict reached."

    print(f"  LINE VERDICT: {line_verdict}")
    print()
    # Wrap reason
    words = line_reason.split()
    line = "  Reason: "
    for w in words:
        if len(line) + len(w) + 1 > 78:
            print(line)
            line = "          " + w
        else:
            line += (" " if len(line) > len("  Reason: ") else "") + w
    if line.strip():
        print(line)
    print()

    # Governance reminders
    print("  GOVERNANCE REMINDERS:")
    print("  - auth_reward_pct is a point-in-time snapshot; rates can change")
    print("  - CONTINUE does not mean execution approval")
    print("  - market_competitiveness proxy is supporting evidence only")
    print("  - Break-even share analysis from official_rewards_truth still applies")
    print("  - No orders were submitted or approved by this run")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticated reward truth for 4 exec-positive survivors.\n"
            "Three-way comparison: model 5% / competitiveness proxy / authenticated %.\n"
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
        "--skip-discovery", action="store_true",
        help=(
            "Skip discovery + EV pass. Official proxy data will be unavailable "
            "(shows N/A in column 2). Use for fast auth-only testing."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full per-survivor detail including raw API payloads",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    print()
    _section("reward_aware_authenticated_rewards_truth_line")
    print(f"  Timestamp      : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Targets        : {len(args.slugs)} survivors")
    print(f"  Model share    : {REWARD_POOL_SHARE_FRACTION:.1%}")
    print()
    print("  Three-way comparison:")
    print("    Column 1: model_share_fraction (5%) — probe constant")
    print("    Column 2: implied_share_official (market_competitiveness proxy)")
    print("               NOTE: proxy — unit semantics INFERRED, not confirmed spec")
    print("    Column 3: auth_reward_pct — authenticated user-level % (ground truth)")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Credential check
    # -----------------------------------------------------------------------
    _section("Step 1: Credential Check")
    creds = load_credentials_from_env()
    missing = get_missing_env_vars()

    if creds is None:
        print("  STATUS: CREDENTIALS NOT AVAILABLE")
        print()
        missing_required = [v for v in missing if "optional" not in v.lower()]
        if missing_required:
            print(f"  Missing required env vars: {missing_required}")
        print()
        print("  Authenticated column (Column 3) will be N/A for all survivors.")
        print("  Running in PROXY-ONLY mode (Column 1 + Column 2 only).")
        print()
    else:
        print("  STATUS: CREDENTIALS LOADED")
        print(f"  api_key present    : YES  (len={len(creds.api_key)})")
        print(f"  api_secret present : YES  (len={len(creds.api_secret)})")
        print(f"  passphrase present : YES  (len={len(creds.passphrase)})")
        print(f"  wallet_address     : {'SET' if creds.wallet_address else 'NOT SET (optional)'}")
        print()
        if missing:
            optional_missing = [v for v in missing if "optional" in v.lower()]
            if optional_missing:
                print(f"  Optional not set: {optional_missing}")
        print("  Will attempt authenticated calls to /rewards/percentages, /user-shares,")
        print("  /market-scores for each survivor.")
        print()

    # -----------------------------------------------------------------------
    # Step 2: Discovery + EV + Official configs
    # -----------------------------------------------------------------------
    raw_markets: list = []
    ev_by_full_slug: dict = {}
    market_by_slug: dict = {}
    official_analyses: dict[str, OfficialTruthAnalysis] = {}

    if not args.skip_discovery:
        _section("Step 2: Discovery + EV + Official Proxy (Column 2)")
        print("  Fetching rewarded market universe...")
        raw_markets = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
        ev_results  = evaluate_batch(raw_markets)
        ev_by_full_slug = {r.market_slug: r for r in ev_results}

        for slug in args.slugs:
            m = _find_market(slug, raw_markets)
            if m:
                market_by_slug[slug] = m
                ev_r = ev_by_full_slug.get(m.market_slug)
                ev_tag = f"ev={ev_r.reward_adjusted_raw_ev:.5f}" if ev_r else "ev=N/A"
                print(
                    f"  Found : {m.market_slug[:55]}  cid={m.market_id[:18]}..."
                    f"  rate=${m.reward_daily_rate_usdc:.0f}/d  {ev_tag}"
                )
            else:
                print(f"  MISSING: {slug}  -- not in current universe")

        print()
        if not market_by_slug:
            print("  No target slugs found in universe. Aborting.")
            return

        # Fetch official configs
        print("  Fetching official reward configs...")
        with httpx.Client() as client:
            for slug, market in market_by_slug.items():
                official = fetch_official_reward_config(
                    host=args.clob_host,
                    condition_id=market.market_id,
                    client=client,
                )
                ev_r = ev_by_full_slug.get(market.market_slug)
                analysis = build_official_truth_analysis(
                    slug=slug,
                    market=market,
                    ev_result=ev_r,
                    official=official,
                )
                official_analyses[slug] = analysis
                comp = official.market_competitiveness
                impl = analysis.implied_share_official
                impl_str = f"{impl:.2%}" if impl is not None else "N/A"
                comp_str = f"{comp:.2f}" if comp is not None else "N/A"
                print(
                    f"  {slug[:46]:<46}  comp={comp_str:>8}  impl={impl_str:>7}  "
                    f"{analysis.share_assumption_verdict}"
                )
        print()
    else:
        _section("Step 2: SKIPPED (--skip-discovery)")
        print("  Official proxy data (Column 2) not available. Column 2 will show N/A.")
        print()
        # Still need basic market info for token_ids (use empty stubs)
        for slug in args.slugs:
            market_by_slug[slug] = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Step 3: Authenticated calls (Column 3)
    # /user/markets fetched ONCE; condition_id lookup per survivor
    # -----------------------------------------------------------------------
    _section("Step 3: Authenticated Reward % — /user/markets Identity Mapping")

    auth_truths: dict[str, AuthRewardTruth] = {}
    user_markets_fetch: Optional[UserMarketsRawFetch] = None

    if creds is None:
        print("  SKIPPED — no credentials. Column 3 will be N/A.")
        print()
    else:
        print("  Auth method  : CLOB L2 — HMAC-SHA256 (stdlib hmac + hashlib)")
        print("  Primary path : /rewards/user/markets  (fetch-once, all pages)")
        print("  Mapping order: condition_id → token_id → market_slug")
        print("  Secondary    : /rewards/user/percentages (401 expected, diagnostic only)")
        print()

        with httpx.Client() as client:
            # ── Fetch /user/markets once ──────────────────────────────────
            user_markets_fetch = fetch_all_user_markets(args.clob_host, creds, client)
            print(f"  /user/markets  HTTP {user_markets_fetch.http_status}  "
                  f"entries={user_markets_fetch.total_entries}  "
                  f"pages={user_markets_fetch.pages_fetched}")
            if user_markets_fetch.first_entry_keys:
                print(f"  first entry keys : {user_markets_fetch.first_entry_keys}")
            print()

            # ── Fetch /user/percentages for one market (diagnostic only) ──
            sample_token = next(
                (
                    market_by_slug[s].yes_token_id
                    for s in args.slugs
                    if market_by_slug.get(s)
                ),
                "",
            )
            pct_resp_sample = None
            if sample_token:
                pct_resp_sample = fetch_auth_percentages(
                    args.clob_host, sample_token, creds, client
                )
                print(f"  /user/percentages HTTP {pct_resp_sample.http_status} "
                      f"(diagnostic — 401 expected if not user-scoped)")
                if pct_resp_sample.http_status == 200 and pct_resp_sample.raw_payload:
                    print(f"  /user/percentages keys: "
                          f"{list(pct_resp_sample.raw_payload.keys())[:10]}")
            print()

            # ── Per-survivor lookup (no network calls) ────────────────────
            print(f"  {'Slug':<46}  {'mapping':<11}  {'matchKey':<22}  auth%")
            print(f"  {'-'*46}  {'-'*11}  {'-'*22}  -----")
            for slug in args.slugs:
                market = market_by_slug.get(slug)
                off    = official_analyses.get(slug)

                token_id     = market.yes_token_id if market else ""
                condition_id = market.market_id if market else ""
                rate_per_day: float = market.reward_daily_rate_usdc if market else 0.0
                impl_share   = off.implied_share_official if off else None

                # Pass per-survivor pct_response as None (secondary path, 401)
                auth_truth = build_auth_reward_truth(
                    slug=slug,
                    token_id=token_id,
                    condition_id=condition_id,
                    rate_per_day=rate_per_day,
                    implied_share_official=impl_share,
                    creds=creds,
                    user_markets_fetch=user_markets_fetch,
                    pct_response=None,
                )
                auth_truths[slug] = auth_truth

                auth_pct_str = (
                    f"{auth_truth.auth_reward_pct:.2%}"
                    if auth_truth.auth_reward_pct is not None else "N/A"
                )
                print(
                    f"  {slug[:46]:<46}  "
                    f"{auth_truth.mapping_status:<11}  "
                    f"{auth_truth.mapping_key[:22]:<22}  "
                    f"{auth_pct_str}"
                )

    # Fill UNVERIFIABLE stubs for any slugs not yet built (no-credentials path)
    for slug in args.slugs:
        if slug not in auth_truths:
            market  = market_by_slug.get(slug)
            off     = official_analyses.get(slug)
            auth_truths[slug] = build_auth_reward_truth(
                slug=slug,
                token_id=market.yes_token_id if market else "",
                condition_id=market.market_id if market else "",
                rate_per_day=market.reward_daily_rate_usdc if market else 0.0,
                implied_share_official=off.implied_share_official if off else None,
                creds=None,
                user_markets_fetch=None,
                pct_response=None,
            )

    print()

    # -----------------------------------------------------------------------
    # Step 4: Three-way comparison table
    # -----------------------------------------------------------------------
    _section("Step 4: Three-Way Comparison Table")
    print(
        "  Columns: model%(5%) | proxy%(competitiveness) | auth%(authenticated)"
    )
    print(
        "  NOTE: proxy% is supporting evidence only — unit semantics not confirmed."
    )
    print(
        "  NOTE: auth% is user-level point-in-time — not a guaranteed future rate."
    )
    print()
    _print_three_way_table(
        official_analyses=official_analyses,
        auth_truths=auth_truths,
        slugs=args.slugs,
    )
    print()

    # -----------------------------------------------------------------------
    # Step 5: Per-survivor detail
    # -----------------------------------------------------------------------
    _section("Step 5: Per-Survivor Detail")
    for slug in args.slugs:
        off = official_analyses.get(slug)
        auth = auth_truths.get(slug)
        if auth is None:
            print(f"  {slug}: no auth truth built")
            continue
        _print_per_survivor_auth_detail(slug=slug, off=off, auth=auth, verbose=args.verbose)

    # -----------------------------------------------------------------------
    # Step 6: Final judgment
    # -----------------------------------------------------------------------
    auth_list = [auth_truths[s] for s in args.slugs if s in auth_truths]
    _print_final_judgment(auth_truths=auth_list, credentials_present=creds is not None)

    # -----------------------------------------------------------------------
    # Credential guidance (always shown when creds absent)
    # -----------------------------------------------------------------------
    if creds is None:
        _section("Credential Setup Guidance")
        print(CREDENTIAL_GUIDANCE)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------
    run_ts = datetime.now(timezone.utc).isoformat()
    results_out = {
        "run_timestamp": run_ts,
        "line": "reward_aware_authenticated_rewards_truth_line",
        "credentials_present": creds is not None,
        "slugs": args.slugs,
        "auth_truths": {
            slug: auth_truths[slug].to_dict()
            for slug in args.slugs
            if slug in auth_truths
        },
        "official_analyses": {
            slug: official_analyses[slug].to_dict()
            for slug in args.slugs
            if slug in official_analyses
        },
    }

    # Always write to default data dir
    output_path = args.output
    if output_path is None:
        AUTH_TRUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_path = AUTH_TRUTH_DATA_DIR / f"auth_rewards_truth_{ts_tag}.json"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_out, f, indent=2, default=str)
        print(f"  JSON written: {output_path}")
    except Exception as exc:
        print(f"  [WARN] JSON write failed: {exc}")

    print()
    print("reward_aware_authenticated_rewards_truth_line")


if __name__ == "__main__":
    main()
