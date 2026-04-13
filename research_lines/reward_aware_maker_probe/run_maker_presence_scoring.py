"""
reward_aware_maker_presence_scoring_line — CLI
polyarb_lab / research_line / validation-only

Diagnoses WHY earning_percentage = 0% for the 4 exec-positive survivors.

Context:
  authenticated_rewards_truth_line established:
    - 3/4 survivors FOUND in /rewards/user/markets (hungary, vance, rubio)
    - 1/4 NOT_FOUND (netanyahu)
    - All 3 found survivors: earning_percentage = 0.00%
    - Previous verdict: DOWNGRADE (incorrectly strong)

  This line resolves the governance error:
    earning_percentage = 0% is account-state truth, NOT market-thesis falsification.
    0% means the account is not currently participating as a qualifying maker.
    This does NOT prove the market reward opportunity is invalid.

  This runner diagnoses the participation gap per survivor:
    NO_ACTIVE_PRESENCE  — no open orders or no maker address
    OUTSIDE_WINDOW      — orders outside rewards_max_spread window
    BELOW_MIN_SIZE      — orders in window but size < rewards_min_size
    NO_BILATERAL_QUOTES — only one side quoted; rewards require both
    NOT_IN_USER_MARKETS — market not registered in this account
    SCORING_ZERO        — qualifying orders present but still 0% (true failure)

  CONTINUE if all 0% cases explained by non-participation.
  DOWNGRADE only if SCORING_ZERO detected.

Credential env vars:
  POLY_API_KEY        — CLOB API key
  POLY_API_SECRET     — CLOB API secret
  POLY_PASSPHRASE     — CLOB passphrase
  POLY_WALLET_ADDRESS — Wallet address (optional)

Usage (Windows PowerShell from repo root):
  py -3 research_lines/reward_aware_maker_probe/run_maker_presence_scoring.py
  py -3 research_lines/reward_aware_maker_probe/run_maker_presence_scoring.py --verbose
  py -3 research_lines/reward_aware_maker_probe/run_maker_presence_scoring.py --skip-discovery

STRICT RULES:
  - Paper only. No order submission.
  - No mainline contamination.
  - No profitability claim.
  - CONTINUE verdict does not mean execution approval.
  - Results go to data/research/reward_aware_maker_probe/maker_presence_scoring/ only.
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
    OfficialRewardConfig,
    OfficialTruthAnalysis,
    build_official_truth_analysis,
    fetch_official_reward_config,
)
from research_lines.reward_aware_maker_probe.modules.auth_rewards_truth import (
    CREDENTIAL_GUIDANCE,
    UserMarketsRawFetch,
    fetch_all_user_markets,
    get_missing_env_vars,
    load_credentials_from_env,
)
from research_lines.reward_aware_maker_probe.modules.maker_presence_scoring import (
    DIAG_NO_ACTIVE_PRESENCE,
    DIAG_IDENTITY_MISMATCH,
    DIAG_NON_SCORING_PRESENCE,
    DIAG_NOT_IN_USER_MARKETS,
    DIAG_SCORING_ZERO,
    DIAG_UNVERIFIABLE,
    JUDGMENT_CONTINUE,
    JUDGMENT_DOWNGRADE,
    JUDGMENT_UNVERIFIABLE,
    PresenceDiagnosis,
    build_presence_diagnosis,
)

DEFAULT_TARGET_SLUGS = [
    "netanyahu-out-by-june-30-383-244-575",
    "will-the-next-prime-minister-of-hungary-be-pter-magyar",
    "will-jd-vance-win-the-2028-republican-presidential-nomi",
    "will-marco-rubio-win-the-2028-republican-presidential-n",
]

# Fallback reward params when discovery is skipped (from last confirmed official run)
# rewards_max_spread in cents; rewards_min_size in shares
_FALLBACK_REWARD_PARAMS: dict[str, dict] = {
    "netanyahu-out-by-june-30-383-244-575": {
        "rewards_max_spread_cents": 4.0,
        "rewards_min_size": 10.0,
    },
    "will-the-next-prime-minister-of-hungary-be-pter-magyar": {
        "rewards_max_spread_cents": 4.0,
        "rewards_min_size": 10.0,
    },
    "will-jd-vance-win-the-2028-republican-presidential-nomi": {
        "rewards_max_spread_cents": 4.0,
        "rewards_min_size": 10.0,
    },
    "will-marco-rubio-win-the-2028-republican-presidential-n": {
        "rewards_max_spread_cents": 4.0,
        "rewards_min_size": 10.0,
    },
}

CLOB_HOST = "https://clob.polymarket.com"
PRESENCE_DATA_DIR = Path(
    "data/research/reward_aware_maker_probe/maker_presence_scoring"
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


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_summary_table(diagnoses: list[PresenceDiagnosis]) -> None:
    """
    Print one-line per survivor summary:
      Slug | found | earning% | has_orders | scoring | diagnosis_code | judgment
    """
    header = (
        f"  {'Slug':<46}  {'found':>5}  {'earn%':>6}  "
        f"{'presence':>8}  {'scoring':>11}  {'diagnosis':<22}  judgment"
    )
    print(header)
    print(
        f"  {'-'*46}  {'-'*5}  {'-'*6}  "
        f"{'-'*8}  {'-'*11}  {'-'*22}  -------"
    )
    for d in diagnoses:
        found_str    = "YES" if d.found_in_user_markets else "NO"
        earn_str     = f"{d.current_earning_pct*100:.2f}%" if d.found_in_user_markets else "N/A"
        presence_str = "YES" if d.has_active_presence else "NO"
        scoring_str  = d.scoring_status
        diag_str     = d.diagnosis_code
        jdg_str      = d.judgment
        slug_d       = d.slug[:46]
        print(
            f"  {slug_d:<46}  {found_str:>5}  {earn_str:>6}  "
            f"{presence_str:>8}  {scoring_str:>11}  {diag_str:<22}  {jdg_str}"
        )


def _print_per_survivor_detail(d: PresenceDiagnosis, verbose: bool) -> None:
    _sep("-", 68)
    print(f"  {d.slug}")
    print(f"  condition_id : {d.condition_id or 'N/A'}")
    print(f"  token_id     : {d.token_id[:40] if d.token_id else 'N/A'}...")
    print()

    # /user/markets fields
    print(f"  /user/markets entry:")
    print(f"    found_in_user_markets  : {d.found_in_user_markets}")
    if d.found_in_user_markets:
        print(f"    current_earning_pct    : {d.current_earning_pct * 100:.4f}%  *** account-state truth ***")
        from research_lines.reward_aware_maker_probe.modules.maker_presence_scoring import _is_null_address
        is_null = _is_null_address(d.maker_address)
        print(f"    maker_address          : {d.maker_address or '(empty)'}  null={is_null}")
        print(f"    quoted_spread          : {_na(d.quoted_spread, '.4f')} (price units; window={d.rewards_max_spread_fraction:.4f})")
        spread_ok = (
            d.quoted_spread is not None
            and d.quoted_spread <= d.rewards_max_spread_fraction
        )
        print(f"    spread_in_window       : {spread_ok if d.quoted_spread is not None else 'N/A'}")
        print(f"    cumulative_earnings    : ${_na(d.cumulative_earnings, '.4f')}")
        print(f"    market_competitiveness : {_na(d.market_competitiveness, '.2f')}")
    print()

    # Reward window params
    print(f"  Reward window params:")
    print(f"    rewards_max_spread     : {d.rewards_max_spread_fraction:.4f} (price units = {d.rewards_max_spread_fraction*100:.1f} cents)")
    print(f"    rewards_min_size       : {d.rewards_min_size:.0f} shares")
    print()

    # Open orders
    o = d.orders
    if o is None:
        print(f"  Open orders: NOT FETCHED (no credentials)")
    else:
        print(f"  Open orders (L2 auth — 6 endpoint variants tried):")
        print(f"    http_status            : {o.orders_http_status}")
        print(f"    fetch_ok               : {o.fetch_ok}")
        if o.orders_fetch_blocker:
            print(f"    orders_fetch_blocker   : {o.orders_fetch_blocker}")
        if o.fetch_ok:
            print(f"    total_orders           : {o.total_orders}")
            print(f"    bid_orders             : {o.bid_orders}")
            print(f"    ask_orders             : {o.ask_orders}")
            print(f"    bid_orders_in_window   : {o.bid_orders_in_window}")
            print(f"    ask_orders_in_window   : {o.ask_orders_in_window}")
            print(f"    max_bid_size_in_window : {_na(o.max_bid_size_in_window, '.1f')} sh")
            print(f"    max_ask_size_in_window : {_na(o.max_ask_size_in_window, '.1f')} sh")
        else:
            print(f"    error                  : {o.error_detail}")
        if verbose and o.variant_statuses:
            print(f"    variant_statuses:")
            for label, st in o.variant_statuses:
                print(f"      HTTP {st}  {label}")
        if verbose and o.raw_orders_sample:
            print(f"    raw_orders_sample[0]   : {o.raw_orders_sample[0]}")
    print()

    # Book context
    b = d.book
    if b is None:
        print(f"  Book context: NOT FETCHED")
    else:
        print(f"  Book context (public /book):")
        print(f"    fetch_ok               : {b.fetch_ok}")
        if b.fetch_ok:
            print(f"    best_bid               : {_na(b.best_bid, '.4f')}")
            print(f"    best_ask               : {_na(b.best_ask, '.4f')}")
            print(f"    spread                 : {_na(b.spread, '.4f')}")
            print(f"    bid_levels_in_window   : {b.bid_levels_in_window}")
            print(f"    ask_levels_in_window   : {b.ask_levels_in_window}")
    print()

    # Diagnosis
    print(f"  DIAGNOSIS: {d.diagnosis_code}")
    print(f"  scoring_status: {d.scoring_status}")
    print(f"  has_active_presence: {d.has_active_presence}")
    print()
    # Word-wrap detail
    words = d.diagnosis_detail.split()
    line = "  Detail: "
    for w in words:
        if len(line) + len(w) + 1 > 78:
            print(line)
            line = "          " + w
        else:
            line += (" " if len(line) > len("  Detail: ") else "") + w
    if line.strip():
        print(line)
    print()
    print(f"  JUDGMENT: {d.judgment}")
    print()


def _print_final_judgment(diagnoses: list[PresenceDiagnosis]) -> None:
    _section("FINAL JUDGMENT: MAKER PRESENCE SCORING LINE")

    total           = len(diagnoses)
    continue_count  = sum(1 for d in diagnoses if d.judgment == JUDGMENT_CONTINUE)
    downgrade_count = sum(1 for d in diagnoses if d.judgment == JUDGMENT_DOWNGRADE)
    unverif_count   = sum(1 for d in diagnoses if d.judgment == JUDGMENT_UNVERIFIABLE)
    scoring_zero    = [d for d in diagnoses if d.diagnosis_code == DIAG_SCORING_ZERO]

    print(f"  Survivors evaluated : {total}")
    print(f"  CONTINUE            : {continue_count}")
    print(f"  DOWNGRADE           : {downgrade_count}  (SCORING_ZERO only)")
    print(f"  UNVERIFIABLE        : {unverif_count}")
    print()

    # Determine line verdict
    # Rule: UNVERIFIABLE cases must never elevate the verdict to CONTINUE.
    # Verdict hierarchy: DOWNGRADE > UNVERIFIABLE > CONTINUE.
    if downgrade_count > 0:
        line_verdict = JUDGMENT_DOWNGRADE
        downs = [d.slug[:40] for d in scoring_zero]
        reason = (
            f"{downgrade_count}/{total} survivor(s) have qualifying maker orders "
            f"(bilateral, in-window, above min_size) but earning_percentage remains 0%: "
            f"{downs}. "
            "This is SCORING_ZERO — the only condition warranting DOWNGRADE. "
            "Competitor density is sufficient to exclude this account despite qualifying quotes. "
            "Reduce confidence in reward-share model for these markets."
        )
    elif unverif_count > 0:
        # UNVERIFIABLE survivors are genuinely unresolved.
        # Do NOT issue CONTINUE while any case remains unexplained.
        line_verdict   = JUDGMENT_UNVERIFIABLE
        unverif_slugs  = [d.slug[:40] for d in diagnoses if d.judgment == JUDGMENT_UNVERIFIABLE]
        unverif_codes  = sorted({d.diagnosis_code for d in diagnoses if d.judgment == JUDGMENT_UNVERIFIABLE})
        continue_codes = sorted({d.diagnosis_code for d in diagnoses if d.judgment == JUDGMENT_CONTINUE})
        reason = (
            f"{unverif_count}/{total} survivor(s) are still unresolved: {unverif_codes} "
            f"({unverif_slugs}). "
            f"{continue_count}/{total} explained by non-participation: {continue_codes}. "
            "UNVERIFIABLE blocks CONTINUE at line level. "
            "Resolve open-order state (check /orders endpoint availability) before "
            "upgrading this line to CONTINUE."
        )
    elif continue_count == total:
        line_verdict = JUDGMENT_CONTINUE
        diag_codes = sorted({d.diagnosis_code for d in diagnoses})
        reason = (
            f"All {total} survivors explained by non-participation: {diag_codes}. "
            "earning_percentage = 0% is consistent with no active maker presence — "
            "not with competitive exclusion of qualifying quotes. "
            "Market reward opportunity remains intact. "
            "Model assumption (5% share) is not contradicted by this evidence. "
            "Next step: place qualifying bilateral orders to validate scoring."
        )
    else:
        line_verdict = JUDGMENT_UNVERIFIABLE
        reason = (
            f"Mixed results: {continue_count} CONTINUE, {unverif_count} UNVERIFIABLE. "
            "No DOWNGRADE triggered. Continue with caution; resolve UNVERIFIABLE cases."
        )

    print(f"  LINE VERDICT: {line_verdict}")
    print()
    words = reason.split()
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

    print("  GOVERNANCE REMINDERS:")
    print("  - 0% earning = account not currently participating as qualifying maker")
    print("  - 0% earning ≠ market reward opportunity invalid")
    print("  - CONTINUE does not mean execution approval")
    print("  - No orders were submitted or approved by this run")
    print("  - Place bilateral quotes within window to activate scoring")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Maker presence scoring for 4 exec-positive survivors.\n"
            "Diagnoses WHY earning_percentage = 0% per survivor.\n"
            "Paper-only. No order submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clob-host", default=CLOB_HOST)
    parser.add_argument(
        "--slugs", nargs="+", default=DEFAULT_TARGET_SLUGS,
        help="Target slugs to diagnose",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON results here",
    )
    parser.add_argument(
        "--skip-discovery", action="store_true",
        help=(
            "Skip discovery + EV pass. Uses fallback reward params "
            "(max_spread=4 cents, min_size=10 shares). "
            "Use for fast presence-only testing."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full per-survivor detail including raw order samples",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    print()
    _section("reward_aware_maker_presence_scoring_line")
    print(f"  Timestamp      : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Targets        : {len(args.slugs)} survivors")
    print(f"  Model share    : {REWARD_POOL_SHARE_FRACTION:.1%}")
    print()
    print("  Governance context:")
    print("    auth_rewards_truth_line found earning_pct = 0% for 3 FOUND survivors.")
    print("    This line diagnoses the participation gap — 0% ≠ market thesis invalid.")
    print("    DOWNGRADE only if SCORING_ZERO (qualifying orders present, still 0%).")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Credential check
    # -----------------------------------------------------------------------
    _section("Step 1: Credential Check")
    creds = load_credentials_from_env()
    missing = get_missing_env_vars()

    if creds is None:
        print("  STATUS: CREDENTIALS NOT AVAILABLE")
        if missing:
            missing_req = [v for v in missing if "optional" not in v.lower()]
            if missing_req:
                print(f"  Missing required env vars: {missing_req}")
        print()
        print("  /orders endpoint requires L2 auth — order fetch will be skipped.")
        print("  Presence diagnosis will rely on /user/markets entry fields only.")
        print()
    else:
        print("  STATUS: CREDENTIALS LOADED")
        print(f"  api_key present    : YES  (len={len(creds.api_key)})")
        print(f"  api_secret present : YES  (len={len(creds.api_secret)})")
        print(f"  passphrase present : YES  (len={len(creds.passphrase)})")
        print(f"  wallet_address     : {'SET' if creds.wallet_address else 'NOT SET (optional)'}")
        print()

    # -----------------------------------------------------------------------
    # Step 2: Discovery + Official configs (for reward params)
    # -----------------------------------------------------------------------
    market_by_slug: dict = {}
    official_by_slug: dict[str, OfficialRewardConfig] = {}

    if not args.skip_discovery:
        _section("Step 2: Discovery + Official Reward Params")
        print("  Fetching rewarded market universe...")
        raw_markets = discover_fee_enabled_rewarded_markets(clob_host=args.clob_host)
        ev_results  = evaluate_batch(raw_markets)
        ev_by_slug  = {r.market_slug: r for r in ev_results}

        for slug in args.slugs:
            m = _find_market(slug, raw_markets)
            if m:
                market_by_slug[slug] = m
                ev_r = ev_by_slug.get(m.market_slug)
                ev_tag = f"ev={ev_r.reward_adjusted_raw_ev:.5f}" if ev_r else "ev=N/A"
                print(
                    f"  Found : {m.market_slug[:50]}  "
                    f"spread={m.rewards_max_spread_cents}c  "
                    f"minsize={m.rewards_min_size}sh  {ev_tag}"
                )
            else:
                print(f"  MISSING: {slug}  -- not in current universe")

        print()
        if not market_by_slug:
            print("  No target slugs found in universe. Aborting.")
            return

        # Fetch official reward configs for max_spread / min_size ground truth
        print("  Fetching official reward configs for window parameters...")
        with httpx.Client() as client:
            for slug, market in market_by_slug.items():
                off_cfg = fetch_official_reward_config(
                    host=args.clob_host,
                    condition_id=market.market_id,
                    client=client,
                )
                official_by_slug[slug] = off_cfg
                spread = off_cfg.official_rewards_max_spread
                minsize = off_cfg.official_rewards_min_size
                print(
                    f"  {slug[:46]:<46}  "
                    f"official_spread={_na(spread, '.1f')}c  "
                    f"official_minsize={_na(minsize, '.0f')}sh"
                )
        print()
    else:
        _section("Step 2: SKIPPED (--skip-discovery)")
        print("  Using fallback reward params:")
        for slug in args.slugs:
            fp = _FALLBACK_REWARD_PARAMS.get(slug, {})
            print(
                f"  {slug[:46]:<46}  "
                f"spread={fp.get('rewards_max_spread_cents', 4.0):.1f}c  "
                f"minsize={fp.get('rewards_min_size', 10.0):.0f}sh  (fallback)"
            )
            market_by_slug[slug] = None  # no market object
        print()

    # -----------------------------------------------------------------------
    # Step 3: Fetch /user/markets once
    # -----------------------------------------------------------------------
    _section("Step 3: /rewards/user/markets — Fetch Once")
    user_markets_fetch: Optional[UserMarketsRawFetch] = None

    if creds is None:
        print("  SKIPPED — no credentials.")
        print("  Presence diagnosis will use /user/markets data from previous run if available.")
        print()
    else:
        with httpx.Client() as _pre_client:
            user_markets_fetch = fetch_all_user_markets(
                args.clob_host, creds, _pre_client
            )
        print(
            f"  /user/markets  HTTP {user_markets_fetch.http_status}  "
            f"entries={user_markets_fetch.total_entries}  "
            f"pages={user_markets_fetch.pages_fetched}"
        )
        if not user_markets_fetch.fetch_ok:
            print(f"  [FETCH FAILED]  error_detail : {user_markets_fetch.error_detail}")
            if user_markets_fetch.raw_error_text:
                print(f"  [FETCH FAILED]  raw_response : {user_markets_fetch.raw_error_text[:400]}")
            print()
            print("  DIAGNOSIS GUIDANCE for /user/markets 400 / non-200 responses:")
            st = user_markets_fetch.http_status
            if st == 400:
                print("    HTTP 400 = Bad Request.  Most common causes:")
                print("    (a) API key / secret / passphrase is expired or invalid")
                print("        → Check POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE are current")
                print("    (b) HMAC signature format rejected by server")
                print("        → Check raw_response above for server error message")
                print("    (c) Server-side API change (new required parameter)")
                print("        → Check raw_response above for parameter hint")
            elif st == 401:
                print("    HTTP 401 = Unauthorized — credentials not accepted")
                print("    → Rotate API key on Polymarket dashboard and update env vars")
            elif st == 403:
                print("    HTTP 403 = Forbidden — key valid but insufficient permissions")
            elif st == -1:
                print("    HTTP -1 = Network error — check connectivity to clob.polymarket.com")
        if user_markets_fetch.first_entry_keys:
            print(f"  first entry keys : {user_markets_fetch.first_entry_keys}")
        print()

    # -----------------------------------------------------------------------
    # Step 4: Per-survivor presence diagnosis
    # -----------------------------------------------------------------------
    _section("Step 4: Per-Survivor Presence Diagnosis")
    diagnoses: list[PresenceDiagnosis] = []

    with httpx.Client() as http_client:
        for slug in args.slugs:
            market = market_by_slug.get(slug)

            # Resolve reward params: official > discovery > fallback
            if market is not None:
                off_cfg = official_by_slug.get(slug)
                if off_cfg and off_cfg.official_rewards_max_spread is not None:
                    spread_cents = off_cfg.official_rewards_max_spread
                else:
                    spread_cents = market.rewards_max_spread_cents or 4.0

                if off_cfg and off_cfg.official_rewards_min_size is not None:
                    min_size = off_cfg.official_rewards_min_size
                else:
                    min_size = market.rewards_min_size or 10.0

                token_id     = market.yes_token_id
                condition_id = market.market_id
            else:
                fp           = _FALLBACK_REWARD_PARAMS.get(slug, {})
                spread_cents = fp.get("rewards_max_spread_cents", 4.0)
                min_size     = fp.get("rewards_min_size", 10.0)
                token_id     = ""
                condition_id = ""

            print(
                f"  Diagnosing: {slug[:50]}  "
                f"spread={spread_cents:.1f}c  minsize={min_size:.0f}sh"
            )

            diag = build_presence_diagnosis(
                slug=slug,
                condition_id=condition_id,
                token_id=token_id,
                rewards_max_spread_cents=spread_cents,
                rewards_min_size=min_size,
                user_markets_fetch=user_markets_fetch,
                host=args.clob_host,
                creds=creds,
                http_client=http_client,
                wallet_address=creds.wallet_address if creds else None,
            )
            diagnoses.append(diag)
            print(
                f"    -> {diag.diagnosis_code:<22}  "
                f"presence={diag.has_active_presence}  "
                f"scoring={diag.scoring_status}  "
                f"judgment={diag.judgment}"
            )

    print()

    # -----------------------------------------------------------------------
    # Step 5: Summary table
    # -----------------------------------------------------------------------
    _section("Step 5: Summary Table")
    _print_summary_table(diagnoses)
    print()

    # -----------------------------------------------------------------------
    # Step 6: Per-survivor detail
    # -----------------------------------------------------------------------
    _section("Step 6: Per-Survivor Detail")
    for d in diagnoses:
        _print_per_survivor_detail(d, verbose=args.verbose)

    # -----------------------------------------------------------------------
    # Step 7: Final judgment
    # -----------------------------------------------------------------------
    _print_final_judgment(diagnoses)

    # -----------------------------------------------------------------------
    # Credential guidance (when creds absent and verbose)
    # -----------------------------------------------------------------------
    if creds is None and args.verbose:
        _section("Credential Setup Guidance")
        print(CREDENTIAL_GUIDANCE)

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------
    run_ts = datetime.now(timezone.utc).isoformat()
    results_out = {
        "run_timestamp": run_ts,
        "line": "reward_aware_maker_presence_scoring_line",
        "credentials_present": creds is not None,
        "user_markets_entries": (
            user_markets_fetch.total_entries
            if user_markets_fetch else None
        ),
        "slugs": args.slugs,
        "diagnoses": {d.slug: d.to_dict() for d in diagnoses},
    }

    output_path = args.output
    if output_path is None:
        PRESENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_path = PRESENCE_DATA_DIR / f"maker_presence_scoring_{ts_tag}.json"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_out, f, indent=2, default=str)
        print(f"  JSON written: {output_path}")
    except Exception as exc:
        print(f"  [WARN] JSON write failed: {exc}")

    print()
    print("reward_aware_maker_presence_scoring_line")


if __name__ == "__main__":
    main()
