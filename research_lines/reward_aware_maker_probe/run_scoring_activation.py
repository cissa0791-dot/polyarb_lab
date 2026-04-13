"""
reward_aware_scoring_activation_line — CLI
polyarb_lab / research_line / validation

Tests whether qualifying bilateral maker quotes activate reward scoring
on the 3 FOUND survivors (hungary, vance, rubio).

Context:
  maker_presence_scoring_line result (2026-03-26):
    - hungary, vance, rubio: NO_ACTIVE_PRESENCE (null maker_address)
    - netanyahu: NOT_IN_USER_MARKETS
    - All 0% earning explained by non-participation
    - Market reward thesis intact

  This line answers: "Can scoring activate with qualifying bilateral participation?"

  Scoring truth sources:
    1. is_order_scoring(order_id)      — immediate per-order CLOB state
    2. /rewards/user/markets earn%     — account-level (lags by period)

  CONTINUE if scoring activates (even if earning% lags)
  NOT_SCORING warrants 24h re-observation, not immediate DOWNGRADE
  DOWNGRADE only if: scoring confirmed + earning% = 0% after 24h+ (competitive exclusion)

Default target: hungary (lowest competitiveness=46.9, rate $150/day, implied share ~81%)
  Condition ID : 0x1480b819d03d4b6388d70e848b0384adf38c38d955cb783cdbcf6d4a436dee14
  Token ID     : 94192784911459194325909253314484842244405314804074606736702592885535642919725
  YES price ref: 0.625  (62.5% as of 2026-03-26)

  Survivor competitiveness ranking (lower = less competition = better share):
    hungary : 46.9  → implied share ~81%  ← PRIMARY TEST TARGET
    rubio   : 104.1 → implied share ~66%  ← backup
    vance   : 576.9 → implied share ~26%  ← highest volume but most competed

Qualifying bilateral quote (auto-computed from live midpoint):
  Spread  : 3 cents (within 3.5-cent max)
  Size    : max(10, rewards_min_size) shares per side
  Type    : GTC  (live in book until cancelled)

Modes:
  --dry-run (default): compute and print qualifying quotes, no order submission
  --live              : submit actual orders, observe scoring, cancel after window

Credential env vars:
  POLYMARKET_PRIVATE_KEY       EVM private key for L1 order signing
  POLYMARKET_API_KEY           CLOB API key
  POLYMARKET_API_SECRET        CLOB API secret
  POLYMARKET_API_PASSPHRASE    CLOB passphrase
  POLYMARKET_CHAIN_ID          137 for Polygon mainnet (default if omitted)

  Fallback (research line vars also accepted):
  POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE
  POLY_WALLET_ADDRESS  (used as private key if 64 hex chars)

Usage (Windows PowerShell from repo root):
  # Step 1: dry-run to verify quotes before submitting
  py -3 research_lines/reward_aware_maker_probe/run_scoring_activation.py

  # Step 2: live submission with 30-min observation
  py -3 research_lines/reward_aware_maker_probe/run_scoring_activation.py --live

  # Target specific survivor
  py -3 research_lines/reward_aware_maker_probe/run_scoring_activation.py --target hungary --live

  # Extend observation window
  py -3 research_lines/reward_aware_maker_probe/run_scoring_activation.py --live --observe-minutes 60

STRICT RULES:
  - No profitability claim.
  - CONTINUE verdict does not mean execution approval.
  - Minimum size (10 shares) to limit directional exposure.
  - Orders automatically cancelled after observation window (unless --skip-cancel).
  - Results written to data/research/reward_aware_maker_probe/scoring_activation/ only.
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

from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
    SLUG_ALIASES,
    SURVIVOR_DATA,
    VERDICT_DRY_RUN,
    VERDICT_PREFLIGHT_FAILED,
    VERDICT_SCORING_ACTIVE,
    VERDICT_NOT_SCORING,
    VERDICT_INCONCLUSIVE,
    ActivationResult,
    load_activation_credentials,
    get_missing_credential_vars,
    run_activation_test,
)

CLOB_HOST  = "https://clob.polymarket.com"
OUTPUT_DIR = Path("data/research/reward_aware_maker_probe/scoring_activation")

DEFAULT_TARGET        = "hungary"
DEFAULT_OBSERVE_MIN   = 30
DEFAULT_POLL_MIN      = 5


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _section(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.4f}%"


def _print_preflight(result: ActivationResult, dry_run: bool) -> None:
    _section("Pre-flight: Market & Quote Parameters")
    print(f"  slug           : {result.slug}")
    print(f"  condition_id   : {result.condition_id}")
    print(f"  token_id       : {result.token_id[:20]}...")
    print()
    print(f"  Reward config source : {result.reward_config.source}")
    print(f"    max_spread         : {result.reward_config.max_spread_cents:.1f} cents "
          f"({result.reward_config.max_spread_fraction:.4f} price units)")
    print(f"    min_size           : {result.reward_config.min_size:.0f} shares")
    print(f"    daily_rate_usdc    : ${result.reward_config.daily_rate_usdc:.2f}")
    print()
    print(f"  Midpoint : {result.midpoint} (source={result.midpoint_source})")
    print()

    if result.quotes:
        q = result.quotes
        print(f"  Qualifying bilateral quotes:")
        print(f"    BID  : {q.bid_price:.4f}  ({q.bid_price * 100:.2f}¢)")
        print(f"    ASK  : {q.ask_price:.4f}  ({q.ask_price * 100:.2f}¢)")
        print(f"    spread       : {q.spread * 100:.2f}¢  "
              f"(max={q.max_spread_fraction * 100:.1f}¢ — {'' if q.spread_ok else 'EXCEEDS MAX!'} OK)")
        print(f"    size per side: {q.size:.0f} shares")
        print(f"    qualifying   : {q.qualifying}")
        print()
        print(f"  Capital at risk (BID side only): "
              f"${q.bid_price * q.size:.2f}  "
              f"(ask side: ${q.ask_price * q.size:.2f})")
        print(f"  Max directional loss if both filled immediately: "
              f"~${q.spread * q.size:.3f}  (spread × size)")
    print()


def _print_observation(result: ActivationResult) -> None:
    _section("Scoring Observation Log")
    if not result.observations:
        print("  No observations recorded.")
        return

    header = f"  {'elapsed_min':>11}  {'bid_scoring':>11}  {'ask_scoring':>11}  " \
             f"{'both':>5}  {'earn%':>8}"
    print(header)
    print(f"  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*5}  {'-'*8}")
    for o in result.observations:
        print(
            f"  {o.elapsed_minutes:>11.1f}  "
            f"{'True' if o.bid_scoring else ('False' if o.bid_scoring is False else 'N/A'):>11}  "
            f"{'True' if o.ask_scoring else ('False' if o.ask_scoring is False else 'N/A'):>11}  "
            f"{'YES' if o.both_scoring else 'no':>5}  "
            f"{_pct(o.earning_pct):>8}"
        )
    print()


def _print_orders(result: ActivationResult) -> None:
    if result.bid_order or result.ask_order:
        _section("Order State")
        for order in [result.bid_order, result.ask_order]:
            if order is None:
                continue
            print(f"  {order.side:<4}  id={order.order_id}")
            print(f"        price={order.price:.4f}  size={order.size:.0f}")
            print(f"        filled={order.filled}  cancelled={order.cancelled}"
                  f"  cancel_ok={order.cancel_ok}")
        print()


def _print_final_judgment(result: ActivationResult) -> None:
    _section("FINAL JUDGMENT: SCORING ACTIVATION LINE")

    verdict_labels = {
        VERDICT_SCORING_ACTIVE  : "SCORING_ACTIVE  ← scoring confirmed",
        VERDICT_NOT_SCORING     : "NOT_SCORING     ← 24h re-observation required",
        VERDICT_INCONCLUSIVE    : "INCONCLUSIVE    ← order(s) filled; re-run required",
        VERDICT_DRY_RUN         : "DRY_RUN         ← no orders submitted",
        VERDICT_PREFLIGHT_FAILED: "PREFLIGHT_FAILED← see verdict_detail",
    }
    label = verdict_labels.get(result.verdict, result.verdict)
    print(f"  VERDICT: {label}")
    print()

    # Word-wrap detail
    words = result.verdict_detail.split()
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

    print("  GOVERNANCE REMINDERS:")
    print("  - CONTINUE verdict does not mean execution approval")
    print("  - 30-min observation may not span a full reward period")
    print("  - NOT_SCORING after 30min warrants 24h re-observation")
    print("  - DOWNGRADE only if scoring confirmed + earning=0% after 24h+")
    print("  - Scoring activation ≠ profitability; reward share fraction is unconfirmed")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "reward_aware_scoring_activation_line\n"
            "Tests whether qualifying bilateral maker quotes activate reward scoring.\n"
            "Default: --dry-run (no order submission).\n"
            "Use --live to submit actual orders."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET,
        choices=list(SLUG_ALIASES.keys()) + list(SURVIVOR_DATA.keys()),
        help=(
            f"Target survivor to test. Aliases: {list(SLUG_ALIASES.keys())}. "
            f"Default: {DEFAULT_TARGET!r}"
        ),
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Submit actual orders. Without this flag, --dry-run mode (no orders submitted).",
    )
    parser.add_argument(
        "--observe-minutes", type=int, default=DEFAULT_OBSERVE_MIN,
        help=f"Minutes to observe scoring after order placement (default: {DEFAULT_OBSERVE_MIN})",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=DEFAULT_POLL_MIN,
        help=f"Polling interval in minutes (default: {DEFAULT_POLL_MIN})",
    )
    parser.add_argument(
        "--skip-cancel", action="store_true",
        help=(
            "Do not cancel orders after observation window. "
            "Use only if you intend to manage orders manually."
        ),
    )
    parser.add_argument(
        "--spread-cents", type=float, default=3.0,
        help="Quote spread in cents (default: 3.0; max allowed by reward config: 3.5)",
    )
    parser.add_argument(
        "--clob-host", default=CLOB_HOST,
        help="CLOB base URL",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write JSON result to this path (default: auto-timestamped in output dir)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    dry_run = not args.live

    # Resolve slug alias
    slug = SLUG_ALIASES.get(args.target, args.target)

    print()
    _section("reward_aware_scoring_activation_line")
    print(f"  Timestamp      : {datetime.now(timezone.utc).isoformat()}")
    print(f"  Target         : {args.target} → {slug[:55]}")
    print(f"  Mode           : {'DRY-RUN (no order submission)' if dry_run else 'LIVE — orders will be submitted'}")
    if not dry_run:
        print(f"  Observe window : {args.observe_minutes} min")
        print(f"  Poll interval  : {args.poll_interval} min")
        print(f"  Skip cancel    : {args.skip_cancel}")
    print()

    if not dry_run:
        print("  *** LIVE MODE: qualifying bilateral orders WILL be submitted ***")
        print("  *** Orders will be cancelled after observation window         ***")
        print()

    # ── Credentials ──────────────────────────────────────────────────────
    _section("Step 1: Credential Check")
    creds = load_activation_credentials()
    missing = get_missing_credential_vars()

    if creds is None:
        # Only fires when POLYMARKET_PRIVATE_KEY is absent entirely.
        print("  STATUS: PRIVATE KEY NOT AVAILABLE")
        if missing:
            print(f"  Missing: {missing}")
        print()
        print("  Required env var:")
        print("    POLYMARKET_PRIVATE_KEY  — EVM private key (hex, with or without 0x)")
        print()
        print("  Optional (auto-derived from private key if absent):")
        print("    POLYMARKET_API_KEY / POLYMARKET_API_SECRET / POLYMARKET_API_PASSPHRASE")
        print("    POLYMARKET_CHAIN_ID     — default 137")
        print("    POLYMARKET_SIGNATURE_TYPE — default 0 (EOA); use 2 for proxy wallet")
        print("    POLYMARKET_FUNDER       — proxy wallet address (required for sig_type=2)")
        print()
        if not dry_run:
            print("  ABORT: --live mode requires POLYMARKET_PRIVATE_KEY.")
            sys.exit(1)
        print("  Continuing in dry-run mode without credentials.")
        print("  Midpoint will be estimated (0.50); reward params from fallback.")
        print()
        # Proceed without creds for dry-run only — stub private key, no API calls made
        from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
            ActivationCredentials,
        )
        creds = ActivationCredentials(
            private_key="0x" + "0" * 64,
            api_key="stub",
            api_secret="stub",
            api_passphrase="stub",
            chain_id=137,
            needs_api_derivation=True,
            credential_source="stub_dry_run",
        )
    else:
        # creds loaded — API trio may be configured or will be derived automatically.
        print("  STATUS: CREDENTIALS LOADED")
        print(f"  private_key       : SET (len={len(creds.private_key)})")
        print(f"  needs_derivation  : {creds.needs_api_derivation}")
        if creds.api_key:
            print(f"  api_key (config)  : {creds.api_key[:8]}...")
        else:
            print("  api_key (config)  : (absent — will derive from private key)")
        print(f"  chain_id          : {creds.chain_id}")
        print(f"  signature_type    : {creds.signature_type}  "
              f"({'EOA' if creds.signature_type == 0 else 'POLY_PROXY' if creds.signature_type == 1 else 'POLY_GNOSIS_SAFE'})")
        print(f"  funder            : {creds.funder!r}")
        print()
        print("  API trio resolution: build_clob_client() calls derive_api_key() first.")
        print("  If configured key mismatches signer EOA, derived key is used automatically.")
        print("  credential_source and effective_api_key are printed at client build time.")
        print()

    # ── Step 1.5: External metadata verification (Dome — observability only) ──
    # Runs only when DOME_API_KEY is set.  Silent skip otherwise.
    # Prints Dome's condition_id and token IDs next to the hardcoded
    # SURVIVOR_DATA values for alignment verification.
    # Result is never used downstream — no variable is modified.
    import os as _os
    if _os.environ.get("DOME_API_KEY"):
        _section("Step 1.5: External Market Verification (Dome — observability only)")
        try:
            from src.external.dome_market_lookup import lookup as _dome_lookup
            from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
                SURVIVOR_DATA as _SURVIVOR_DATA,
            )
            _dome = _dome_lookup(slug)
            _local = _SURVIVOR_DATA.get(slug, {})
            print(f"  slug           : {slug}")
            print(f"  dome source    : {_dome['source']}")
            if _dome["error"]:
                print(f"  dome error     : {_dome['error']}")
            else:
                _cid_match = (
                    _dome["condition_id"] == _local.get("condition_id")
                    if _local.get("condition_id") else "local_not_found"
                )
                _sid_match = (
                    _dome["side_a_id"] == _local.get("token_id")
                    if _local.get("token_id") else "local_not_found"
                )
                print(f"  dome condition_id : {_dome['condition_id']}")
                print(f"  local condition_id: {_local.get('condition_id', '(not in SURVIVOR_DATA)')}")
                print(f"  condition_id_match: {_cid_match}")
                print()
                print(f"  dome side_a_id    : {_dome['side_a_id']}")
                print(f"  dome side_b_id    : {_dome['side_b_id']}")
                print(f"  local token_id    : {_local.get('token_id', '(not in SURVIVOR_DATA)')}")
                print(f"  side_a==token_id  : {_sid_match}")
                print(f"  dome status       : {_dome['status']}")
            print()
            print("  [observability only — no execution path modified]")
        except Exception as _dome_exc:
            print(f"  [WARN] Dome lookup failed (non-fatal): {_dome_exc}")
        print()

    # ── Run activation test ──────────────────────────────────────────────
    _section("Step 2: Preflight — Midpoint + Reward Config + Quote Computation")
    result = run_activation_test(
        slug=slug,
        creds=creds,
        host=args.clob_host,
        dry_run=dry_run,
        observe_minutes=args.observe_minutes,
        poll_interval_minutes=args.poll_interval,
        skip_cancel=args.skip_cancel,
        spread_cents=args.spread_cents,
    )

    _print_preflight(result, dry_run)

    if result.verdict == VERDICT_PREFLIGHT_FAILED:
        print(f"  PREFLIGHT FAILED: {result.verdict_detail}")
        print()
    else:
        if not dry_run:
            if result.earning_pct_before is not None:
                print(f"  earning_pct BEFORE orders : {_pct(result.earning_pct_before)}")
                print()

            _section("Step 3: Order Placement + Observation")
            _print_orders(result)
            _print_observation(result)

            if result.earning_pct_after is not None:
                print(f"  earning_pct AFTER observation : {_pct(result.earning_pct_after)}")
                print()

            print(f"  orders_cancelled : {result.orders_cancelled}")
            print()

    _print_final_judgment(result)

    # ── JSON output ──────────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_path = OUTPUT_DIR / f"scoring_activation_{ts_tag}.json"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"  JSON written: {output_path}")
    except Exception as exc:
        print(f"  [WARN] JSON write failed: {exc}")

    print()
    print("reward_aware_maker_probe")


if __name__ == "__main__":
    main()
