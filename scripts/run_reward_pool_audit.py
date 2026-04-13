"""
run_reward_pool_audit.py — Reward pool / Q-score capture audit for anchor markets.

Purpose
-------
Surfaces the largest remaining uncertainty in the maker-MM EV model:
competition_factor / reward-share capture.  The model assumes a competition_factor
(fraction of the reward pool we expect to capture).  This script makes that
assumption observable by computing our modeled Q-score vs the visible book depth
Q-scores.

What is observable (no auth needed):
  - Gamma: rewardsDailyRate, rewardsMinSize, rewardsMaxSpread, clob_rewards
  - CLOB:  full orderbook depth (all resting bid/ask levels + sizes)
  - Q-score formula (official Polymarket): S(v,s) = ((v-s)/v)^2 * b
      v = max_spread (cents), s = |price - mid| * 100, b = order size (shares)

What this computes:
  - Our modeled Q-score for each anchor market
  - Sum of visible book Q-scores within reward zone (eligible competitors)
  - Implied pool-share upper bound: our_Q / (our_Q + competitor_Q_sum)
  - Comparison against the competition_factor assumed by compute_maker_mm_ev()

What remains unobservable:
  - Which book orders are from MMs vs position-takers (all sizes counted)
  - Book orders placed AFTER our snapshot (stale book = stale competitor estimate)
  - Actual on-chain reward payouts (requires live orders + Polymarket API auth)
  - Multi-session TWAP of book depth (single snapshot is noisy)

Usage
-----
    python scripts/run_reward_pool_audit.py
    python scripts/run_reward_pool_audit.py --events next-prime-minister-of-hungary netanyahu-out-before-2027
    python scripts/run_reward_pool_audit.py --json   # machine-readable output

Isolation: read-only, no DB writes, no network mutation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.gamma import fetch_events, fetch_markets
from src.ingest.clob import ReadOnlyClob
from src.intelligence.market_intelligence import build_event_market_registry
from scripts.run_wide_maker_scan import compute_maker_mm_ev
from src.scanner.maker_scan_quote_planner import (
    plan_quote,
    MAKER_RISK_AVERSION,
    MAKER_K,
    PRIOR_BELIEF_VAR,
)

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST  = "https://clob.polymarket.com"

DEFAULT_EVENTS = ["next-prime-minister-of-hungary"]


# ---------------------------------------------------------------------------
# Q-score formula (official Polymarket)
# ---------------------------------------------------------------------------

def q_score(price: float, mid_p: float, size: float, max_spread_cents: float) -> float:
    """
    Official Polymarket Q-score: S(v,s) = ((v-s)/v)^2 * b
    v = max_spread_cents, s = |price - mid_p| * 100, b = size (shares).
    Returns 0 if s >= v (outside reward zone).
    """
    s = abs(price - mid_p) * 100.0
    if s >= max_spread_cents or max_spread_cents <= 0:
        return 0.0
    return ((max_spread_cents - s) / max_spread_cents) ** 2 * size


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardPoolAuditResult:
    event_slug           : str
    market_slug          : str
    question             : str

    # Reward config from Gamma
    reward_daily_rate    : float   # USDC/day (pool size)
    max_spread_cents     : float   # reward eligibility zone
    min_size_shares      : int     # minimum order size for eligibility
    rewards_amount       : float   # clob_rewards.rewardsAmount (usually 0)
    reward_start_date    : str
    reward_end_date      : str

    # Live book
    best_bid             : float
    best_ask             : float
    mid_p                : float
    book_spread_cents    : float
    bid_levels           : int     # total bid levels in book
    ask_levels           : int     # total ask levels in book

    # Our modeled quote
    our_quote_bid        : float
    our_quote_ask        : float
    our_quote_size       : float
    our_bid_spread_cents : float
    our_ask_spread_cents : float
    our_q_bid            : float   # Q-score of our resting bid
    our_q_ask            : float   # Q-score of our resting ask

    # Competitor book Q-scores (all visible orders within reward zone)
    competitor_q_bid_sum : float   # sum of Q-scores of all bids within max_spread
    competitor_q_ask_sum : float   # sum of Q-scores of all asks within max_spread
    eligible_bid_orders  : int     # book bid orders within max_spread
    eligible_ask_orders  : int     # book ask orders within max_spread

    # Implied pool-share bounds
    implied_share_bid    : float   # our_q_bid / (our_q_bid + competitor_q_bid_sum)
    implied_share_ask    : float   # our_q_ask / (our_q_ask + competitor_q_ask_sum)
    implied_share_avg    : float   # average of bid/ask share

    # Model assumptions vs observed
    modeled_competition_factor : float   # Q-multiplier from compute_maker_mm_ev (total_Q = our_Q × cf)
    modeled_pool_share         : float   # 1 / competition_factor (fraction of reward pool)
    modeled_reward_ev          : float   # USDC/day assumed by EV model
    modeled_total_ev           : float

    # Gap summary
    share_model_vs_implied : float   # modeled_pool_share - implied_share_avg
    gap_direction          : str     # "OVERESTIMATED" / "UNDERESTIMATED" / "WITHIN_RANGE"

    # What remains unobservable
    unobservable_note      : str


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

def audit_market(
    m         : dict[str, Any],
    book_bids : list,   # list of (price, size) from CLOB
    book_asks : list,
) -> RewardPoolAuditResult:
    slug        = m["market_slug"]
    ev_slug     = m["event_slug"]
    best_bid    = m["best_bid"]
    best_ask    = m["best_ask"]
    mid_p       = (best_bid + best_ask) / 2.0
    max_sp      = m["rewards_max_spread"]   # cents
    min_sz      = int(m["rewards_min_size"])
    rate        = m["reward_daily_rate"]

    # Our modeled quote
    quote_size = max(float(min_sz), 20.0)
    m_plan = dict(m)
    m_plan["rewards_min_size"] = quote_size
    plan = plan_quote(m_plan)

    our_q_bid = q_score(plan.quote_bid, mid_p, quote_size, max_sp)
    our_q_ask = q_score(plan.quote_ask, mid_p, quote_size, max_sp)

    # Competitor Q-scores from visible book
    comp_q_bid   = 0.0
    elig_bids    = 0
    for price, size in book_bids:
        qs = q_score(price, mid_p, size, max_sp)
        if qs > 0:
            comp_q_bid += qs
            elig_bids  += 1

    comp_q_ask   = 0.0
    elig_asks    = 0
    for price, size in book_asks:
        qs = q_score(price, mid_p, size, max_sp)
        if qs > 0:
            comp_q_ask += qs
            elig_asks  += 1

    # Implied pool-share (treat all visible book volume as competitor)
    implied_bid = our_q_bid / (our_q_bid + comp_q_bid) if (our_q_bid + comp_q_bid) > 0 else 0.0
    implied_ask = our_q_ask / (our_q_ask + comp_q_ask) if (our_q_ask + comp_q_ask) > 0 else 0.0
    implied_avg = (implied_bid + implied_ask) / 2.0

    # EV model assumptions
    ev = compute_maker_mm_ev(
        best_bid                  = best_bid,
        best_ask                  = best_ask,
        rewards_min_size          = min_sz,
        rewards_max_spread_cents  = max_sp,
        reward_daily_rate         = rate,
        volume_num                = float(m.get("volume_num") or 0),
    )

    # competition_factor is a Q-multiplier: estimated_total_q = our_q * cf.
    # Modeled pool share = 1 / cf (fraction of reward pool we expect to capture).
    mod_cf = ev["competition_factor"]
    modeled_pool_share = 1.0 / mod_cf if mod_cf > 0 else 0.0

    # Gap: positive = model assumes more share than visible book implies
    gap = modeled_pool_share - implied_avg
    if abs(gap) < 0.01:
        direction = "WITHIN_RANGE"
    elif gap > 0:
        direction = "OVERESTIMATED"   # model assumes more pool share than observable
    else:
        direction = "UNDERESTIMATED"

    clob_cr = m.get("clob_rewards") or [{}]
    cr0 = clob_cr[0] if clob_cr else {}

    return RewardPoolAuditResult(
        event_slug           = ev_slug,
        market_slug          = slug,
        question             = str(m.get("question") or ""),

        reward_daily_rate    = rate,
        max_spread_cents     = max_sp,
        min_size_shares      = min_sz,
        rewards_amount       = float(cr0.get("rewardsAmount") or 0),
        reward_start_date    = str(cr0.get("startDate") or ""),
        reward_end_date      = str(cr0.get("endDate") or ""),

        best_bid             = best_bid,
        best_ask             = best_ask,
        mid_p                = mid_p,
        book_spread_cents    = (best_ask - best_bid) * 100.0,
        bid_levels           = len(book_bids),
        ask_levels           = len(book_asks),

        our_quote_bid        = round(plan.quote_bid, 6),
        our_quote_ask        = round(plan.quote_ask, 6),
        our_quote_size       = quote_size,
        our_bid_spread_cents = round(plan.bid_spread_cents, 3),
        our_ask_spread_cents = round(plan.ask_spread_cents, 3),
        our_q_bid            = round(our_q_bid, 2),
        our_q_ask            = round(our_q_ask, 2),

        competitor_q_bid_sum = round(comp_q_bid, 2),
        competitor_q_ask_sum = round(comp_q_ask, 2),
        eligible_bid_orders  = elig_bids,
        eligible_ask_orders  = elig_asks,

        implied_share_bid    = round(implied_bid, 4),
        implied_share_ask    = round(implied_ask, 4),
        implied_share_avg    = round(implied_avg, 4),

        modeled_competition_factor = round(mod_cf, 4),
        modeled_pool_share         = round(modeled_pool_share, 4),
        modeled_reward_ev          = round(ev["reward_ev"], 4),
        modeled_total_ev           = round(ev["total_ev"], 4),

        share_model_vs_implied = round(gap, 4),
        gap_direction          = direction,

        unobservable_note = (
            "Book orders counted as 100% competitor volume (conservative upper bound). "
            "True MM-only Q-score pool requires on-chain reward data. "
            "Position-taker orders resting in book inflate competitor_Q and understate our share."
        ),
    )


# ---------------------------------------------------------------------------
# Fetch + run
# ---------------------------------------------------------------------------

def run_audit(event_slugs: list[str], as_json: bool = False) -> list[RewardPoolAuditResult]:
    print(f"REWARD POOL / Q-SCORE AUDIT")
    print(f"Events: {event_slugs}")
    print(f"CLOB Q-score formula: S(v,s) = ((v-s)/v)^2 * b  (official Polymarket)\n")

    # 1. Fetch Gamma registry
    events  = fetch_events(GAMMA_HOST,  limit=500)
    markets = fetch_markets(GAMMA_HOST, limit=500)
    registry = build_event_market_registry(events, markets)

    slug_set = set(event_slugs)
    eligible: list[dict] = []

    for ev in registry.get("events", []):
        if ev.get("slug") not in slug_set:
            continue
        for m in ev.get("markets", []):
            rewards = m.get("clob_rewards") or []
            rate    = sum(float(r.get("rewardsDailyRate", 0) or 0) for r in rewards)
            if rate <= 0:
                continue
            min_sz  = float(m.get("rewards_min_size") or 0)
            max_sp  = float(m.get("rewards_max_spread") or 0)
            bid     = float(m.get("best_bid") or 0)
            ask     = float(m.get("best_ask") or 0)
            if not (min_sz > 0 and max_sp > 0 and bid > 0 and ask > bid):
                continue
            yes_tok = m.get("yes_token_id")
            no_tok  = m.get("no_token_id")
            if not yes_tok or not no_tok:
                continue
            eligible.append({
                "event_slug":        ev.get("slug"),
                "market_slug":       m.get("slug"),
                "question":          m.get("question"),
                "yes_token_id":      str(yes_tok),
                "best_bid":          bid,
                "best_ask":          ask,
                "rewards_min_size":  min_sz,
                "rewards_max_spread": max_sp,
                "reward_daily_rate": rate,
                "volume_num":        float(m.get("volume_num") or 0),
                "clob_rewards":      rewards,
                "neg_risk":          bool(m.get("neg_risk")),
            })

    print(f"Found {len(eligible)} reward-eligible markets\n")
    if not eligible:
        return []

    # 2. Fetch CLOB books
    clob      = ReadOnlyClob(CLOB_HOST)
    yes_toks  = [m["yes_token_id"] for m in eligible]
    books     = clob.prefetch_books(yes_toks, max_workers=8)

    results: list[RewardPoolAuditResult] = []

    for m in eligible:
        tok  = m["yes_token_id"]
        book = books.get(tok)
        book_bids = [(b.price, b.size) for b in (book.bids if book else [])]
        book_asks = [(a.price, a.size) for a in (book.asks if book else [])]

        # Update best_bid/ask from live CLOB if available
        if book and book.bids and book.asks:
            m["best_bid"] = book.bids[0].price
            m["best_ask"] = book.asks[0].price

        result = audit_market(m, book_bids, book_asks)
        results.append(result)

    if as_json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return results

    # 3. Human-readable report
    sep = "=" * 72
    for r in results:
        print(sep)
        print(f"  {r.market_slug}")
        print(f"  {r.question[:70]}")
        print(sep)

        print(f"\n  REWARD CONFIG (Gamma)")
        print(f"    daily_rate_usdc   : ${r.reward_daily_rate:.2f}/day")
        print(f"    max_spread_cents  : {r.max_spread_cents:.1f}¢")
        print(f"    min_size_shares   : {r.min_size_shares}")
        print(f"    rewards_amount    : {r.rewards_amount}  (0 = pool not yet distributed)")
        print(f"    reward_window     : {r.reward_start_date} → {r.reward_end_date}")

        print(f"\n  LIVE BOOK")
        print(f"    bid/ask           : {r.best_bid:.4f} / {r.best_ask:.4f}")
        print(f"    mid               : {r.mid_p:.4f}")
        print(f"    book spread       : {r.book_spread_cents:.2f}¢")
        print(f"    book depth        : {r.bid_levels} bid levels, {r.ask_levels} ask levels")

        print(f"\n  OUR MODELED QUOTE (γ={MAKER_RISK_AVERSION}, k={MAKER_K}, σ²={PRIOR_BELIEF_VAR})")
        print(f"    quote bid / ask   : {r.our_quote_bid:.4f} / {r.our_quote_ask:.4f}")
        print(f"    bid spread        : {r.our_bid_spread_cents:.2f}¢  (from mid)")
        print(f"    ask spread        : {r.our_ask_spread_cents:.2f}¢  (from mid)")
        print(f"    quote size        : {r.our_quote_size:.0f} shares")
        print(f"    Q-score bid/ask   : {r.our_q_bid:.1f} / {r.our_q_ask:.1f}")

        print(f"\n  COMPETITOR Q-SCORES (all visible book volume within {r.max_spread_cents}¢ zone)")
        print(f"    eligible bid orders : {r.eligible_bid_orders}  (Q-sum={r.competitor_q_bid_sum:.0f})")
        print(f"    eligible ask orders : {r.eligible_ask_orders}  (Q-sum={r.competitor_q_ask_sum:.0f})")
        print(f"    NOTE: counts all book orders as competitor (conservative upper bound)")

        print(f"\n  POOL-SHARE ESTIMATE")
        print(f"    implied share bid : {r.implied_share_bid:.4f}  ({r.implied_share_bid*100:.2f}%)")
        print(f"    implied share ask : {r.implied_share_ask:.4f}  ({r.implied_share_ask*100:.2f}%)")
        print(f"    implied share avg : {r.implied_share_avg:.4f}  ({r.implied_share_avg*100:.2f}%)")

        print(f"\n  EV MODEL ASSUMPTIONS vs OBSERVED")
        print(f"    competition_factor (Q-mult): {r.modeled_competition_factor:.1f}×  (total pool Q = our Q × cf)")
        print(f"    modeled pool share         : {r.modeled_pool_share:.4f}  ({r.modeled_pool_share*100:.2f}%)  [= 1/cf]")
        print(f"    implied pool share         : {r.implied_share_avg:.4f}  ({r.implied_share_avg*100:.2f}%)  [from book depth]")
        print(f"    gap (model - implied)      : {r.share_model_vs_implied:+.4f}  → {r.gap_direction}")
        print(f"    modeled reward_ev          : ${r.modeled_reward_ev:.4f}/day")
        print(f"    modeled total_ev           : ${r.modeled_total_ev:.4f}/day")

        # Recalculate reward_ev using implied share
        implied_reward_ev = r.reward_daily_rate * r.implied_share_avg
        print(f"    implied reward_ev          : ${implied_reward_ev:.4f}/day  (rate × implied_share)")
        ev_adj = implied_reward_ev - r.modeled_reward_ev
        print(f"    reward EV adjustment       : {ev_adj:+.4f}/day")

        print(f"\n  WHAT REMAINS UNOBSERVABLE")
        print(f"    {r.unobservable_note}")
        print()

    print(sep)
    print(f"SUMMARY: {len(results)} markets audited")
    overest = sum(1 for r in results if r.gap_direction == "OVERESTIMATED")
    underest = sum(1 for r in results if r.gap_direction == "UNDERESTIMATED")
    within = sum(1 for r in results if r.gap_direction == "WITHIN_RANGE")
    print(f"  OVERESTIMATED  (model > implied): {overest}")
    print(f"  WITHIN_RANGE   (|gap| < 5%):      {within}")
    print(f"  UNDERESTIMATED (model < implied): {underest}")
    print(sep)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reward pool / Q-score audit")
    parser.add_argument(
        "--events", nargs="+", default=DEFAULT_EVENTS,
        help="Event slugs to audit (default: Hungary)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of human report",
    )
    args = parser.parse_args()
    run_audit(args.events, as_json=args.json)


if __name__ == "__main__":
    main()
