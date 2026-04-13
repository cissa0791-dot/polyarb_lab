"""
Step 1: Wide maker-MM scan + fill-realism stress testing.

Bypasses the watchlist bottleneck and directly scans ALL reward-eligible
markets from the Gamma API with live orderbook data.

Usage:
    python scripts/run_wide_maker_scan.py --market-limit 500 --top-events 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.gamma import fetch_events, fetch_markets
from src.ingest.clob import ReadOnlyClob
from src.intelligence.market_intelligence import build_event_market_registry


# ---------------------------------------------------------------------------
# Q-Score model (from Polymarket docs)
# S(v, s) = ((v - s) / v)^2 * b
# v = max spread, s = distance from midpoint, b = order size
# Two-sided = full score, one-sided = 1/3
# ---------------------------------------------------------------------------

def q_score_per_side(max_spread_cents: float, distance_from_mid_cents: float, size_shares: float) -> float:
    """Compute Q-score contribution for one side of a two-sided quote."""
    v = max_spread_cents
    s = distance_from_mid_cents
    if v <= 0 or s >= v:
        return 0.0
    return ((v - s) / v) ** 2 * size_shares


def estimate_daily_reward_share(
    reward_daily_rate: float,
    our_q_score: float,
    estimated_total_q_score: float,
) -> float:
    """Estimate our share of daily reward pool based on Q-score fraction."""
    if estimated_total_q_score <= 0:
        return 0.0
    return reward_daily_rate * (our_q_score / estimated_total_q_score)


def estimate_implied_reward_ev(
    daily_rate      : float,
    our_q_score     : float,
    competitor_q_sum: float,
) -> float:
    """
    Conservative reward EV using observed book depth as the competitor pool.

    Treats every order resting within the reward zone as a competing MM.
    This is a lower bound: position-taker orders inflate competitor_q, so
    the true MM pool share is likely higher than this estimate.

    Formula: daily_rate * our_q / (our_q + competitor_q_sum)

    Parameters
    ----------
    daily_rate       : reward pool size (USDC/day)
    our_q_score      : our two-sided Q-score (bid + ask)
    competitor_q_sum : sum of Q-scores of all book orders within max_spread zone

    Returns 0.0 if inputs are degenerate.
    """
    total = our_q_score + competitor_q_sum
    if total <= 0 or our_q_score <= 0:
        return 0.0
    return daily_rate * (our_q_score / total)


# ---------------------------------------------------------------------------
# EV model for maker-MM
# ---------------------------------------------------------------------------

def compute_maker_mm_ev(
    *,
    best_bid: float,
    best_ask: float,
    rewards_min_size: float,
    rewards_max_spread_cents: float,
    reward_daily_rate: float,
    volume_num: float = 0.0,
    liquidity_num: float = 0.0,
    fees_enabled: bool = False,
) -> dict[str, Any]:
    """
    Compute EV components for a two-sided maker quote.

    Revenue sources:
    1. Spread capture: when both sides fill, we capture the spread
    2. Liquidity rewards: Q-score based daily USDC payout

    Costs:
    1. Adverse selection: informed traders pick us off
    2. Inventory risk: one-sided fills leave directional exposure
    3. Cancel/replace friction
    4. Fees (if fee-enabled market)
    """
    current_spread = best_ask - best_bid
    max_spread = rewards_max_spread_cents / 100.0  # convert to probability units
    midpoint = (best_bid + best_ask) / 2.0

    # Our quote: tight to midpoint to maximize Q-score
    # Use half the max spread or current spread, whichever is tighter
    our_half_spread = min(current_spread / 2.0, max_spread / 2.0)
    our_half_spread = max(our_half_spread, 0.005)  # at least 0.5 cent

    quote_bid = round(max(0.01, midpoint - our_half_spread), 4)
    quote_ask = round(min(0.99, midpoint + our_half_spread), 4)
    quote_spread = quote_ask - quote_bid
    quote_size = max(rewards_min_size, 20.0)  # at least 20 shares

    # ---- Q-Score estimation ----
    distance_from_mid = our_half_spread * 100  # in cents
    our_q_score_per_side = q_score_per_side(rewards_max_spread_cents, distance_from_mid, quote_size)
    our_q_score = our_q_score_per_side * 2  # two-sided

    # Estimate competition: CONSERVATIVE — assume 10-30 makers with similar scores
    # Higher reward pools attract more competition; higher volume markets too.
    # Base: 10x (we capture ~10% of pool). High-reward markets: up to 30x.
    reward_competition = min(30.0, max(10.0, 10.0 + (reward_daily_rate / 20.0)))
    volume_competition = min(5.0, volume_num / 100000.0) if volume_num > 0 else 0.0
    competition_factor = reward_competition + volume_competition
    estimated_total_q = our_q_score * competition_factor

    reward_share = estimate_daily_reward_share(reward_daily_rate, our_q_score, estimated_total_q)

    # ---- Fill probability estimation ----
    # Based on volume and how tight our quotes are
    volume_factor = min(1.0, volume_num / 50000.0) if volume_num > 0 else 0.1
    tightness_factor = max(0.0, 1.0 - (distance_from_mid / rewards_max_spread_cents))

    # Probability of at least one fill per day (each side independently)
    fill_prob_per_side = min(0.90, max(0.05, 0.10 + 0.50 * volume_factor + 0.30 * tightness_factor))

    # ---- Revenue components ----
    # Spread capture: both sides fill -> capture full spread
    both_fill_prob = fill_prob_per_side ** 2 * 0.5  # correlation discount
    spread_capture_ev = both_fill_prob * quote_spread * quote_size

    # Reward income (our estimated share)
    reward_ev = reward_share

    # ---- Cost components ----
    # Adverse selection: informed traders hit our quotes, we lose on the fill
    # Higher for markets with more news flow
    adverse_selection_rate = 0.15  # 15% of fills are adversely selected
    adverse_move = current_spread * 2  # when adversely selected, lose ~2x spread
    adverse_cost = fill_prob_per_side * adverse_selection_rate * adverse_move * quote_size

    # Inventory cost: one-sided fills leave exposure
    one_sided_prob = fill_prob_per_side * (1 - fill_prob_per_side) * 2
    inventory_move = current_spread  # expected adverse move on held inventory
    inventory_cost = one_sided_prob * inventory_move * quote_size * 0.5

    # Cancel/replace cost: re-quoting friction (gas, latency)
    cancel_cost = 0.001  # negligible on Polygon

    # Fee cost
    fee_cost = 0.0
    if fees_enabled:
        # Taker fees apply to the OTHER side; maker is free
        # But if we get hit, the counterparty pays taker fee, not us
        fee_cost = 0.0  # Makers don't pay fees on Polymarket

    total_ev = round(spread_capture_ev + reward_ev - adverse_cost - inventory_cost - cancel_cost, 6)

    return {
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
        "quote_spread": round(quote_spread, 6),
        "quote_size": quote_size,
        "midpoint": round(midpoint, 4),
        "our_q_score": round(our_q_score, 2),
        "competition_factor": round(competition_factor, 2),
        "reward_share_est": round(reward_share, 6),
        "fill_prob_per_side": round(fill_prob_per_side, 4),
        "both_fill_prob": round(both_fill_prob, 4),
        "spread_capture_ev": round(spread_capture_ev, 6),
        "reward_ev": round(reward_ev, 6),
        "adverse_cost": round(adverse_cost, 6),
        "inventory_cost": round(inventory_cost, 6),
        "cancel_cost": round(cancel_cost, 6),
        "fee_cost": round(fee_cost, 6),
        "total_ev": total_ev,
    }


# ---------------------------------------------------------------------------
# Stress testing views
# ---------------------------------------------------------------------------

def stress_test_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Apply fill-realism stress views to a maker-MM plan.

    Views:
    1. spread_haircut_25: 25% wider spreads → less spread capture
    2. spread_haircut_50: 50% wider spreads
    3. spread_haircut_75: 75% wider spreads
    4. one_sided_fill: only one side fills regularly
    5. adverse_move: higher adverse selection (30% instead of 15%)
    6. high_competition: 2x more competitors for rewards
    7. conservative_combined: all pessimistic assumptions together
    """
    ev = plan["ev"]
    base_ev = ev["total_ev"]

    views = {}

    # 1-3: Spread haircuts
    for pct in [25, 50, 75]:
        factor = 1 - pct / 100
        adjusted_spread_capture = ev["spread_capture_ev"] * factor
        adjusted_ev = adjusted_spread_capture + ev["reward_ev"] - ev["adverse_cost"] - ev["inventory_cost"] - ev["cancel_cost"]
        views[f"spread_haircut_{pct}"] = {
            "total_ev": round(adjusted_ev, 6),
            "positive": adjusted_ev > 0,
        }

    # 4: One-sided fill (no spread capture, higher inventory cost)
    one_sided_ev = ev["reward_ev"] - ev["adverse_cost"] - ev["inventory_cost"] * 2 - ev["cancel_cost"]
    views["one_sided_fill"] = {
        "total_ev": round(one_sided_ev, 6),
        "positive": one_sided_ev > 0,
    }

    # 5: High adverse selection (double)
    high_adverse_ev = ev["spread_capture_ev"] + ev["reward_ev"] - ev["adverse_cost"] * 2 - ev["inventory_cost"] - ev["cancel_cost"]
    views["adverse_move"] = {
        "total_ev": round(high_adverse_ev, 6),
        "positive": high_adverse_ev > 0,
    }

    # 6: High competition (halve reward share)
    high_comp_ev = ev["spread_capture_ev"] + ev["reward_ev"] * 0.5 - ev["adverse_cost"] - ev["inventory_cost"] - ev["cancel_cost"]
    views["high_competition"] = {
        "total_ev": round(high_comp_ev, 6),
        "positive": high_comp_ev > 0,
    }

    # 7: Conservative combined: 50% spread haircut + high adverse + high competition + one-sided fill penalty
    conservative_ev = (
        ev["spread_capture_ev"] * 0.5  # 50% spread haircut
        + ev["reward_ev"] * 0.5        # 2x competition
        - ev["adverse_cost"] * 2        # 2x adverse selection
        - ev["inventory_cost"] * 1.5    # higher inventory cost
        - ev["cancel_cost"]
    )
    views["conservative_combined"] = {
        "total_ev": round(conservative_ev, 6),
        "positive": conservative_ev > 0,
    }

    # Classification
    all_positive = all(v["positive"] for v in views.values())
    conservative_positive = views["conservative_combined"]["positive"]

    if all_positive:
        label = "ROBUST_PAPER_MM"
    elif conservative_positive:
        label = "MODERATE_PAPER_MM"
    elif base_ev > 0:
        label = "FRAGILE_PAPER_MM"
    else:
        label = "NEGATIVE_EV"

    return {
        "base_ev": round(base_ev, 6),
        "label": label,
        "views": views,
    }


# ---------------------------------------------------------------------------
# Queue-realism views (Step 2)
# ---------------------------------------------------------------------------

def queue_realism_test(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Apply queue-realism stress views for Step 2 validation.

    Views:
    1. delayed_fill: fills take 2x longer → less spread capture per day
    2. quote_replaced: 30% of our quotes get replaced before filling
    3. partial_fill: average fill is only 60% of quote size
    4. reduced_reward_dwell: only 70% of time our quotes are eligible
    5. conservative_combined: all queue effects together
    """
    ev = plan["ev"]

    views = {}

    # 1: Delayed fill - half the fill rate
    delayed_ev = ev["spread_capture_ev"] * 0.5 + ev["reward_ev"] - ev["adverse_cost"] * 0.5 - ev["inventory_cost"] * 0.5 - ev["cancel_cost"]
    views["delayed_fill"] = {"total_ev": round(delayed_ev, 6), "positive": delayed_ev > 0}

    # 2: Quote replaced - 30% of quotes don't fill
    replaced_ev = ev["spread_capture_ev"] * 0.7 + ev["reward_ev"] * 0.7 - ev["adverse_cost"] * 0.7 - ev["inventory_cost"] - ev["cancel_cost"] * 3
    views["quote_replaced"] = {"total_ev": round(replaced_ev, 6), "positive": replaced_ev > 0}

    # 3: Partial fill - 60% fill rate
    partial_ev = ev["spread_capture_ev"] * 0.6 + ev["reward_ev"] - ev["adverse_cost"] * 0.6 - ev["inventory_cost"] * 1.2 - ev["cancel_cost"]
    views["partial_fill"] = {"total_ev": round(partial_ev, 6), "positive": partial_ev > 0}

    # 4: Reduced reward dwell - only 70% eligible time
    dwell_ev = ev["spread_capture_ev"] + ev["reward_ev"] * 0.7 - ev["adverse_cost"] - ev["inventory_cost"] - ev["cancel_cost"]
    views["reduced_reward_dwell"] = {"total_ev": round(dwell_ev, 6), "positive": dwell_ev > 0}

    # 5: Conservative combined
    combined_ev = (
        ev["spread_capture_ev"] * 0.5 * 0.7 * 0.6  # delayed + replaced + partial
        + ev["reward_ev"] * 0.7 * 0.7               # replaced + reduced dwell
        - ev["adverse_cost"] * 0.5                   # less fills = less adverse
        - ev["inventory_cost"] * 1.5                 # more inventory from partials
        - ev["cancel_cost"] * 3                      # more requoting
    )
    views["conservative_combined"] = {"total_ev": round(combined_ev, 6), "positive": combined_ev > 0}

    all_positive = all(v["positive"] for v in views.values())
    conservative_positive = views["conservative_combined"]["positive"]

    label = "QUEUE_RESILIENT" if all_positive else ("MODERATE_QUEUE" if conservative_positive else "FRAGILE")

    return {"label": label, "views": views}


# ---------------------------------------------------------------------------
# Calibration feedback penalties
# ---------------------------------------------------------------------------

def _load_calib_penalties(calib_db_path: str) -> dict[str, dict]:
    """
    Load per-market calibration feedback from maker_paper_calib.db.

    Returns a dict keyed by market_slug.  Only markets that meet a penalty
    threshold are included.  Penalty multipliers are applied to total_ev
    before sorting/ranking so that fill-poor or model-optimistic markets
    fall naturally in the output.

    Penalty rules (cumulative, multiplicative):
      FILL_POOR           — ≥10 obs, 0 crosses          → 0.30×
      MARKET_STATIC       — avg bid-cross-dist > 10c    → 0.60×
      MODEL_TOO_OPTIMISTIC— avg implied/modeled < 0.30  → 0.70×
    """
    import sqlite3 as _sqlite3
    from pathlib import Path as _Path

    path = _Path(calib_db_path)
    if not path.exists():
        return {}

    conn = _sqlite3.connect(str(path))
    conn.row_factory = _sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                market_slug,
                COUNT(*)                                                        AS n_obs,
                SUM(CASE WHEN status IN
                    ('BID_CROSSED','ASK_CROSSED','BOTH_CROSSED') THEN 1 ELSE 0
                    END)                                                        AS n_crosses,
                AVG(bid_cross_distance_cents)                                   AS avg_bid_dist,
                AVG(CASE WHEN implied_reward_ev > 0 AND modeled_reward_ev > 0
                         THEN implied_reward_ev / modeled_reward_ev END)        AS avg_reward_ratio
            FROM quote_observations
            GROUP BY market_slug
            """
        ).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    penalties: dict[str, dict] = {}
    for r in rows:
        flags: list[str] = []
        ev_multiplier = 1.0

        n_obs     = r["n_obs"] or 0
        n_crosses = r["n_crosses"] or 0
        avg_dist  = r["avg_bid_dist"]
        avg_ratio = r["avg_reward_ratio"]

        if n_obs >= 10 and n_crosses == 0:
            flags.append("FILL_POOR")
            ev_multiplier *= 0.30

        if avg_dist is not None and avg_dist > 10.0:
            flags.append("MARKET_STATIC")
            ev_multiplier *= 0.60

        if avg_ratio is not None and avg_ratio < 0.30:
            flags.append("MODEL_TOO_OPTIMISTIC")
            ev_multiplier *= 0.70

        if flags:
            penalties[r["market_slug"]] = {
                "flags":          flags,
                "ev_multiplier":  round(ev_multiplier, 4),
                "n_obs":          n_obs,
                "n_crosses":      n_crosses,
            }

    return penalties


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wide maker-MM scan + stress test")
    parser.add_argument("--gamma-host", default="https://gamma-api.polymarket.com")
    parser.add_argument("--clob-host", default="https://clob.polymarket.com")
    parser.add_argument("--market-limit", type=int, default=500)
    parser.add_argument("--top-events", type=int, default=30)
    parser.add_argument("--out-dir", default="data/reports")
    parser.add_argument("--fetch-books", action="store_true", default=True, help="Fetch live orderbooks")
    parser.add_argument("--no-fetch-books", dest="fetch_books", action="store_false")
    parser.add_argument(
        "--calib-db",
        default="data/processed/maker_paper_calib.db",
        help="Path to maker_paper_calib.db for calibration feedback penalties",
    )
    parser.add_argument(
        "--exclude-events",
        default="",
        help="Comma-separated event slugs to hide from active-anchor shortlist output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_events: set[str] = {
        s.strip() for s in args.exclude_events.split(",") if s.strip()
    }

    print(f"Loading calibration penalties from {args.calib_db} ...")
    calib_penalties = _load_calib_penalties(args.calib_db)
    if calib_penalties:
        print(f"  {len(calib_penalties)} markets have calibration penalties")
        for slug, p in calib_penalties.items():
            print(f"  {slug}: {p['flags']} → {p['ev_multiplier']}× EV")
    else:
        print("  No calibration penalties loaded (DB absent or no penalised markets)")

    print(f"[1/5] Fetching events and markets from Gamma API (limit={args.market_limit})...")
    events = fetch_events(args.gamma_host, args.market_limit)
    markets = fetch_markets(args.gamma_host, args.market_limit)
    print(f"       Got {len(events)} events, {len(markets)} markets")

    registry = build_event_market_registry(events, markets)

    # Extract all reward-eligible markets directly from registry
    print("[2/5] Filtering reward-eligible markets...")
    eligible: list[dict[str, Any]] = []
    events_with_eligible: set[str] = set()

    for event in registry.get("events", []):
        event_slug = event.get("slug", "")
        for m in event.get("markets", []):
            if not m.get("is_binary_yes_no"):
                continue
            if not m.get("enable_orderbook"):
                continue
            rewards = m.get("clob_rewards") or []
            reward_rate = sum(float(r.get("rewardsDailyRate", 0) or 0) for r in rewards)
            if reward_rate <= 0:
                continue
            min_size = float(m.get("rewards_min_size") or 0)
            max_spread = float(m.get("rewards_max_spread") or 0)
            if min_size <= 0 or max_spread <= 0:
                continue

            best_bid = float(m.get("best_bid") or 0)
            best_ask = float(m.get("best_ask") or 0)
            if best_bid <= 0 or best_ask <= best_bid:
                continue

            eligible.append({
                "event_slug": event_slug,
                "event_title": event.get("title"),
                "market_slug": m.get("slug"),
                "question": m.get("question"),
                "yes_token_id": m.get("yes_token_id"),
                "no_token_id": m.get("no_token_id"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "rewards_min_size": min_size,
                "rewards_max_spread": max_spread,
                "reward_daily_rate": reward_rate,
                "fees_enabled": bool(m.get("fees_enabled")),
                "volume_num": float(m.get("volume_num") or 0),
                "liquidity_num": float(m.get("liquidity_num") or 0),
                "neg_risk": bool(m.get("neg_risk")),
            })
            events_with_eligible.add(event_slug)

    print(f"       {len(eligible)} eligible markets across {len(events_with_eligible)} events")

    # Optionally fetch live orderbooks for more accurate pricing
    if args.fetch_books and eligible:
        print("[3/5] Fetching live orderbooks for eligible markets...")
        clob = ReadOnlyClob(args.clob_host)
        token_ids = []
        for m in eligible:
            if m["yes_token_id"]:
                token_ids.append(m["yes_token_id"])

        books = clob.prefetch_books(token_ids[:200], max_workers=8)
        print(f"       Fetched {len(books)} orderbooks")

        # Update best bid/ask from live data
        for m in eligible:
            book = books.get(m["yes_token_id"])
            if book and book.bids and book.asks:
                m["best_bid"] = book.bids[0].price
                m["best_ask"] = book.asks[0].price
                m["live_book"] = True
            else:
                m["live_book"] = False
    else:
        print("[3/5] Skipping live orderbook fetch")
        for m in eligible:
            m["live_book"] = False

    # Run EV analysis on all eligible markets
    print("[4/5] Running EV analysis + stress tests...")
    plans: list[dict[str, Any]] = []
    reject_counter: Counter[str] = Counter()

    for m in eligible:
        # Skip if bid/ask is stale or crossed
        if m["best_ask"] <= m["best_bid"]:
            reject_counter["crossed_book"] += 1
            continue

        ev = compute_maker_mm_ev(
            best_bid=m["best_bid"],
            best_ask=m["best_ask"],
            rewards_min_size=m["rewards_min_size"],
            rewards_max_spread_cents=m["rewards_max_spread"],
            reward_daily_rate=m["reward_daily_rate"],
            volume_num=m.get("volume_num", 0),
            liquidity_num=m.get("liquidity_num", 0),
            fees_enabled=m.get("fees_enabled", False),
        )

        # Apply calibration feedback penalties before ranking.
        # Multipliers are cumulative; FILL_POOR alone drops EV to 30% of model.
        calib = calib_penalties.get(m["market_slug"])
        if calib:
            mult = calib["ev_multiplier"]
            ev["total_ev"]          = round(ev["total_ev"]          * mult, 6)
            ev["reward_ev"]         = round(ev["reward_ev"]         * mult, 6)
            ev["spread_capture_ev"] = round(ev["spread_capture_ev"] * mult, 6)

        if ev["total_ev"] <= 0:
            reject_counter["negative_ev"] += 1
            continue

        plan = {
            "event_slug": m["event_slug"],
            "event_title": m["event_title"],
            "market_slug": m["market_slug"],
            "question": m["question"],
            "yes_token_id": m.get("yes_token_id"),
            "no_token_id": m.get("no_token_id"),
            "best_bid": m["best_bid"],
            "best_ask": m["best_ask"],
            "rewards_min_size": m["rewards_min_size"],
            "rewards_max_spread": m["rewards_max_spread"],
            "reward_daily_rate": m["reward_daily_rate"],
            "fees_enabled": m["fees_enabled"],
            "neg_risk": m["neg_risk"],
            "live_book": m["live_book"],
            "volume_num": m.get("volume_num", 0),
            "ev": ev,
            "calib_flags":        calib["flags"]          if calib else [],
            "calib_ev_multiplier": calib["ev_multiplier"] if calib else 1.0,
        }

        # Fill-realism stress test
        stress = stress_test_plan(plan)
        plan["stress"] = stress

        # Queue-realism test
        queue = queue_realism_test(plan)
        plan["queue_realism"] = queue

        plans.append(plan)

    # Sort by total EV descending
    plans.sort(key=lambda p: p["ev"]["total_ev"], reverse=True)

    # Classify
    robust_plans = [p for p in plans if p["stress"]["label"] == "ROBUST_PAPER_MM"]
    moderate_plans = [p for p in plans if p["stress"]["label"] == "MODERATE_PAPER_MM"]
    fragile_plans = [p for p in plans if p["stress"]["label"] == "FRAGILE_PAPER_MM"]

    robust_events = set(p["event_slug"] for p in robust_plans)
    queue_resilient = [p for p in robust_plans if p["queue_realism"]["label"] == "QUEUE_RESILIENT"]
    queue_resilient_events = set(p["event_slug"] for p in queue_resilient)

    # Verdict
    if len(robust_plans) >= 3 and len(robust_events) >= 2:
        thesis_verdict = "THESIS_GENERALIZED"
    else:
        thesis_verdict = "STILL_SPECIAL_CASE"

    if len(queue_resilient) >= 3 and len(queue_resilient_events) >= 2:
        execution_verdict = "READY_FOR_EXECUTION"
    else:
        execution_verdict = "NEEDS_MORE_WORK"

    # Build report
    print("[5/5] Building report...")
    report = {
        "report_type": "wide_maker_mm_scan",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "scan_params": {
            "market_limit": args.market_limit,
            "top_events": args.top_events,
            "gamma_host": args.gamma_host,
        },
        "summary": {
            "events_fetched": len(events),
            "markets_fetched": len(markets),
            "reward_eligible_markets": len(eligible),
            "reward_eligible_events": len(events_with_eligible),
            "positive_ev_plans": len(plans),
            "robust_plans": len(robust_plans),
            "moderate_plans": len(moderate_plans),
            "fragile_plans": len(fragile_plans),
            "robust_events": len(robust_events),
            "queue_resilient_plans": len(queue_resilient),
            "queue_resilient_events": len(queue_resilient_events),
            "reject_reasons": dict(reject_counter.most_common()),
        },
        "verdicts": {
            "thesis": thesis_verdict,
            "execution": execution_verdict,
        },
        "robust_plans": [
            {
                "event_slug": p["event_slug"],
                "market_slug": p["market_slug"],
                "question": p["question"],
                "reward_daily_rate": p["reward_daily_rate"],
                "total_ev": p["ev"]["total_ev"],
                "reward_ev": p["ev"]["reward_ev"],
                "spread_capture_ev": p["ev"]["spread_capture_ev"],
                "conservative_ev": p["stress"]["views"]["conservative_combined"]["total_ev"],
                "queue_label": p["queue_realism"]["label"],
                "queue_conservative_ev": p["queue_realism"]["views"]["conservative_combined"]["total_ev"],
                "fees_enabled": p["fees_enabled"],
                "neg_risk": p["neg_risk"],
            }
            for p in robust_plans
        ],
        "all_plans": [
            {
                "event_slug":          p["event_slug"],
                "market_slug":         p["market_slug"],
                "total_ev":            p["ev"]["total_ev"],
                "stress_label":        p["stress"]["label"],
                "queue_label":         p["queue_realism"]["label"],
                "reward_daily_rate":   p["reward_daily_rate"],
                "reward_ev":           p["ev"]["reward_ev"],
                "calib_flags":         p.get("calib_flags", []),
                "calib_ev_multiplier": p.get("calib_ev_multiplier", 1.0),
            }
            for p in plans
        ],
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"wide_maker_scan_{stamp}.json"
    latest_path = out_dir / "wide_maker_scan_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")

    # Print summary
    print("\n" + "=" * 70)
    print(f"STEP 1 RESULTS: {thesis_verdict}")
    print("=" * 70)
    print(f"Eligible markets:     {len(eligible)}")
    print(f"Positive EV plans:    {len(plans)}")
    print(f"ROBUST plans:         {len(robust_plans)} across {len(robust_events)} events")
    print(f"MODERATE plans:       {len(moderate_plans)}")
    print(f"FRAGILE plans:        {len(fragile_plans)}")
    print(f"Queue-resilient:      {len(queue_resilient)} across {len(queue_resilient_events)} events")
    print(f"Rejects: {dict(reject_counter.most_common())}")

    # Active-anchor shortlist: exclude monitor-only events from display
    shortlist = [p for p in robust_plans if p["event_slug"] not in excluded_events]

    if robust_plans:
        print(f"\nTop ROBUST plans (active-anchor shortlist, {len(excluded_events)} event(s) excluded):")
        rank = 0
        for p in robust_plans[:20]:
            q_label      = p["queue_realism"]["label"]
            flags        = p.get("calib_flags", [])
            mult         = p.get("calib_ev_multiplier", 1.0)
            is_excluded  = p["event_slug"] in excluded_events
            flag_str     = f"  CALIB:{','.join(flags)} ×{mult}" if flags else ""
            excl_str     = "  [MONITOR-ONLY, excluded from shortlist]" if is_excluded else ""
            rank         += 0 if is_excluded else 1
            rank_str     = f"  {rank}." if not is_excluded else "   -."
            print(f"{rank_str} [{q_label}] {p['event_slug']}/{p['market_slug']}{flag_str}{excl_str}")
            print(f"     EV=${p['ev']['total_ev']:.4f}/day  reward=${p['ev']['reward_ev']:.4f}  spread=${p['ev']['spread_capture_ev']:.4f}")
            print(f"     conservative=${p['stress']['views']['conservative_combined']['total_ev']:.4f}")

    if excluded_events:
        print(f"\nMonitor-only (excluded from shortlist): {sorted(excluded_events)}")

    print(f"\nThesis verdict:     {thesis_verdict}")
    print(f"Execution verdict:  {execution_verdict}")
    print(f"Report: {report_path}")

    return report


if __name__ == "__main__":
    main()
