"""Scanner helpers for the wide-scan maker-MM bridge integration.

Provides market fetching, EV computation, and RawCandidate construction
for the MAKER_REWARDED_EVENT_MM_V1 strategy family.  Used by the runner's
_run_maker_mm_scan() and by scripts/run_wide_scan_bridge.py.

These functions contain no runner-specific state and have no side effects.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.ingest.gamma import fetch_events, fetch_markets
from src.intelligence.market_intelligence import build_event_market_registry
from src.opportunity.models import CandidateLeg, RawCandidate, StrategyFamily


def fetch_wide_scan_maker_mm_candidates(
    gamma_host: str,
    event_slugs: str | set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Fetch Gamma registry and return reward-eligible markets with token IDs.

    Eligible criteria:
    - is_binary_yes_no and enable_orderbook
    - reward_daily_rate > 0, rewards_min_size > 0, rewards_max_spread > 0
    - best_bid > 0 and best_ask > best_bid
    - yes_token_id and no_token_id present

    event_slugs may be a single slug string or a collection of slugs.
    """
    slug_set: set[str] = {event_slugs} if isinstance(event_slugs, str) else set(event_slugs)

    events = fetch_events(gamma_host, limit=500)
    markets = fetch_markets(gamma_host, limit=500)
    registry = build_event_market_registry(events, markets)

    candidates: list[dict[str, Any]] = []
    for event in registry.get("events", []):
        if event.get("slug") not in slug_set:
            continue
        for m in event.get("markets", []):
            if not m.get("is_binary_yes_no") or not m.get("enable_orderbook"):
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
            yes_token_id = m.get("yes_token_id")
            no_token_id = m.get("no_token_id")
            if not yes_token_id or not no_token_id:
                continue

            candidates.append({
                "event_slug":         str(event.get("slug") or ""),
                "event_title":        event.get("title"),
                "market_slug":        m.get("slug"),
                "question":           m.get("question"),
                "yes_token_id":       str(yes_token_id),
                "no_token_id":        str(no_token_id),
                "best_bid":           best_bid,
                "best_ask":           best_ask,
                "rewards_min_size":   min_size,
                "rewards_max_spread": max_spread,
                "reward_daily_rate":  reward_rate,
                "fees_enabled":       bool(m.get("fees_enabled")),
                "neg_risk":           bool(m.get("neg_risk")),
                "volume_num":         float(m.get("volume_num") or 0),
            })

    return candidates


def compute_wide_scan_ev(m: dict[str, Any]) -> dict[str, Any]:
    """Compute expected value for a two-sided maker-MM quote on market m."""
    best_bid        = m["best_bid"]
    best_ask        = m["best_ask"]
    min_size        = m["rewards_min_size"]
    max_spread_cents = m["rewards_max_spread"]
    reward_rate     = m["reward_daily_rate"]
    volume_num      = m.get("volume_num", 0)

    current_spread  = best_ask - best_bid
    max_spread      = max_spread_cents / 100.0
    midpoint        = (best_bid + best_ask) / 2.0
    our_half_spread = min(current_spread / 2.0, max_spread / 2.0)
    our_half_spread = max(our_half_spread, 0.005)
    quote_bid       = round(max(0.01, midpoint - our_half_spread), 4)
    quote_ask       = round(min(0.99, midpoint + our_half_spread), 4)
    quote_spread    = quote_ask - quote_bid
    quote_size      = max(min_size, 20.0)

    distance_from_mid   = our_half_spread * 100
    v                   = max_spread_cents
    s                   = distance_from_mid
    q_per_side          = ((v - s) / v) ** 2 * quote_size if v > 0 and s < v else 0.0
    our_q_score         = q_per_side * 2
    competition_factor  = min(30.0, max(10.0, 10.0 + reward_rate / 20.0))
    estimated_total_q   = our_q_score * competition_factor
    reward_share        = reward_rate * (our_q_score / estimated_total_q) if estimated_total_q > 0 else 0.0

    volume_factor       = min(1.0, volume_num / 50000.0) if volume_num > 0 else 0.1
    tightness_factor    = max(0.0, 1.0 - (distance_from_mid / max_spread_cents))
    fill_prob_per_side  = min(0.90, max(0.05, 0.10 + 0.50 * volume_factor + 0.30 * tightness_factor))
    both_fill_prob      = fill_prob_per_side ** 2 * 0.5
    spread_capture_ev   = both_fill_prob * quote_spread * quote_size
    reward_ev           = reward_share

    adverse_cost   = fill_prob_per_side * 0.15 * current_spread * 2 * quote_size
    inventory_cost = fill_prob_per_side * (1 - fill_prob_per_side) * 2 * current_spread * quote_size * 0.5
    cancel_cost    = 0.001
    total_ev       = round(spread_capture_ev + reward_ev - adverse_cost - inventory_cost - cancel_cost, 6)

    return {
        "quote_bid":         quote_bid,
        "quote_ask":         quote_ask,
        "quote_spread":      round(quote_spread, 6),
        "quote_size":        quote_size,
        "midpoint":          round(midpoint, 4),
        "spread_capture_ev": round(spread_capture_ev, 6),
        "reward_ev":         round(reward_ev, 6),
        "adverse_cost":      round(adverse_cost, 6),
        "inventory_cost":    round(inventory_cost, 6),
        "cancel_cost":       round(cancel_cost, 6),
        "total_ev":          total_ev,
    }


def build_wide_scan_raw_candidate(
    m: dict[str, Any],
    ev: dict[str, Any],
    yes_token_id: str,
    no_token_id: str,
) -> RawCandidate:
    """Build a RawCandidate from market metadata and EV computation.

    expected_payout encoding:
      For neg-risk: pair cost = $1.00 (YES + NO redeems to $1 by construction).
      For non-neg-risk: pair cost estimated from live bid/ask.
      expected_payout = quote_size * pair_cost + total_ev
      This encodes redemption value + net daily maker reward so that
      qualification arithmetic (pair_vwap subtraction) yields a valid gross_edge.
    """
    quote_size        = ev["quote_size"]
    total_ev          = ev["total_ev"]
    market_slug       = m["market_slug"]
    pair_cost_per_unit = 1.0 if m.get("neg_risk") else (m["best_ask"] * 2)

    legs = [
        CandidateLeg(
            token_id=yes_token_id,
            market_slug=market_slug,
            action="BUY",
            side="YES",
            required_shares=quote_size,
        ),
        CandidateLeg(
            token_id=no_token_id,
            market_slug=market_slug,
            action="BUY",
            side="NO",
            required_shares=quote_size,
        ),
    ]

    return RawCandidate(
        strategy_id="wide_scan_maker_mm",
        strategy_family=StrategyFamily.MAKER_REWARDED_EVENT_MM_V1,
        candidate_id=f"wsmm_{market_slug[:32]}_{uuid.uuid4().hex[:8]}",
        kind="maker_mm_quote",
        detection_name=f"wide_scan_maker_mm:{m['event_slug']}:{market_slug}",
        market_slugs=[market_slug],
        gross_edge_cents=round(total_ev / max(quote_size, 1e-9), 6),
        expected_payout=quote_size * pair_cost_per_unit + total_ev,
        target_notional_usd=quote_size * pair_cost_per_unit,
        target_shares=quote_size,
        execution_mode="paper_eligible",
        research_only=False,
        legs=legs,
        metadata={
            "event_slug":         m["event_slug"],
            "event_title":        m.get("event_title"),
            "question":           m.get("question"),
            "neg_risk":           m.get("neg_risk"),
            "reward_daily_rate":  m["reward_daily_rate"],
            "rewards_min_size":   m["rewards_min_size"],
            "rewards_max_spread": m["rewards_max_spread"],
            "wide_scan_ev":       ev,
        },
        ts=datetime.now(timezone.utc),
    )
