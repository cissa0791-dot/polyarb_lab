from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.models import ArbOpportunity


MAKER_MM_MISSING_REWARD_METADATA = "MAKER_MM_MISSING_REWARD_METADATA"
MAKER_MM_UNSTABLE_BOOK = "MAKER_MM_UNSTABLE_BOOK"
MAKER_MM_NON_POSITIVE_EV = "MAKER_MM_NON_POSITIVE_EV"


def load_latest_event_first_inputs(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads((out_dir / "market_intelligence_registry_latest.json").read_text(encoding="utf-8"))
    watchlist = json.loads((out_dir / "market_intelligence_watchlist_latest.json").read_text(encoding="utf-8"))
    return registry, watchlist


def _reward_daily_rate(market: dict[str, Any]) -> float:
    rewards = list(market.get("clob_rewards") or [])
    rates: list[float] = []
    for reward in rewards:
        try:
            rates.append(float(reward.get("rewardsDailyRate", 0.0) or 0.0))
        except Exception:
            continue
    return round(sum(rate for rate in rates if rate > 0.0), 6)


def _has_complete_book(market: dict[str, Any]) -> bool:
    best_bid = market.get("best_bid")
    best_ask = market.get("best_ask")
    if best_bid is None or best_ask is None:
        return False
    try:
        return float(best_bid) > 0.0 and float(best_ask) > float(best_bid)
    except Exception:
        return False


def _market_watchlist_row(event_row: dict[str, Any], market_slug: str) -> dict[str, Any] | None:
    for row in list(event_row.get("top_markets") or []):
        if str(row.get("market_slug") or "") == market_slug:
            return row
    return None


def build_eligible_rewarded_market_groups(
    *,
    registry: dict[str, Any],
    watchlist_report: dict[str, Any],
    max_events: int = 10,
    min_stability_hits: int = 1,
) -> list[dict[str, Any]]:
    registry_by_event = {
        str(event.get("slug") or ""): event
        for event in registry.get("events", [])
        if str(event.get("slug") or "")
    }

    selected_events = list(watchlist_report.get("top_events") or [])[:max_events]
    groups: list[dict[str, Any]] = []
    for event_row in selected_events:
        event_slug = str(event_row.get("event_slug") or "")
        registry_event = registry_by_event.get(event_slug)
        if registry_event is None:
            continue

        eligible_markets: list[dict[str, Any]] = []
        for market in registry_event.get("markets", []):
            market_slug = str(market.get("slug") or "")
            if not market_slug:
                continue

            watch_market = _market_watchlist_row(event_row, market_slug)
            if watch_market is None:
                continue
            if not bool(market.get("enable_orderbook")):
                continue
            if bool(market.get("fees_enabled")):
                continue
            if not bool(market.get("is_binary_yes_no")):
                continue
            if not market.get("yes_token_id") or not market.get("no_token_id"):
                continue
            if not _has_complete_book(market):
                continue
            if market.get("rewards_min_size") in (None, "") or market.get("rewards_max_spread") in (None, ""):
                continue
            if _reward_daily_rate(market) <= 0.0:
                continue
            if int(watch_market.get("stability_score", 0) or 0) < min_stability_hits:
                continue

            eligible_markets.append(
                {
                    "market_slug": market_slug,
                    "question": market.get("question"),
                    "yes_token_id": str(market.get("yes_token_id") or ""),
                    "no_token_id": str(market.get("no_token_id") or ""),
                    "best_bid": float(market.get("best_bid") or 0.0),
                    "best_ask": float(market.get("best_ask") or 0.0),
                    "rewards_min_size": float(market.get("rewards_min_size") or 0.0),
                    "rewards_max_spread": float(market.get("rewards_max_spread") or 0.0),
                    "reward_daily_rate": _reward_daily_rate(market),
                    "fees_enabled": bool(market.get("fees_enabled")),
                    "activity_events": int(watch_market.get("activity_events", 0) or 0),
                    "spread_changes": int(watch_market.get("spread_changes", 0) or 0),
                    "liquidity_changes": int(watch_market.get("liquidity_changes", 0) or 0),
                    "stability_score": int(watch_market.get("stability_score", 0) or 0),
                    "watchlist_score": float(watch_market.get("watchlist_score", 0.0) or 0.0),
                }
            )

        if eligible_markets:
            groups.append(
                {
                    "event_slug": event_slug,
                    "event_title": registry_event.get("title"),
                    "event_watchlist_score": float(event_row.get("watchlist_score", 0.0) or 0.0),
                    "markets": eligible_markets,
                }
            )
    return groups


def analyze_maker_rewarded_market(*, event_group: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    best_bid = float(market.get("best_bid") or 0.0)
    best_ask = float(market.get("best_ask") or 0.0)
    rewards_min_size = float(market.get("rewards_min_size") or 0.0)
    rewards_max_spread_cents = float(market.get("rewards_max_spread") or 0.0)
    reward_daily_rate = float(market.get("reward_daily_rate") or 0.0)
    activity_events = int(market.get("activity_events", 0) or 0)
    spread_changes = int(market.get("spread_changes", 0) or 0)
    liquidity_changes = int(market.get("liquidity_changes", 0) or 0)
    stability_score = int(market.get("stability_score", 0) or 0)

    if rewards_min_size <= 0.0 or rewards_max_spread_cents <= 0.0 or reward_daily_rate <= 0.0:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_reward_filter",
                "failure_reason": MAKER_MM_MISSING_REWARD_METADATA,
                "event_slug": event_group.get("event_slug"),
                "market_slug": market.get("market_slug"),
            },
        }

    if stability_score <= 0 or activity_events <= 0 or best_ask <= best_bid:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_stability_filter",
                "failure_reason": MAKER_MM_UNSTABLE_BOOK,
                "event_slug": event_group.get("event_slug"),
                "market_slug": market.get("market_slug"),
            },
        }

    current_spread = best_ask - best_bid
    reward_max_spread = rewards_max_spread_cents / 100.0
    quote_spread = min(current_spread, reward_max_spread)
    midpoint = (best_bid + best_ask) / 2.0
    quote_bid = round(max(0.0, midpoint - (quote_spread / 2.0)), 6)
    quote_ask = round(min(1.0, midpoint + (quote_spread / 2.0)), 6)
    quote_size = round(rewards_min_size, 6)

    activity_intensity = min(1.0, activity_events / 20.0)
    stability_ratio = min(1.0, stability_score / max(activity_events, 1))
    spread_churn = min(1.0, spread_changes / max(activity_events, 1))
    liquidity_support = min(1.0, liquidity_changes / max(activity_events, 1))

    fill_probability = max(
        0.05,
        min(0.95, 0.20 + (0.35 * activity_intensity) + (0.25 * liquidity_support) + (0.20 * stability_ratio)),
    )
    quote_competitiveness = max(0.05, min(1.0, 1.0 - (quote_spread / max(reward_max_spread, 1e-9)) + 0.25))
    reward_eligibility = min(1.0, quote_competitiveness * stability_ratio)

    spread_capture_ev = round(quote_size * quote_spread * fill_probability * 0.5, 6)
    liquidity_reward_ev = round(reward_daily_rate * reward_eligibility, 6)
    adverse_selection_cost_proxy = round(quote_size * quote_spread * max(0.0, 1.0 - stability_ratio) * 0.35, 6)
    inventory_cost_proxy = round(quote_size * quote_spread * max(0.0, 1.0 - liquidity_support) * 0.20, 6)
    cancel_replace_cost_proxy = round(quote_size * quote_spread * spread_churn * 0.15, 6)
    total_ev = round(
        spread_capture_ev
        + liquidity_reward_ev
        - adverse_selection_cost_proxy
        - inventory_cost_proxy
        - cancel_replace_cost_proxy,
        6,
    )

    if total_ev <= 0.0:
        return {
            "opportunity": None,
            "audit": {
                "failure_stage": "pre_candidate_ev_filter",
                "failure_reason": MAKER_MM_NON_POSITIVE_EV,
                "event_slug": event_group.get("event_slug"),
                "market_slug": market.get("market_slug"),
                "ev_components": {
                    "spread_capture_ev": spread_capture_ev,
                    "liquidity_reward_ev": liquidity_reward_ev,
                    "adverse_selection_cost_proxy": adverse_selection_cost_proxy,
                    "inventory_cost_proxy": inventory_cost_proxy,
                    "cancel_replace_cost_proxy": cancel_replace_cost_proxy,
                    "total_ev": total_ev,
                },
            },
        }

    return {
        "opportunity": ArbOpportunity(
            kind="maker_rewarded_event_mm",
            name=f"{event_group.get('event_slug')}:{market.get('market_slug')}",
            edge_cents=round(total_ev / max(quote_size, 1e-9), 6),
            gross_profit=total_ev,
            est_fill_cost=round(midpoint * quote_size, 6),
            est_payout=round(midpoint * quote_size, 6),
            notional=round(midpoint * quote_size, 6),
            details={
                "event_slug": event_group.get("event_slug"),
                "event_title": event_group.get("event_title"),
                "event_watchlist_score": event_group.get("event_watchlist_score"),
                "market_slug": market.get("market_slug"),
                "question": market.get("question"),
                "maker_only": True,
                "paper_only": True,
                "quote_bid": quote_bid,
                "quote_ask": quote_ask,
                "quote_size": quote_size,
                "quote_spread": round(quote_spread, 6),
                "reward_daily_rate": reward_daily_rate,
                "rewards_min_size": rewards_min_size,
                "rewards_max_spread": rewards_max_spread_cents,
                "fill_probability_proxy": round(fill_probability, 6),
                "reward_eligibility_proxy": round(reward_eligibility, 6),
                "spread_capture_ev": spread_capture_ev,
                "liquidity_reward_ev": liquidity_reward_ev,
                "adverse_selection_cost_proxy": adverse_selection_cost_proxy,
                "inventory_cost_proxy": inventory_cost_proxy,
                "cancel_replace_cost_proxy": cancel_replace_cost_proxy,
                "total_ev": total_ev,
                "activity_events": activity_events,
                "spread_changes": spread_changes,
                "liquidity_changes": liquidity_changes,
                "stability_score": stability_score,
            },
            ts=datetime.now(timezone.utc),
        ),
        "audit": None,
    }


def write_maker_rewarded_mm_report(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"maker_rewarded_event_mm_v1_{stamp}.json"
    latest_report_path = out_dir / "maker_rewarded_event_mm_v1_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_report_path.write_text(payload, encoding="utf-8")
    return {
        "report_path": report_path,
        "latest_report_path": latest_report_path,
    }
