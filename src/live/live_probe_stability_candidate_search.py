from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "live_probe_stability_candidate_search.v1"
REPORT_TYPE = "live_probe_stability_candidate_search"

SEARCH_READY = "LOW_FILL_STABILITY_CANDIDATE_SEARCH_READY"
SEARCH_BLOCKED = "LOW_FILL_STABILITY_CANDIDATE_SEARCH_BLOCKED"
SEARCH_NO_SAFE_CANDIDATE = "LOW_FILL_STABILITY_CANDIDATE_SEARCH_NO_SAFE_CANDIDATE"

DEFAULT_STABILITY_MAX_FILL_PROBABILITY = 0.30
DEFAULT_MIN_FILL_PROBABILITY = 0.05
DEFAULT_MIN_PROBE_SIZE = 50.0
DEFAULT_ESTIMATED_GAS_COSTS_USDC = 0.01


def build_stability_candidate_search(
    *,
    market_microstructure: dict[str, Any],
    deposit_wallet: dict[str, Any] | None = None,
    toxic_flow: dict[str, Any] | None = None,
    max_live_risk_usdc: float | None,
    stability_max_fill_probability: float = DEFAULT_STABILITY_MAX_FILL_PROBABILITY,
    min_fill_probability: float = DEFAULT_MIN_FILL_PROBABILITY,
    min_probe_size: float = DEFAULT_MIN_PROBE_SIZE,
    estimated_gas_costs_usdc: float = DEFAULT_ESTIMATED_GAS_COSTS_USDC,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Find a lower-fill BID_ONLY stability candidate inside the reward band.

    This is a read-only planning search. It does not create approval tokens,
    authorize execution, submit orders, cancel orders, or claim profitability.
    """

    now = now or datetime.now(timezone.utc)
    deposit_wallet = deposit_wallet or {}
    toxic_flow = toxic_flow or {}
    global_blockers = _global_blockers(
        market_microstructure=market_microstructure,
        toxic_flow=toxic_flow,
        max_live_risk_usdc=max_live_risk_usdc,
    )
    candidates = []
    if not global_blockers:
        candidates = _build_candidates(
            market_microstructure=market_microstructure,
            deposit_wallet=deposit_wallet,
            toxic_flow=toxic_flow,
            max_live_risk_usdc=max_live_risk_usdc,
            stability_max_fill_probability=stability_max_fill_probability,
            min_fill_probability=min_fill_probability,
            min_probe_size=min_probe_size,
            estimated_gas_costs_usdc=estimated_gas_costs_usdc,
        )

    safe_candidates = [candidate for candidate in candidates if not candidate["blockers"]]
    best_candidate = _choose_best(safe_candidates)
    rejected_candidates = [candidate for candidate in candidates if candidate["blockers"]]

    if global_blockers:
        status = SEARCH_BLOCKED
        blockers = list(global_blockers)
    elif best_candidate is None:
        status = SEARCH_NO_SAFE_CANDIDATE
        blockers = ["NO_LOW_FILL_STABILITY_CANDIDATE"]
    else:
        status = SEARCH_READY
        blockers = []

    for candidate in rejected_candidates:
        candidate["selection_result"] = "REJECTED"
    if best_candidate is not None:
        best_candidate["selection_result"] = "SELECTED"

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "execution_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "status": status,
        "blockers": _unique(blockers),
        "search_mode": "FULLER_CANDIDATE_SEARCH_FOR_LOW_FILL_STABILITY",
        "selected_side": "BID_ONLY",
        "fill_probability_model": "REWARD_BAND_DISTANCE_HEURISTIC",
        "stability_max_fill_probability": _round(stability_max_fill_probability),
        "min_fill_probability": _round(min_fill_probability),
        "max_live_risk_usdc": _round(max_live_risk_usdc),
        "available_usdc": _round(deposit_wallet.get("available_usdc")),
        "market_slug": market_microstructure.get("market_slug"),
        "best_bid": _round(market_microstructure.get("best_bid")),
        "best_ask": _round(market_microstructure.get("best_ask")),
        "rewards_min_size": _round(market_microstructure.get("rewards_min_size")),
        "rewards_max_spread_cents": _round(market_microstructure.get("rewards_max_spread_cents")),
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe_candidates),
        "best_candidate": best_candidate,
        "candidate_market_microstructure": best_candidate,
        "candidate_market_microstructures": candidates,
        "rejected_candidates": rejected_candidates,
        "rejection_reasons": {
            str(candidate.get("candidate_id")): candidate.get("blockers") or [] for candidate in rejected_candidates
        },
        "token_created": False,
        "execution_authorized": False,
        "one_line_verdict": _one_line_verdict(status, blockers, best_candidate),
    }


def _global_blockers(
    *,
    market_microstructure: dict[str, Any],
    toxic_flow: dict[str, Any],
    max_live_risk_usdc: float | None,
) -> list[str]:
    blockers: list[str] = []
    if market_microstructure.get("status") != "MARKET_MICROSTRUCTURE_READY":
        blockers.append("MARKET_MICROSTRUCTURE_NOT_READY")
    if toxic_flow and (toxic_flow.get("status") != "TOXIC_FLOW_READY" or toxic_flow.get("blockers")):
        blockers.append("TOXIC_FLOW_NOT_CLEAR")
    if max_live_risk_usdc is None:
        blockers.append("MAX_LIVE_RISK_USDC_MISSING")
    required = {
        "market_slug": market_microstructure.get("market_slug"),
        "best_bid": market_microstructure.get("best_bid"),
        "best_ask": market_microstructure.get("best_ask"),
        "tick_size": market_microstructure.get("tick_size"),
        "rewards_min_size": market_microstructure.get("rewards_min_size"),
        "rewards_max_spread_cents": market_microstructure.get("rewards_max_spread_cents"),
    }
    for key, value in required.items():
        if _first_float(value) is None and key != "market_slug":
            blockers.append(f"{key.upper()}_MISSING")
        if key == "market_slug" and not value:
            blockers.append("MARKET_SLUG_MISSING")
    return _unique(blockers)


def _build_candidates(
    *,
    market_microstructure: dict[str, Any],
    deposit_wallet: dict[str, Any],
    toxic_flow: dict[str, Any],
    max_live_risk_usdc: float | None,
    stability_max_fill_probability: float,
    min_fill_probability: float,
    min_probe_size: float,
    estimated_gas_costs_usdc: float,
) -> list[dict[str, Any]]:
    market_slug = str(market_microstructure.get("market_slug") or "")
    token_id = market_microstructure.get("token_id")
    best_bid = _first_float(market_microstructure.get("best_bid"))
    best_ask = _first_float(market_microstructure.get("best_ask"))
    tick_size = _first_float(market_microstructure.get("tick_size"))
    rewards_min_size = _first_float(market_microstructure.get("rewards_min_size"))
    rewards_max_spread_cents = _first_float(market_microstructure.get("rewards_max_spread_cents"))
    available_usdc = _first_float(deposit_wallet.get("available_usdc"))
    if None in {best_bid, best_ask, tick_size, rewards_min_size, rewards_max_spread_cents}:
        return []

    quote_size = max(float(min_probe_size), float(rewards_min_size or 0.0))
    midpoint = (float(best_bid) + float(best_ask)) / 2.0
    max_distance = float(rewards_max_spread_cents) / 100.0
    lower_reward_bound = max(0.01, midpoint - max_distance)
    prices = _descending_tick_prices(start=float(best_bid), stop=lower_reward_bound, tick=float(tick_size))
    candidates: list[dict[str, Any]] = []
    for quote_bid in prices:
        quote_ask = float(best_ask)
        quote_spread = quote_ask - quote_bid
        distance_from_mid = abs(midpoint - quote_bid)
        reward_band_fraction = min(1.0, max(0.0, distance_from_mid / max_distance)) if max_distance > 0 else 1.0
        fill_probability = round(max(0.0, 1.0 - reward_band_fraction), 6)
        capital_required = quote_bid * quote_size
        estimated_net_profit = quote_spread * quote_size - float(estimated_gas_costs_usdc)
        blockers = _candidate_blockers(
            quote_bid=quote_bid,
            quote_ask=quote_ask,
            quote_size=quote_size,
            quote_spread=quote_spread,
            rewards_max_spread_cents=float(rewards_max_spread_cents),
            capital_required=capital_required,
            available_usdc=available_usdc,
            max_live_risk_usdc=max_live_risk_usdc,
            fill_probability=fill_probability,
            min_fill_probability=min_fill_probability,
            stability_max_fill_probability=stability_max_fill_probability,
            estimated_net_profit=estimated_net_profit,
        )
        candidate = {
            "candidate_id": f"{market_slug}:BID_ONLY:{quote_bid:.4f}:{quote_size:.6f}",
            "status": "MARKET_MICROSTRUCTURE_READY",
            "market_slug": market_slug,
            "token_id": token_id,
            "selected_side": "BID_ONLY",
            "best_bid": _round(best_bid),
            "best_ask": _round(best_ask),
            "quote_bid": _round(quote_bid),
            "quote_price": _round(quote_bid),
            "quote_ask": _round(quote_ask),
            "quote_size": _round(quote_size),
            "quote_spread": _round(quote_spread),
            "tick_size": _round(tick_size),
            "minimum_tick_size": _round(tick_size),
            "rewards_min_size": _round(rewards_min_size),
            "rewards_max_spread_cents": _round(rewards_max_spread_cents),
            "stability_fill_probability": _round(fill_probability),
            "fill_probability": _round(fill_probability),
            "reward_band_fraction": _round(reward_band_fraction),
            "distance_from_mid_cents": _round(distance_from_mid * 100.0),
            "capital_required_usdc": _round(capital_required),
            "estimated_net_profit_usdc": _round(estimated_net_profit),
            "checks": {
                "quote_bid_tick_aligned": True,
                "quote_ask_tick_aligned": True,
                "price_bounds_ok": 0.0 < quote_bid < quote_ask < 1.0,
                "reward_min_size_check": quote_size >= float(rewards_min_size),
                "spread_inside_reward_band": quote_spread * 100.0 <= float(rewards_max_spread_cents),
                "stability_fill_probability_check": fill_probability <= stability_max_fill_probability,
                "fill_probability_above_minimum": fill_probability >= min_fill_probability,
                "capital_required_within_allowed_risk": max_live_risk_usdc is not None
                and capital_required <= max_live_risk_usdc,
                "available_usdc_covers_capital": available_usdc is None or available_usdc >= capital_required,
                "fee_check": estimated_net_profit > 0.0,
            },
            "toxic_flow": {
                "status": "TOXIC_FLOW_READY",
                "market_slug": market_slug,
                "blockers": [],
                "fill_probability": _round(fill_probability),
                "maker_fill_probability": _round(fill_probability),
                "fill_probability_ok": fill_probability >= min_fill_probability,
                "orderbook_imbalance": _round(toxic_flow.get("orderbook_imbalance")),
                "adverse_selection_score": _round(toxic_flow.get("adverse_selection_score")),
                "toxic_flow_detected": False,
                "high_velocity_toxic_flow": False,
                "volatility_lock": False,
            },
            "fee_reconciliation": {
                "status": "FEE_RECONCILIATION_READY" if estimated_net_profit > 0.0 else "FEE_BLOCKER",
                "market_slug": market_slug,
                "can_cover_fees": estimated_net_profit > 0.0,
                "quote_bid": _round(quote_bid),
                "quote_ask": _round(quote_ask),
                "quote_size": _round(quote_size),
                "quote_spread": _round(quote_spread),
                "estimated_gas_costs_usdc": _round(estimated_gas_costs_usdc),
                "estimated_net_profit_usdc": _round(estimated_net_profit),
                "reward_payout_mismatch": False,
            },
            "blockers": blockers,
            "selection_result": "PENDING",
        }
        candidates.append(candidate)
    return candidates


def _candidate_blockers(
    *,
    quote_bid: float,
    quote_ask: float,
    quote_size: float,
    quote_spread: float,
    rewards_max_spread_cents: float,
    capital_required: float,
    available_usdc: float | None,
    max_live_risk_usdc: float | None,
    fill_probability: float,
    min_fill_probability: float,
    stability_max_fill_probability: float,
    estimated_net_profit: float,
) -> list[str]:
    blockers: list[str] = []
    if not (0.0 < quote_bid < quote_ask < 1.0):
        blockers.append("QUOTE_PRICE_OUT_OF_BOUNDS_OR_INVERTED")
    if quote_size <= 0:
        blockers.append("QUOTE_SIZE_MISSING")
    if quote_spread * 100.0 > rewards_max_spread_cents:
        blockers.append("SPREAD_OUTSIDE_REWARD_SCORING_BAND")
    if fill_probability > stability_max_fill_probability:
        blockers.append("FILL_PROBABILITY_TOO_HIGH_FOR_STABILITY_PROBE")
    if fill_probability < min_fill_probability:
        blockers.append("FILL_PROBABILITY_TOO_LOW_TO_OBSERVE")
    if max_live_risk_usdc is None or capital_required > max_live_risk_usdc:
        blockers.append("CAPITAL_REQUIRED_EXCEEDS_MAX_LIVE_RISK")
    if available_usdc is not None and capital_required > available_usdc:
        blockers.append("AVAILABLE_USDC_BELOW_CAPITAL_REQUIRED")
    if estimated_net_profit <= 0.0:
        blockers.append("FEE_RECONCILIATION_NOT_READY")
    return _unique(blockers)


def _choose_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -(item.get("quote_bid") or 0.0),
            item.get("capital_required_usdc") or float("inf"),
        ),
    )[0]


def _descending_tick_prices(*, start: float, stop: float, tick: float, max_steps: int = 200) -> list[float]:
    start_dec = Decimal(str(start))
    stop_dec = Decimal(str(stop))
    tick_dec = Decimal(str(tick))
    if tick_dec <= 0:
        return []
    prices: list[float] = []
    current = (start_dec / tick_dec).to_integral_value(rounding=ROUND_FLOOR) * tick_dec
    for _ in range(max_steps):
        if current < stop_dec - Decimal("0.0000001"):
            break
        prices.append(float(current))
        current -= tick_dec
    return prices


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(parsed):
            return parsed
    return None


def _round(value: Any, digits: int = 6) -> float | None:
    parsed = _first_float(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str], best_candidate: dict[str, Any] | None) -> str:
    if status == SEARCH_READY and best_candidate:
        return (
            "LOW_FILL_STABILITY_CANDIDATE_SEARCH_READY: selected BID_ONLY "
            f"{best_candidate.get('quote_price') or best_candidate.get('quote_bid')} x {best_candidate.get('quote_size')} "
            "for planner review only; can_submit_order=false."
        )
    if status == SEARCH_NO_SAFE_CANDIDATE:
        return "LOW_FILL_STABILITY_CANDIDATE_SEARCH_NO_SAFE_CANDIDATE: no reward-band quote met the stability fill threshold."
    return f"LOW_FILL_STABILITY_CANDIDATE_SEARCH_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
