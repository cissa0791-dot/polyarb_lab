from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "toxic_flow_detector.v1"
REPORT_TYPE = "toxic_flow_detector"

READY = "TOXIC_FLOW_READY"
BLOCKED = "TOXIC_FLOW_BLOCKED"

DEFAULT_IMBALANCE_THRESHOLD = 0.75
DEFAULT_LOOKBACK_SECONDS = 60.0
DEFAULT_VOLATILITY_SPREAD_MULTIPLIER = 1.5
DEFAULT_MIN_FILL_PROBABILITY = 0.05


def calc_imbalance(*, bid_size: float | int | str | None, ask_size: float | int | str | None) -> float | None:
    """Return signed top-of-book imbalance: +1 means bid-heavy, -1 ask-heavy."""

    bid = _first_float(bid_size)
    ask = _first_float(ask_size)
    if bid is None or ask is None:
        return None
    total = bid + ask
    if total <= 0:
        return None
    return round((bid - ask) / total, 6)


def calc_price_momentum(
    snapshots: Iterable[dict[str, Any]],
    *,
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS,
    now: datetime | None = None,
) -> float | None:
    """Return signed midpoint move over the lookback window."""

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=max(0.0, float(lookback_seconds)))
    points = []
    for snapshot in snapshots:
        ts = _timestamp(snapshot)
        midpoint = _midpoint(snapshot)
        if ts is None or midpoint is None or ts < cutoff or ts > now:
            continue
        points.append((ts, midpoint))
    if len(points) < 2:
        return None
    points.sort(key=lambda item: item[0])
    return round(points[-1][1] - points[0][1], 6)


def build_toxic_flow_report(
    *,
    market_microstructure: dict[str, Any] | None = None,
    explicit: dict[str, Any] | None = None,
    snapshots: Iterable[dict[str, Any]] = (),
    now: datetime | None = None,
    imbalance_threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS,
    volatility_spread_multiplier: float = DEFAULT_VOLATILITY_SPREAD_MULTIPLIER,
    min_fill_probability: float = DEFAULT_MIN_FILL_PROBABILITY,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    market_microstructure = market_microstructure or {}
    explicit = explicit or {}

    market_slug = _first_text(explicit.get("market_slug"), market_microstructure.get("market_slug"))
    best_bid = _first_float(explicit.get("best_bid"), market_microstructure.get("best_bid"))
    best_ask = _first_float(explicit.get("best_ask"), market_microstructure.get("best_ask"))
    best_bid_size = _first_float(
        explicit.get("best_bid_size"),
        market_microstructure.get("best_bid_size"),
        market_microstructure.get("bid_size"),
    )
    best_ask_size = _first_float(
        explicit.get("best_ask_size"),
        market_microstructure.get("best_ask_size"),
        market_microstructure.get("ask_size"),
    )
    quote_spread = _first_float(explicit.get("quote_spread"), market_microstructure.get("quote_spread"))
    if quote_spread is None and best_bid is not None and best_ask is not None:
        quote_spread = best_ask - best_bid

    imbalance = calc_imbalance(bid_size=best_bid_size, ask_size=best_ask_size)
    imbalance_abs = None if imbalance is None else abs(imbalance)
    momentum = calc_price_momentum(snapshots, lookback_seconds=lookback_seconds, now=now)
    momentum_abs = None if momentum is None else abs(momentum)
    volatility_threshold = (
        None if quote_spread is None else max(0.0, quote_spread) * float(volatility_spread_multiplier)
    )
    adverse_selection_risk = imbalance_abs is not None and imbalance_abs > float(imbalance_threshold)
    volatility_lock = (
        momentum_abs is not None
        and volatility_threshold is not None
        and momentum_abs > volatility_threshold
    )
    fill_probability = _fill_probability_proxy(imbalance_abs, volatility_lock=volatility_lock)
    adverse_selection_score = _adverse_selection_score(
        imbalance_abs=imbalance_abs,
        momentum_abs=momentum_abs,
        volatility_threshold=volatility_threshold,
    )
    missing_inputs = []
    if imbalance is None:
        missing_inputs.append("ORDERBOOK_IMBALANCE_INPUT_MISSING")
    if quote_spread is None:
        missing_inputs.append("QUOTE_SPREAD_MISSING")
    blockers = list(missing_inputs)
    if adverse_selection_risk:
        blockers.append("ADVERSE_SELECTION_RISK")
    if volatility_lock:
        blockers.append("VOLATILITY_LOCK")
    blockers = _unique(blockers)
    toxic_flow_detected = bool(adverse_selection_risk or volatility_lock)
    status = READY if not blockers else BLOCKED

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
        "market_slug": market_slug,
        "best_bid": _round(best_bid),
        "best_ask": _round(best_ask),
        "best_bid_size": _round(best_bid_size),
        "best_ask_size": _round(best_ask_size),
        "quote_spread": _round(quote_spread),
        "orderbook_imbalance": _round(imbalance),
        "orderbook_imbalance_abs": _round(imbalance_abs),
        "imbalance_threshold": _round(float(imbalance_threshold)),
        "adverse_selection_risk": adverse_selection_risk,
        "price_momentum_60s": _round(momentum),
        "price_momentum_abs_60s": _round(momentum_abs),
        "volatility_threshold": _round(volatility_threshold),
        "volatility_spread_multiplier": _round(float(volatility_spread_multiplier)),
        "volatility_lock": volatility_lock,
        "toxic_flow_detected": toxic_flow_detected,
        "high_velocity_toxic_flow": volatility_lock,
        "adverse_selection_score": _round(adverse_selection_score),
        "fill_probability": _round(fill_probability),
        "maker_fill_probability": _round(fill_probability),
        "min_fill_probability": _round(float(min_fill_probability)),
        "fill_probability_ok": fill_probability is not None and fill_probability >= float(min_fill_probability),
        "missing_inputs": missing_inputs,
        "blockers": blockers,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def _fill_probability_proxy(imbalance_abs: float | None, *, volatility_lock: bool) -> float | None:
    if imbalance_abs is None:
        return None
    penalty = min(1.0, max(0.0, imbalance_abs))
    if volatility_lock:
        penalty = min(1.0, penalty + 0.5)
    return max(0.0, round(1.0 - penalty, 6))


def _adverse_selection_score(
    *,
    imbalance_abs: float | None,
    momentum_abs: float | None,
    volatility_threshold: float | None,
) -> float | None:
    if imbalance_abs is None:
        return None
    momentum_component = 0.0
    if momentum_abs is not None and volatility_threshold is not None and volatility_threshold > 0:
        momentum_component = min(1.0, momentum_abs / volatility_threshold)
    return min(1.0, round((0.7 * imbalance_abs) + (0.3 * momentum_component), 6))


def _timestamp(snapshot: dict[str, Any]) -> datetime | None:
    for key in ("timestamp_utc", "generated_at_utc", "fetched_at_utc", "ts"):
        value = snapshot.get(key)
        if not value:
            continue
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _midpoint(snapshot: dict[str, Any]) -> float | None:
    candidate = snapshot.get("midpoint")
    if candidate is not None:
        return _first_float(candidate)
    bid = _first_float(snapshot.get("best_bid"))
    ask = _first_float(snapshot.get("best_ask"))
    if bid is None or ask is None:
        nested = snapshot.get("orderbook_snapshot")
        if isinstance(nested, dict):
            return _midpoint(nested)
        return None
    return (bid + ask) / 2.0


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(parsed):
            return parsed
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value in {None, ""}:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == READY:
        return "TOXIC_FLOW_READY: orderbook imbalance and momentum are inside safety thresholds; can_submit_order=false."
    return f"TOXIC_FLOW_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
