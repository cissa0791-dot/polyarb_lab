from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any, Callable


REPORT_SCHEMA_VERSION = "live_market_microstructure.v1"
REPORT_TYPE = "live_market_microstructure"

TickSizeReader = Callable[[str], Any]


def build_market_microstructure_report(
    *,
    candidate_report: dict[str, Any] | None = None,
    health_report: dict[str, Any] | None = None,
    explicit: dict[str, Any] | None = None,
    tick_size_reader: TickSizeReader | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    candidate_report = candidate_report or {}
    health_report = health_report or {}
    explicit = explicit or {}
    health_target = _health_target_market(health_report)
    candidate = _candidate_quote(candidate_report)

    market_slug = _first_text(explicit.get("market_slug"), candidate.get("market_slug"), health_target.get("market_slug"))
    token_id = _first_text(explicit.get("token_id"), candidate.get("token_id"), health_target.get("token_id"))
    quote_bid = _first_float(explicit.get("quote_bid"), candidate.get("quote_bid"), health_target.get("quote_bid"))
    quote_ask = _first_float(explicit.get("quote_ask"), candidate.get("quote_ask"), health_target.get("quote_ask"))
    quote_size = _first_float(explicit.get("quote_size"), candidate.get("quote_size"), health_target.get("quote_size"))
    best_bid = _first_float(explicit.get("best_bid"), candidate.get("best_bid"), health_target.get("best_bid"))
    best_ask = _first_float(explicit.get("best_ask"), candidate.get("best_ask"), health_target.get("best_ask"))
    tick_size, tick_source, tick_error = _resolve_tick_size(
        explicit=explicit,
        candidate=candidate,
        health_target=health_target,
        token_id=token_id,
        tick_size_reader=tick_size_reader,
    )
    rewards_min_size = _first_float(
        explicit.get("rewards_min_size"),
        candidate.get("rewards_min_size"),
        health_target.get("rewards_min_size"),
    )
    rewards_max_spread_cents = _first_float(
        explicit.get("rewards_max_spread_cents"),
        candidate.get("rewards_max_spread_cents"),
        health_target.get("rewards_max_spread_cents"),
        health_target.get("rewards_max_spread"),
    )

    checks = {
        "candidate_present": bool(candidate) or bool(explicit) or bool(health_target),
        "market_slug_present": bool(market_slug),
        "token_id_present": bool(token_id),
        "tick_size_present": tick_size is not None and tick_size > 0.0,
        "quote_bid_present": quote_bid is not None,
        "quote_ask_present": quote_ask is not None,
        "quote_bid_below_quote_ask": quote_bid is not None and quote_ask is not None and quote_bid < quote_ask,
        "quote_bid_tick_aligned": _price_tick_aligned(quote_bid, tick_size),
        "quote_ask_tick_aligned": _price_tick_aligned(quote_ask, tick_size),
        "price_bounds_ok": _price_in_bounds(quote_bid) and _price_in_bounds(quote_ask),
    }
    blockers = _blockers(checks)
    if tick_error:
        blockers.append("TICK_SIZE_READ_FAILED")
    blockers = _unique(blockers)
    status = "MARKET_MICROSTRUCTURE_READY" if not blockers else "MARKET_MICROSTRUCTURE_BLOCKED"
    spread = None if quote_bid is None or quote_ask is None else quote_ask - quote_bid

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
        "token_id": token_id,
        "best_bid": _round(best_bid),
        "best_ask": _round(best_ask),
        "quote_bid": _round(quote_bid),
        "quote_ask": _round(quote_ask),
        "quote_size": _round(quote_size),
        "quote_spread": _round(spread),
        "tick_size": _round(tick_size),
        "minimum_tick_size": _round(tick_size),
        "tick_size_source": tick_source,
        "tick_size_read_error": tick_error,
        "rewards_min_size": _round(rewards_min_size),
        "rewards_max_spread_cents": _round(rewards_max_spread_cents),
        "candidate_source": candidate.get("candidate_source") or "HEALTH_TARGET_MARKET_FALLBACK",
        "checks": checks,
        "blockers": blockers,
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def _health_target_market(health_report: dict[str, Any]) -> dict[str, Any]:
    target = health_report.get("target_market")
    if isinstance(target, dict):
        return target
    checks = health_report.get("checks")
    if isinstance(checks, dict) and isinstance(checks.get("target_market"), dict):
        return checks["target_market"]
    return {}


def _candidate_quote(candidate_report: dict[str, Any]) -> dict[str, Any]:
    sources = [
        ("EXPLICIT_MARKET_MICROSTRUCTURE_REPORT", candidate_report),
        ("VIRTUAL_MAKER_QUOTE", candidate_report.get("virtual_maker_quote")),
        ("VIRTUAL_MAKER_QUOTE_EVENT", candidate_report.get("virtual_maker_quote_event")),
        (
            "AUTO_SIZING_CANDIDATE",
            (candidate_report.get("auto_sizing_decision") or {}).get("candidate")
            if isinstance(candidate_report.get("auto_sizing_decision"), dict)
            else None,
        ),
        ("HEALTH_TARGET_MARKET", _health_target_market(candidate_report)),
    ]
    for source_name, payload in sources:
        if not isinstance(payload, dict):
            continue
        if any(payload.get(key) is not None for key in ("quote_bid", "quote_ask", "tick_size", "minimum_tick_size")):
            return {**payload, "candidate_source": source_name}
    return {}


def _resolve_tick_size(
    *,
    explicit: dict[str, Any],
    candidate: dict[str, Any],
    health_target: dict[str, Any],
    token_id: str | None,
    tick_size_reader: TickSizeReader | None,
) -> tuple[float | None, str | None, str | None]:
    source_values = [
        ("EXPLICIT_ARG", explicit.get("tick_size") or explicit.get("minimum_tick_size")),
        ("CANDIDATE_REPORT", candidate.get("tick_size") or candidate.get("minimum_tick_size")),
        ("HEALTH_TARGET_MARKET", health_target.get("tick_size") or health_target.get("minimum_tick_size")),
    ]
    for source, value in source_values:
        parsed = _first_float(value)
        if parsed is not None and parsed > 0.0:
            return parsed, source, None
    if tick_size_reader is not None and token_id:
        try:
            parsed = _first_float(tick_size_reader(token_id))
        except Exception as exc:
            return None, "CLOB_GET_TICK_SIZE", str(exc)
        if parsed is not None and parsed > 0.0:
            return parsed, "CLOB_GET_TICK_SIZE", None
    return None, None, None


def _blockers(checks: dict[str, bool]) -> list[str]:
    out: list[str] = []
    if not checks.get("candidate_present"):
        out.append("CANDIDATE_QUOTE_MISSING")
    if not checks.get("tick_size_present"):
        out.append("TICK_SIZE_MISSING")
    if not checks.get("quote_bid_present"):
        out.append("QUOTE_BID_MISSING")
    if not checks.get("quote_ask_present"):
        out.append("QUOTE_ASK_MISSING")
    if checks.get("quote_bid_present") and checks.get("quote_ask_present") and not checks.get("quote_bid_below_quote_ask"):
        out.append("QUOTE_PRICE_INVERSION")
    if checks.get("quote_bid_present") and not checks.get("quote_bid_tick_aligned"):
        out.append("QUOTE_BID_TICK_MISALIGNED")
    if checks.get("quote_ask_present") and not checks.get("quote_ask_tick_aligned"):
        out.append("QUOTE_ASK_TICK_MISALIGNED")
    if not checks.get("price_bounds_ok"):
        out.append("QUOTE_PRICE_OUT_OF_BOUNDS")
    return out


def _price_tick_aligned(price: float | None, tick: float | None) -> bool:
    if price is None or tick is None or tick <= 0:
        return False
    try:
        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(tick))
        units = price_decimal / tick_decimal
    except (InvalidOperation, ZeroDivisionError):
        return False
    return units == units.to_integral_value()


def _price_in_bounds(price: float | None) -> bool:
    return price is not None and 0.0 < price < 1.0


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
    if status == "MARKET_MICROSTRUCTURE_READY":
        return "MARKET_MICROSTRUCTURE_READY: quote prices are tick-aligned, in bounds, and non-inverted; can_submit_order=false."
    return f"MARKET_MICROSTRUCTURE_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
