from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = "reward_adjusted_ev_lifecycle.v1"
REPORT_TYPE = "reward_adjusted_ev_lifecycle_summary"

DEFAULT_MODEL_VERSION = "proxy.v0"


def build_lifecycle_event(
    *,
    lifecycle_id: str,
    event_type: str,
    market: str,
    side: str,
    price: float,
    size: float,
    best_bid: float | None = None,
    best_ask: float | None = None,
    bid_depth: float | None = None,
    ask_depth: float | None = None,
    fill_probability: float | None = None,
    fill_probability_model_version: str = DEFAULT_MODEL_VERSION,
    toxic_flow_score: float | None = None,
    reward_scoring_state: str | None = None,
    expected_reward_usdc: float = 0.0,
    pending_reward_usdc: float = 0.0,
    confirmed_reward_usdc: float = 0.0,
    confirmed_reward_source: str | None = None,
    fill_result: str | None = None,
    inventory_result: str | None = None,
    cash_delta_usdc: float | None = None,
    realized_spread_pnl_usdc: float = 0.0,
    unrealized_pnl_usdc: float = 0.0,
    fees_usdc: float = 0.0,
    exit_result: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build one durable reward-adjusted EV lifecycle JSONL event.

    Forecast reward and pending reward are intentionally excluded from
    realized_cash_pnl_usdc. Confirmed reward contributes only when an explicit
    confirmed source is supplied.
    """

    timestamp = timestamp or datetime.now(timezone.utc)
    spread = None if best_bid is None or best_ask is None else round(float(best_ask) - float(best_bid), 6)
    confirmed_reward_counted = bool(confirmed_reward_source and confirmed_reward_usdc)
    realized_cash_pnl = realized_spread_pnl_usdc - fees_usdc + (confirmed_reward_usdc if confirmed_reward_counted else 0.0)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": timestamp.isoformat(),
        "lifecycle_id": lifecycle_id,
        "event_type": event_type,
        "market": market,
        "side": side,
        "price": round(float(price), 6),
        "size": round(float(size), 6),
        "orderbook": {
            "best_bid": _round(best_bid),
            "best_ask": _round(best_ask),
            "spread": spread,
            "bid_depth": _round(bid_depth),
            "ask_depth": _round(ask_depth),
        },
        "fill_probability": {
            "value": _round(fill_probability),
            "model_version": fill_probability_model_version,
        },
        "toxic_flow_score": _round(toxic_flow_score),
        "reward_accounting": {
            "reward_scoring_state": reward_scoring_state,
            "expected_reward_usdc": _round(expected_reward_usdc) or 0.0,
            "pending_reward_usdc": _round(pending_reward_usdc) or 0.0,
            "confirmed_reward_usdc": _round(confirmed_reward_usdc) or 0.0,
            "confirmed_reward_source": confirmed_reward_source,
            "expected_reward_is_forecast_only": True,
            "pending_reward_counted_as_confirmed_reward": False,
            "confirmed_reward_counted_in_realized_cash_pnl": confirmed_reward_counted,
        },
        "fill_result": fill_result,
        "inventory_result": inventory_result,
        "cash_delta_usdc": _round(cash_delta_usdc),
        "pnl_accounting": {
            "realized_spread_pnl_usdc": _round(realized_spread_pnl_usdc) or 0.0,
            "unrealized_pnl_usdc": _round(unrealized_pnl_usdc) or 0.0,
            "fees_usdc": _round(fees_usdc) or 0.0,
            "realized_cash_pnl_usdc": round(realized_cash_pnl, 6),
            "estimated_reward_counted_as_realized_cash_pnl": False,
        },
        "exit_result": exit_result,
    }


def append_lifecycle_event(path: str | Path, event: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def read_lifecycle_events(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = Path(path)
    if not source.exists():
        return [], {"malformed_row_count": 0, "non_object_row_count": 0}
    events: list[dict[str, Any]] = []
    malformed = 0
    non_object = 0
    for line in source.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(parsed, dict):
            non_object += 1
            continue
        events.append(parsed)
    return events, {"malformed_row_count": malformed, "non_object_row_count": non_object}


def summarize_lifecycle_events(events: Iterable[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    rows = list(events)
    realized_spread = 0.0
    realized_cash = 0.0
    confirmed_reward = 0.0
    pending_reward = 0.0
    expected_reward = 0.0
    fill_counts: dict[str, int] = {}
    lifecycle_ids: set[str] = set()
    for row in rows:
        lifecycle_id = row.get("lifecycle_id")
        if lifecycle_id:
            lifecycle_ids.add(str(lifecycle_id))
        fill_result = str(row.get("fill_result") or "UNKNOWN")
        fill_counts[fill_result] = fill_counts.get(fill_result, 0) + 1
        reward = row.get("reward_accounting") if isinstance(row.get("reward_accounting"), dict) else {}
        pnl = row.get("pnl_accounting") if isinstance(row.get("pnl_accounting"), dict) else {}
        expected_reward += float(reward.get("expected_reward_usdc") or 0.0)
        pending_reward += float(reward.get("pending_reward_usdc") or 0.0)
        if reward.get("confirmed_reward_counted_in_realized_cash_pnl") is True:
            confirmed_reward += float(reward.get("confirmed_reward_usdc") or 0.0)
        realized_spread += float(pnl.get("realized_spread_pnl_usdc") or 0.0)
        realized_cash += float(pnl.get("realized_cash_pnl_usdc") or 0.0)
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "status": "REWARD_ADJUSTED_EV_LIFECYCLE_SUMMARY_READY",
        "event_count": len(rows),
        "lifecycle_count": len(lifecycle_ids),
        "fill_result_counts": fill_counts,
        "forecast_accounting": {
            "expected_reward_usdc": round(expected_reward, 6),
            "pending_reward_usdc": round(pending_reward, 6),
            "expected_reward_is_forecast_only": True,
            "pending_reward_counted_as_confirmed_reward": False,
        },
        "cash_accounting": {
            "realized_spread_pnl_usdc": round(realized_spread, 6),
            "confirmed_reward_usdc": round(confirmed_reward, 6),
            "realized_cash_pnl_usdc": round(realized_cash, 6),
            "estimated_reward_counted_as_realized_cash_pnl": False,
        },
        "profitability_claimed": False,
    }


def _round(value: Any, digits: int = 6) -> float | None:
    if value in {None, ""}:
        return None
    return round(float(value), digits)
