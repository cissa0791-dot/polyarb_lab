from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "fill_lifecycle_logger.v1"
REPORT_TYPE = "fill_lifecycle_summary"


def build_fill_lifecycle_event(
    *,
    lifecycle_id: str,
    event_type: str,
    market: str,
    side: str,
    order_id: str | None,
    price: float,
    size: float,
    fill_quantity: float,
    inventory_delta_shares: float | None,
    cash_delta_usdc: float | None,
    fees_usdc: float,
    realized_spread_pnl_usdc: float,
    remaining_qty: float | None = None,
    expected_reward_usdc: float = 0.0,
    pending_reward_usdc: float = 0.0,
    confirmed_reward_usdc: float = 0.0,
    confirmed_reward_source: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build the minimum durable event needed before any C fill probe.

    This event is intentionally smaller than the full reward-adjusted EV log.
    It records enough to reconstruct fill, inventory, cash, fee, and PnL
    separation without granting execution authority.
    """

    timestamp = timestamp or datetime.now(timezone.utc)
    confirmed_reward_counted = bool(confirmed_reward_source and confirmed_reward_usdc)
    realized_cash_pnl = realized_spread_pnl_usdc - fees_usdc + (confirmed_reward_usdc if confirmed_reward_counted else 0.0)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "timestamp_utc": timestamp.isoformat(),
        "lifecycle_id": lifecycle_id,
        "event_type": event_type,
        "market": market,
        "market_slug": market,
        "side": side,
        "order_id": order_id,
        "price": round(float(price), 6),
        "size": round(float(size), 6),
        "fill_quantity": round(float(fill_quantity), 6),
        "fill_qty": round(float(fill_quantity), 6),
        "remaining_qty": _round(remaining_qty),
        "inventory_delta_shares": _round(inventory_delta_shares),
        "inventory_delta": _round(inventory_delta_shares),
        "cash_delta_usdc": _round(cash_delta_usdc),
        "fees_usdc": _round(fees_usdc) or 0.0,
        "fee": _round(fees_usdc) or 0.0,
        "realized_spread_pnl_usdc": _round(realized_spread_pnl_usdc) or 0.0,
        "reward_separation": {
            "expected_reward_usdc": _round(expected_reward_usdc) or 0.0,
            "pending_reward_usdc": _round(pending_reward_usdc) or 0.0,
            "confirmed_reward_usdc": _round(confirmed_reward_usdc) or 0.0,
            "confirmed_reward": _round(confirmed_reward_usdc) or 0.0,
            "confirmed_reward_source": confirmed_reward_source,
            "expected_reward_is_forecast_only": True,
            "pending_reward_counted_as_confirmed_reward": False,
            "confirmed_reward_counted_in_realized_cash_pnl": confirmed_reward_counted,
            "reward_count_is_actual_earned_reward": confirmed_reward_counted,
        },
        "realized_cash_pnl_usdc": round(realized_cash_pnl, 6),
        "estimated_reward_counted_as_realized_cash_pnl": False,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }


def append_fill_lifecycle_event(path: str | Path, event: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def read_fill_lifecycle_events(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = Path(path)
    if not source.exists():
        return [], {"malformed_row_count": 0, "non_object_row_count": 0}
    rows: list[dict[str, Any]] = []
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
        rows.append(parsed)
    return rows, {"malformed_row_count": malformed, "non_object_row_count": non_object}


def summarize_fill_lifecycle_events(events: list[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    fill_qty = 0.0
    inventory_delta = 0.0
    cash_delta = 0.0
    fees = 0.0
    spread_pnl = 0.0
    realized_cash = 0.0
    expected_reward = 0.0
    pending_reward = 0.0
    confirmed_reward = 0.0
    for event in events:
        fill_qty += float(event.get("fill_quantity") or 0.0)
        inventory_delta += float(event.get("inventory_delta_shares") or 0.0)
        cash_delta += float(event.get("cash_delta_usdc") or 0.0)
        fees += float(event.get("fees_usdc") or 0.0)
        spread_pnl += float(event.get("realized_spread_pnl_usdc") or 0.0)
        realized_cash += float(event.get("realized_cash_pnl_usdc") or 0.0)
        reward = event.get("reward_separation") if isinstance(event.get("reward_separation"), dict) else {}
        expected_reward += float(reward.get("expected_reward_usdc") or 0.0)
        pending_reward += float(reward.get("pending_reward_usdc") or 0.0)
        if reward.get("confirmed_reward_counted_in_realized_cash_pnl") is True:
            confirmed_reward += float(reward.get("confirmed_reward_usdc") or 0.0)
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "status": "FILL_LIFECYCLE_SUMMARY_READY",
        "event_count": len(events),
        "fill_quantity": round(fill_qty, 6),
        "inventory_delta_shares": round(inventory_delta, 6),
        "cash_delta_usdc": round(cash_delta, 6),
        "fees_usdc": round(fees, 6),
        "realized_spread_pnl_usdc": round(spread_pnl, 6),
        "realized_cash_pnl_usdc": round(realized_cash, 6),
        "reward_separation": {
            "expected_reward_usdc": round(expected_reward, 6),
            "pending_reward_usdc": round(pending_reward, 6),
            "confirmed_reward_usdc": round(confirmed_reward, 6),
            "expected_reward_is_forecast_only": True,
            "pending_reward_counted_as_confirmed_reward": False,
        },
        "estimated_reward_counted_as_realized_cash_pnl": False,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }


def _round(value: Any, digits: int = 6) -> float | None:
    if value in {None, ""}:
        return None
    return round(float(value), digits)
