from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.live.auth import load_live_credentials
from src.live.client import LiveOpenOrder, LiveWriteClient, clean_live_error_message


REPORT_SCHEMA_VERSION = "inventory_state_readiness.v1"
REPORT_TYPE = "inventory_state_readiness"

READY_STATUS = "INVENTORY_STATE_CLEAR"
BLOCKED_STATUS = "INVENTORY_STATE_BLOCKED"
CLOB_HOST = "https://clob.polymarket.com"
POSITIONS_API_URL = "https://data-api.polymarket.com/positions"
DEFAULT_DUST_THRESHOLD_SHARES = 0.001


def build_inventory_state_report(
    *,
    open_orders: Iterable[LiveOpenOrder | dict[str, Any]] | None = None,
    positions: Iterable[dict[str, Any]] | None = None,
    read_errors: Iterable[str] | None = None,
    wallet_address: str | None = None,
    dust_threshold_shares: float = DEFAULT_DUST_THRESHOLD_SHARES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only inventory/open-order report from supplied evidence."""

    now = now or datetime.now(timezone.utc)
    order_rows = [_normalise_order(order) for order in (open_orders or [])]
    position_rows = [_normalise_position(position) for position in (positions or [])]
    non_dust_positions = [row for row in position_rows if abs(float(row["shares"])) > dust_threshold_shares]
    open_order_count = len(order_rows)
    total_position_shares = sum(float(row["shares"]) for row in non_dust_positions)
    errors = list(read_errors or [])

    blockers: list[str] = []
    if errors:
        blockers.append("INVENTORY_SOURCE_READ_FAILED")
    if open_order_count > 0:
        blockers.append("OPEN_ORDERS_PRESENT")
    if non_dust_positions:
        blockers.append("NON_USDC_POSITION_PRESENT")

    status = READY_STATUS if not blockers else BLOCKED_STATUS
    open_order_status = "NO_OPEN_ORDER" if open_order_count == 0 else "OPEN_ORDER_PRESENT"
    inventory_status = "FLAT" if not non_dust_positions else "NON_USDC_POSITION_PRESENT"

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
        "wallet_address": wallet_address,
        "dust_threshold_shares": dust_threshold_shares,
        "inventory_status": inventory_status,
        "current_inventory_status": inventory_status,
        "open_order_status": open_order_status,
        "open_order_count": open_order_count,
        "token_open_order_count": open_order_count,
        "token_balance_shares": round(total_position_shares, 6),
        "non_usdc_position_count": len(non_dust_positions),
        "partial_fill_unresolved": bool(non_dust_positions or order_rows),
        "orders": order_rows,
        "positions": position_rows,
        "non_dust_positions": non_dust_positions,
        "blockers": _unique(blockers),
        "errors": errors,
        "one_line_verdict": _one_line_verdict(status, _unique(blockers), open_order_count, len(non_dust_positions)),
    }


def collect_live_inventory_report(
    *,
    host: str = CLOB_HOST,
    env: Mapping[str, str] | None = None,
    positions_api_url: str = POSITIONS_API_URL,
    timeout_sec: float = 10.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Collect read-only live inventory evidence from CLOB and Data API."""

    env_map = os.environ if env is None else env
    signature_type = _optional_int(env_map.get("POLYMARKET_SIGNATURE_TYPE"))
    funder = str(env_map.get("POLYMARKET_FUNDER") or "").strip() or None
    errors: list[str] = []
    open_orders: list[LiveOpenOrder | dict[str, Any]] = []
    positions: list[dict[str, Any]] = []

    if not funder:
        errors.append("POLYMARKET_FUNDER_MISSING")
    try:
        creds = load_live_credentials()
        client = LiveWriteClient.from_credentials(
            creds,
            host=host,
            dry_run=False,
            signature_type=signature_type,
            funder=funder,
        )
        open_orders = client.get_all_open_orders()
    except Exception as exc:
        errors.append(f"CLOB_OPEN_ORDERS_READ_FAILED: {clean_live_error_message(exc) or type(exc).__name__}")

    if funder:
        try:
            positions = fetch_data_api_positions(
                wallet_address=funder,
                positions_api_url=positions_api_url,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            errors.append(f"DATA_API_POSITIONS_READ_FAILED: {clean_live_error_message(exc) or type(exc).__name__}")

    return build_inventory_state_report(
        open_orders=open_orders,
        positions=positions,
        read_errors=errors,
        wallet_address=funder,
        now=now,
    )


def fetch_data_api_positions(
    *,
    wallet_address: str,
    positions_api_url: str = POSITIONS_API_URL,
    timeout_sec: float = 10.0,
) -> list[dict[str, Any]]:
    query = urlencode({"user": wallet_address, "limit": 500})
    separator = "&" if "?" in positions_api_url else "?"
    url = f"{positions_api_url}{separator}{query}"
    request = Request(url, headers={"User-Agent": "polyarb-lab-inventory-readiness/1.0"})
    with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - fixed/operator supplied read-only URL
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("positions") or payload.get("results") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("POSITIONS_RESPONSE_NOT_LIST")
    return [row for row in rows if isinstance(row, dict)]


def _normalise_order(order: LiveOpenOrder | dict[str, Any]) -> dict[str, Any]:
    if isinstance(order, LiveOpenOrder):
        row = asdict(order)
    else:
        row = dict(order)
    return {
        "order_id": str(row.get("order_id") or row.get("id") or row.get("orderID") or ""),
        "side": str(row.get("side") or "").upper() or None,
        "price": _first_float(row.get("price")),
        "size": _first_float(row.get("size")),
        "size_matched": _first_float(row.get("size_matched"), row.get("sizeMatched"), row.get("matched_size")),
        "size_remaining": _first_float(row.get("size_remaining"), row.get("sizeRemaining"), row.get("remaining_size")),
        "status": str(row.get("status") or "open"),
        "token_id": row.get("token_id") or row.get("asset_id") or row.get("assetId"),
    }


def _normalise_position(position: dict[str, Any]) -> dict[str, Any]:
    shares = _first_float(
        position.get("size"),
        position.get("shares"),
        position.get("amount"),
        position.get("balance"),
        position.get("quantity"),
    )
    shares = shares or 0.0
    return {
        "market_slug": position.get("marketSlug") or position.get("market_slug") or position.get("slug"),
        "asset": position.get("asset") or position.get("outcome") or position.get("title"),
        "token_id": position.get("assetId") or position.get("asset_id") or position.get("token_id"),
        "shares": round(float(shares), 6),
        "raw": position,
    }


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str], order_count: int, position_count: int) -> str:
    if status == READY_STATUS:
        return "INVENTORY_STATE_CLEAR: no open orders and no non-USDC positions above dust; can_submit_order=false."
    return (
        "INVENTORY_STATE_BLOCKED: "
        f"open_orders={order_count}, non_usdc_positions={position_count}; "
        f"{', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
    )
