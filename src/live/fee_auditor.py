from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPORT_SCHEMA_VERSION = "fee_reconciliation.v1"
REPORT_TYPE = "fee_reconciliation"

READY = "FEE_RECONCILIATION_READY"
BLOCKED = "FEE_BLOCKER"

DEFAULT_RPC_TIMEOUT_SEC = 2.0

PriorityFeeReader = Callable[[str, float], "PriorityFeeResult"]


@dataclass(frozen=True)
class PriorityFeeResult:
    priority_fee_gwei: Decimal | None
    source: str
    latency_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None


def read_polygon_priority_fee_gwei(
    rpc_url: str,
    timeout_sec: float = DEFAULT_RPC_TIMEOUT_SEC,
) -> PriorityFeeResult:
    """Read Polygon priority fee through a JSON-RPC endpoint.

    This is read-only and does not sign, submit, cancel, or mutate any order.
    """

    started = time.perf_counter()
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_maxPriorityFeePerGas", "params": []}).encode(
        "utf-8"
    )
    request = Request(
        rpc_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "polyarb-lab-fee-auditor/1.0"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator supplied RPC URL
            raw = response.read(4096)
            latency_ms = (time.perf_counter() - started) * 1000.0
    except HTTPError as exc:
        return PriorityFeeResult(
            priority_fee_gwei=None,
            source="POLYGON_RPC",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code=f"HTTP_{exc.code}",
            error_message=str(exc.reason),
        )
    except (TimeoutError, URLError, OSError) as exc:
        return PriorityFeeResult(
            priority_fee_gwei=None,
            source="POLYGON_RPC",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code=exc.__class__.__name__,
            error_message=str(exc),
        )

    try:
        decoded = json.loads(raw.decode("utf-8"))
        result = decoded.get("result")
        if not isinstance(result, str):
            raise ValueError("JSON_RPC_RESULT_MISSING")
        wei = Decimal(int(result, 16))
    except (json.JSONDecodeError, ValueError, TypeError, InvalidOperation) as exc:
        return PriorityFeeResult(
            priority_fee_gwei=None,
            source="POLYGON_RPC",
            latency_ms=latency_ms,
            error_code="INVALID_RPC_RESPONSE",
            error_message=str(exc),
        )

    return PriorityFeeResult(
        priority_fee_gwei=wei / Decimal("1000000000"),
        source="POLYGON_RPC",
        latency_ms=latency_ms,
    )


def build_fee_reconciliation_report(
    *,
    market_microstructure: dict[str, Any] | None = None,
    explicit: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    priority_fee_reader: PriorityFeeReader | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    market_microstructure = market_microstructure or {}
    explicit = explicit or {}
    env = env if env is not None else os.environ

    market_slug = _first_text(explicit.get("market_slug"), market_microstructure.get("market_slug"))
    quote_bid = _first_decimal(explicit.get("quote_bid"), market_microstructure.get("quote_bid"))
    quote_ask = _first_decimal(explicit.get("quote_ask"), market_microstructure.get("quote_ask"))
    quote_size = _first_decimal(explicit.get("quote_size"), market_microstructure.get("quote_size"))
    quote_spread = _first_decimal(explicit.get("quote_spread"), market_microstructure.get("quote_spread"))
    if quote_spread is None and quote_bid is not None and quote_ask is not None:
        quote_spread = quote_ask - quote_bid

    maker_fee_rate = _first_decimal(
        explicit.get("maker_fee_rate"),
        market_microstructure.get("maker_fee_rate"),
        env.get("POLYMARKET_MAKER_FEE_RATE"),
    )
    taker_fee_rate = _first_decimal(
        explicit.get("taker_fee_rate"),
        market_microstructure.get("taker_fee_rate"),
        env.get("POLYMARKET_TAKER_FEE_RATE"),
    )
    entry_role = _fee_role(_first_text(explicit.get("entry_liquidity_role"), env.get("POLYMARKET_ENTRY_FEE_ROLE")), "maker")
    exit_role = _fee_role(_first_text(explicit.get("exit_liquidity_role"), env.get("POLYMARKET_EXIT_FEE_ROLE")), "taker")

    priority_fee = _resolve_priority_fee(
        explicit=explicit,
        env=env,
        priority_fee_reader=priority_fee_reader or read_polygon_priority_fee_gwei,
    )
    gas_units_per_cancel = _first_decimal(
        explicit.get("gas_units_per_cancel"),
        env.get("POLYGON_GAS_UNITS_PER_CANCEL"),
    )
    gas_asset_usdc = _first_decimal(
        explicit.get("gas_asset_usdc"),
        env.get("POLYGON_GAS_ASSET_USDC"),
    )
    cancel_tx_count = _first_decimal(
        explicit.get("cancel_tx_count"),
        env.get("POLYGON_CANCEL_TX_COUNT"),
        1,
    )
    explicit_gas_cost = _first_decimal(
        explicit.get("estimated_gas_costs_usdc"),
        env.get("POLYGON_ESTIMATED_GAS_COST_USDC"),
    )
    estimated_gas_costs, gas_cost_source = _estimated_gas_costs(
        explicit_gas_cost=explicit_gas_cost,
        priority_fee_gwei=priority_fee.priority_fee_gwei,
        gas_units_per_cancel=gas_units_per_cancel,
        gas_asset_usdc=gas_asset_usdc,
        cancel_tx_count=cancel_tx_count,
    )

    reward_payout_mismatch = _first_bool(explicit.get("reward_payout_mismatch"), market_microstructure.get("reward_payout_mismatch"), False)

    gross_spread_profit = _safe_mul(quote_spread, quote_size)
    entry_fee_rate = _rate_for_role(entry_role, maker_fee_rate=maker_fee_rate, taker_fee_rate=taker_fee_rate)
    exit_fee_rate = _rate_for_role(exit_role, maker_fee_rate=maker_fee_rate, taker_fee_rate=taker_fee_rate)
    estimated_trading_fees = _estimated_trading_fees(
        quote_bid=quote_bid,
        quote_ask=quote_ask,
        quote_size=quote_size,
        entry_fee_rate=entry_fee_rate,
        exit_fee_rate=exit_fee_rate,
    )
    total_costs = _safe_add(estimated_trading_fees, estimated_gas_costs)
    estimated_net_profit = _safe_sub(gross_spread_profit, total_costs)
    break_even_spread = None if total_costs is None or quote_size is None or quote_size <= 0 else total_costs / quote_size

    missing_inputs = _missing_inputs(
        quote_spread=quote_spread,
        quote_size=quote_size,
        quote_bid=quote_bid,
        quote_ask=quote_ask,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        estimated_gas_costs=estimated_gas_costs,
    )
    projected_fee_unknown = bool(missing_inputs)
    can_cover_fees = estimated_net_profit is not None and estimated_net_profit > 0 and not reward_payout_mismatch
    blockers = _blockers(
        missing_inputs=missing_inputs,
        quote_spread=quote_spread,
        quote_size=quote_size,
        reward_payout_mismatch=reward_payout_mismatch,
        estimated_net_profit=estimated_net_profit,
    )
    status = READY if not blockers else BLOCKED
    generated_at = now.isoformat()

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "last_audit_timestamp": generated_at,
        "read_only": True,
        "live_actions_enabled": False,
        "execution_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "status": status,
        "market_slug": market_slug,
        "quote_bid": _float(quote_bid),
        "quote_ask": _float(quote_ask),
        "quote_size": _float(quote_size),
        "quote_spread": _float(quote_spread),
        "gross_spread_profit_usdc": _float(gross_spread_profit),
        "maker_fee_rate": _float(maker_fee_rate),
        "taker_fee_rate": _float(taker_fee_rate),
        "entry_liquidity_role": entry_role,
        "exit_liquidity_role": exit_role,
        "maker_fee_model_present": maker_fee_rate is not None,
        "taker_fee_model_present": taker_fee_rate is not None,
        "estimated_trading_fees_usdc": _float(estimated_trading_fees),
        "priority_fee_gwei": _float(priority_fee.priority_fee_gwei),
        "priority_fee_source": priority_fee.source,
        "priority_fee_latency_ms": _round_float(priority_fee.latency_ms),
        "priority_fee_error_code": priority_fee.error_code,
        "priority_fee_error_message": priority_fee.error_message,
        "gas_units_per_cancel": _float(gas_units_per_cancel),
        "gas_asset_usdc": _float(gas_asset_usdc),
        "cancel_tx_count": _float(cancel_tx_count),
        "estimated_gas_costs_usdc": _float(estimated_gas_costs),
        "gas_cost_source": gas_cost_source,
        "estimated_total_costs_usdc": _float(total_costs),
        "estimated_net_profit": _float(estimated_net_profit),
        "estimated_net_profit_usdc": _float(estimated_net_profit),
        "break_even_spread": _float(break_even_spread),
        "can_cover_fees": can_cover_fees,
        "projected_fee_unknown": projected_fee_unknown,
        "missing_fee_inputs": missing_inputs,
        "reward_payout_mismatch": reward_payout_mismatch,
        "blockers": blockers,
        "one_line_verdict": _one_line_verdict(status, estimated_net_profit, blockers),
    }


def _resolve_priority_fee(
    *,
    explicit: dict[str, Any],
    env: Mapping[str, str],
    priority_fee_reader: PriorityFeeReader,
) -> PriorityFeeResult:
    explicit_priority = _first_decimal(explicit.get("priority_fee_gwei"), env.get("POLYGON_PRIORITY_FEE_GWEI"))
    if explicit_priority is not None:
        return PriorityFeeResult(priority_fee_gwei=explicit_priority, source="EXPLICIT_OR_ENV")
    if _first_decimal(explicit.get("estimated_gas_costs_usdc"), env.get("POLYGON_ESTIMATED_GAS_COST_USDC")) is not None:
        return PriorityFeeResult(priority_fee_gwei=None, source="NOT_REQUIRED_EXPLICIT_GAS_COST")
    rpc_url = _first_text(explicit.get("polygon_rpc_url"), env.get("POLYGON_RPC_URL"))
    if not rpc_url:
        return PriorityFeeResult(priority_fee_gwei=None, source="MISSING_POLYGON_RPC_OR_PRIORITY_FEE")
    timeout_sec = float(_first_decimal(explicit.get("rpc_timeout_sec"), env.get("POLYGON_RPC_TIMEOUT_SEC"), DEFAULT_RPC_TIMEOUT_SEC))
    return priority_fee_reader(rpc_url, timeout_sec)


def _estimated_gas_costs(
    *,
    explicit_gas_cost: Decimal | None,
    priority_fee_gwei: Decimal | None,
    gas_units_per_cancel: Decimal | None,
    gas_asset_usdc: Decimal | None,
    cancel_tx_count: Decimal | None,
) -> tuple[Decimal | None, str | None]:
    if explicit_gas_cost is not None:
        return explicit_gas_cost, "EXPLICIT_OR_ENV_ESTIMATED_GAS_COST_USDC"
    if None in {priority_fee_gwei, gas_units_per_cancel, gas_asset_usdc, cancel_tx_count}:
        return None, None
    assert priority_fee_gwei is not None
    assert gas_units_per_cancel is not None
    assert gas_asset_usdc is not None
    assert cancel_tx_count is not None
    return (
        priority_fee_gwei
        * Decimal("0.000000001")
        * gas_units_per_cancel
        * gas_asset_usdc
        * cancel_tx_count,
        "PRIORITY_FEE_GWEI_X_GAS_UNITS_X_GAS_ASSET_USDC",
    )


def _estimated_trading_fees(
    *,
    quote_bid: Decimal | None,
    quote_ask: Decimal | None,
    quote_size: Decimal | None,
    entry_fee_rate: Decimal | None,
    exit_fee_rate: Decimal | None,
) -> Decimal | None:
    if None in {quote_bid, quote_ask, quote_size, entry_fee_rate, exit_fee_rate}:
        return None
    assert quote_bid is not None
    assert quote_ask is not None
    assert quote_size is not None
    assert entry_fee_rate is not None
    assert exit_fee_rate is not None
    return (quote_bid * quote_size * entry_fee_rate) + (quote_ask * quote_size * exit_fee_rate)


def _missing_inputs(**values: Decimal | None) -> list[str]:
    out: list[str] = []
    for key, value in values.items():
        if value is None:
            out.append(key.upper() + "_MISSING")
    return out


def _blockers(
    *,
    missing_inputs: list[str],
    quote_spread: Decimal | None,
    quote_size: Decimal | None,
    reward_payout_mismatch: bool,
    estimated_net_profit: Decimal | None,
) -> list[str]:
    blockers = list(missing_inputs)
    if quote_spread is not None and quote_spread <= 0:
        blockers.append("QUOTE_SPREAD_NOT_POSITIVE")
    if quote_size is not None and quote_size <= 0:
        blockers.append("QUOTE_SIZE_NOT_POSITIVE")
    if reward_payout_mismatch:
        blockers.append("REWARD_PAYOUT_MISMATCH")
    if estimated_net_profit is not None and estimated_net_profit <= 0:
        blockers.append("NON_POSITIVE_NET_PROFIT_AFTER_FEES")
    return _unique(blockers)


def _fee_role(value: str | None, default: str) -> str:
    text = (value or default).strip().lower()
    return text if text in {"maker", "taker"} else default


def _rate_for_role(
    role: str,
    *,
    maker_fee_rate: Decimal | None,
    taker_fee_rate: Decimal | None,
) -> Decimal | None:
    return maker_fee_rate if role == "maker" else taker_fee_rate


def _first_decimal(*values: Any) -> Decimal | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, ValueError):
            continue
        if parsed.is_finite():
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


def _first_bool(*values: Any) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return False


def _safe_mul(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    if left is None or right is None:
        return None
    return left * right


def _safe_add(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    if left is None or right is None:
        return None
    return left + right


def _safe_sub(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    if left is None or right is None:
        return None
    return left - right


def _float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _round_float(value: float | None) -> float | None:
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


def _one_line_verdict(status: str, estimated_net_profit: Decimal | None, blockers: list[str]) -> str:
    if status == READY:
        return (
            "FEE_RECONCILIATION_READY: estimated net profit after trading fees and gas costs is "
            f"{_float(estimated_net_profit)} USDC; can_submit_order=false."
        )
    return f"FEE_BLOCKER: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
