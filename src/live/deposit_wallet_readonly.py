from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from src.live.auth import build_authenticated_client, load_live_credentials
from src.live.client import clean_live_error_message
from src.live.clob_compat import AssetType, BalanceAllowanceParams


CLOB_HOST = "https://clob.polymarket.com"
REPORT_SCHEMA_VERSION = "deposit_wallet_readonly.v1"
REPORT_TYPE = "deposit_wallet_readonly"
MICRO_USDC = 1_000_000.0

BalanceReader = Callable[[], dict[str, Any]]


def build_deposit_wallet_readonly_report(
    *,
    env_file_status: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    host: str = CLOB_HOST,
    balance_reader: BalanceReader | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only Deposit Wallet balance report.

    The report is intentionally acquisition-only. It never calls deposit,
    approve, transfer, update allowance, cancel, or order-placement methods.
    """

    now = now or datetime.now(timezone.utc)
    env_map = os.environ if env is None else env
    signature_type = _optional_int(env_map.get("POLYMARKET_SIGNATURE_TYPE"))
    deposit_wallet_address = str(env_map.get("POLYMARKET_FUNDER") or "").strip() or None
    blockers: list[str] = []
    errors: list[str] = []
    balance_payload: dict[str, Any] = {}

    if not deposit_wallet_address:
        blockers.append("DEPOSIT_WALLET_ADDRESS_MISSING")

    if deposit_wallet_address:
        try:
            balance_payload = balance_reader() if balance_reader else _read_live_deposit_wallet_balance(
                host=host,
                signature_type=signature_type,
                funder=deposit_wallet_address,
            )
        except Exception as exc:
            message = clean_live_error_message(exc) or type(exc).__name__
            blockers.append("DEPOSIT_WALLET_BALANCE_READ_FAILED")
            errors.append(message)

    raw_balance = _first_float(balance_payload.get("raw_balance"))
    available_usdc = _first_float(balance_payload.get("available_usdc"))
    if available_usdc is None and raw_balance is not None:
        available_usdc = raw_balance / MICRO_USDC
    raw_allowance = _first_float(balance_payload.get("raw_allowance"))
    allowance_usdc = _first_float(balance_payload.get("allowance_usdc"))
    if allowance_usdc is None and raw_allowance is not None:
        allowance_usdc = raw_allowance / MICRO_USDC
    if deposit_wallet_address and available_usdc is None and "DEPOSIT_WALLET_BALANCE_READ_FAILED" not in blockers:
        blockers.append("DEPOSIT_WALLET_BALANCE_MISSING")

    status = "DEPOSIT_WALLET_READY" if not blockers else "DEPOSIT_WALLET_BLOCKED"
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
        "wallet_type": "DEPOSIT_WALLET" if deposit_wallet_address else "EOA_OR_UNKNOWN",
        "balance_source": "CLOB_GET_BALANCE_ALLOWANCE_COLLATERAL" if available_usdc is not None else None,
        "deposit_wallet_address": deposit_wallet_address,
        "signature_type": signature_type,
        "available_usdc": _round(available_usdc),
        "raw_balance": raw_balance,
        "raw_balance_units": "micro_usdc" if raw_balance is not None else None,
        "allowance_usdc": _round(allowance_usdc),
        "raw_allowance": raw_allowance,
        "env_file": env_file_status or {"loaded": False, "reason": "NOT_PROVIDED", "keys": []},
        "credential_source": {
            "type": "process_env_or_env_file",
            "required_address_var": "POLYMARKET_FUNDER",
            "secret_values_reported": False,
        },
        "blockers": _unique(blockers),
        "errors": errors,
        "one_line_verdict": _one_line_verdict(status, _unique(blockers)),
    }


def _read_live_deposit_wallet_balance(
    *,
    host: str,
    signature_type: int | None,
    funder: str,
) -> dict[str, Any]:
    creds = load_live_credentials()
    client = build_authenticated_client(creds, host, signature_type=signature_type, funder=funder)
    raw = _get_collateral_balance_allowance(client, signature_type=signature_type)
    raw_balance = _raw_field(raw, "balance")
    raw_allowance = _raw_field(raw, "allowance")
    balance = _first_float(raw_balance)
    allowance = _first_float(raw_allowance)
    return {
        "raw_balance": balance,
        "available_usdc": None if balance is None else balance / MICRO_USDC,
        "raw_allowance": allowance,
        "allowance_usdc": None if allowance is None else allowance / MICRO_USDC,
    }


def _get_collateral_balance_allowance(client: Any, *, signature_type: int | None) -> Any:
    attempts: list[dict[str, Any]] = []
    if signature_type is not None:
        attempts.append({"asset_type": AssetType.COLLATERAL, "signature_type": signature_type})
    attempts.append({"asset_type": AssetType.COLLATERAL})
    errors: list[str] = []
    for kwargs in attempts:
        try:
            params = BalanceAllowanceParams(**kwargs)
            return client.get_balance_allowance(params)
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError("CLOB_COLLATERAL_BALANCE_PARAMS_UNSUPPORTED: " + "; ".join(errors))


def _raw_field(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


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
    if status == "DEPOSIT_WALLET_READY":
        return "DEPOSIT_WALLET_READY: read-only collateral balance acquired; can_submit_order=false."
    return f"DEPOSIT_WALLET_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
