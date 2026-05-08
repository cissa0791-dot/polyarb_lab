from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from src.live.auth import CredentialError, assert_clob_v2_available, build_authenticated_client, load_live_credentials
from src.live.clob_compat import ClobClient
from src.live.client import clean_live_error_message


CLOB_HOST = "https://clob.polymarket.com"
REPORT_SCHEMA_VERSION = "auth_scope_readiness.v1"
REPORT_TYPE = "auth_scope_readiness"
DEFAULT_MAX_LIVE_RISK_USDC = 300.0
DEFAULT_FEE_BUFFER_USDC = 5.0
DEFAULT_CANCEL_BUFFER_USDC = 5.0

AuthScopeReader = Callable[[], "AuthScopeSnapshot"]


@dataclass(frozen=True)
class AuthScopeSnapshot:
    configured_api_key: str | None
    level_1_auth_ok: bool
    level_2_auth_ok: bool
    clob_v2_available: bool
    signer_address: str | None = None
    trading_api_keys: tuple[str, ...] = ()
    readonly_api_keys: tuple[str, ...] = ()
    rate_limit_degraded: bool = False
    errors: tuple[str, ...] = ()


def build_auth_scope_readiness_report(
    *,
    env_file_status: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    deposit_wallet_report: dict[str, Any] | None = None,
    host: str = CLOB_HOST,
    auth_reader: AuthScopeReader | None = None,
    max_live_risk_usdc: float = DEFAULT_MAX_LIVE_RISK_USDC,
    fee_buffer_usdc: float = DEFAULT_FEE_BUFFER_USDC,
    cancel_buffer_usdc: float = DEFAULT_CANCEL_BUFFER_USDC,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only API key / signer scope report.

    This validator proves identity and permission posture. It never creates,
    revokes, deposits, cancels, signs an order, or posts an order.
    """

    now = now or datetime.now(timezone.utc)
    env_map = os.environ if env is None else env
    deposit_wallet_report = deposit_wallet_report or {}
    funder_address = _clean_str(env_map.get("POLYMARKET_FUNDER"))
    signature_type = _optional_int(env_map.get("POLYMARKET_SIGNATURE_TYPE"))
    required_usdc = max_live_risk_usdc + fee_buffer_usdc + cancel_buffer_usdc
    available_usdc = _first_float(deposit_wallet_report.get("available_usdc"))
    deposit_wallet_address = _clean_str(deposit_wallet_report.get("deposit_wallet_address"))

    try:
        snapshot = auth_reader() if auth_reader else _read_live_auth_scope(
            host=host,
            signature_type=signature_type,
            funder=funder_address,
        )
    except Exception as exc:
        snapshot = AuthScopeSnapshot(
            configured_api_key=_clean_str(env_map.get("POLYMARKET_API_KEY")),
            level_1_auth_ok=False,
            level_2_auth_ok=False,
            clob_v2_available=False,
            errors=(clean_live_error_message(exc) or type(exc).__name__,),
        )

    configured_key = snapshot.configured_api_key or _clean_str(env_map.get("POLYMARKET_API_KEY"))
    trading_keys = tuple(str(key) for key in snapshot.trading_api_keys if key)
    readonly_keys = tuple(str(key) for key in snapshot.readonly_api_keys if key)
    configured_key_is_trading_key = bool(configured_key and configured_key in set(trading_keys))
    configured_key_is_readonly_key = bool(configured_key and configured_key in set(readonly_keys))
    funder_matches_deposit_wallet = bool(
        funder_address and deposit_wallet_address and funder_address.lower() == deposit_wallet_address.lower()
    )
    deposit_wallet_balance_sufficient = available_usdc is not None and available_usdc >= required_usdc
    is_signing_enabled = all(
        [
            snapshot.clob_v2_available,
            snapshot.level_1_auth_ok,
            snapshot.level_2_auth_ok,
            configured_key_is_trading_key,
            not configured_key_is_readonly_key,
        ]
    )

    checks = {
        "auth_report_present": True,
        "clob_v2_available": snapshot.clob_v2_available,
        "level_1_auth_ok": snapshot.level_1_auth_ok,
        "level_2_auth_ok": snapshot.level_2_auth_ok,
        "configured_key_present": bool(configured_key),
        "configured_key_is_trading_key": configured_key_is_trading_key,
        "configured_key_is_readonly_key": configured_key_is_readonly_key,
        "is_signing_enabled": is_signing_enabled,
        "funder_address_present": bool(funder_address),
        "deposit_wallet_report_present": bool(deposit_wallet_report),
        "deposit_wallet_address_present": bool(deposit_wallet_address),
        "funder_matches_deposit_wallet": funder_matches_deposit_wallet,
        "deposit_wallet_balance_present": available_usdc is not None,
        "deposit_wallet_balance_sufficient": deposit_wallet_balance_sufficient,
        "rate_limit_degraded": snapshot.rate_limit_degraded,
        "no_auth_errors": not snapshot.errors,
    }
    blockers = _blockers(checks)
    status = "AUTH_SCOPE_READY" if not blockers else "AUTH_SCOPE_BLOCKED"
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
        "host": host,
        "signature_type": signature_type,
        "is_signing_enabled": is_signing_enabled,
        "level_1_auth_ok": snapshot.level_1_auth_ok,
        "level_2_auth_ok": snapshot.level_2_auth_ok,
        "clob_v2_available": snapshot.clob_v2_available,
        "configured_api_key_present": bool(configured_key),
        "configured_api_key_fingerprint": _key_fingerprint(configured_key),
        "configured_key_is_trading_key": configured_key_is_trading_key,
        "configured_key_is_readonly_key": configured_key_is_readonly_key,
        "trading_key_count": len(trading_keys),
        "readonly_key_count": len(readonly_keys),
        "signer_address": _normalise_address(snapshot.signer_address),
        "funder_address": _normalise_address(funder_address),
        "deposit_wallet_address": _normalise_address(deposit_wallet_address),
        "funder_matches_deposit_wallet": funder_matches_deposit_wallet,
        "available_usdc": _round(available_usdc),
        "required_usdc": _round(required_usdc),
        "deposit_wallet_balance_sufficient": deposit_wallet_balance_sufficient,
        "rate_limit_degraded": snapshot.rate_limit_degraded,
        "abnormal_restrictions": _abnormal_restrictions(checks),
        "checks": checks,
        "blockers": blockers,
        "errors": tuple(_redacted_errors(snapshot.errors, configured_key=configured_key)),
        "env_file": env_file_status or {"loaded": False, "reason": "NOT_PROVIDED", "keys": []},
        "credential_source": {
            "type": "process_env_or_env_file",
            "secret_values_reported": False,
            "required_funder_var": "POLYMARKET_FUNDER",
        },
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def _read_live_auth_scope(
    *,
    host: str,
    signature_type: int | None,
    funder: str | None,
) -> AuthScopeSnapshot:
    creds = load_live_credentials()
    errors: list[str] = []
    clob_v2_available = True
    try:
        assert_clob_v2_available()
    except CredentialError as exc:
        clob_v2_available = False
        errors.append(clean_live_error_message(exc) or type(exc).__name__)

    level_1_auth_ok = False
    level_2_auth_ok = False
    signer_address: str | None = None
    trading_api_keys: tuple[str, ...] = ()
    readonly_api_keys: tuple[str, ...] = ()
    rate_limit_degraded = False

    try:
        l1_client = ClobClient(host=host, chain_id=creds.chain_id, key=creds.private_key)
        _safe_client_call(l1_client.assert_level_1_auth)
        level_1_auth_ok = True
        signer_address = _derive_signer_address(l1_client)
    except Exception as exc:
        errors.append(_clean_error(exc, creds))

    try:
        l2_client = build_authenticated_client(creds, host, signature_type=signature_type, funder=funder)
        _safe_client_call(l2_client.assert_level_2_auth)
        level_2_auth_ok = True
        trading_api_keys = tuple(_extract_key_list(_safe_client_call(l2_client.get_api_keys), "apiKeys"))
        readonly_api_keys = tuple(_extract_key_list(_safe_client_call(l2_client.get_readonly_api_keys), "readonlyApiKeys"))
    except Exception as exc:
        cleaned = _clean_error(exc, creds)
        errors.append(cleaned)
        rate_limit_degraded = _is_rate_limit(cleaned)

    return AuthScopeSnapshot(
        configured_api_key=creds.api_key,
        level_1_auth_ok=level_1_auth_ok,
        level_2_auth_ok=level_2_auth_ok,
        clob_v2_available=clob_v2_available,
        signer_address=signer_address,
        trading_api_keys=trading_api_keys,
        readonly_api_keys=readonly_api_keys,
        rate_limit_degraded=rate_limit_degraded,
        errors=tuple(errors),
    )


def _safe_client_call(method: Any, *args: Any, **kwargs: Any) -> Any:
    return method(*args, **kwargs)


def _derive_signer_address(client: Any) -> str | None:
    signer = getattr(client, "signer", None)
    address = getattr(signer, "address", None)
    if callable(address):
        return _clean_str(address())
    return _clean_str(address)


def _extract_key_list(raw: Any, key: str) -> list[str]:
    if isinstance(raw, dict):
        values = raw.get(key) or []
    else:
        values = getattr(raw, key, []) or []
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if item]


def _blockers(checks: dict[str, bool]) -> list[str]:
    out: list[str] = []
    if not checks["clob_v2_available"]:
        out.append("CLOB_V2_NOT_AVAILABLE")
    if not checks["level_1_auth_ok"]:
        out.append("LEVEL_1_AUTH_FAILED")
    if not checks["level_2_auth_ok"]:
        out.append("LEVEL_2_AUTH_FAILED")
    if not checks["configured_key_present"]:
        out.append("API_KEY_MISSING")
    if not checks["configured_key_is_trading_key"]:
        out.append("API_KEY_NOT_REGISTERED_FOR_TRADING")
    if checks["configured_key_is_readonly_key"]:
        out.append("API_KEY_IS_READONLY")
    if not checks["is_signing_enabled"]:
        out.append("SIGNING_NOT_ENABLED")
    if not checks["funder_address_present"]:
        out.append("FUNDER_ADDRESS_MISSING")
    if not checks["deposit_wallet_report_present"]:
        out.append("DEPOSIT_WALLET_REPORT_MISSING")
    elif not checks["deposit_wallet_address_present"]:
        out.append("DEPOSIT_WALLET_ADDRESS_MISSING")
    elif not checks["funder_matches_deposit_wallet"]:
        out.append("FUNDER_DEPOSIT_WALLET_MISMATCH")
    if not checks["deposit_wallet_balance_present"]:
        out.append("DEPOSIT_WALLET_BALANCE_MISSING")
    elif not checks["deposit_wallet_balance_sufficient"]:
        out.append("DEPOSIT_WALLET_BALANCE_BELOW_REQUIRED")
    if checks["rate_limit_degraded"]:
        out.append("AUTH_RATE_LIMIT_DEGRADED")
    if not checks["no_auth_errors"]:
        out.append("AUTH_SCOPE_READ_ERROR")
    return _unique(out)


def _abnormal_restrictions(checks: dict[str, bool]) -> list[str]:
    restrictions: list[str] = []
    if checks["configured_key_is_readonly_key"]:
        restrictions.append("READONLY_API_KEY")
    if checks["rate_limit_degraded"]:
        restrictions.append("RATE_LIMIT_DEGRADED")
    if not checks["configured_key_is_trading_key"]:
        restrictions.append("TRADING_KEY_NOT_LISTED")
    return restrictions


def _redacted_errors(errors: tuple[str, ...], *, configured_key: str | None) -> list[str]:
    return [_redact(str(error), configured_key) for error in errors if error]


def _clean_error(exc: Exception, creds: Any) -> str:
    text = clean_live_error_message(exc) or type(exc).__name__
    for value in (
        getattr(creds, "private_key", None),
        getattr(creds, "api_key", None),
        getattr(creds, "api_secret", None),
        getattr(creds, "api_passphrase", None),
    ):
        text = _redact(text, value)
    return text


def _redact(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(str(secret), "[REDACTED]")


def _key_fingerprint(api_key: str | None) -> dict[str, str] | None:
    if not api_key:
        return None
    text = str(api_key)
    return {
        "prefix": text[:4] + "****",
        "sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }


def _normalise_address(value: str | None) -> str | None:
    text = _clean_str(value)
    return text.lower() if text else None


def _is_rate_limit(value: str) -> bool:
    text = value.lower()
    return any(marker in text for marker in ("rate limit", "rate-limit", "http 429", "status=429", "1015"))


def _clean_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


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
    if status == "AUTH_SCOPE_READY":
        return "AUTH_SCOPE_READY: Level-2 auth, signing scope, trading key, and Deposit Wallet binding proven; can_submit_order=false."
    return f"AUTH_SCOPE_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; can_submit_order=false."
