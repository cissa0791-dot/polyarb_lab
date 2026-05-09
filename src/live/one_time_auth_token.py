from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "single_side_probe_authorization.v1"
TOKEN_SCHEMA_VERSION = "single_side_probe_token.v1"
REPORT_TYPE = "single_side_probe_authorization"

READY_STATUS = "SINGLE_SIDE_PROBE_AUTHORIZATION_READY"
BLOCKED_STATUS = "SINGLE_SIDE_PROBE_AUTHORIZATION_BLOCKED"

DEFAULT_MODE = "MAKER_SINGLE_SIDE_LIVE_REHEARSAL"
DEFAULT_SIDE = "BID_ONLY"
DEFAULT_TTL_SECONDS = 300
DEFAULT_TOKEN_PATH = Path("/tmp/polyarb_single_side_bid_probe_authorization.json")


def create_authorization_token(
    *,
    market_slug: str,
    max_live_risk_usdc: float,
    quote_price: float,
    quote_size: float,
    hold_seconds: float | None = None,
    planner_hash: str | None = None,
    planner_snapshot_ts: str | None = None,
    planner_expires_at: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    mode: str = DEFAULT_MODE,
    side: str = DEFAULT_SIDE,
    now: datetime | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Create a short-lived, single-use authorization token payload.

    The token contains no secrets and does not enable execution by itself. A
    future execution runner must validate and consume it atomically before any
    live submit path can run.
    """

    now = now or datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=int(ttl_seconds))
    payload: dict[str, Any] = {
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "created_at_utc": now.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "ttl_seconds": int(ttl_seconds),
        "nonce": nonce or secrets.token_hex(16),
        "status": "ISSUED_UNUSED",
        "mode": mode,
        "side": side,
        "market_slug": market_slug,
        "max_live_risk_usdc": _round(max_live_risk_usdc),
        "quote_price": _round(quote_price),
        "quote_size": _round(quote_size),
        "hold_seconds": None if hold_seconds is None else _round(hold_seconds),
        "planner_hash": planner_hash,
        "planner_snapshot_ts": planner_snapshot_ts,
        "planner_expires_at": planner_expires_at,
        "max_order_count": 1,
        "auto_retry": False,
        "maker_both_sides_live_allowed": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }
    payload["token_hash"] = token_hash(payload)
    return payload


def write_authorization_token(token: dict[str, Any], path: str | Path = DEFAULT_TOKEN_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(token, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_authorization_token(path: str | Path = DEFAULT_TOKEN_PATH) -> tuple[dict[str, Any], str | None]:
    source = Path(path)
    if not source.exists():
        return {}, "TOKEN_FILE_MISSING"
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}, "TOKEN_FILE_INVALID_JSON"
    if not isinstance(payload, dict):
        return {}, "TOKEN_FILE_NOT_OBJECT"
    return payload, None


def build_authorization_report(
    *,
    token: dict[str, Any] | None,
    token_file: str | Path = DEFAULT_TOKEN_PATH,
    expected_market_slug: str,
    expected_max_live_risk_usdc: float,
    expected_quote_price: float,
    expected_quote_size: float,
    expected_hold_seconds: float | None = None,
    expected_planner_hash: str | None = None,
    expected_mode: str = DEFAULT_MODE,
    expected_side: str = DEFAULT_SIDE,
    now: datetime | None = None,
    load_error: str | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    token = token or {}
    blockers: list[str] = []
    if load_error:
        blockers.append(load_error)
    hash_matches = bool(token) and token.get("token_hash") == token_hash(token)
    expires_at = _parse_ts(token.get("expires_at_utc"))
    ttl_remaining = None if expires_at is None else (expires_at - now).total_seconds()
    expired = ttl_remaining is None or ttl_remaining <= 0

    checks = {
        "token_present": bool(token),
        "token_hash_matches": hash_matches,
        "token_unused": token.get("status") == "ISSUED_UNUSED",
        "not_expired": not expired,
        "mode_matches": token.get("mode") == expected_mode,
        "side_matches": token.get("side") == expected_side,
        "market_matches": token.get("market_slug") == expected_market_slug,
        "max_live_risk_matches": _same_float(token.get("max_live_risk_usdc"), expected_max_live_risk_usdc),
        "quote_price_matches": _same_float(token.get("quote_price"), expected_quote_price),
        "quote_size_matches": _same_float(token.get("quote_size"), expected_quote_size),
        "hold_seconds_matches": expected_hold_seconds is None
        or _same_float(token.get("hold_seconds"), expected_hold_seconds),
        "planner_hash_matches": expected_planner_hash is None
        or (bool(expected_planner_hash) and token.get("planner_hash") == expected_planner_hash),
        "max_order_count_one": token.get("max_order_count") == 1,
        "auto_retry_false": token.get("auto_retry") is False,
        "both_sides_rejected": token.get("maker_both_sides_live_allowed") is False,
        "token_does_not_submit_order": token.get("can_submit_order") is False and token.get("live_order_sent") is False,
    }
    for key, ok in checks.items():
        if not ok:
            blockers.append(_blocker_for_check(key))

    status = READY_STATUS if not blockers else BLOCKED_STATUS
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "token_file": str(token_file),
        "status": status,
        "authorization_token_valid": status == READY_STATUS,
        "execution_release_ready": status == READY_STATUS,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "mode": expected_mode,
        "side": expected_side,
        "market_slug": expected_market_slug,
        "expected_planner_hash": expected_planner_hash,
        "token_planner_hash": token.get("planner_hash"),
        "expected_hold_seconds": None if expected_hold_seconds is None else _round(expected_hold_seconds),
        "token_hold_seconds": token.get("hold_seconds"),
        "ttl_remaining_seconds": None if ttl_remaining is None else round(ttl_remaining, 6),
        "token_status": token.get("status"),
        "token_hash_prefix": str(token.get("token_hash") or "")[:12] or None,
        "checks": checks,
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def token_hash(token: dict[str, Any]) -> str:
    payload = {key: value for key, value in token.items() if key != "token_hash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mark_token_expended(token: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    updated = dict(token)
    updated["status"] = "EXPENDED"
    updated["expended_at_utc"] = now.isoformat()
    updated["token_hash"] = token_hash(updated)
    return updated


def _parse_ts(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _same_float(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _round(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _blocker_for_check(check: str) -> str:
    return {
        "token_present": "AUTH_TOKEN_MISSING",
        "token_hash_matches": "AUTH_TOKEN_HASH_MISMATCH",
        "token_unused": "AUTH_TOKEN_ALREADY_EXPENDED",
        "not_expired": "AUTH_TOKEN_EXPIRED",
        "mode_matches": "AUTH_TOKEN_MODE_MISMATCH",
        "side_matches": "AUTH_TOKEN_SIDE_MISMATCH",
        "market_matches": "AUTH_TOKEN_MARKET_MISMATCH",
        "max_live_risk_matches": "AUTH_TOKEN_RISK_MISMATCH",
        "quote_price_matches": "AUTH_TOKEN_QUOTE_PRICE_MISMATCH",
        "quote_size_matches": "AUTH_TOKEN_QUOTE_SIZE_MISMATCH",
        "hold_seconds_matches": "AUTH_TOKEN_HOLD_SECONDS_MISMATCH",
        "planner_hash_matches": "AUTH_TOKEN_PLANNER_HASH_MISMATCH",
        "max_order_count_one": "AUTH_TOKEN_ORDER_COUNT_NOT_ONE",
        "auto_retry_false": "AUTH_TOKEN_AUTO_RETRY_NOT_FALSE",
        "both_sides_rejected": "AUTH_TOKEN_BOTH_SIDES_NOT_REJECTED",
        "token_does_not_submit_order": "AUTH_TOKEN_ILLEGAL_SUBMIT_FLAG",
    }.get(check, f"AUTH_TOKEN_{check.upper()}_FAILED")


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return "SINGLE_SIDE_PROBE_AUTHORIZATION_READY: token valid for one future BID probe; execution still unauthorized here."
    return f"SINGLE_SIDE_PROBE_AUTHORIZATION_BLOCKED: {', '.join(_unique(blockers)) or 'UNKNOWN'}."
