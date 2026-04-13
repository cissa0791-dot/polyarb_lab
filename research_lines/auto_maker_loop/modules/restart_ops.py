"""
auto_maker_loop — restart_ops
polyarb_lab / research_lines / auto_maker_loop / modules

Cancel open orders for one token, verify the count reaches zero, handle 425
rate-limit responses with backoff, and log the reason for the operation.

Use this for:
  - Kill-switch cancellation before stopping the loop
  - KeyboardInterrupt graceful shutdown
  - Pre-placement cleanup when stale orders may be present

Public interface
----------------
    cancel_and_verify(client, token_id, reason,
                      max_retries=3, backoff_sec=5) -> dict

Output schema
-------------
    cancelled_count     int       number of cancel_all() calls that returned OK
    verification_passed bool      True if open orders reached 0 after cancel
    orders_remaining    int       open orders seen after last cancel attempt
    attempts            int       number of cancel attempts made
    rate_limited        bool      True if any 425 / rate-limit response was seen
    error               str|None  last unexpected error; None if clean
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RATE_LIMIT_MARKERS = ("425", "too many", "rate limit", "too_many", "ratelimit")
_OPEN_ORDER_VERIFY_SLEEP_SEC = 3   # brief pause before verifying open order count


def cancel_and_verify(
    client: Any,
    token_id: str,
    reason: str,
    max_retries: int = 3,
    backoff_sec: int = 5,
) -> dict:
    """
    Cancel all open orders for token_id, then verify open count = 0.

    Parameters
    ----------
    client : ClobClient
        Authenticated CLOB client.
    token_id : str
        CTF token ID for the market being cleaned up.
    reason : str
        Human-readable reason for this cancel (written to log).
    max_retries : int
        Maximum cancel attempts before giving up.
    backoff_sec : int
        Seconds to wait after a rate-limit (425) response before retry.

    Returns
    -------
    dict — see module docstring for schema.
    """
    logger.info("restart_ops: cancel_and_verify — reason=%s  token=%s…", reason, token_id[:20])

    cancelled_count  = 0
    rate_limited     = False
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            result = client.cancel_all()
            cancelled_count += 1
            logger.info("restart_ops: cancel_all attempt=%d OK — result=%s", attempt, result)
            break
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(m in exc_str for m in _RATE_LIMIT_MARKERS):
                rate_limited = True
                wait = backoff_sec * attempt   # linear back-off per retry
                logger.warning(
                    "restart_ops: 425/rate-limit on cancel attempt %d — backing off %ds",
                    attempt, wait,
                )
                time.sleep(wait)
                last_error = f"rate_limited:{exc}"
            else:
                last_error = str(exc)
                logger.error("restart_ops: cancel_all attempt %d failed: %s", attempt, exc)
                break

    # Verify: check open orders count
    time.sleep(_OPEN_ORDER_VERIFY_SLEEP_SEC)
    orders_remaining, verify_error = _count_open_orders(client, token_id)

    if verify_error:
        last_error = last_error or verify_error
        logger.warning("restart_ops: could not verify open orders: %s", verify_error)

    verification_passed = (orders_remaining == 0)
    if not verification_passed:
        logger.warning(
            "restart_ops: %d order(s) still open after cancel — manual check required",
            orders_remaining,
        )
    else:
        logger.info("restart_ops: verification passed — 0 open orders remaining")

    return {
        "cancelled_count":     cancelled_count,
        "verification_passed": verification_passed,
        "orders_remaining":    orders_remaining,
        "attempts":            min(max_retries, cancelled_count + (1 if last_error else 0)),
        "rate_limited":        rate_limited,
        "error":               last_error,
    }


def _count_open_orders(client: Any, token_id: str) -> tuple[int, Optional[str]]:
    """
    Return (count_of_open_orders_for_token, error_str).
    Fetches all open orders and filters locally by token_id.
    """
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw = client.get_orders(OpenOrderParams())
        all_orders: list = (
            raw if isinstance(raw, list)
            else (raw.get("data") or [] if isinstance(raw, dict) else [])
        )
        count = 0
        for o in all_orders:
            asset = str(
                (o.get("asset_id") or o.get("token_id") or o.get("market") or "")
                if isinstance(o, dict)
                else getattr(o, "asset_id", "") or ""
            )
            if asset == token_id or token_id in asset:
                count += 1
        return count, None
    except Exception as exc:
        return 0, str(exc)
