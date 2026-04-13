"""
auto_maker_loop — order_scoring_verifier
polyarb_lab / research_lines / auto_maker_loop / modules

Wrap is_order_scoring() with retry, structured output, and unified error path.
All scoring checks in the loop go through here — no ad-hoc is_order_scoring()
calls in mainline code.

Public interface
----------------
    verify(client, order_ids, max_retries=3) -> list[dict]

Output schema — one dict per order_id
--------------------------------------
    order_id   str          the order ID passed in
    scoring    bool|None    True/False from API; None if all retries exhausted
    attempts   int          number of API calls made
    error      str|None     last error detail; None if scoring is not None
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RETRY_SLEEP_SEC = 2   # fixed wait between retries


def verify(
    client: Any,
    order_ids: list[str],
    max_retries: int = 3,
) -> list[dict]:
    """
    Check is_order_scoring() for each order ID.

    Parameters
    ----------
    client : ClobClient
        Authenticated CLOB client.
    order_ids : list[str]
        One or two order IDs (bid + ask).  Empty strings are skipped.
    max_retries : int
        Maximum API attempts per order before returning scoring=None.

    Returns
    -------
    list[dict] — one entry per non-empty order_id, in input order.
    """
    results = []
    for oid in order_ids:
        if not oid or oid.startswith("NONE") or oid.startswith("reduce_only"):
            continue
        results.append(_check_one(client, oid, max_retries))
    return results


def _check_one(client: Any, order_id: str, max_retries: int) -> dict:
    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            from py_clob_client.clob_types import OrderScoringParams
            raw = client.is_order_scoring(OrderScoringParams(orderId=order_id))
            scoring: Optional[bool] = None
            if isinstance(raw, bool):
                scoring = raw
            elif isinstance(raw, dict):
                scoring = bool(raw.get("scoring") or raw.get("is_scoring"))
                if scoring is False:
                    logger.info(
                        "order_scoring raw NOT_SCORING order_id=%s: %s",
                        order_id[:16], raw,
                    )
            if scoring is not None:
                return {
                    "order_id": order_id,
                    "scoring":  scoring,
                    "attempts": attempt,
                    "error":    None,
                }
            last_error = f"unexpected_response_type:{type(raw).__name__}"
        except Exception as exc:
            last_error = str(exc)
            logger.debug("order_scoring_verifier: attempt %d/%d failed for %s: %s",
                         attempt, max_retries, order_id[:16], exc)
        if attempt < max_retries:
            time.sleep(_RETRY_SLEEP_SEC)

    logger.warning(
        "order_scoring_verifier: exhausted %d retries for %s — last_error=%s",
        max_retries, order_id[:16], last_error,
    )
    return {
        "order_id": order_id,
        "scoring":  None,
        "attempts": max_retries,
        "error":    last_error,
    }


def all_scoring(verify_results: list[dict]) -> bool:
    """Return True only if every result has scoring=True."""
    if not verify_results:
        return False
    return all(r["scoring"] is True for r in verify_results)


def format_summary(verify_results: list[dict]) -> str:
    """One-line summary string for logging/printing."""
    parts = []
    for r in verify_results:
        tag = "SCORING" if r["scoring"] else ("UNKNOWN" if r["scoring"] is None else "NOT_SCORING")
        parts.append(f"{r['order_id'][:12]}…={tag}(attempts={r['attempts']})")
    return "  ".join(parts) if parts else "(no orders checked)"
