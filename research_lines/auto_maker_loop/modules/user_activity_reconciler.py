"""
auto_maker_loop — user_activity_reconciler
polyarb_lab / research_lines / auto_maker_loop / modules

Minimal REST-based user-activity feed.
Replaces the planned ws_market_user_bridge for now — same data, REST polling
instead of websocket stream.  Provides the two fields inventory_governor was
blind to:

    same_side_pending_shares   open unmatched BID shares for the current token
    global_total_shares        YES shares held across all survivor markets

Also flags unrecorded inventory changes since the last runs.jsonl entry.

Sources
-------
1. client.get_orders(OpenOrderParams())
       All open orders for the authed account.
       Filtered locally: side=BUY, asset_id matches token_id, status not MATCHED/CANCELLED.
       Computes: same_side_pending_shares = sum(size - size_matched) for matching orders.

2. Polymarket data-API positions
       GET https://data-api.polymarket.com/positions?user={holder}&sizeThreshold=0.01
       Holder = creds.funder (sig_type=2) or client.signer.address() (EOA).
       Computes: position_by_token (token_id → shares), global_total_shares.

3. runs.jsonl last entry (optional)
       Compares inventory_after_shares from last completed cycle record
       against current data-API position.
       Computes: unrecorded_inventory_delta = current - last_recorded.
       Positive delta = unexpected acquisition since last cycle.
       Negative delta = unexpected sell/fill since last cycle.

Public interface
----------------
    reconcile(
        client, creds, survivor_data: dict,
        token_id: str,
        runs_jsonl_path=None,
    ) -> ReconcileResult

Output schema
-------------
    same_side_pending_shares  float       live unmatched BID shares (this token)
    open_bid_count            int         number of live BID orders (this token)
    global_total_shares       float       sum of YES shares across all survivor tokens
    position_by_token         dict        token_id → float shares (data-API)
    current_position_shares   float       data-API position for this token specifically
    unrecorded_inventory_delta float      current - last_runs_recorded (0 if no record)
    reconcile_ok              bool        True if both sources responded without error
    errors                    list[str]   any fetch errors encountered
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests as _req

logger = logging.getLogger(__name__)

_DATA_API_BASE   = "https://data-api.polymarket.com"
_ORDER_TERMINAL  = {"MATCHED", "FILLED", "CANCELLED", "CANCELED", "EXPIRED"}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReconcileResult:
    same_side_pending_shares:   float
    open_bid_count:             int
    global_total_shares:        float
    position_by_token:          dict        # token_id → shares
    current_position_shares:    float       # data-API position for the queried token
    unrecorded_inventory_delta: float       # current_position - last_runs_recorded
    reconcile_ok:               bool
    errors:                     list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def reconcile(
    client: Any,
    creds: Any,
    survivor_data: dict,
    token_id: str,
    runs_jsonl_path: Optional[str] = None,
) -> ReconcileResult:
    """
    Fetch live user activity and return reconciled inventory exposure fields.

    Parameters
    ----------
    client : ClobClient
        Authenticated CLOB client.
    creds : ActivationCredentials
        Used to resolve the holder address (funder or EOA).
    survivor_data : dict
        SURVIVOR_DATA from scoring_activation — keyed by full slug.
        Used to build the set of survivor token IDs for global aggregation.
    token_id : str
        The CTF token ID for the market being traded this cycle.
    runs_jsonl_path : str | Path | None
        Path to runs.jsonl for last-cycle comparison.  If None, unrecorded
        delta defaults to 0.

    Returns
    -------
    ReconcileResult
    """
    errors: list[str] = []
    all_survivor_token_ids = {d["token_id"] for d in survivor_data.values()}

    # ── Source 1: open BID orders from CLOB ──────────────────────────────
    pending_shares, open_bid_count, src1_err = _fetch_pending_bids(client, token_id)
    if src1_err:
        errors.append(f"open_orders:{src1_err}")

    # ── Source 2: positions from data-API ────────────────────────────────
    holder = _resolve_holder(client, creds)
    position_by_token, src2_err = _fetch_positions(holder, all_survivor_token_ids)
    if src2_err:
        errors.append(f"data_api:{src2_err}")

    current_position = position_by_token.get(token_id, 0.0)
    global_total     = sum(position_by_token.values())

    # ── Source 3: runs.jsonl last-cycle comparison ────────────────────────
    unrecorded_delta = 0.0
    if runs_jsonl_path:
        last_rec = _load_last_run(runs_jsonl_path)
        if last_rec is not None:
            last_inv = last_rec.get("inventory_after_shares")
            last_slug_tok = _slug_to_token(last_rec.get("slug", ""), survivor_data)
            if (
                last_inv is not None
                and last_slug_tok == token_id
            ):
                unrecorded_delta = round(current_position - float(last_inv), 4)
                if abs(unrecorded_delta) > 1.0:
                    logger.warning(
                        "reconciler: unrecorded inventory delta=%.2f "
                        "(current=%.2f last_recorded=%.2f)",
                        unrecorded_delta, current_position, last_inv,
                    )

    reconcile_ok = not bool(errors)
    return ReconcileResult(
        same_side_pending_shares=pending_shares,
        open_bid_count=open_bid_count,
        global_total_shares=global_total,
        position_by_token=position_by_token,
        current_position_shares=current_position,
        unrecorded_inventory_delta=unrecorded_delta,
        reconcile_ok=reconcile_ok,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Source 1: open BID orders
# ---------------------------------------------------------------------------

def _fetch_pending_bids(client: Any, token_id: str) -> tuple[float, int, Optional[str]]:
    """
    Return (pending_shares, open_bid_count, error_str).

    pending_shares = sum of (size - size_matched) for live BID orders
    whose asset_id matches token_id.
    """
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw = client.get_orders(OpenOrderParams())
        orders: list = (
            raw if isinstance(raw, list)
            else (raw.get("data") or [] if isinstance(raw, dict) else [])
        )
    except Exception as exc:
        return 0.0, 0, str(exc)

    pending   = 0.0
    bid_count = 0

    for o in orders:
        d = o if isinstance(o, dict) else (o.__dict__ if hasattr(o, "__dict__") else {})

        # Status filter: skip terminal orders
        status = str(d.get("status", "")).upper()
        if status in _ORDER_TERMINAL:
            continue

        # Side filter: BUY only
        side = str(d.get("side", "")).upper()
        if side != "BUY":
            continue

        # Token match
        asset = str(
            d.get("asset_id") or d.get("token_id")
            or d.get("market") or d.get("outcome_token_id") or ""
        )
        if not (asset == token_id or token_id in asset):
            continue

        # Remaining size
        total   = _f(d.get("original_size") or d.get("size") or 0)
        matched = _f(d.get("size_matched") or d.get("filled") or 0)
        remaining = max(0.0, total - matched)
        pending  += remaining
        bid_count += 1

    return round(pending, 4), bid_count, None


# ---------------------------------------------------------------------------
# Source 2: data-API positions
# ---------------------------------------------------------------------------

def _fetch_positions(
    holder: str,
    survivor_token_ids: set,
) -> tuple[dict, Optional[str]]:
    """
    Return (position_by_token, error_str).

    position_by_token maps each matched survivor token_id to shares held.
    Paginates up to 10 pages (cursor-based, stops on "LTE=").
    """
    url    = f"{_DATA_API_BASE}/positions"
    params = {"user": holder, "sizeThreshold": "0.01"}
    position_by_token: dict = {}

    cursor = None
    pages  = 0
    try:
        while pages < 10:
            req_params = dict(params)
            if cursor:
                req_params["next_cursor"] = cursor
            resp = _req.get(url, params=req_params, timeout=8)
            if resp.status_code != 200:
                return position_by_token, f"HTTP_{resp.status_code}"
            body = resp.json()
            rows = body if isinstance(body, list) else (body.get("data") or body.get("positions") or [])
            pages += 1

            for row in rows:
                asset = str(
                    row.get("asset") or row.get("asset_id")
                    or row.get("token_id") or row.get("outcomeIndex") or ""
                )
                if asset in survivor_token_ids:
                    size = _f(row.get("size") or row.get("shares") or row.get("balance") or 0)
                    position_by_token[asset] = round(size, 4)

            # Pagination
            cursor = body.get("next_cursor") if isinstance(body, dict) else None
            if not cursor or cursor in ("", "LTE="):
                break

        return position_by_token, None
    except Exception as exc:
        return position_by_token, str(exc)


# ---------------------------------------------------------------------------
# Source 3: runs.jsonl last record
# ---------------------------------------------------------------------------

def _load_last_run(path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    last = None
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return last


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_holder(client: Any, creds: Any) -> str:
    """Return the address that actually holds the tokens (funder or EOA)."""
    holder = getattr(creds, "funder", None)
    if holder:
        return holder
    try:
        return client.signer.address()
    except Exception:
        return ""


def _slug_to_token(slug: str, survivor_data: dict) -> Optional[str]:
    """Return token_id for a given slug, or None if not found."""
    d = survivor_data.get(slug)
    return d["token_id"] if d else None


def _f(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
