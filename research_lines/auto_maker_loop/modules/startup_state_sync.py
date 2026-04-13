"""
auto_maker_loop — startup_state_sync
polyarb_lab / research_lines / auto_maker_loop / modules

Read-only startup state sync.  Called once at process start before the
first cycle.  Consolidates three truth sources into one diagnostic block:

  1. CLOB   GET /balance-allowance (CONDITIONAL)  — usable token balance per survivor
  2. Data API GET /positions                        — on-chain position per survivor
  3. CLOB   GET /orders (open)                     — pending BID / ASK exposure

Computes effective_inventory = data_api_position - open_ask_shares.
This is the amount available to place new ASK orders.  If the CLOB balance
equals the Data API position, the state is consistent.  Divergence indicates
an address mismatch, unapproved allowance, or stale on-chain sync.

Non-goals:
  - Does NOT split / merge / redeem tokens.
  - Does NOT modify any local state.
  - Does NOT trigger retry logic.

Public interface
----------------
    sync(client, creds, survivor_data) -> StartupState

Output schema (StartupState)
----------------------------
    tokens                 list[TokenState]  — one entry per survivor
    total_position_shares  float             — sum of data-API positions
    total_clob_balance     float             — sum of CLOB balances
    total_open_bid_shares  float             — total pending BID exposure
    total_open_ask_shares  float             — total pending ASK exposure
    sync_ok                bool              — True if all sources responded
    errors                 list[str]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DATA_API_BASE  = "https://data-api.polymarket.com"
_ORDER_TERMINAL = {"MATCHED", "FILLED", "CANCELLED", "CANCELED", "EXPIRED"}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class TokenState:
    slug:                    str
    token_id:                str
    clob_balance_shares:     float    # GET /balance-allowance CONDITIONAL
    data_api_position:       float    # GET /positions (on-chain ground truth)
    open_bid_shares:         float    # unmatched open BID orders
    open_ask_shares:         float    # unmatched open ASK orders
    effective_inventory:     float    # data_api_position - open_ask_shares
    clob_balance_ok:         bool     # CLOB fetch succeeded without error
    balance_mismatch:        bool     # data_api_position != clob_balance_shares


@dataclass
class StartupState:
    tokens:                  list[TokenState]
    total_position_shares:   float            # sum data-API positions
    total_clob_balance:      float            # sum CLOB balances
    total_open_bid_shares:   float
    total_open_ask_shares:   float
    holder_address:          str              # address used for Data API query
    sync_ok:                 bool
    errors:                  list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def sync(
    client: Any,
    creds: Any,
    survivor_data: dict,
) -> StartupState:
    """
    Fetch and print the startup state diagnostic.  Returns StartupState.

    Parameters
    ----------
    client        : authenticated ClobClient
    creds         : ActivationCredentials (used to resolve holder address)
    survivor_data : SURVIVOR_DATA dict from scoring_activation
    """
    errors: list[str] = []

    # ── Resolve holder address ────────────────────────────────────────────
    holder = _resolve_holder(client, creds)

    # ── RAW AUDIT: Hungary-only truth dump (temporary diagnostic) ─────────
    _raw_audit_hungary(client, creds, survivor_data, holder)

    # ── Source 1: Data API positions (one call for all survivors) ─────────
    all_token_ids = {d["token_id"] for d in survivor_data.values()}
    position_by_token, pos_err = _fetch_positions(holder, all_token_ids)
    if pos_err:
        errors.append(f"data_api_positions:{pos_err}")

    # ── Source 2: CLOB open orders (one call for all) ─────────────────────
    open_bids_by_token, open_asks_by_token, orders_err = _fetch_open_orders(
        client, all_token_ids
    )
    if orders_err:
        errors.append(f"clob_open_orders:{orders_err}")

    # ── Source 3: CLOB balance-allowance (one call per token) ────────────
    tokens: list[TokenState] = []
    for slug, d in survivor_data.items():
        tid = d["token_id"]
        clob_bal, bal_ok, bal_err = _fetch_clob_balance(client, tid)
        if bal_err:
            errors.append(f"clob_balance_{slug[:20]}:{bal_err}")

        data_pos = position_by_token.get(tid, 0.0)
        bid_exp  = open_bids_by_token.get(tid, 0.0)
        ask_exp  = open_asks_by_token.get(tid, 0.0)
        eff_inv  = round(data_pos - ask_exp, 4)
        mismatch = abs(data_pos - clob_bal) > 0.5  # >0.5 share divergence is notable

        tokens.append(TokenState(
            slug=slug,
            token_id=tid,
            clob_balance_shares=clob_bal,
            data_api_position=data_pos,
            open_bid_shares=bid_exp,
            open_ask_shares=ask_exp,
            effective_inventory=eff_inv,
            clob_balance_ok=bal_ok,
            balance_mismatch=mismatch,
        ))

    state = StartupState(
        tokens=tokens,
        total_position_shares=sum(t.data_api_position for t in tokens),
        total_clob_balance=sum(t.clob_balance_shares for t in tokens),
        total_open_bid_shares=sum(t.open_bid_shares for t in tokens),
        total_open_ask_shares=sum(t.open_ask_shares for t in tokens),
        holder_address=holder,
        sync_ok=not bool(errors),
        errors=errors,
    )

    _print_diagnostic(state)
    return state


# ---------------------------------------------------------------------------
# RAW AUDIT — Hungary only (temporary, no filtering, no logic)
# ---------------------------------------------------------------------------

def _raw_audit_hungary(
    client: Any,
    creds: Any,
    survivor_data: dict,
    holder: str,
) -> None:
    """
    Print raw, unfiltered payloads for Hungary to diagnose inventory mismatch.
    Temporary diagnostic — safe to remove once root cause is identified.
    """
    import requests as _req
    import json as _json

    HUNGARY_SLUG_FRAGMENT = "hungary"  # matches any slug containing this

    # ── Identify Hungary entry ────────────────────────────────────────────
    hungary_slug = None
    hungary_data = None
    for slug, d in survivor_data.items():
        if HUNGARY_SLUG_FRAGMENT in slug.lower():
            hungary_slug = slug
            hungary_data = d
            break

    print("\n" + "#" * 70)
    print("  RAW AUDIT — HUNGARY TRUTH DUMP")
    print("#" * 70)

    # ── Address resolution ────────────────────────────────────────────────
    funder_attr = getattr(creds, "funder", None)
    signer_addr = None
    try:
        signer_addr = client.signer.address()
    except Exception as exc:
        signer_addr = f"<error: {exc}>"

    print(f"  creds.funder      : {funder_attr!r}")
    print(f"  signer.address()  : {signer_addr!r}")
    print(f"  resolved holder   : {holder!r}")

    if not hungary_slug:
        print("  !! No Hungary slug found in survivor_data — keys:")
        for k in survivor_data:
            print(f"       {k}")
        print("#" * 70 + "\n")
        return

    token_id     = hungary_data.get("token_id", "<missing>")
    condition_id = hungary_data.get("condition_id", "<missing>")
    print(f"\n  hungary slug      : {hungary_slug}")
    print(f"  token_id          : {token_id!r}")
    print(f"  condition_id      : {condition_id!r}")

    # ── Raw Data API /positions (first page, unfiltered) ─────────────────
    print("\n--- RAW DATA API /positions (first page, no filter) ---")
    try:
        url    = f"{_DATA_API_BASE}/positions"
        params = {"user": holder, "sizeThreshold": "0.01"}
        resp   = _req.get(url, params=params, timeout=8)
        print(f"  status_code : {resp.status_code}")
        print(f"  url         : {resp.url}")
        try:
            body = resp.json()
            rows = (
                body if isinstance(body, list)
                else (body.get("data") or body.get("positions") or [])
            )
            print(f"  rows_on_page1 : {len(rows)}")
            if isinstance(body, dict):
                next_cur = body.get("next_cursor")
                print(f"  next_cursor   : {next_cur!r}")
            # Print ALL rows (field names + values) so we can see the key names
            print("  --- all rows ---")
            for i, row in enumerate(rows):
                print(f"    row[{i}] keys={list(row.keys() if isinstance(row, dict) else [])}  raw={row}")
            if not rows:
                print("  (no rows returned — printing raw body)")
                print(f"  raw_body : {_json.dumps(body)[:2000]}")
        except Exception as parse_exc:
            print(f"  parse_error : {parse_exc}")
            print(f"  raw_text    : {resp.text[:1000]}")
    except Exception as exc:
        print(f"  request_error : {exc}")

    # ── Raw CLOB /balance-allowance CONDITIONAL for Hungary token ─────────
    print("\n--- RAW CLOB get_balance_allowance (CONDITIONAL, Hungary token_id) ---")
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams
        resp_bal = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type="CONDITIONAL",
                token_id=token_id,
                signature_type=-1,
            )
        )
        print(f"  raw response : {resp_bal!r}")
        if isinstance(resp_bal, dict):
            print(f"  keys         : {list(resp_bal.keys())}")
            # Dump allowances sub-dict if present
            allowances = resp_bal.get("allowances")
            print(f"  allowances   : {allowances!r}")
    except Exception as exc:
        print(f"  error : {exc}")

    # ── Also try with condition_id as token_id ────────────────────────────
    if condition_id and condition_id != "<missing>" and condition_id != token_id:
        print("\n--- RAW CLOB get_balance_allowance (CONDITIONAL, condition_id as token_id) ---")
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams
            resp_cid = client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=condition_id,
                    signature_type=-1,
                )
            )
            print(f"  raw response : {resp_cid!r}")
        except Exception as exc:
            print(f"  error : {exc}")

    # ── Raw CLOB open orders (first 10 entries, all fields) ───────────────
    print("\n--- RAW CLOB get_orders (OpenOrderParams, first 10) ---")
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw_orders = client.get_orders(OpenOrderParams())
        orders: list = (
            raw_orders if isinstance(raw_orders, list)
            else (raw_orders.get("data") or [] if isinstance(raw_orders, dict) else [])
        )
        print(f"  total_orders_returned : {len(orders)}")
        for i, o in enumerate(orders[:10]):
            d = o if isinstance(o, dict) else (o.__dict__ if hasattr(o, "__dict__") else {})
            print(f"    order[{i}] keys={list(d.keys())}  asset_id={d.get('asset_id') or d.get('token_id') or '?'}  side={d.get('side')}  status={d.get('status')}  size={d.get('size') or d.get('original_size')}")
        if not orders:
            print("  (no open orders returned)")
    except Exception as exc:
        print(f"  error : {exc}")

    print("#" * 70 + "\n")


# ---------------------------------------------------------------------------
# Diagnostic printer
# ---------------------------------------------------------------------------

def _print_diagnostic(state: StartupState) -> None:
    print("\n" + "=" * 60)
    print("  STARTUP STATE SYNC")
    print(f"  holder              : {state.holder_address or 'unknown'}")
    print(f"  sync_ok             : {state.sync_ok}")
    if state.errors:
        for e in state.errors:
            print(f"  error               : {e}")
    print("=" * 60)

    for t in state.tokens:
        short_slug = t.slug.split("-")[-3:][:2]
        label = "-".join(short_slug) if short_slug else t.slug[:22]
        mismatch_tag = "  !! BALANCE_MISMATCH" if t.balance_mismatch else ""
        print(
            f"  {label:<22}"
            f"  clob={t.clob_balance_shares:>7.1f}"
            f"  data_api={t.data_api_position:>7.1f}"
            f"  bid_exp={t.open_bid_shares:>6.1f}"
            f"  ask_exp={t.open_ask_shares:>6.1f}"
            f"  eff_inv={t.effective_inventory:>7.1f}"
            f"{mismatch_tag}"
        )

    print("-" * 60)
    print(
        f"  {'TOTAL':<22}"
        f"  clob={state.total_clob_balance:>7.1f}"
        f"  data_api={state.total_position_shares:>7.1f}"
        f"  bid_exp={state.total_open_bid_shares:>6.1f}"
        f"  ask_exp={state.total_open_ask_shares:>6.1f}"
    )

    # Verdict line
    ready_tokens = [t for t in state.tokens if t.effective_inventory >= 200.0]
    if ready_tokens:
        print(f"\n  INVENTORY VERDICT   : READY for bilateral quoting on "
              f"{len(ready_tokens)} token(s)")
        for t in ready_tokens:
            print(f"    ready             : {t.slug[:48]}"
                  f"  eff_inv={t.effective_inventory:.0f}")
    else:
        print("\n  INVENTORY VERDICT   : NO TOKEN has >= 200 shares of effective inventory")
        if state.total_position_shares > 0:
            print("    !! Data API shows positions exist but none meet 200-share threshold")
            for t in state.tokens:
                if t.data_api_position > 0:
                    print(f"    position found    : {t.slug[:48]}"
                          f"  data_api={t.data_api_position:.0f}"
                          f"  clob={t.clob_balance_shares:.0f}")
        else:
            print("    !! Data API shows zero positions across all survivor tokens")
            print("    !! To bilateral-quote, you need >= 200 YES shares per market")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Source: Data API positions
# ---------------------------------------------------------------------------

def _fetch_positions(
    holder: str,
    token_ids: set,
) -> tuple[dict, Optional[str]]:
    """Return (position_by_token, error_str).  Paginates up to 10 pages."""
    try:
        import requests as _req
        url    = f"{_DATA_API_BASE}/positions"
        params = {"user": holder, "sizeThreshold": "0.01"}
        result: dict = {}
        cursor: Optional[str] = None
        pages = 0

        while pages < 10:
            req_params = dict(params)
            if cursor:
                req_params["next_cursor"] = cursor
            resp = _req.get(url, params=req_params, timeout=8)
            if resp.status_code != 200:
                return result, f"HTTP_{resp.status_code}"
            body = resp.json()
            rows = (
                body if isinstance(body, list)
                else (body.get("data") or body.get("positions") or [])
            )
            pages += 1
            for row in rows:
                asset = str(
                    row.get("asset") or row.get("asset_id")
                    or row.get("token_id") or row.get("conditionId") or ""
                )
                if asset in token_ids:
                    size = _f(
                        row.get("size") or row.get("shares")
                        or row.get("balance") or 0
                    )
                    result[asset] = round(size, 4)
            cursor = body.get("next_cursor") if isinstance(body, dict) else None
            if not cursor or cursor in ("", "LTE="):
                break

        return result, None
    except Exception as exc:
        return {}, str(exc)


# ---------------------------------------------------------------------------
# Source: CLOB open orders
# ---------------------------------------------------------------------------

def _fetch_open_orders(
    client: Any,
    token_ids: set,
) -> tuple[dict, dict, Optional[str]]:
    """
    Return (bids_by_token, asks_by_token, error_str).

    Both dicts map token_id → float unmatched shares.
    """
    bids: dict = {}
    asks: dict = {}
    try:
        from py_clob_client.clob_types import OpenOrderParams
        raw = client.get_orders(OpenOrderParams())
        orders: list = (
            raw if isinstance(raw, list)
            else (raw.get("data") or [] if isinstance(raw, dict) else [])
        )
    except Exception as exc:
        return bids, asks, str(exc)

    for o in orders:
        d = o if isinstance(o, dict) else (o.__dict__ if hasattr(o, "__dict__") else {})

        status = str(d.get("status", "")).upper()
        if status in _ORDER_TERMINAL:
            continue

        asset = str(
            d.get("asset_id") or d.get("token_id")
            or d.get("market") or d.get("outcome_token_id") or ""
        )
        if asset not in token_ids:
            continue

        total     = _f(d.get("original_size") or d.get("size") or 0)
        matched   = _f(d.get("size_matched") or d.get("filled") or 0)
        remaining = max(0.0, total - matched)
        side      = str(d.get("side", "")).upper()

        if side == "BUY":
            bids[asset] = bids.get(asset, 0.0) + remaining
        elif side == "SELL":
            asks[asset] = asks.get(asset, 0.0) + remaining

    return bids, asks, None


# ---------------------------------------------------------------------------
# Source: CLOB balance-allowance
# ---------------------------------------------------------------------------

def _fetch_clob_balance(
    client: Any,
    token_id: str,
) -> tuple[float, bool, Optional[str]]:
    """Return (balance_shares, ok, error_str)."""
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams
        # Sync CLOB cache from chain first (non-destructive GET)
        try:
            client.update_balance_allowance(
                BalanceAllowanceParams(
                    asset_type="CONDITIONAL",
                    token_id=token_id,
                    signature_type=-1,
                )
            )
        except Exception:
            pass  # non-fatal
        resp = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type="CONDITIONAL",
                token_id=token_id,
                signature_type=-1,
            )
        )
        if not isinstance(resp, dict):
            return 0.0, False, f"unexpected_type:{type(resp).__name__}"
        balance_raw = int(resp.get("balance") or 0)
        return round(balance_raw / 1_000_000, 4), True, None
    except Exception as exc:
        return 0.0, False, str(exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_holder(client: Any, creds: Any) -> str:
    holder = getattr(creds, "funder", None)
    if holder:
        return holder
    try:
        return client.signer.address()
    except Exception:
        return ""


def _f(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
