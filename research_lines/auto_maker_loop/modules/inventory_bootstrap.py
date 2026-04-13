"""
auto_maker_loop — inventory_bootstrap
polyarb_lab / research_lines / auto_maker_loop / modules

Pre-cycle YES token seeder.

Buys YES tokens for the selected target via aggressive CLOB limit order
when balance < required_shares.  Called once per cycle, before _run_cycle().

Mechanism
---------
Places a BUY limit order priced at (live ask + 1 tick) — aggressive enough to
cross the spread and fill as a taker against any resting ask.  If no ask is
available at poll time, the GTC order remains open and fills when a seller
appears; the existing in-cycle inventory gate (_run_cycle line ~301) is the
hard stop if the position still isn't ready by execution time.

No web3.py.  No relayer.  No splitPosition.
Uses the same py_clob_client signing stack already live in the main loop.

Public interface
----------------
    check_needs_bootstrap(client, token_id, required) -> float
        Returns current balance_shares. Caller decides if bootstrap is needed.

    run_bootstrap(client, token_id, creds, required_shares, dry_run) -> dict
        Buys shortfall shares if below required.  Returns verdict dict.

Verdicts
--------
    ALREADY_READY           : balance >= required; nothing to do
    DRY_RUN                 : dry_run=True; shortfall computed but no order placed
    COLLATERAL_INSUFFICIENT : USDC.e balance < estimated cost; skip bootstrap
    ORDER_FAILED            : CLOB order placement rejected
    BOOTSTRAP_OK            : order placed and balance now >= required
    BOOTSTRAP_PARTIAL       : order placed but balance still below required
                              (GTC order is live; in-cycle gate is the backstop)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default seed target — matches run_auto_maker.py --base-inventory-shares default
SEED_SHARES: float = 200.0

# How long to wait for taker fill before leaving order open (GTC)
FILL_POLL_SECS: int   = 10
FILL_TIMEOUT_SECS: int = 120

BUY_SIDE  = "BUY"
TICK_SIZE = 0.01


# ---------------------------------------------------------------------------
# Lazy import — avoid circular dependency on scoring_activation
# ---------------------------------------------------------------------------

_sa = None


def _get_sa():
    global _sa
    if _sa is None:
        import importlib
        _sa = importlib.import_module(
            "research_lines.reward_aware_maker_probe.modules.scoring_activation"
        )
    return _sa


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def check_needs_bootstrap(
    client: Any,
    token_id: str,
    required: float = SEED_SHARES,
) -> float:
    """
    Read current YES token balance.  Returns balance_shares.
    Caller decides whether bootstrap is needed (balance < required).
    No side effects.
    """
    sa = _get_sa()
    inv = sa._check_sell_inventory(client, token_id, required_shares=required)
    return inv.get("balance_shares", 0.0)


def run_bootstrap(
    client: Any,
    token_id: str,
    creds: Any,
    required_shares: float = SEED_SHARES,
    dry_run: bool = False,
    price_ref: Optional[float] = None,
) -> dict:
    """
    Ensure YES token balance >= required_shares.

    Parameters
    ----------
    client         : authenticated py_clob_client ClobClient
    token_id       : YES token ID for the target market
    creds          : ActivationCredentials (used for logging only; client already auth'd)
    required_shares: target minimum balance (default SEED_SHARES = 200)
    dry_run        : if True, compute shortfall but do not place any order

    Returns
    -------
    dict with keys:
        verdict        : str  (see module docstring)
        balance_shares : float — current balance (post-action if live; pre-action if dry_run)
        shortfall      : float — shares still needed (0 if ALREADY_READY / BOOTSTRAP_OK)
        order_id       : str | None
        filled         : bool | None
        usdc_balance   : float | None — only present on COLLATERAL_INSUFFICIENT
        cost_estimate  : float | None — only present on COLLATERAL_INSUFFICIENT
    """
    sa = _get_sa()

    # ── 1. Current balance ────────────────────────────────────────────────
    inv = sa._check_sell_inventory(client, token_id, required_shares=required_shares)
    current = inv.get("balance_shares", 0.0)
    shortfall = max(0.0, required_shares - current)

    if shortfall <= 0.0:
        logger.info("bootstrap: ALREADY_READY  balance=%.1f >= required=%.1f", current, required_shares)
        return {
            "verdict": "ALREADY_READY",
            "balance_shares": current,
            "shortfall": 0.0,
            "order_id": None,
            "filled": None,
        }

    logger.info(
        "bootstrap: shortfall=%.1f shares  token=%s",
        shortfall, token_id[:20],
    )

    if dry_run:
        return {
            "verdict": "DRY_RUN",
            "balance_shares": current,
            "shortfall": shortfall,
            "order_id": None,
            "filled": None,
        }

    # ── 2. Verify collateral (USDC.e) covers estimated cost ──────────────
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams
        col = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL", signature_type=-1)
        )
        usdc_bal = float(col.get("balance", 0) or 0) / 1_000_000
    except Exception as exc:
        logger.warning("bootstrap: collateral balance check failed: %s", exc)
        usdc_bal = 0.0

    # Fetch live midpoint — abort if unavailable; never use a hardcoded fallback
    mid, mid_src = sa.fetch_midpoint(client, token_id, price_ref)
    if mid is None:
        logger.error(
            "bootstrap: PRICE_UNAVAILABLE — fetch_midpoint returned None "
            "(live=failed, book=failed, price_ref=%s); aborting to avoid overpay",
            price_ref,
        )
        return {
            "verdict": "PRICE_UNAVAILABLE",
            "balance_shares": current,
            "shortfall": shortfall,
            "order_id": None,
            "filled": None,
        }
    logger.info("bootstrap: mid=%.4f (src=%s)", mid, mid_src)
    estimated_ask = min(0.97, round(mid + 0.03, 2))
    cost_estimate = shortfall * estimated_ask

    if usdc_bal < cost_estimate:
        logger.warning(
            "bootstrap: COLLATERAL_INSUFFICIENT  have=%.4f USDC  need≈%.4f USDC",
            usdc_bal, cost_estimate,
        )
        return {
            "verdict": "COLLATERAL_INSUFFICIENT",
            "balance_shares": current,
            "shortfall": shortfall,
            "order_id": None,
            "filled": None,
            "usdc_balance": usdc_bal,
            "cost_estimate": cost_estimate,
        }

    # ── 3. Place aggressive BUY (crosses ask, fills as taker) ────────────
    buy_price = min(0.98, round(estimated_ask + TICK_SIZE, 2))
    logger.info(
        "bootstrap: placing BUY  price=%.4f  size=%.1f  token=%s",
        buy_price, shortfall, token_id[:20],
    )
    order_id, err = sa._place_order(client, token_id, buy_price, shortfall, BUY_SIDE)
    if not order_id:
        logger.error("bootstrap: ORDER_FAILED  error=%s", err)
        return {
            "verdict": "ORDER_FAILED",
            "balance_shares": current,
            "shortfall": shortfall,
            "order_id": None,
            "filled": False,
            "error": err,
        }

    logger.info(
        "bootstrap: BUY order placed  order_id=%s  price=%.4f  size=%.1f",
        order_id[:16], buy_price, shortfall,
    )

    # ── 4. Poll for taker fill (up to FILL_TIMEOUT_SECS) ─────────────────
    deadline = time.monotonic() + FILL_TIMEOUT_SECS
    filled = False
    while time.monotonic() < deadline:
        time.sleep(FILL_POLL_SECS)
        if sa._is_filled(client, order_id):
            filled = True
            logger.info("bootstrap: BUY order filled  order_id=%s", order_id[:16])
            break

    if not filled:
        # GTC order remains live — in-cycle gate is the backstop
        logger.info(
            "bootstrap: BUY order not yet filled — leaving GTC  order_id=%s",
            order_id[:16],
        )

    # ── 5. Verify final balance ───────────────────────────────────────────
    inv_after = sa._check_sell_inventory(client, token_id, required_shares=required_shares)
    final_bal = inv_after.get("balance_shares", 0.0)
    remaining_shortfall = max(0.0, required_shares - final_bal)

    verdict = "BOOTSTRAP_OK" if inv_after["verdict"] == "SELL_INVENTORY_READY" else "BOOTSTRAP_PARTIAL"
    logger.info(
        "bootstrap: %s  final_balance=%.1f  shortfall_remaining=%.1f",
        verdict, final_bal, remaining_shortfall,
    )

    return {
        "verdict": verdict,
        "balance_shares": final_bal,
        "shortfall": remaining_shortfall,
        "order_id": order_id,
        "filled": filled,
    }
