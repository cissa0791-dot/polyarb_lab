"""
auto_maker_loop — inventory_governor
polyarb_lab / research_lines / auto_maker_loop / modules

Single source of truth for inventory state.  Every placement decision in the
mainline reads from this module.  The governor does not place, cancel, or read
from APIs — it is pure arithmetic over the inputs it receives.

States (ordered by severity)
-----------------------------
NORMAL       total <= market_cap * normal_threshold (default 0.80)
             Both BID and ASK may be placed at full size.

SOFT_CAP     total > market_cap * normal_threshold AND total <= market_cap
             BID size is reduced by size_factor.  ASK may be placed at full.
             New BIDs are allowed but at reduced size.

REDUCE_ONLY  total > market_cap  OR  reward_cover_hours > reduce_only_rch_threshold
             BID suppressed.  Only ASK is allowed (inventory exit).
             Size stays at full for ASK.

STOP         total > global_cap  OR  consecutive errors >= stop_error_limit
             All placement suppressed.  Manual intervention required.

Same-side suppression
---------------------
If same_side_pending_shares > 0 (a BID is already live and unfilled),
bid_ok is forced False regardless of state.  Prevents stacking BIDs.

Public interface
----------------
    assess(params: InventoryParams) -> InventoryDecision

Config fields (all have defaults — only override what you need)
--------------------------------------------------------------
    market_cap              float   per-market hard cap on YES shares (default 400)
    global_cap              float   total across all markets (default 600; not enforced
                                    here — caller must pass combined total)
    normal_threshold        float   fraction of market_cap below which state=NORMAL (0.80)
    reduce_only_rch_threshold float  reward_cover_hours above which state=REDUCE_ONLY (24.0)
    stop_error_limit        int     consecutive errors that trigger STOP (3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_NORMAL      = "NORMAL"
STATE_SOFT_CAP    = "SOFT_CAP"
STATE_REDUCE_ONLY = "REDUCE_ONLY"
STATE_STOP        = "STOP"


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InventoryParams:
    """
    Inputs to the inventory governor for one assessment.

    All share counts are in shares (float), not raw units.
    """
    current_shares:           float           # YES shares currently held (this market)
    market_cap:               float = 400.0   # per-market hard cap
    global_cap:               float = 600.0   # global cap (caller passes combined total)
    global_total_shares:      float = 0.0     # combined shares across all markets
    normal_threshold:         float = 0.80    # NORMAL if current < market_cap * this
    same_side_pending_shares: float = 0.0     # shares in live unfilled BID (same side)
    reward_cover_hours:       Optional[float] = None
    reduce_only_rch_threshold: float = 24.0   # hours above which REDUCE_ONLY
    consecutive_errors:       int   = 0
    stop_error_limit:         int   = 3


@dataclass
class InventoryDecision:
    """
    Output of the inventory governor.

    Mainline reads bid_ok / ask_ok / size_factor before placement.
    """
    inventory_state: str        # NORMAL | SOFT_CAP | REDUCE_ONLY | STOP
    bid_ok:          bool       # True if a new BID may be placed
    ask_ok:          bool       # True if an ASK may be placed
    size_factor:     float      # multiply intended size by this (1.0 = full, 0.5 = half)
    reason:          str        # human-readable explanation
    excess_shares:   float      # max(0, current_shares - market_cap * normal_threshold)
    hard_stop:       bool       # True when state=STOP


# ---------------------------------------------------------------------------
# Core assessment
# ---------------------------------------------------------------------------

def assess(params: InventoryParams) -> InventoryDecision:
    """
    Evaluate current inventory state and return placement permissions.

    Parameters
    ----------
    params : InventoryParams

    Returns
    -------
    InventoryDecision
    """
    cur   = params.current_shares
    mcap  = params.market_cap
    gcap  = params.global_cap
    gtot  = params.global_total_shares
    norm  = params.normal_threshold
    rch   = params.reward_cover_hours
    errs  = params.consecutive_errors
    stop_lim = params.stop_error_limit
    pend  = params.same_side_pending_shares

    normal_ceiling = mcap * norm
    excess = max(0.0, cur - normal_ceiling)

    # ── STOP: global cap breached or consecutive error limit hit ──────────
    if cur > gcap or gtot > gcap or errs >= stop_lim:
        reason = (
            f"global_cap_breached({cur:.0f}>{gcap:.0f})" if cur > gcap or gtot > gcap
            else f"consecutive_errors={errs}>={stop_lim}"
        )
        return InventoryDecision(
            inventory_state=STATE_STOP,
            bid_ok=False, ask_ok=False,
            size_factor=0.0,
            reason=reason,
            excess_shares=excess,
            hard_stop=True,
        )

    # ── REDUCE_ONLY: market cap breached or reward cover too long ─────────
    rch_trigger = (
        rch is not None
        and rch > params.reduce_only_rch_threshold
        and excess > 0
    )
    if cur > mcap or rch_trigger:
        reason = (
            f"market_cap_breached({cur:.0f}>{mcap:.0f})" if cur > mcap
            else f"reward_cover_hours={rch:.1f}>{params.reduce_only_rch_threshold:.0f}"
        )
        return InventoryDecision(
            inventory_state=STATE_REDUCE_ONLY,
            bid_ok=False, ask_ok=True,
            size_factor=1.0,
            reason=reason,
            excess_shares=excess,
            hard_stop=False,
        )

    # ── SOFT_CAP: above normal ceiling but within market cap ──────────────
    if cur > normal_ceiling:
        # Size reduction: proportional to how far above the ceiling we are
        # factor = 1 - ((cur - normal_ceiling) / (mcap - normal_ceiling))
        # clamped to [0.25, 0.75]
        denom = mcap - normal_ceiling
        raw_factor = 1.0 - ((cur - normal_ceiling) / denom) if denom > 0 else 0.25
        size_factor = round(max(0.25, min(0.75, raw_factor)), 4)

        # Same-side suppression: don't stack BIDs
        bid_allowed = pend <= 0
        return InventoryDecision(
            inventory_state=STATE_SOFT_CAP,
            bid_ok=bid_allowed, ask_ok=True,
            size_factor=size_factor,
            reason=(
                f"soft_cap(cur={cur:.0f}>ceiling={normal_ceiling:.0f})"
                + (f"  same_side_suppressed(pending={pend:.0f})" if not bid_allowed else "")
            ),
            excess_shares=excess,
            hard_stop=False,
        )

    # ── NORMAL ────────────────────────────────────────────────────────────
    # Same-side suppression even in NORMAL state
    bid_allowed = pend <= 0
    return InventoryDecision(
        inventory_state=STATE_NORMAL,
        bid_ok=bid_allowed, ask_ok=True,
        size_factor=1.0,
        reason=(
            f"normal(cur={cur:.0f}<=ceiling={normal_ceiling:.0f})"
            + (f"  same_side_suppressed(pending={pend:.0f})" if not bid_allowed else "")
        ),
        excess_shares=excess,
        hard_stop=False,
    )
