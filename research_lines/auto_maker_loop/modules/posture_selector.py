"""
auto_maker_loop — posture_selector
polyarb_lab / research_lines / auto_maker_loop / modules

Selects the quoting posture for each cycle based on inventory state, reward
config quality, and market suitability.  Returns one of four postures.

Postures
--------
BILATERAL         Both BID and ASK are placed at standard prices.
                  Requires: inventory_state=NORMAL, reward zone known, suitable.

ASYMMETRIC_ASK_LEAN  ASK is placed closer to midpoint; BID is placed further.
                  Applies when SOFT_CAP but inventory governor permits BID.
                  Earns reward with lower fill probability on BID side.

EXIT_ONLY         Only ASK is placed.  BID suppressed.
                  Applies when inventory_state is REDUCE_ONLY.
                  Also applies when reward_config_ok=False AND competitiveness > threshold.

SKIP              No orders placed this cycle.
                  Applies when: STOP state, not suitable, or missing config.

Decision logic (evaluated top to bottom — first match wins)
-----------------------------------------------------------
1. inventory_state == STOP                             → SKIP
2. not suitable                                        → SKIP
3. inventory_state == REDUCE_ONLY                      → EXIT_ONLY
4. reward_config_ok=False AND competitiveness > 200    → EXIT_ONLY
   (quoting without known reward zone risks non-scoring placement)
5. inventory_state == SOFT_CAP AND bid_ok              → ASYMMETRIC_ASK_LEAN
6. inventory_state in (NORMAL, SOFT_CAP) AND bid_ok    → BILATERAL
7. bid_ok=False AND ask_ok                             → EXIT_ONLY
8. fallback                                            → SKIP

Public interface
----------------
    select(params: PostureParams) -> PostureDecision
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Posture constants
POSTURE_BILATERAL          = "BILATERAL"
POSTURE_ASYMMETRIC_ASK_LEAN = "ASYMMETRIC_ASK_LEAN"
POSTURE_EXIT_ONLY          = "EXIT_ONLY"
POSTURE_SKIP               = "SKIP"

_HIGH_COMPETITIVENESS = 200.0


@dataclass
class PostureParams:
    """Inputs consumed by the posture selector."""
    inventory_state:  str           # from inventory_governor: NORMAL|SOFT_CAP|REDUCE_ONLY|STOP
    bid_ok:           bool          # from inventory_governor
    ask_ok:           bool          # from inventory_governor
    suitable:         bool          # from market_suitability_snapshot
    reward_config_ok: bool          # from reward_config_digest
    quotes_in_zone:   Optional[bool] = None   # from reward_config_digest + midpoint
    competitiveness:  Optional[float] = None


@dataclass
class PostureDecision:
    """Output of the posture selector."""
    posture:   str    # BILATERAL | ASYMMETRIC_ASK_LEAN | EXIT_ONLY | SKIP
    place_bid: bool
    place_ask: bool
    reason:    str


def select(params: PostureParams) -> PostureDecision:
    """
    Select the quoting posture for this cycle.

    Parameters
    ----------
    params : PostureParams

    Returns
    -------
    PostureDecision
    """
    inv   = params.inventory_state
    bid_ok = params.bid_ok
    ask_ok = params.ask_ok
    suit  = params.suitable
    rok   = params.reward_config_ok
    comp  = params.competitiveness or 0.0

    # 1. Hard stop state
    if inv == "STOP":
        return PostureDecision(
            posture=POSTURE_SKIP,
            place_bid=False, place_ask=False,
            reason=f"inventory_state=STOP — manual intervention required",
        )

    # 2. Not suitable
    if not suit:
        return PostureDecision(
            posture=POSTURE_SKIP,
            place_bid=False, place_ask=False,
            reason="market_suitability=False — midpoint or reward zone check failed",
        )

    # 3. REDUCE_ONLY
    if inv == "REDUCE_ONLY":
        return PostureDecision(
            posture=POSTURE_EXIT_ONLY,
            place_bid=False, place_ask=ask_ok,
            reason=f"inventory_state=REDUCE_ONLY — ASK only",
        )

    # 4. Fallback config + high competition → don't expand inventory
    if not rok and comp > _HIGH_COMPETITIVENESS:
        return PostureDecision(
            posture=POSTURE_EXIT_ONLY,
            place_bid=False, place_ask=ask_ok,
            reason=(
                f"reward_config_ok=False AND competitiveness={comp:.0f}>{_HIGH_COMPETITIVENESS:.0f} "
                f"— quoting without live reward zone risks non-scoring BID"
            ),
        )

    # 5. SOFT_CAP with BID permitted → lean toward ASK
    if inv == "SOFT_CAP" and bid_ok:
        return PostureDecision(
            posture=POSTURE_ASYMMETRIC_ASK_LEAN,
            place_bid=True, place_ask=True,
            reason=f"inventory_state=SOFT_CAP — asymmetric (ask-lean) bilateral",
        )

    # 6. NORMAL or SOFT_CAP with BID permitted → full bilateral
    if inv in ("NORMAL", "SOFT_CAP") and bid_ok:
        return PostureDecision(
            posture=POSTURE_BILATERAL,
            place_bid=True, place_ask=True,
            reason=f"inventory_state={inv} — bilateral",
        )

    # 7. BID suppressed but ASK allowed
    if not bid_ok and ask_ok:
        return PostureDecision(
            posture=POSTURE_EXIT_ONLY,
            place_bid=False, place_ask=True,
            reason="bid_ok=False (same-side suppression or inventory limit) — ASK only",
        )

    # 8. Fallback
    return PostureDecision(
        posture=POSTURE_SKIP,
        place_bid=False, place_ask=False,
        reason=f"no eligible posture (inv={inv} bid_ok={bid_ok} ask_ok={ask_ok})",
    )
