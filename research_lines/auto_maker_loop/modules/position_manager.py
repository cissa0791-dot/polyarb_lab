"""
auto_maker_loop — position_manager
polyarb_lab / research_lines / auto_maker_loop / modules

State machine for one bilateral position lifecycle.

States
------
QUOTING      Both orders live, no fill yet.  Polls for fill or scoring.
BID_FILLED   BID matched — 200 YES tokens acquired.  ASK still live.
             System now manages the exit automatically.
EXITING      Exit order in flight (cancel + re-place at lower price, or taker fill).
DONE         Position closed.  Outcome available in PositionState.

Exit triggers (checked every poll_interval_sec while in BID_FILLED state)
--------------------------------------------------------------------------
NATURAL      ASK order filled at full ask_price — best outcome.
TIME_LIMIT   Hold duration >= max_hold_minutes.  Cancel ASK; place taker SELL
             at bid-side (cross spread to exit immediately).
STOP_LOSS    Live midpoint fell below entry_price - stop_loss_cents.
             Cancel ASK; cross spread to exit.
CHASE        Hold duration >= chase_after_minutes AND midpoint has moved.
             Cancel ASK; re-place ASK one tick lower.  Repeat up to max_chases.
CANCELLED    BID was cancelled (dry-run, or manual intervention).

Reuses from scoring_activation.py (imported dynamically to avoid circular deps):
    _is_filled(client, order_id) -> bool
    _cancel_order(client, order_id) -> bool
    _place_order(client, token_id, price, size, side) -> (order_id, error)
    fetch_midpoint(client, token_id, price_ref) -> (midpoint, source)

Public interface
----------------
    run_position_loop(client, state, config) -> PositionState
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import helpers from scoring_activation (same repo, no circular dependency)
_sa = None  # lazy import


def _get_sa():
    global _sa
    if _sa is None:
        import importlib
        _sa = importlib.import_module(
            "research_lines.reward_aware_maker_probe.modules.scoring_activation"
        )
    return _sa


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PositionConfig:
    """
    Two-layer control parameters for one position.

    Layer 1 — Dynamic (evaluated each poll):
        drift_threshold_cents    : refresh quotes if mid moves this far from our bid
        ask_cancel_distance_cents: cancel hanging ASK if mid rises this far above it
        chase_after_minutes      : lower ASK after this many minutes unfilled (backstop)
        stop_loss_cents          : exit if mid drops this far below entry

    Layer 2 — Hard limits (unconditional):
        max_hold_minutes         : forced taker-sell after this many minutes (backstop only)
        max_chases               : cap on ASK re-quotes downward
        consecutive_fail_limit   : kill switch after this many consecutive order failures
        reduce_only              : if True, skip BID — only work SELL side (set by outer loop)
        close_buffer             : if True, no new BIDs allowed (set by outer loop)
    """
    poll_interval_sec: int          = 30      # seconds between fill checks
    max_hold_minutes: float         = 90.0   # hard backstop: forced exit after this many minutes
    chase_after_minutes: float      = 45.0    # lower ASK after this many minutes unfilled
    max_chases: int                 = 2       # max ASK re-quotes downward
    chase_tick: float               = 0.01    # lower ASK by this amount per chase
    stop_loss_cents: float          = 3.0     # exit if mid drops this far below entry
    drift_threshold_cents: float    = 2.0     # re-quote if mid drifts this far from bid in QUOTING
    ask_cancel_distance_cents: float = 2.0    # cancel ASK if mid rises this far above it in BID_FILLED
    consecutive_fail_limit: int     = 2       # kill switch after this many consecutive failures
    reduce_only: bool               = False   # outer loop: True = SELL-only (inventory cap hit)
    close_buffer: bool              = False   # outer loop: True = near resolution, no new BIDs
    dry_run: bool                   = False   # no order submission; simulate fills


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """
    Live state for one bilateral position.  Mutated in-place by run_position_loop().
    """
    slug: str
    token_id: str
    bid_order_id: str
    ask_order_id: str
    entry_price: float        # price BID was placed at
    ask_price: float          # current live ASK price (updated on chase)
    size: float               # shares per side (e.g. 200)
    placed_at: datetime       # when BID+ASK were submitted

    state: str                = "QUOTING"   # QUOTING | BID_FILLED | DONE
    bid_filled: bool          = False
    ask_filled: bool          = False
    ask_chases: int           = 0           # how many times ASK was re-quoted lower
    ask_repriced_up: int      = 0           # how many times ASK was re-placed higher (mid rose)
    exit_reason: str          = ""          # NATURAL | TIME_LIMIT | STOP_LOSS | REQUOTE_NEEDED | KILL_SWITCH
    exit_price: float         = 0.0         # actual fill price of exit leg
    hold_minutes: float       = 0.0         # minutes from BID fill to exit
    pnl_cents: float          = 0.0         # (exit_price - entry_price) * 100
    last_midpoint: float      = 0.0         # most recent live midpoint
    min_midpoint_hold: float  = 0.0         # lowest mid seen during BID_FILLED
    max_midpoint_hold: float  = 0.0         # highest mid seen during BID_FILLED
    kill_switch: bool         = False       # hard stop: error or consecutive failures
    drift_triggered: bool     = False       # QUOTING drift re-quote fired
    consecutive_failures: int = 0           # consecutive order/cancel failures
    notes: list[str]          = field(default_factory=list)

    def elapsed_minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.placed_at).total_seconds() / 60.0

    def to_dict(self) -> dict:
        return {
            "slug":           self.slug,
            "token_id":       self.token_id[:20] + "...",
            "bid_order_id":   self.bid_order_id,
            "ask_order_id":   self.ask_order_id,
            "entry_price":    self.entry_price,
            "ask_price":      self.ask_price,
            "size":           self.size,
            "placed_at":      self.placed_at.isoformat(),
            "state":              self.state,
            "min_midpoint_hold":  self.min_midpoint_hold,
            "max_midpoint_hold":  self.max_midpoint_hold,
            "bid_filled":         self.bid_filled,
            "ask_filled":         self.ask_filled,
            "ask_chases":         self.ask_chases,
            "ask_repriced_up":    self.ask_repriced_up,
            "drift_triggered":    self.drift_triggered,
            "kill_switch":        self.kill_switch,
            "consecutive_failures": self.consecutive_failures,
            "exit_reason":        self.exit_reason,
            "exit_price":         self.exit_price,
            "hold_minutes":       round(self.hold_minutes, 2),
            "pnl_cents":          round(self.pnl_cents, 4),
            "last_midpoint":  self.last_midpoint,
            "notes":          self.notes,
        }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_position_loop(
    client: Any,
    state: PositionState,
    config: PositionConfig,
    price_ref: Optional[float] = None,
) -> PositionState:
    """
    Poll until the position reaches state=DONE.

    Handles all exit triggers automatically.  Mutates `state` in-place
    and returns it when done.

    Parameters
    ----------
    client       : py_clob_client ClobClient (authenticated)
    state        : PositionState (created by run_auto_maker.py after quote placement)
    config       : PositionConfig (exit thresholds)
    price_ref    : float, optional — fallback midpoint if live fetch fails
    """
    sa = _get_sa()
    bid_filled_at: Optional[datetime] = None

    # Reduce-only / pre-owned inventory entry: state is already BID_FILLED on
    # arrival, so the QUOTING→BID_FILLED transition never fires and bid_filled_at
    # would otherwise remain None.  Initialize it to now so hold_minutes
    # accumulates correctly from the moment the loop starts.
    if state.state == "BID_FILLED":
        bid_filled_at = datetime.now(timezone.utc)

    logger.info(
        "position_loop START slug=%s bid=%s ask=%s entry=%.4f ask=%.4f size=%.0f",
        state.slug[:30], state.bid_order_id[:12], state.ask_order_id[:12],
        state.entry_price, state.ask_price, state.size,
    )

    if config.dry_run:
        state.state       = "DONE"
        state.exit_reason = "DRY_RUN"
        state.exit_price  = state.ask_price
        state.pnl_cents   = round((state.ask_price - state.entry_price) * 100, 4)
        state.notes.append("dry_run: no real orders; assumed natural fill")
        return state

    while state.state != "DONE":
        time.sleep(config.poll_interval_sec)

        elapsed = state.elapsed_minutes()

        # ── Fetch live midpoint (for stop-loss and chase logic) ───────────
        mid, mid_src = sa.fetch_midpoint(client, state.token_id, price_ref)
        if mid:
            state.last_midpoint = mid

        # ── State: QUOTING — both orders live, no fill yet ────────────────
        if state.state == "QUOTING":
            bid_filled = sa._is_filled(client, state.bid_order_id)
            ask_filled = sa._is_filled(client, state.ask_order_id)

            if ask_filled and bid_filled:
                # Both filled (rare but possible: crossed quotes)
                state.bid_filled  = True
                state.ask_filled  = True
                state.exit_reason = "NATURAL"
                state.exit_price  = state.ask_price
                state.hold_minutes = 0.0
                state.pnl_cents   = round((state.ask_price - state.entry_price) * 100, 4)
                state.state       = "DONE"
                logger.info("QUOTING → DONE (NATURAL — both filled simultaneously)")
                break

            if ask_filled and not bid_filled:
                # ASK filled first without BID — taker bought our YES inventory
                state.ask_filled  = True
                state.exit_reason = "ASK_ONLY"
                state.exit_price  = state.ask_price
                state.state       = "DONE"
                state.notes.append("ASK filled before BID — inventory sold without acquisition cost")
                logger.info("QUOTING → DONE (ASK_ONLY — ASK filled before BID)")
                break

            if bid_filled:
                state.bid_filled = True
                state.state      = "BID_FILLED"
                bid_filled_at    = datetime.now(timezone.utc)
                logger.info(
                    "QUOTING → BID_FILLED  entry=%.4f  elapsed=%.1f min",
                    state.entry_price, elapsed,
                )
                # Fall through immediately to BID_FILLED logic in next iteration
                continue

            # ── Dynamic: drift check — re-quote if mid moved too far from our bid ──
            if mid and abs(mid - state.entry_price) * 100 > config.drift_threshold_cents:
                logger.info(
                    "DRIFT: mid=%.4f entry=%.4f drift=%.2f¢ > threshold=%.2f¢ — cancelling and re-quoting",
                    mid, state.entry_price,
                    abs(mid - state.entry_price) * 100,
                    config.drift_threshold_cents,
                )
                _safe_cancel(client, state.bid_order_id, "BID (drift)", state)
                _safe_cancel(client, state.ask_order_id, "ASK (drift)", state)
                state.drift_triggered = True
                state.exit_reason     = "REQUOTE_NEEDED"
                state.state           = "DONE"
                break

            # ── Hard limit: time limit for quoting phase ───────────────────
            if elapsed >= config.max_hold_minutes:
                _safe_cancel(client, state.bid_order_id, "BID (no fill, time limit)", state)
                _safe_cancel(client, state.ask_order_id, "ASK (no fill, time limit)", state)
                state.exit_reason = "NO_FILL_TIME_LIMIT"
                state.state       = "DONE"
                logger.info("QUOTING → DONE (NO_FILL_TIME_LIMIT at %.1f min)", elapsed)
                break

            logger.info(
                "QUOTING poll: elapsed=%.1f min  bid_filled=%s  ask_filled=%s  mid=%s",
                elapsed, bid_filled, ask_filled, f"{mid:.4f}" if mid else "N/A",
            )
            continue

        # ── State: BID_FILLED — holding YES tokens, managing exit ─────────
        if state.state == "BID_FILLED":
            hold_sec = (
                (datetime.now(timezone.utc) - bid_filled_at).total_seconds()
                if bid_filled_at else 0.0
            )
            hold_min = hold_sec / 60.0
            state.hold_minutes = round(hold_min, 2)

            # Track excursion bounds during hold
            if mid:
                if state.min_midpoint_hold == 0.0 or mid < state.min_midpoint_hold:
                    state.min_midpoint_hold = mid
                if mid > state.max_midpoint_hold:
                    state.max_midpoint_hold = mid

            # 1. Natural exit: ASK filled
            ask_filled = sa._is_filled(client, state.ask_order_id)
            if ask_filled:
                state.ask_filled  = True
                state.exit_reason = "NATURAL"
                state.exit_price  = state.ask_price
                state.pnl_cents   = round((state.ask_price - state.entry_price) * 100, 4)
                state.state       = "DONE"
                logger.info(
                    "BID_FILLED → DONE (NATURAL)  pnl=%.2f¢  hold=%.1f min",
                    state.pnl_cents, hold_min,
                )
                break

            # 2. Dynamic: hanging ASK stale-above-mid — mid rose above our ask
            # Mid rising above ask means: (a) we are selling too cheap, and (b) a taker
            # may sweep our ask soon anyway — but if the distance is large, reprice upward
            # to capture more of the move.
            if mid and (mid - state.ask_price) * 100 > config.ask_cancel_distance_cents:
                new_ask = round(mid + 0.01, 4)  # 1 tick above new midpoint
                new_ask = min(new_ask, 0.99)
                if new_ask > state.entry_price:
                    logger.info(
                        "ASK_STALE_ABOVE_MID: mid=%.4f ask=%.4f dist=%.2f¢ — re-pricing up to %.4f",
                        mid, state.ask_price,
                        (mid - state.ask_price) * 100,
                        new_ask,
                    )
                    _safe_cancel(client, state.ask_order_id, "ASK (stale above mid)", state)
                    new_ask_id, err = sa._place_order(
                        client, state.token_id, new_ask, state.size, "SELL"
                    )
                    if new_ask_id:
                        state.ask_order_id  = new_ask_id
                        state.ask_price     = new_ask
                        state.ask_repriced_up += 1
                        state.notes.append(
                            f"ask_repriced_up: new={new_ask:.4f} id={new_ask_id[:12]}"
                        )
                    else:
                        state.consecutive_failures += 1
                        state.notes.append(f"ask_reprice_up_failed: {err}")
                        if state.consecutive_failures >= config.consecutive_fail_limit:
                            logger.error("KILL_SWITCH: consecutive failures=%d", state.consecutive_failures)
                            state.kill_switch = True
                            state.exit_reason = "KILL_SWITCH"
                            state.state       = "DONE"
                            break

            # 3. Hard limit: stop-loss — midpoint dropped far below entry
            if mid and mid <= state.entry_price - (config.stop_loss_cents / 100.0):
                logger.warning(
                    "STOP_LOSS: mid=%.4f  entry=%.4f  threshold=%.4f  hold=%.1f min",
                    mid, state.entry_price,
                    state.entry_price - config.stop_loss_cents / 100.0,
                    hold_min,
                )
                _safe_cancel(client, state.ask_order_id, "ASK (stop-loss)", state)
                exit_price = _taker_sell(client, state, config)
                state.exit_reason = "STOP_LOSS"
                state.exit_price  = exit_price
                state.pnl_cents   = round((exit_price - state.entry_price) * 100, 4)
                state.state       = "DONE"
                logger.info(
                    "BID_FILLED → DONE (STOP_LOSS)  exit=%.4f  pnl=%.2f¢",
                    exit_price, state.pnl_cents,
                )
                break

            # 3. Chase: lower ASK price after chase_after_minutes
            if (
                hold_min >= config.chase_after_minutes
                and state.ask_chases < config.max_chases
            ):
                new_ask = round(state.ask_price - config.chase_tick, 4)
                new_ask = max(new_ask, state.entry_price + config.chase_tick)  # never below entry + 1 tick
                if new_ask != state.ask_price and new_ask > state.entry_price:
                    logger.info(
                        "CHASE %d/%d: cancel ASK %.4f → re-quote %.4f  hold=%.1f min",
                        state.ask_chases + 1, config.max_chases,
                        state.ask_price, new_ask, hold_min,
                    )
                    _safe_cancel(client, state.ask_order_id, "ASK (chase)", state)
                    new_ask_id, err = sa._place_order(
                        client, state.token_id, new_ask, state.size, "SELL"
                    )
                    if new_ask_id:
                        state.ask_order_id = new_ask_id
                        state.ask_price    = new_ask
                        state.ask_chases  += 1
                        state.notes.append(
                            f"chase {state.ask_chases}: new ASK={new_ask:.4f} id={new_ask_id[:12]}"
                        )
                    else:
                        state.notes.append(f"chase failed: {err}")
                        logger.warning("chase re-quote failed: %s", err)

            # 4. Time limit: forced exit
            if hold_min >= config.max_hold_minutes:
                logger.warning(
                    "TIME_LIMIT: hold=%.1f min >= max=%.0f min — forcing exit",
                    hold_min, config.max_hold_minutes,
                )
                _safe_cancel(client, state.ask_order_id, "ASK (time limit)", state)
                exit_price = _taker_sell(client, state, config)
                state.exit_reason = "TIME_LIMIT"
                state.exit_price  = exit_price
                state.pnl_cents   = round((exit_price - state.entry_price) * 100, 4)
                state.state       = "DONE"
                logger.info(
                    "BID_FILLED → DONE (TIME_LIMIT)  exit=%.4f  pnl=%.2f¢",
                    exit_price, state.pnl_cents,
                )
                break

            logger.info(
                "BID_FILLED poll: hold=%.1f min  mid=%s  ask=%.4f  chases=%d",
                hold_min,
                f"{mid:.4f}" if mid else "N/A",
                state.ask_price,
                state.ask_chases,
            )
            continue

    logger.info(
        "position_loop END slug=%s  state=%s  exit=%s  pnl=%.2f¢  hold=%.1f min",
        state.slug[:30], state.state, state.exit_reason,
        state.pnl_cents, state.hold_minutes,
    )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_cancel(client: Any, order_id: str, label: str, state: PositionState) -> None:
    sa = _get_sa()
    ok = sa._cancel_order(client, order_id)
    state.notes.append(f"cancel {label}: {'ok' if ok else 'failed'}")
    logger.info("cancel %s order_id=%s ok=%s", label, order_id[:16], ok)


def _taker_sell(
    client: Any,
    state: PositionState,
    config: PositionConfig,
) -> float:
    """
    Place a taker SELL to immediately exit the YES inventory.

    Strategy: place a SELL limit at entry_price - 1 tick (crosses the spread
    to ensure immediate execution as a taker).  This guarantees exit at the
    cost of giving up part of the spread.

    Returns the exit price used (or entry_price - 1 tick if placement fails).
    """
    sa = _get_sa()
    tick = 0.01
    taker_price = round(state.entry_price - tick, 4)
    taker_price = max(tick, taker_price)  # never below 0.01

    logger.info(
        "taker_sell: SELL %.0f shares @ %.4f  (entry=%.4f - 1 tick)",
        state.size, taker_price, state.entry_price,
    )

    order_id, err = sa._place_order(
        client, state.token_id, taker_price, state.size, "SELL"
    )
    if order_id:
        state.notes.append(f"taker_sell placed: {order_id[:16]} @ {taker_price:.4f}")
        return taker_price
    else:
        state.notes.append(f"taker_sell failed: {err}")
        logger.error("taker_sell failed: %s", err)
        # Return entry_price as estimate — position may not be closed
        return state.entry_price
