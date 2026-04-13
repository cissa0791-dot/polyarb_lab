"""
neg_risk_structure_research_line — Module 5: Live Gap Scanner
polyarb_lab / research_line / active

Replaces the stale Gamma outcomePrices-based gap check with live CLOB bids
fetched in parallel across all legs of all baskets simultaneously.

Live tradeable edge definition:
  live_gap = 1 - sum(best_bid_i for all YES legs, parallel CLOB fetch)

If live_gap > FEE_HURDLE: passive BUY on all YES legs may survive round-trip fees.

Pipeline position: after normalizer.py, before executable_filter.py.

Why this module exists:
  normalizer.py uses Gamma outcomePrices (cached, stale) for implied_sum.
  structural_check.py gates on that stale gap.
  executable_filter.py fetches CLOB books sequentially — by the time the Nth
  leg is fetched, the first prices are already stale.

  This module fetches all legs of all events in a single parallel batch,
  minimising the timestamp spread across legs to network latency only,
  not sequential call overhead.

Hard constraints:
  - No order submission. Paper-only.
  - No mainline imports.
  - Read-only CLOB access only.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal, Optional

import httpx

from .normalizer import NegRiskEvent, NegRiskOutcome

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
CLOB_BOOK_PATH = "/book"

# Fee hurdle: minimum live_gap to be worth pursuing.
# Conservative: ~2% round-trip (1% taker each way on Polymarket fee markets).
FEE_HURDLE = 0.020

# Parallel fetch configuration
MAX_WORKERS = 6           # thread pool size — kept low to avoid rate limiting
FETCH_TIMEOUT_SEC = 8     # per-request timeout (connect + read)

# Hard cap: maximum number of leg fetches per scan cycle.
# Prevents submitting thousands of tasks when the event universe is large.
# Events are sorted by descending gamma_gap before capping — best candidates first.
MAX_FETCH_LEGS = 2000

# Batch wall-clock timeout: if the parallel fetch takes longer than this,
# cancel remaining futures and return what has completed so far.
# Prevents the indefinite hang when threads stall on slow/unresponsive connections.
BATCH_TIMEOUT_SEC = 180

# Pre-filter: scan events where Gamma gap is within this threshold of zero.
# Relaxed from -0.01 to -0.40 to cover events where Gamma prices are stale
# but live CLOB asks may still sum below $1.00.
# Guards 2 and 3 in _fetch_leg_bid filter inactive/unquoted legs before fetch.
PRE_FILTER_GAMMA_GAP_MIN = -0.40   # was -0.01

# Sweep depth profile sizes (shares) — computed for confirmed ASK_ARB events
# to determine how far into the book the gap survives.
SWEEP_SIZES = [10, 25, 50, 100, 250, 500]

# Thread-local store: one persistent httpx.Client per worker thread.
# Avoids creating + SSL-handshaking a new client for every single leg fetch.
_thread_local = threading.local()


def _get_thread_client() -> httpx.Client:
    """Return a per-thread httpx.Client, creating one on first use."""
    if not hasattr(_thread_local, "http_client"):
        _thread_local.http_client = httpx.Client(timeout=FETCH_TIMEOUT_SEC)
    return _thread_local.http_client

# Multi-cycle defaults
DEFAULT_CYCLES = 1
MAX_CYCLES = 20
DEFAULT_CYCLE_DELAY_SEC = 2.0

GapClass = Literal[
    "LIVE_EDGE",      # live_gap > FEE_HURDLE — may survive round-trip fees
    "SUBTHRESHOLD",   # live_gap > 0 but <= FEE_HURDLE — positive but fee-negative
    "NO_EDGE",        # live_gap <= 0 — no positive gap at current bids
    "PARTIAL_BOOK",   # one or more legs missing live bid — gap unreliable
    "NOT_SCANNED",    # all legs skipped via cap or pre-filter — no data
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LegBidResult:
    """Live CLOB bid result for one YES-side leg."""
    outcome_id: str
    token_id: str
    question: str
    best_bid: Optional[float]        # best YES bid from CLOB (None if unavailable)
    bid_depth_usd: Optional[float]   # USD value at best bid level (size × price)
    fetch_ok: bool                   # True if CLOB responded (even if book was empty)
    fetch_error: Optional[str] = None
    best_ask: Optional[float] = None      # best YES ask from CLOB (None if unavailable)
    ask_depth_usd: Optional[float] = None # USD value at best ask level (size × price)
    ask_levels: list = field(default_factory=list)  # [(price, size), ...] all levels, asc


@dataclass
class LiveGapResult:
    """
    Live gap scan result for one neg-risk basket.

    Core output: live_gap = 1 - sum(best_bid_i across all YES legs).
    If live_gap > FEE_HURDLE → LIVE_EDGE.

    gap_drift = live_gap - gamma_gap.
    Positive drift means the live bid-based gap is larger than the stale
    Gamma outcomePrices gap — common when Gamma lags the actual market.
    """
    event_id: str
    slug: str
    title: str
    n_outcomes: int

    # Live bid data per leg (one entry per outcome, order matches event.outcomes)
    leg_bids: list[LegBidResult]

    # Live gap — None if any leg is missing a best_bid
    live_implied_sum: Optional[float]    # sum of all best_bid_i
    live_gap: Optional[float]            # 1 - live_implied_sum

    # Gamma-based gap (stale outcomePrices) kept for drift comparison
    gamma_implied_sum: float             # sum(yes_mid_i) from normalizer
    gamma_gap: float                     # 1 - gamma_implied_sum

    # Drift: how much does the live bid gap differ from the stale Gamma gap?
    gap_drift: Optional[float]           # live_gap - gamma_gap

    gap_class: GapClass
    legs_with_bid: int
    legs_missing_bid: int

    # Timing
    fetch_start: datetime
    fetch_latency_ms: float              # wall time for the parallel batch fetch

    # Multi-cycle persistence (populated when cycles > 1)
    cycle_count: int = 1
    positive_cycles: int = 0            # cycles where live_gap > 0
    fee_threshold_cycles: int = 0       # cycles where live_gap > FEE_HURDLE
    min_live_gap: Optional[float] = None
    max_live_gap: Optional[float] = None
    mean_live_gap: Optional[float] = None

    # Ask-side validation — None if any leg has no ask
    ask_implied_sum: Optional[float] = None    # sum of all best_ask_i
    ask_gap: Optional[float] = None            # 1 - ask_implied_sum (<0 = spread artifact, >0 = real buy arb)
    total_bid_depth_usd: Optional[float] = None
    total_ask_depth_usd: Optional[float] = None

    # Constrained profit estimate — max fill bounded by shallowest leg depth
    constrained_shares: Optional[float] = None
    constrained_cost_usd: Optional[float] = None
    constrained_gross_profit_usd: Optional[float] = None
    constrained_fee_estimate_usd: Optional[float] = None
    constrained_net_profit_usd: Optional[float] = None
    binding_leg_outcome_id: Optional[str] = None
    binding_leg_ask: Optional[float] = None
    binding_leg_depth_usd: Optional[float] = None

    # Full book sweep — how far does the ask_gap survive?
    sweep_profile: Optional[dict] = None      # {str(shares): ask_gap} at SWEEP_SIZES
    max_profitable_shares: Optional[float] = None  # largest fill where ask_gap > 0

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "slug": self.slug,
            "title": self.title,
            "n_outcomes": self.n_outcomes,
            "live_implied_sum": (
                round(self.live_implied_sum, 6)
                if self.live_implied_sum is not None else None
            ),
            "live_gap": (
                round(self.live_gap, 6) if self.live_gap is not None else None
            ),
            "gamma_implied_sum": round(self.gamma_implied_sum, 6),
            "gamma_gap": round(self.gamma_gap, 6),
            "gap_drift": (
                round(self.gap_drift, 6) if self.gap_drift is not None else None
            ),
            "ask_implied_sum": (
                round(self.ask_implied_sum, 6) if self.ask_implied_sum is not None else None
            ),
            "ask_gap": (
                round(self.ask_gap, 6) if self.ask_gap is not None else None
            ),
            "total_bid_depth_usd": (
                round(self.total_bid_depth_usd, 2) if self.total_bid_depth_usd is not None else None
            ),
            "total_ask_depth_usd": (
                round(self.total_ask_depth_usd, 2) if self.total_ask_depth_usd is not None else None
            ),
            "constrained_shares": self.constrained_shares,
            "constrained_cost_usd": self.constrained_cost_usd,
            "constrained_gross_profit_usd": self.constrained_gross_profit_usd,
            "constrained_fee_estimate_usd": self.constrained_fee_estimate_usd,
            "constrained_net_profit_usd": self.constrained_net_profit_usd,
            "binding_leg_outcome_id": self.binding_leg_outcome_id,
            "binding_leg_ask": self.binding_leg_ask,
            "binding_leg_depth_usd": self.binding_leg_depth_usd,
            "sweep_profile": self.sweep_profile,
            "max_profitable_shares": self.max_profitable_shares,
            "gap_class": self.gap_class,
            "legs_with_bid": self.legs_with_bid,
            "legs_missing_bid": self.legs_missing_bid,
            "fetch_latency_ms": round(self.fetch_latency_ms, 1),
            "cycle_count": self.cycle_count,
            "positive_cycles": self.positive_cycles,
            "fee_threshold_cycles": self.fee_threshold_cycles,
            "min_live_gap": (
                round(self.min_live_gap, 6) if self.min_live_gap is not None else None
            ),
            "max_live_gap": (
                round(self.max_live_gap, 6) if self.max_live_gap is not None else None
            ),
            "mean_live_gap": (
                round(self.mean_live_gap, 6) if self.mean_live_gap is not None else None
            ),
            "fetch_start": self.fetch_start.isoformat(),
            "leg_bids": [
                {
                    "outcome_id": lb.outcome_id,
                    "token_id": lb.token_id,
                    "question": lb.question[:60],
                    "best_bid": lb.best_bid,
                    "bid_depth_usd": (
                        round(lb.bid_depth_usd, 2)
                        if lb.bid_depth_usd is not None else None
                    ),
                    "best_ask": lb.best_ask,
                    "ask_depth_usd": (
                        round(lb.ask_depth_usd, 2)
                        if lb.ask_depth_usd is not None else None
                    ),
                    "fetch_ok": lb.fetch_ok,
                    "fetch_error": lb.fetch_error,
                }
                for lb in self.leg_bids
            ],
        }


# ---------------------------------------------------------------------------
# Per-leg CLOB fetch
# Uses a thread-local httpx.Client — one persistent connection per worker thread,
# reused across all tasks assigned to that thread.
# ---------------------------------------------------------------------------

def _fetch_leg_bid(
    outcome: NegRiskOutcome,
    clob_host: str,
) -> LegBidResult:
    """
    Fetch live best_bid for one YES-side token from CLOB /book.

    Uses _get_thread_client() — one persistent client per worker thread,
    avoiding SSL handshake overhead on every request.
    Never raises — all errors are captured in LegBidResult.fetch_error.
    """
    # A valid CLOB token_id is a uint256 decimal integer — all digits, ~77 chars.
    # normalizer._extract_token_id falls back to conditionId (0x-hex, 66 chars),
    # marketId, or Gamma internal id (short integers) when clobTokenIds is missing.
    # All of these return 404 from CLOB /book. Reject anything that is not a
    # long all-digit string.
    tid = outcome.token_id or ""
    if not tid or not tid.isdigit() or len(tid) < 70:
        return LegBidResult(
            outcome_id=outcome.outcome_id,
            token_id=tid,
            question=outcome.question,
            best_bid=None,
            bid_depth_usd=None,
            fetch_ok=False,
            fetch_error="invalid_token_id_format",
        )

    # Guard 2: Per-leg active status.
    # Gamma marks individual legs active=False / closed=True when they are
    # near-resolved or settled. These always 404 on CLOB /book.
    if not outcome.market_active:
        return LegBidResult(
            outcome_id=outcome.outcome_id,
            token_id=tid,
            question=outcome.question,
            best_bid=None,
            bid_depth_usd=None,
            fetch_ok=False,
            fetch_error="market_inactive",
        )

    # Guard 3: Gamma quote presence.
    # If Gamma has no bid AND no ask for this outcome, the CLOB book almost
    # certainly doesn't exist (near-resolved, inactive, or not yet open).
    # Skipping avoids ~70% of 404 requests with no information loss —
    # a market with no Gamma quote cannot contribute a usable live_gap anyway.
    if outcome.yes_bid is None and outcome.yes_ask is None:
        return LegBidResult(
            outcome_id=outcome.outcome_id,
            token_id=tid,
            question=outcome.question,
            best_bid=None,
            bid_depth_usd=None,
            fetch_ok=False,
            fetch_error="no_gamma_quote",
        )

    url = f"{clob_host.rstrip('/')}{CLOB_BOOK_PATH}"
    try:
        client = _get_thread_client()
        resp = client.get(url, params={"token_id": outcome.token_id})
        resp.raise_for_status()
        data = resp.json()

        bids = data.get("bids") or []
        asks = data.get("asks") or []
        if not bids:
            return LegBidResult(
                outcome_id=outcome.outcome_id,
                token_id=outcome.token_id,
                question=outcome.question,
                best_bid=None,
                bid_depth_usd=None,
                fetch_ok=True,
                fetch_error="empty_bids",
            )

        sorted_bids = sorted(
            bids, key=lambda b: float(b.get("price", 0)), reverse=True
        )
        best = sorted_bids[0]
        try:
            best_bid = float(best.get("price", 0))
            best_size = float(best.get("size", 0))
        except (TypeError, ValueError):
            best_bid = 0.0
            best_size = 0.0

        if not isfinite(best_bid) or best_bid <= 0.0:
            return LegBidResult(
                outcome_id=outcome.outcome_id,
                token_id=outcome.token_id,
                question=outcome.question,
                best_bid=None,
                bid_depth_usd=None,
                fetch_ok=True,
                fetch_error="invalid_bid_price",
            )

        bid_depth = (
            round(best_bid * best_size, 4)
            if isfinite(best_size) and best_size > 0
            else None
        )

        # Best ask: sort ascending (lowest ask = best price for a buyer)
        best_ask_val: Optional[float] = None
        ask_depth: Optional[float] = None
        if asks:
            sorted_asks = sorted(
                asks, key=lambda a: float(a.get("price", 999)), reverse=False
            )
            best_a = sorted_asks[0]
            try:
                raw_ask = float(best_a.get("price", 0))
                ask_size = float(best_a.get("size", 0))
            except (TypeError, ValueError):
                raw_ask = 0.0
                ask_size = 0.0
            if isfinite(raw_ask) and raw_ask > 0.0:
                best_ask_val = round(raw_ask, 6)
                if isfinite(ask_size) and ask_size > 0:
                    ask_depth = round(raw_ask * ask_size, 4)

        # Store all ask levels for sweep depth analysis
        ask_levels_list: list = []
        if asks:
            _sorted = sorted(asks, key=lambda a: float(a.get("price", 999)))
            for _a in _sorted:
                try:
                    _p = float(_a.get("price", 0))
                    _s = float(_a.get("size", 0))
                    if isfinite(_p) and isfinite(_s) and _p > 0 and _s > 0:
                        ask_levels_list.append((round(_p, 6), round(_s, 4)))
                except (TypeError, ValueError):
                    pass

        return LegBidResult(
            outcome_id=outcome.outcome_id,
            token_id=outcome.token_id,
            question=outcome.question,
            best_bid=round(best_bid, 6),
            bid_depth_usd=bid_depth,
            fetch_ok=True,
            best_ask=best_ask_val,
            ask_depth_usd=ask_depth,
            ask_levels=ask_levels_list,
        )

    except Exception as exc:
        return LegBidResult(
            outcome_id=outcome.outcome_id,
            token_id=outcome.token_id,
            question=outcome.question,
            best_bid=None,
            bid_depth_usd=None,
            fetch_ok=False,
            fetch_error=str(exc)[:160],
        )


# ---------------------------------------------------------------------------
# Parallel batch fetch
# ---------------------------------------------------------------------------

def _fetch_all_parallel(
    events: list[NegRiskEvent],
    clob_host: str,
    max_workers: int,
) -> dict[str, dict[str, LegBidResult]]:
    """
    Fetch live best_bids for all legs of all events in a single parallel batch.

    All tasks are submitted at once so the spread in fetch timestamps across
    legs is minimised to network latency, not sequential call overhead.

    Returns: {event_id: {outcome_id: LegBidResult}}
    """
    # Pre-filter: only scan events near parity — events with deeply negative
    # gamma_gap cannot show positive live_gap regardless of bid movement.
    # Sort best gamma_gap first so the cap keeps the most promising events.
    scannable = sorted(
        [e for e in events if (1.0 - e.implied_sum) >= PRE_FILTER_GAMMA_GAP_MIN],
        key=lambda e: (1.0 - e.implied_sum),
        reverse=True,
    )
    skipped_filter = len(events) - len(scannable)

    # Hard cap on total leg fetches — collect in best-first order.
    tasks: list[tuple[str, NegRiskOutcome]] = []
    for event in scannable:
        for outcome in event.outcomes:
            if len(tasks) >= MAX_FETCH_LEGS:
                break
            tasks.append((event.event_id, outcome))
        if len(tasks) >= MAX_FETCH_LEGS:
            break

    scanned_ids = {eid for eid, _ in tasks}
    logger.info(
        "Fetch plan: %d leg tasks | events=%d | pre-filter skipped=%d | "
        "cap skipped=%d | workers=%d | batch_timeout=%ds",
        len(tasks),
        len(scanned_ids),
        skipped_filter,
        len(scannable) - len(scanned_ids),
        max_workers,
        BATCH_TIMEOUT_SEC,
    )

    # Pre-allocate result map for ALL events (including skipped).
    bid_map: dict[str, dict[str, LegBidResult]] = {}
    for event in events:
        bid_map[event.event_id] = {}
        if event.event_id not in scanned_ids:
            reason = (
                "pre_filtered"
                if (1.0 - event.implied_sum) < PRE_FILTER_GAMMA_GAP_MIN
                else "cap_exceeded"
            )
            for outcome in event.outcomes:
                bid_map[event.event_id][outcome.outcome_id] = LegBidResult(
                    outcome_id=outcome.outcome_id,
                    token_id=outcome.token_id,
                    question=outcome.question,
                    best_bid=None,
                    bid_depth_usd=None,
                    fetch_ok=False,
                    fetch_error=reason,
                )

    if not tasks:
        logger.info("No tasks after pre-filter and cap.")
        return bid_map

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_fetch_leg_bid, outcome, clob_host): (event_id, outcome)
            for event_id, outcome in tasks
        }
        completed = 0
        try:
            for future in as_completed(future_to_task, timeout=BATCH_TIMEOUT_SEC):
                event_id, outcome = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = LegBidResult(
                        outcome_id=outcome.outcome_id,
                        token_id=outcome.token_id,
                        question=outcome.question,
                        best_bid=None,
                        bid_depth_usd=None,
                        fetch_ok=False,
                        fetch_error=f"executor_error:{exc}",
                    )
                bid_map[event_id][outcome.outcome_id] = result
                completed += 1
        except FutureTimeoutError:
            logger.warning(
                "Batch timeout after %ds — %d/%d tasks completed. "
                "Returning partial results.",
                BATCH_TIMEOUT_SEC, completed, len(future_to_task),
            )
            for f in future_to_task:
                f.cancel()

    return bid_map


# ---------------------------------------------------------------------------
# Sweep depth analysis
# ---------------------------------------------------------------------------

def _sweep_cost_at_n(lb: "LegBidResult", n_shares: float) -> Optional[float]:
    """
    Cost to buy n_shares of this leg by sweeping through all ask levels.
    Returns total cost in USD, or None if the book cannot fill n_shares.
    """
    if not lb.ask_levels:
        return None
    remaining = n_shares
    cost = 0.0
    for price, size in lb.ask_levels:  # already sorted ascending
        if remaining <= 0.0:
            break
        fill = min(remaining, size)
        cost += fill * price
        remaining -= fill
    if remaining > 1e-9:
        return None  # book too thin to fill n_shares
    return cost


def _sweep_ask_gap_at_n(leg_bids: list, n_shares: float) -> Optional[float]:
    """
    ask_gap if buying n_shares of the full basket, sweeping through book levels.
    Returns 1 - (total_cost / n_shares) which equals 1 - weighted_avg_ask_sum.
    None if any leg cannot be filled.
    """
    total_cost = 0.0
    for lb in leg_bids:
        leg_cost = _sweep_cost_at_n(lb, n_shares)
        if leg_cost is None:
            return None
        total_cost += leg_cost
    weighted_sum = total_cost / n_shares
    return round(1.0 - weighted_sum, 6)


def _compute_sweep_profile(leg_bids: list) -> dict[str, Optional[float]]:
    """
    Compute ask_gap at SWEEP_SIZES share levels for full-book depth picture.
    Returns {str(shares): ask_gap_or_None}.
    A positive value at size N means gap survives at that fill size.
    """
    return {str(n): _sweep_ask_gap_at_n(leg_bids, float(n)) for n in SWEEP_SIZES}


def _max_profitable_shares(leg_bids: list) -> Optional[float]:
    """
    Binary-search for the largest integer share count where ask_gap > 0.
    Returns None if even top-of-book (1 share) is not profitable.
    """
    # Quick gate: top-of-book gap must be positive
    top_gap = _sweep_ask_gap_at_n(leg_bids, 1.0)
    if top_gap is None or top_gap <= 0.0:
        return None
    lo, hi = 1, 10_000  # search up to 10k shares
    best = 1.0
    while lo <= hi:
        mid = (lo + hi) // 2
        gap = _sweep_ask_gap_at_n(leg_bids, float(mid))
        if gap is not None and gap > 0.0:
            best = float(mid)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------------------
# Gap computation and classification
# ---------------------------------------------------------------------------

_NOT_SCANNED_ERRORS = frozenset({"cap_exceeded", "pre_filtered"})


def _classify_gap(
    live_gap: Optional[float],
    has_all_bids: bool,
    leg_bids: list["LegBidResult"],
) -> GapClass:
    # NOT_SCANNED: every leg was skipped by cap or pre-filter — no fetch attempted
    if leg_bids and all(lb.fetch_error in _NOT_SCANNED_ERRORS for lb in leg_bids):
        return "NOT_SCANNED"
    if not has_all_bids or live_gap is None:
        return "PARTIAL_BOOK"
    if live_gap > FEE_HURDLE:
        return "LIVE_EDGE"
    if live_gap > 0.0:
        return "SUBTHRESHOLD"
    return "NO_EDGE"


def _build_result(
    event: NegRiskEvent,
    bid_map: dict[str, LegBidResult],
    fetch_start: datetime,
    fetch_latency_ms: float,
) -> LiveGapResult:
    """Assemble LiveGapResult for one event from the parallel fetch output."""
    leg_bids: list[LegBidResult] = []
    for outcome in event.outcomes:
        lb = bid_map.get(outcome.outcome_id)
        if lb is None:
            lb = LegBidResult(
                outcome_id=outcome.outcome_id,
                token_id=outcome.token_id,
                question=outcome.question,
                best_bid=None,
                bid_depth_usd=None,
                fetch_ok=False,
                fetch_error="no_result",
            )
        leg_bids.append(lb)

    legs_with_bid = sum(1 for lb in leg_bids if lb.best_bid is not None)
    legs_missing_bid = len(leg_bids) - legs_with_bid
    has_all_bids = legs_missing_bid == 0

    live_implied_sum: Optional[float] = None
    live_gap: Optional[float] = None
    if has_all_bids and leg_bids:
        live_implied_sum = round(
            sum(lb.best_bid for lb in leg_bids),  # type: ignore[arg-type]
            6,
        )
        live_gap = round(1.0 - live_implied_sum, 6)

    # Ask-side: None if any leg has no ask (mirrors bid-side logic)
    has_all_asks = all(lb.best_ask is not None for lb in leg_bids)
    ask_implied_sum: Optional[float] = None
    ask_gap: Optional[float] = None
    if has_all_asks and leg_bids:
        ask_implied_sum = round(
            sum(lb.best_ask for lb in leg_bids),  # type: ignore[arg-type]
            6,
        )
        ask_gap = round(1.0 - ask_implied_sum, 6)

    # Depth totals across all legs
    bid_depths = [lb.bid_depth_usd for lb in leg_bids if lb.bid_depth_usd is not None]
    ask_depths = [lb.ask_depth_usd for lb in leg_bids if lb.ask_depth_usd is not None]
    total_bid_depth_usd = round(sum(bid_depths), 4) if bid_depths else None
    total_ask_depth_usd = round(sum(ask_depths), 4) if ask_depths else None

    # Constrained profit: max fill bounded by the shallowest leg's ask depth.
    # available shares per leg = ask_depth_usd / best_ask
    # constrained fill = min across all legs (the binding constraint)
    # cost = constrained_shares × ask_implied_sum
    # gross = constrained_shares × 1.0 - cost  (guaranteed $1 payout per share)
    constrained_shares: Optional[float] = None
    constrained_cost_usd: Optional[float] = None
    constrained_gross_profit_usd: Optional[float] = None
    constrained_fee_estimate_usd: Optional[float] = None
    constrained_net_profit_usd: Optional[float] = None
    binding_leg_outcome_id: Optional[str] = None
    binding_leg_ask: Optional[float] = None
    binding_leg_depth_usd: Optional[float] = None

    if ask_implied_sum is not None and has_all_asks:
        legs_with_depth = [
            lb for lb in leg_bids
            if lb.best_ask is not None and lb.ask_depth_usd is not None and lb.best_ask > 0
        ]
        if len(legs_with_depth) == len(leg_bids):
            available = [
                (lb.ask_depth_usd / lb.best_ask, lb)  # type: ignore[operator]
                for lb in legs_with_depth
            ]
            min_shares_val, binding_lb = min(available, key=lambda t: t[0])
            _cost = round(min_shares_val * ask_implied_sum, 6)
            _gross = round(min_shares_val - _cost, 6)
            _fee = round(_cost * FEE_HURDLE, 6)
            _net = round(_gross - _fee, 6)
            constrained_shares = round(min_shares_val, 4)
            constrained_cost_usd = round(_cost, 4)
            constrained_gross_profit_usd = round(_gross, 4)
            constrained_fee_estimate_usd = round(_fee, 4)
            constrained_net_profit_usd = round(_net, 4)
            binding_leg_outcome_id = binding_lb.outcome_id
            binding_leg_ask = binding_lb.best_ask
            binding_leg_depth_usd = binding_lb.ask_depth_usd

    # Full book sweep — only computed when ask_gap > 0 (ASK_ARB candidates)
    sweep_profile: Optional[dict] = None
    max_profitable_shares: Optional[float] = None
    if ask_gap is not None and ask_gap > 0.0 and has_all_asks:
        sweep_profile = _compute_sweep_profile(leg_bids)
        max_profitable_shares = _max_profitable_shares(leg_bids)

    gamma_gap = round(1.0 - event.implied_sum, 6)
    gap_drift: Optional[float] = (
        round(live_gap - gamma_gap, 6) if live_gap is not None else None
    )
    gap_class = _classify_gap(live_gap, has_all_bids, leg_bids)

    return LiveGapResult(
        event_id=event.event_id,
        slug=event.slug,
        title=event.title,
        n_outcomes=event.n_outcomes,
        leg_bids=leg_bids,
        live_implied_sum=live_implied_sum,
        live_gap=live_gap,
        gamma_implied_sum=event.implied_sum,
        gamma_gap=gamma_gap,
        gap_drift=gap_drift,
        gap_class=gap_class,
        legs_with_bid=legs_with_bid,
        legs_missing_bid=legs_missing_bid,
        fetch_start=fetch_start,
        fetch_latency_ms=fetch_latency_ms,
        ask_implied_sum=ask_implied_sum,
        ask_gap=ask_gap,
        total_bid_depth_usd=total_bid_depth_usd,
        total_ask_depth_usd=total_ask_depth_usd,
        constrained_shares=constrained_shares,
        constrained_cost_usd=constrained_cost_usd,
        constrained_gross_profit_usd=constrained_gross_profit_usd,
        constrained_fee_estimate_usd=constrained_fee_estimate_usd,
        constrained_net_profit_usd=constrained_net_profit_usd,
        binding_leg_outcome_id=binding_leg_outcome_id,
        binding_leg_ask=binding_leg_ask,
        binding_leg_depth_usd=binding_leg_depth_usd,
        sweep_profile=sweep_profile,
        max_profitable_shares=max_profitable_shares,
    )


# ---------------------------------------------------------------------------
# Public API — single cycle
# ---------------------------------------------------------------------------

def scan_one_cycle(
    events: list[NegRiskEvent],
    clob_host: str = CLOB_HOST,
    max_workers: int = MAX_WORKERS,
) -> tuple[list[LiveGapResult], float]:
    """
    Fetch live bids for all events in one parallel batch and compute live gaps.

    Returns:
        (results, fetch_latency_ms)
    """
    if not events:
        return [], 0.0

    fetch_start = datetime.now(timezone.utc)
    t0 = time.monotonic()
    bid_map = _fetch_all_parallel(events, clob_host, max_workers)
    fetch_latency_ms = (time.monotonic() - t0) * 1000.0

    results = [
        _build_result(event, bid_map.get(event.event_id, {}), fetch_start, fetch_latency_ms)
        for event in events
    ]

    logger.info(
        "Live gap scan: %d events | LIVE_EDGE=%d SUBTHRESHOLD=%d NO_EDGE=%d PARTIAL=%d"
        " | latency=%.0fms",
        len(results),
        sum(1 for r in results if r.gap_class == "LIVE_EDGE"),
        sum(1 for r in results if r.gap_class == "SUBTHRESHOLD"),
        sum(1 for r in results if r.gap_class == "NO_EDGE"),
        sum(1 for r in results if r.gap_class == "PARTIAL_BOOK"),
        fetch_latency_ms,
    )
    return results, fetch_latency_ms


# ---------------------------------------------------------------------------
# Public API — multi-cycle persistence scan
# ---------------------------------------------------------------------------

def scan_with_persistence(
    events: list[NegRiskEvent],
    clob_host: str = CLOB_HOST,
    cycles: int = DEFAULT_CYCLES,
    cycle_delay_sec: float = DEFAULT_CYCLE_DELAY_SEC,
    max_workers: int = MAX_WORKERS,
) -> list[LiveGapResult]:
    """
    Run N cycles of parallel live gap scans and attach persistence metrics.

    Each cycle fetches all legs in parallel. After each cycle, waits
    cycle_delay_sec before the next. Returns one LiveGapResult per event,
    with multi-cycle fields (positive_cycles, fee_threshold_cycles,
    min/max/mean live_gap) populated from the full cycle history.

    For cycles=1 (default), equivalent to scan_one_cycle.
    """
    cycles = min(max(cycles, 1), MAX_CYCLES)

    if not events:
        return []

    gap_history: dict[str, list[float]] = {e.event_id: [] for e in events}
    last_results: list[LiveGapResult] = []

    for cycle_num in range(cycles):
        if cycle_num > 0:
            time.sleep(cycle_delay_sec)

        cycle_results, _ = scan_one_cycle(events, clob_host, max_workers)
        last_results = cycle_results

        for r in cycle_results:
            if r.live_gap is not None:
                gap_history[r.event_id].append(r.live_gap)

        logger.debug(
            "Cycle %d/%d | LIVE_EDGE=%d",
            cycle_num + 1,
            cycles,
            sum(1 for r in cycle_results if r.gap_class == "LIVE_EDGE"),
        )

    # Merge persistence stats into the last cycle's results
    merged: list[LiveGapResult] = []
    for r in last_results:
        history = gap_history[r.event_id]
        if history:
            r.positive_cycles = sum(1 for g in history if g > 0.0)
            r.fee_threshold_cycles = sum(1 for g in history if g > FEE_HURDLE)
            r.min_live_gap = round(min(history), 6)
            r.max_live_gap = round(max(history), 6)
            r.mean_live_gap = round(sum(history) / len(history), 6)
        r.cycle_count = cycles
        merged.append(r)

    logger.info(
        "Multi-cycle scan complete: %d cycles | %d events"
        " | LIVE_EDGE (final)=%d | persistent (>=1 fee-threshold cycle)=%d",
        cycles,
        len(merged),
        sum(1 for r in merged if r.gap_class == "LIVE_EDGE"),
        sum(1 for r in merged if r.fee_threshold_cycles > 0),
    )
    return merged


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def scan_summary(results: list[LiveGapResult]) -> dict[str, Any]:
    """Build a concise summary dict from a list of LiveGapResult."""
    if not results:
        return {"total_events": 0}

    from collections import Counter
    counts: Counter[str] = Counter(r.gap_class for r in results)

    live_edge = [r for r in results if r.gap_class == "LIVE_EDGE"]
    live_gaps = [r.live_gap for r in results if r.live_gap is not None]
    latencies = [r.fetch_latency_ms for r in results]
    gap_drifts = [r.gap_drift for r in results if r.gap_drift is not None]
    ask_gaps = [r.ask_gap for r in results if r.ask_gap is not None]
    ask_arb_count = sum(1 for r in results if r.ask_gap is not None and r.ask_gap > FEE_HURDLE)

    best = (
        max(results, key=lambda r: (r.live_gap if r.live_gap is not None else -999))
        if results else None
    )

    return {
        "total_events": len(results),
        "gap_class_counts": dict(counts),
        "live_edge_count": counts.get("LIVE_EDGE", 0),
        "subthreshold_count": counts.get("SUBTHRESHOLD", 0),
        "no_edge_count": counts.get("NO_EDGE", 0),
        "partial_book_count": counts.get("PARTIAL_BOOK", 0),
        "not_scanned_count": counts.get("NOT_SCANNED", 0),
        "scanned_count": len(results) - counts.get("NOT_SCANNED", 0),
        "best_live_gap_event": (
            best.slug
            if best and best.live_gap is not None and best.live_gap > 0
            else None
        ),
        "best_live_gap": (
            round(best.live_gap, 6)
            if best and best.live_gap is not None
            else None
        ),
        "live_gap_stats": {
            "min": round(min(live_gaps), 6) if live_gaps else None,
            "max": round(max(live_gaps), 6) if live_gaps else None,
            "mean": round(sum(live_gaps) / len(live_gaps), 6) if live_gaps else None,
        },
        "gap_drift_stats": {
            "min": round(min(gap_drifts), 6) if gap_drifts else None,
            "max": round(max(gap_drifts), 6) if gap_drifts else None,
            "mean": (
                round(sum(gap_drifts) / len(gap_drifts), 6) if gap_drifts else None
            ),
        },
        "ask_gap_stats": {
            "min": round(min(ask_gaps), 6) if ask_gaps else None,
            "max": round(max(ask_gaps), 6) if ask_gaps else None,
            "mean": round(sum(ask_gaps) / len(ask_gaps), 6) if ask_gaps else None,
        },
        "ask_arb_count": ask_arb_count,
        "ask_arb_profit_summary": {
            "events": [
                {
                    "slug": r.slug,
                    "ask_gap": round(r.ask_gap, 6) if r.ask_gap is not None else None,
                    "constrained_shares": r.constrained_shares,
                    "constrained_gross_profit_usd": r.constrained_gross_profit_usd,
                    "constrained_net_profit_usd": r.constrained_net_profit_usd,
                    "binding_leg_outcome_id": r.binding_leg_outcome_id,
                    "binding_leg_depth_usd": r.binding_leg_depth_usd,
                }
                for r in results
                if r.ask_gap is not None and r.ask_gap > FEE_HURDLE
            ],
            "total_net_profit_usd": round(
                sum(
                    r.constrained_net_profit_usd for r in results
                    if r.ask_gap is not None and r.ask_gap > FEE_HURDLE
                    and r.constrained_net_profit_usd is not None
                ),
                4,
            ),
        },
        "avg_fetch_latency_ms": (
            round(sum(latencies) / len(latencies), 1) if latencies else None
        ),
        "live_edge_events": [
            {
                "slug": r.slug,
                "live_gap": round(r.live_gap, 6) if r.live_gap is not None else None,
                "ask_gap": round(r.ask_gap, 6) if r.ask_gap is not None else None,
                "total_bid_depth_usd": round(r.total_bid_depth_usd, 2) if r.total_bid_depth_usd is not None else None,
                "total_ask_depth_usd": round(r.total_ask_depth_usd, 2) if r.total_ask_depth_usd is not None else None,
                "n_outcomes": r.n_outcomes,
                "fee_threshold_cycles": r.fee_threshold_cycles,
                "cycle_count": r.cycle_count,
            }
            for r in sorted(
                live_edge,
                key=lambda r: r.live_gap if r.live_gap is not None else -999,
                reverse=True,
            )
        ],
    }
