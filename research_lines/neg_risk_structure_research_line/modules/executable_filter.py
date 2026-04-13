"""
neg_risk_structure_research_line — Module 4: Executable Filter
polyarb_lab / research_line / active

Applies execution quality gates to events that passed structural_check.py.

Gates applied (in order):
  1. Depth gate: minimum CLOB ask depth per leg >= MIN_DEPTH_USD
  2. Spread gate: maximum bid/ask spread per leg <= MAX_SPREAD
  3. Slippage estimate: estimated slippage at PAPER_SIZE_USD
  4. Liquidity quality score

This module DOES fetch CLOB order books for events that pass structural check.
CLOB fetches are read-only — no order submission ever.

Output: ExecutionQualityResult per event.

Read-only. No order submission. No mainline imports.

CLOB API via py_clob_client is the preferred book source.
Falls back to Gamma bid/ask fields if CLOB is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal, Optional

import httpx

from .normalizer import NegRiskEvent, NegRiskOutcome
from .structural_check import StructuralCheckResult

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"

# Hard gates (paper scale)
MIN_DEPTH_USD = 20.0        # minimum ask depth in USD per leg (paper floor)
MAX_SPREAD = 0.10           # maximum bid/ask spread allowed (10 cents per $1)
PAPER_SIZE_USD = 50.0       # reference paper position size per leg
MIN_LEGS_SUFFICIENT = 1     # at least this many legs must pass depth gate

# Slippage estimation
SLIPPAGE_WARNING_THRESHOLD = 0.015  # > 1.5% slippage relative to mid: warn


ExecutionClass = Literal[
    "EXECUTABLE",              # all gates pass — suitable for paper position
    "RESEARCH_VALUE_ONLY",     # structural gap exists but execution quality insufficient
    "NOISE",                   # no meaningful gap AND execution quality poor
    "CLOB_UNAVAILABLE",        # could not fetch CLOB book — classify as research only
]


# ---------------------------------------------------------------------------
# CLOB book fetch (read-only)
# ---------------------------------------------------------------------------

@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class ClobBook:
    token_id: str
    bids: list[BookLevel]
    asks: list[BookLevel]
    fetched_at: datetime
    source: Literal["clob", "gamma_fallback"]
    error: Optional[str] = None


def _fetch_clob_book(token_id: str, clob_host: str, client: httpx.Client) -> ClobBook:
    """
    Fetch CLOB order book for a single token_id.

    Uses the CLOB REST API: GET /book?token_id={token_id}
    Returns a ClobBook. On failure, returns a ClobBook with error set.
    """
    fetched_at = datetime.now(timezone.utc)
    url = f"{clob_host.rstrip('/')}/book"
    try:
        resp = client.get(url, params={"token_id": token_id}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        bids = _parse_levels(data.get("bids") or [])
        asks = _parse_levels(data.get("asks") or [])
        return ClobBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            fetched_at=fetched_at,
            source="clob",
        )
    except Exception as exc:
        logger.debug("CLOB book fetch failed for token %s: %s", token_id, exc)
        return ClobBook(
            token_id=token_id,
            bids=[],
            asks=[],
            fetched_at=fetched_at,
            source="clob",
            error=str(exc),
        )


def _parse_levels(raw_levels: list[Any]) -> list[BookLevel]:
    levels: list[BookLevel] = []
    for raw in raw_levels:
        try:
            if isinstance(raw, dict):
                price = float(raw.get("price", 0))
                size = float(raw.get("size", 0))
            else:
                continue
            if price > 0 and size > 0 and isfinite(price) and isfinite(size):
                levels.append(BookLevel(price=price, size=size))
        except (ValueError, TypeError):
            continue
    # Sort: bids descending, asks ascending (normalize regardless of source order)
    # Caller is responsible for knowing which list is which
    return levels


def _gamma_fallback_book(outcome: NegRiskOutcome) -> ClobBook:
    """Create a minimal ClobBook from Gamma bid/ask fields."""
    fetched_at = datetime.now(timezone.utc)
    bids: list[BookLevel] = []
    asks: list[BookLevel] = []
    if outcome.yes_bid is not None and outcome.yes_bid > 0:
        bids.append(BookLevel(price=outcome.yes_bid, size=MIN_DEPTH_USD))  # size unknown
    if outcome.yes_ask is not None and outcome.yes_ask > 0:
        asks.append(BookLevel(price=outcome.yes_ask, size=MIN_DEPTH_USD))  # size unknown
    return ClobBook(
        token_id=outcome.token_id,
        bids=bids,
        asks=asks,
        fetched_at=fetched_at,
        source="gamma_fallback",
    )


# ---------------------------------------------------------------------------
# Per-leg metrics
# ---------------------------------------------------------------------------

@dataclass
class LegExecutionMetrics:
    outcome_index: int
    outcome_id: str
    question: str
    token_id: str

    # Book-derived
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]        # best_ask - best_bid
    spread_pct: Optional[float]    # spread / best_ask

    # Depth at PAPER_SIZE_USD
    ask_depth_usd: float           # total USD value fillable at paper size
    bid_depth_usd: float

    # Slippage estimate (buying YES at paper size)
    vwap_ask: Optional[float]      # volume-weighted average ask price at paper size
    slippage_vs_mid: Optional[float]  # vwap_ask - yes_mid

    # Gate results
    passes_depth_gate: bool        # ask_depth_usd >= MIN_DEPTH_USD
    passes_spread_gate: bool       # spread <= MAX_SPREAD (if spread available)

    # Source
    book_source: Literal["clob", "gamma_fallback", "no_data"]


def _compute_vwap_ask(asks: list[BookLevel], target_usd: float) -> Optional[float]:
    """Compute volume-weighted average ask price for buying up to target_usd dollars."""
    if not asks:
        return None
    # Sort ascending by price
    sorted_asks = sorted(asks, key=lambda l: l.price)
    total_cost = 0.0
    total_shares = 0.0
    remaining = target_usd

    for level in sorted_asks:
        # shares available at this level (shares = size, cost = shares * price)
        level_cost = level.size * level.price
        if level_cost <= remaining:
            total_cost += level_cost
            total_shares += level.size
            remaining -= level_cost
        else:
            # Partial fill
            shares = remaining / level.price
            total_cost += remaining
            total_shares += shares
            remaining = 0.0
            break

    if total_shares <= 0:
        return None
    return total_cost / total_shares


def _ask_depth_usd(asks: list[BookLevel], cap_usd: float) -> float:
    """Total USD fillable on ask side up to cap_usd."""
    total = 0.0
    for level in sorted(asks, key=lambda l: l.price):
        total += level.size * level.price
        if total >= cap_usd:
            return min(total, cap_usd)
    return total


def _bid_depth_usd(bids: list[BookLevel], cap_usd: float) -> float:
    """Total USD fillable on bid side up to cap_usd."""
    total = 0.0
    for level in sorted(bids, key=lambda l: l.price, reverse=True):
        total += level.size * level.price
        if total >= cap_usd:
            return min(total, cap_usd)
    return total


def _compute_leg_metrics(
    outcome: NegRiskOutcome,
    book: ClobBook,
) -> LegExecutionMetrics:
    asks_sorted = sorted(book.asks, key=lambda l: l.price)
    bids_sorted = sorted(book.bids, key=lambda l: l.price, reverse=True)

    best_ask = asks_sorted[0].price if asks_sorted else None
    best_bid = bids_sorted[0].price if bids_sorted else None

    spread = (best_ask - best_bid) if (best_ask is not None and best_bid is not None) else None
    spread_pct = (spread / best_ask) if (spread is not None and best_ask and best_ask > 0) else None

    ask_depth = _ask_depth_usd(book.asks, PAPER_SIZE_USD)
    bid_depth = _bid_depth_usd(book.bids, PAPER_SIZE_USD)

    vwap = _compute_vwap_ask(book.asks, PAPER_SIZE_USD)
    slippage = (vwap - outcome.yes_mid) if (vwap is not None) else None

    passes_depth = ask_depth >= MIN_DEPTH_USD
    passes_spread = (spread is None or spread <= MAX_SPREAD)  # None = unknown, allow

    if book.source == "no_data":
        book_src: Literal["clob", "gamma_fallback", "no_data"] = "no_data"
    else:
        book_src = book.source

    return LegExecutionMetrics(
        outcome_index=outcome.outcome_index,
        outcome_id=outcome.outcome_id,
        question=outcome.question,
        token_id=outcome.token_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        spread_pct=spread_pct,
        ask_depth_usd=ask_depth,
        bid_depth_usd=bid_depth,
        vwap_ask=vwap,
        slippage_vs_mid=slippage,
        passes_depth_gate=passes_depth,
        passes_spread_gate=passes_spread,
        book_source=book_src,
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ExecutionQualityResult:
    """Full execution quality result for one event."""
    event_id: str
    slug: str
    title: str
    n_outcomes: int
    abs_gap: float
    constraint_class: str

    leg_metrics: list[LegExecutionMetrics]

    # Aggregate gates
    legs_passing_depth: int
    legs_passing_spread: int
    min_ask_depth_usd: float
    max_spread: Optional[float]
    max_slippage_vs_mid: Optional[float]

    # Liquidity quality score (0.0–1.0)
    # = (legs_passing_depth + legs_passing_spread/2) / n_outcomes
    liquidity_quality_score: float

    execution_class: ExecutionClass
    classification_reason: str

    clob_fetch_attempted: bool
    clob_fetch_failed_count: int

    checked_at: datetime

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "slug": self.slug,
            "title": self.title,
            "n_outcomes": self.n_outcomes,
            "abs_gap": round(self.abs_gap, 6),
            "constraint_class": self.constraint_class,
            "execution_class": self.execution_class,
            "classification_reason": self.classification_reason,
            "legs_passing_depth": self.legs_passing_depth,
            "legs_passing_spread": self.legs_passing_spread,
            "min_ask_depth_usd": round(self.min_ask_depth_usd, 2),
            "max_spread": round(self.max_spread, 6) if self.max_spread is not None else None,
            "max_slippage_vs_mid": (
                round(self.max_slippage_vs_mid, 6)
                if self.max_slippage_vs_mid is not None else None
            ),
            "liquidity_quality_score": round(self.liquidity_quality_score, 4),
            "clob_fetch_attempted": self.clob_fetch_attempted,
            "clob_fetch_failed_count": self.clob_fetch_failed_count,
            "checked_at": self.checked_at.isoformat(),
            "leg_metrics": [
                {
                    "outcome_index": m.outcome_index,
                    "outcome_id": m.outcome_id,
                    "question": m.question,
                    "best_bid": m.best_bid,
                    "best_ask": m.best_ask,
                    "spread": round(m.spread, 6) if m.spread is not None else None,
                    "ask_depth_usd": round(m.ask_depth_usd, 2),
                    "vwap_ask": round(m.vwap_ask, 6) if m.vwap_ask is not None else None,
                    "slippage_vs_mid": (
                        round(m.slippage_vs_mid, 6) if m.slippage_vs_mid is not None else None
                    ),
                    "passes_depth_gate": m.passes_depth_gate,
                    "passes_spread_gate": m.passes_spread_gate,
                    "book_source": m.book_source,
                }
                for m in self.leg_metrics
            ],
        }


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _classify_execution(
    check_result: StructuralCheckResult,
    leg_metrics: list[LegExecutionMetrics],
    clob_fetch_failed_count: int,
    n_outcomes: int,
) -> tuple[ExecutionClass, str]:
    """Determine ExecutionClass and reason string."""

    legs_passing_depth = sum(1 for m in leg_metrics if m.passes_depth_gate)
    legs_passing_spread = sum(1 for m in leg_metrics if m.passes_spread_gate)
    all_clob_failed = clob_fetch_failed_count == n_outcomes

    if all_clob_failed:
        return (
            "CLOB_UNAVAILABLE",
            "All CLOB book fetches failed. Cannot assess execution quality. "
            "Classify as research only."
        )

    # No structural gap — classify as noise
    if not check_result.passes_gap_threshold:
        return (
            "NOISE",
            f"Constraint gap {check_result.abs_gap:.4f} below non-trivial threshold "
            f"({0.010:.3f}). No structural inconsistency to exploit."
        )

    # Gap exists but execution quality is poor
    if legs_passing_depth < MIN_LEGS_SUFFICIENT:
        return (
            "RESEARCH_VALUE_ONLY",
            f"Structural gap confirmed ({check_result.abs_gap:.4f}, "
            f"{check_result.constraint_class}) but no legs meet depth gate "
            f"(min_depth={MIN_DEPTH_USD} USD). Execution not feasible at paper scale."
        )

    # Gap exists, some depth, but does not clear fee hurdle
    if not check_result.passes_fee_hurdle:
        return (
            "RESEARCH_VALUE_ONLY",
            f"Structural gap ({check_result.abs_gap:.4f}) exists but does not clear "
            f"estimated round-trip fee ({0.020:.3f}). Research value: gap direction "
            f"and persistence. No executable edge at current gap."
        )

    # Gap clears fee hurdle and depth is sufficient
    spread_warning = ""
    if legs_passing_spread < n_outcomes:
        spread_warning = (
            f" Warning: {n_outcomes - legs_passing_spread} legs exceed spread threshold "
            f"({MAX_SPREAD:.2f}). Execution quality degraded."
        )

    return (
        "EXECUTABLE",
        f"Structural gap ({check_result.abs_gap:.4f}) clears fee hurdle. "
        f"{legs_passing_depth}/{n_outcomes} legs pass depth gate. "
        f"Paper-scale position may be feasible.{spread_warning}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_execution_quality(
    event: NegRiskEvent,
    check_result: StructuralCheckResult,
    clob_host: str = CLOB_HOST,
    fetch_clob: bool = True,
) -> ExecutionQualityResult:
    """
    Classify execution quality for an event that passed structural check.

    Args:
        event:        Normalized NegRiskEvent.
        check_result: StructuralCheckResult from structural_check.py.
        clob_host:    CLOB API base URL.
        fetch_clob:   If True, fetch CLOB order books. If False, use Gamma fallback.

    Returns:
        ExecutionQualityResult (paper-only — no order submission).
    """
    checked_at = datetime.now(timezone.utc)
    books: dict[str, ClobBook] = {}
    clob_fetch_attempted = fetch_clob
    clob_fetch_failed_count = 0

    if fetch_clob:
        token_ids = [o.token_id for o in event.outcomes if o.token_id]
        if not token_ids:
            logger.warning("Event %s: no token_ids — cannot fetch CLOB books", event.event_id)
            clob_fetch_attempted = False

        if clob_fetch_attempted and token_ids:
            with httpx.Client() as client:
                for outcome in event.outcomes:
                    if not outcome.token_id:
                        clob_fetch_failed_count += 1
                        continue
                    book = _fetch_clob_book(outcome.token_id, clob_host, client)
                    if book.error:
                        clob_fetch_failed_count += 1
                        # Fall back to Gamma data for this leg
                        books[outcome.outcome_id] = _gamma_fallback_book(outcome)
                    else:
                        books[outcome.outcome_id] = book

    # Ensure all outcomes have a book entry
    for outcome in event.outcomes:
        if outcome.outcome_id not in books:
            if outcome.yes_bid is not None or outcome.yes_ask is not None:
                books[outcome.outcome_id] = _gamma_fallback_book(outcome)
            else:
                books[outcome.outcome_id] = ClobBook(
                    token_id=outcome.token_id,
                    bids=[],
                    asks=[],
                    fetched_at=checked_at,
                    source="gamma_fallback",
                    error="no_data",
                )

    # Compute per-leg metrics
    leg_metrics: list[LegExecutionMetrics] = []
    for outcome in event.outcomes:
        book = books.get(outcome.outcome_id)
        if book is None:
            book = ClobBook(
                token_id=outcome.token_id, bids=[], asks=[],
                fetched_at=checked_at, source="gamma_fallback", error="no_data"
            )
        metrics = _compute_leg_metrics(outcome, book)
        if book.error == "no_data":
            metrics.book_source = "no_data"
        leg_metrics.append(metrics)

    legs_passing_depth = sum(1 for m in leg_metrics if m.passes_depth_gate)
    legs_passing_spread = sum(1 for m in leg_metrics if m.passes_spread_gate)
    min_ask_depth = min((m.ask_depth_usd for m in leg_metrics), default=0.0)

    spreads = [m.spread for m in leg_metrics if m.spread is not None]
    max_spread = max(spreads) if spreads else None

    slippages = [m.slippage_vs_mid for m in leg_metrics if m.slippage_vs_mid is not None]
    max_slippage = max(slippages) if slippages else None

    # Liquidity quality score
    liq_score = (
        (legs_passing_depth + legs_passing_spread * 0.5)
        / (event.n_outcomes * 1.5)
        if event.n_outcomes > 0 else 0.0
    )
    liq_score = min(liq_score, 1.0)

    execution_class, reason = _classify_execution(
        check_result, leg_metrics, clob_fetch_failed_count, event.n_outcomes
    )

    return ExecutionQualityResult(
        event_id=event.event_id,
        slug=event.slug,
        title=event.title,
        n_outcomes=event.n_outcomes,
        abs_gap=event.abs_gap,
        constraint_class=check_result.constraint_class,
        leg_metrics=leg_metrics,
        legs_passing_depth=legs_passing_depth,
        legs_passing_spread=legs_passing_spread,
        min_ask_depth_usd=min_ask_depth,
        max_spread=max_spread,
        max_slippage_vs_mid=max_slippage,
        liquidity_quality_score=round(liq_score, 4),
        execution_class=execution_class,
        classification_reason=reason,
        clob_fetch_attempted=clob_fetch_attempted,
        clob_fetch_failed_count=clob_fetch_failed_count,
        checked_at=checked_at,
    )


def filter_batch(
    events: list[NegRiskEvent],
    check_results: list[StructuralCheckResult],
    clob_host: str = CLOB_HOST,
    fetch_clob: bool = True,
    only_gap_threshold: bool = True,
) -> list[ExecutionQualityResult]:
    """
    Apply execution filter to a batch of events.

    Args:
        only_gap_threshold: If True, skip CLOB fetch for events that don't
                            pass the gap threshold (saves API calls).
    """
    check_by_id = {r.event_id: r for r in check_results}
    results: list[ExecutionQualityResult] = []

    for event in events:
        check_result = check_by_id.get(event.event_id)
        if check_result is None:
            logger.warning("No check result for event %s — skipping filter", event.event_id)
            continue

        # Skip CLOB fetch for events below gap threshold
        do_clob = fetch_clob and (
            not only_gap_threshold or check_result.passes_gap_threshold
        )

        result = classify_execution_quality(
            event, check_result, clob_host=clob_host, fetch_clob=do_clob
        )
        results.append(result)

    executable = sum(1 for r in results if r.execution_class == "EXECUTABLE")
    research_only = sum(1 for r in results if r.execution_class == "RESEARCH_VALUE_ONLY")
    noise = sum(1 for r in results if r.execution_class == "NOISE")
    unavailable = sum(1 for r in results if r.execution_class == "CLOB_UNAVAILABLE")

    logger.info(
        "Execution filter complete: %d results | "
        "EXECUTABLE=%d, RESEARCH_VALUE_ONLY=%d, NOISE=%d, CLOB_UNAVAILABLE=%d",
        len(results), executable, research_only, noise, unavailable,
    )
    return results
