"""
neg_risk_structure_research_line — Module 2: Normalizer
polyarb_lab / research_line / active

Converts a raw NegRiskEventRaw (from discovery.py) into a unified
structural representation: NegRiskEvent.

The NegRiskEvent contains:
  - Per-outcome prices (YES mid, YES bid, YES ask, derived NO prices)
  - Implied sum of all YES prices
  - Constraint gap from 1.0
  - Price source (CLOB vs Gamma outcomePrices)

This module does NOT fetch CLOB books — it works from data already present
in the raw Gamma event/market dicts. CLOB enrichment is done in
executable_filter.py for events that pass structural_check.py.

Read-only. No order submission. No mainline imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from .discovery import NegRiskEventRaw, RawMarket

logger = logging.getLogger(__name__)

PriceSource = Literal["clob_mid", "gamma_outcome_prices", "gamma_last_trade", "unknown"]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class NegRiskOutcome:
    """Structural representation of one outcome in a neg-risk event."""
    outcome_index: int          # position within the event (0-based)
    outcome_id: str             # Gamma market id for this outcome
    token_id: str               # CLOB token id (for book fetch)
    question: str               # outcome label / question text
    slug: str                   # market slug

    # Prices — from best available source
    yes_mid: float              # midpoint of YES bid+ask, or outcomePrices value
    yes_bid: Optional[float]    # best YES bid (None if unavailable)
    yes_ask: Optional[float]    # best YES ask (None if unavailable)
    no_mid: float               # 1 - yes_mid (derived)

    price_source: PriceSource = "unknown"
    liquidity_raw: Optional[float] = None  # Gamma liquidityNum if available
    market_active: bool = True             # Gamma per-leg active flag (False = skip CLOB fetch)


@dataclass
class NegRiskEvent:
    """
    Unified structural representation of a neg-risk event.

    implied_sum = sum of all outcome yes_mid prices.
    Under no-arbitrage: implied_sum should equal 1.0.
    constraint_gap = implied_sum - 1.0.
    """
    event_id: str
    slug: str
    title: str
    end_date_str: Optional[str]
    neg_risk_flag: bool
    outcomes: list[NegRiskOutcome]
    fetched_at: datetime
    normalized_at: datetime

    # Derived constraint metrics
    implied_sum: float          # sum(yes_mid_i)
    constraint_gap: float       # implied_sum - 1.0
    abs_gap: float              # |constraint_gap|
    n_outcomes: int

    # Quality flags
    has_all_prices: bool        # True if all outcomes have a valid yes_mid
    min_liquidity: Optional[float]  # minimum liquidityNum across all outcomes
    dominant_price_source: PriceSource  # most common price source across outcomes

    def leading_outcome(self) -> NegRiskOutcome:
        """Outcome with highest yes_mid price (most likely to resolve YES)."""
        return max(self.outcomes, key=lambda o: o.yes_mid)

    def trailing_outcomes(self) -> list[NegRiskOutcome]:
        """All outcomes except the leading one."""
        leader = self.leading_outcome()
        return [o for o in self.outcomes if o.outcome_id != leader.outcome_id]

    def to_log_dict(self) -> dict[str, Any]:
        """Serializable dict for JSON logging."""
        return {
            "event_id": self.event_id,
            "slug": self.slug,
            "title": self.title,
            "end_date_str": self.end_date_str,
            "neg_risk_flag": self.neg_risk_flag,
            "n_outcomes": self.n_outcomes,
            "implied_sum": round(self.implied_sum, 6),
            "constraint_gap": round(self.constraint_gap, 6),
            "abs_gap": round(self.abs_gap, 6),
            "has_all_prices": self.has_all_prices,
            "dominant_price_source": self.dominant_price_source,
            "min_liquidity": self.min_liquidity,
            "fetched_at": self.fetched_at.isoformat(),
            "normalized_at": self.normalized_at.isoformat(),
            "outcomes": [
                {
                    "index": o.outcome_index,
                    "outcome_id": o.outcome_id,
                    "token_id": o.token_id,
                    "question": o.question,
                    "slug": o.slug,
                    "yes_mid": round(o.yes_mid, 6),
                    "yes_bid": round(o.yes_bid, 6) if o.yes_bid is not None else None,
                    "yes_ask": round(o.yes_ask, 6) if o.yes_ask is not None else None,
                    "no_mid": round(o.no_mid, 6),
                    "price_source": o.price_source,
                    "liquidity_raw": o.liquidity_raw,
                }
                for o in self.outcomes
            ],
        }


# ---------------------------------------------------------------------------
# Price extraction helpers
# ---------------------------------------------------------------------------

def _extract_outcome_prices(market: RawMarket) -> tuple[Optional[float], PriceSource]:
    """
    Extract best available YES mid price from a raw market dict.

    Priority:
    1. outcomePrices[0] from the market dict (Gamma provides this as a JSON string array)
    2. bestBid / bestAsk from market-level fields
    3. lastTradePrice
    4. None (unknown)

    Returns (yes_mid, price_source).
    """
    # Priority 1: outcomePrices — Gamma returns this as ["0.73", "0.27"] for binary
    # For neg-risk multi-outcome, the first element is the YES price for this outcome
    outcome_prices = market.get("outcomePrices")
    if outcome_prices and isinstance(outcome_prices, (list, str)):
        try:
            if isinstance(outcome_prices, str):
                import json
                outcome_prices = json.loads(outcome_prices)
            if isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
                val = float(outcome_prices[0])
                if 0.0 < val < 1.0:
                    return val, "gamma_outcome_prices"
        except (ValueError, TypeError, Exception):
            pass

    # Priority 2: best bid/ask
    bid = market.get("bestBid") or market.get("best_bid")
    ask = market.get("bestAsk") or market.get("best_ask")
    try:
        bid_f = float(bid) if bid is not None else None
        ask_f = float(ask) if ask is not None else None
        if bid_f is not None and ask_f is not None and bid_f > 0 and ask_f > 0:
            mid = (bid_f + ask_f) / 2.0
            return mid, "gamma_outcome_prices"
    except (ValueError, TypeError):
        pass

    # Priority 3: last trade price
    last_price = market.get("lastTradePrice") or market.get("price")
    try:
        val = float(last_price) if last_price is not None else None
        if val is not None and 0.0 < val < 1.0:
            return val, "gamma_last_trade"
    except (ValueError, TypeError):
        pass

    return None, "unknown"


def _extract_bid_ask(market: RawMarket) -> tuple[Optional[float], Optional[float]]:
    """Extract best bid and ask from raw market dict (may be None)."""
    bid = market.get("bestBid") or market.get("best_bid")
    ask = market.get("bestAsk") or market.get("best_ask")
    try:
        bid_f = float(bid) if bid is not None else None
    except (ValueError, TypeError):
        bid_f = None
    try:
        ask_f = float(ask) if ask is not None else None
    except (ValueError, TypeError):
        ask_f = None
    return bid_f, ask_f


def _extract_token_id(market: RawMarket) -> str:
    """Extract CLOB token id from market dict."""
    # Gamma may return clobTokenIds as a JSON string array or actual list
    clob_token_ids = market.get("clobTokenIds")
    if clob_token_ids:
        try:
            if isinstance(clob_token_ids, str):
                import json
                clob_token_ids = json.loads(clob_token_ids)
            if isinstance(clob_token_ids, list) and len(clob_token_ids) >= 1:
                return str(clob_token_ids[0])
        except Exception:
            pass
    # Fallback fields
    for field_name in ("conditionId", "condition_id", "marketId", "id"):
        val = market.get(field_name)
        if val:
            return str(val)
    return ""


def _extract_liquidity(market: RawMarket) -> Optional[float]:
    for fname in ("liquidityNum", "liquidity", "liquidityClob"):
        val = market.get(fname)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def _dominant_source(outcomes: list[NegRiskOutcome]) -> PriceSource:
    from collections import Counter
    counts: Counter[str] = Counter(o.price_source for o in outcomes)
    return counts.most_common(1)[0][0] if counts else "unknown"  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(raw: NegRiskEventRaw) -> Optional[NegRiskEvent]:
    """
    Normalize a NegRiskEventRaw into a NegRiskEvent.

    Returns None if:
    - No markets are present
    - Fewer than 2 outcomes have parseable prices (implied_sum cannot be computed)

    All outcomes are included in the result, but has_all_prices will be False
    if any outcome lacks a parseable price (using 0.0 for the missing price
    to still compute a partial implied_sum, but flagging the event accordingly).
    """
    if not raw.markets:
        logger.debug("Event %s has no markets — skipping normalization", raw.event_id)
        return None

    normalized_at = datetime.now(timezone.utc)
    outcomes: list[NegRiskOutcome] = []
    missing_price_count = 0

    for idx, mkt in enumerate(raw.markets):
        yes_mid, price_source = _extract_outcome_prices(mkt)
        yes_bid, yes_ask = _extract_bid_ask(mkt)
        liquidity = _extract_liquidity(mkt)
        token_id = _extract_token_id(mkt)
        # Per-leg active status: Gamma sets active=False for near-resolved /
        # closed individual legs even when the parent event is still active.
        _active_raw = mkt.get("active")
        _closed_raw = mkt.get("closed")
        market_active = (
            (_active_raw is not False and _active_raw != "false")
            and (_closed_raw is not True and _closed_raw != "true")
        )

        if yes_mid is None:
            missing_price_count += 1
            yes_mid = 0.0  # placeholder to allow partial implied_sum
            price_source = "unknown"

        question = (
            mkt.get("question")
            or mkt.get("groupItemQuestion")
            or mkt.get("outcome")
            or f"Outcome {idx}"
        )
        slug = mkt.get("slug") or mkt.get("market_slug") or ""
        outcome_id = str(mkt.get("id") or mkt.get("condition_id") or f"{raw.event_id}_{idx}")

        outcomes.append(NegRiskOutcome(
            outcome_index=idx,
            outcome_id=outcome_id,
            token_id=token_id,
            question=str(question),
            slug=str(slug),
            yes_mid=yes_mid,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_mid=round(1.0 - yes_mid, 6),
            price_source=price_source,
            liquidity_raw=liquidity,
            market_active=market_active,
        ))

    if len(outcomes) < 2:
        logger.debug("Event %s has fewer than 2 parseable outcomes — skipping", raw.event_id)
        return None

    has_all_prices = missing_price_count == 0
    if not has_all_prices:
        logger.debug(
            "Event %s: %d/%d outcomes missing prices",
            raw.event_id, missing_price_count, len(outcomes)
        )

    implied_sum = sum(o.yes_mid for o in outcomes)
    constraint_gap = implied_sum - 1.0
    abs_gap = abs(constraint_gap)

    liq_values = [o.liquidity_raw for o in outcomes if o.liquidity_raw is not None]
    min_liquidity = min(liq_values) if liq_values else None

    return NegRiskEvent(
        event_id=raw.event_id,
        slug=raw.slug,
        title=raw.title,
        end_date_str=raw.end_date_str,
        neg_risk_flag=raw.neg_risk_flag,
        outcomes=outcomes,
        fetched_at=raw.fetched_at,
        normalized_at=normalized_at,
        implied_sum=round(implied_sum, 6),
        constraint_gap=round(constraint_gap, 6),
        abs_gap=round(abs_gap, 6),
        n_outcomes=len(outcomes),
        has_all_prices=has_all_prices,
        min_liquidity=min_liquidity,
        dominant_price_source=_dominant_source(outcomes),
    )


def normalize_batch(
    raw_events: list[NegRiskEventRaw],
) -> tuple[list[NegRiskEvent], list[str]]:
    """
    Normalize a batch of raw events.

    Returns:
        (normalized_events, failed_event_ids)
    """
    normalized: list[NegRiskEvent] = []
    failed: list[str] = []

    for raw in raw_events:
        try:
            result = normalize(raw)
            if result is not None:
                normalized.append(result)
            else:
                failed.append(raw.event_id)
        except Exception as exc:
            logger.warning("Normalization failed for event %s: %s", raw.event_id, exc)
            failed.append(raw.event_id)

    logger.info(
        "Normalization complete: %d normalized, %d failed",
        len(normalized), len(failed)
    )
    return normalized, failed
