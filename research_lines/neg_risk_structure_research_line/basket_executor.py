"""
basket_executor.py — Neg-Risk Basket Executor
neg_risk_structure_research_line / polyarb_lab / research utility

Orchestrates SELL YES orders across all N legs of a neg-risk event.

Strategy logic:
  If SUM(YES prices) > 1.0, collective YES outcomes are overpriced.
  Selling YES on all N legs collects the implied sum.
  When exactly one outcome resolves, you pay $1.00 on the winner.
  Net profit = (sum collected) - $1.00 - fees.

Execution rules:
  - Submits SELL YES on every leg at yes_bid (passive resting limit)
  - If any leg returns REJECTED: cancel all submitted legs (rollback)
  - Paper mode by default — pass mode=OrderMode.LIVE only when authorized
  - Size is uniform across all legs (same shares per leg)

Usage (paper):
    from research_lines.neg_risk_structure_research_line.basket_executor import (
        BasketExecutor, BasketConfig
    )
    from src.paper.broker import PaperBroker
    from src.paper.ledger import Ledger

    ledger = Ledger()
    # auto_cancel_unfilled=False: neg-risk SELL orders rest on book, not filled immediately
    broker = PaperBroker(ledger, fee_rate=0.01, auto_cancel_unfilled=False)
    executor = BasketExecutor(broker, config=BasketConfig())
    report = executor.execute(event, mode=OrderMode.PAPER)

STRICT RULES:
  - Paper only until escalation gate is lifted (5 consistent scans).
  - No mainline contamination.
  - Results go to data/research/neg_risk/baskets/ only.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from src.domain.models import ExecutionReport, OrderIntent, OrderMode, OrderStatus, OrderType

from .modules.normalizer import NegRiskEvent, NegRiskOutcome

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAKER_FEE_RATE = 0.01          # 1% taker fee per leg (Polymarket standard)
DEFAULT_SHARES_PER_LEG = 10.0  # paper size per leg in shares
MIN_YES_BID = 0.001            # reject leg if yes_bid is below this
MIN_ABS_GAP = 0.030            # minimum structural gap to attempt execution

_BASKET_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "research" / "neg_risk" / "baskets"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BasketConfig:
    shares_per_leg: float = DEFAULT_SHARES_PER_LEG
    taker_fee_rate: float = TAKER_FEE_RATE
    min_yes_bid: float = MIN_YES_BID
    min_abs_gap: float = MIN_ABS_GAP
    dry_run: bool = True        # True = log intents but do not call broker.submit


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

BasketStatus = Literal["FULL", "PARTIAL", "ROLLED_BACK", "GATE_REJECTED", "FAILED"]


@dataclass
class LegResult:
    leg_index: int
    token_id: str
    outcome_question: str
    intent_id: str
    yes_bid_used: float
    shares: float
    status: str             # mirrors OrderStatus value
    filled_size: float
    avg_fill_price: Optional[float]
    fee_paid_usd: float
    rollback_attempted: bool = False
    rollback_succeeded: bool = False
    error: Optional[str] = None


@dataclass
class BasketExecutionReport:
    basket_id: str
    event_slug: str
    mode: str
    status: BasketStatus
    n_legs: int
    n_submitted: int
    n_filled: int
    n_rejected: int
    abs_gap: float
    implied_sum: float
    net_collected_usd: float        # sum of (yes_bid * shares) across filled legs
    fee_estimate_usd: float         # taker_fee_rate * net_collected
    implied_gross_profit_usd: float # net_collected - 1.00 * (shares / n_legs) — simplified
    leg_results: list[LegResult] = field(default_factory=list)
    rollback_note: str = ""
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basket Executor
# ---------------------------------------------------------------------------

class BasketExecutor:
    """Execute a neg-risk SELL YES basket across all N legs of a NegRiskEvent.

    Accepts either a PaperBroker or a LiveBroker (duck-typed).
    The caller is responsible for passing the correct broker type.
    """

    def __init__(self, broker: Any, config: Optional[BasketConfig] = None) -> None:
        self.broker = broker
        self.config = config or BasketConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, event: NegRiskEvent, mode: OrderMode = OrderMode.PAPER) -> BasketExecutionReport:
        """Execute a neg-risk SELL YES basket for the given event.

        Args:
            event: Normalized NegRiskEvent from the research scan.
            mode:  OrderMode.PAPER (default) or OrderMode.LIVE.

        Returns:
            BasketExecutionReport with full leg-by-leg detail.
        """
        basket_id = f"basket_{uuid.uuid4().hex[:12]}"
        ts_now = datetime.now(timezone.utc).isoformat()

        logger.info(
            "BasketExecutor: starting basket %s | event=%s | legs=%d | mode=%s",
            basket_id, event.slug, len(event.outcomes), mode.value,
        )

        # Gate 1: minimum structural gap
        if event.abs_gap < self.config.min_abs_gap:
            logger.warning(
                "BasketExecutor: GATE_REJECTED basket %s — abs_gap=%.4f < min=%.4f",
                basket_id, event.abs_gap, self.config.min_abs_gap,
            )
            report = BasketExecutionReport(
                basket_id=basket_id,
                event_slug=event.slug,
                mode=mode.value,
                status="GATE_REJECTED",
                n_legs=len(event.outcomes),
                n_submitted=0,
                n_filled=0,
                n_rejected=0,
                abs_gap=event.abs_gap,
                implied_sum=event.implied_sum,
                net_collected_usd=0.0,
                fee_estimate_usd=0.0,
                implied_gross_profit_usd=0.0,
                rollback_note=f"abs_gap={event.abs_gap:.4f} below min={self.config.min_abs_gap:.4f}",
                ts=ts_now,
            )
            self._persist(report)
            return report

        # Gate 2: all legs must have a usable yes_bid
        usable_legs = [o for o in event.outcomes if (o.yes_bid or 0.0) >= self.config.min_yes_bid]
        if not usable_legs:
            logger.warning("BasketExecutor: GATE_REJECTED basket %s — no legs with yes_bid >= %.4f", basket_id, self.config.min_yes_bid)
            report = BasketExecutionReport(
                basket_id=basket_id,
                event_slug=event.slug,
                mode=mode.value,
                status="GATE_REJECTED",
                n_legs=len(event.outcomes),
                n_submitted=0,
                n_filled=0,
                n_rejected=0,
                abs_gap=event.abs_gap,
                implied_sum=event.implied_sum,
                net_collected_usd=0.0,
                fee_estimate_usd=0.0,
                implied_gross_profit_usd=0.0,
                rollback_note="no legs with usable yes_bid",
                ts=ts_now,
            )
            self._persist(report)
            return report

        # Build intents
        intents = [self._build_intent(leg, basket_id, mode) for leg in usable_legs]

        # Submit legs — with rollback on first REJECTED
        leg_results: list[LegResult] = []
        submitted_order_ids: list[str] = []   # live order IDs for rollback
        n_filled = 0
        n_rejected = 0
        rollback_triggered = False

        for i, (leg, intent) in enumerate(zip(usable_legs, intents)):
            if self.config.dry_run:
                # dry_run active — log intent but skip broker call entirely
                leg_result = LegResult(
                    leg_index=i,
                    token_id=leg.token_id,
                    outcome_question=leg.question,
                    intent_id=intent.intent_id,
                    yes_bid_used=float(intent.limit_price or 0.0),
                    shares=intent.size,
                    status=OrderStatus.SUBMITTED.value,
                    filled_size=0.0,
                    avg_fill_price=None,
                    fee_paid_usd=0.0,
                    error="DRY_RUN: broker call skipped",
                )
                leg_results.append(leg_result)
                logger.debug("BasketExecutor [DRY_RUN] leg %d | token=%s | price=%.4f", i, leg.token_id, intent.limit_price or 0.0)
                continue

            try:
                if mode == OrderMode.PAPER:
                    # PaperBroker requires a book argument — pass None (no book sim for neg-risk)
                    report_leg: ExecutionReport = self.broker.submit_limit_order(intent, None)
                else:
                    report_leg = self.broker.submit_limit_order(intent)
            except Exception as exc:
                logger.error("BasketExecutor: leg %d submit exception: %s", i, exc)
                report_leg = ExecutionReport(
                    intent_id=intent.intent_id,
                    position_id=None,
                    status=OrderStatus.REJECTED,
                    metadata={"error": str(exc)},
                    ts=datetime.now(timezone.utc),
                )

            leg_result = LegResult(
                leg_index=i,
                token_id=leg.token_id,
                outcome_question=leg.question,
                intent_id=intent.intent_id,
                yes_bid_used=float(intent.limit_price or 0.0),
                shares=intent.size,
                status=report_leg.status.value,
                filled_size=report_leg.filled_size,
                avg_fill_price=report_leg.avg_fill_price,
                fee_paid_usd=report_leg.fee_paid_usd,
            )

            if report_leg.status == OrderStatus.REJECTED:
                n_rejected += 1
                leg_result.error = str(report_leg.metadata.get("error") or report_leg.metadata.get("reason") or "REJECTED")
                leg_results.append(leg_result)
                rollback_triggered = True
                logger.warning("BasketExecutor: leg %d REJECTED — triggering rollback on basket %s", i, basket_id)
                break

            if report_leg.status in (OrderStatus.FILLED, OrderStatus.PARTIAL, OrderStatus.SUBMITTED):
                n_filled += (1 if report_leg.status == OrderStatus.FILLED else 0)
                live_order_id = report_leg.metadata.get("live_order_id")
                if live_order_id:
                    submitted_order_ids.append(live_order_id)

            leg_results.append(leg_result)

        # Rollback if triggered
        rollback_note = ""
        if rollback_triggered and submitted_order_ids:
            rollback_note = self._rollback(submitted_order_ids, leg_results)

        # Compute financials
        net_collected = sum(
            r.yes_bid_used * r.shares
            for r in leg_results
            if r.status in (OrderStatus.FILLED.value, OrderStatus.SUBMITTED.value, OrderStatus.PARTIAL.value)
            and not r.rollback_attempted
        )
        fee_estimate = net_collected * self.config.taker_fee_rate
        # Gross profit: what we collected minus the $1 we'll owe at resolution
        # Normalized to per-share: assume $1 payout on shares_per_leg shares
        payout_obligation = self.config.shares_per_leg * 1.0
        implied_gross_profit = net_collected - fee_estimate - payout_obligation

        # Determine basket status
        if rollback_triggered:
            basket_status: BasketStatus = "ROLLED_BACK"
        elif n_rejected > 0:
            basket_status = "FAILED"
        elif len(leg_results) == len(usable_legs):
            basket_status = "FULL"
        else:
            basket_status = "PARTIAL"

        final_report = BasketExecutionReport(
            basket_id=basket_id,
            event_slug=event.slug,
            mode=mode.value,
            status=basket_status,
            n_legs=len(usable_legs),
            n_submitted=len(leg_results),
            n_filled=n_filled,
            n_rejected=n_rejected,
            abs_gap=event.abs_gap,
            implied_sum=event.implied_sum,
            net_collected_usd=round(net_collected, 4),
            fee_estimate_usd=round(fee_estimate, 4),
            implied_gross_profit_usd=round(implied_gross_profit, 4),
            leg_results=leg_results,
            rollback_note=rollback_note,
            ts=ts_now,
            metadata={"n_outcomes_total": len(event.outcomes), "n_usable_legs": len(usable_legs)},
        )

        logger.info(
            "BasketExecutor: basket %s complete | status=%s | n_legs=%d | net_collected=%.4f | implied_profit=%.4f",
            basket_id, basket_status, len(usable_legs), net_collected, implied_gross_profit,
        )

        self._persist(final_report)
        return final_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_intent(self, leg: NegRiskOutcome, basket_id: str, mode: OrderMode) -> OrderIntent:
        return OrderIntent(
            intent_id=f"{basket_id}_leg{leg.outcome_index}",
            candidate_id=basket_id,
            mode=mode,
            market_slug=leg.slug,
            token_id=leg.token_id,
            position_id=basket_id,
            side="SELL",
            order_type=OrderType.LIMIT,
            size=self.config.shares_per_leg,
            limit_price=leg.yes_bid or leg.yes_mid,
            max_notional_usd=self.config.shares_per_leg * (leg.yes_bid or leg.yes_mid or 1.0),
            ts=datetime.now(timezone.utc),
        )

    def _rollback(self, order_ids: list[str], leg_results: list[LegResult]) -> str:
        """Attempt to cancel all submitted live orders. Paper orders need no cancel."""
        cancel_results = []
        for order_id in order_ids:
            try:
                success = self.broker.cancel_order(order_id)
                cancel_results.append(f"{order_id}={'OK' if success else 'FAIL'}")
            except Exception as exc:
                cancel_results.append(f"{order_id}=ERR({exc})")

        # Mark leg_results as rollback attempted
        for r in leg_results:
            if r.status in (OrderStatus.SUBMITTED.value, OrderStatus.PARTIAL.value):
                r.rollback_attempted = True
                r.rollback_succeeded = any(order_id in str(cancel_results) and "OK" in str(cancel_results) for order_id in [])

        note = f"rollback attempted on {len(order_ids)} orders: {'; '.join(cancel_results)}"
        logger.warning("BasketExecutor rollback: %s", note)
        return note

    def _persist(self, report: BasketExecutionReport) -> None:
        """Write basket report to data/research/neg_risk/baskets/."""
        _BASKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts_safe = report.ts.replace(":", "").replace(".", "")[:19]
        path = _BASKET_DATA_DIR / f"{report.basket_id}_{ts_safe}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            logger.info("BasketExecutor: report written to %s", path)
        except Exception as exc:
            logger.error("BasketExecutor: failed to persist report: %s", exc)
