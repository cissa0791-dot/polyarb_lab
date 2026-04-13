from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml

from src.config_runtime.loader import load_runtime_config
from src.config_runtime.models import ExecutionConfig, OpportunityConfig, PaperConfig, RiskConfig
from src.core.fees import total_buffer_cents
from src.core.normalize import build_yes_no_pairs
from src.core.orderbook_validation import (
    FEASIBILITY_FAILURE,
    build_fetch_failure_validation,
    orderbook_failure_class,
    validate_orderbook,
)
from src.domain.models import (
    ExecutionReport,
    OrderIntent,
    OrderMode,
    OrderStatus,
    OrderType,
    PositionState,
    RejectionEvent,
    RejectionReason,
    RiskDecision,
    Severity,
    SystemEvent,
)
from src.ingest.clob import ReadOnlyClob
from src.ingest.gamma import fetch_events, fetch_markets, fetch_markets_from_events, fetch_markets_with_slice, flatten_event_markets
from src.monitoring.logger import configure_logging, get_logger
from src.opportunity.audit import QualificationAuditor
from src.opportunity.models import RankedOpportunity
from src.opportunity.qualification import ExecutionFeasibilityEvaluator, OpportunityRanker
from src.live.broker import LiveBroker
from src.live.reconciler import CompletedOrder, FillReconciler
from src.paper.broker import PaperBroker
from src.paper.exit_policy import evaluate_exit
from src.paper.ledger import Ledger
from src.domain.models import RunSummary
from src.reporting.summary import RunSummaryBuilder
from src.risk.manager import RiskManager
from src.sizing.engine import DepthCappedSizer
from src.storage.event_store import ResearchStore
from src.runtime.parameter_sets import apply_runtime_parameter_set
from src.runtime.market_universe import (
    HotMarketWebsocketClient,
    MarketUniverseManager,
    MarketTier,
    RefreshPlan,
    build_pair_snapshot,
)
from src.strategies.opportunity_strategies import (
    CrossMarketConstraintStrategy,
    CrossMarketExecutionGrossConstraintStrategy,
    CrossMarketGrossConstraintStrategy,
    NegRiskRebalancingStrategy,
    SingleMarketMispricingStrategy,
    SingleMarketTouchMispricingStrategy,
)
from src.utils.db import OpportunityStore
from src.sidecar.ab_bridge import ABSidecar


def _read_yaml(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _token_map_from_pairs(pairs: list) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for pair in pairs:
        out[(pair.market_slug, "YES")] = pair.yes_token_id
        out[(pair.market_slug, "NO")] = pair.no_token_id
    return out


def _simplified_market_lookup(simplified_markets: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for market in simplified_markets:
        tokens = market.get("tokens") or []
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            if not isinstance(token, dict):
                continue
            token_id = str(token.get("token_id") or token.get("asset_id") or "")
            if token_id:
                lookup[token_id] = market
    return lookup


def _augment_markets_with_simplified(markets: list[dict], simplified_lookup: dict[str, dict]) -> list[dict]:
    if not simplified_lookup:
        return markets
    augmented: list[dict] = []
    for market in markets:
        token_ids = market.get("clobTokenIds") or market.get("clob_token_ids") or []
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = []
        simplified = None
        for token_id in token_ids:
            simplified = simplified_lookup.get(str(token_id))
            if simplified is not None:
                break
        if simplified is None:
            augmented.append(market)
            continue
        merged = dict(market)
        merged.setdefault("acceptingOrders", simplified.get("accepting_orders"))
        merged.setdefault("active", simplified.get("active", market.get("active")))
        merged.setdefault("closed", simplified.get("closed", market.get("closed")))
        merged.setdefault("archived", simplified.get("archived", market.get("archived")))
        augmented.append(merged)
    return augmented


def _limit_price_for_target_shares(levels: list, shares: float) -> float | None:
    remaining = float(shares)
    last_price: float | None = None
    for level in levels:
        size = float(level.size)
        if size <= 0:
            continue
        last_price = float(level.price)
        remaining -= min(remaining, size)
        if remaining <= 1e-9:
            return last_price
    return None


def _complement_side(side: str) -> str:
    return "NO" if str(side).upper() == "YES" else "YES"


class _BidFillBook:
    """Presents bid levels as 'asks' so PaperBroker's BUY fill path simulates
    a passive maker fill at bid-side prices instead of taker ask-side prices."""

    def __init__(self, book: object) -> None:
        object.__setattr__(self, "_book", book)

    def __getattr__(self, name: str) -> object:
        book = object.__getattribute__(self, "_book")
        if name == "asks":
            return getattr(book, "bids", [])
        return getattr(book, name)


def _best_bid_ask(book: object) -> tuple[float | None, float | None, float | None, float | None]:
    bids = getattr(book, "bids", [])
    asks = getattr(book, "asks", [])
    best_bid = float(bids[0].price) if bids else None
    best_ask = float(asks[0].price) if asks else None
    bid_size = float(bids[0].size) if bids else None
    ask_size = float(asks[0].size) if asks else None
    return best_bid, best_ask, bid_size, ask_size


def _midpoint_from_book(book: object) -> float | None:
    best_bid, best_ask, _, _ = _best_bid_ask(book)
    if best_bid is None or best_ask is None or best_ask <= best_bid:
        return None
    return round((best_bid + best_ask) / 2.0, 6)


def _current_inventory_shares(position_records: dict, token_id: str) -> float:
    return round(
        sum(
            rec.remaining_shares
            for rec in position_records.values()
            if rec.symbol == token_id and rec.is_open
        ),
        6,
    )


def _inventory_bias(side: str, inventory_shares: float) -> str:
    side_upper = side.upper()
    if side_upper == "BUY":
        if inventory_shares > 1e-9:
            return "worsen"
        if inventory_shares < -1e-9:
            return "favor"
        return "flat"
    if inventory_shares < -1e-9:
        return "worsen"
    if inventory_shares > 1e-9:
        return "favor"
    return "flat"


def _inventory_penalty_and_suppression(
    side: str,
    inventory_shares: float,
    config: ExecutionConfig,
) -> dict[str, float | bool | str]:
    bias = _inventory_bias(side, inventory_shares)
    soft_limit = max(0.0, float(config.maker_quote_inventory_soft_limit_shares))
    hard_limit = max(soft_limit, float(config.maker_quote_inventory_hard_limit_shares))
    skew_cents = max(0.0, float(config.maker_quote_inventory_skew_cents))
    inv_abs = abs(inventory_shares)

    suppressed = bool(bias == "worsen" and hard_limit > 0.0 and inv_abs >= hard_limit)
    if hard_limit <= soft_limit:
        pressure = 1.0 if inv_abs >= hard_limit and hard_limit > 0.0 else 0.0
    else:
        pressure = min(max(inv_abs - soft_limit, 0.0) / max(hard_limit - soft_limit, 1e-9), 1.0)

    penalty = skew_cents * pressure if bias == "worsen" else 0.0
    return {
        "bias": bias,
        "pressure": round(pressure, 6),
        "inventory_penalty_cents": round(penalty, 6),
        "suppressed": suppressed,
    }


def _apply_inventory_skew(
    side: str,
    quote_price: float,
    inventory_shares: float,
    config: ExecutionConfig,
) -> dict[str, float | bool | str]:
    penalty = _inventory_penalty_and_suppression(side, inventory_shares, config)
    bias = str(penalty["bias"])
    skew_cents = float(config.maker_quote_inventory_skew_cents) * float(penalty["pressure"])
    adjusted_price = quote_price
    if bias == "worsen":
        adjusted_price = quote_price - skew_cents if side.upper() == "BUY" else quote_price + skew_cents
    elif bias == "favor":
        adjusted_price = quote_price + skew_cents if side.upper() == "BUY" else quote_price - skew_cents
    adjusted_price = round(min(0.999, max(0.001, adjusted_price)), 6)
    return {
        **penalty,
        "skew_cents_applied": round(skew_cents, 6),
        "adjusted_quote_price": adjusted_price,
    }


def _evaluate_quote_health(
    *,
    side: str,
    quote_price: float,
    size: float,
    book: object,
    inventory_shares: float,
    posted_ts: datetime,
    now_ts: datetime,
    opportunity_config: OpportunityConfig,
    execution_config: ExecutionConfig,
    qualified_gross_edge_cents: float | None = None,
    qualified_fee_impact_cents: float | None = None,
    qualified_slippage_cents: float | None = None,
    qualified_net_edge_cents: float | None = None,
) -> dict[str, float | bool | str | None]:
    best_bid, best_ask, bid_size, ask_size = _best_bid_ask(book)
    fair_value = _midpoint_from_book(book)
    inventory = _inventory_penalty_and_suppression(side, inventory_shares, execution_config)
    age_sec = max(0.0, (now_ts - posted_ts).total_seconds())
    threshold = float(execution_config.maker_quote_min_expected_net_edge_cents)
    using_qualified_edge = qualified_gross_edge_cents is not None and qualified_net_edge_cents is not None

    if using_qualified_edge:
        gross_edge = round(float(qualified_gross_edge_cents), 6)
        expected_fee = round(float(qualified_fee_impact_cents or 0.0), 6)
        slippage_proxy = round(float(qualified_slippage_cents or 0.0), 6)
        expected_net_edge_pre_inventory = round(float(qualified_net_edge_cents), 6)
    else:
        gross_edge = 0.0
        expected_fee = round(opportunity_config.fee_buffer_cents, 6)
        slippage_proxy = round(opportunity_config.slippage_buffer_cents, 6)
        expected_net_edge_pre_inventory = round(gross_edge - expected_fee - slippage_proxy, 6)

    inventory_penalty = float(inventory["inventory_penalty_cents"])
    expected_net_edge = round(expected_net_edge_pre_inventory - inventory_penalty, 6)

    if fair_value is None or best_bid is None or best_ask is None:
        return {
            "should_cancel": True,
            "cancel_reason": "price_guard",
            "current_fair_value": fair_value,
            "quote_distance_from_fair": None,
            "expected_gross_edge": gross_edge,
            "expected_fee_impact": expected_fee,
            "slippage_depth_proxy": slippage_proxy,
            "inventory_penalty": inventory_penalty,
            "expected_net_edge": expected_net_edge,
            "age_sec": round(age_sec, 6),
            "price_guard_failed": True,
            "inventory_suppressed": bool(inventory["suppressed"]),
            "inventory_bias": str(inventory["bias"]),
            "threshold_used": threshold,
            "comparison_operator": "<",
            "edge_unit": "usd_per_share_price_points",
            "baseline_source": "qualification" if using_qualified_edge else "quote_local",
        }

    quote_distance = round(quote_price - fair_value, 6)
    if side.upper() == "BUY":
        local_gross_edge = fair_value - quote_price
        same_side_top_size = bid_size or 0.0
        price_guard_failed = quote_price >= best_ask
    else:
        local_gross_edge = quote_price - fair_value
        same_side_top_size = ask_size or 0.0
        price_guard_failed = quote_price <= best_bid

    if not using_qualified_edge:
        depth_shortfall_ratio = min(max(size - same_side_top_size, 0.0) / max(size, 1e-9), 1.0) if size > 1e-9 else 0.0
        slippage_proxy = round(opportunity_config.slippage_buffer_cents * depth_shortfall_ratio, 6)
        expected_fee = round(opportunity_config.fee_buffer_cents, 6)
        gross_edge = round(local_gross_edge, 6)
        expected_net_edge_pre_inventory = round(gross_edge - expected_fee - slippage_proxy, 6)
        expected_net_edge = round(expected_net_edge_pre_inventory - inventory_penalty, 6)
    drift_too_wide = abs(quote_distance) > execution_config.maker_quote_max_fair_value_drift_cents
    stale_quote = age_sec > execution_config.maker_quote_max_age_sec
    inventory_suppressed = bool(inventory["suppressed"])

    cancel_reason: str | None = None
    if inventory_suppressed:
        cancel_reason = "inventory_suppressed"
    elif price_guard_failed or drift_too_wide:
        cancel_reason = "price_guard"
    elif stale_quote:
        cancel_reason = "stale_quote"
    elif expected_net_edge < threshold:
        cancel_reason = "profitability"

    return {
        "should_cancel": cancel_reason is not None,
        "cancel_reason": cancel_reason,
        "current_fair_value": round(fair_value, 6),
        "quote_distance_from_fair": quote_distance,
        "expected_gross_edge": round(gross_edge, 6),
        "expected_fee_impact": expected_fee,
        "slippage_depth_proxy": slippage_proxy,
        "inventory_penalty": inventory_penalty,
        "expected_net_edge": expected_net_edge,
        "age_sec": round(age_sec, 6),
        "price_guard_failed": price_guard_failed or drift_too_wide,
        "inventory_suppressed": inventory_suppressed,
        "inventory_bias": str(inventory["bias"]),
        "threshold_used": threshold,
        "comparison_operator": "<",
        "edge_unit": "usd_per_share_price_points",
        "baseline_source": "qualification" if using_qualified_edge else "quote_local",
    }


def _quote_age_bucket(age_sec: float) -> str:
    if age_sec < 30.0:
        return "lt_30s"
    if age_sec < 60.0:
        return "30s_to_60s"
    return "gte_60s"


def _terminal_order_status(completed: CompletedOrder) -> OrderStatus:
    """Map a completed live order's CLOB state to a terminal OrderStatus.

    Always returns FILLED or CANCELED — never SUBMITTED or PARTIAL — so that
    load_pending_live_orders() correctly excludes the new terminal row from
    startup recovery on the next process start.

    Mapping:
      CLOB "matched"                         → FILLED
      any status with full fill detected     → FILLED  (size_matched >= intent.size)
      "canceled" / "cancelled" / "expired"   → CANCELED (including cancel-with-partial)
    """
    if (
        completed.final_clob_status.lower() == "matched"
        or completed.final_size_matched >= completed.intent.size - 1e-9
    ):
        return OrderStatus.FILLED
    return OrderStatus.CANCELED


def _build_basket_audit(
    position_records: dict,
    candidate_id: str | None,
    now_ts: datetime,
) -> dict:
    """Compute basket-level audit fields for all legs sharing candidate_id.

    Returns a dict suitable for merging into TradeSummary.metadata.
    Reads only existing state from position_records — no side-effects.
    """
    if candidate_id is None:
        return {"basket_audit": None}

    basket_legs = [rec for rec in position_records.values() if rec.candidate_id == candidate_id]
    if not basket_legs:
        return {"basket_audit": None}

    # Per-leg PnL: realized for closed legs, last_unrealized for open legs.
    leg_pnls = {
        rec.position_id: rec.realized_pnl_usd if not rec.is_open else rec.last_unrealized_pnl_usd
        for rec in basket_legs
    }

    basket_unrealized_pnl = round(sum(leg_pnls.values()), 6)
    basket_peak_unrealized_pnl = round(sum(rec.peak_unrealized_pnl_usd for rec in basket_legs), 6)
    basket_peak_to_current_drawdown = round(basket_peak_unrealized_pnl - basket_unrealized_pnl, 6)

    # Dominant loss leg share.
    negative_pnls = [pnl for pnl in leg_pnls.values() if pnl < 0]
    total_loss = sum(abs(p) for p in negative_pnls) if negative_pnls else 0.0
    if total_loss > 1e-9:
        dominant_loss_pnl = min(leg_pnls.values())
        dominant_loss_leg_share = round(abs(dominant_loss_pnl) / total_loss, 6)
    else:
        dominant_loss_leg_share = 0.0

    # Trigger leg = ED-closed leg with earliest closed_ts.
    ed_closed = [
        rec for rec in basket_legs
        if not rec.is_open and rec.close_reason == "EDGE_DECAY" and rec.closed_ts is not None
    ]
    trigger_leg_id: str | None = None
    trigger_leg_loss_share: float | None = None
    if ed_closed:
        trigger_rec = min(ed_closed, key=lambda r: r.closed_ts)  # type: ignore[arg-type]
        trigger_leg_id = trigger_rec.position_id
        if total_loss > 1e-9:
            trigger_pnl = leg_pnls.get(trigger_leg_id, 0.0)
            trigger_leg_loss_share = round(abs(trigger_pnl) / total_loss, 6) if trigger_pnl < 0 else 0.0

    # Time since first adverse state across all basket legs.
    adverse_timestamps = [
        rec.first_adverse_ts for rec in basket_legs if rec.first_adverse_ts is not None
    ]
    if adverse_timestamps:
        earliest_adverse = min(adverse_timestamps)
        try:
            adverse_dt = datetime.fromisoformat(earliest_adverse)
            if adverse_dt.tzinfo is None:
                adverse_dt = adverse_dt.replace(tzinfo=timezone.utc)
            time_since_first_adverse_state = round((now_ts - adverse_dt).total_seconds(), 1)
        except (ValueError, TypeError):
            time_since_first_adverse_state = -1.0
    else:
        time_since_first_adverse_state = -1.0

    # exit_path_classification.
    if not ed_closed:
        exit_path_classification = "MHA_only"
    elif not negative_pnls:
        # Basket profitable overall despite ED exit (trigger forced exit while basket was fine).
        exit_path_classification = "dominant_leg_candidate"
    elif dominant_loss_leg_share < 0.40:
        # No single leg dominates; loss is spread across multiple legs.
        exit_path_classification = "aggregate_basket_deterioration"
    else:
        # Single dominant loser exists.
        dominant_position_id = min(leg_pnls, key=lambda pid: leg_pnls[pid])
        if dominant_position_id == trigger_leg_id:
            exit_path_classification = "dominant_leg_candidate"
        else:
            exit_path_classification = "minor_leg_candidate"

    return {
        "basket_audit": {
            "basket_unrealized_pnl": basket_unrealized_pnl,
            "basket_peak_unrealized_pnl": basket_peak_unrealized_pnl,
            "basket_peak_to_current_drawdown": basket_peak_to_current_drawdown,
            "dominant_loss_leg_share": dominant_loss_leg_share,
            "trigger_leg_loss_share": trigger_leg_loss_share,
            "time_since_first_adverse_state": time_since_first_adverse_state,
            "exit_path_classification": exit_path_classification,
            "basket_leg_count": len(basket_legs),
            "trigger_leg_id": trigger_leg_id,
        }
    }


def _basket_exit_confirmed(
    position_records: dict,
    candidate_id: str | None,
    this_position_id: str,
    config: "PaperConfig",
) -> bool:
    """Return True if this natural EDGE_DECAY trigger passes the basket-level dominance gate.

    Gate is disabled (always returns True) when config.basket_dominance_threshold <= 0.0,
    preserving current behavior for sessions that do not set the threshold.

    When active, the exit is confirmed if ANY of the following paths is satisfied:

    Path A — dominant-leg confirmation:
      - this_position_id is the dominant loss leg (worst unrealized/realized PnL in basket), OR
      - dominant_loss_leg_share >= basket_dominance_threshold

    Path B — aggregate basket deterioration override:
      - basket_peak_to_current_drawdown >= basket_drawdown_exit_threshold (if threshold > 0), OR
      - basket_unrealized_pnl <= basket_unrealized_pnl_floor (if floor < 0)

    Only called for natural EDGE_DECAY exits (force_exit=False).
    Cascade exits (force_exit=True) bypass this function entirely.
    """
    if config.basket_dominance_threshold <= 0.0:
        return True
    if candidate_id is None:
        return True

    basket_legs = [rec for rec in position_records.values() if rec.candidate_id == candidate_id]
    if not basket_legs:
        return True

    leg_pnls = {
        rec.position_id: rec.realized_pnl_usd if not rec.is_open else rec.last_unrealized_pnl_usd
        for rec in basket_legs
    }

    basket_pnl = sum(leg_pnls.values())
    basket_peak = sum(rec.peak_unrealized_pnl_usd for rec in basket_legs)
    basket_drawdown = basket_peak - basket_pnl

    # Path B: aggregate basket deterioration overrides dominance check.
    if config.basket_drawdown_exit_threshold > 0.0 and basket_drawdown >= config.basket_drawdown_exit_threshold:
        return True
    if config.basket_unrealized_pnl_floor < 0.0 and basket_pnl <= config.basket_unrealized_pnl_floor:
        return True

    # Path A: dominant-leg confirmation.
    negative_pnls = [pnl for pnl in leg_pnls.values() if pnl < 0]
    total_loss = sum(abs(p) for p in negative_pnls) if negative_pnls else 0.0

    if total_loss <= 1e-9:
        # Basket has no negative-PnL legs; trigger fired on a profitable basket. Confirm.
        return True

    dominant_position_id = min(leg_pnls, key=lambda pid: leg_pnls[pid])
    dominant_loss_leg_share = abs(leg_pnls[dominant_position_id]) / total_loss

    return dominant_position_id == this_position_id or dominant_loss_leg_share >= config.basket_dominance_threshold


def _has_basket_ed_exit(
    position_records: dict,
    candidate_id: str | None,
    this_position_id: str,
) -> bool:
    """Return True if any sibling position in the same basket has been closed via EDGE_DECAY.

    Used by _manage_open_positions to cascade a basket-level exit when one leg
    triggers EDGE_DECAY: all remaining open legs in the same basket are then
    force-exited with the same reason code.
    """
    if candidate_id is None:
        return False
    return any(
        rec.close_reason == "EDGE_DECAY"
        for rec in position_records.values()
        if rec.candidate_id == candidate_id
        and rec.position_id != this_position_id
        and not rec.is_open
    )


def _has_basket_sl_exit(
    position_records: dict,
    candidate_id: str | None,
    this_position_id: str,
) -> bool:
    """Return True if any sibling position in the same basket has been closed via STOP_LOSS.

    Used by _manage_open_positions to cascade a basket-level exit when one leg
    triggers STOP_LOSS: all remaining open legs in the same basket are then
    force-exited with the same reason code.
    """
    if candidate_id is None:
        return False
    return any(
        rec.close_reason == "STOP_LOSS"
        for rec in position_records.values()
        if rec.candidate_id == candidate_id
        and rec.position_id != this_position_id
        and not rec.is_open
    )


def _has_basket_idle_release(
    position_records: dict,
    candidate_id: str | None,
    this_position_id: str,
) -> bool:
    """Return True if any sibling position in the same basket already closed via IDLE_HOLD_RELEASE."""
    if candidate_id is None:
        return False
    return any(
        rec.close_reason == "IDLE_HOLD_RELEASE"
        for rec in position_records.values()
        if rec.candidate_id == candidate_id
        and rec.position_id != this_position_id
        and not rec.is_open
    )


def _basket_idle_release_eligible(
    position_records: dict,
    candidate_id: str | None,
    config: "PaperConfig",
    now_ts: datetime,
) -> bool:
    """Return True only for clearly inert baskets eligible for early idle release.

    This intentionally targets only the no_state_evolution bucket:
    no adverse state, no EDGE_DECAY candidate formation, near-flat PnL/drawdown,
    and minimal repricing after a mid-hold checkpoint.
    """
    if config.idle_hold_release_check_sec <= 0.0 or candidate_id is None:
        return False

    basket_legs = [
        rec
        for rec in position_records.values()
        if rec.candidate_id == candidate_id and rec.is_open
    ]
    if not basket_legs:
        return False

    min_age_sec = min(
        (now_ts - datetime.fromisoformat(rec.opened_ts)).total_seconds()
        for rec in basket_legs
    )
    if min_age_sec < config.idle_hold_release_check_sec:
        return False
    if config.max_holding_sec > 0.0 and min_age_sec >= config.max_holding_sec:
        return False

    if any(rec.first_adverse_ts is not None for rec in basket_legs):
        return False
    if any(getattr(rec, "edge_decay_candidate_count", 0) > 0 for rec in basket_legs):
        return False

    total_reprices = sum(getattr(rec, "repricing_event_count", 0) for rec in basket_legs)
    if total_reprices > config.idle_hold_release_max_repricing_events:
        return False

    basket_unrealized_pnl = sum(rec.last_unrealized_pnl_usd for rec in basket_legs)
    basket_peak_unrealized_pnl = sum(rec.peak_unrealized_pnl_usd for rec in basket_legs)
    basket_drawdown = basket_peak_unrealized_pnl - basket_unrealized_pnl

    if abs(basket_unrealized_pnl) > config.idle_hold_release_max_abs_unrealized_pnl:
        return False
    if basket_drawdown > config.idle_hold_release_max_drawdown:
        return False

    return True


# Qualification-stage reasons that indicate a marginal near-miss (candidate nearly passed).
# ABSOLUTE_DEPTH_BELOW_FLOOR: excluded — hard liquidity floor failure, not a marginal rejection.
# SIZED_NOTIONAL_TOO_SMALL: handled at the sizing stage (see _qualify_and_rank_candidate).
_QUALIFICATION_NEAR_MISS_REASONS: frozenset[str] = frozenset({
    RejectionReason.EDGE_BELOW_THRESHOLD.value,
    RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value,
    RejectionReason.INSUFFICIENT_DEPTH.value,
    RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value,
})


class ResearchRunner:
    def __init__(
        self,
        settings_path: str = "config/settings.yaml",
        constraints_path: str = "config/constraints.yaml",
        debug_output_dir: str | Path | None = None,
    ):
        self._base_config = load_runtime_config(settings_path)
        self.config = self._base_config.model_copy(deep=True)
        configure_logging(
            level=self.config.monitoring.log_level,
            json_logs=self.config.monitoring.json_logs,
        )
        self.logger = get_logger("polyarb.runtime")
        self.constraints_path = constraints_path
        self.store = ResearchStore(self.config.storage.sqlite_url)
        self.opportunity_store = OpportunityStore(self.config.storage.sqlite_url)
        self.clob = ReadOnlyClob(self.config.market_data.clob_host)
        self.market_universe = MarketUniverseManager(self.config.market_data)
        self.hot_market_stream = HotMarketWebsocketClient()
        self.paper_ledger = Ledger(cash=self.config.paper.starting_cash)
        self.paper_broker = PaperBroker(self.paper_ledger)
        self.live_broker: LiveBroker | None = None
        self.fill_reconciler: FillReconciler | None = None
        self._startup_recovery_done: bool = False
        self.single_market_strategy = SingleMarketMispricingStrategy()
        self.single_market_touch_strategy = SingleMarketTouchMispricingStrategy()
        self.cross_market_strategy = CrossMarketConstraintStrategy()
        self.cross_market_gross_strategy = CrossMarketGrossConstraintStrategy()
        self.cross_market_execution_gross_strategy = CrossMarketExecutionGrossConstraintStrategy()
        self.neg_risk_strategy = NegRiskRebalancingStrategy()
        self._rebuild_runtime_components()
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir is not None else Path("data/reports/orderbook_debug")
        self._current_run_id: str | None = None
        self._current_summary: RunSummaryBuilder | None = None
        self._experiment_context: dict[str, object] = {}
        self._invalid_orderbook_export_path: Path | None = None
        self._run_sequence = 0
        self._current_auditor: QualificationAuditor | None = None
        self._liquidity_skip_state: dict[tuple[str, str, str], dict[str, int]] = {}
        self._empty_asks_skip_threshold = 2
        self._empty_asks_skip_cooldown_runs = 2
        # A+B sidecar bridge — observation only, no Track A logic affected.
        self._ab_sidecar = ABSidecar(self.config.storage.sqlite_url)
        self._quote_guard_metrics: dict[str, object] = {}
        self._cycle_metrics: dict[str, object] = {}
        self._hot_tier_zero_cycles = 0
        self._seed_market_universe_from_history()

    def apply_runtime_parameter_set(self, parameter_set_label: str) -> None:
        self.config = apply_runtime_parameter_set(self._base_config, parameter_set_label)
        self._rebuild_runtime_components()

    def reset_paper_state(self) -> None:
        """Reset in-memory paper ledger and broker to a clean starting state.

        Replaces the current Ledger and PaperBroker instances with fresh ones
        initialised from config.paper.starting_cash.  The SQLite store is NOT
        touched; all prior run records remain intact.

        Called by BatchResearchRunner between cycles when
        config.paper.inter_cycle_reset is True.  No-op if called outside that
        context (safe to call manually).
        """
        self.paper_ledger = Ledger(cash=self.config.paper.starting_cash)
        self.paper_broker = PaperBroker(self.paper_ledger)

    # ------------------------------------------------------------------
    # L3/L4 dry-run guard and execution-mode routing
    # ------------------------------------------------------------------

    @property
    def _effective_live_enabled(self) -> bool:
        """True only when both live_enabled=True AND dry_run=False.

        dry_run=True always wins regardless of live_enabled, preventing
        accidental live order submission before the dry_run gate is
        explicitly cleared.  L4 routing reads this property instead of
        config.execution.live_enabled directly.
        """
        return self.config.execution.live_enabled and not self.config.execution.dry_run

    @property
    def _order_mode(self) -> OrderMode:
        """Resolved execution mode for new order intents.

        Returns OrderMode.LIVE when _effective_live_enabled is True,
        OrderMode.PAPER otherwise.  All 4 intent creation sites read
        this property so routing logic lives in exactly one place.
        """
        return OrderMode.LIVE if self._effective_live_enabled else OrderMode.PAPER

    def _check_live_dry_run_conflict(self) -> None:
        """Warn when live_enabled and dry_run are simultaneously True.

        The combination is contradictory: the caller intends live
        execution but the dry_run gate is still active.  Paper mode
        runs normally; the warning surfaces the misconfiguration in the
        log without blocking execution.
        """
        if self.config.execution.live_enabled and self.config.execution.dry_run:
            self.logger.warning(
                "live_enabled=True is overridden by dry_run=True — "
                "running in paper mode. Set execution.dry_run=False "
                "to enable live order submission."
            )

    def _poll_live_fills(self) -> None:
        """Reconcile fill deltas for pending live orders into the paper ledger.

        Called once at the start of each run_once() cycle so that any fills
        that arrived since the previous cycle are visible to position
        management and exit evaluation before market scanning begins.

        For each order that reached terminal state during this poll, writes
        exactly one terminal ExecutionReport row to the store.  This closes
        the audit gap where asynchronously-completed live orders had no
        terminal row in execution_reports.  The terminal row's status is
        always FILLED or CANCELED so load_pending_live_orders() correctly
        excludes it from future startup recovery.
        """
        if self.fill_reconciler is None:
            return
        completed = self.fill_reconciler.poll(self.paper_ledger)
        for c in completed:
            terminal_status = _terminal_order_status(c)
            report = ExecutionReport(
                intent_id=c.intent.intent_id,
                position_id=c.intent.position_id,
                status=terminal_status,
                filled_size=round(c.final_size_matched, 6),
                avg_fill_price=(
                    c.final_avg_price
                    if c.final_size_matched > 1e-9
                    else None
                ),
                metadata={
                    "live_order_id": c.live_order_id,
                    "live_status": c.final_clob_status,
                    "terminal": True,
                },
                ts=datetime.now(timezone.utc),
            )
            self.store.save_execution_report(report)
            if c.final_size_matched > 1e-9:
                self._record_quote_fill_metrics(c.intent, report.avg_fill_price, report.ts)
            # Mirror fill/cancel counts into the current RunSummary so async
            # live fills are counted identically to synchronous paper fills.
            self._record_report_stats(report)
            # Emit position_opened event for async fills that produced shares,
            # closing the audit gap where live fills had no position_event row.
            if c.final_size_matched > 1e-9 and c.intent.position_id:
                position = self.paper_ledger.position_records.get(c.intent.position_id)
                if position is not None:
                    self.store.save_position_event(
                        position_id=position.position_id,
                        candidate_id=position.candidate_id,
                        event_type="position_opened",
                        symbol=position.symbol,
                        market_slug=position.market_slug,
                        state=position.state.value,
                        reason_code=None,
                        payload={
                            "filled_size": report.filled_size,
                            "avg_fill_price": report.avg_fill_price,
                            "intent_id": report.intent_id,
                            "live_order_id": c.live_order_id,
                            "source": "async_fill_reconciler",
                        },
                        ts=report.ts,
                    )

    def _refresh_active_maker_quotes(self) -> None:
        """Refresh pending live maker quotes and cancel stale/unprofitable ones."""
        if not self._effective_live_enabled:
            return
        if self.live_broker is None or self.fill_reconciler is None:
            return

        for live_order_id, intent, _, _ in self.fill_reconciler.snapshot_pending():
            guard_meta = (intent.metadata or {}).get("quote_guard") or {}
            if not guard_meta:
                continue
            try:
                book = self.clob.get_book(intent.token_id)
            except Exception as exc:
                self._record_event(
                    "maker_quote_refresh_failed",
                    Severity.WARNING,
                    f"maker quote refresh book fetch failed: {exc}",
                    {"live_order_id": live_order_id, "token_id": intent.token_id, "error": str(exc)},
                )
                continue

            inventory_shares = _current_inventory_shares(self.paper_ledger.position_records, intent.token_id)
            health = _evaluate_quote_health(
                side=intent.side,
                quote_price=float(intent.limit_price or 0.0),
                size=intent.size,
                book=book,
                inventory_shares=inventory_shares,
                posted_ts=intent.ts,
                now_ts=datetime.now(timezone.utc),
                opportunity_config=self.config.opportunity,
                execution_config=self.config.execution,
                qualified_gross_edge_cents=float(guard_meta["qualified_gross_edge_cents"]) if guard_meta.get("qualified_gross_edge_cents") is not None else None,
                qualified_fee_impact_cents=float(guard_meta["qualified_fee_impact_cents"]) if guard_meta.get("qualified_fee_impact_cents") is not None else None,
                qualified_slippage_cents=float(guard_meta["qualified_slippage_cents"]) if guard_meta.get("qualified_slippage_cents") is not None else None,
                qualified_net_edge_cents=float(guard_meta["qualified_net_edge_cents"]) if guard_meta.get("qualified_net_edge_cents") is not None else None,
            )
            if not health["should_cancel"]:
                continue

            if not self.live_broker.cancel_order(live_order_id):
                self._record_event(
                    "maker_quote_cancel_failed",
                    Severity.WARNING,
                    "maker quote cancel failed",
                    {"live_order_id": live_order_id, "candidate_id": intent.candidate_id},
                )
                continue

            self.fill_reconciler.unregister(live_order_id)
            self._record_quote_cancel_metric(str(health["cancel_reason"]))
            ts = datetime.now(timezone.utc)
            self.store.save_execution_report(
                ExecutionReport(
                    intent_id=intent.intent_id,
                    position_id=intent.position_id,
                    status=OrderStatus.CANCELED,
                    metadata={
                        "live_order_id": live_order_id,
                        "cancel_reason": health["cancel_reason"],
                        "quote_guard": health,
                    },
                    ts=ts,
                )
            )
            if self._current_summary is not None:
                self._current_summary.cancellations += 1
            self._record_event(
                "maker_quote_canceled",
                Severity.INFO,
                "maker quote canceled by refresh guard",
                {
                    "live_order_id": live_order_id,
                    "candidate_id": intent.candidate_id,
                    "cancel_reason": health["cancel_reason"],
                    "quote_guard": health,
                },
            )

    def _recover_pending_live_orders(self) -> None:
        """Re-register prior-session live orders with the fill reconciler on startup.

        Queries the persistent store for every live order whose last recorded
        execution-report status is SUBMITTED or PARTIAL, then registers each
        with self.fill_reconciler so the next _poll_live_fills() cycle will
        fetch current CLOB state and mirror fill deltas into the paper ledger.

        Since the paper ledger is in-memory only (empty on every process start),
        each re-registered order is treated as size_applied=0.  The first poll
        then applies ALL confirmed fills from scratch, correctly reconstructing
        the live position in the ledger.

        No-op conditions (returns immediately):
          - fill_reconciler is None  — live path not configured; paper mode unaffected
          - _effective_live_enabled is False — contradictory config; paper path wins
          - _startup_recovery_done is True  — already ran this session

        This method must be called before _poll_live_fills() each first cycle.
        """
        if self.fill_reconciler is None:
            return
        if not self._effective_live_enabled:
            return

        pending = self.store.load_pending_live_orders()
        if pending:
            self.logger.info(
                "startup recovery: re-registering pending live orders",
                extra={"payload": {"count": len(pending)}},
            )
        for intent, live_order_id in pending:
            self.fill_reconciler.register(live_order_id, intent)

    def _reconstruct_open_live_positions(self) -> None:
        """Replay prior-session filled live positions into the in-memory Ledger.

        On process restart the Ledger is empty.  _manage_open_positions reads
        paper_ledger.get_open_positions() to find positions eligible for exit
        evaluation.  Without reconstruction those positions are invisible and no
        live exits ever fire — the realized-PnL pipeline is permanently blocked.

        Algorithm:
          1. Query position_events for position_opened rows with no matching
             position_closed / position_expired / position_force_closed row.
          2. For each open position: place a synthetic limit order in the ledger
             (order_id = position_id) and immediately apply the recorded fill,
             restoring the PaperPositionRecord as if created this session.

        No-op conditions (returns immediately without touching the ledger):
          - fill_reconciler is None — live path not configured
          - _effective_live_enabled is False — paper mode active

        Idempotent: skips any position_id already present in position_records.
        Safe to call after _recover_pending_live_orders — the two methods use
        disjoint order states (SUBMITTED/PARTIAL vs FILLED).
        """
        if self.fill_reconciler is None:
            return
        if not self._effective_live_enabled:
            return

        rows = self.store.load_open_live_positions()
        if not rows:
            return

        self.logger.info(
            "startup reconstruction: replaying open live positions into ledger",
            extra={"payload": {"count": len(rows)}},
        )

        reconstructed = 0
        for row in rows:
            position_id = row["position_id"]
            if position_id in self.paper_ledger.position_records:
                continue  # already present — idempotency guard

            filled_size = float(row["filled_size"] or 0.0)
            avg_price   = float(row["avg_fill_price"] or row["limit_price"] or 0.0)
            side        = str(row["side"] or "BUY").upper()

            if filled_size <= 0 or avg_price <= 0:
                self.logger.warning(
                    "startup reconstruction: skipping position with missing fill data",
                    extra={"payload": {"position_id": position_id}},
                )
                continue

            # Synthetic order key: reuse position_id so apply_fill can locate it.
            placed = self.paper_ledger.place_limit_order(
                order_id=position_id,
                symbol=row["symbol"],
                market_slug=row["market_slug"],
                side=side,
                shares=filled_size,
                limit_price=avg_price,
                ts=str(row["ts"]),
                candidate_id=row["candidate_id"],
                position_id=position_id,
            )
            if not placed:
                self.logger.warning(
                    "startup reconstruction: place_limit_order failed",
                    extra={"payload": {"position_id": position_id}},
                )
                continue

            applied = self.paper_ledger.apply_fill(
                order_id=position_id,
                shares=filled_size,
                price=avg_price,
            )
            if not applied:
                self.logger.warning(
                    "startup reconstruction: apply_fill failed",
                    extra={"payload": {"position_id": position_id}},
                )
                continue

            reconstructed += 1

        self.logger.info(
            "startup reconstruction: complete",
            extra={"payload": {
                "reconstructed": reconstructed,
                "skipped": len(rows) - reconstructed,
            }},
        )

    def _abort_cycle_early(self) -> RunSummary:
        """Build and return a partial RunSummary when a data error ends the cycle early.

        Called when halt_on_data_errors=False and a qualifying data fetch failure
        has already been recorded via _record_event.  Mirrors the cleanup at the
        normal end of run_once() so runner state (summary, run_id, export path) is
        always consistent on return.
        """
        ended_ts = datetime.now(timezone.utc)
        assert self._current_summary is not None, "_abort_cycle_early called outside run_once"
        if self._current_auditor is not None:
            funnel_report = self._current_auditor.report()
            self._current_summary.record_qualification_funnel(funnel_report)
            self.store.save_qualification_funnel_report(funnel_report)
            self._current_auditor = None
        summary = self._current_summary.build(ended_ts=ended_ts)
        self.store.save_run_summary(summary)
        if self.config.monitoring.emit_console_summary:
            self.logger.info(
                "run summary (aborted — data error)",
                extra={"payload": summary.model_dump(mode="json")},
            )
        self._current_summary = None
        self._current_run_id = None
        self._invalid_orderbook_export_path = None
        return summary

    def _dispatch_order(self, intent: OrderIntent, book: object) -> ExecutionReport:
        """Route an order intent to the appropriate broker.

        Routes to live_broker when _order_mode is LIVE.  Raises RuntimeError
        if live_broker has not been configured.  Falls through to paper_broker
        for all other modes, covering default paper and dry-run cases.

        Args:
            intent: Order intent stamped with the resolved mode.
            book:   Order-book snapshot for paper fill simulation.
                    Not used by the live path.

        Returns:
            ExecutionReport from whichever broker handled the intent.

        Raises:
            RuntimeError: if _order_mode is LIVE but live_broker is None.
        """
        if self._order_mode == OrderMode.LIVE:
            if self.live_broker is None:
                raise RuntimeError(
                    "live_broker is not configured — assign runner.live_broker "
                    "before enabling live execution"
                )
            report = self.live_broker.submit_limit_order(intent)
            if (
                self.fill_reconciler is not None
                and report.status != OrderStatus.REJECTED
                and report.metadata.get("live_order_id")
            ):
                self.fill_reconciler.register(report.metadata["live_order_id"], intent)
            return report
        return self.paper_broker.submit_limit_order(intent, book)

    def _blocked_by_human_review_gate(
        self,
        decision: RiskDecision,
        candidate_id: str,
        metadata: dict,
    ) -> bool:
        """Block live-intended submission when risk requires human confirmation.

        Returns True (and records a rejection) when both conditions hold:
          1. _effective_live_enabled is True — order would go to the live client
          2. decision.human_review_required is True — risk flags it for review

        Returns False immediately in paper mode so paper execution is
        completely unaffected.  The caller must ``continue`` on True.
        """
        if not self._effective_live_enabled:
            return False
        if not decision.human_review_required:
            return False
        self._record_rejection(
            stage="human_review_gate",
            reason_code=RejectionReason.HUMAN_REVIEW_REQUIRED.value,
            candidate_id=candidate_id,
            metadata=metadata,
        )
        self.logger.warning(
            "live order blocked: human_review_required",
            extra={"payload": {"candidate_id": candidate_id}},
        )
        return True

    def _rebuild_runtime_components(self) -> None:
        self.risk = RiskManager(self.config.risk, self.config.opportunity, self.config.execution)
        self.feasibility = ExecutionFeasibilityEvaluator(self.config.opportunity)
        self.ranker = OpportunityRanker(self.config.opportunity)
        self.sizer = DepthCappedSizer(self.config.paper, self.config.opportunity)
        self.market_universe.reconfigure(self.config.market_data)
        self.clob.configure_negative_cache(
            no_orderbook_ttl_sec=self.config.market_data.no_orderbook_negative_cache_ttl_sec,
            invalid_token_retry_interval_sec=self.config.market_data.invalid_token_retry_interval_sec,
        )
        self.total_buffer = total_buffer_cents(
            {
                "fee_buffer_cents": self.config.opportunity.fee_buffer_cents,
                "slippage_buffer_cents": self.config.opportunity.slippage_buffer_cents,
            }
        )

    def _seed_market_universe_from_history(self) -> None:
        try:
            seeded_at = datetime.now(timezone.utc)
            productive_markets = self.store.load_recent_productive_market_slugs(
                limit=max(8, int(self.config.market_data.hot_market_count)),
            )
            recent_qualified_markets = self.store.load_recent_candidate_market_slugs(
                limit=max(16, int(self.config.market_data.warm_market_count)),
            )
            recent_near_miss_markets = self.store.load_recent_rejection_market_slugs(
                reason_codes=list(_QUALIFICATION_NEAR_MISS_REASONS),
                limit=max(16, int(self.config.market_data.warm_market_count)),
            )
            self.market_universe.seed_productive_markets(productive_markets, observed_at=seeded_at)
            self.market_universe.seed_recent_qualified_markets(recent_qualified_markets, observed_at=seeded_at)
            self.market_universe.seed_recent_near_miss_markets(recent_near_miss_markets, observed_at=seeded_at)
        except Exception:
            # History seeding is opportunistic only; runtime correctness must not depend on it.
            return

    def _reset_cycle_metrics(self) -> None:
        self.clob.reset_request_stats()
        self._neg_risk_family_audit_payloads = []
        self._neg_risk_family_parity_payloads = []
        self._cycle_metrics = {
            "discovery_duration_ms": 0.0,
            "refresh_duration_ms": 0.0,
            "candidate_eval_duration_ms": 0.0,
            "total_cycle_duration_ms": 0.0,
            "markets_in_hot_warm_cold": {
                MarketTier.HOT.value: 0,
                MarketTier.WARM.value: 0,
                MarketTier.COLD.value: 0,
            },
            "books_refreshed_by_tier": {
                MarketTier.HOT.value: 0,
                MarketTier.WARM.value: 0,
                MarketTier.COLD.value: 0,
                "backstop": 0,
            },
            "batch_request_counts_by_endpoint": Counter(),
            "websocket_update_count": 0,
            "recompute_trigger_count": 0,
            "recompute_changed_count": 0,
            "recompute_due_count": 0,
            "recompute_skip_count": 0,
            "markets_with_any_signal": 0,
            "pinned_productive_markets_evaluated": 0,
            "hot_tier_seed_count": 0,
            "hot_tier_zero_cycles": self._hot_tier_zero_cycles,
            "backstop_markets_refreshed": 0,
            "negative_cache_active_count": 0,
            "negative_cache_hits": 0,
            "negative_cache_expired_rechecks": 0,
            "productive_family_coverage_rate": 0.0,
            "families_considered": 0,
            "neg_risk_event_groups_available": 0,
            "neg_risk_audit_watchlist_matches": 0,
            "neg_risk_audit_forced_families": 0,
            "neg_risk_family_qualification_audit_count": 0,
            "families_recomputed_due_to_change": 0,
            "families_recomputed_due_to_due_interval": 0,
            "pinned_productive_families_evaluated": 0,
            "recent_near_miss_families_evaluated": 0,
            "family_backstop_recompute_count": 0,
            "avg_markets_per_family_recompute": 0.0,
            "neg_risk_family_coverage_rate": 0.0,
            "raw_candidates_by_family_per_cycle": {},
            "watched_families_audited": 0,
            "watched_families_fast_vs_broad_mismatch_count": 0,
            "watched_families_true_market_deterioration_count": 0,
            "watched_families_incomplete_input_count": 0,
            "watched_families_time_skew_count": 0,
            "watched_families_negative_cache_side_effect_count": 0,
            "watched_families_depth_failure_count": 0,
            "watched_families_net_profit_failure_count": 0,
            "productive_families_audited": 0,
            "productive_families_currently_viable_count": 0,
            "productive_families_marginal_count": 0,
            "productive_families_edge_vanished_count": 0,
            "productive_families_depth_failure_count": 0,
            "productive_families_spread_failure_count": 0,
            "selector_refresh_families_audited": 0,
            "selector_refresh_currently_viable_count": 0,
            "selector_refresh_near_miss_count": 0,
            "selector_refresh_marginal_count": 0,
            "selector_refresh_downgrade_count": 0,
            "selector_refresh_park_count": 0,
            "families_with_complete_leg_set": 0,
            "families_with_incomplete_leg_set": 0,
            "avg_missing_legs_per_family": 0.0,
            "families_blocked_by_negative_cache": 0,
            "avg_family_snapshot_time_skew_ms": 0.0,
            "families_failing_due_to_incomplete_inputs": 0,
            "raw_candidates_rejected_due_to_depth": 0,
            "raw_candidates_rejected_due_to_spread": 0,
            "raw_candidates_rejected_due_to_net_profit": 0,
            "raw_candidates_rejected_due_to_concentration": 0,
            "raw_candidates_per_cycle": 0,
            "raw_candidates_per_minute": 0.0,
            "qualified_candidates_per_cycle": 0,
            "qualified_candidates_per_minute": 0.0,
            "near_miss_candidates_per_minute": 0.0,
        }

    def _record_batch_request(self, endpoint: str, count: int = 1) -> None:
        batch_counts = self._cycle_metrics.get("batch_request_counts_by_endpoint")
        if isinstance(batch_counts, Counter):
            batch_counts[endpoint] += count

    def _discover_market_universe(
        self,
        *,
        cycle_started: datetime,
        market_slice: dict | None,
    ) -> tuple[list[dict], list[dict], list, dict[tuple[str, str], str]]:
        discovery_started = perf_counter()
        events: list[dict] = self.market_universe.current_events

        if self.market_universe.discovery_due(cycle_started) or not self.market_universe.current_pairs:
            if isinstance(market_slice, dict) and market_slice:
                markets = fetch_markets_with_slice(
                    self.config.market_data.gamma_host,
                    self.config.market_data.market_limit,
                    market_slice=market_slice,
                )
                self._record_batch_request("gamma_markets_slice")
            else:
                events = fetch_events(self.config.market_data.gamma_host, limit=100)
                self._record_batch_request("gamma_events")
                markets = flatten_event_markets(events, limit=self.config.market_data.market_limit)
                if len(markets) < self.config.market_data.market_limit:
                    markets = fetch_markets_from_events(
                        self.config.market_data.gamma_host,
                        self.config.market_data.market_limit,
                    )
                    self._record_batch_request("gamma_events_market_flatten")
                if self.config.market_data.discovery_use_simplified_markets:
                    simplified = self.clob.fetch_simplified_markets(limit=max(self.config.market_data.market_limit, 1000))
                    self._record_batch_request("clob_simplified_markets")
                    markets = _augment_markets_with_simplified(markets, _simplified_market_lookup(simplified))
            pairs = build_yes_no_pairs(markets)
            self.market_universe.update_discovery(
                events=events,
                markets=markets,
                pairs=pairs,
                discovered_at=cycle_started,
            )
            self._save_raw_snapshot("gamma", "markets", markets, cycle_started)
        else:
            markets = self.market_universe.current_markets
            pairs = self.market_universe.current_pairs

        token_map = _token_map_from_pairs(pairs)
        self._cycle_metrics["discovery_duration_ms"] = round((perf_counter() - discovery_started) * 1000.0, 3)
        self._cycle_metrics["markets_in_hot_warm_cold"] = self.market_universe.tier_counts(cycle_started)
        return events, markets, pairs, token_map

    def _collect_hot_tier_updates(self, refresh_plan: RefreshPlan, cycle_started: datetime) -> tuple[set[str], bool]:
        if not self.config.market_data.enable_hot_tier_websocket:
            return set(), False
        hot_entries = refresh_plan.hot or self.market_universe.all_hot_entries(cycle_started)
        token_ids = [token_id for entry in hot_entries for token_id in entry.token_ids]
        ws_result = self.hot_market_stream.collect_updates(
            token_ids=token_ids,
            timeout_sec=self.config.market_data.hot_tier_websocket_poll_timeout_sec,
            stale_sec=self.config.market_data.hot_tier_websocket_stale_sec,
        )
        updated_tokens = list(ws_result.get("updated_tokens") or [])
        changed_market_slugs = self.market_universe.record_websocket_updates(updated_tokens, cycle_started)
        self._cycle_metrics["websocket_update_count"] = int(ws_result.get("update_count") or 0)
        return changed_market_slugs, bool(ws_result.get("fallback_to_rest"))

    def _inventory_abs_for_pair(self, pair) -> float:
        yes_inv = abs(_current_inventory_shares(self.paper_ledger.position_records, pair.yes_token_id))
        no_inv = abs(_current_inventory_shares(self.paper_ledger.position_records, pair.no_token_id))
        return round(yes_inv + no_inv, 6)

    def _refresh_market_books(
        self,
        *,
        cycle_started: datetime,
        refresh_plan: RefreshPlan,
        book_cache: dict[str, object],
        forced_hot_market_slugs: set[str],
        force_rest_hot: bool,
    ) -> tuple[set[str], set[str]]:
        if "books_refreshed_by_tier" not in self._cycle_metrics:
            self._reset_cycle_metrics()
        refresh_started = perf_counter()
        changed_market_slugs: set[str] = set()
        due_market_slugs: set[str] = set(refresh_plan.force_scan_market_slugs)

        tier_entries = refresh_plan.entries_by_tier()
        if force_rest_hot:
            tier_entries[MarketTier.HOT.value] = list({
                entry.market_slug: entry
                for entry in [*tier_entries[MarketTier.HOT.value], *self.market_universe.all_hot_entries(cycle_started)]
            }.values())
        if forced_hot_market_slugs:
            forced_entries = [
                entry
                for entry in self.market_universe.all_hot_entries(cycle_started)
                if entry.market_slug in forced_hot_market_slugs
            ]
            existing = {entry.market_slug for entry in tier_entries[MarketTier.HOT.value]}
            tier_entries[MarketTier.HOT.value].extend(
                entry for entry in forced_entries if entry.market_slug not in existing
            )
            due_market_slugs.update(forced_hot_market_slugs)

        backstop_refreshed = 0
        for tier_name in (MarketTier.HOT.value, MarketTier.WARM.value, MarketTier.COLD.value, "backstop"):
            entries = tier_entries[tier_name]
            token_ids = [token_id for entry in entries for token_id in entry.token_ids]
            if not token_ids:
                continue

            books_by_token = self.clob.fetch_books_batch(token_ids)
            self._record_batch_request("clob_books")
            self._register_book_fetch(len(books_by_token))
            midpoints = self.clob.fetch_midpoints_batch(token_ids)
            self._record_batch_request("clob_midpoints")
            spreads = self.clob.fetch_spreads_batch(token_ids)
            self._record_batch_request("clob_spreads")
            prices = self.clob.fetch_prices_batch(token_ids)
            self._record_batch_request("clob_prices")

            _ = midpoints, spreads, prices  # refresh telemetry uses these calls even when scanners consume books.
            self._cycle_metrics["books_refreshed_by_tier"][tier_name] = len(books_by_token)

            for token_id, book in books_by_token.items():
                book_cache[token_id] = book
                self._save_raw_snapshot("clob", token_id, book.model_dump(mode="json"), cycle_started)

            for entry in entries:
                yes_book = books_by_token.get(entry.pair.yes_token_id)
                no_book = books_by_token.get(entry.pair.no_token_id)
                if yes_book is None or no_book is None:
                    continue
                snapshot = build_pair_snapshot(
                    yes_book=yes_book,
                    no_book=no_book,
                    inventory_abs=self._inventory_abs_for_pair(entry.pair),
                    source="rest",
                    ts=cycle_started,
                )
                recompute = self.market_universe.evaluate_recompute(
                    market_slug=entry.market_slug,
                    snapshot=snapshot,
                    stream_event=entry.market_slug in forced_hot_market_slugs,
                )
                if recompute.triggered:
                    changed_market_slugs.add(entry.market_slug)
                    self._cycle_metrics["recompute_trigger_count"] += 1
                    self._cycle_metrics["recompute_changed_count"] += 1
                else:
                    self._cycle_metrics["recompute_skip_count"] += 1
                    if entry.market_slug in due_market_slugs:
                        self._cycle_metrics["recompute_due_count"] += 1
                if tier_name == "backstop":
                    backstop_refreshed += 1

        self._cycle_metrics["refresh_duration_ms"] = round((perf_counter() - refresh_started) * 1000.0, 3)
        self._cycle_metrics["markets_in_hot_warm_cold"] = self.market_universe.tier_counts(cycle_started)
        self._cycle_metrics["backstop_markets_refreshed"] = backstop_refreshed
        return changed_market_slugs, due_market_slugs

    def _get_or_fetch_book(self, token_id: str, cycle_started: datetime, book_cache: dict[str, object]):
        book = book_cache.get(token_id)
        if book is not None:
            return book
        book = self.clob.get_book(token_id)
        book_cache[token_id] = book
        self._register_book_fetch()
        self._save_raw_snapshot("clob", token_id, book.model_dump(mode="json"), cycle_started)
        return book

    def _apply_ranked_opportunity_hook(self, candidates: list[RankedOpportunity]) -> list[RankedOpportunity]:
        if not candidates:
            return []
        sort_key = str(self._experiment_context.get("ranked_opportunity_sort") or "ranking_score")
        if sort_key == "expected_profit_usd":
            key_fn = lambda candidate: (float(candidate.expected_profit_usd), float(candidate.ranking_score))
        else:
            key_fn = lambda candidate: (float(candidate.ranking_score), float(candidate.expected_profit_usd))
        return sorted(candidates, key=key_fn, reverse=True)

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config.market_data.scan_interval_sec)

    def set_experiment_context(self, **context) -> None:
        self._experiment_context = {
            key: value
            for key, value in context.items()
            if value is not None
        }

    def run_once(self, experiment_context: dict[str, object] | None = None):
        cycle_started = datetime.now(timezone.utc)
        cycle_timer = perf_counter()
        self._run_sequence += 1
        self._current_run_id = str(uuid4())
        self._current_summary = RunSummaryBuilder(run_id=self._current_run_id, started_ts=cycle_started)
        self._reset_quote_guard_metrics()
        self._reset_cycle_metrics()
        self._current_auditor = QualificationAuditor(run_id=self._current_run_id)
        self._invalid_orderbook_export_path = self._prepare_invalid_orderbook_export_path(cycle_started)
        if experiment_context is not None:
            self.set_experiment_context(**experiment_context)
        if self._current_summary is not None:
            self._current_summary.metadata.update(self._build_experiment_metadata())
            self._current_summary.metadata["invalid_orderbooks_export_path"] = str(self._invalid_orderbook_export_path)
        self._check_live_dry_run_conflict()
        if not self._startup_recovery_done:
            self._recover_pending_live_orders()
            self._reconstruct_open_live_positions()
            self._startup_recovery_done = True
        self._poll_live_fills()
        self._refresh_active_maker_quotes()
        # ----------------------------------------------------------------
        # STAGE 1: RESEARCH — fetch market data and produce raw signals.
        # Scanners detect raw_signal candidates (RawCandidate) from
        # live orderbook and market data.  No execution decisions here.
        # ----------------------------------------------------------------
        constraints = _read_yaml(self.constraints_path)
        book_cache: dict[str, object] = {}

        try:
            market_slice = self._experiment_context.get("market_slice")
            events, markets, pairs, token_map = self._discover_market_universe(
                cycle_started=cycle_started,
                market_slice=market_slice if isinstance(market_slice, dict) else None,
            )
            self._current_summary.markets_scanned = len(pairs)
            self.logger.info(
                "loaded market pairs",
                extra={
                    "payload": {
                        "pairs": len(pairs),
                        "run_id": self._current_run_id,
                        "market_slice_name": market_slice.get("name") if isinstance(market_slice, dict) else None,
                    }
                },
            )
        except Exception as exc:
            self._record_event("market_fetch_failed", Severity.ERROR, "Failed to fetch or normalize markets", {"error": str(exc)})
            if self.config.risk.halt_on_data_errors:
                raise
            return self._abort_cycle_early()

        # ----------------------------------------------------------------
        # STAGE 2: DECISION — qualify, risk-check, size, and gate to live.
        # Each scan method transitions: RawCandidate → ExecutableCandidate
        # → (risk gate) → LiveTradableOpportunity → OrderIntent.
        # Open position exits (evaluate_exit) also happen in this stage.
        # ----------------------------------------------------------------
        candidate_eval_started = perf_counter()
        refresh_plan = self.market_universe.select_refresh_plan(cycle_started, cycle_index=self._run_sequence)
        self._cycle_metrics["hot_tier_seed_count"] = self.market_universe.seeded_hot_count(cycle_started)
        hot_count = int(self._cycle_metrics["markets_in_hot_warm_cold"][MarketTier.HOT.value])
        if hot_count == 0:
            self._hot_tier_zero_cycles += 1
        self._cycle_metrics["hot_tier_zero_cycles"] = self._hot_tier_zero_cycles
        websocket_market_slugs, force_rest_hot = self._collect_hot_tier_updates(refresh_plan, cycle_started)
        changed_market_slugs, due_market_slugs = self._refresh_market_books(
            cycle_started=cycle_started,
            refresh_plan=refresh_plan,
            book_cache=book_cache,
            forced_hot_market_slugs=websocket_market_slugs,
            force_rest_hot=force_rest_hot,
        )
        scan_market_slugs = set(changed_market_slugs) | set(due_market_slugs)
        self._cycle_metrics["pinned_productive_markets_evaluated"] = len(
            scan_market_slugs & self.market_universe.pinned_productive_market_slugs()
        )
        productive_event_slugs = self.market_universe.productive_event_slugs()
        if productive_event_slugs:
            evaluated_productive_events = {
                entry.event_slug
                for market_slug in scan_market_slugs
                for entry in [self.market_universe.active_entry_for_slug(market_slug)]
                if entry is not None and entry.event_slug in productive_event_slugs
            }
            self._cycle_metrics["productive_family_coverage_rate"] = round(
                len(evaluated_productive_events) / max(len(productive_event_slugs), 1),
                6,
            )
        single_market_pairs = [
            pair
            for pair in pairs
            if pair.market_slug in scan_market_slugs
        ]
        if self._active_single_market_strategies():
            self._run_single_market_scan(single_market_pairs, cycle_started, book_cache)
        if self._active_cross_market_strategies():
            self._run_cross_market_scan(constraints, token_map, cycle_started, book_cache, changed_market_slugs=scan_market_slugs)
        if self._active_neg_risk_strategies():
            self._run_neg_risk_scan(events, cycle_started, book_cache, changed_market_slugs=scan_market_slugs)
        _mm_targets = self._target_strategy_families()
        if _mm_targets is not None and "maker_rewarded_event_mm_v1" in _mm_targets:
            self._run_maker_mm_scan(cycle_started, book_cache)
        self._manage_open_positions(cycle_started, book_cache, force_reason="RUN_END_FLATTEN" if self.config.paper.flatten_on_run_end else None)
        self._cycle_metrics["candidate_eval_duration_ms"] = round((perf_counter() - candidate_eval_started) * 1000.0, 3)

        # ----------------------------------------------------------------
        # STAGE 3: EXECUTION — persist account state and run telemetry.
        # Broker dispatch occurred inside scan methods above.  This block
        # records the final snapshot and emits the run summary.
        # ----------------------------------------------------------------
        final_snapshot = self.paper_ledger.snapshot()
        self.store.save_account_snapshot(final_snapshot)
        self._current_summary.open_positions = final_snapshot.open_positions
        self._current_summary.realized_pnl = final_snapshot.realized_pnl
        self._current_summary.unrealized_pnl = final_snapshot.unrealized_pnl

        ended_ts = datetime.now(timezone.utc)
        if self._current_auditor is not None:
            funnel_report = self._current_auditor.report()
            self._current_summary.record_qualification_funnel(funnel_report)
            self.store.save_qualification_funnel_report(funnel_report)
            self.logger.info("qualification funnel", extra={"payload": funnel_report.model_dump(mode="json")})
            self._current_auditor = None
        self._cycle_metrics["raw_candidates_per_cycle"] = int(sum(self._current_summary.raw_candidate_family_counts.values()))
        self._cycle_metrics["markets_with_any_signal"] = self.market_universe.markets_with_any_signal()
        self._cycle_metrics["qualified_candidates_per_cycle"] = int(self._current_summary.candidates_generated)
        total_minutes = max((ended_ts - cycle_started).total_seconds() / 60.0, 1e-9)
        self._cycle_metrics["raw_candidates_per_minute"] = round(
            float(self._cycle_metrics["raw_candidates_per_cycle"]) / total_minutes,
            6,
        )
        self._cycle_metrics["qualified_candidates_per_minute"] = round(
            float(self._current_summary.candidates_generated) / total_minutes,
            6,
        )
        self._cycle_metrics["near_miss_candidates_per_minute"] = round(
            float(self._current_summary.near_miss_candidates) / total_minutes,
            6,
        )
        self._cycle_metrics["total_cycle_duration_ms"] = round((perf_counter() - cycle_timer) * 1000.0, 3)
        self._cycle_metrics.update(self.clob.request_stats_snapshot())
        batch_counts = self._cycle_metrics.get("batch_request_counts_by_endpoint")
        self._current_summary.metadata["scan_metrics"] = {
            **self._cycle_metrics,
            "batch_request_counts_by_endpoint": dict(batch_counts) if isinstance(batch_counts, Counter) else batch_counts,
        }
        self._current_summary.metadata["maker_quote_guard"] = self._quote_guard_summary()
        summary = self._current_summary.build(ended_ts=ended_ts)
        self.store.save_run_summary(summary)
        if self.config.monitoring.emit_console_summary:
            self.logger.info("run summary", extra={"payload": summary.model_dump(mode="json")})
        self._current_summary = None
        self._current_run_id = None
        self._invalid_orderbook_export_path = None
        return summary

    def force_flatten_open_positions(self, reason: str = "MANUAL_FORCE_FLATTEN"):
        started_ts = datetime.now(timezone.utc)
        self._current_run_id = self._current_run_id or str(uuid4())
        self._current_summary = self._current_summary or RunSummaryBuilder(run_id=self._current_run_id, started_ts=started_ts)
        self._invalid_orderbook_export_path = self._invalid_orderbook_export_path or self._prepare_invalid_orderbook_export_path(started_ts)
        self._manage_open_positions(started_ts, book_cache={}, force_reason=reason)
        snapshot = self.paper_ledger.snapshot()
        self.store.save_account_snapshot(snapshot)
        self._current_summary.open_positions = snapshot.open_positions
        self._current_summary.realized_pnl = snapshot.realized_pnl
        self._current_summary.unrealized_pnl = snapshot.unrealized_pnl
        summary = self._current_summary.build(ended_ts=datetime.now(timezone.utc))
        self.store.save_run_summary(summary)
        self._current_summary = None
        self._current_run_id = None
        self._current_auditor = None
        self._invalid_orderbook_export_path = None
        return summary

    def _run_single_market_scan(self, pairs: list, cycle_started: datetime, book_cache: dict[str, object]) -> None:
        active_strategies = self._active_single_market_strategies()
        if not active_strategies:
            return
        for pair in pairs:
            for strategy in active_strategies:
                self._record_strategy_family_market_considered(strategy.strategy_family.value, pair.market_slug)
            yes_skip_key = (pair.market_slug, "YES", "BUY")
            no_skip_key = (pair.market_slug, "NO", "BUY")
            if self._should_skip_market_leg(*yes_skip_key) or self._should_skip_market_leg(*no_skip_key):
                if self._current_summary is not None:
                    self._current_summary.books_skipped_due_to_recent_empty_asks += 1
                self.logger.info(
                    "skipping market due to repeated empty asks",
                    extra={
                        "payload": {
                            "market_slug": pair.market_slug,
                            "skip_keys": [
                                key
                                for key in (yes_skip_key, no_skip_key)
                                if self._should_skip_market_leg(*key)
                            ],
                            "run_id": self._current_run_id,
                        }
                    },
                )
                continue
            try:
                try:
                    yes_book = self._get_or_fetch_book(pair.yes_token_id, cycle_started, book_cache)
                    for strategy in active_strategies:
                        self._record_strategy_family_book_fetch(strategy.strategy_family.value)
                except Exception as exc:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=build_fetch_failure_validation(pair.yes_token_id, exc),
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.yes_token_id,
                            "side": "YES",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue

                try:
                    no_book = self._get_or_fetch_book(pair.no_token_id, cycle_started, book_cache)
                    for strategy in active_strategies:
                        self._record_strategy_family_book_fetch(strategy.strategy_family.value)
                except Exception as exc:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=build_fetch_failure_validation(pair.no_token_id, exc),
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.no_token_id,
                            "side": "NO",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue
                yes_validation = validate_orderbook(yes_book, required_action="BUY")
                self._record_book_validation_result(yes_validation)
                for strategy in active_strategies:
                    self._record_strategy_family_book_validation_result(strategy.strategy_family.value, yes_validation)
                if not yes_validation.passed:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=yes_validation,
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.yes_token_id,
                            "side": "YES",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue
                self._update_liquidity_skip_state(pair.market_slug, "YES", "BUY", "PASSED", "candidate_filter")

                no_validation = validate_orderbook(no_book, required_action="BUY")
                self._record_book_validation_result(no_validation)
                for strategy in active_strategies:
                    self._record_strategy_family_book_validation_result(strategy.strategy_family.value, no_validation)
                if not no_validation.passed:
                    self._record_invalid_orderbook(
                        stage="candidate_filter",
                        validation=no_validation,
                        metadata={
                            "market_slug": pair.market_slug,
                            "token_id": pair.no_token_id,
                            "side": "NO",
                            "strategy_family": self.single_market_strategy.strategy_family.value,
                        },
                    )
                    continue
                self._update_liquidity_skip_state(pair.market_slug, "NO", "BUY", "PASSED", "candidate_filter")

                for strategy in active_strategies:
                    raw_candidate, pre_candidate_audit = strategy.detect_with_audit(
                        pair,
                        yes_book,
                        no_book,
                        max_notional=self.config.paper.max_notional_per_arb,
                        total_buffer_cents=self.total_buffer,
                    )
                    if raw_candidate is None:
                        if pre_candidate_audit is not None:
                            self._record_pre_candidate_pricing_failure(pre_candidate_audit)
                        continue
                    raw_candidate = self._decorate_raw_candidate(raw_candidate)
                    self._record_raw_candidate(raw_candidate)

                    account_snapshot = self.paper_ledger.snapshot(ts=cycle_started)
                    candidate = self._qualify_and_rank_candidate(
                        raw_candidate=raw_candidate,
                        books_by_token={pair.yes_token_id: yes_book, pair.no_token_id: no_book},
                        account_snapshot=account_snapshot,
                    )
                    if candidate is None:
                        continue
                    self._record_qualified_candidate(candidate)

                    self._current_summary.candidates_generated += 1
                    self._current_summary.market_counts[pair.market_slug] += 1
                    self._current_summary.opportunity_type_counts[candidate.kind] += 1
                    self.store.save_candidate(candidate)
                    self._ab_sidecar.observe(raw_candidate, candidate)  # A+B sidecar — observation only
                    self.opportunity_store.save(raw_candidate.to_legacy_opportunity())
                    try:
                        decision = self.risk.evaluate(candidate, account_snapshot)
                    except Exception as exc:
                        self._record_rejection(
                            stage="risk",
                            reason_code=RejectionReason.RISK_ENGINE_ERROR.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "market_slug": pair.market_slug,
                                "error": str(exc),
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                        continue

                    self.store.save_risk_decision(decision)

                    if decision.status.name in {"BLOCKED", "HALTED"}:
                        self._current_summary.risk_rejected += 1
                        for reason_code in decision.reason_codes:
                            self._record_rejection(
                                stage="risk",
                                reason_code=reason_code,
                                candidate_id=candidate.candidate_id,
                                metadata={
                                    "market_slug": pair.market_slug,
                                    "strategy_family": candidate.strategy_family.value,
                                },
                            )
                        self.logger.warning(
                            "candidate blocked by risk",
                            extra={"payload": {"candidate_id": candidate.candidate_id, "reasons": decision.reason_codes}},
                        )
                        continue

                    self._current_summary.risk_accepted += 1
                    if self._blocked_by_human_review_gate(
                        decision,
                        candidate.candidate_id,
                        {"market_slug": pair.market_slug, "strategy_family": candidate.strategy_family.value},
                    ):
                        continue
                    try:
                        intents, reports = self._submit_pair_orders(
                            candidate,
                            pair.market_slug,
                            pair.yes_token_id,
                            pair.no_token_id,
                            yes_book,
                            no_book,
                            candidate.sizing_hint_shares,
                        )
                    except Exception as exc:
                        self._record_rejection(
                            stage="execution",
                            reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "market_slug": pair.market_slug,
                                "error": str(exc),
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                        continue

                    self._current_summary.paper_orders_created += len(intents)
                    for intent in intents:
                        self.store.save_order_intent(intent)
                    for report in reports:
                        self.store.save_execution_report(report)
                        self._record_report_stats(report)

                    for report in reports:
                        if report.filled_size > 0 and report.position_id:
                            position = self.paper_ledger.position_records.get(report.position_id)
                            if position is not None:
                                self.store.save_position_event(
                                    position_id=position.position_id,
                                    candidate_id=position.candidate_id,
                                    event_type="position_opened",
                                    symbol=position.symbol,
                                    market_slug=position.market_slug,
                                    state=position.state.value,
                                    reason_code=None,
                                    payload={
                                        "filled_size": report.filled_size,
                                        "avg_fill_price": report.avg_fill_price,
                                        "intent_id": report.intent_id,
                                    },
                                    ts=report.ts,
                                )

                    snapshot = self.paper_ledger.snapshot()
                    self.store.save_account_snapshot(snapshot)

                    if any(report.status != OrderStatus.FILLED for report in reports):
                        self._record_event(
                            "paper_pair_incomplete_fill",
                            Severity.WARNING,
                            f"Incomplete pair fill for {pair.market_slug}",
                            {
                                "candidate_id": candidate.candidate_id,
                                "reports": [report.model_dump(mode="json") for report in reports],
                            },
                        )
            except Exception as exc:
                self._record_event(
                    "single_market_scan_failed",
                    Severity.WARNING,
                    f"Failed to scan market {pair.market_slug}",
                    {"market_slug": pair.market_slug, "error": str(exc)},
                )

    def _submit_pair_orders(
        self,
        candidate: RankedOpportunity,
        market_slug: str,
        yes_token_id: str,
        no_token_id: str,
        yes_book,
        no_book,
        shares: float,
    ) -> tuple[list[OrderIntent], list]:
        if shares <= 0:
            raise ValueError(f"Invalid paper order size {shares} for {market_slug}")

        ts = datetime.now(timezone.utc)
        yes_limit = _limit_price_for_target_shares(getattr(yes_book, "asks", []), shares)
        no_limit = _limit_price_for_target_shares(getattr(no_book, "asks", []), shares)
        if yes_limit is None or no_limit is None:
            raise ValueError(f"Insufficient depth to simulate pair fill for {market_slug}")

        yes_intent = OrderIntent(
            intent_id=str(uuid4()),
            candidate_id=candidate.candidate_id,
            mode=self._order_mode,
            market_slug=market_slug,
            token_id=yes_token_id,
            position_id=str(uuid4()),
            side="BUY",
            order_type=OrderType.LIMIT,
            size=shares,
            limit_price=yes_limit,
            max_notional_usd=shares * yes_limit,
            ts=ts,
        )
        no_intent = OrderIntent(
            intent_id=str(uuid4()),
            candidate_id=candidate.candidate_id,
            mode=self._order_mode,
            market_slug=market_slug,
            token_id=no_token_id,
            position_id=str(uuid4()),
            side="BUY",
            order_type=OrderType.LIMIT,
            size=shares,
            limit_price=no_limit,
            max_notional_usd=shares * no_limit,
            ts=ts,
        )

        yes_report = self._dispatch_order(yes_intent, yes_book)
        no_report = self._dispatch_order(no_intent, no_book)
        return [yes_intent, no_intent], [yes_report, no_report]

    def _submit_candidate_orders(
        self,
        candidate: RankedOpportunity,
        books_by_token: dict[str, object],
        shares: float,
    ) -> tuple[list[OrderIntent], list]:
        if shares <= 0:
            raise ValueError(f"Invalid paper order size {shares} for candidate {candidate.candidate_id}")

        ts = datetime.now(timezone.utc)
        planned_orders: list[tuple[OrderIntent, object]] = []
        reports: list = []

        for leg in candidate.legs:
            if leg.action.upper() != "BUY":
                raise ValueError(f"Unsupported paper action {leg.action} for candidate {candidate.candidate_id}")
            book = books_by_token.get(leg.token_id)
            if book is None:
                raise ValueError(f"Missing book for token {leg.token_id}")

            is_maker = (leg.metadata or {}).get("maker_first", False)
            if is_maker:
                bid_levels = getattr(book, "bids", [])
                if not bid_levels:
                    raise ValueError(f"No bid levels for maker leg {leg.token_id}")
                # Set limit_price just above the best bid so the paper broker's
                # price filter (px <= limit_price) passes all bid levels and fills
                # at bid-side prices, matching maker execution economics.
                limit_price = float(bid_levels[0].price) + 1e-9
                sim_book = _BidFillBook(book)
                inventory_shares = _current_inventory_shares(self.paper_ledger.position_records, leg.token_id)
                skew = _apply_inventory_skew("BUY", limit_price, inventory_shares, self.config.execution)
                if bool(skew["suppressed"]):
                    self._record_quote_cancel_metric("inventory_suppressed")
                    self._record_event(
                        "maker_quote_suppressed",
                        Severity.INFO,
                        "maker quote suppressed by inventory skew",
                        {
                            "candidate_id": candidate.candidate_id,
                            "market_slug": leg.market_slug,
                            "token_id": leg.token_id,
                            "inventory_shares": inventory_shares,
                            "skew": skew,
                        },
                    )
                    raise ValueError(f"Maker quote suppressed by inventory skew for token {leg.token_id}")
                limit_price = float(skew["adjusted_quote_price"])
                health = _evaluate_quote_health(
                    side="BUY",
                    quote_price=limit_price,
                    size=shares,
                    book=book,
                    inventory_shares=inventory_shares,
                    posted_ts=ts,
                    now_ts=ts,
                    opportunity_config=self.config.opportunity,
                    execution_config=self.config.execution,
                    qualified_gross_edge_cents=candidate.gross_edge_cents,
                    qualified_fee_impact_cents=candidate.fee_estimate_cents,
                    qualified_slippage_cents=candidate.slippage_estimate_cents,
                    qualified_net_edge_cents=candidate.net_edge_cents,
                )
                if bool(health["should_cancel"]):
                    self._record_quote_post_metric(float(health["expected_net_edge"]))
                    self._record_quote_cancel_metric(str(health["cancel_reason"]))
                    self._record_event(
                        "maker_quote_blocked",
                        Severity.INFO,
                        "maker quote blocked before post",
                        {
                            "candidate_id": candidate.candidate_id,
                            "market_slug": leg.market_slug,
                            "token_id": leg.token_id,
                            "quote_guard": health,
                        },
                    )
                    raise ValueError(f"Maker quote blocked before post for token {leg.token_id}: {health['cancel_reason']}")
            else:
                limit_price = _limit_price_for_target_shares(getattr(book, "asks", []), shares)
                sim_book = book
                skew = None
                health = None

            if limit_price is None:
                raise ValueError(f"Insufficient depth to simulate fill for token {leg.token_id}")

            intent = OrderIntent(
                intent_id=str(uuid4()),
                candidate_id=candidate.candidate_id,
                mode=self._order_mode,
                market_slug=leg.market_slug,
                token_id=leg.token_id,
                position_id=str(uuid4()),
                side="BUY",
                order_type=OrderType.LIMIT,
                size=shares,
                limit_price=limit_price,
                max_notional_usd=shares * limit_price,
                ts=ts,
                metadata={
                    "maker_quote": is_maker,
                    "quote_guard": {
                        **(health or {}),
                        "qualified_gross_edge_cents": candidate.gross_edge_cents,
                        "qualified_fee_impact_cents": candidate.fee_estimate_cents,
                        "qualified_slippage_cents": candidate.slippage_estimate_cents,
                        "qualified_net_edge_cents": candidate.net_edge_cents,
                        "skew": skew,
                    } if is_maker else {},
                },
            )
            if is_maker and health is not None:
                self._record_quote_post_metric(float(health["expected_net_edge"]))
            planned_orders.append((intent, sim_book))

        intents: list[OrderIntent] = []
        for intent, sim_book in planned_orders:
            report = self._dispatch_order(intent, sim_book)
            intents.append(intent)
            reports.append(report)
            if intent.mode == OrderMode.PAPER and report.filled_size > 1e-9:
                self._record_quote_fill_metrics(intent, report.avg_fill_price, report.ts)
        return intents, reports

    def _run_cross_market_scan(
        self,
        constraints: dict,
        token_map: dict[tuple[str, str], str],
        cycle_started: datetime,
        book_cache: dict[str, object],
        *,
        changed_market_slugs: set[str] | None = None,
    ) -> None:
        active_strategies = self._active_cross_market_strategies()
        if not active_strategies:
            return
        for rule in constraints.get("cross_market", []):
            if rule.get("relation") != "leq":
                continue

            lhs_slug = rule["lhs"]["market_slug"]
            rhs_slug = rule["rhs"]["market_slug"]
            if changed_market_slugs is not None and lhs_slug not in changed_market_slugs and rhs_slug not in changed_market_slugs:
                continue
            for strategy in active_strategies:
                self._record_strategy_family_market_considered(strategy.strategy_family.value, lhs_slug)
                self._record_strategy_family_market_considered(strategy.strategy_family.value, rhs_slug)
            lhs_side = rule["lhs"].get("side", "YES").upper()
            rhs_side = rule["rhs"].get("side", "YES").upper()
            lhs_token = token_map.get((lhs_slug, lhs_side))
            rhs_token = token_map.get((rhs_slug, rhs_side))
            lhs_exec_side = _complement_side(lhs_side)
            rhs_exec_side = rhs_side
            lhs_exec_token = token_map.get((lhs_slug, lhs_exec_side))
            rhs_exec_token = token_map.get((rhs_slug, rhs_exec_side))
            if not lhs_token or not rhs_token or not lhs_exec_token or not rhs_exec_token:
                for strategy in active_strategies:
                    self._record_rejection(
                        stage="candidate_filter",
                        reason_code="MISSING_CONSTRAINT_TOKEN",
                        candidate_id=None,
                        metadata={
                            "strategy_family": strategy.strategy_family.value,
                            "failure_stage": "pre_candidate_relation",
                            "constraint_name": rule.get("name"),
                            "lhs_market_slug": lhs_slug,
                            "rhs_market_slug": rhs_slug,
                            "lhs_relation_side": lhs_side,
                            "rhs_relation_side": rhs_side,
                            "lhs_execution_side": lhs_exec_side,
                            "rhs_execution_side": rhs_exec_side,
                        },
                    )
                continue

            try:
                relation_lhs_book = self._get_or_fetch_book(lhs_token, cycle_started, book_cache)
                for strategy in active_strategies:
                    self._record_strategy_family_book_fetch(strategy.strategy_family.value)
                relation_rhs_book = self._get_or_fetch_book(rhs_token, cycle_started, book_cache)
                for strategy in active_strategies:
                    self._record_strategy_family_book_fetch(strategy.strategy_family.value)
                lhs_exec_book = self._get_or_fetch_book(lhs_exec_token, cycle_started, book_cache)
                for strategy in active_strategies:
                    self._record_strategy_family_book_fetch(strategy.strategy_family.value)
                rhs_exec_book = self._get_or_fetch_book(rhs_exec_token, cycle_started, book_cache)
                for strategy in active_strategies:
                    self._record_strategy_family_book_fetch(strategy.strategy_family.value)
                for book in (relation_lhs_book, relation_rhs_book, lhs_exec_book, rhs_exec_book):
                    validation = validate_orderbook(book, required_action="BUY")
                    self._record_book_validation_result(validation)
                    for strategy in active_strategies:
                        self._record_strategy_family_book_validation_result(strategy.strategy_family.value, validation)

                for strategy in active_strategies:
                    raw_candidate, relation_audit = strategy.detect_with_audit(
                        rule,
                        relation_lhs_book,
                        relation_rhs_book,
                        lhs_exec_token,
                        lhs_exec_side,
                        lhs_exec_book,
                        rhs_exec_token,
                        rhs_exec_side,
                        rhs_exec_book,
                        self.config.paper.max_notional_per_arb,
                        self.total_buffer,
                    )
                    if raw_candidate is None:
                        if relation_audit is not None:
                            self._record_cross_market_pre_candidate_failure(relation_audit)
                        continue

                    raw_candidate = self._decorate_raw_candidate(raw_candidate)
                    self._record_raw_candidate(raw_candidate)

                    account_snapshot = self.paper_ledger.snapshot(ts=cycle_started)
                    candidate = self._qualify_and_rank_candidate(
                        raw_candidate=raw_candidate,
                        books_by_token={
                            lhs_exec_token: lhs_exec_book,
                            rhs_exec_token: rhs_exec_book,
                        },
                        account_snapshot=account_snapshot,
                    )
                    if candidate is None:
                        continue

                    self._record_qualified_candidate(candidate)
                    self._current_summary.candidates_generated += 1
                    for market_slug in candidate.market_slugs:
                        self._current_summary.market_counts[market_slug] += 1
                    self._current_summary.opportunity_type_counts[candidate.kind] += 1
                    self.store.save_candidate(candidate)
                    self._ab_sidecar.observe(raw_candidate, candidate)  # A+B sidecar — observation only
                    self.opportunity_store.save(raw_candidate.to_legacy_opportunity())
                    try:
                        decision = self.risk.evaluate(candidate, account_snapshot)
                    except Exception as exc:
                        self._record_rejection(
                            stage="risk",
                            reason_code=RejectionReason.RISK_ENGINE_ERROR.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "constraint_name": rule["name"],
                                "error": str(exc),
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                        continue

                    self.store.save_risk_decision(decision)

                    if decision.status.name in {"BLOCKED", "HALTED"}:
                        self._current_summary.risk_rejected += 1
                        for reason_code in decision.reason_codes:
                            self._record_rejection(
                                stage="risk",
                                reason_code=reason_code,
                                candidate_id=candidate.candidate_id,
                                metadata={
                                    "constraint_name": rule["name"],
                                    "strategy_family": candidate.strategy_family.value,
                                },
                            )
                        self.logger.warning(
                            "candidate blocked by risk",
                            extra={"payload": {"candidate_id": candidate.candidate_id, "reasons": decision.reason_codes}},
                        )
                        continue

                    self._current_summary.risk_accepted += 1
                    if self._blocked_by_human_review_gate(
                        decision,
                        candidate.candidate_id,
                        {"constraint_name": rule["name"], "strategy_family": candidate.strategy_family.value},
                    ):
                        continue
                    try:
                        intents, reports = self._submit_candidate_orders(
                            candidate,
                            books_by_token={
                                lhs_exec_token: lhs_exec_book,
                                rhs_exec_token: rhs_exec_book,
                            },
                            shares=candidate.sizing_hint_shares,
                        )
                    except Exception as exc:
                        self._record_rejection(
                            stage="execution",
                            reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "constraint_name": rule["name"],
                                "error": str(exc),
                                "strategy_family": candidate.strategy_family.value,
                            },
                        )
                        continue

                    self._current_summary.paper_orders_created += len(intents)
                    for intent in intents:
                        self.store.save_order_intent(intent)
                    for report in reports:
                        self.store.save_execution_report(report)
                        self._record_report_stats(report)

                    for report in reports:
                        if report.filled_size > 0 and report.position_id:
                            position = self.paper_ledger.position_records.get(report.position_id)
                            if position is not None:
                                self.store.save_position_event(
                                    position_id=position.position_id,
                                    candidate_id=position.candidate_id,
                                    event_type="position_opened",
                                    symbol=position.symbol,
                                    market_slug=position.market_slug,
                                    state=position.state.value,
                                    reason_code=None,
                                    payload={
                                        "filled_size": report.filled_size,
                                        "avg_fill_price": report.avg_fill_price,
                                        "intent_id": report.intent_id,
                                    },
                                    ts=report.ts,
                                )

                    snapshot = self.paper_ledger.snapshot()
                    self.store.save_account_snapshot(snapshot)

                    if any(report.status != OrderStatus.FILLED for report in reports):
                        self._record_event(
                            "paper_candidate_incomplete_fill",
                            Severity.WARNING,
                            f"Incomplete candidate fill for {rule['name']}",
                            {
                                "candidate_id": candidate.candidate_id,
                                "constraint_name": rule["name"],
                                "reports": [report.model_dump(mode="json") for report in reports],
                            },
                        )
            except Exception as exc:
                self._record_event(
                    "cross_market_scan_failed",
                    Severity.WARNING,
                    f"Failed to scan relation {rule['name']}",
                    {"rule": rule["name"], "error": str(exc)},
                )

    def _manage_open_positions(self, cycle_started: datetime, book_cache: dict[str, object], force_reason: str | None = None) -> None:
        open_positions = list(self.paper_ledger.get_open_positions())
        for position in open_positions:
            try:
                book = book_cache.get(position.symbol)
                if book is None:
                    try:
                        book = self.clob.get_book(position.symbol)
                    except Exception as exc:
                        self._record_invalid_orderbook(
                            stage="markout",
                            validation=build_fetch_failure_validation(position.symbol, exc),
                            metadata={
                                "candidate_id": position.candidate_id,
                                "position_id": position.position_id,
                                "market_slug": position.market_slug,
                                "token_id": position.symbol,
                                "side": "SELL",
                            },
                        )
                        continue
                    book_cache[position.symbol] = book
                    self._save_raw_snapshot("clob", position.symbol, book.model_dump(mode="json"), cycle_started)

                markout_validation = validate_orderbook(book, required_action="SELL")
                if not markout_validation.passed:
                    self._record_invalid_orderbook(
                        stage="markout",
                        validation=markout_validation,
                        metadata={
                            "candidate_id": position.candidate_id,
                            "position_id": position.position_id,
                            "market_slug": position.market_slug,
                            "token_id": position.symbol,
                            "side": "SELL",
                        },
                    )
                    continue

                best_bid = float(book.bids[0].price) if getattr(book, "bids", []) else None
                best_ask = float(book.asks[0].price) if getattr(book, "asks", []) else None

                mark = self.paper_ledger.mark_position(
                    position_id=position.position_id,
                    mark_price=best_bid,
                    ts=datetime.now(timezone.utc),
                    source_bid=best_bid,
                    source_ask=best_ask,
                )
                if mark is None:
                    continue

                self.store.save_position_mark(mark)
                self.store.save_position_event(
                    position_id=mark.position_id,
                    candidate_id=mark.candidate_id,
                    event_type="position_marked",
                    symbol=mark.symbol,
                    market_slug=mark.market_slug,
                    state=mark.state.value,
                    reason_code=None,
                    payload=mark.model_dump(mode="json"),
                    ts=mark.ts,
                )

                # Basket-level EDGE_DECAY cascade: if any sibling leg of this position has
                # already been closed via EDGE_DECAY, force-exit this leg with the same reason.
                effective_force_reason = force_reason
                if effective_force_reason is None and _has_basket_ed_exit(
                    self.paper_ledger.position_records,
                    position.candidate_id,
                    position.position_id,
                ):
                    effective_force_reason = "EDGE_DECAY"
                elif effective_force_reason is None and _has_basket_sl_exit(
                    self.paper_ledger.position_records,
                    position.candidate_id,
                    position.position_id,
                ):
                    effective_force_reason = "STOP_LOSS"
                elif effective_force_reason is None and _has_basket_idle_release(
                    self.paper_ledger.position_records,
                    position.candidate_id,
                    position.position_id,
                ):
                    effective_force_reason = "IDLE_HOLD_RELEASE"
                elif effective_force_reason is None and _basket_idle_release_eligible(
                    self.paper_ledger.position_records,
                    position.candidate_id,
                    self.config.paper,
                    mark.ts,
                ):
                    effective_force_reason = "IDLE_HOLD_RELEASE"

                exit_signal = evaluate_exit(mark, self.config.paper, force_reason=effective_force_reason)
                if exit_signal is None:
                    continue

                if (
                    effective_force_reason is None
                    and exit_signal.reason_code == "EDGE_DECAY"
                    and not exit_signal.force_exit
                ):
                    record = self.paper_ledger.position_records.get(position.position_id)
                    if record is not None:
                        record.edge_decay_candidate_count += 1

                # Basket-level dominance gate: suppress natural EDGE_DECAY exits where the
                # trigger leg is not the dominant loss leg and basket deterioration thresholds
                # are not met.  Cascade exits (force_exit=True) bypass this gate unconditionally
                # so that a confirmed trigger always exits its full basket.
                if exit_signal.reason_code == "EDGE_DECAY" and not exit_signal.force_exit:
                    if not _basket_exit_confirmed(
                        self.paper_ledger.position_records,
                        position.candidate_id,
                        position.position_id,
                        self.config.paper,
                    ):
                        self.store.save_position_event(
                            position_id=exit_signal.position_id,
                            candidate_id=exit_signal.candidate_id,
                            event_type="exit_suppressed",
                            symbol=exit_signal.symbol,
                            market_slug=exit_signal.market_slug,
                            state=position.state.value,
                            reason_code="MINOR_LEG_GATE",
                            payload={
                                **exit_signal.model_dump(mode="json"),
                                "gate_reason": "basket_dominance_threshold_not_met",
                            },
                            ts=exit_signal.ts,
                        )
                        continue

                self.store.save_position_event(
                    position_id=exit_signal.position_id,
                    candidate_id=exit_signal.candidate_id,
                    event_type="exit_signal_generated",
                    symbol=exit_signal.symbol,
                    market_slug=exit_signal.market_slug,
                    state=position.state.value,
                    reason_code=exit_signal.reason_code,
                    payload=exit_signal.model_dump(mode="json"),
                    ts=exit_signal.ts,
                )

                limit_price = _limit_price_for_target_shares(getattr(book, "bids", []), position.remaining_shares)
                if limit_price is None:
                    self._record_rejection(
                        stage="execution",
                        reason_code=RejectionReason.INSUFFICIENT_DEPTH.value,
                        candidate_id=position.candidate_id,
                        metadata={"position_id": position.position_id, "market_slug": position.market_slug},
                    )
                    continue

                intent = OrderIntent(
                    intent_id=str(uuid4()),
                    candidate_id=position.candidate_id or "unknown",
                    mode=self._order_mode,
                    market_slug=position.market_slug,
                    token_id=position.symbol,
                    position_id=position.position_id,
                    side="SELL",
                    order_type=OrderType.LIMIT,
                    size=position.remaining_shares,
                    limit_price=limit_price,
                    max_notional_usd=position.remaining_shares * limit_price,
                    ts=exit_signal.ts,
                )
                report = self._dispatch_order(intent, book)
                self.store.save_order_intent(intent)
                self.store.save_execution_report(report)
                self._current_summary.paper_orders_created += 1
                self._record_report_stats(report)

                updated_position = self.paper_ledger.position_records.get(position.position_id)
                if report.filled_size <= 0:
                    self._record_rejection(
                        stage="execution",
                        reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                        candidate_id=position.candidate_id,
                        metadata={"position_id": position.position_id, "status": report.status.value},
                    )
                    continue

                if updated_position is not None and updated_position.is_open:
                    self.store.save_position_event(
                        position_id=updated_position.position_id,
                        candidate_id=updated_position.candidate_id,
                        event_type="position_reduced",
                        symbol=updated_position.symbol,
                        market_slug=updated_position.market_slug,
                        state=updated_position.state.value,
                        reason_code=exit_signal.reason_code,
                        payload=report.model_dump(mode="json"),
                        ts=report.ts,
                    )
                    continue

                final_state = self._final_position_state(exit_signal.reason_code)
                self.paper_ledger.set_position_state(position.position_id, final_state, reason_code=exit_signal.reason_code, ts=report.ts)
                trade_summary = self.paper_ledger.build_trade_summary(position.position_id)
                if trade_summary is not None:
                    audit = _build_basket_audit(
                        self.paper_ledger.position_records,
                        position.candidate_id,
                        report.ts,
                    )
                    trade_summary.metadata.update(audit)
                    self.store.save_trade_summary(trade_summary)
                    if (
                        exit_signal.reason_code == "EDGE_DECAY"
                        and float(trade_summary.realized_pnl_usd) > 0.0
                    ):
                        self.market_universe.record_productive_outcome(trade_summary.market_slug, report.ts)
                    if self._current_summary is not None:
                        self._current_summary.closed_positions += 1
                    self.store.save_position_event(
                        position_id=trade_summary.position_id,
                        candidate_id=trade_summary.candidate_id,
                        event_type=self._close_event_type(final_state),
                        symbol=trade_summary.symbol,
                        market_slug=trade_summary.market_slug,
                        state=trade_summary.state.value,
                        reason_code=exit_signal.reason_code,
                        payload=trade_summary.model_dump(mode="json"),
                        ts=trade_summary.closed_ts,
                    )
            except Exception as exc:
                self._record_event(
                    "position_management_failed",
                    Severity.WARNING,
                    f"Failed to manage paper position {position.position_id}",
                    {"position_id": position.position_id, "error": str(exc)},
                )

    def _record_raw_candidate(self, raw_candidate) -> None:
        self.market_universe.record_raw_signal(list(raw_candidate.market_slugs), raw_candidate.ts)
        if self._current_summary is None:
            return
        strategy_family = raw_candidate.strategy_family.value
        self._current_summary.raw_candidate_family_counts[strategy_family] += 1
        self._current_summary.record_strategy_family_signal(strategy_family, list(raw_candidate.market_slugs))

    def _record_qualified_candidate(self, candidate: RankedOpportunity) -> None:
        self.market_universe.record_qualified_candidate(list(candidate.market_slugs), candidate.ts)
        if self._current_summary is None:
            return
        self._current_summary.qualified_candidate_family_counts[candidate.strategy_family.value] += 1
        if candidate.research_only:
            self._current_summary.research_only_family_counts[candidate.strategy_family.value] += 1

    def _decorate_raw_candidate(self, raw_candidate):
        return raw_candidate.model_copy(
            update={
                "metadata": {
                    **raw_candidate.metadata,
                    **self._runtime_metadata(),
                    "strategy_family": raw_candidate.strategy_family.value,
                    "execution_mode": raw_candidate.execution_mode,
                    "research_only": raw_candidate.research_only,
                }
            }
        )

    def _qualify_and_rank_candidate(
        self,
        raw_candidate,
        books_by_token: dict[str, object],
        account_snapshot,
        *,
        audit_sink: dict[str, object] | None = None,
    ) -> RankedOpportunity | None:
        decision = self.feasibility.qualify(raw_candidate, books_by_token)
        if self._current_auditor is not None:
            self._current_auditor.record(decision)
        if audit_sink is not None:
            audit_sink["actual_leg_count_used"] = len(list(raw_candidate.legs or []))
            audit_sink["raw_candidate_generated"] = True
            audit_sink["raw_candidate_edge_inputs"] = {
                "gross_edge_cents": round(float(raw_candidate.gross_edge_cents), 6),
                "gross_profit_usd": round(float(raw_candidate.gross_profit_usd), 6),
                "expected_payout": round(float(raw_candidate.expected_payout), 6),
                "target_notional_usd": round(float(raw_candidate.target_notional_usd), 6),
                "target_shares": round(float(raw_candidate.target_shares), 6),
                "est_fill_cost_usd": round(float(raw_candidate.est_fill_cost_usd), 6),
                "basket_bid_sum": raw_candidate.metadata.get("basket_bid_sum"),
                "maker_edge_cents": raw_candidate.metadata.get("maker_edge_cents"),
            }
            audit_sink.update(
                self._build_neg_risk_gate_values(
                    raw_candidate=raw_candidate,
                    qualification_metadata=decision.metadata if isinstance(decision.metadata, dict) else {},
                )
            )
            audit_sink["qualification_rejection_reasons"] = list(decision.reason_codes)
            audit_sink["qualified"] = bool(decision.passed and decision.executable_candidate is not None)
        if not decision.passed or decision.executable_candidate is None:
            if self._current_summary is not None and any(
                reason in _QUALIFICATION_NEAR_MISS_REASONS
                for reason in decision.reason_codes
            ):
                self._current_summary.near_miss_candidates += 1
                self._current_summary.near_miss_family_counts[raw_candidate.strategy_family.value] += 1
                self.market_universe.record_near_miss(list(raw_candidate.market_slugs), decision.ts)
            for reason_code in decision.reason_codes:
                self._record_rejection(
                    stage="qualification",
                    reason_code=reason_code,
                    candidate_id=raw_candidate.candidate_id,
                    metadata={
                        "strategy_family": raw_candidate.strategy_family.value,
                        "execution_mode": raw_candidate.execution_mode,
                        "research_only": raw_candidate.research_only,
                        "market_slugs": raw_candidate.market_slugs,
                        "raw_candidate": raw_candidate.model_dump(mode="json"),
                        "qualification": decision.metadata,
                    },
                )
            return None

        ranked = self.ranker.rank(decision.executable_candidate)
        sizing = self.sizer.size(ranked, account_snapshot)
        if sizing.notional_usd <= 1e-9 or sizing.shares <= 1e-9:
            if audit_sink is not None:
                reasons = list(audit_sink.get("qualification_rejection_reasons") or [])
                reasons.append(RejectionReason.ORDER_SIZE_LIMIT.value)
                audit_sink["qualification_rejection_reasons"] = sorted(set(reasons))
                audit_sink["qualified"] = False
            if (
                self._current_summary is not None
                and sizing.rejection_reason == RejectionReason.SIZED_NOTIONAL_TOO_SMALL.value
            ):
                self._current_summary.near_miss_candidates += 1
                self._current_summary.near_miss_family_counts[raw_candidate.strategy_family.value] += 1
            self._record_rejection(
                stage="qualification",
                reason_code=RejectionReason.ORDER_SIZE_LIMIT.value,
                candidate_id=raw_candidate.candidate_id,
                metadata={
                    "strategy_family": raw_candidate.strategy_family.value,
                    "execution_mode": raw_candidate.execution_mode,
                    "research_only": raw_candidate.research_only,
                    "market_slugs": raw_candidate.market_slugs,
                    "sizing": {
                        "notional_usd": sizing.notional_usd,
                        "shares": sizing.shares,
                        "reason": sizing.reason,
                        "metadata": sizing.metadata,
                    },
                },
            )
            return None

        scale = sizing.notional_usd / max(ranked.target_notional_usd, 1e-9)
        scaled_legs = [
            leg.model_copy(update={"required_shares": sizing.shares})
            for leg in ranked.legs
        ]
        hooked = ranked.model_copy(
            update={
                "target_notional_usd": sizing.notional_usd,
                "expected_payout": round(ranked.expected_payout * scale, 6),
                "estimated_net_profit_usd": round(ranked.estimated_net_profit_usd * scale, 6),
                "expected_gross_profit_usd": round(ranked.expected_gross_profit_usd * scale, 6),
                "expected_fee_usd": round(ranked.expected_fee_usd * scale, 6),
                "expected_slippage_usd": round(ranked.expected_slippage_usd * scale, 6),
                "required_depth_usd": round(ranked.required_depth_usd * scale, 6),
                "required_shares": sizing.shares,
                "sizing_hint_usd": sizing.notional_usd,
                "sizing_hint_shares": sizing.shares,
                "legs": scaled_legs,
                "metadata": {
                    **ranked.metadata,
                    **self._runtime_metadata(),
                    "strategy_family": ranked.strategy_family.value,
                    "execution_mode": ranked.execution_mode,
                    "research_only": ranked.research_only,
                    "qualification": ranked.qualification_metadata,
                    "sizing_decision": {
                        "notional_usd": sizing.notional_usd,
                        "shares": sizing.shares,
                        "reason": sizing.reason,
                        "metadata": sizing.metadata,
                    },
                },
            }
        )
        return self._apply_ranked_opportunity_hook([hooked])[0]

    def _build_experiment_metadata(self) -> dict[str, object]:
        context = self._experiment_context.copy()
        context.setdefault("execution_mode", self.config.execution.mode)
        context["parameter_set"] = self._parameter_snapshot()
        context["scan_scope"] = {
            "market_limit": self.config.market_data.market_limit,
            "scan_interval_sec": self.config.market_data.scan_interval_sec,
        }
        return context

    def _target_strategy_families(self) -> set[str] | None:
        raw_targets = self._experiment_context.get("campaign_target_strategy_families")
        if raw_targets is None:
            return None
        if isinstance(raw_targets, str):
            candidates = [raw_targets]
        elif isinstance(raw_targets, (list, tuple, set)):
            candidates = list(raw_targets)
        else:
            return None
        targets = {
            str(candidate).strip()
            for candidate in candidates
            if str(candidate).strip()
        }
        return targets or None

    def _active_single_market_strategies(self) -> list:
        targets = self._target_strategy_families()
        if targets is None:
            return [self.single_market_strategy]

        active = []
        if self.single_market_strategy.strategy_family.value in targets:
            active.append(self.single_market_strategy)
        if self.single_market_touch_strategy.strategy_family.value in targets:
            active.append(self.single_market_touch_strategy)
        return active

    def _active_cross_market_strategies(self) -> list:
        targets = self._target_strategy_families()
        if targets is None:
            return [self.cross_market_strategy]

        active = []
        if self.cross_market_strategy.strategy_family.value in targets:
            active.append(self.cross_market_strategy)
        if self.cross_market_gross_strategy.strategy_family.value in targets:
            active.append(self.cross_market_gross_strategy)
        if self.cross_market_execution_gross_strategy.strategy_family.value in targets:
            active.append(self.cross_market_execution_gross_strategy)
        return active

    def _active_neg_risk_strategies(self) -> list:
        targets = self._target_strategy_families()
        if targets is None:
            return []
        active = []
        if self.neg_risk_strategy.strategy_family.value in targets:
            active.append(self.neg_risk_strategy)
        return active

    def _neg_risk_family_key(self, event_group: dict) -> str:
        return str(event_group.get("event_slug") or event_group.get("event_id") or "").strip()

    def _neg_risk_family_market_slugs(self, event_group: dict) -> list[str]:
        return [
            str(market.get("slug") or "").strip()
            for market in event_group.get("markets", [])
            if str(market.get("slug") or "").strip()
        ]

    def _normalize_neg_risk_watch_key(self, value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-")

    def _neg_risk_watchlist_descriptors(self) -> list[dict[str, object]]:
        remembered_market_slugs = sorted(self.market_universe.remembered_watchlist_market_slugs())
        active_market_entries = list(self.market_universe.active_entries())
        active_family_entries = list(self.market_universe.active_family_entries())
        family_memory_entries = [
            entry
            for entry in active_family_entries
            if (
                entry.seeded_productive
                or entry.productive_outcome_count > 0
                or entry.seeded_recent_qualified
                or entry.qualified_count > 0
                or entry.seeded_recent_near_miss
                or entry.near_miss_count > 0
                or entry.raw_signal_count > 0
            )
        ]
        raw_entries = [
            *list(self.config.market_data.neg_risk_family_audit_watchlist),
            *remembered_market_slugs,
            *[entry.family_slug for entry in family_memory_entries if str(entry.family_slug).strip()],
        ]
        descriptors: dict[str, dict[str, object]] = {}
        configured = {
            self._normalize_neg_risk_watch_key(item)
            for item in self.config.market_data.neg_risk_family_audit_watchlist
        }
        productive_market_keys = {
            self._normalize_neg_risk_watch_key(entry.market_slug)
            for entry in active_market_entries
            if entry.seeded_productive or entry.productive_outcome_count > 0
        }
        recent_qualified_market_keys = {
            self._normalize_neg_risk_watch_key(entry.market_slug)
            for entry in active_market_entries
            if entry.seeded_recent_qualified or entry.qualified_count > 0 or entry.last_qualified_at is not None
        }
        recent_near_miss_market_keys = {
            self._normalize_neg_risk_watch_key(entry.market_slug)
            for entry in active_market_entries
            if entry.seeded_recent_near_miss or entry.near_miss_count > 0 or entry.last_near_miss_at is not None
        }
        productive_family_keys = {
            self._normalize_neg_risk_watch_key(entry.family_slug)
            for entry in active_family_entries
            if entry.seeded_productive or entry.productive_outcome_count > 0
        }
        recent_qualified_family_keys = {
            self._normalize_neg_risk_watch_key(entry.family_slug)
            for entry in active_family_entries
            if entry.seeded_recent_qualified or entry.qualified_count > 0 or entry.last_qualified_at is not None
        }
        recent_near_miss_family_keys = {
            self._normalize_neg_risk_watch_key(entry.family_slug)
            for entry in active_family_entries
            if entry.seeded_recent_near_miss or entry.near_miss_count > 0 or entry.last_near_miss_at is not None
        }
        semi_productive_family_keys = {
            self._normalize_neg_risk_watch_key(entry.family_slug)
            for entry in active_family_entries
            if (
                (entry.qualified_count > 0 or entry.last_qualified_at is not None)
                and not (entry.seeded_productive or entry.productive_outcome_count > 0)
            )
        }
        raw_positive_narrow_fail_family_keys = {
            self._normalize_neg_risk_watch_key(entry.family_slug)
            for entry in active_family_entries
            if (
                (entry.raw_signal_count > 0 or entry.last_raw_signal_at is not None)
                and (entry.seeded_recent_near_miss or entry.near_miss_count > 0 or entry.last_near_miss_at is not None)
            )
        }
        for raw_entry in raw_entries:
            normalized = self._normalize_neg_risk_watch_key(raw_entry)
            if not normalized:
                continue
            descriptor = descriptors.get(normalized)
            if descriptor is None:
                descriptor = {
                    "watchlist_entry": str(raw_entry),
                    "watchlist_family_key": normalized,
                    "expected_slugs": [normalized],
                    "source_tags": set(),
                }
                descriptors[normalized] = descriptor
            source_tags = descriptor["source_tags"]
            if isinstance(source_tags, set):
                if normalized in configured:
                    source_tags.add("config")
                if normalized in {
                    self._normalize_neg_risk_watch_key(slug) for slug in remembered_market_slugs
                }:
                    source_tags.add("history")
                if normalized in productive_market_keys or normalized in productive_family_keys:
                    source_tags.add("history_productive")
                if normalized in recent_qualified_market_keys or normalized in recent_qualified_family_keys:
                    source_tags.add("history_recent_qualified")
                if normalized in recent_near_miss_market_keys or normalized in recent_near_miss_family_keys:
                    source_tags.add("history_near_miss")
                if normalized in semi_productive_family_keys:
                    source_tags.add("history_semi_productive")
                if normalized in raw_positive_narrow_fail_family_keys:
                    source_tags.add("history_raw_positive")

            family_entry = self.market_universe.family_entries.get(normalized)
            if family_entry is None:
                market_entry = self.market_universe.active_entry_for_slug(str(raw_entry))
                if market_entry is not None and str(market_entry.event_slug or "").strip():
                    family_entry = self.market_universe.family_entries.get(str(market_entry.event_slug))
            if family_entry is not None and family_entry.market_slugs:
                expected = {
                    self._normalize_neg_risk_watch_key(slug)
                    for slug in family_entry.market_slugs
                    if self._normalize_neg_risk_watch_key(slug)
                }
                expected.add(normalized)
                descriptor["expected_slugs"] = sorted(expected)
                if isinstance(source_tags, set):
                    if family_entry.seeded_productive or family_entry.productive_outcome_count > 0:
                        source_tags.add("history_productive")
                    if family_entry.seeded_recent_qualified or family_entry.qualified_count > 0 or family_entry.last_qualified_at is not None:
                        source_tags.add("history_recent_qualified")
                    if family_entry.seeded_recent_near_miss or family_entry.near_miss_count > 0 or family_entry.last_near_miss_at is not None:
                        source_tags.add("history_near_miss")
                    if (
                        (family_entry.qualified_count > 0 or family_entry.last_qualified_at is not None)
                        and not (family_entry.seeded_productive or family_entry.productive_outcome_count > 0)
                    ):
                        source_tags.add("history_semi_productive")
                    if (
                        (family_entry.raw_signal_count > 0 or family_entry.last_raw_signal_at is not None)
                        and (
                            family_entry.seeded_recent_near_miss
                            or family_entry.near_miss_count > 0
                            or family_entry.last_near_miss_at is not None
                        )
                    ):
                        source_tags.add("history_raw_positive")
                    if {
                        "history_productive",
                        "history_recent_qualified",
                        "history_near_miss",
                        "history_semi_productive",
                        "history_raw_positive",
                    } & source_tags:
                        source_tags.add("history")

        for descriptor in descriptors.values():
            source_tags = descriptor.get("source_tags")
            if isinstance(source_tags, set):
                descriptor["source_tags"] = sorted(source_tags)
        return list(descriptors.values())

    def _neg_risk_group_aliases(self, event_group: dict) -> set[str]:
        aliases: set[str] = set()
        family_key = self._normalize_neg_risk_watch_key(self._neg_risk_family_key(event_group))
        if family_key:
            aliases.add(family_key)
        family_id = self._normalize_neg_risk_watch_key(event_group.get("event_id"))
        if family_id:
            aliases.add(family_id)
        aliases.update(
            self._normalize_neg_risk_watch_key(slug)
            for slug in self._neg_risk_family_market_slugs(event_group)
            if self._normalize_neg_risk_watch_key(slug)
        )
        return aliases

    def _neg_risk_match_watch_descriptor(
        self,
        event_group: dict,
        descriptor: dict[str, object],
    ) -> dict[str, object] | None:
        watch_key = str(descriptor.get("watchlist_family_key") or "")
        if not watch_key:
            return None
        expected_slugs = {
            self._normalize_neg_risk_watch_key(slug)
            for slug in list(descriptor.get("expected_slugs") or [])
            if self._normalize_neg_risk_watch_key(slug)
        }
        aliases = self._neg_risk_group_aliases(event_group)
        matched_aliases = set(expected_slugs & aliases)
        if watch_key in aliases:
            matched_aliases.add(watch_key)
        if not matched_aliases:
            return None
        return {
            "family_key": self._neg_risk_family_key(event_group),
            "event_group": event_group,
            "matched_slugs_count": len(matched_aliases),
            "missing_slugs": sorted(expected_slugs - matched_aliases),
        }

    def _neg_risk_condition_monitor_mode_enabled(self) -> bool:
        return bool(
            self.config.market_data.neg_risk_condition_monitor_mode_enabled
            and list(self.config.market_data.neg_risk_condition_monitor_watchlist)
        )

    def _neg_risk_condition_monitor_family_keys(self) -> set[str]:
        return {
            self._normalize_neg_risk_watch_key(item)
            for item in self.config.market_data.neg_risk_condition_monitor_watchlist
            if self._normalize_neg_risk_watch_key(item)
        }

    def _neg_risk_condition_monitor_descriptors(self) -> list[dict[str, object]]:
        descriptors: dict[str, dict[str, object]] = {}
        for raw_entry in self.config.market_data.neg_risk_condition_monitor_watchlist:
            normalized = self._normalize_neg_risk_watch_key(raw_entry)
            if not normalized:
                continue
            descriptor = {
                "watchlist_entry": str(raw_entry),
                "watchlist_family_key": normalized,
                "expected_slugs": [normalized],
                "source_tags": ["condition_monitor"],
            }
            family_entry = self.market_universe.family_entries.get(normalized)
            if family_entry is None:
                market_entry = self.market_universe.active_entry_for_slug(str(raw_entry))
                if market_entry is not None and str(market_entry.event_slug or "").strip():
                    family_entry = self.market_universe.family_entries.get(str(market_entry.event_slug))
            if family_entry is not None and family_entry.market_slugs:
                expected = {
                    self._normalize_neg_risk_watch_key(slug)
                    for slug in family_entry.market_slugs
                    if self._normalize_neg_risk_watch_key(slug)
                }
                expected.add(normalized)
                descriptor["expected_slugs"] = sorted(expected)
            descriptors[normalized] = descriptor
        return list(descriptors.values())

    def _neg_risk_condition_monitor_match(self, event_group: dict) -> bool:
        if not self._neg_risk_condition_monitor_mode_enabled():
            return True
        for descriptor in self._neg_risk_condition_monitor_descriptors():
            if self._neg_risk_match_watch_descriptor(event_group, descriptor) is not None:
                return True
        return False

    def _neg_risk_similar_groups(
        self,
        descriptor: dict[str, object],
        event_groups: list[dict],
        *,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        watch_key = str(descriptor.get("watchlist_family_key") or "")
        if not watch_key:
            return []
        watch_tokens = {token for token in watch_key.split("-") if token}
        scored: list[tuple[int, str, list[str]]] = []
        for event_group in event_groups:
            family_key = self._neg_risk_family_key(event_group)
            if not family_key:
                continue
            alias_tokens: set[str] = set()
            for alias in self._neg_risk_group_aliases(event_group):
                alias_tokens.update(token for token in alias.split("-") if token)
            shared = sorted(watch_tokens & alias_tokens)
            if not shared:
                continue
            scored.append((len(shared), family_key, shared))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "family_key": family_key,
                "shared_tokens": shared_tokens,
            }
            for _, family_key, shared_tokens in scored[:limit]
        ]

    def _reconcile_neg_risk_watchlist_groups(
        self,
        event_groups: list[dict],
    ) -> tuple[list[dict], list[dict[str, object]]]:
        from src.scanner.neg_risk import build_eligible_neg_risk_event_groups

        descriptors = self._neg_risk_watchlist_descriptors()
        if not descriptors:
            return list(event_groups), []

        diagnostics: list[dict[str, object]] = []
        current_family_keys = {
            self._neg_risk_family_key(event_group)
            for event_group in event_groups
            if self._neg_risk_family_key(event_group)
        }
        unmatched_descriptors: list[tuple[dict[str, object], list[dict[str, object]]]] = []

        for descriptor in descriptors:
            current_match = next(
                (
                    match
                    for event_group in event_groups
                    if (match := self._neg_risk_match_watch_descriptor(event_group, descriptor)) is not None
                ),
                None,
            )
            similar_groups = self._neg_risk_similar_groups(descriptor, event_groups)
            if current_match is not None:
                diagnostics.append(
                    {
                        "watchlist_family_key": descriptor["watchlist_family_key"],
                        "watchlist_entry": descriptor["watchlist_entry"],
                        "expected_slugs": list(descriptor.get("expected_slugs") or []),
                        "matched_slugs_count": current_match["matched_slugs_count"],
                        "missing_slugs": current_match["missing_slugs"],
                        "discovered_candidate_groups_considered_similar": similar_groups,
                        "match_status": "active_discovery",
                        "matched_family_key": current_match["family_key"],
                        "injected_for_diagnosis": False,
                        "mismatch_reason": None,
                        "source_tags": list(descriptor.get("source_tags") or []),
                    }
                )
                continue
            unmatched_descriptors.append((descriptor, similar_groups))

        if not unmatched_descriptors:
            return list(event_groups), diagnostics

        reconciliation_limit = max(
            int(self.config.market_data.neg_risk_watchlist_reconciliation_event_limit),
            100,
        )
        reconciliation_budget = max(0, int(self.config.market_data.neg_risk_family_audit_budget))
        extra_events = fetch_events(self.config.market_data.gamma_host, limit=reconciliation_limit)
        self._record_batch_request("gamma_events_watchlist_reconciliation")
        extra_event_groups = build_eligible_neg_risk_event_groups(extra_events)
        injected_groups: list[dict] = []

        for descriptor, similar_groups in unmatched_descriptors:
            reconciled_match = next(
                (
                    match
                    for event_group in extra_event_groups
                    if (match := self._neg_risk_match_watch_descriptor(event_group, descriptor)) is not None
                ),
                None,
            )
            if reconciled_match is None:
                diagnostics.append(
                    {
                        "watchlist_family_key": descriptor["watchlist_family_key"],
                        "watchlist_entry": descriptor["watchlist_entry"],
                        "expected_slugs": list(descriptor.get("expected_slugs") or []),
                        "matched_slugs_count": 0,
                        "missing_slugs": list(descriptor.get("expected_slugs") or []),
                        "discovered_candidate_groups_considered_similar": similar_groups,
                        "match_status": "unmatched",
                        "matched_family_key": None,
                        "injected_for_diagnosis": False,
                        "mismatch_reason": "no_active_neg_risk_family_match_found",
                        "source_tags": list(descriptor.get("source_tags") or []),
                    }
                )
                continue

            reconciled_group = reconciled_match["event_group"]
            reconciled_family_key = str(reconciled_match["family_key"] or "")
            injected = False
            if (
                reconciled_family_key
                and reconciled_family_key not in current_family_keys
                and len(injected_groups) < reconciliation_budget
            ):
                injected_groups.append(reconciled_group)
                current_family_keys.add(reconciled_family_key)
                injected = True

            diagnostics.append(
                {
                    "watchlist_family_key": descriptor["watchlist_family_key"],
                    "watchlist_entry": descriptor["watchlist_entry"],
                    "expected_slugs": list(descriptor.get("expected_slugs") or []),
                    "matched_slugs_count": reconciled_match["matched_slugs_count"],
                    "missing_slugs": reconciled_match["missing_slugs"],
                    "discovered_candidate_groups_considered_similar": similar_groups,
                    "match_status": "bounded_rescan_injected" if injected else "bounded_rescan_matched",
                    "matched_family_key": reconciled_family_key or None,
                    "injected_for_diagnosis": injected,
                    "mismatch_reason": (
                        "outside_primary_discovery_window"
                        if injected
                        else "matched_but_injection_budget_exhausted"
                    ),
                    "source_tags": list(descriptor.get("source_tags") or []),
                }
            )

        return [*event_groups, *injected_groups], diagnostics

    def _neg_risk_audit_watchlist_match(self, event_group: dict) -> bool:
        for descriptor in self._neg_risk_watchlist_descriptors():
            if self._neg_risk_match_watch_descriptor(event_group, descriptor) is not None:
                return True
        return False

    def _select_neg_risk_event_groups_for_scan(
        self,
        *,
        event_groups: list[dict],
        changed_market_slugs: set[str] | None,
        cycle_started: datetime,
    ) -> tuple[list[dict], dict[str, object], set[str], set[str]]:
        selected_event_groups, family_metrics = self.market_universe.select_neg_risk_groups_for_recompute(
            event_groups=event_groups,
            changed_market_slugs=changed_market_slugs,
            now=cycle_started,
            cycle_index=self._run_sequence,
        )
        if self._neg_risk_condition_monitor_mode_enabled():
            selected_event_groups = [
                event_group
                for event_group in selected_event_groups
                if self._neg_risk_condition_monitor_match(event_group)
            ]
            family_metrics = {
                **family_metrics,
                "families_considered": len(selected_event_groups),
            }
        naturally_selected_keys = {
            self._neg_risk_family_key(event_group)
            for event_group in selected_event_groups
        }
        audit_forced_keys: set[str] = set()
        if not self.config.market_data.neg_risk_family_audit_mode_enabled:
            return selected_event_groups, family_metrics, naturally_selected_keys, audit_forced_keys

        budget = max(0, int(self.config.market_data.neg_risk_family_audit_budget))
        if budget <= 0:
            return selected_event_groups, family_metrics, naturally_selected_keys, audit_forced_keys

        forced_groups: list[dict] = []
        for event_group in event_groups:
            family_key = self._neg_risk_family_key(event_group)
            if not family_key or family_key in naturally_selected_keys:
                continue
            if self._neg_risk_condition_monitor_mode_enabled():
                if not self._neg_risk_condition_monitor_match(event_group):
                    continue
            elif not self._neg_risk_audit_watchlist_match(event_group):
                continue
            forced_groups.append(event_group)
            audit_forced_keys.add(family_key)
            if len(forced_groups) >= budget:
                break

        if forced_groups:
            selected_event_groups = [*selected_event_groups, *forced_groups]
            family_metrics = {
                **family_metrics,
                "families_considered": int(family_metrics.get("families_considered", 0)) + len(forced_groups),
            }

        if self._neg_risk_condition_monitor_mode_enabled():
            return selected_event_groups, family_metrics, naturally_selected_keys, audit_forced_keys

        selector_budget = max(0, int(self.config.market_data.neg_risk_selector_refresh_budget))
        remaining_budget = max(0, selector_budget - len(audit_forced_keys))
        if remaining_budget <= 0:
            return selected_event_groups, family_metrics, naturally_selected_keys, audit_forced_keys

        refresh_forced_groups: list[dict] = []
        ranked_candidates: list[tuple[int, int, str, dict]] = []
        for event_group in event_groups:
            family_key = self._neg_risk_family_key(event_group)
            if not family_key or family_key in naturally_selected_keys or family_key in audit_forced_keys:
                continue
            source_tags = set(
                self._neg_risk_family_source_tags(
                    family_key,
                    include_current_discovery_fallback=True,
                )
            )
            if "history_recent_qualified" in source_tags:
                priority = 0
            elif "history_near_miss" in source_tags:
                priority = 1
            elif "history_semi_productive" in source_tags:
                priority = 2
            elif "history_raw_positive" in source_tags:
                priority = 3
            elif "current_discovery_sample" in source_tags:
                priority = 4
            else:
                priority = 5
            ranked_candidates.append((priority, -len(self._neg_risk_family_market_slugs(event_group)), family_key, event_group))

        ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _priority, _market_count_sort, family_key, event_group in ranked_candidates:
            refresh_forced_groups.append(event_group)
            audit_forced_keys.add(family_key)
            if len(refresh_forced_groups) >= remaining_budget:
                break

        if refresh_forced_groups:
            selected_event_groups = [*selected_event_groups, *refresh_forced_groups]
            family_metrics = {
                **family_metrics,
                "families_considered": int(family_metrics.get("families_considered", 0)) + len(refresh_forced_groups),
            }
        return selected_event_groups, family_metrics, naturally_selected_keys, audit_forced_keys

    def _neg_risk_watch_source_tags_by_family(
        self,
        diagnostics: list[dict[str, object]],
    ) -> dict[str, list[str]]:
        tags_by_family: dict[str, set[str]] = {}
        for diagnostic in diagnostics:
            family_key = str(diagnostic.get("matched_family_key") or "")
            if not family_key:
                continue
            tags = {
                str(tag)
                for tag in list(diagnostic.get("source_tags") or [])
                if str(tag)
            }
            if not tags:
                continue
            tags_by_family.setdefault(family_key, set()).update(tags)
        return {
            family_key: sorted(tags)
            for family_key, tags in tags_by_family.items()
        }

    def _neg_risk_family_source_tags(
        self,
        family_key: str,
        base_tags: list[str] | None = None,
        *,
        include_current_discovery_fallback: bool = False,
    ) -> list[str]:
        tags = {
            str(tag)
            for tag in list(base_tags or [])
            if str(tag)
        }
        normalized = self._normalize_neg_risk_watch_key(family_key)
        family_entry = self.market_universe.family_entries.get(normalized)
        if family_entry is None:
            family_entry = self.market_universe.family_entries.get(str(family_key or "").strip())
        if family_entry is not None:
            if family_entry.seeded_productive or family_entry.productive_outcome_count > 0:
                tags.add("history_productive")
            if family_entry.seeded_recent_qualified or family_entry.qualified_count > 0 or family_entry.last_qualified_at is not None:
                tags.add("history_recent_qualified")
            if family_entry.seeded_recent_near_miss or family_entry.near_miss_count > 0 or family_entry.last_near_miss_at is not None:
                tags.add("history_near_miss")
            if (
                (family_entry.qualified_count > 0 or family_entry.last_qualified_at is not None)
                and not (family_entry.seeded_productive or family_entry.productive_outcome_count > 0)
            ):
                tags.add("history_semi_productive")
            if (
                (family_entry.raw_signal_count > 0 or family_entry.last_raw_signal_at is not None)
                and (
                    family_entry.seeded_recent_near_miss
                    or family_entry.near_miss_count > 0
                    or family_entry.last_near_miss_at is not None
                )
            ):
                tags.add("history_raw_positive")
        if {"history_productive", "history_recent_qualified", "history_near_miss", "history_semi_productive", "history_raw_positive"} & tags:
            tags.add("history")
        if include_current_discovery_fallback and not tags:
            tags.add("current_discovery_sample")
        return sorted(tags)

    def _build_neg_risk_family_input_audit(
        self,
        *,
        event_group: dict,
        books_by_token: dict[str, object],
        cycle_started: datetime,
        naturally_selected: bool,
        audit_forced: bool,
        source_tags: list[str] | None = None,
    ) -> dict[str, object]:
        market_slugs = self._neg_risk_family_market_slugs(event_group)
        expected_leg_count = len(list(event_group.get("markets") or []))
        valid_book_leg_count = 0
        negative_cached_leg_count = 0
        missing_leg_count = 0
        snapshot_ages_ms: list[float] = []
        negative_cached_legs: list[str] = []
        missing_legs: list[str] = []
        actual_slugs_used: list[str] = []

        for market in list(event_group.get("markets") or []):
            market_slug = str(market.get("slug") or "")
            token_id = str(market.get("yes_token_id") or "")
            if token_id and self.clob.negative_cache_reason(token_id) is not None:
                negative_cached_leg_count += 1
                negative_cached_legs.append(market_slug)
            book = books_by_token.get(token_id)
            if book is None:
                missing_leg_count += 1
                missing_legs.append(market_slug)
                continue
            bids = getattr(book, "bids", []) or []
            if bids:
                try:
                    best_bid = float(bids[0].price)
                    best_size = float(bids[0].size)
                except Exception:
                    best_bid = 0.0
                    best_size = 0.0
                if best_bid > 0.0 and best_size > 0.0:
                    valid_book_leg_count += 1
                    actual_slugs_used.append(market_slug)
            book_ts = getattr(book, "ts", None)
            if isinstance(book_ts, datetime):
                snapshot_ages_ms.append(max(0.0, (cycle_started - book_ts).total_seconds() * 1000.0))

        family_snapshot_age_max_ms = round(max(snapshot_ages_ms), 3) if snapshot_ages_ms else None
        family_snapshot_age_min_ms = round(min(snapshot_ages_ms), 3) if snapshot_ages_ms else None
        family_snapshot_time_skew_ms = round(
            max(snapshot_ages_ms) - min(snapshot_ages_ms),
            3,
        ) if len(snapshot_ages_ms) >= 2 else 0.0
        same_window = (
            valid_book_leg_count == expected_leg_count
            and negative_cached_leg_count == 0
            and family_snapshot_age_max_ms is not None
            and family_snapshot_age_max_ms <= float(self.config.market_data.stale_book_sec) * 1000.0
            and family_snapshot_time_skew_ms <= float(self.config.market_data.scan_interval_sec) * 1000.0
        )
        inputs_coherent = (
            valid_book_leg_count == expected_leg_count
            and missing_leg_count == 0
            and negative_cached_leg_count == 0
            and bool(same_window)
        )
        return {
            "family_id": str(event_group.get("event_id") or ""),
            "family_key": self._neg_risk_family_key(event_group),
            "family_considered": True,
            "strategy_family": self.neg_risk_strategy.strategy_family.value,
            "source_tags": sorted({str(tag) for tag in list(source_tags or []) if str(tag)}),
            "expected_slugs": market_slugs,
            "actual_slugs_used": actual_slugs_used,
            "market_slugs": market_slugs,
            "expected_leg_count": expected_leg_count,
            "actual_leg_count_used": valid_book_leg_count,
            "valid_book_leg_count": valid_book_leg_count,
            "negative_cached_leg_count": negative_cached_leg_count,
            "missing_leg_count": missing_leg_count,
            "all_expected_legs_present": bool(missing_leg_count == 0 and negative_cached_leg_count == 0),
            "negative_cached_legs": negative_cached_legs,
            "missing_legs": missing_legs,
            "family_snapshot_age_max_ms": family_snapshot_age_max_ms,
            "family_snapshot_age_min_ms": family_snapshot_age_min_ms,
            "family_snapshot_time_skew_ms": family_snapshot_time_skew_ms,
            "all_legs_refreshed_same_window": bool(same_window),
            "inputs_coherent": bool(inputs_coherent),
            "naturally_selected": naturally_selected,
            "audit_forced": audit_forced,
            "raw_candidate_generated": False,
            "qualified": False,
            "qualification_rejection_reasons": [],
            "raw_detection_failure_reason": None,
            "raw_candidate_edge_inputs": {},
            "gross_edge": None,
            "net_edge": None,
            "expected_net_profit_usd": None,
            "spread": None,
            "absolute_depth": None,
            "depth_multiple": None,
            "single_leg_concentration": None,
            "partial_fill_risk": None,
            "non_atomic_risk": None,
        }

    def _build_neg_risk_gate_values(
        self,
        *,
        raw_candidate,
        qualification_metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        metadata = qualification_metadata or {}
        legs = list(metadata.get("legs") or [])
        spreads = [
            float(leg.get("spread_cents"))
            for leg in legs
            if leg.get("spread_cents") is not None
        ]
        available_notional = [
            float(leg.get("available_notional_usd"))
            for leg in legs
            if leg.get("available_notional_usd") is not None
        ]
        best_prices = [
            float(leg.get("best_price"))
            for leg in legs
            if leg.get("best_price") is not None
        ]
        required_depth = float(metadata.get("required_depth_usd", 0.0) or 0.0)
        available_depth = float(metadata.get("available_depth_usd", 0.0) or 0.0)
        gross_edge = float(metadata.get("expected_gross_edge_cents", raw_candidate.gross_edge_cents))
        net_edge = float(
            metadata.get(
                "expected_net_edge_cents",
                gross_edge - self.config.opportunity.fee_buffer_cents - self.config.opportunity.slippage_buffer_cents,
            )
        )
        expected_net_profit = float(
            metadata.get(
                "expected_net_profit_usd",
                raw_candidate.gross_profit_usd
                - (raw_candidate.target_shares * self.config.opportunity.fee_buffer_cents)
                - (raw_candidate.target_shares * self.config.opportunity.slippage_buffer_cents),
            )
        )
        return {
            "gross_edge": round(gross_edge, 6),
            "gross_edge_cents": round(gross_edge, 6),
            "net_edge": round(net_edge, 6),
            "net_edge_cents": round(net_edge, 6),
            "expected_net_profit_usd": round(expected_net_profit, 6),
            "spread": round(max(spreads), 6) if spreads else None,
            "max_spread_observed": round(max(spreads), 6) if spreads else None,
            "absolute_depth": round(min(available_notional), 6) if available_notional else None,
            "required_depth_usd": round(required_depth, 6),
            "available_depth_usd": round(available_depth, 6),
            "depth_multiple": round(available_depth / required_depth, 6) if required_depth > 1e-9 else None,
            "single_leg_concentration": round(max(best_prices), 6) if best_prices else None,
            "partial_fill_risk": round(float(metadata.get("partial_fill_risk_score", 0.0) or 0.0), 6) if metadata else None,
            "partial_fill_risk_score": round(float(metadata.get("partial_fill_risk_score", 0.0) or 0.0), 6) if metadata else None,
            "non_atomic_risk": round(float(metadata.get("non_atomic_execution_risk_score", 0.0) or 0.0), 6) if metadata else None,
            "non_atomic_execution_risk_score": round(float(metadata.get("non_atomic_execution_risk_score", 0.0) or 0.0), 6) if metadata else None,
        }

    def _record_neg_risk_family_audit(self, payload: dict[str, object]) -> None:
        self._neg_risk_family_audit_payloads.append(payload)
        if self._current_summary is None:
            return
        self._current_summary.metadata["neg_risk_family_qualification_audit"] = list(self._neg_risk_family_audit_payloads)

    def _evaluate_neg_risk_family_path_for_audit(
        self,
        *,
        event_group: dict,
        books_by_token: dict[str, object],
        cycle_started: datetime,
        source_tags: list[str],
        naturally_selected: bool,
        audit_forced: bool,
    ) -> dict[str, object]:
        audit_payload = self._build_neg_risk_family_input_audit(
            event_group=event_group,
            books_by_token=books_by_token,
            cycle_started=cycle_started,
            naturally_selected=naturally_selected,
            audit_forced=audit_forced,
            source_tags=source_tags,
        )
        raw_candidate, detection_audit = self.neg_risk_strategy.detect_with_audit(
            event_group,
            books_by_token,
            self.config.paper.max_notional_per_arb,
        )
        if raw_candidate is None:
            audit_payload["raw_detection_failure_reason"] = (detection_audit or {}).get("failure_reason")
            audit_payload["rejection_reason_codes"] = []
            return audit_payload

        audit_payload["actual_leg_count"] = len(list(raw_candidate.legs or []))
        audit_payload["actual_slugs_used"] = list(raw_candidate.market_slugs)
        audit_payload["raw_candidate_generated"] = True
        audit_payload["raw_candidate_edge_inputs"] = {
            "gross_edge_cents": round(float(raw_candidate.gross_edge_cents), 6),
            "gross_profit_usd": round(float(raw_candidate.gross_profit_usd), 6),
            "expected_payout": round(float(raw_candidate.expected_payout), 6),
            "target_notional_usd": round(float(raw_candidate.target_notional_usd), 6),
            "target_shares": round(float(raw_candidate.target_shares), 6),
            "est_fill_cost_usd": round(float(raw_candidate.est_fill_cost_usd), 6),
            "basket_bid_sum": raw_candidate.metadata.get("basket_bid_sum"),
            "maker_edge_cents": raw_candidate.metadata.get("maker_edge_cents"),
        }
        decision = self.feasibility.qualify(raw_candidate, books_by_token)
        audit_payload.update(
            self._build_neg_risk_gate_values(
                raw_candidate=raw_candidate,
                qualification_metadata=decision.metadata if isinstance(decision.metadata, dict) else {},
            )
        )
        audit_payload["qualification_rejection_reasons"] = list(decision.reason_codes)
        audit_payload["rejection_reason_codes"] = list(decision.reason_codes)
        audit_payload["qualified"] = bool(decision.passed and decision.executable_candidate is not None)
        return audit_payload

    def _fetch_neg_risk_family_broad_books(
        self,
        *,
        event_group: dict,
    ) -> dict[str, object]:
        token_ids = [
            str(market.get("yes_token_id") or "")
            for market in list(event_group.get("markets") or [])
            if str(market.get("yes_token_id") or "")
        ]
        if not token_ids:
            return {}
        books = self.clob.fetch_books_batch(token_ids)
        self._record_batch_request("clob_books_neg_risk_parity")
        self._register_book_fetch(len(books))
        return books

    @staticmethod
    def _neg_risk_numeric_delta(
        fast_value: object,
        broad_value: object,
    ) -> float | None:
        if fast_value is None or broad_value is None:
            return None
        try:
            return round(float(broad_value) - float(fast_value), 6)
        except Exception:
            return None

    def _classify_neg_risk_family_parity_failure(
        self,
        *,
        fast_path_result: dict[str, object],
        broad_path_result: dict[str, object],
    ) -> str:
        fast_reasons = set(str(reason) for reason in list(fast_path_result.get("qualification_rejection_reasons") or []))
        broad_reasons = set(str(reason) for reason in list(broad_path_result.get("qualification_rejection_reasons") or []))
        fast_missing = int(fast_path_result.get("missing_leg_count", 0) or 0)
        broad_missing = int(broad_path_result.get("missing_leg_count", 0) or 0)
        fast_negative_cached = int(fast_path_result.get("negative_cached_leg_count", 0) or 0)
        broad_negative_cached = int(broad_path_result.get("negative_cached_leg_count", 0) or 0)
        fast_skew = float(fast_path_result.get("family_snapshot_time_skew_ms", 0.0) or 0.0)
        broad_skew = float(broad_path_result.get("family_snapshot_time_skew_ms", 0.0) or 0.0)
        fast_qualified = bool(fast_path_result.get("qualified"))
        broad_qualified = bool(broad_path_result.get("qualified"))
        fast_raw = bool(fast_path_result.get("raw_candidate_generated"))
        broad_raw = bool(broad_path_result.get("raw_candidate_generated"))
        mismatch = (
            fast_qualified != broad_qualified
            or fast_raw != broad_raw
            or fast_reasons != broad_reasons
        )
        depth_reasons = {
            RejectionReason.INSUFFICIENT_DEPTH.value,
            RejectionReason.ABSOLUTE_DEPTH_BELOW_FLOOR.value,
            RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value,
            RejectionReason.NON_ATOMIC_RISK_TOO_HIGH.value,
            RejectionReason.SINGLE_LEG_CONCENTRATION.value,
            RejectionReason.SPREAD_TOO_WIDE.value,
        }
        edge_reasons = {
            RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value,
            RejectionReason.EDGE_BELOW_THRESHOLD.value,
        }
        if fast_negative_cached > broad_negative_cached and (mismatch or broad_missing < fast_missing):
            return "negative_cache_side_effect"
        if fast_negative_cached > 0 and broad_qualified and not fast_qualified:
            return "negative_cache_side_effect"
        if fast_missing > broad_missing and (mismatch or broad_qualified or broad_raw):
            return "incomplete_family_input"
        if fast_skew > broad_skew + 1000.0 and (
            mismatch
            or (self._neg_risk_numeric_delta(fast_path_result.get("expected_net_profit_usd"), broad_path_result.get("expected_net_profit_usd")) or 0.0) > 0.0
        ):
            return "time_skewed_family_input"
        if (fast_reasons | broad_reasons) & depth_reasons:
            if not mismatch or bool(broad_reasons & depth_reasons):
                return "concentration_depth_degradation"
        if (
            not mismatch
            and not fast_qualified
            and not broad_qualified
            and (fast_reasons | broad_reasons) & edge_reasons
            and fast_missing == 0
            and broad_missing == 0
            and fast_negative_cached == 0
            and broad_negative_cached == 0
        ):
            return "edge_vanished_for_real"
        if not mismatch and not fast_qualified and not broad_qualified:
            return "true_market_deterioration"
        return "unclear"

    def _build_neg_risk_family_parity_payload(
        self,
        *,
        fast_path_result: dict[str, object],
        broad_path_result: dict[str, object],
    ) -> dict[str, object]:
        fast_reasons = set(str(reason) for reason in list(fast_path_result.get("qualification_rejection_reasons") or []))
        broad_reasons = set(str(reason) for reason in list(broad_path_result.get("qualification_rejection_reasons") or []))
        qualification_outcome_differs = (
            bool(fast_path_result.get("qualified")) != bool(broad_path_result.get("qualified"))
            or bool(fast_path_result.get("raw_candidate_generated")) != bool(broad_path_result.get("raw_candidate_generated"))
            or fast_reasons != broad_reasons
        )
        return {
            "family_key": fast_path_result.get("family_key"),
            "source_tags": list(fast_path_result.get("source_tags") or []),
            "expected_slugs": list(fast_path_result.get("expected_slugs") or []),
            "fast_path_result": fast_path_result,
            "broad_path_result": broad_path_result,
            "qualification_outcome_differs": qualification_outcome_differs,
            "metric_deltas": {
                "net_edge_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("net_edge_cents"),
                    broad_path_result.get("net_edge_cents"),
                ),
                "expected_net_profit_usd_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("expected_net_profit_usd"),
                    broad_path_result.get("expected_net_profit_usd"),
                ),
                "available_depth_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("available_depth_usd"),
                    broad_path_result.get("available_depth_usd"),
                ),
                "spread_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("max_spread_observed"),
                    broad_path_result.get("max_spread_observed"),
                ),
                "concentration_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("single_leg_concentration"),
                    broad_path_result.get("single_leg_concentration"),
                ),
                "snapshot_skew_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("family_snapshot_time_skew_ms"),
                    broad_path_result.get("family_snapshot_time_skew_ms"),
                ),
                "missing_legs_delta": self._neg_risk_numeric_delta(
                    fast_path_result.get("missing_leg_count"),
                    broad_path_result.get("missing_leg_count"),
                ),
            },
            "failure_class": self._classify_neg_risk_family_parity_failure(
                fast_path_result=fast_path_result,
                broad_path_result=broad_path_result,
            ),
        }

    def _record_neg_risk_family_parity(self, payload: dict[str, object]) -> None:
        self._neg_risk_family_parity_payloads.append(payload)
        if self._current_summary is None:
            return
        self._current_summary.metadata["neg_risk_family_qualification_parity_audit"] = list(
            self._neg_risk_family_parity_payloads
        )

    def _summarize_neg_risk_family_audits(self) -> dict[str, object]:
        payloads = list(getattr(self, "_neg_risk_family_audit_payloads", []) or [])
        if not payloads:
            return {
                "families_with_complete_leg_set": 0,
                "families_with_incomplete_leg_set": 0,
                "avg_missing_legs_per_family": 0.0,
                "families_blocked_by_negative_cache": 0,
                "avg_family_snapshot_time_skew_ms": 0.0,
                "families_failing_due_to_incomplete_inputs": 0,
                "raw_candidates_rejected_due_to_depth": 0,
                "raw_candidates_rejected_due_to_spread": 0,
                "raw_candidates_rejected_due_to_net_profit": 0,
                "raw_candidates_rejected_due_to_concentration": 0,
                "top_qualification_failure_reason": None,
            }

        complete = sum(1 for payload in payloads if int(payload.get("missing_leg_count", 0)) == 0 and int(payload.get("negative_cached_leg_count", 0)) == 0)
        incomplete = sum(1 for payload in payloads if int(payload.get("missing_leg_count", 0)) > 0 or int(payload.get("negative_cached_leg_count", 0)) > 0)
        missing_total = sum(int(payload.get("missing_leg_count", 0)) for payload in payloads)
        negative_cached = sum(1 for payload in payloads if int(payload.get("negative_cached_leg_count", 0)) > 0)
        skew_values = [
            float(payload.get("family_snapshot_time_skew_ms", 0.0) or 0.0)
            for payload in payloads
        ]
        incomplete_failures = sum(
            1
            for payload in payloads
            if (
                (int(payload.get("missing_leg_count", 0)) > 0 or int(payload.get("negative_cached_leg_count", 0)) > 0)
                and not bool(payload.get("qualified"))
            )
        )
        rejection_counter = Counter(
            reason
            for payload in payloads
            for reason in list(payload.get("qualification_rejection_reasons") or [])
        )
        depth_reasons = {
            RejectionReason.INSUFFICIENT_DEPTH.value,
            RejectionReason.ABSOLUTE_DEPTH_BELOW_FLOOR.value,
            RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value,
            RejectionReason.NON_ATOMIC_RISK_TOO_HIGH.value,
        }
        return {
            "families_with_complete_leg_set": complete,
            "families_with_incomplete_leg_set": incomplete,
            "avg_missing_legs_per_family": round(missing_total / max(len(payloads), 1), 6),
            "families_blocked_by_negative_cache": negative_cached,
            "avg_family_snapshot_time_skew_ms": round(sum(skew_values) / max(len(skew_values), 1), 6),
            "families_failing_due_to_incomplete_inputs": incomplete_failures,
            "raw_candidates_rejected_due_to_depth": sum(
                1 for payload in payloads if bool(payload.get("raw_candidate_generated")) and set(payload.get("qualification_rejection_reasons") or []) & depth_reasons
            ),
            "raw_candidates_rejected_due_to_spread": sum(
                1 for payload in payloads if bool(payload.get("raw_candidate_generated")) and RejectionReason.SPREAD_TOO_WIDE.value in set(payload.get("qualification_rejection_reasons") or [])
            ),
            "raw_candidates_rejected_due_to_net_profit": sum(
                1 for payload in payloads if bool(payload.get("raw_candidate_generated")) and (
                    RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value in set(payload.get("qualification_rejection_reasons") or [])
                    or RejectionReason.EDGE_BELOW_THRESHOLD.value in set(payload.get("qualification_rejection_reasons") or [])
                )
            ),
            "raw_candidates_rejected_due_to_concentration": sum(
                1 for payload in payloads if bool(payload.get("raw_candidate_generated")) and RejectionReason.SINGLE_LEG_CONCENTRATION.value in set(payload.get("qualification_rejection_reasons") or [])
            ),
            "top_qualification_failure_reason": rejection_counter.most_common(1)[0][0] if rejection_counter else None,
        }

    def _summarize_neg_risk_family_parity(self) -> dict[str, object]:
        payloads = list(getattr(self, "_neg_risk_family_parity_payloads", []) or [])
        if not payloads:
            return {
                "watched_families_audited": 0,
                "watched_families_fast_vs_broad_mismatch_count": 0,
                "watched_families_true_market_deterioration_count": 0,
                "watched_families_incomplete_input_count": 0,
                "watched_families_time_skew_count": 0,
                "watched_families_negative_cache_side_effect_count": 0,
                "watched_families_depth_failure_count": 0,
                "watched_families_net_profit_failure_count": 0,
            }
        edge_reasons = {
            RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value,
            RejectionReason.EDGE_BELOW_THRESHOLD.value,
        }
        failure_counter = Counter(str(payload.get("failure_class") or "unclear") for payload in payloads)
        return {
            "watched_families_audited": len(payloads),
            "watched_families_fast_vs_broad_mismatch_count": sum(
                1 for payload in payloads if bool(payload.get("qualification_outcome_differs"))
            ),
            "watched_families_true_market_deterioration_count": int(failure_counter.get("true_market_deterioration", 0)),
            "watched_families_incomplete_input_count": int(failure_counter.get("incomplete_family_input", 0)),
            "watched_families_time_skew_count": int(failure_counter.get("time_skewed_family_input", 0)),
            "watched_families_negative_cache_side_effect_count": int(failure_counter.get("negative_cache_side_effect", 0)),
            "watched_families_depth_failure_count": int(failure_counter.get("concentration_depth_degradation", 0)),
            "watched_families_net_profit_failure_count": sum(
                1
                for payload in payloads
                if edge_reasons
                & (
                    set(payload.get("fast_path_result", {}).get("qualification_rejection_reasons") or [])
                    | set(payload.get("broad_path_result", {}).get("qualification_rejection_reasons") or [])
                )
            ),
        }

    def _neg_risk_economic_truth_path(
        self,
        parity_payload: dict[str, object],
    ) -> dict[str, object]:
        if bool(parity_payload.get("qualification_outcome_differs")):
            return dict(parity_payload.get("broad_path_result") or {})
        return dict(parity_payload.get("fast_path_result") or {})

    def _classify_neg_risk_family_current_economics(
        self,
        parity_payload: dict[str, object],
    ) -> str:
        result = self._neg_risk_economic_truth_path(parity_payload)
        rejection_reasons = set(str(reason) for reason in list(result.get("qualification_rejection_reasons") or []))
        if bool(result.get("qualified")):
            return "still_productive_now"
        if RejectionReason.SPREAD_TOO_WIDE.value in rejection_reasons:
            return "spread_too_wide_now"
        if rejection_reasons & {
            RejectionReason.INSUFFICIENT_DEPTH.value,
            RejectionReason.ABSOLUTE_DEPTH_BELOW_FLOOR.value,
            RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value,
            RejectionReason.NON_ATOMIC_RISK_TOO_HIGH.value,
            RejectionReason.SINGLE_LEG_CONCENTRATION.value,
        }:
            return "depth_insufficient_now"
        net_edge = float(result.get("net_edge_cents", 0.0) or 0.0)
        expected_net_profit = float(result.get("expected_net_profit_usd", 0.0) or 0.0)
        if rejection_reasons & {
            RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value,
            RejectionReason.EDGE_BELOW_THRESHOLD.value,
        }:
            if net_edge > 0.0 or expected_net_profit > 0.0:
                return "economically_marginal_now"
            return "edge_vanished_now"
        parity_failure = str(parity_payload.get("failure_class") or "")
        if parity_failure == "edge_vanished_for_real":
            return "edge_vanished_now"
        if parity_failure == "concentration_depth_degradation":
            return "depth_insufficient_now"
        return "unclear"

    def _recommend_neg_risk_family_watchlist_action(
        self,
        *,
        economics_class: str,
        source_tags: list[str],
    ) -> str:
        source_tag_set = {str(tag) for tag in list(source_tags or []) if str(tag)}
        historically_productive = "history_productive" in source_tag_set
        if economics_class == "still_productive_now":
            return "viable_now"
        if economics_class in {"depth_insufficient_now", "spread_too_wide_now"}:
            return "near_miss_now"
        if economics_class == "economically_marginal_now":
            return "economically_marginal_now"
        if economics_class == "edge_vanished_now":
            return "downgrade" if historically_productive else "park"
        return "park"

    def _build_neg_risk_productive_family_economics_report(self) -> list[dict[str, object]]:
        payloads = list(getattr(self, "_neg_risk_family_parity_payloads", []) or [])
        report: list[dict[str, object]] = []
        for payload in payloads:
            source_tags = list(payload.get("source_tags") or [])
            result = self._neg_risk_economic_truth_path(payload)
            economics_class = self._classify_neg_risk_family_current_economics(payload)
            watchlist_recommendation = self._recommend_neg_risk_family_watchlist_action(
                economics_class=economics_class,
                source_tags=source_tags,
            )
            historically_productive = "history_productive" in set(source_tags)
            current_viable = bool(result.get("qualified"))
            report.append(
                {
                    "family_key": payload.get("family_key"),
                    "source_tags": source_tags,
                    "pass_fail_status": "PASS" if current_viable else "FAIL",
                    "gross_edge_cents": result.get("gross_edge_cents"),
                    "net_edge_cents": result.get("net_edge_cents"),
                    "expected_net_profit_usd": result.get("expected_net_profit_usd"),
                    "required_depth_usd": result.get("required_depth_usd"),
                    "available_depth_usd": result.get("available_depth_usd"),
                    "max_spread_observed": result.get("max_spread_observed"),
                    "partial_fill_risk_score": result.get("partial_fill_risk_score"),
                    "non_atomic_execution_risk_score": result.get("non_atomic_execution_risk_score"),
                    "rejection_reason_codes": list(result.get("qualification_rejection_reasons") or []),
                    "economics_class": economics_class,
                    "historically_productive_family": historically_productive,
                    "current_audit_result": economics_class,
                    "remains_viable_now": current_viable,
                    "reason_if_not_viable": None if current_viable else economics_class,
                    "watchlist_recommendation": watchlist_recommendation,
                }
            )
        return report

    def _summarize_neg_risk_productive_family_economics(
        self,
        report: list[dict[str, object]],
    ) -> dict[str, object]:
        if not report:
            return {
                "productive_families_audited": 0,
                "productive_families_currently_viable_count": 0,
                "productive_families_marginal_count": 0,
                "productive_families_edge_vanished_count": 0,
                "productive_families_depth_failure_count": 0,
                "productive_families_spread_failure_count": 0,
            }
        class_counter = Counter(str(row.get("economics_class") or "unclear") for row in report)
        return {
            "productive_families_audited": len(report),
            "productive_families_currently_viable_count": int(class_counter.get("still_productive_now", 0)),
            "productive_families_marginal_count": int(class_counter.get("economically_marginal_now", 0)),
            "productive_families_edge_vanished_count": int(class_counter.get("edge_vanished_now", 0)),
            "productive_families_depth_failure_count": int(class_counter.get("depth_insufficient_now", 0)),
            "productive_families_spread_failure_count": int(class_counter.get("spread_too_wide_now", 0)),
        }

    def _build_neg_risk_selector_refresh_report(self) -> list[dict[str, object]]:
        economics_report = self._build_neg_risk_productive_family_economics_report()
        selector_rows: list[dict[str, object]] = []
        for row in economics_report:
            source_tags = [str(tag) for tag in list(row.get("source_tags") or []) if str(tag)]
            selector_rows.append(
                {
                    "family_key": row.get("family_key"),
                    "source_tags": source_tags,
                    "historically_productive_family": bool(row.get("historically_productive_family")),
                    "current_audit_result": row.get("current_audit_result"),
                    "remains_viable_now": bool(row.get("remains_viable_now")),
                    "economics_class": row.get("economics_class"),
                    "gross_edge_cents": row.get("gross_edge_cents"),
                    "net_edge_cents": row.get("net_edge_cents"),
                    "expected_net_profit_usd": row.get("expected_net_profit_usd"),
                    "required_depth_usd": row.get("required_depth_usd"),
                    "available_depth_usd": row.get("available_depth_usd"),
                    "max_spread_observed": row.get("max_spread_observed"),
                    "rejection_reason_codes": list(row.get("rejection_reason_codes") or []),
                    "watchlist_recommendation": row.get("watchlist_recommendation"),
                }
            )
        if self._neg_risk_condition_monitor_mode_enabled():
            allowed_family_keys = self._neg_risk_condition_monitor_family_keys()
            selector_rows = [
                row
                for row in selector_rows
                if self._normalize_neg_risk_watch_key(row.get("family_key")) in allowed_family_keys
            ]
        return selector_rows

    def _summarize_neg_risk_selector_refresh(
        self,
        report: list[dict[str, object]],
    ) -> dict[str, object]:
        if not report:
            return {
                "selector_refresh_families_audited": 0,
                "selector_refresh_currently_viable_count": 0,
                "selector_refresh_near_miss_count": 0,
                "selector_refresh_marginal_count": 0,
                "selector_refresh_downgrade_count": 0,
                "selector_refresh_park_count": 0,
            }
        recommendation_counter = Counter(str(row.get("watchlist_recommendation") or "park") for row in report)
        return {
            "selector_refresh_families_audited": len(report),
            "selector_refresh_currently_viable_count": int(recommendation_counter.get("viable_now", 0)),
            "selector_refresh_near_miss_count": int(recommendation_counter.get("near_miss_now", 0)),
            "selector_refresh_marginal_count": int(recommendation_counter.get("economically_marginal_now", 0)),
            "selector_refresh_downgrade_count": int(recommendation_counter.get("downgrade", 0)),
            "selector_refresh_park_count": int(recommendation_counter.get("park", 0)),
        }

    def _build_neg_risk_selector_refresh_watchlist(
        self,
        report: list[dict[str, object]],
    ) -> dict[str, list[dict[str, object]]]:
        grouped: dict[str, list[dict[str, object]]] = {
            "viable_now": [],
            "near_miss_now": [],
            "economically_marginal_now": [],
            "downgrade": [],
            "park": [],
        }
        target_keys = {
            "viable_now": "viable_now",
            "near_miss_now": "near_miss_now",
            "economically_marginal_now": "economically_marginal_now",
            "downgrade": "downgrade",
            "park": "park",
        }
        for row in report:
            target_key = target_keys.get(str(row.get("watchlist_recommendation") or "park"), "park")
            grouped[target_key].append(
                {
                    "family_key": row.get("family_key"),
                    "source_tags": list(row.get("source_tags") or []),
                    "economics_class": row.get("economics_class"),
                    "rejection_reason_codes": list(row.get("rejection_reason_codes") or []),
                }
            )
        for key, values in grouped.items():
            grouped[key] = sorted(values, key=lambda item: str(item.get("family_key") or ""))
        return grouped

    def _run_neg_risk_scan(
        self,
        events: list,
        cycle_started: datetime,
        book_cache: dict[str, object],
        *,
        changed_market_slugs: set[str] | None = None,
    ) -> None:
        from src.scanner.neg_risk import build_eligible_neg_risk_event_groups
        active_strategies = self._active_neg_risk_strategies()
        if not active_strategies:
            return
        try:
            event_groups = build_eligible_neg_risk_event_groups(events)
        except Exception as exc:
            self._record_event(
                "neg_risk_scan_error",
                Severity.ERROR,
                f"Neg-risk event grouping error: {exc}",
                {"error": str(exc)},
            )
            return
        event_groups, watchlist_diagnostics = self._reconcile_neg_risk_watchlist_groups(event_groups)
        if self._current_summary is not None:
            self._current_summary.metadata["neg_risk_watchlist_reconciliation_diagnostics"] = watchlist_diagnostics
        watch_source_tags_by_family = self._neg_risk_watch_source_tags_by_family(watchlist_diagnostics)
        self._cycle_metrics["neg_risk_event_groups_available"] = len(event_groups)
        self._cycle_metrics["neg_risk_audit_watchlist_matches"] = sum(
            1
            for diagnostic in watchlist_diagnostics
            if str(diagnostic.get("match_status") or "") in {"active_discovery", "bounded_rescan_injected", "bounded_rescan_matched"}
        )
        selected_event_groups, family_metrics, naturally_selected_keys, audit_forced_keys = self._select_neg_risk_event_groups_for_scan(
            event_groups=event_groups,
            changed_market_slugs=changed_market_slugs,
            cycle_started=cycle_started,
        )
        self._cycle_metrics["neg_risk_audit_forced_families"] = len(audit_forced_keys)
        self._cycle_metrics["families_considered"] = int(family_metrics.get("families_considered", 0))
        self._cycle_metrics["families_recomputed_due_to_change"] = int(
            family_metrics.get("families_recomputed_due_to_change", 0)
        )
        self._cycle_metrics["families_recomputed_due_to_due_interval"] = int(
            family_metrics.get("families_recomputed_due_to_due_interval", 0)
        )
        self._cycle_metrics["pinned_productive_families_evaluated"] = int(
            family_metrics.get("pinned_productive_families_evaluated", 0)
        )
        self._cycle_metrics["recent_near_miss_families_evaluated"] = int(
            family_metrics.get("recent_near_miss_families_evaluated", 0)
        )
        self._cycle_metrics["family_backstop_recompute_count"] = int(
            family_metrics.get("family_backstop_recompute_count", 0)
        )
        self._cycle_metrics["avg_markets_per_family_recompute"] = float(
            family_metrics.get("avg_markets_per_family_recompute", 0.0)
        )
        self._cycle_metrics["neg_risk_family_coverage_rate"] = float(
            family_metrics.get("neg_risk_family_coverage_rate", 0.0)
        )
        self._cycle_metrics["raw_candidates_by_family_per_cycle"] = dict(
            family_metrics.get("raw_candidates_by_family_per_cycle", {})
        )
        self._cycle_metrics["neg_risk_family_qualification_audit_count"] = len(
            getattr(self, "_neg_risk_family_audit_payloads", []) or []
        )
        if not selected_event_groups:
            neg_risk_audit_summary = self._summarize_neg_risk_family_audits()
            neg_risk_parity_summary = self._summarize_neg_risk_family_parity()
            neg_risk_economics_report = self._build_neg_risk_productive_family_economics_report()
            neg_risk_economics_summary = self._summarize_neg_risk_productive_family_economics(neg_risk_economics_report)
            selector_refresh_report = self._build_neg_risk_selector_refresh_report()
            selector_refresh_summary = self._summarize_neg_risk_selector_refresh(selector_refresh_report)
            selector_refresh_watchlist = self._build_neg_risk_selector_refresh_watchlist(selector_refresh_report)
            self._cycle_metrics.update(neg_risk_audit_summary)
            self._cycle_metrics.update(neg_risk_parity_summary)
            self._cycle_metrics.update(neg_risk_economics_summary)
            self._cycle_metrics.update(selector_refresh_summary)
            if self._current_summary is not None:
                self._current_summary.metadata["neg_risk_family_audit_summary"] = neg_risk_audit_summary
                self._current_summary.metadata["neg_risk_family_qualification_parity_summary"] = neg_risk_parity_summary
                self._current_summary.metadata["neg_risk_productive_family_economics_report"] = neg_risk_economics_report
                self._current_summary.metadata["neg_risk_productive_family_economics_summary"] = neg_risk_economics_summary
                self._current_summary.metadata["neg_risk_selector_refresh_report"] = selector_refresh_report
                self._current_summary.metadata["neg_risk_selector_refresh_summary"] = selector_refresh_summary
                self._current_summary.metadata["neg_risk_selector_refresh_watchlist"] = selector_refresh_watchlist
                if self._neg_risk_condition_monitor_mode_enabled():
                    self._current_summary.metadata["neg_risk_condition_monitor_report"] = selector_refresh_report
                    self._current_summary.metadata["neg_risk_condition_monitor_summary"] = selector_refresh_summary
            return

        max_notional = self.config.paper.max_notional_per_arb

        # Collect all YES token IDs across selected event groups and fetch books once.
        neg_risk_token_ids = list({
            m["yes_token_id"]
            for g in selected_event_groups
            for m in g.get("markets", [])
            if m.get("yes_token_id")
        })
        books_by_token: dict[str, object] = {
            token_id: book_cache[token_id]
            for token_id in neg_risk_token_ids
            if token_id in book_cache
        }
        missing_token_ids = [token_id for token_id in neg_risk_token_ids if token_id not in books_by_token]
        if missing_token_ids:
            fetched_books = self.clob.fetch_books_batch(missing_token_ids)
            self._record_batch_request("clob_books")
            self._register_book_fetch(len(fetched_books))
            books_by_token.update(fetched_books)
            for token_id, book in fetched_books.items():
                book_cache[token_id] = book
                self._save_raw_snapshot("clob", token_id, book.model_dump(mode="json"), cycle_started)

        for event_group in selected_event_groups:
            family_slug = str(event_group.get("event_slug") or event_group.get("event_id") or "")
            market_slugs = {
                str(market.get("slug") or "")
                for market in event_group.get("markets", [])
                if str(market.get("slug") or "")
            }
            self.market_universe.record_family_recompute(
                family_slug=family_slug,
                observed_at=cycle_started,
                event_title=event_group.get("event_title"),
                market_slugs=market_slugs,
            )

        for strategy in active_strategies:
            for event_group in selected_event_groups:
                family_key = self._neg_risk_family_key(event_group)
                source_tags = self._neg_risk_family_source_tags(
                    family_key,
                    watch_source_tags_by_family.get(family_key, []),
                    include_current_discovery_fallback=True,
                )
                audit_payload = self._build_neg_risk_family_input_audit(
                    event_group=event_group,
                    books_by_token=books_by_token,
                    cycle_started=cycle_started,
                    naturally_selected=family_key in naturally_selected_keys,
                    audit_forced=family_key in audit_forced_keys,
                    source_tags=source_tags,
                )
                try:
                    raw_candidate, _audit = strategy.detect_with_audit(event_group, books_by_token, max_notional)
                    if raw_candidate is None:
                        audit_payload["raw_detection_failure_reason"] = (_audit or {}).get("failure_reason")
                        self._record_neg_risk_family_audit(audit_payload)
                        if source_tags:
                            broad_payload = self._evaluate_neg_risk_family_path_for_audit(
                                event_group=event_group,
                                books_by_token=self._fetch_neg_risk_family_broad_books(event_group=event_group),
                                cycle_started=cycle_started,
                                source_tags=source_tags,
                                naturally_selected=family_key in naturally_selected_keys,
                                audit_forced=family_key in audit_forced_keys,
                            )
                            self._record_neg_risk_family_parity(
                                self._build_neg_risk_family_parity_payload(
                                    fast_path_result=dict(audit_payload),
                                    broad_path_result=broad_payload,
                                )
                            )
                        continue

                    raw_candidate = self._decorate_raw_candidate(raw_candidate)
                    self._record_raw_candidate(raw_candidate)
                    raw_candidates_by_family = self._cycle_metrics.get("raw_candidates_by_family_per_cycle")
                    if isinstance(raw_candidates_by_family, dict) and family_key:
                        raw_candidates_by_family[family_key] = int(raw_candidates_by_family.get(family_key, 0)) + 1

                    account_snapshot = self.paper_ledger.snapshot(ts=cycle_started)
                    candidate = self._qualify_and_rank_candidate(
                        raw_candidate=raw_candidate,
                        books_by_token=books_by_token,
                        account_snapshot=account_snapshot,
                        audit_sink=audit_payload,
                    )
                    if candidate is None:
                        self._record_neg_risk_family_audit(audit_payload)
                        if source_tags:
                            broad_payload = self._evaluate_neg_risk_family_path_for_audit(
                                event_group=event_group,
                                books_by_token=self._fetch_neg_risk_family_broad_books(event_group=event_group),
                                cycle_started=cycle_started,
                                source_tags=source_tags,
                                naturally_selected=family_key in naturally_selected_keys,
                                audit_forced=family_key in audit_forced_keys,
                            )
                            self._record_neg_risk_family_parity(
                                self._build_neg_risk_family_parity_payload(
                                    fast_path_result=dict(audit_payload),
                                    broad_path_result=broad_payload,
                                )
                            )
                        continue

                    self._record_qualified_candidate(candidate)
                    audit_payload["qualified"] = True
                    self._record_neg_risk_family_audit(audit_payload)
                    if source_tags:
                        broad_payload = self._evaluate_neg_risk_family_path_for_audit(
                            event_group=event_group,
                            books_by_token=self._fetch_neg_risk_family_broad_books(event_group=event_group),
                            cycle_started=cycle_started,
                            source_tags=source_tags,
                            naturally_selected=family_key in naturally_selected_keys,
                            audit_forced=family_key in audit_forced_keys,
                        )
                        self._record_neg_risk_family_parity(
                            self._build_neg_risk_family_parity_payload(
                                fast_path_result=dict(audit_payload),
                                broad_path_result=broad_payload,
                            )
                        )
                    self._current_summary.candidates_generated += 1
                    for market_slug in candidate.market_slugs:
                        self._current_summary.market_counts[market_slug] += 1
                    self._current_summary.opportunity_type_counts[candidate.kind] += 1
                    self.store.save_candidate(candidate)
                    self._ab_sidecar.observe(raw_candidate, candidate)
                    self.opportunity_store.save(raw_candidate.to_legacy_opportunity())
                    try:
                        decision = self.risk.evaluate(candidate, account_snapshot)
                    except Exception as exc:
                        self._record_rejection(
                            stage="risk",
                            reason_code=RejectionReason.RISK_ENGINE_ERROR.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "strategy_family": candidate.strategy_family.value,
                                "error": str(exc),
                            },
                        )
                        continue

                    self.store.save_risk_decision(decision)

                    if decision.status.name in {"BLOCKED", "HALTED"}:
                        self._current_summary.risk_rejected += 1
                        for reason_code in decision.reason_codes:
                            self._record_rejection(
                                stage="risk",
                                reason_code=reason_code,
                                candidate_id=candidate.candidate_id,
                                metadata={
                                    "strategy_family": candidate.strategy_family.value,
                                },
                            )
                        self.logger.warning(
                            "candidate blocked by risk",
                            extra={"payload": {"candidate_id": candidate.candidate_id, "reasons": decision.reason_codes}},
                        )
                        continue

                    self._current_summary.risk_accepted += 1
                    if self._blocked_by_human_review_gate(
                        decision,
                        candidate.candidate_id,
                        {"strategy_family": candidate.strategy_family.value},
                    ):
                        continue
                    try:
                        intents, reports = self._submit_candidate_orders(
                            candidate,
                            books_by_token=books_by_token,
                            shares=candidate.sizing_hint_shares,
                        )
                    except Exception as exc:
                        self._record_rejection(
                            stage="execution",
                            reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                            candidate_id=candidate.candidate_id,
                            metadata={
                                "strategy_family": candidate.strategy_family.value,
                                "error": str(exc),
                            },
                        )
                        continue

                    self._current_summary.paper_orders_created += len(intents)
                    for intent in intents:
                        self.store.save_order_intent(intent)
                    for report in reports:
                        self.store.save_execution_report(report)
                        self._record_report_stats(report)
                except Exception as exc:
                    if audit_payload not in self._neg_risk_family_audit_payloads:
                        self._record_neg_risk_family_audit(audit_payload)
                    self._record_event(
                        "neg_risk_scan_error",
                        Severity.ERROR,
                        f"Neg-risk scan error: {exc}",
                        {
                            "error": str(exc),
                            "family_key": family_key,
                            "strategy_family": strategy.strategy_family.value,
                        },
                    )
                    raise

        neg_risk_audit_summary = self._summarize_neg_risk_family_audits()
        neg_risk_parity_summary = self._summarize_neg_risk_family_parity()
        neg_risk_economics_report = self._build_neg_risk_productive_family_economics_report()
        neg_risk_economics_summary = self._summarize_neg_risk_productive_family_economics(neg_risk_economics_report)
        selector_refresh_report = self._build_neg_risk_selector_refresh_report()
        selector_refresh_summary = self._summarize_neg_risk_selector_refresh(selector_refresh_report)
        selector_refresh_watchlist = self._build_neg_risk_selector_refresh_watchlist(selector_refresh_report)
        self._cycle_metrics.update(neg_risk_audit_summary)
        self._cycle_metrics.update(neg_risk_parity_summary)
        self._cycle_metrics.update(neg_risk_economics_summary)
        self._cycle_metrics.update(selector_refresh_summary)
        self._cycle_metrics["neg_risk_family_qualification_audit_count"] = len(
            getattr(self, "_neg_risk_family_audit_payloads", []) or []
        )
        if self._current_summary is not None:
            self._current_summary.metadata["neg_risk_family_audit_summary"] = neg_risk_audit_summary
            self._current_summary.metadata["neg_risk_family_qualification_parity_summary"] = neg_risk_parity_summary
            self._current_summary.metadata["neg_risk_productive_family_economics_report"] = neg_risk_economics_report
            self._current_summary.metadata["neg_risk_productive_family_economics_summary"] = neg_risk_economics_summary
            self._current_summary.metadata["neg_risk_selector_refresh_report"] = selector_refresh_report
            self._current_summary.metadata["neg_risk_selector_refresh_summary"] = selector_refresh_summary
            self._current_summary.metadata["neg_risk_selector_refresh_watchlist"] = selector_refresh_watchlist
            if self._neg_risk_condition_monitor_mode_enabled():
                self._current_summary.metadata["neg_risk_condition_monitor_report"] = selector_refresh_report
                self._current_summary.metadata["neg_risk_condition_monitor_summary"] = selector_refresh_summary

    def _is_strategy_family_targeted(self, strategy_family: str) -> bool:
        targets = self._target_strategy_families()
        if targets is None:
            return True
        return strategy_family in targets

    def _parameter_snapshot(self) -> dict[str, float | int | bool]:
        return {
            "min_edge_cents": self.config.opportunity.min_edge_cents,
            "fee_buffer_cents": self.config.opportunity.fee_buffer_cents,
            "slippage_buffer_cents": self.config.opportunity.slippage_buffer_cents,
            "min_depth_multiple": self.config.opportunity.min_depth_multiple,
            "max_spread_cents": self.config.opportunity.max_spread_cents,
            "min_net_profit_usd": self.config.opportunity.min_net_profit_usd,
            "max_partial_fill_risk": self.config.opportunity.max_partial_fill_risk,
            "max_non_atomic_risk": self.config.opportunity.max_non_atomic_risk,
            "max_notional_per_arb": self.config.paper.max_notional_per_arb,
        }

    def _runtime_metadata(self) -> dict[str, object]:
        metadata = self._build_experiment_metadata()
        if self._current_run_id is not None:
            metadata["run_id"] = self._current_run_id
        return metadata

    def _save_raw_snapshot(self, source: str, entity_id: str, payload, ts: datetime) -> None:
        self.store.save_raw_snapshot(source, entity_id, payload, ts)
        if self._current_summary is not None:
            self._current_summary.snapshots_stored += 1

    def _record_report_stats(self, report) -> None:
        if self._current_summary is None:
            return
        if report.status == OrderStatus.FILLED:
            self._current_summary.fills += 1
        elif report.status == OrderStatus.PARTIAL:
            self._current_summary.partial_fills += 1
            self._current_summary.cancellations += 1
            self._record_rejection(
                stage="execution",
                reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                candidate_id=None,
                metadata={"intent_id": report.intent_id, "status": report.status.value},
            )
        elif report.status == OrderStatus.CANCELED:
            self._current_summary.cancellations += 1
            self._record_rejection(
                stage="execution",
                reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                candidate_id=None,
                metadata={"intent_id": report.intent_id, "status": report.status.value},
            )
        elif report.status == OrderStatus.REJECTED:
            self._record_rejection(
                stage="execution",
                reason_code=RejectionReason.PAPER_ORDER_REJECTED.value,
                candidate_id=None,
                metadata={"intent_id": report.intent_id},
            )

    def _record_rejection(self, stage: str, reason_code: str, candidate_id: str | None, metadata: dict) -> None:
        metadata = {**self._runtime_metadata(), **metadata}
        if self._current_summary is not None:
            self._current_summary.rejection_reason_counts[reason_code] += 1
            strategy_family = metadata.get("strategy_family")
            if strategy_family:
                self._current_summary.rejection_counts_by_family[str(strategy_family)][reason_code] += 1
        if self._current_run_id is None:
            return
        self.store.save_rejection_event(
            RejectionEvent(
                run_id=self._current_run_id,
                candidate_id=candidate_id,
                stage=stage,
                reason_code=reason_code,
                metadata=metadata,
                ts=datetime.now(timezone.utc),
            )
        )

    # Reason codes that signal a pre-candidate structural check failure.
    _PRE_CANDIDATE_REASON_CODES = frozenset({
        RejectionReason.EMPTY_ASKS.value,
    })

    def _record_invalid_orderbook(self, stage: str, validation, metadata: dict) -> None:
        reason_code = validation.reason_code or RejectionReason.INVALID_ORDERBOOK.value
        failure_class = orderbook_failure_class(reason_code)
        debug_payload = {
            **self._runtime_metadata(),
            **metadata,
            **validation.to_debug_payload(),
            "failure_class": failure_class,
            "reason_code": reason_code,
            "stage": stage,
        }
        if stage == "candidate_filter" and self._current_summary is not None:
            failure_stage = str(
                debug_payload.get("failure_stage")
                or (
                    "pre_candidate_precheck"
                    if reason_code in self._PRE_CANDIDATE_REASON_CODES
                    else "validate"
                )
            )
            strategy_family = str(metadata.get("strategy_family") or "")
            self._current_summary.candidate_filter_failure_stage_counts[failure_stage] += 1
            if strategy_family:
                self._current_summary.candidate_filter_failure_stage_counts_by_family[strategy_family][failure_stage] += 1
        self._update_liquidity_skip_state(
            market_slug=str(metadata.get("market_slug") or ""),
            side=str(metadata.get("side") or ""),
            required_action=str(validation.required_action),
            reason_code=reason_code,
            stage=stage,
        )
        self._append_invalid_orderbook_export(debug_payload)
        self.logger.warning("invalid orderbook rejected", extra={"payload": debug_payload})
        self._record_rejection(
            stage=stage,
            reason_code=reason_code,
            candidate_id=metadata.get("candidate_id"),
            metadata=debug_payload,
        )

    def _record_pre_candidate_pricing_failure(self, audit_payload: dict[str, object]) -> None:
        reason_code = str(audit_payload.get("failure_reason") or "PRE_CANDIDATE_PRICING_FAILED")
        debug_payload = {
            **self._runtime_metadata(),
            **audit_payload,
        }
        self.logger.info("single-market pre-candidate pricing rejected", extra={"payload": debug_payload})
        self._record_rejection(
            stage="candidate_filter",
            reason_code=reason_code,
            candidate_id=None,
            metadata=debug_payload,
        )

    def _record_cross_market_pre_candidate_failure(self, audit_payload: dict[str, object]) -> None:
        reason_code = str(audit_payload.get("failure_reason") or "PRE_CANDIDATE_RELATION_FAILED")
        debug_payload = {
            **self._runtime_metadata(),
            **audit_payload,
        }
        self.logger.info("cross-market pre-candidate relation rejected", extra={"payload": debug_payload})
        self._record_rejection(
            stage="candidate_filter",
            reason_code=reason_code,
            candidate_id=None,
            metadata=debug_payload,
        )

    def _record_event(self, event_type: str, severity: Severity, message: str, payload: dict) -> None:
        self.store.save_system_event(
            SystemEvent(
                event_type=event_type,
                severity=severity,
                message=message,
                payload=payload,
                ts=datetime.now(timezone.utc),
            )
        )
        if self._current_summary is not None and severity in {Severity.WARNING, Severity.ERROR, Severity.CRITICAL}:
            self._current_summary.system_errors += 1
        log_method = getattr(self.logger, severity.value.lower(), logging.info)
        log_method(message, extra={"payload": payload})

    def _register_book_fetch(self, count: int = 1) -> None:
        if self._current_summary is not None:
            self._current_summary.books_fetched += int(count)

    def _record_book_validation_result(self, validation) -> None:
        if self._current_summary is None:
            return
        failure_class = orderbook_failure_class(validation.reason_code)
        if validation.passed or failure_class == FEASIBILITY_FAILURE:
            self._current_summary.books_structurally_valid += 1
        if validation.passed:
            self._current_summary.books_execution_feasible += 1

    def _record_strategy_family_market_considered(self, strategy_family: str, market_slug: str) -> None:
        if self._current_summary is not None:
            self._current_summary.record_strategy_family_market_considered(strategy_family, market_slug)

    def _record_strategy_family_book_fetch(self, strategy_family: str) -> None:
        if self._current_summary is not None:
            self._current_summary.record_strategy_family_book_fetch(strategy_family)

    def _record_strategy_family_book_validation_result(self, strategy_family: str, validation) -> None:
        if self._current_summary is None:
            return
        failure_class = orderbook_failure_class(validation.reason_code)
        self._current_summary.record_strategy_family_book_validation(
            strategy_family,
            structurally_valid=bool(validation.passed or failure_class == FEASIBILITY_FAILURE),
            execution_feasible=bool(validation.passed),
        )

    def _should_skip_market_leg(self, market_slug: str, side: str, required_action: str) -> bool:
        state = self._liquidity_skip_state.get((market_slug, side, required_action))
        if not state:
            return False
        return self._run_sequence < int(state.get("next_allowed_run_sequence", 0))

    def _update_liquidity_skip_state(
        self,
        market_slug: str,
        side: str,
        required_action: str,
        reason_code: str,
        stage: str,
    ) -> None:
        if stage != "candidate_filter" or not market_slug or not side:
            return

        key = (market_slug, side, required_action)
        if reason_code == RejectionReason.EMPTY_ASKS.value:
            state = self._liquidity_skip_state.get(key, {"consecutive_failures": 0, "next_allowed_run_sequence": 0})
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
            if state["consecutive_failures"] >= self._empty_asks_skip_threshold:
                state["next_allowed_run_sequence"] = self._run_sequence + self._empty_asks_skip_cooldown_runs
            self._liquidity_skip_state[key] = state
            return

        self._liquidity_skip_state.pop(key, None)

    def _prepare_invalid_orderbook_export_path(self, started_ts: datetime) -> Path:
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        run_id = self._current_run_id or "unknown-run"
        filename = f"{started_ts.strftime('%Y%m%dT%H%M%S')}_{run_id}_invalid_orderbooks.jsonl"
        return self.debug_output_dir / filename

    def _append_invalid_orderbook_export(self, payload: dict[str, object]) -> None:
        if self._invalid_orderbook_export_path is None:
            return
        with self._invalid_orderbook_export_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    def _final_position_state(self, reason_code: str) -> PositionState:
        if reason_code == "RUN_END_FLATTEN" or reason_code == "MANUAL_FORCE_FLATTEN":
            return PositionState.FORCE_CLOSED
        if reason_code == "MAX_HOLDING_AGE":
            return PositionState.EXPIRED
        return PositionState.CLOSED

    # -----------------------------------------------------------------------
    # Maker-MM wide-scan integration (MAKER_REWARDED_EVENT_MM_V1)
    # Opt-in only: only runs when "maker_rewarded_event_mm_v1" is explicitly
    # listed in the campaign_target_strategy_families experiment context.
    # Uses local G5/G6 config adapters — does not modify self.config or
    # the shared feasibility/risk/sizer instances used by other scan paths.
    # -----------------------------------------------------------------------

    # Maker-MM controls now live in self.config.maker_mm (MakerMMConfig).
    # Experiment-context keys override config values at run-level:
    #   maker_mm_cohort, maker_mm_min_edge, maker_mm_g6_margin

    def _run_maker_mm_scan(self, cycle_started: datetime, book_cache: dict) -> None:
        from src.scanner.wide_scan_maker_mm import (
            fetch_wide_scan_maker_mm_candidates,
            compute_wide_scan_ev,
            build_wide_scan_raw_candidate,
        )

        # Resolve run-level controls: experiment context overrides config defaults.
        cohort    = self._experiment_context.get("maker_mm_cohort",    list(self.config.maker_mm.cohort))
        min_edge  = self._experiment_context.get("maker_mm_min_edge",  self.config.maker_mm.min_edge_cents)
        g6_margin = self._experiment_context.get("maker_mm_g6_margin", self.config.maker_mm.g6_margin)

        # G5 config adapters: path-level, shared across all cohort markets.
        mm_opp_cfg     = OpportunityConfig(min_edge_cents=min_edge)
        mm_feasibility = ExecutionFeasibilityEvaluator(mm_opp_cfg)
        mm_ranker      = OpportunityRanker(mm_opp_cfg)

        try:
            candidates = fetch_wide_scan_maker_mm_candidates(
                self.config.market_data.gamma_host,
                cohort,
            )
        except Exception as exc:
            self._record_event(
                "maker_mm_scan_error",
                Severity.ERROR,
                f"maker_mm market fetch failed: {exc}",
                {"error": str(exc)},
            )
            return

        for m in candidates:
            try:
                yes_book = self.clob.get_book(m["yes_token_id"])
                no_book  = self.clob.get_book(m["no_token_id"])
            except Exception as exc:
                self._record_event(
                    "maker_mm_scan_error",
                    Severity.WARNING,
                    f"book fetch failed for {m.get('market_slug')}: {exc}",
                    {"market_slug": m.get("market_slug"), "error": str(exc)},
                )
                continue

            book_cache[m["yes_token_id"]] = yes_book
            book_cache[m["no_token_id"]]  = no_book
            fetch_ts = datetime.now(timezone.utc)
            self._save_raw_snapshot("clob", m["yes_token_id"], yes_book.model_dump(mode="json"), fetch_ts)
            self._save_raw_snapshot("clob", m["no_token_id"],  no_book.model_dump(mode="json"), fetch_ts)

            # Update m with live book prices before EV computation.
            if yes_book.bids and yes_book.asks:
                m["best_bid"] = yes_book.bids[0].price
                m["best_ask"] = yes_book.asks[0].price

            ev = compute_wide_scan_ev(m)
            if ev["total_ev"] <= 0:
                continue

            books_by_token = {
                m["yes_token_id"]: yes_book,
                m["no_token_id"]:  no_book,
            }

            raw = build_wide_scan_raw_candidate(m, ev, m["yes_token_id"], m["no_token_id"])
            raw = self._decorate_raw_candidate(raw)
            self._record_raw_candidate(raw)

            account_snapshot = self.paper_ledger.snapshot(ts=cycle_started)

            # G6 config: conditional per-market based on rewards_min_size.
            g6_cap       = max(self.config.maker_mm.default_notional_usd,
                               round(m["rewards_min_size"] * g6_margin))
            mm_paper_cfg = PaperConfig(max_notional_per_arb=g6_cap)
            mm_risk_cfg  = RiskConfig(max_order_notional_usd=g6_cap)
            mm_sizer     = DepthCappedSizer(mm_paper_cfg, mm_opp_cfg)
            mm_risk      = RiskManager(mm_risk_cfg, mm_opp_cfg, self.config.execution)

            # Qualify / rank / size with local adapted components.
            decision = mm_feasibility.qualify(raw, books_by_token)
            if self._current_auditor is not None:
                self._current_auditor.record(decision)
            if not decision.passed or decision.executable_candidate is None:
                for code in decision.reason_codes:
                    self._record_rejection(
                        stage="qualification",
                        reason_code=code,
                        candidate_id=raw.candidate_id,
                        metadata={
                            "strategy_family": raw.strategy_family.value,
                            "market_slugs": raw.market_slugs,
                        },
                    )
                continue

            ranked  = mm_ranker.rank(decision.executable_candidate)
            sizing  = mm_sizer.size(ranked, account_snapshot)

            if sizing.notional_usd <= 1e-9 or sizing.shares <= 1e-9:
                self._record_rejection(
                    stage="qualification",
                    reason_code=RejectionReason.ORDER_SIZE_LIMIT.value,
                    candidate_id=raw.candidate_id,
                    metadata={
                        "strategy_family": raw.strategy_family.value,
                        "market_slugs": raw.market_slugs,
                        "sizing": {"notional_usd": sizing.notional_usd, "shares": sizing.shares},
                    },
                )
                continue

            # Mirror the leg-scaling logic in _qualify_and_rank_candidate.
            scale       = sizing.notional_usd / max(ranked.target_notional_usd, 1e-9)
            scaled_legs = [leg.model_copy(update={"required_shares": sizing.shares}) for leg in ranked.legs]
            ranked = ranked.model_copy(
                update={
                    "target_notional_usd":       sizing.notional_usd,
                    "expected_payout":           round(ranked.expected_payout * scale, 6),
                    "required_shares":           sizing.shares,
                    "sizing_hint_usd":           sizing.notional_usd,
                    "sizing_hint_shares":         sizing.shares,
                    "legs":                      scaled_legs,
                    "metadata": {
                        **ranked.metadata,
                        **self._runtime_metadata(),
                        "strategy_family":  ranked.strategy_family.value,
                        "execution_mode":   ranked.execution_mode,
                        "research_only":    ranked.research_only,
                        "sizing_decision":  {
                            "notional_usd": sizing.notional_usd,
                            "shares":       sizing.shares,
                            "reason":       sizing.reason,
                            "metadata":     sizing.metadata,
                        },
                    },
                }
            )

            self._record_qualified_candidate(ranked)
            if self._current_summary is not None:
                self._current_summary.candidates_generated += 1
                self._current_summary.opportunity_type_counts[ranked.kind] += 1
            self.store.save_candidate(ranked)
            self._ab_sidecar.observe(raw, ranked)  # A+B sidecar — observation only
            self.opportunity_store.save(raw.to_legacy_opportunity())

            # Risk evaluation with local adapted config.
            try:
                risk_decision = mm_risk.evaluate(ranked, account_snapshot)
            except Exception as exc:
                self._record_rejection(
                    stage="risk",
                    reason_code=RejectionReason.RISK_ENGINE_ERROR.value,
                    candidate_id=ranked.candidate_id,
                    metadata={
                        "market_slugs":    raw.market_slugs,
                        "error":           str(exc),
                        "strategy_family": ranked.strategy_family.value,
                    },
                )
                continue

            self.store.save_risk_decision(risk_decision)

            if risk_decision.status.name in {"BLOCKED", "HALTED"}:
                if self._current_summary is not None:
                    self._current_summary.risk_rejected += 1
                for code in risk_decision.reason_codes:
                    self._record_rejection(
                        stage="risk",
                        reason_code=code,
                        candidate_id=ranked.candidate_id,
                        metadata={
                            "market_slugs":    raw.market_slugs,
                            "strategy_family": ranked.strategy_family.value,
                        },
                    )
                continue

            if self._current_summary is not None:
                self._current_summary.risk_accepted += 1
            if self._blocked_by_human_review_gate(
                risk_decision,
                ranked.candidate_id,
                {"market_slugs": raw.market_slugs, "strategy_family": ranked.strategy_family.value},
            ):
                continue

            try:
                intents, reports = self._submit_candidate_orders(
                    ranked, books_by_token, ranked.sizing_hint_shares
                )
            except Exception as exc:
                self._record_rejection(
                    stage="execution",
                    reason_code=RejectionReason.EXECUTION_SIMULATION_FAILED.value,
                    candidate_id=ranked.candidate_id,
                    metadata={
                        "market_slugs":    raw.market_slugs,
                        "error":           str(exc),
                        "strategy_family": ranked.strategy_family.value,
                    },
                )
                continue

            if self._current_summary is not None:
                self._current_summary.paper_orders_created += len(intents)
            for intent in intents:
                self.store.save_order_intent(intent)
            for report in reports:
                self.store.save_execution_report(report)
                self._record_report_stats(report)

            for report in reports:
                if report.filled_size > 0 and report.position_id:
                    position = self.paper_ledger.position_records.get(report.position_id)
                    if position is not None:
                        self.store.save_position_event(
                            position_id=position.position_id,
                            candidate_id=position.candidate_id,
                            event_type="position_opened",
                            symbol=position.symbol,
                            market_slug=position.market_slug,
                            state=position.state.value,
                            reason_code=None,
                            payload={
                                "filled_size":    report.filled_size,
                                "avg_fill_price": report.avg_fill_price,
                                "intent_id":      report.intent_id,
                            },
                            ts=report.ts,
                        )

            snapshot = self.paper_ledger.snapshot()
            self.store.save_account_snapshot(snapshot)

            if any(r.status != OrderStatus.FILLED for r in reports):
                self._record_event(
                    "paper_pair_incomplete_fill",
                    Severity.WARNING,
                    f"Incomplete maker_mm fill for {m.get('market_slug')}",
                    {
                        "candidate_id": ranked.candidate_id,
                        "reports":      [r.model_dump(mode="json") for r in reports],
                    },
                )

    def _reset_quote_guard_metrics(self) -> None:
        self._quote_guard_metrics = {
            "profitability_cancel_count": 0,
            "stale_quote_cancel_count": 0,
            "price_guard_cancel_count": 0,
            "inventory_skew_suppressed_quote_count": 0,
            "quote_guard_reason_counts": Counter(),
            "expected_net_edge_post_sum": 0.0,
            "expected_net_edge_post_count": 0,
            "expected_net_edge_fill_sum": 0.0,
            "expected_net_edge_fill_count": 0,
            "adverse_fill_count": 0,
            "adverse_fill_30s_count": 0,
            "fill_to_mid_move_bps_sum": 0.0,
            "fill_to_mid_move_bps_count": 0,
            "fills_by_quote_age_bucket": Counter(),
        }

    def _record_quote_post_metric(self, expected_net_edge: float) -> None:
        self._quote_guard_metrics["expected_net_edge_post_sum"] = float(self._quote_guard_metrics["expected_net_edge_post_sum"]) + expected_net_edge
        self._quote_guard_metrics["expected_net_edge_post_count"] = int(self._quote_guard_metrics["expected_net_edge_post_count"]) + 1

    def _record_quote_cancel_metric(self, cancel_reason: str) -> None:
        reason_counts = self._quote_guard_metrics["quote_guard_reason_counts"]
        if isinstance(reason_counts, Counter):
            reason_counts[cancel_reason] += 1
        if cancel_reason == "profitability":
            self._quote_guard_metrics["profitability_cancel_count"] = int(self._quote_guard_metrics["profitability_cancel_count"]) + 1
        elif cancel_reason == "stale_quote":
            self._quote_guard_metrics["stale_quote_cancel_count"] = int(self._quote_guard_metrics["stale_quote_cancel_count"]) + 1
        elif cancel_reason == "price_guard":
            self._quote_guard_metrics["price_guard_cancel_count"] = int(self._quote_guard_metrics["price_guard_cancel_count"]) + 1
        elif cancel_reason == "inventory_suppressed":
            self._quote_guard_metrics["inventory_skew_suppressed_quote_count"] = int(self._quote_guard_metrics["inventory_skew_suppressed_quote_count"]) + 1

    def _record_quote_fill_metrics(self, intent: OrderIntent, fill_price: float | None, ts: datetime) -> None:
        if fill_price is None:
            return
        guard_meta = (intent.metadata or {}).get("quote_guard") or {}
        if not guard_meta:
            return
        try:
            book = self.clob.get_book(intent.token_id)
        except Exception:
            return
        qualified_gross_edge_cents = guard_meta.get("qualified_gross_edge_cents")
        qualified_fee_impact_cents = guard_meta.get("qualified_fee_impact_cents")
        qualified_slippage_cents = guard_meta.get("qualified_slippage_cents")
        qualified_net_edge_cents = guard_meta.get("qualified_net_edge_cents")
        inventory_shares = _current_inventory_shares(self.paper_ledger.position_records, intent.token_id)
        health = _evaluate_quote_health(
            side=intent.side,
            quote_price=float(intent.limit_price or 0.0),
            size=intent.size,
            book=book,
            inventory_shares=inventory_shares,
            posted_ts=intent.ts,
            now_ts=ts,
            opportunity_config=self.config.opportunity,
            execution_config=self.config.execution,
            qualified_gross_edge_cents=float(qualified_gross_edge_cents) if qualified_gross_edge_cents is not None else None,
            qualified_fee_impact_cents=float(qualified_fee_impact_cents) if qualified_fee_impact_cents is not None else None,
            qualified_slippage_cents=float(qualified_slippage_cents) if qualified_slippage_cents is not None else None,
            qualified_net_edge_cents=float(qualified_net_edge_cents) if qualified_net_edge_cents is not None else None,
        )
        expected_net_edge = float(health["expected_net_edge"])
        self._quote_guard_metrics["expected_net_edge_fill_sum"] = float(self._quote_guard_metrics["expected_net_edge_fill_sum"]) + expected_net_edge
        self._quote_guard_metrics["expected_net_edge_fill_count"] = int(self._quote_guard_metrics["expected_net_edge_fill_count"]) + 1

        fair_value = health.get("current_fair_value")
        if fair_value is not None and fair_value > 1e-9:
            if intent.side.upper() == "BUY":
                adverse_move_bps = max(0.0, ((float(fill_price) - float(fair_value)) / float(fair_value)) * 10000.0)
            else:
                adverse_move_bps = max(0.0, ((float(fair_value) - float(fill_price)) / float(fair_value)) * 10000.0)
            self._quote_guard_metrics["fill_to_mid_move_bps_sum"] = float(self._quote_guard_metrics["fill_to_mid_move_bps_sum"]) + adverse_move_bps
            self._quote_guard_metrics["fill_to_mid_move_bps_count"] = int(self._quote_guard_metrics["fill_to_mid_move_bps_count"]) + 1
            if adverse_move_bps > 0.0:
                self._quote_guard_metrics["adverse_fill_count"] = int(self._quote_guard_metrics["adverse_fill_count"]) + 1
                quote_age_sec = max(0.0, (ts - intent.ts).total_seconds())
                if quote_age_sec <= 30.0:
                    self._quote_guard_metrics["adverse_fill_30s_count"] = int(self._quote_guard_metrics["adverse_fill_30s_count"]) + 1

        fills_by_bucket = self._quote_guard_metrics["fills_by_quote_age_bucket"]
        if isinstance(fills_by_bucket, Counter):
            fills_by_bucket[_quote_age_bucket(max(0.0, (ts - intent.ts).total_seconds()))] += 1

    def _quote_guard_summary(self) -> dict[str, object]:
        post_count = int(self._quote_guard_metrics["expected_net_edge_post_count"])
        fill_count = int(self._quote_guard_metrics["expected_net_edge_fill_count"])
        move_count = int(self._quote_guard_metrics["fill_to_mid_move_bps_count"])
        fills_by_bucket = self._quote_guard_metrics["fills_by_quote_age_bucket"]
        reason_counts = self._quote_guard_metrics["quote_guard_reason_counts"]
        return {
            "profitability_cancel_count": int(self._quote_guard_metrics["profitability_cancel_count"]),
            "stale_quote_cancel_count": int(self._quote_guard_metrics["stale_quote_cancel_count"]),
            "price_guard_cancel_count": int(self._quote_guard_metrics["price_guard_cancel_count"]),
            "inventory_skew_suppressed_quote_count": int(self._quote_guard_metrics["inventory_skew_suppressed_quote_count"]),
            "avg_expected_net_edge_at_post": round(float(self._quote_guard_metrics["expected_net_edge_post_sum"]) / post_count, 6) if post_count else None,
            "avg_expected_net_edge_at_fill": round(float(self._quote_guard_metrics["expected_net_edge_fill_sum"]) / fill_count, 6) if fill_count else None,
            "adverse_fill_count": int(self._quote_guard_metrics["adverse_fill_count"]),
            "adverse_fill_30s_count": int(self._quote_guard_metrics["adverse_fill_30s_count"]),
            "avg_fill_to_mid_move_bps": round(float(self._quote_guard_metrics["fill_to_mid_move_bps_sum"]) / move_count, 6) if move_count else None,
            "fills_by_quote_age_bucket": dict(fills_by_bucket) if isinstance(fills_by_bucket, Counter) else {},
            "quote_guard_reason_counts": dict(reason_counts) if isinstance(reason_counts, Counter) else {},
        }

    def _close_event_type(self, state: PositionState) -> str:
        if state == PositionState.FORCE_CLOSED:
            return "position_force_closed"
        if state == PositionState.EXPIRED:
            return "position_expired"
        return "position_closed"
