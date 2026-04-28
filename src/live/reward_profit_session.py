from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from src.ingest.gamma import fetch_events, fetch_markets
from src.intelligence.market_intelligence import build_event_market_registry
from src.live.auth import load_live_credentials
from src.live.broker import LiveBroker
from src.live.client import LiveClientError, LiveOrderStatus, LiveWriteClient
from src.live.rewards import RewardClient, RewardClientError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class RewardMarketStatus(str, Enum):
    SCANNED = "SCANNED"
    SELECTED = "SELECTED"
    INVENTORY_BUILT = "INVENTORY_BUILT"
    QUOTING = "QUOTING"
    PAUSED = "PAUSED"
    EXITING = "EXITING"
    CLOSED = "CLOSED"


@dataclass
class RewardProfitCandidate:
    event_slug: str
    event_title: str | None
    market_slug: str
    question: str | None
    token_id: str
    best_bid: float
    best_ask: float
    midpoint: float
    quote_bid: float
    quote_ask: float
    quote_size: float
    capital_basis_usdc: float
    reward_daily_rate: float
    rewards_max_spread_cents: float
    volume_num: float
    neg_risk: bool
    tick_size: str | None
    expected_reward_per_hour_lower: float
    expected_drawdown_cost_per_hour: float
    reward_minus_drawdown_per_hour: float
    reward_per_dollar_inventory_per_hour: float
    immediate_entry_cost_usdc: float
    immediate_entry_cost_pct: float
    break_even_hours: float
    true_break_even_hours: float
    expected_spread_capture_per_hour: float
    expected_net_edge_per_hour: float
    kelly_raw_fraction: float
    kelly_position_shares: float
    liquidity_factor: float
    activity_factor: float
    sizing_mode: str = "reward_min_size"
    effective_quote_size: float = 0.0
    sizing_reason: str = "KELLY_DISABLED_REWARD_MIN_SIZE"


@dataclass
class RewardProfitConfig:
    settings_path: str = "config/settings.yaml"
    gamma_host: str = "https://gamma-api.polymarket.com"
    out_dir: str = "data/reports"
    state_path: str = "data/reports/reward_profit_state_latest.json"
    pnl_path: str = "data/reports/reward_profit_pnl_latest.json"
    capital_limit_usdc: float = 300.0
    per_market_cap_usdc: float = 120.0
    max_markets: int = 2
    max_markets_per_event: int = 1
    max_drawdown_per_market: float = 3.0
    max_drawdown_pct_of_capital: float = 0.0
    max_daily_loss: float = 10.0
    max_entry_cost_usdc: float = 0.5
    max_entry_cost_pct: float = 0.02
    max_break_even_hours: float = 0.5
    max_true_break_even_hours: float = 0.0
    min_reward_minus_drawdown_per_hour: float = 0.02
    min_reward_per_dollar_inventory_per_hour: float = 0.0005
    projection_horizon_hours: float = 0.0
    min_projected_net_at_horizon_usdc: float = 0.0
    max_quoting_hours_without_fills: float = 0.0
    reward_share_floor: float = 0.005
    reward_share_ceiling: float = 0.02
    reward_share_quality_weight: float = 0.015
    maker_take_share: float = 0.30
    drawdown_factor_per_day: float = 0.30
    use_kelly_sizing: bool = False
    kelly_fraction_scale: float = 0.25
    kelly_horizon_hours: float = 1.0
    reward_calibration_factor: float = 1.0
    reward_correlation_log_path: str = "data/reports/reward_correlation_log.json"
    dry_run_fill_simulation: bool = True
    entry_mode: str = "maker_first"
    exit_cooldown_minutes: float = 45.0
    repeat_exit_cooldown_minutes: float = 90.0
    max_reentries_per_market: int = 1
    actual_reward_zero_cycle_limit: int = 0
    min_actual_reward_delta_usdc: float = 0.0
    min_daily_reward_for_actual_gate_usdc: float = 1.0
    max_adverse_midpoint_move_cents_per_hour: float = 0.0
    min_inventory_risk_coverage_ratio: float = 0.0
    market_intel_path: str = "data/reports/market_intel_latest.json"
    enable_market_intel_filter: bool = False
    max_market_intel_risk_score: float = 0.70
    exit_target_capture_ratio: float = 0.85
    volume_spike_multiple: float = 3.0
    stale_thesis_hours: float = 24.0
    stale_thesis_max_price_change: float = 0.02
    live_order_max_age_sec: float = 120.0
    live_requote_price_move_cents: float = 1.0
    event_limit: int = 200
    market_limit: int = 400
    cycles: int = 1
    interval_sec: int = 0
    live: bool = False
    verbose: bool = False
    show_progress: bool = False
    cancel_open_orders_on_finish: bool = True


@dataclass
class RewardMarketState:
    event_slug: str
    event_title: str | None
    market_slug: str
    question: str | None
    token_id: str
    status: str = RewardMarketStatus.SCANNED.value
    inventory_shares: float = 0.0
    avg_inventory_cost: float = 0.0
    bid_order_id: str | None = None
    ask_order_id: str | None = None
    bid_order_created_ts: str | None = None
    ask_order_created_ts: str | None = None
    bid_order_price: float | None = None
    ask_order_price: float | None = None
    bid_order_size: float = 0.0
    ask_order_size: float = 0.0
    bid_order_filled_size: float = 0.0
    ask_order_filled_size: float = 0.0
    bid_order_age_sec: float = 0.0
    ask_order_age_sec: float = 0.0
    bid_filled_delta: float = 0.0
    ask_filled_delta: float = 0.0
    bid_order_remaining_size: float = 0.0
    ask_order_remaining_size: float = 0.0
    bid_order_status: str | None = None
    ask_order_status: str | None = None
    cancel_count: int = 0
    requote_count: int = 0
    last_order_error: str | None = None
    last_cancel_reason: str | None = None
    hours_in_reward_zone: float = 0.0
    reward_accrued_estimate_usdc: float = 0.0
    reward_accrued_actual_usdc: float = 0.0
    inventory_mtm_pnl_usdc: float = 0.0
    inventory_realized_pnl_usdc: float = 0.0
    spread_realized_usdc: float = 0.0
    simulated_bid_fill_shares: float = 0.0
    simulated_ask_fill_shares: float = 0.0
    simulated_spread_capture_usdc: float = 0.0
    total_cost_proxy_usdc: float = 0.0
    net_after_reward_usdc: float = 0.0
    net_after_reward_and_cost_usdc: float = 0.0
    capital_in_use_usdc: float = 0.0
    last_best_bid: float | None = None
    last_best_ask: float | None = None
    last_midpoint: float | None = None
    entry_midpoint: float | None = None
    max_midpoint_seen: float = 0.0
    last_volume_num: float | None = None
    avg_volume_delta_per_hour: float = 0.0
    last_volume_delta_per_hour: float = 0.0
    adverse_midpoint_move_usdc: float = 0.0
    adverse_midpoint_move_cents_per_hour: float = 0.0
    inventory_risk_coverage_ratio: float = 0.0
    last_reward_rate: float = 0.0
    last_actual_reward_delta_usdc: float = 0.0
    actual_reward_zero_cycles: int = 0
    last_cycle_ts: str | None = None
    last_selected_cycle: int = 0
    last_exit_reason: str | None = None
    cooldown_until_ts: str | None = None
    last_exit_ts: str | None = None
    last_exit_cycle: int = 0
    reentry_count: int = 0
    entry_spread_cost_usdc: float = 0.0
    total_entry_spread_cost_usdc: float = 0.0
    reward_estimate_at_last_actual_usdc: float = 0.0
    reward_actual_vs_estimate_ratio: float = 0.0
    selection_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardProfitSessionState:
    session_id: str
    mode: str
    started_ts: str
    updated_ts: str
    cycle_index: int = 0
    halted: bool = False
    halt_reason: str | None = None
    capital_limit_usdc: float = 300.0
    per_market_cap_usdc: float = 120.0
    max_markets: int = 2
    max_markets_per_event: int = 1
    max_drawdown_per_market: float = 3.0
    max_daily_loss: float = 10.0
    last_scanned_candidate_count: int = 0
    last_eligible_candidate_count: int = 0
    last_filter_reasons: dict[str, int] = field(default_factory=dict)
    last_selection_reasons: dict[str, int] = field(default_factory=dict)
    last_scan_diagnostics: dict[str, Any] = field(default_factory=dict)
    actual_reward_baseline_usdc: float = 0.0
    actual_reward_latest_usdc: float = 0.0
    reward_epoch_id: str | None = None
    selected_market_slugs: list[str] = field(default_factory=list)
    markets: dict[str, RewardMarketState] = field(default_factory=dict)
    cycle_history: list[dict[str, Any]] = field(default_factory=list)


class RewardOrderManager:
    def __init__(self, *, live: bool, write_client: LiveWriteClient | None = None, broker: LiveBroker | None = None) -> None:
        self.live = live
        self.write_client = write_client
        self.broker = broker

    def build_inventory(self, market: RewardMarketState, candidate: RewardProfitCandidate) -> tuple[float, float]:
        if not self.live:
            return candidate.quote_size, candidate.best_ask

        if self.broker is None:
            raise LiveClientError("Live broker not configured")
        intent = _make_order_intent(
            candidate_id=f"reward_profit:{candidate.market_slug}",
            market_slug=candidate.market_slug,
            token_id=candidate.token_id,
            side="BUY",
            size=candidate.quote_size,
            limit_price=candidate.best_ask,
            neg_risk=candidate.neg_risk,
            tick_size=candidate.tick_size,
        )
        report = self.broker.submit_limit_order(intent)
        if report.status.value == "rejected":
            raise LiveClientError(str(report.metadata.get("error") or "inventory order rejected"))
        return float(report.filled_size or 0.0), float(report.avg_fill_price or candidate.best_ask)

    def ensure_quote_orders(self, market: RewardMarketState, candidate: RewardProfitCandidate) -> tuple[str | None, str | None]:
        if not self.live:
            bid_id = market.bid_order_id or f"dry-bid-{uuid.uuid4().hex[:10]}"
            ask_id = market.ask_order_id or f"dry-ask-{uuid.uuid4().hex[:10]}"
            return bid_id, ask_id

        if self.broker is None:
            raise LiveClientError("Live broker not configured")

        bid_order_id = market.bid_order_id
        ask_order_id = market.ask_order_id

        if not bid_order_id:
            bid_intent = _make_order_intent(
                candidate_id=f"reward_profit:{candidate.market_slug}:bid",
                market_slug=candidate.market_slug,
                token_id=candidate.token_id,
                side="BUY",
                size=candidate.quote_size,
                limit_price=candidate.quote_bid,
                neg_risk=candidate.neg_risk,
                tick_size=candidate.tick_size,
            )
            bid_report = self.broker.submit_limit_order(bid_intent)
            if bid_report.metadata.get("error"):
                print(f"[BROKER] bid REJECTED for {candidate.market_slug}: {bid_report.metadata['error']}", flush=True)
            bid_order_id = str(bid_report.metadata.get("live_order_id") or "")

        if not ask_order_id and market.inventory_shares > 0.0:
            ask_intent = _make_order_intent(
                candidate_id=f"reward_profit:{candidate.market_slug}:ask",
                market_slug=candidate.market_slug,
                token_id=candidate.token_id,
                side="SELL",
                size=min(candidate.quote_size, market.inventory_shares),
                limit_price=candidate.quote_ask,
                neg_risk=candidate.neg_risk,
                tick_size=candidate.tick_size,
            )
            ask_report = self.broker.submit_limit_order(ask_intent)
            ask_order_id = str(ask_report.metadata.get("live_order_id") or "")

        return bid_order_id or None, ask_order_id or None

    def cancel_order(self, order_id: str | None) -> bool:
        if not order_id:
            return True
        if not self.live:
            return True
        if self.broker is None:
            raise LiveClientError("Live broker not configured")
        return self.broker.cancel_order(order_id)

    def get_order_status(self, order_id: str | None) -> LiveOrderStatus | None:
        if not order_id or not self.live or self.write_client is None:
            return None
        return self.write_client.get_order_status(order_id)


def _make_order_intent(
    *,
    candidate_id: str,
    market_slug: str,
    token_id: str,
    side: str,
    size: float,
    limit_price: float,
    neg_risk: bool = False,
    tick_size: str | None = None,
):
    from src.domain.models import OrderIntent, OrderMode, OrderType

    return OrderIntent(
        intent_id=str(uuid.uuid4()),
        candidate_id=candidate_id,
        mode=OrderMode.LIVE,
        market_slug=market_slug,
        token_id=token_id,
        position_id=str(uuid.uuid4()),
        side=side,
        order_type=OrderType.LIMIT,
        size=round(size, 6),
        limit_price=round(limit_price, 6),
        max_notional_usd=round(size * limit_price, 6),
        ts=_utc_now(),
        metadata={
            "neg_risk": bool(neg_risk),
            "tick_size": tick_size,
        },
    )


class RewardProfitSelector:
    def __init__(
        self,
        *,
        reward_share_floor: float = 0.005,
        reward_share_ceiling: float = 0.02,
        reward_share_quality_weight: float = 0.015,
        maker_take_share: float = 0.30,
        drawdown_factor_per_day: float = 0.30,
        use_kelly_sizing: bool = False,
        kelly_fraction_scale: float = 0.25,
        kelly_horizon_hours: float = 1.0,
        max_drawdown_per_market: float = 3.0,
        per_market_cap_usdc: float = 120.0,
        reward_calibration_factor: float = 1.0,
    ) -> None:
        self.reward_share_floor = reward_share_floor
        self.reward_share_ceiling = reward_share_ceiling
        self.reward_share_quality_weight = reward_share_quality_weight
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction_scale = kelly_fraction_scale
        self.kelly_horizon_hours = kelly_horizon_hours
        self.max_drawdown_per_market = max_drawdown_per_market
        self.per_market_cap_usdc = per_market_cap_usdc
        self.reward_calibration_factor = reward_calibration_factor
        self.maker_take_share = maker_take_share
        self.drawdown_factor_per_day = drawdown_factor_per_day

    def build_candidates(self, registry: dict[str, Any]) -> list[RewardProfitCandidate]:
        candidates, _diagnostics = self.build_candidates_with_diagnostics(registry)
        return candidates

    def build_candidates_with_diagnostics(
        self,
        registry: dict[str, Any],
    ) -> tuple[list[RewardProfitCandidate], dict[str, Any]]:
        candidates: list[RewardProfitCandidate] = []
        prefilter_reasons: dict[str, int] = {}
        raw_events = list(registry.get("events") or [])
        raw_summary = dict(registry.get("summary") or {})
        raw_market_count = int(raw_summary.get("markets_seen") or 0)
        registry_market_count = 0
        reward_market_count = 0
        orderbook_market_count = 0
        binary_market_count = 0
        top_of_book_market_count = 0
        for event in registry.get("events", []):
            event_slug = str(event.get("slug") or "")
            event_title = event.get("title")
            for market in event.get("markets", []):
                registry_market_count += 1
                if bool(market.get("enable_orderbook")):
                    orderbook_market_count += 1
                if bool(market.get("is_binary_yes_no")):
                    binary_market_count += 1
                if list(market.get("clob_rewards") or []):
                    reward_market_count += 1
                try:
                    best_bid = float(market.get("best_bid") or 0.0)
                    best_ask = float(market.get("best_ask") or 0.0)
                except Exception:
                    best_bid = 0.0
                    best_ask = 0.0
                if best_bid > 0.0 and best_ask > best_bid:
                    top_of_book_market_count += 1

                reason = self._candidate_prefilter_reason(market)
                if reason is not None:
                    prefilter_reasons[reason] = prefilter_reasons.get(reason, 0) + 1
                    continue
                candidate = self._build_candidate(event_slug=event_slug, event_title=event_title, market=market)
                if candidate is not None:
                    candidates.append(candidate)
                else:
                    prefilter_reasons["CANDIDATE_BUILD_FAILED"] = prefilter_reasons.get("CANDIDATE_BUILD_FAILED", 0) + 1
        diagnostics = {
            "raw_events_seen": int(raw_summary.get("events_seen") or len(raw_events)),
            "raw_markets_seen": raw_market_count,
            "registry_events": len(raw_events),
            "registry_markets": registry_market_count,
            "orderbook_enabled_markets": orderbook_market_count,
            "binary_yes_no_markets": binary_market_count,
            "clob_reward_markets": reward_market_count,
            "top_of_book_markets": top_of_book_market_count,
            "scored_candidates": len(candidates),
            "candidate_prefilter_reasons": prefilter_reasons,
        }
        return candidates, diagnostics

    def _candidate_prefilter_reason(self, market: dict[str, Any]) -> str | None:
        if not bool(market.get("enable_orderbook")):
            return "NO_ORDERBOOK"
        if bool(market.get("fees_enabled")):
            return "FEES_ENABLED"
        if not bool(market.get("is_binary_yes_no")):
            return "NOT_BINARY_YES_NO"
        if not market.get("yes_token_id"):
            return "MISSING_YES_TOKEN"

        try:
            best_bid = float(market.get("best_bid") or 0.0)
            best_ask = float(market.get("best_ask") or 0.0)
            quote_size = float(market.get("rewards_min_size") or 0.0)
            reward_daily_rate = sum(float(row.get("rewardsDailyRate") or 0.0) for row in list(market.get("clob_rewards") or []))
            rewards_max_spread_cents = float(market.get("rewards_max_spread") or 0.0)
        except Exception:
            return "INVALID_NUMERIC_FIELDS"

        if best_bid <= 0.0 or best_ask <= best_bid:
            return "INVALID_TOP_OF_BOOK"
        if quote_size <= 0.0:
            return "NO_REWARD_MIN_SIZE"
        if reward_daily_rate <= 0.0:
            return "NO_CLOB_REWARD_RATE"
        if rewards_max_spread_cents <= 0.0:
            return "NO_REWARD_MAX_SPREAD"
        return None

    def select(
        self,
        candidates: list[RewardProfitCandidate],
        *,
        capital_limit_usdc: float,
        per_market_cap_usdc: float,
        max_markets: int,
        max_markets_per_event: int,
    ) -> list[RewardProfitCandidate]:
        selected, _selection_reasons = self.select_with_reasons(
            candidates,
            capital_limit_usdc=capital_limit_usdc,
            per_market_cap_usdc=per_market_cap_usdc,
            max_markets=max_markets,
            max_markets_per_event=max_markets_per_event,
        )
        return selected

    def select_with_reasons(
        self,
        candidates: list[RewardProfitCandidate],
        *,
        capital_limit_usdc: float,
        per_market_cap_usdc: float,
        max_markets: int,
        max_markets_per_event: int,
    ) -> tuple[list[RewardProfitCandidate], dict[str, int]]:
        selected: list[RewardProfitCandidate] = []
        selection_reasons: dict[str, int] = {}
        total_capital = 0.0
        event_counts: dict[str, int] = {}

        ordered = sorted(
            candidates,
            key=lambda item: (
                item.break_even_hours,
                -item.reward_minus_drawdown_per_hour,
                -item.reward_per_dollar_inventory_per_hour,
                -item.expected_reward_per_hour_lower,
                item.capital_basis_usdc,
                item.market_slug,
            ),
        )
        for candidate in ordered:
            if len(selected) >= max_markets:
                selection_reasons["SELECT_MAX_MARKETS"] = selection_reasons.get("SELECT_MAX_MARKETS", 0) + 1
                continue
            if candidate.capital_basis_usdc > per_market_cap_usdc:
                selection_reasons["SELECT_PER_MARKET_CAP"] = selection_reasons.get("SELECT_PER_MARKET_CAP", 0) + 1
                continue
            if total_capital + candidate.capital_basis_usdc > capital_limit_usdc + 1e-9:
                selection_reasons["SELECT_CAPITAL_LIMIT"] = selection_reasons.get("SELECT_CAPITAL_LIMIT", 0) + 1
                continue
            if event_counts.get(candidate.event_slug, 0) >= max_markets_per_event:
                selection_reasons["SELECT_EVENT_LIMIT"] = selection_reasons.get("SELECT_EVENT_LIMIT", 0) + 1
                continue
            selected.append(candidate)
            total_capital += candidate.capital_basis_usdc
            event_counts[candidate.event_slug] = event_counts.get(candidate.event_slug, 0) + 1
        if selected:
            selection_reasons["SELECTED"] = len(selected)
        return selected, selection_reasons

    def _build_candidate(self, *, event_slug: str, event_title: str | None, market: dict[str, Any]) -> RewardProfitCandidate | None:
        if not bool(market.get("enable_orderbook")):
            return None
        if bool(market.get("fees_enabled")):
            return None
        if not bool(market.get("is_binary_yes_no")):
            return None
        if not market.get("yes_token_id"):
            return None

        try:
            best_bid = float(market.get("best_bid") or 0.0)
            best_ask = float(market.get("best_ask") or 0.0)
            quote_size = float(market.get("rewards_min_size") or 0.0)
            reward_daily_rate = sum(float(row.get("rewardsDailyRate") or 0.0) for row in list(market.get("clob_rewards") or []))
            rewards_max_spread_cents = float(market.get("rewards_max_spread") or 0.0)
            neg_risk = bool(market.get("neg_risk"))
            raw_tick_size = market.get("minimum_tick_size") or market.get("tick_size")
            tick_size = str(raw_tick_size) if raw_tick_size not in (None, "") else None
        except Exception:
            return None

        if best_bid <= 0.0 or best_ask <= best_bid:
            return None
        if quote_size <= 0.0 or reward_daily_rate <= 0.0 or rewards_max_spread_cents <= 0.0:
            return None

        current_spread = best_ask - best_bid
        midpoint = (best_bid + best_ask) / 2.0
        reward_max_spread = rewards_max_spread_cents / 100.0
        quote_spread = min(current_spread, reward_max_spread)
        quote_bid = round(max(0.0, midpoint - (quote_spread / 2.0)), 6)
        quote_ask = round(min(1.0, midpoint + (quote_spread / 2.0)), 6)

        liquidity_num = float(market.get("liquidity_num") or 0.0)
        volume_num = float(market.get("volume_num") or 0.0)
        liquidity_factor = _clamp(liquidity_num / 5000.0, 0.05, 1.0)
        activity_factor = _clamp(volume_num / 5000.0, 0.05, 1.0)

        # Q-score 份额估算（基于官方公式 Q = b × ((v-s)/v)²）
        # 我们贴盘口报价，s ≈ quote_spread/2，v = reward_max_spread
        # spread_quality = max(0, (v - s) / v)² ≈ (1 - quote_spread/(2*v))²
        spread_quality = max(0.0, 1.0 - quote_spread / (2.0 * max(reward_max_spread, 1e-9))) ** 2
        # 竞争惩罚：流动性越高说明市场越热门、竞争者越多
        # liquidity>$50k → 约 20–50 个做市商 → 份额偏低
        # liquidity<$5k  → 约 5–15 个做市商 → 份额偏高
        competition_penalty = _clamp(1.0 - (liquidity_num - 5000.0) / 95000.0, 0.2, 1.0)
        share_floor = float(self.reward_share_floor)
        share_ceiling = float(self.reward_share_ceiling)
        share_quality_weight = float(self.reward_share_quality_weight)
        lower_reward_share = _clamp(
            share_floor + share_quality_weight * spread_quality * competition_penalty,
            share_floor,
            share_ceiling,
        )

        capital_basis_usdc = round(quote_size * best_ask, 6)
        immediate_entry_cost_usdc = round(quote_size * max(best_ask - best_bid, 0.0), 6)
        immediate_entry_cost_pct = round(
            immediate_entry_cost_usdc / capital_basis_usdc,
            8,
        ) if capital_basis_usdc > 0.0 else 0.0
        expected_reward_per_hour_lower = round(
            (reward_daily_rate * lower_reward_share) / 24.0 * float(self.reward_calibration_factor),
            6,
        )

        maker_take_share = float(self.maker_take_share)
        hourly_volume_shares = (volume_num / 24.0) / midpoint if midpoint > 0.0 else 0.0
        expected_filled_shares_per_hour = min(
            hourly_volume_shares * maker_take_share,
            quote_size,
        )
        expected_spread_capture_per_hour = round(
            expected_filled_shares_per_hour * quote_spread,
            6,
        )

        drawdown_factor = float(self.drawdown_factor_per_day)
        drawdown_cost_day = immediate_entry_cost_usdc * (
            drawdown_factor + (1.0 - liquidity_factor) * 0.30 + (1.0 - activity_factor) * 0.15
        )
        expected_drawdown_cost_per_hour = round(drawdown_cost_day / 24.0, 6)
        reward_minus_drawdown_per_hour = round(expected_reward_per_hour_lower - expected_drawdown_cost_per_hour, 6)
        expected_net_edge_per_hour = round(
            expected_reward_per_hour_lower
            + expected_spread_capture_per_hour
            - expected_drawdown_cost_per_hour,
            6,
        )
        break_even_hours = round(
            immediate_entry_cost_usdc / expected_reward_per_hour_lower,
            6,
        ) if expected_reward_per_hour_lower > 0.0 else 999999.0
        true_break_even_hours = round(
            immediate_entry_cost_usdc / expected_net_edge_per_hour,
            6,
        ) if expected_net_edge_per_hour > 0.0 else 999999.0
        reward_per_dollar_inventory_per_hour = round(
            expected_reward_per_hour_lower / capital_basis_usdc,
            8,
        ) if capital_basis_usdc > 0.0 else 0.0

        if expected_net_edge_per_hour <= 0.0:
            return None

        # ── Kelly position sizing ────────────────────────────────────────────
        # Kelly for market maker, framed as a bet on each entry:
        #   Bankroll B     = per_market_cap_usdc
        #   Loss amount L  = immediate_entry_cost_usdc
        #                    (the spread cost you definitely pay to enter)
        #   Win amount W   = expected_net_edge_per_hour × horizon
        #                    (reward + spread capture − drawdown, over horizon)
        #   b              = W / L  (win/loss ratio)
        #   p              = base 0.5 + edge_boost, capped at 0.95
        #                    edge_boost = min(W / (2×L), 0.45)
        #                    → when W = L:   p = 0.75
        #                    → when W >> L:  p → 0.95
        #   f*             = (p×b − (1−p)) / b   (raw Kelly fraction of B)
        #   effective_f    = f* × kelly_fraction_scale  (quarter-Kelly default)
        #   kelly_shares   = effective_f × B / best_ask
        #   final_shares   = clamp(kelly_shares, rewards_min_size, B/best_ask)
        #
        # Using entry_cost as L (not max_drawdown) because:
        #   · entry_cost is the certain, immediate cash outlay
        #   · max_drawdown is a guardrail that fires separately
        #   · b_ratio = W/L is scale-invariant (both scale with quote_size when
        #     volume is abundant), so Kelly fraction is correctly independent of
        #     the initial quote_size used for the calculation
        H = float(self.kelly_horizon_hours)
        L = max(immediate_entry_cost_usdc, 1e-6)
        W = expected_net_edge_per_hour * H
        b_ratio = W / L if L > 0.0 else 0.0
        edge_boost = min(W / (2.0 * L), 0.45) if L > 0.0 else 0.0
        p_win = 0.5 + edge_boost
        f_raw = max(0.0, (p_win * b_ratio - (1.0 - p_win)) / b_ratio) if b_ratio > 0.0 else 0.0
        kelly_raw_fraction = round(f_raw, 6)

        if self.use_kelly_sizing and f_raw > 0.0:
            effective_f = f_raw * float(self.kelly_fraction_scale)
            B = float(self.per_market_cap_usdc)
            kelly_capital = effective_f * B
            kelly_shares_raw = kelly_capital / best_ask if best_ask > 0.0 else 0.0
            kelly_position_shares = round(
                max(kelly_shares_raw, quote_size),
                2,
            )
            kelly_position_shares = min(kelly_position_shares, B / best_ask)
            effective_quote_size = kelly_position_shares
            sizing_mode = "kelly"
            sizing_reason = "KELLY_ENABLED_CAPPED_BY_PER_MARKET_CAP"
        else:
            kelly_position_shares = round(quote_size, 2)
            effective_quote_size = quote_size
            sizing_mode = "reward_min_size"
            sizing_reason = "KELLY_DISABLED_REWARD_MIN_SIZE"

        # Recompute all size-dependent metrics with effective_quote_size
        capital_basis_usdc = round(effective_quote_size * best_ask, 6)
        immediate_entry_cost_usdc = round(effective_quote_size * max(best_ask - best_bid, 0.0), 6)
        immediate_entry_cost_pct = round(
            immediate_entry_cost_usdc / capital_basis_usdc, 8
        ) if capital_basis_usdc > 0.0 else 0.0
        expected_filled_shares_per_hour = min(
            hourly_volume_shares * maker_take_share,
            effective_quote_size,
        )
        expected_spread_capture_per_hour = round(expected_filled_shares_per_hour * quote_spread, 6)
        drawdown_cost_day = immediate_entry_cost_usdc * (
            drawdown_factor + (1.0 - liquidity_factor) * 0.30 + (1.0 - activity_factor) * 0.15
        )
        expected_drawdown_cost_per_hour = round(drawdown_cost_day / 24.0, 6)
        reward_minus_drawdown_per_hour = round(expected_reward_per_hour_lower - expected_drawdown_cost_per_hour, 6)
        expected_net_edge_per_hour = round(
            expected_reward_per_hour_lower + expected_spread_capture_per_hour - expected_drawdown_cost_per_hour, 6
        )
        break_even_hours = round(
            immediate_entry_cost_usdc / expected_reward_per_hour_lower, 6
        ) if expected_reward_per_hour_lower > 0.0 else 999999.0
        true_break_even_hours = round(
            immediate_entry_cost_usdc / expected_net_edge_per_hour, 6
        ) if expected_net_edge_per_hour > 0.0 else 999999.0
        reward_per_dollar_inventory_per_hour = round(
            expected_reward_per_hour_lower / capital_basis_usdc, 8
        ) if capital_basis_usdc > 0.0 else 0.0
        # ────────────────────────────────────────────────────────────────────

        return RewardProfitCandidate(
            event_slug=event_slug,
            event_title=event_title,
            market_slug=str(market.get("slug") or ""),
            question=market.get("question"),
            token_id=str(market.get("yes_token_id") or ""),
            best_bid=best_bid,
            best_ask=best_ask,
            midpoint=round(midpoint, 6),
            quote_bid=quote_bid,
            quote_ask=quote_ask,
            quote_size=round(effective_quote_size, 6),
            capital_basis_usdc=capital_basis_usdc,
            reward_daily_rate=round(reward_daily_rate, 6),
            rewards_max_spread_cents=rewards_max_spread_cents,
            volume_num=round(volume_num, 6),
            neg_risk=neg_risk,
            tick_size=tick_size,
            expected_reward_per_hour_lower=expected_reward_per_hour_lower,
            expected_drawdown_cost_per_hour=expected_drawdown_cost_per_hour,
            reward_minus_drawdown_per_hour=reward_minus_drawdown_per_hour,
            reward_per_dollar_inventory_per_hour=reward_per_dollar_inventory_per_hour,
            immediate_entry_cost_usdc=immediate_entry_cost_usdc,
            immediate_entry_cost_pct=immediate_entry_cost_pct,
            break_even_hours=break_even_hours,
            true_break_even_hours=true_break_even_hours,
            expected_spread_capture_per_hour=expected_spread_capture_per_hour,
            expected_net_edge_per_hour=expected_net_edge_per_hour,
            kelly_raw_fraction=kelly_raw_fraction,
            kelly_position_shares=round(kelly_position_shares, 6),
            sizing_mode=sizing_mode,
            effective_quote_size=round(effective_quote_size, 6),
            sizing_reason=sizing_reason,
            liquidity_factor=round(liquidity_factor, 6),
            activity_factor=round(activity_factor, 6),
        )


class RewardProfitSessionEngine:
    def __init__(
        self,
        config: RewardProfitConfig,
        *,
        selector: RewardProfitSelector | None = None,
        reward_client_factory: Callable[[bool], RewardClient | None] | None = None,
        order_manager: RewardOrderManager | None = None,
        registry_provider: Callable[[RewardProfitConfig], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.selector = selector or RewardProfitSelector(
            reward_share_floor=config.reward_share_floor,
            reward_share_ceiling=config.reward_share_ceiling,
            reward_share_quality_weight=config.reward_share_quality_weight,
            maker_take_share=config.maker_take_share,
            drawdown_factor_per_day=config.drawdown_factor_per_day,
            use_kelly_sizing=config.use_kelly_sizing,
            kelly_fraction_scale=config.kelly_fraction_scale,
            kelly_horizon_hours=config.kelly_horizon_hours,
            max_drawdown_per_market=config.max_drawdown_per_market,
            per_market_cap_usdc=config.per_market_cap_usdc,
            reward_calibration_factor=config.reward_calibration_factor,
        )
        self.registry_provider = registry_provider or _default_registry_provider
        self.reward_client_factory = reward_client_factory or _default_reward_client_factory
        self.reward_client = self.reward_client_factory(not config.live)
        self.order_manager = order_manager or _default_order_manager(config.live)
        self.market_intel = self._load_market_intel()

    def run(self) -> tuple[RewardProfitSessionState, dict[str, Any]]:
        import time

        state = self._load_state()
        total_cycles = max(1, self.config.cycles)
        started_monotonic = time.monotonic()
        try:
            for cycle_number in range(1, total_cycles + 1):
                state = self.run_cycle(state)
                if self.config.show_progress:
                    self._print_progress(cycle_number, total_cycles, started_monotonic, state)
                if state.halted:
                    break
                if self.config.interval_sec > 0 and self.config.cycles > 1:
                    time.sleep(self.config.interval_sec)
        except KeyboardInterrupt:
            state.halted = True
            state.halt_reason = "INTERRUPTED"
        if self.config.live and self.config.cancel_open_orders_on_finish:
            self._cancel_open_orders_on_finish(state)
        pnl_report = self._build_pnl_report(state)
        self._write_reports(state, pnl_report)
        return state, pnl_report

    def _print_progress(
        self,
        cycle_number: int,
        total_cycles: int,
        started_monotonic: float,
        state: RewardProfitSessionState,
    ) -> None:
        import time

        elapsed_sec = max(0.0, time.monotonic() - started_monotonic)
        percent = (cycle_number / total_cycles) * 100.0
        avg_cycle_sec = elapsed_sec / cycle_number if cycle_number > 0 else 0.0
        remaining_cycles = max(0, total_cycles - cycle_number)
        remaining_sec = (remaining_cycles * self.config.interval_sec) + (remaining_cycles * avg_cycle_sec)
        summary = self._build_pnl_report(state)["summary"]
        selection_reasons = dict(state.last_selection_reasons)
        selection_note = ""
        if not state.selected_market_slugs and selection_reasons:
            top_reason, top_count = max(selection_reasons.items(), key=lambda item: item[1])
            selection_note = f" | select_block {top_reason}:{top_count}"
        print(
            "Progress: "
            f"{cycle_number}/{total_cycles} "
            f"({percent:.1f}%) | "
            f"ETA {self._format_duration(remaining_sec)} | "
            f"selected {len(state.selected_market_slugs)} | "
            f"active {summary['active_quote_market_count']} | "
            f"raw {summary['scan_diagnostics'].get('raw_markets_seen', 0)} | "
            f"scanned {state.last_scanned_candidate_count} | "
            f"eligible {state.last_eligible_candidate_count} | "
            f"orders b{summary['open_bid_order_count']}/a{summary['open_ask_order_count']} "
            f"age {summary['max_order_age_sec']:.0f}s | "
            f"fill b{summary['bid_order_filled_shares']:.4f}/a{summary['ask_order_filled_shares']:.4f} | "
            f"reward ${summary['reward_accrued_estimate_usdc'] + summary['reward_accrued_actual_usdc']:.4f} | "
            f"spread ${summary['spread_realized_usdc']:.4f} | "
            f"mtm ${summary['inventory_mtm_pnl_usdc']:.4f} | "
            f"cost ${summary['cost_proxy_usdc']:.4f} | "
            f"modeled_net ${summary['net_after_reward_and_cost_usdc']:.4f} | "
            f"verified_net ${summary['verified_net_after_reward_and_cost_usdc']:.4f}"
            f"{selection_note}",
            flush=True,
        )

    def _candidate_filter_reason(
        self,
        item: RewardProfitCandidate,
        *,
        state: RewardProfitSessionState | None = None,
        now: datetime | None = None,
        horizon: float,
        max_true_be: float,
    ) -> str | None:
        if state is not None and now is not None:
            existing = state.markets.get(item.market_slug)
            if existing is not None and self._market_in_cooldown(existing, now):
                return "COOLDOWN_AFTER_EXIT"
        if self.config.max_entry_cost_usdc > 0.0 and item.immediate_entry_cost_usdc > self.config.max_entry_cost_usdc + 1e-9:
            return "ENTRY_COST_USDC"
        if self.config.max_entry_cost_pct > 0.0 and item.immediate_entry_cost_pct > self.config.max_entry_cost_pct + 1e-9:
            return "ENTRY_COST_PCT"
        if self.config.max_break_even_hours > 0.0 and item.break_even_hours > self.config.max_break_even_hours + 1e-9:
            return "BREAK_EVEN_HOURS"
        if self.config.min_reward_minus_drawdown_per_hour > 0.0 and item.reward_minus_drawdown_per_hour < self.config.min_reward_minus_drawdown_per_hour:
            return "REWARD_MINUS_DRAWDOWN"
        if self.config.min_reward_per_dollar_inventory_per_hour > 0.0 and item.reward_per_dollar_inventory_per_hour < self.config.min_reward_per_dollar_inventory_per_hour:
            return "REWARD_PER_DOLLAR"
        if max_true_be > 0.0 and item.true_break_even_hours > max_true_be + 1e-9:
            return "TRUE_BREAK_EVEN_HOURS"
        if (
            horizon > 0.0
            and (item.expected_net_edge_per_hour * horizon - item.immediate_entry_cost_usdc)
            < self.config.min_projected_net_at_horizon_usdc - 1e-9
        ):
            return "PROJECTED_NET_AT_HORIZON"
        if self._candidate_blocked_by_market_intel(item):
            return "MARKET_INTEL_RISK"
        return None

    @staticmethod
    def _market_in_cooldown(market_state: RewardMarketState, now: datetime) -> bool:
        if not market_state.cooldown_until_ts:
            return False
        try:
            cooldown_until = datetime.fromisoformat(market_state.cooldown_until_ts)
        except ValueError:
            return False
        return cooldown_until > now

    def _load_market_intel(self) -> dict[str, Any]:
        if not self.config.enable_market_intel_filter:
            return {}
        path = Path(self.config.market_intel_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        markets = payload.get("markets") if isinstance(payload, dict) else {}
        return markets if isinstance(markets, dict) else {}

    def _candidate_blocked_by_market_intel(self, item: RewardProfitCandidate) -> bool:
        if not self.config.enable_market_intel_filter:
            return False
        intel = self.market_intel.get(item.market_slug)
        if not isinstance(intel, dict):
            return False
        if bool(intel.get("blocked") or intel.get("cooldown")):
            return True
        if intel.get("allow") is False:
            return True
        risk_score = float(intel.get("risk_score") or 0.0)
        return risk_score > self.config.max_market_intel_risk_score

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds_int = max(0, int(round(seconds)))
        minutes, seconds_left = divmod(seconds_int, 60)
        hours, minutes_left = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h{minutes_left:02d}m{seconds_left:02d}s"
        return f"{minutes_left}m{seconds_left:02d}s"

    def run_cycle(
        self,
        state: RewardProfitSessionState | None = None,
        *,
        scanned_candidates: list[RewardProfitCandidate] | None = None,
        cycle_ts: datetime | None = None,
    ) -> RewardProfitSessionState:
        state = state or self._load_state()
        now = cycle_ts or _utc_now()
        state.cycle_index += 1
        state.updated_ts = now.isoformat()

        if scanned_candidates is None:
            registry = self.registry_provider(self.config)
            scanned_candidates, scan_diagnostics = self.selector.build_candidates_with_diagnostics(registry)
        else:
            registry = {"events": []}
            scan_diagnostics = {
                "raw_events_seen": 0,
                "raw_markets_seen": 0,
                "registry_events": 0,
                "registry_markets": 0,
                "orderbook_enabled_markets": 0,
                "binary_yes_no_markets": 0,
                "clob_reward_markets": 0,
                "top_of_book_markets": 0,
                "scored_candidates": len(scanned_candidates),
                "candidate_prefilter_reasons": {},
                "source": "injected_scanned_candidates",
            }

        horizon = self.config.projection_horizon_hours
        max_true_be = self.config.max_true_break_even_hours
        eligible_candidates = []
        filter_reasons: dict[str, int] = {}
        for item in scanned_candidates:
            reason = self._candidate_filter_reason(item, state=state, now=now, horizon=horizon, max_true_be=max_true_be)
            if reason is None:
                eligible_candidates.append(item)
            else:
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
        state.last_scanned_candidate_count = len(scanned_candidates)
        state.last_eligible_candidate_count = len(eligible_candidates)
        state.last_filter_reasons = filter_reasons
        scan_diagnostics["eligible_candidates"] = len(eligible_candidates)
        scan_diagnostics["candidate_filter_reasons"] = dict(filter_reasons)
        state.last_scan_diagnostics = scan_diagnostics
        selected, selection_reasons = self.selector.select_with_reasons(
            eligible_candidates,
            capital_limit_usdc=self.config.capital_limit_usdc,
            per_market_cap_usdc=self.config.per_market_cap_usdc,
            max_markets=self.config.max_markets,
            max_markets_per_event=self.config.max_markets_per_event,
        )
        state.last_selection_reasons = selection_reasons
        state.last_scan_diagnostics["selected_markets"] = len(selected)
        state.last_scan_diagnostics["selection_reasons"] = dict(selection_reasons)
        selected_by_slug = {item.market_slug: item for item in selected}
        state.selected_market_slugs = [item.market_slug for item in selected]

        self._refresh_actual_reward(state)

        for market_slug, market_state in list(state.markets.items()):
            if market_slug not in selected_by_slug and market_state.status not in {RewardMarketStatus.CLOSED.value, RewardMarketStatus.PAUSED.value}:
                self._pause_market(market_state, reason="NOT_SELECTED", state=state, now=now)

        for candidate in selected:
            market_state = state.markets.get(candidate.market_slug)
            if market_state is None:
                market_state = RewardMarketState(
                    event_slug=candidate.event_slug,
                    event_title=candidate.event_title,
                    market_slug=candidate.market_slug,
                    question=candidate.question,
                    token_id=candidate.token_id,
                )
                state.markets[candidate.market_slug] = market_state

            self._advance_market_state(state, market_state, candidate, now)

        pnl_report = self._build_pnl_report(state)
        state.halted = pnl_report["summary"]["net_after_reward_and_cost_usdc"] <= -abs(self.config.max_daily_loss)
        state.halt_reason = "MAX_DAILY_LOSS" if state.halted else None
        state.cycle_history.append(
            {
                "cycle_index": state.cycle_index,
                "ts": now.isoformat(),
                "selected_market_slugs": list(state.selected_market_slugs),
                "summary": pnl_report["summary"],
            }
        )
        return state

    def _advance_market_state(
        self,
        state: RewardProfitSessionState,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
        now: datetime,
    ) -> None:
        previous_ts = datetime.fromisoformat(market_state.last_cycle_ts) if market_state.last_cycle_ts else now
        dt_hours = max(0.0, (now - previous_ts).total_seconds() / 3600.0)

        market_state.status = RewardMarketStatus.SELECTED.value
        if market_state.last_exit_ts and market_state.inventory_shares <= 0.0:
            market_state.reentry_count += 1
            market_state.cooldown_until_ts = None
        market_state.last_selected_cycle = state.cycle_index
        market_state.selection_metrics = asdict(candidate)
        market_state.last_best_bid = candidate.best_bid
        market_state.last_best_ask = candidate.best_ask
        market_state.last_midpoint = candidate.midpoint
        market_state.last_reward_rate = candidate.reward_daily_rate
        market_state.bid_filled_delta = 0.0
        market_state.ask_filled_delta = 0.0

        self._refresh_live_order_marks(market_state, now)
        self._manage_live_order_lifecycle(market_state, candidate, now)

        if market_state.inventory_shares <= 0.0 and self.config.entry_mode == "inventory_first":
            filled_size, fill_price = self.order_manager.build_inventory(market_state, candidate)
            market_state.inventory_shares = round(filled_size, 6)
            market_state.avg_inventory_cost = round(fill_price, 6) if filled_size > 0.0 else 0.0
            entry_spread_cost = max(0.0, filled_size * (fill_price - candidate.best_bid))
            market_state.entry_spread_cost_usdc = round(entry_spread_cost, 6)
            market_state.total_entry_spread_cost_usdc = round(market_state.total_entry_spread_cost_usdc + entry_spread_cost, 6)
            market_state.entry_midpoint = candidate.midpoint if filled_size > 0.0 else market_state.entry_midpoint
            market_state.max_midpoint_seen = candidate.midpoint if filled_size > 0.0 else market_state.max_midpoint_seen
            market_state.status = RewardMarketStatus.INVENTORY_BUILT.value if filled_size > 0.0 else RewardMarketStatus.SELECTED.value

        previous_bid_id = market_state.bid_order_id
        previous_ask_id = market_state.ask_order_id
        bid_order_id, ask_order_id = self._ensure_quote_orders(market_state, candidate, now)
        market_state.bid_order_id = bid_order_id
        market_state.ask_order_id = ask_order_id
        self._record_new_order_metadata(market_state, candidate, now, previous_bid_id, previous_ask_id)
        if market_state.inventory_shares > 0.0 and market_state.ask_order_id:
            market_state.status = RewardMarketStatus.QUOTING.value

        self._refresh_live_order_marks(market_state, now)
        if (
            self.config.live
            and self.config.entry_mode == "maker_first"
            and market_state.inventory_shares > 0.0
            and market_state.ask_order_id is None
        ):
            previous_bid_id = market_state.bid_order_id
            previous_ask_id = market_state.ask_order_id
            bid_order_id, ask_order_id = self._ensure_quote_orders(market_state, candidate, now)
            market_state.bid_order_id = bid_order_id
            market_state.ask_order_id = ask_order_id
            self._record_new_order_metadata(market_state, candidate, now, previous_bid_id, previous_ask_id)
            self._refresh_live_order_marks(market_state, now)
        self._simulate_dry_run_fills(market_state, candidate, dt_hours)
        if self.config.entry_mode == "maker_first" and market_state.inventory_shares > 0.0 and market_state.ask_order_id is None:
            market_state.ask_order_id = f"dry-ask-{uuid.uuid4().hex[:10]}" if not self.config.live else None
            if market_state.ask_order_id:
                self._record_new_order_metadata(market_state, candidate, now, market_state.bid_order_id, None)
        if market_state.inventory_shares > 0.0 and market_state.ask_order_id:
            market_state.status = RewardMarketStatus.QUOTING.value

        if market_state.status == RewardMarketStatus.QUOTING.value:
            market_state.hours_in_reward_zone = round(market_state.hours_in_reward_zone + dt_hours, 6)
            market_state.reward_accrued_estimate_usdc = round(
                market_state.reward_accrued_estimate_usdc + candidate.expected_reward_per_hour_lower * dt_hours,
                6,
            )
            market_state.total_cost_proxy_usdc = round(
                market_state.total_cost_proxy_usdc + candidate.expected_drawdown_cost_per_hour * dt_hours,
                6,
            )

        mark_price = candidate.best_bid
        market_state.inventory_mtm_pnl_usdc = round(
            market_state.inventory_shares * (mark_price - market_state.avg_inventory_cost),
            6,
        )
        market_state.capital_in_use_usdc = round(
            market_state.inventory_shares * market_state.avg_inventory_cost,
            6,
        )
        market_state.net_after_reward_usdc = round(
            market_state.reward_accrued_estimate_usdc
            + market_state.reward_accrued_actual_usdc
            + market_state.spread_realized_usdc
            + market_state.inventory_realized_pnl_usdc
            + market_state.inventory_mtm_pnl_usdc,
            6,
        )
        market_state.net_after_reward_and_cost_usdc = round(
            market_state.net_after_reward_usdc - market_state.total_cost_proxy_usdc,
            6,
        )
        self._update_inventory_risk_metrics(market_state, candidate)
        self._update_exit_behavior_metrics(market_state, candidate, dt_hours)
        market_state.last_cycle_ts = now.isoformat()

        # 百分比止损（Kelly 大仓位时用这个，与仓位大小同步缩放）
        # 绝对值止损（保底兜底，防极端行情）
        pct_stop = self.config.max_drawdown_pct_of_capital
        effective_drawdown_limit = abs(self.config.max_drawdown_per_market)
        if pct_stop > 0.0 and market_state.capital_in_use_usdc > 0.0:
            pct_limit = pct_stop * market_state.capital_in_use_usdc
            effective_drawdown_limit = pct_limit if abs(self.config.max_drawdown_per_market) <= 0.0 else min(pct_limit, abs(self.config.max_drawdown_per_market))
        if market_state.inventory_mtm_pnl_usdc <= -effective_drawdown_limit:
            self._close_market(market_state, exit_price=candidate.best_bid, reason="MAX_DRAWDOWN", state=state, now=now)
            return

        if self._inventory_drift_exceeds_gate(market_state):
            self._close_market(market_state, exit_price=candidate.best_bid, reason="ADVERSE_INVENTORY_DRIFT", state=state, now=now)
            return

        exit_reason = self._exit_behavior_reason(market_state, candidate)
        if exit_reason is not None:
            self._close_market(market_state, exit_price=candidate.best_bid, reason=exit_reason, state=state, now=now)
            return

        if (
            self.config.actual_reward_zero_cycle_limit > 0
            and market_state.status == RewardMarketStatus.QUOTING.value
            and market_state.actual_reward_zero_cycles >= self.config.actual_reward_zero_cycle_limit
            and market_state.last_reward_rate >= self.config.min_daily_reward_for_actual_gate_usdc
        ):
            self._close_market(market_state, exit_price=candidate.best_bid, reason="ACTUAL_REWARD_ZERO_STREAK", state=state, now=now)
            return

        max_stagnation = self.config.max_quoting_hours_without_fills
        if (
            max_stagnation > 0.0
            and market_state.status == RewardMarketStatus.QUOTING.value
            and market_state.hours_in_reward_zone >= max_stagnation - 1e-9
            and (market_state.bid_order_filled_size + market_state.ask_order_filled_size) <= 0.0
        ):
            self._close_market(market_state, exit_price=candidate.best_bid, reason="STAGNATION_NO_FILLS", state=state, now=now)

    def _ensure_quote_orders(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
        now: datetime,
    ) -> tuple[str | None, str | None]:
        if self.config.entry_mode != "maker_first":
            return self.order_manager.ensure_quote_orders(market_state, candidate)
        if self.config.live:
            return self.order_manager.ensure_quote_orders(market_state, candidate)

        bid_order_id = market_state.bid_order_id or f"dry-bid-{uuid.uuid4().hex[:10]}"
        ask_order_id = market_state.ask_order_id if market_state.inventory_shares > 0.0 else None
        if market_state.inventory_shares > 0.0:
            ask_order_id = ask_order_id or f"dry-ask-{uuid.uuid4().hex[:10]}"
        return bid_order_id, ask_order_id

    def _update_exit_behavior_metrics(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
        dt_hours: float,
    ) -> None:
        market_state.max_midpoint_seen = round(max(market_state.max_midpoint_seen or 0.0, candidate.midpoint), 6)
        if market_state.last_volume_num is None:
            market_state.last_volume_num = candidate.volume_num
            return
        if dt_hours <= 0.0:
            market_state.last_volume_num = candidate.volume_num
            return
        volume_delta = max(0.0, candidate.volume_num - market_state.last_volume_num)
        volume_delta_per_hour = volume_delta / dt_hours
        market_state.last_volume_delta_per_hour = round(volume_delta_per_hour, 6)
        if market_state.avg_volume_delta_per_hour <= 0.0:
            market_state.avg_volume_delta_per_hour = round(volume_delta_per_hour, 6)
        elif volume_delta_per_hour > market_state.avg_volume_delta_per_hour * self.config.volume_spike_multiple:
            pass
        else:
            market_state.avg_volume_delta_per_hour = round(
                (market_state.avg_volume_delta_per_hour * 0.8) + (volume_delta_per_hour * 0.2),
                6,
            )
        market_state.last_volume_num = candidate.volume_num

    def _exit_behavior_reason(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
    ) -> str | None:
        entry_midpoint = market_state.entry_midpoint
        if entry_midpoint is None or market_state.status != RewardMarketStatus.QUOTING.value:
            return None

        intel = self.market_intel.get(candidate.market_slug) if isinstance(self.market_intel, dict) else None
        expected_gap = 0.0
        if isinstance(intel, dict):
            expected_gap = float(intel.get("expected_gap") or intel.get("expected_gap_usdc") or 0.0)
        if expected_gap > 0.0 and candidate.midpoint >= entry_midpoint + (expected_gap * self.config.exit_target_capture_ratio):
            return "TARGET_HIT"

        avg_volume = market_state.avg_volume_delta_per_hour
        if (
            avg_volume > 0.0
            and market_state.last_volume_delta_per_hour > avg_volume * self.config.volume_spike_multiple
        ):
            return "VOLUME_EXIT"

        price_change = candidate.midpoint - entry_midpoint
        if (
            market_state.hours_in_reward_zone > self.config.stale_thesis_hours
            and abs(price_change) < self.config.stale_thesis_max_price_change
        ):
            return "STALE_THESIS"
        return None

    def _update_inventory_risk_metrics(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
    ) -> None:
        if market_state.entry_midpoint is None:
            market_state.entry_midpoint = candidate.midpoint
        adverse_move = max(0.0, float(market_state.entry_midpoint or candidate.midpoint) - candidate.midpoint)
        market_state.adverse_midpoint_move_usdc = round(adverse_move * market_state.inventory_shares, 6)
        if market_state.hours_in_reward_zone > 0.0:
            market_state.adverse_midpoint_move_cents_per_hour = round(
                (adverse_move * 100.0) / market_state.hours_in_reward_zone,
                6,
            )
        else:
            market_state.adverse_midpoint_move_cents_per_hour = 0.0

        risk_income = (
            market_state.reward_accrued_actual_usdc
            + market_state.reward_accrued_estimate_usdc
            + market_state.spread_realized_usdc
            - market_state.total_cost_proxy_usdc
        )
        market_state.inventory_risk_coverage_ratio = round(
            risk_income / market_state.adverse_midpoint_move_usdc,
            6,
        ) if market_state.adverse_midpoint_move_usdc > 0.0 else 999999.0

    def _inventory_drift_exceeds_gate(self, market_state: RewardMarketState) -> bool:
        if market_state.status != RewardMarketStatus.QUOTING.value:
            return False
        max_drift = self.config.max_adverse_midpoint_move_cents_per_hour
        if max_drift > 0.0 and market_state.adverse_midpoint_move_cents_per_hour > max_drift:
            return True
        min_coverage = self.config.min_inventory_risk_coverage_ratio
        return (
            min_coverage > 0.0
            and market_state.adverse_midpoint_move_usdc > 0.0
            and market_state.inventory_risk_coverage_ratio < min_coverage
        )

    def _record_new_order_metadata(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
        now: datetime,
        previous_bid_id: str | None,
        previous_ask_id: str | None,
    ) -> None:
        if market_state.bid_order_id and market_state.bid_order_id != previous_bid_id:
            market_state.bid_order_created_ts = now.isoformat()
            market_state.bid_order_price = round(candidate.quote_bid, 6)
            market_state.bid_order_size = round(candidate.quote_size, 6)
            market_state.bid_order_status = "submitted"
            market_state.bid_order_age_sec = 0.0
            market_state.bid_order_remaining_size = round(candidate.quote_size, 6)
            market_state.last_order_error = None
            if previous_bid_id is not None:
                market_state.requote_count += 1
        if market_state.ask_order_id and market_state.ask_order_id != previous_ask_id:
            ask_size = min(candidate.quote_size, market_state.inventory_shares)
            market_state.ask_order_created_ts = now.isoformat()
            market_state.ask_order_price = round(candidate.quote_ask, 6)
            market_state.ask_order_size = round(ask_size, 6)
            market_state.ask_order_status = "submitted"
            market_state.ask_order_age_sec = 0.0
            market_state.ask_order_remaining_size = round(ask_size, 6)
            market_state.last_order_error = None
            if previous_ask_id is not None:
                market_state.requote_count += 1

    def _manage_live_order_lifecycle(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
        now: datetime,
    ) -> None:
        if not self.config.live:
            return

        max_age = float(self.config.live_order_max_age_sec)
        move_threshold = max(0.0, float(self.config.live_requote_price_move_cents)) / 100.0
        if market_state.bid_order_id:
            self._update_order_age(market_state, "bid", now)
            stale = max_age > 0.0 and market_state.bid_order_age_sec >= max_age and market_state.bid_order_filled_size <= 1e-9
            moved = (
                move_threshold > 0.0
                and market_state.bid_order_price is not None
                and abs(candidate.quote_bid - market_state.bid_order_price) >= move_threshold - 1e-12
            )
            if stale or moved:
                self._cancel_side_order(
                    market_state,
                    "bid",
                    "STALE_UNFILLED_BID" if stale else "REQUOTE_BID_PRICE_MOVED",
                )

        if market_state.ask_order_id:
            self._update_order_age(market_state, "ask", now)
            stale = max_age > 0.0 and market_state.ask_order_age_sec >= max_age and market_state.ask_order_filled_size <= 1e-9
            moved = (
                move_threshold > 0.0
                and market_state.ask_order_price is not None
                and abs(candidate.quote_ask - market_state.ask_order_price) >= move_threshold - 1e-12
            )
            no_inventory = market_state.inventory_shares <= 1e-9
            if stale or moved or no_inventory:
                if no_inventory:
                    reason = "CANCEL_ASK_NO_INVENTORY"
                else:
                    reason = "STALE_UNFILLED_ASK" if stale else "REQUOTE_ASK_PRICE_MOVED"
                self._cancel_side_order(market_state, "ask", reason)

    def _refresh_live_order_marks(self, market_state: RewardMarketState, now: datetime) -> None:
        if not self.config.live:
            return
        bid_status = self._safe_get_order_status(market_state.bid_order_id, market_state)
        ask_status = self._safe_get_order_status(market_state.ask_order_id, market_state)

        if bid_status is not None:
            self._apply_live_order_status(market_state, "bid", bid_status, now)
        if ask_status is not None:
            self._apply_live_order_status(market_state, "ask", ask_status, now)

    def _safe_get_order_status(
        self,
        order_id: str | None,
        market_state: RewardMarketState,
    ) -> LiveOrderStatus | None:
        if not order_id:
            return None
        try:
            return self.order_manager.get_order_status(order_id)
        except Exception as exc:
            market_state.last_order_error = str(exc)
            return None

    def _apply_live_order_status(
        self,
        market_state: RewardMarketState,
        side: str,
        status: LiveOrderStatus,
        now: datetime,
    ) -> None:
        self._update_order_age(market_state, side, now)
        terminal_statuses = {"filled", "matched", "cancelled", "canceled", "expired"}
        status_text = str(status.status or "unknown")
        if side == "bid":
            market_state.bid_order_status = status_text
            market_state.bid_order_remaining_size = round(status.size_remaining, 6)
            delta = max(0.0, status.size_matched - market_state.bid_order_filled_size)
            if delta > 0.0:
                market_state.bid_filled_delta = round(delta, 6)
                avg_price = status.avg_price or market_state.bid_order_price or market_state.avg_inventory_cost or 0.0
                total_cost = (market_state.inventory_shares * market_state.avg_inventory_cost) + (delta * avg_price)
                market_state.inventory_shares = round(market_state.inventory_shares + delta, 6)
                market_state.avg_inventory_cost = round(total_cost / market_state.inventory_shares, 6) if market_state.inventory_shares > 0.0 else 0.0
                market_state.bid_order_filled_size = round(status.size_matched, 6)
                if market_state.entry_midpoint is None and market_state.last_midpoint is not None:
                    market_state.entry_midpoint = market_state.last_midpoint
                    market_state.max_midpoint_seen = market_state.last_midpoint
            if status_text.lower() in terminal_statuses:
                self._clear_side_order(market_state, "bid", keep_status=True)
            return

        market_state.ask_order_status = status_text
        market_state.ask_order_remaining_size = round(status.size_remaining, 6)
        delta = max(0.0, status.size_matched - market_state.ask_order_filled_size)
        if delta > 0.0:
            market_state.ask_filled_delta = round(delta, 6)
            avg_price = status.avg_price or market_state.ask_order_price or market_state.avg_inventory_cost
            realized = delta * (avg_price - market_state.avg_inventory_cost)
            market_state.spread_realized_usdc = round(market_state.spread_realized_usdc + realized, 6)
            market_state.inventory_shares = round(max(0.0, market_state.inventory_shares - delta), 6)
            market_state.ask_order_filled_size = round(status.size_matched, 6)
        if status_text.lower() in terminal_statuses:
            self._clear_side_order(market_state, "ask", keep_status=True)

    def _update_order_age(self, market_state: RewardMarketState, side: str, now: datetime) -> None:
        created_ts = market_state.bid_order_created_ts if side == "bid" else market_state.ask_order_created_ts
        if not created_ts:
            return
        try:
            created = datetime.fromisoformat(created_ts)
        except ValueError:
            return
        age = round(max(0.0, (now - created).total_seconds()), 3)
        if side == "bid":
            market_state.bid_order_age_sec = age
        else:
            market_state.ask_order_age_sec = age

    def _cancel_side_order(
        self,
        market_state: RewardMarketState,
        side: str,
        reason: str,
        *,
        count_requote: bool = True,
    ) -> None:
        order_id = market_state.bid_order_id if side == "bid" else market_state.ask_order_id
        if not order_id:
            return
        canceled = self.order_manager.cancel_order(order_id)
        if canceled:
            market_state.cancel_count += 1
            if count_requote:
                market_state.requote_count += 1
            market_state.last_cancel_reason = reason
            self._clear_side_order(market_state, side, keep_status=False)
        else:
            market_state.last_order_error = f"cancel failed: {reason}"

    @staticmethod
    def _clear_side_order(market_state: RewardMarketState, side: str, *, keep_status: bool) -> None:
        if side == "bid":
            market_state.bid_order_id = None
            market_state.bid_order_created_ts = None
            market_state.bid_order_price = None
            market_state.bid_order_size = 0.0
            market_state.bid_order_age_sec = 0.0
            market_state.bid_order_remaining_size = 0.0
            if not keep_status:
                market_state.bid_order_status = None
            return
        market_state.ask_order_id = None
        market_state.ask_order_created_ts = None
        market_state.ask_order_price = None
        market_state.ask_order_size = 0.0
        market_state.ask_order_age_sec = 0.0
        market_state.ask_order_remaining_size = 0.0
        if not keep_status:
            market_state.ask_order_status = None

    def _simulate_dry_run_fills(
        self,
        market_state: RewardMarketState,
        candidate: RewardProfitCandidate,
        dt_hours: float,
    ) -> None:
        if self.config.live or not self.config.dry_run_fill_simulation:
            return
        if dt_hours <= 0.0:
            return
        if market_state.bid_order_id is None and market_state.ask_order_id is None:
            return

        quote_spread = max(0.0, candidate.quote_ask - candidate.quote_bid)
        if quote_spread <= 0.0 or candidate.expected_spread_capture_per_hour <= 0.0:
            return

        expected_side_fill_per_hour = candidate.expected_spread_capture_per_hour / quote_spread
        side_fill = max(0.0, expected_side_fill_per_hour * dt_hours)
        if side_fill <= 0.0:
            return

        bid_fill = side_fill if market_state.bid_order_id else 0.0
        ask_fill = side_fill if market_state.ask_order_id else 0.0
        round_trip_fill = min(bid_fill, ask_fill)
        simulated_spread = round_trip_fill * quote_spread

        extra_bid_fill = max(0.0, bid_fill - round_trip_fill)
        if extra_bid_fill > 0.0:
            total_cost = (
                market_state.inventory_shares * market_state.avg_inventory_cost
            ) + (extra_bid_fill * candidate.quote_bid)
            market_state.inventory_shares = round(market_state.inventory_shares + extra_bid_fill, 6)
            market_state.avg_inventory_cost = (
                round(total_cost / market_state.inventory_shares, 6)
                if market_state.inventory_shares > 0.0
                else 0.0
            )

        extra_ask_fill = max(0.0, ask_fill - round_trip_fill)
        if extra_ask_fill > 0.0:
            sellable = min(extra_ask_fill, market_state.inventory_shares)
            simulated_spread += sellable * (candidate.quote_ask - market_state.avg_inventory_cost)
            market_state.inventory_shares = round(max(0.0, market_state.inventory_shares - sellable), 6)

        market_state.bid_order_filled_size = round(market_state.bid_order_filled_size + bid_fill, 6)
        market_state.ask_order_filled_size = round(market_state.ask_order_filled_size + ask_fill, 6)
        market_state.simulated_bid_fill_shares = round(market_state.simulated_bid_fill_shares + bid_fill, 6)
        market_state.simulated_ask_fill_shares = round(market_state.simulated_ask_fill_shares + ask_fill, 6)
        market_state.simulated_spread_capture_usdc = round(
            market_state.simulated_spread_capture_usdc + simulated_spread,
            6,
        )
        market_state.spread_realized_usdc = round(market_state.spread_realized_usdc + simulated_spread, 6)
        if market_state.entry_midpoint is None and market_state.inventory_shares > 0.0:
            market_state.entry_midpoint = candidate.midpoint
            market_state.max_midpoint_seen = candidate.midpoint

    def _cancel_open_orders_on_finish(self, state: RewardProfitSessionState) -> None:
        for market_state in state.markets.values():
            had_open_order = bool(market_state.bid_order_id or market_state.ask_order_id)
            if not had_open_order:
                continue
            if market_state.bid_order_id and self.order_manager.cancel_order(market_state.bid_order_id):
                market_state.cancel_count += 1
            if market_state.ask_order_id and self.order_manager.cancel_order(market_state.ask_order_id):
                market_state.cancel_count += 1
            self._clear_side_order(market_state, "bid", keep_status=False)
            self._clear_side_order(market_state, "ask", keep_status=False)
            market_state.last_exit_reason = "RUN_FINISHED_CANCEL_OPEN_ORDERS"
            if market_state.inventory_shares <= 0.0:
                market_state.status = RewardMarketStatus.PAUSED.value

    def _pause_market(
        self,
        market_state: RewardMarketState,
        *,
        reason: str,
        state: RewardProfitSessionState | None = None,
        now: datetime | None = None,
    ) -> None:
        self._cancel_side_order(market_state, "bid", reason, count_requote=False)
        self._cancel_side_order(market_state, "ask", reason, count_requote=False)
        market_state.status = RewardMarketStatus.PAUSED.value
        market_state.last_exit_reason = reason
        market_state.inventory_realized_pnl_usdc = round(
            market_state.inventory_realized_pnl_usdc + market_state.inventory_mtm_pnl_usdc,
            6,
        )
        market_state.inventory_mtm_pnl_usdc = 0.0
        market_state.inventory_shares = 0.0
        market_state.avg_inventory_cost = 0.0
        market_state.capital_in_use_usdc = 0.0
        market_state.net_after_reward_usdc = round(
            market_state.reward_accrued_estimate_usdc
            + market_state.reward_accrued_actual_usdc
            + market_state.spread_realized_usdc
            + market_state.inventory_realized_pnl_usdc,
            6,
        )
        market_state.net_after_reward_and_cost_usdc = round(
            market_state.net_after_reward_usdc - market_state.total_cost_proxy_usdc,
            6,
        )
        if state is not None and now is not None:
            self._start_cooldown(market_state, state=state, now=now)

    def _close_market(
        self,
        market_state: RewardMarketState,
        *,
        exit_price: float,
        reason: str,
        state: RewardProfitSessionState | None = None,
        now: datetime | None = None,
    ) -> None:
        self._cancel_side_order(market_state, "bid", reason, count_requote=False)
        self._cancel_side_order(market_state, "ask", reason, count_requote=False)
        market_state.status = RewardMarketStatus.CLOSED.value
        market_state.last_exit_reason = reason
        realized = market_state.inventory_shares * (exit_price - market_state.avg_inventory_cost)
        market_state.inventory_realized_pnl_usdc = round(market_state.inventory_realized_pnl_usdc + realized, 6)
        market_state.inventory_shares = 0.0
        market_state.avg_inventory_cost = 0.0
        market_state.inventory_mtm_pnl_usdc = 0.0
        market_state.capital_in_use_usdc = 0.0
        market_state.net_after_reward_usdc = round(
            market_state.reward_accrued_estimate_usdc
            + market_state.reward_accrued_actual_usdc
            + market_state.spread_realized_usdc
            + market_state.inventory_realized_pnl_usdc,
            6,
        )
        market_state.net_after_reward_and_cost_usdc = round(
            market_state.net_after_reward_usdc - market_state.total_cost_proxy_usdc,
            6,
        )
        if state is not None and now is not None:
            self._start_cooldown(market_state, state=state, now=now)

    def _start_cooldown(
        self,
        market_state: RewardMarketState,
        *,
        state: RewardProfitSessionState,
        now: datetime,
    ) -> None:
        from datetime import timedelta

        minutes = self.config.exit_cooldown_minutes
        if market_state.reentry_count >= self.config.max_reentries_per_market:
            minutes = self.config.repeat_exit_cooldown_minutes
        market_state.last_exit_ts = now.isoformat()
        market_state.last_exit_cycle = state.cycle_index
        market_state.cooldown_until_ts = (now + timedelta(minutes=max(0.0, minutes))).isoformat()

    def _refresh_actual_reward(self, state: RewardProfitSessionState) -> None:
        if self.reward_client is None:
            return
        try:
            summary = self.reward_client.get_rewards_summary()
        except RewardClientError:
            return

        current_earned = float(summary.get("user_earned_usd") or 0.0)
        if state.actual_reward_baseline_usdc == 0.0 and current_earned >= 0.0:
            state.actual_reward_baseline_usdc = current_earned
        previous_total = max(0.0, state.actual_reward_latest_usdc - state.actual_reward_baseline_usdc)
        state.actual_reward_latest_usdc = current_earned
        state.reward_epoch_id = str(summary.get("epoch_id") or state.reward_epoch_id or "")
        new_total = max(0.0, current_earned - state.actual_reward_baseline_usdc)
        delta = max(0.0, new_total - previous_total)
        active_markets = [market for market in state.markets.values() if market.status == RewardMarketStatus.QUOTING.value]
        for market in active_markets:
            market.last_actual_reward_delta_usdc = 0.0
        if not active_markets:
            return
        if delta <= self.config.min_actual_reward_delta_usdc:
            for market in active_markets:
                if market.last_reward_rate >= self.config.min_daily_reward_for_actual_gate_usdc:
                    market.actual_reward_zero_cycles += 1
            return
        per_market = delta / len(active_markets)
        correlation_entries: list[dict[str, Any]] = []
        for market in active_markets:
            market.last_actual_reward_delta_usdc = round(per_market, 6)
            market.actual_reward_zero_cycles = 0
            market.reward_accrued_actual_usdc = round(market.reward_accrued_actual_usdc + per_market, 6)
            estimate_since_last = round(
                market.reward_accrued_estimate_usdc - market.reward_estimate_at_last_actual_usdc,
                6,
            )
            ratio = round(per_market / estimate_since_last, 6) if estimate_since_last > 1e-9 else 0.0
            market.reward_actual_vs_estimate_ratio = ratio
            market.reward_estimate_at_last_actual_usdc = market.reward_accrued_estimate_usdc
            correlation_entries.append({
                "ts": _utc_now().isoformat(),
                "session_id": state.session_id,
                "epoch_id": state.reward_epoch_id,
                "market_slug": market.market_slug,
                "hours_in_reward_zone": round(market.hours_in_reward_zone, 6),
                "actual_delta_usdc": round(per_market, 6),
                "estimated_since_last_actual_usdc": estimate_since_last,
                "actual_vs_estimate_ratio": ratio,
                "reward_daily_rate": market.last_reward_rate,
            })
        if correlation_entries and self.config.reward_correlation_log_path:
            self._append_correlation_log(correlation_entries)

    def _append_correlation_log(self, entries: list[dict[str, Any]]) -> None:
        log_path = Path(self.config.reward_correlation_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        if log_path.exists():
            try:
                existing = json.loads(log_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.extend(entries)
        # 只保留最近 500 条，防止文件无限增长
        if len(existing) > 500:
            existing = existing[-500:]
        log_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_state(self) -> RewardProfitSessionState:
        path = Path(self.config.state_path)
        if not path.exists():
            now = _utc_now().isoformat()
            return RewardProfitSessionState(
                session_id=f"reward_profit_{uuid.uuid4().hex[:10]}",
                mode="LIVE" if self.config.live else "DRY_RUN",
                started_ts=now,
                updated_ts=now,
                capital_limit_usdc=self.config.capital_limit_usdc,
                per_market_cap_usdc=self.config.per_market_cap_usdc,
                max_markets=self.config.max_markets,
                max_markets_per_event=self.config.max_markets_per_event,
                max_drawdown_per_market=self.config.max_drawdown_per_market,
                max_daily_loss=self.config.max_daily_loss,
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        markets = {
            slug: RewardMarketState(**market_payload)
            for slug, market_payload in dict(payload.get("markets") or {}).items()
        }
        return RewardProfitSessionState(
            session_id=str(payload.get("session_id") or f"reward_profit_{uuid.uuid4().hex[:10]}"),
            mode=str(payload.get("mode") or ("LIVE" if self.config.live else "DRY_RUN")),
            started_ts=str(payload.get("started_ts") or _utc_now().isoformat()),
            updated_ts=str(payload.get("updated_ts") or _utc_now().isoformat()),
            cycle_index=int(payload.get("cycle_index") or 0),
            halted=bool(payload.get("halted") or False),
            halt_reason=payload.get("halt_reason"),
            capital_limit_usdc=float(payload.get("capital_limit_usdc") or self.config.capital_limit_usdc),
            per_market_cap_usdc=float(payload.get("per_market_cap_usdc") or self.config.per_market_cap_usdc),
            max_markets=int(payload.get("max_markets") or self.config.max_markets),
            max_markets_per_event=int(payload.get("max_markets_per_event") or self.config.max_markets_per_event),
                max_drawdown_per_market=float(payload.get("max_drawdown_per_market") or self.config.max_drawdown_per_market),
                max_daily_loss=float(payload.get("max_daily_loss") or self.config.max_daily_loss),
            last_scanned_candidate_count=int(payload.get("last_scanned_candidate_count") or 0),
            last_eligible_candidate_count=int(payload.get("last_eligible_candidate_count") or 0),
            last_filter_reasons=dict(payload.get("last_filter_reasons") or {}),
            last_selection_reasons=dict(payload.get("last_selection_reasons") or {}),
            last_scan_diagnostics=dict(payload.get("last_scan_diagnostics") or {}),
            actual_reward_baseline_usdc=float(payload.get("actual_reward_baseline_usdc") or 0.0),
            actual_reward_latest_usdc=float(payload.get("actual_reward_latest_usdc") or 0.0),
            reward_epoch_id=payload.get("reward_epoch_id"),
            selected_market_slugs=list(payload.get("selected_market_slugs") or []),
            markets=markets,
            cycle_history=list(payload.get("cycle_history") or []),
        )

    def _build_pnl_report(self, state: RewardProfitSessionState) -> dict[str, Any]:
        market_rows = []
        total_capital = 0.0
        total_reward_est = 0.0
        total_reward_actual = 0.0
        total_inventory_mtm = 0.0
        total_inventory_realized = 0.0
        total_spread_realized = 0.0
        total_simulated_spread = 0.0
        total_entry_spread_cost = 0.0
        total_cost_proxy = 0.0
        active_quote_count = 0
        cooldown_count = 0
        open_bid_count = 0
        open_ask_count = 0
        total_bid_filled = 0.0
        total_ask_filled = 0.0
        max_order_age_sec = 0.0
        for market in state.markets.values():
            if market.status == RewardMarketStatus.QUOTING.value:
                active_quote_count += 1
            if market.cooldown_until_ts:
                cooldown_count += 1
            if market.bid_order_id:
                open_bid_count += 1
            if market.ask_order_id:
                open_ask_count += 1
            total_bid_filled += market.bid_order_filled_size
            total_ask_filled += market.ask_order_filled_size
            max_order_age_sec = max(max_order_age_sec, market.bid_order_age_sec, market.ask_order_age_sec)
            total_capital += market.capital_in_use_usdc
            total_reward_est += market.reward_accrued_estimate_usdc
            total_reward_actual += market.reward_accrued_actual_usdc
            total_inventory_mtm += market.inventory_mtm_pnl_usdc
            total_inventory_realized += market.inventory_realized_pnl_usdc
            total_spread_realized += market.spread_realized_usdc
            total_simulated_spread += market.simulated_spread_capture_usdc
            total_entry_spread_cost += market.total_entry_spread_cost_usdc
            total_cost_proxy += market.total_cost_proxy_usdc
            market_row = asdict(market)
            market_row["fill_simulation"] = {
                "simulated_bid_fill_shares": round(market.simulated_bid_fill_shares, 6),
                "simulated_ask_fill_shares": round(market.simulated_ask_fill_shares, 6),
                "simulated_spread_capture_usdc": round(market.simulated_spread_capture_usdc, 6),
                "expected_spread_capture_per_hour": round(
                    float(market.selection_metrics.get("expected_spread_capture_per_hour") or 0.0),
                    6,
                ),
            }
            market_row["live_order_lifecycle"] = {
                "open_bid_order_id": market.bid_order_id,
                "open_ask_order_id": market.ask_order_id,
                "bid_order_status": market.bid_order_status,
                "ask_order_status": market.ask_order_status,
                "bid_order_age_sec": round(market.bid_order_age_sec, 3),
                "ask_order_age_sec": round(market.ask_order_age_sec, 3),
                "bid_filled_delta": round(market.bid_filled_delta, 6),
                "ask_filled_delta": round(market.ask_filled_delta, 6),
                "bid_order_remaining_size": round(market.bid_order_remaining_size, 6),
                "ask_order_remaining_size": round(market.ask_order_remaining_size, 6),
                "cancel_count": market.cancel_count,
                "requote_count": market.requote_count,
                "last_order_error": market.last_order_error,
                "last_cancel_reason": market.last_cancel_reason,
                "sizing_mode": str(market.selection_metrics.get("sizing_mode") or ""),
                "effective_quote_size": round(float(market.selection_metrics.get("effective_quote_size") or 0.0), 6),
                "sizing_reason": str(market.selection_metrics.get("sizing_reason") or ""),
            }
            market_rows.append(market_row)

        net_after_reward = round(
            total_reward_est + total_reward_actual + total_spread_realized + total_inventory_realized + total_inventory_mtm,
            6,
        )
        net_after_reward_and_cost = round(net_after_reward - total_cost_proxy, 6)
        verified_net_after_reward = round(
            total_reward_actual + total_spread_realized + total_inventory_realized + total_inventory_mtm,
            6,
        )
        verified_net_after_reward_and_cost = round(verified_net_after_reward - total_cost_proxy, 6)
        return {
            "report_type": "reward_profit_pnl",
            "session_id": state.session_id,
            "generated_ts": _utc_now().isoformat(),
            "summary": {
                "mode": state.mode,
                "cycle_index": state.cycle_index,
                "halted": state.halted,
                "halt_reason": state.halt_reason,
                "last_scanned_candidate_count": state.last_scanned_candidate_count,
                "last_eligible_candidate_count": state.last_eligible_candidate_count,
                "last_filter_reasons": dict(state.last_filter_reasons),
                "last_selection_reasons": dict(state.last_selection_reasons),
                "scan_diagnostics": dict(state.last_scan_diagnostics),
                "capital_limit_usdc": round(state.capital_limit_usdc, 6),
                "capital_in_use_usdc": round(total_capital, 6),
                "active_quote_market_count": active_quote_count,
                "cooldown_market_count": cooldown_count,
                "open_bid_order_count": open_bid_count,
                "open_ask_order_count": open_ask_count,
                "bid_order_filled_shares": round(total_bid_filled, 6),
                "ask_order_filled_shares": round(total_ask_filled, 6),
                "max_order_age_sec": round(max_order_age_sec, 3),
                "reward_accrued_estimate_usdc": round(total_reward_est, 6),
                "reward_accrued_actual_usdc": round(total_reward_actual, 6),
                "inventory_mtm_pnl_usdc": round(total_inventory_mtm, 6),
                "inventory_realized_pnl_usdc": round(total_inventory_realized, 6),
                "spread_realized_usdc": round(total_spread_realized, 6),
                "simulated_spread_capture_usdc": round(total_simulated_spread, 6),
                "total_entry_spread_cost_usdc": round(total_entry_spread_cost, 6),
                "net_after_reward_usdc": net_after_reward,
                "cost_proxy_usdc": round(total_cost_proxy, 6),
                "net_after_reward_and_cost_usdc": net_after_reward_and_cost,
                "verified_net_after_reward_usdc": verified_net_after_reward,
                "verified_net_after_reward_and_cost_usdc": verified_net_after_reward_and_cost,
                "model_vs_realized": {
                    "modeled_reward_usdc": round(total_reward_est, 6),
                    "actual_reward_usdc": round(total_reward_actual, 6),
                    "simulated_spread_usdc": round(total_simulated_spread, 6),
                    "realized_spread_usdc": round(total_spread_realized, 6),
                    "inventory_mtm_usdc": round(total_inventory_mtm, 6),
                    "inventory_realized_usdc": round(total_inventory_realized, 6),
                    "cost_proxy_usdc": round(total_cost_proxy, 6),
                    "net_after_all_costs_usdc": net_after_reward_and_cost,
                    "verified_net_after_all_costs_usdc": verified_net_after_reward_and_cost,
                },
            },
            "markets": sorted(market_rows, key=lambda row: (-row["net_after_reward_and_cost_usdc"], row["market_slug"])),
        }

    def _write_reports(self, state: RewardProfitSessionState, pnl_report: dict[str, Any]) -> None:
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path = Path(self.config.state_path)
        pnl_path = Path(self.config.pnl_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pnl_path.parent.mkdir(parents=True, exist_ok=True)

        state_payload = asdict(state)
        state_path.write_text(json.dumps(state_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        pnl_path.write_text(json.dumps(pnl_report, indent=2, ensure_ascii=False), encoding="utf-8")

        stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
        (out_dir / f"reward_profit_state_{stamp}.json").write_text(
            json.dumps(state_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / f"reward_profit_pnl_{stamp}.json").write_text(
            json.dumps(pnl_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _default_registry_provider(config: RewardProfitConfig) -> dict[str, Any]:
    events = fetch_events(config.gamma_host, config.event_limit)
    markets = fetch_markets(config.gamma_host, config.market_limit)
    return build_event_market_registry(events, markets)


def _default_reward_client_factory(dry_run: bool) -> RewardClient | None:
    if dry_run:
        return RewardClient(address="0x0000000000000000000000000000000000000000", dry_run=True)
    try:
        creds = load_live_credentials()
    except Exception:
        return None
    return RewardClient.from_credentials(creds, dry_run=False)


def _default_order_manager(live: bool) -> RewardOrderManager:
    if not live:
        return RewardOrderManager(live=False)
    import os
    creds = load_live_credentials()
    raw_sig = os.environ.get("POLYMARKET_SIGNATURE_TYPE")
    signature_type = int(raw_sig) if raw_sig is not None else None
    funder = os.environ.get("POLYMARKET_FUNDER") or None
    write_client = LiveWriteClient.from_credentials(
        creds, host="https://clob.polymarket.com", dry_run=False,
        signature_type=signature_type, funder=funder,
    )
    broker = LiveBroker(write_client)
    return RewardOrderManager(live=True, write_client=write_client, broker=broker)
