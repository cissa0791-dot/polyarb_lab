from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.domain.models import ExecutionReport, OrderIntent, OrderMode, OrderStatus, OrderType
from src.ingest.clob import ReadOnlyClob
from src.ingest.gamma import fetch_events, fetch_markets
from src.intelligence.market_intelligence import build_event_market_registry
from src.live.broker import LiveBroker
from src.live.client import LiveClientError, LiveOrderStatus, LiveWriteClient
from src.live.rewards import RewardClient, RewardClientError
from src.scanner.maker_scan_quote_planner import plan_quote


DEFAULT_REWARD_EVENTS = [
    "next-prime-minister-of-hungary",
    "netanyahu-out-before-2027",
    "balance-of-power-2026-midterms",
    "presidential-election-winner-2028",
    "democratic-presidential-nominee-2028",
    "nba-mvp-694",
]


class RewardExecutionState(str, Enum):
    SCANNED = "SCANNED"
    SELECTED = "SELECTED"
    INVENTORY_BUILT = "INVENTORY_BUILT"
    QUOTING = "QUOTING"
    PAUSED = "PAUSED"
    EXITING = "EXITING"
    CLOSED = "CLOSED"


class RewardOrderKind(str, Enum):
    INVENTORY = "inventory"
    BID = "bid"
    ASK = "ask"
    EXIT = "exit"


class RewardOrderRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: RewardOrderKind
    side: str
    order_id: str
    requested_size: float
    price: float
    matched_size: float = 0.0
    status: str = "created"
    synthetic: bool = False
    created_ts: datetime
    updated_ts: datetime


class RewardMarketCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_slug: str
    market_slug: str
    question: str | None = None
    yes_token_id: str
    no_token_id: str
    best_bid: float
    best_ask: float
    midpoint: float
    neg_risk: bool = False
    reward_daily_rate: float
    rewards_min_size: float
    rewards_max_spread: float
    quote_bid: float
    quote_ask: float
    quote_size: float
    capital_in_use_usdc: float
    implied_reward_ev_daily_usdc: float
    expected_reward_per_hour: float
    expected_drawdown_cost_per_hour: float
    reward_minus_drawdown_per_hour: float
    reward_per_dollar_inventory_per_day: float
    modeled_spread_capture_ev_daily_usdc: float
    modeled_cost_proxy_daily_usdc: float
    selection_score: float
    current_spread_cents: float


class RewardMarketRuntimeState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_slug: str
    market_slug: str
    question: str | None = None
    yes_token_id: str
    no_token_id: str
    status: RewardExecutionState = RewardExecutionState.SCANNED
    inventory_shares: float = 0.0
    avg_inventory_cost: float = 0.0
    capital_in_use_usdc: float = 0.0
    latest_mid_price: float | None = None
    reward_zone_started_ts: datetime | None = None
    reward_zone_elapsed_sec: float = 0.0
    reward_accrued_estimate_usdc: float = 0.0
    inventory_mtm_pnl_usdc: float = 0.0
    spread_realized_usdc: float = 0.0
    net_after_reward_usdc: float = 0.0
    daily_realized_pnl_usdc: float = 0.0
    cooldown_remaining_sessions: int = 0
    selection_count: int = 0
    last_selected_ts: datetime | None = None
    last_update_ts: datetime | None = None
    last_halt_reason: str | None = None
    inventory_order: RewardOrderRecord | None = None
    bid_order: RewardOrderRecord | None = None
    ask_order: RewardOrderRecord | None = None
    exit_order: RewardOrderRecord | None = None
    last_candidate: RewardMarketCandidate | None = None


class RewardLiveConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capital: float = 5000.0
    max_capital_per_market: float = 1200.0
    max_markets: int = 6
    max_markets_per_event: int = 2
    cooldown_sessions: int = 4
    max_drawdown_per_market: float = 5.0
    max_daily_loss: float = 25.0
    min_reward_minus_drawdown_per_hour: float = 0.02
    min_reward_per_dollar_inventory_per_day: float = 0.01
    min_implied_reward_ev_daily_usdc: float = 0.25
    max_current_spread_cents: float = 4.0
    event_fetch_limit: int = 1500
    market_fetch_limit: int = 3000
    replace_drift_cents: float = 0.5
    dry_run: bool = True


class RewardLiveStateReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_ts: datetime
    mode: str
    halted: bool = False
    halt_reason: str | None = None
    selected_market_count: int = 0
    active_quote_market_count: int = 0
    total_capital_in_use_usdc: float = 0.0
    total_reward_accrued_estimate_usdc: float = 0.0
    total_inventory_mtm_pnl_usdc: float = 0.0
    total_spread_realized_usdc: float = 0.0
    total_net_after_reward_usdc: float = 0.0
    actual_reward_summary: dict[str, Any] | None = None
    markets: list[RewardMarketRuntimeState] = Field(default_factory=list)


def _reward_daily_rate(market: dict[str, Any]) -> float:
    rewards = market.get("clob_rewards") or []
    total = 0.0
    for reward in rewards:
        try:
            total += float(reward.get("rewardsDailyRate", 0.0) or 0.0)
        except Exception:
            continue
    return round(total, 6)


def _q_score_per_side(max_spread_cents: float, distance_from_mid_cents: float, size_shares: float) -> float:
    if max_spread_cents <= 0 or distance_from_mid_cents >= max_spread_cents:
        return 0.0
    return ((max_spread_cents - distance_from_mid_cents) / max_spread_cents) ** 2 * size_shares


def _book_competitor_q(book: Any, mid_p: float, max_spread_cents: float) -> float:
    if book is None:
        return 0.0
    total = 0.0
    for level in list(getattr(book, "bids", [])) + list(getattr(book, "asks", [])):
        dist_cents = abs(float(level.price) - mid_p) * 100.0
        total += _q_score_per_side(max_spread_cents, dist_cents, float(level.size))
    return total


def _estimate_implied_reward_ev(daily_rate: float, our_q_score: float, competitor_q_sum: float) -> float:
    total = our_q_score + competitor_q_sum
    if total <= 0 or our_q_score <= 0:
        return 0.0
    return daily_rate * (our_q_score / total)


def _compute_maker_mm_ev(
    *,
    best_bid: float,
    best_ask: float,
    rewards_min_size: float,
    rewards_max_spread_cents: float,
    reward_daily_rate: float,
    volume_num: float = 0.0,
) -> dict[str, float]:
    current_spread = best_ask - best_bid
    max_spread = rewards_max_spread_cents / 100.0
    midpoint = (best_bid + best_ask) / 2.0
    our_half_spread = min(current_spread / 2.0, max_spread / 2.0)
    our_half_spread = max(our_half_spread, 0.005)
    quote_spread = min(our_half_spread * 2.0, current_spread)
    quote_size = max(rewards_min_size, 20.0)
    distance_from_mid = our_half_spread * 100.0
    our_q_score = _q_score_per_side(rewards_max_spread_cents, distance_from_mid, quote_size) * 2
    reward_competition = min(30.0, max(10.0, 10.0 + (reward_daily_rate / 20.0)))
    volume_competition = min(5.0, volume_num / 100000.0) if volume_num > 0 else 0.0
    competition_factor = reward_competition + volume_competition
    estimated_total_q = our_q_score * competition_factor if our_q_score > 0 else 0.0
    reward_ev = reward_daily_rate * (our_q_score / estimated_total_q) if estimated_total_q > 0 else 0.0
    volume_factor = min(1.0, volume_num / 50000.0) if volume_num > 0 else 0.1
    tightness_factor = max(0.0, 1.0 - (distance_from_mid / max(rewards_max_spread_cents, 1e-9)))
    fill_prob_per_side = min(0.90, max(0.05, 0.10 + 0.50 * volume_factor + 0.30 * tightness_factor))
    both_fill_prob = fill_prob_per_side ** 2 * 0.5
    spread_capture_ev = both_fill_prob * quote_spread * quote_size
    adverse_cost = fill_prob_per_side * 0.15 * current_spread * 2.0 * quote_size
    inventory_cost = fill_prob_per_side * (1.0 - fill_prob_per_side) * 2.0 * current_spread * quote_size * 0.5
    cancel_cost = 0.001
    total_ev = spread_capture_ev + reward_ev - adverse_cost - inventory_cost - cancel_cost
    return {
        "reward_ev": round(reward_ev, 6),
        "spread_capture_ev": round(spread_capture_ev, 6),
        "adverse_cost": round(adverse_cost, 6),
        "inventory_cost": round(inventory_cost, 6),
        "cancel_cost": round(cancel_cost, 6),
        "total_ev": round(total_ev, 6),
        "our_q_score": round(our_q_score, 6),
    }


def _capital_basis_usdc(mid_p: float, quote_size: float, neg_risk: bool) -> float:
    if neg_risk:
        return round(quote_size, 6)
    return round(quote_size * max(min(mid_p * 2.0, 1.0), 0.01), 6)


class RewardMarketSelector:
    def __init__(self, *, gamma_host: str, clob_host: str) -> None:
        self.gamma_host = gamma_host
        self.clob_host = clob_host

    def select(
        self,
        *,
        event_slugs: list[str],
        state_by_market: dict[str, RewardMarketRuntimeState],
        config: RewardLiveConfig,
    ) -> list[RewardMarketCandidate]:
        events = fetch_events(self.gamma_host, limit=config.event_fetch_limit)
        markets = fetch_markets(self.gamma_host, limit=config.market_fetch_limit)
        registry = build_event_market_registry(events, markets)
        event_set = set(event_slugs)
        eligible: list[dict[str, Any]] = []
        for event in registry.get("events", []):
            if str(event.get("slug") or "") not in event_set:
                continue
            for market in event.get("markets", []):
                if not bool(market.get("enable_orderbook")) or not bool(market.get("is_binary_yes_no")):
                    continue
                if bool(market.get("fees_enabled")):
                    continue
                if not market.get("yes_token_id") or not market.get("no_token_id"):
                    continue
                best_bid = float(market.get("best_bid") or 0.0)
                best_ask = float(market.get("best_ask") or 0.0)
                if best_bid <= 0 or best_ask <= best_bid:
                    continue
                min_size = float(market.get("rewards_min_size") or 0.0)
                max_spread = float(market.get("rewards_max_spread") or 0.0)
                rate = _reward_daily_rate(market)
                if min_size <= 0 or max_spread <= 0 or rate <= 0:
                    continue
                eligible.append(
                    {
                        "event_slug": str(event.get("slug") or ""),
                        "market_slug": str(market.get("slug") or ""),
                        "question": market.get("question"),
                        "yes_token_id": str(market.get("yes_token_id") or ""),
                        "no_token_id": str(market.get("no_token_id") or ""),
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "rewards_min_size": min_size,
                        "rewards_max_spread": max_spread,
                        "reward_daily_rate": rate,
                        "volume_num": float(market.get("volume_num") or 0.0),
                        "neg_risk": bool(market.get("neg_risk")),
                    }
                )

        clob = ReadOnlyClob(self.clob_host)
        books = clob.prefetch_books([m["yes_token_id"] for m in eligible][:200], max_workers=8)
        candidates: list[RewardMarketCandidate] = []
        for market in eligible:
            book = books.get(market["yes_token_id"])
            if not book or not getattr(book, "bids", []) or not getattr(book, "asks", []):
                continue
            best_bid = float(book.bids[0].price)
            best_ask = float(book.asks[0].price)
            if best_bid <= 0 or best_ask <= best_bid:
                continue
            market["best_bid"] = best_bid
            market["best_ask"] = best_ask
            plan = plan_quote(market)
            if not plan.eligible or plan.quote_size <= 0.0:
                continue
            midpoint = round((best_bid + best_ask) / 2.0, 6)
            bid_dist = abs(plan.quote_bid - midpoint) * 100.0
            ask_dist = abs(plan.quote_ask - midpoint) * 100.0
            our_q_score = _q_score_per_side(market["rewards_max_spread"], bid_dist, plan.quote_size)
            our_q_score += _q_score_per_side(market["rewards_max_spread"], ask_dist, plan.quote_size)
            competitor_q_sum = _book_competitor_q(book, midpoint, market["rewards_max_spread"])
            implied_reward_ev = _estimate_implied_reward_ev(market["reward_daily_rate"], our_q_score, competitor_q_sum)
            ev = _compute_maker_mm_ev(
                best_bid=best_bid,
                best_ask=best_ask,
                rewards_min_size=market["rewards_min_size"],
                rewards_max_spread_cents=market["rewards_max_spread"],
                reward_daily_rate=market["reward_daily_rate"],
                volume_num=float(market["volume_num"] or 0.0),
            )
            capital_basis = _capital_basis_usdc(midpoint, plan.quote_size, bool(market["neg_risk"]))
            expected_reward_per_hour = implied_reward_ev / 24.0
            expected_drawdown_cost_per_hour = (ev["adverse_cost"] + ev["inventory_cost"]) / 24.0
            reward_minus_drawdown_per_hour = expected_reward_per_hour - expected_drawdown_cost_per_hour
            reward_per_dollar_inventory_per_day = implied_reward_ev / max(capital_basis, 1e-9)
            selection_score = reward_minus_drawdown_per_hour + reward_per_dollar_inventory_per_day
            candidates.append(
                RewardMarketCandidate(
                    event_slug=market["event_slug"],
                    market_slug=market["market_slug"],
                    question=market.get("question"),
                    yes_token_id=market["yes_token_id"],
                    no_token_id=market["no_token_id"],
                    best_bid=best_bid,
                    best_ask=best_ask,
                    midpoint=midpoint,
                    neg_risk=bool(market["neg_risk"]),
                    reward_daily_rate=market["reward_daily_rate"],
                    rewards_min_size=market["rewards_min_size"],
                    rewards_max_spread=market["rewards_max_spread"],
                    quote_bid=plan.quote_bid,
                    quote_ask=plan.quote_ask,
                    quote_size=plan.quote_size,
                    capital_in_use_usdc=capital_basis,
                    implied_reward_ev_daily_usdc=round(implied_reward_ev, 6),
                    expected_reward_per_hour=round(expected_reward_per_hour, 6),
                    expected_drawdown_cost_per_hour=round(expected_drawdown_cost_per_hour, 6),
                    reward_minus_drawdown_per_hour=round(reward_minus_drawdown_per_hour, 6),
                    reward_per_dollar_inventory_per_day=round(reward_per_dollar_inventory_per_day, 6),
                    modeled_spread_capture_ev_daily_usdc=ev["spread_capture_ev"],
                    modeled_cost_proxy_daily_usdc=round(ev["adverse_cost"] + ev["inventory_cost"] + ev["cancel_cost"], 6),
                    selection_score=round(selection_score, 6),
                    current_spread_cents=round((best_ask - best_bid) * 100.0, 6),
                )
            )

        candidates = [
            candidate
            for candidate in candidates
            if self._passes_profit_gates(candidate, config)
        ]
        candidates.sort(
            key=lambda candidate: (
                state_by_market.get(candidate.market_slug, RewardMarketRuntimeState(
                    event_slug=candidate.event_slug,
                    market_slug=candidate.market_slug,
                    yes_token_id=candidate.yes_token_id,
                    no_token_id=candidate.no_token_id,
                )).cooldown_remaining_sessions > 0,
                -candidate.selection_score,
                -candidate.implied_reward_ev_daily_usdc,
            )
        )
        selected: list[RewardMarketCandidate] = []
        per_event: dict[str, int] = {}
        total_capital = 0.0
        for candidate in candidates:
            if len(selected) >= config.max_markets:
                break
            if per_event.get(candidate.event_slug, 0) >= config.max_markets_per_event:
                continue
            if total_capital + candidate.capital_in_use_usdc > config.capital + 1e-9:
                continue
            selected.append(candidate)
            total_capital += candidate.capital_in_use_usdc
            per_event[candidate.event_slug] = per_event.get(candidate.event_slug, 0) + 1
        return selected

    @staticmethod
    def _passes_profit_gates(candidate: RewardMarketCandidate, config: RewardLiveConfig) -> bool:
        if candidate.capital_in_use_usdc <= 0.0:
            return False
        if candidate.capital_in_use_usdc > config.max_capital_per_market + 1e-9:
            return False
        if candidate.reward_minus_drawdown_per_hour < config.min_reward_minus_drawdown_per_hour:
            return False
        if candidate.reward_per_dollar_inventory_per_day < config.min_reward_per_dollar_inventory_per_day:
            return False
        if candidate.implied_reward_ev_daily_usdc < config.min_implied_reward_ev_daily_usdc:
            return False
        if candidate.current_spread_cents > config.max_current_spread_cents:
            return False
        return True


class RewardLiveExecutor:
    def __init__(
        self,
        *,
        broker: LiveBroker,
        client: LiveWriteClient | None,
        reward_client: RewardClient | None,
        selector: RewardMarketSelector,
        config: RewardLiveConfig,
    ) -> None:
        self.broker = broker
        self.client = client
        self.reward_client = reward_client
        self.selector = selector
        self.config = config

    def run_cycle(
        self,
        *,
        event_slugs: list[str],
        state_path: Path,
        pnl_path: Path,
    ) -> tuple[RewardLiveStateReport, dict[str, Any]]:
        state_report = self._load_state(state_path)
        now = datetime.now(timezone.utc)
        market_state_map = {market.market_slug: market for market in state_report.markets}
        self._decay_cooldowns(market_state_map)
        self._sync_known_orders(market_state_map, now)
        selected = self.selector.select(
            event_slugs=event_slugs,
            state_by_market=market_state_map,
            config=self.config,
        )
        selected = [
            candidate
            for candidate in selected
            if RewardMarketSelector._passes_profit_gates(candidate, self.config)
        ]
        halted, halt_reason = self._global_halt(market_state_map)
        if not halted:
            self._reconcile_selection(market_state_map, selected, now)
        actual_reward_summary = self._actual_reward_summary()
        state_report = self._build_state_report(market_state_map, selected, halted, halt_reason, actual_reward_summary, now)
        pnl_report = self._build_pnl_report(state_report)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pnl_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(state_report.model_dump_json(indent=2), encoding="utf-8")
        pnl_path.write_text(json.dumps(pnl_report, indent=2), encoding="utf-8")
        return state_report, pnl_report

    def _load_state(self, state_path: Path) -> RewardLiveStateReport:
        if not state_path.exists():
            return RewardLiveStateReport(generated_ts=datetime.now(timezone.utc), mode="dry_run" if self.config.dry_run else "live")
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        return RewardLiveStateReport.model_validate(raw)

    def _decay_cooldowns(self, market_state_map: dict[str, RewardMarketRuntimeState]) -> None:
        for market in market_state_map.values():
            market.cooldown_remaining_sessions = max(0, market.cooldown_remaining_sessions - 1)

    def _sync_known_orders(self, market_state_map: dict[str, RewardMarketRuntimeState], now: datetime) -> None:
        for market in market_state_map.values():
            for attr in ("inventory_order", "bid_order", "ask_order", "exit_order"):
                order = getattr(market, attr)
                if order is None:
                    continue
                if order.synthetic or self.client is None or self.config.dry_run:
                    continue
                if order.status.lower() in {"filled", "canceled", "cancelled", "expired", "rejected"}:
                    continue
                try:
                    status = self.client.get_order_status(order.order_id)
                except LiveClientError:
                    continue
                self._apply_polled_status(market, order, status, now)
            self._refresh_market_metrics(market, now)

    def _apply_polled_status(
        self,
        market: RewardMarketRuntimeState,
        order: RewardOrderRecord,
        status: LiveOrderStatus,
        now: datetime,
    ) -> None:
        delta = max(0.0, float(status.size_matched) - float(order.matched_size))
        if delta > 0.0:
            fill_price = float(status.avg_price) if status.avg_price is not None else float(order.price)
            self._apply_fill_delta(market, order, delta, fill_price)
        order.matched_size = round(float(status.size_matched), 6)
        order.status = str(status.status)
        order.updated_ts = now
        if status.size_remaining <= 1e-9 or str(status.status).lower() in {"matched", "canceled", "cancelled", "expired"}:
            if order.kind == RewardOrderKind.INVENTORY and market.inventory_shares >= (market.last_candidate.quote_size if market.last_candidate else 0.0):
                market.status = RewardExecutionState.INVENTORY_BUILT
            elif order.kind == RewardOrderKind.EXIT and market.inventory_shares <= 1e-9:
                market.status = RewardExecutionState.CLOSED

    def _apply_fill_delta(
        self,
        market: RewardMarketRuntimeState,
        order: RewardOrderRecord,
        delta_size: float,
        fill_price: float,
    ) -> None:
        if order.side.upper() == "BUY":
            total_cost = market.avg_inventory_cost * market.inventory_shares
            total_cost += delta_size * fill_price
            market.inventory_shares = round(market.inventory_shares + delta_size, 6)
            market.avg_inventory_cost = round(total_cost / max(market.inventory_shares, 1e-9), 6)
            if market.inventory_shares > 0:
                market.status = RewardExecutionState.INVENTORY_BUILT
        else:
            matched = min(delta_size, market.inventory_shares)
            market.spread_realized_usdc = round(
                market.spread_realized_usdc + ((fill_price - market.avg_inventory_cost) * matched),
                6,
            )
            market.daily_realized_pnl_usdc = round(
                market.daily_realized_pnl_usdc + ((fill_price - market.avg_inventory_cost) * matched),
                6,
            )
            market.inventory_shares = round(max(0.0, market.inventory_shares - matched), 6)
            if market.inventory_shares <= 1e-9:
                market.avg_inventory_cost = 0.0

    def _reconcile_selection(
        self,
        market_state_map: dict[str, RewardMarketRuntimeState],
        selected: list[RewardMarketCandidate],
        now: datetime,
    ) -> None:
        selected_slugs = {candidate.market_slug for candidate in selected}
        for slug, market in market_state_map.items():
            if slug in selected_slugs:
                continue
            if market.status == RewardExecutionState.QUOTING:
                self._cancel_quote_orders(market)
                market.status = RewardExecutionState.PAUSED
                market.last_halt_reason = "not_selected"
            self._refresh_market_metrics(market, now)

        total_net = sum(market.net_after_reward_usdc for market in market_state_map.values())
        for candidate in selected:
            market = market_state_map.get(candidate.market_slug)
            if market is None:
                market = RewardMarketRuntimeState(
                    event_slug=candidate.event_slug,
                    market_slug=candidate.market_slug,
                    question=candidate.question,
                    yes_token_id=candidate.yes_token_id,
                    no_token_id=candidate.no_token_id,
                )
                market_state_map[candidate.market_slug] = market
            market.question = candidate.question
            market.last_candidate = candidate
            market.latest_mid_price = candidate.midpoint
            market.capital_in_use_usdc = candidate.capital_in_use_usdc
            market.last_update_ts = now
            market.selection_count += 1
            market.cooldown_remaining_sessions = self.config.cooldown_sessions
            market.last_selected_ts = now
            market.status = RewardExecutionState.SELECTED if market.inventory_shares <= 0 else market.status

            if total_net <= -abs(self.config.max_daily_loss):
                market.status = RewardExecutionState.PAUSED
                market.last_halt_reason = "daily_loss_limit"
                self._cancel_quote_orders(market)
                self._refresh_market_metrics(market, now)
                continue

            if market.net_after_reward_usdc <= -abs(self.config.max_drawdown_per_market):
                self._cancel_quote_orders(market)
                market.last_halt_reason = "market_drawdown_limit"
                self._submit_exit_if_needed(market, now)
                self._refresh_market_metrics(market, now)
                continue

            target_inventory = candidate.quote_size
            if market.inventory_shares < target_inventory - 1e-9:
                self._ensure_inventory(market, candidate, now)
            if market.inventory_shares >= target_inventory - 1e-9 and candidate.reward_minus_drawdown_per_hour > 0:
                self._ensure_quotes(market, candidate, now)
            else:
                self._cancel_quote_orders(market)
                market.status = RewardExecutionState.PAUSED
                market.last_halt_reason = "reward_not_profitable"
            self._refresh_market_metrics(market, now)

    def _ensure_inventory(self, market: RewardMarketRuntimeState, candidate: RewardMarketCandidate, now: datetime) -> None:
        if market.inventory_order and market.inventory_order.status.lower() not in {"filled", "canceled", "cancelled", "expired", "rejected"}:
            return
        missing = round(max(candidate.quote_size - market.inventory_shares, 0.0), 6)
        if missing <= 1e-9:
            return
        order = self._submit_order(
            market=market,
            kind=RewardOrderKind.INVENTORY,
            side="BUY",
            price=candidate.best_ask,
            size=missing,
            now=now,
        )
        market.inventory_order = order
        market.status = RewardExecutionState.SELECTED

    def _ensure_quotes(self, market: RewardMarketRuntimeState, candidate: RewardMarketCandidate, now: datetime) -> None:
        ask_size = min(candidate.quote_size, market.inventory_shares)
        if ask_size > 1e-9:
            market.ask_order = self._refresh_quote_order(
                market=market,
                existing=market.ask_order,
                kind=RewardOrderKind.ASK,
                side="SELL",
                price=candidate.quote_ask,
                size=ask_size,
                now=now,
            )
        market.bid_order = self._refresh_quote_order(
            market=market,
            existing=market.bid_order,
            kind=RewardOrderKind.BID,
            side="BUY",
            price=candidate.quote_bid,
            size=candidate.quote_size,
            now=now,
        )
        market.status = RewardExecutionState.QUOTING
        if market.reward_zone_started_ts is None:
            market.reward_zone_started_ts = now

    def _refresh_quote_order(
        self,
        *,
        market: RewardMarketRuntimeState,
        existing: RewardOrderRecord | None,
        kind: RewardOrderKind,
        side: str,
        price: float,
        size: float,
        now: datetime,
    ) -> RewardOrderRecord:
        if existing and existing.status.lower() not in {"filled", "canceled", "cancelled", "expired", "rejected"}:
            if abs(existing.price - price) * 100.0 <= self.config.replace_drift_cents:
                return existing
            self._cancel_order(existing)
        return self._submit_order(market=market, kind=kind, side=side, price=price, size=size, now=now)

    def _submit_order(
        self,
        *,
        market: RewardMarketRuntimeState,
        kind: RewardOrderKind,
        side: str,
        price: float,
        size: float,
        now: datetime,
    ) -> RewardOrderRecord:
        if self.config.dry_run:
            order = RewardOrderRecord(
                kind=kind,
                side=side,
                order_id=f"dryrun-{kind.value}-{market.market_slug}-{uuid4().hex[:8]}",
                requested_size=size,
                price=price,
                matched_size=size if kind in {RewardOrderKind.INVENTORY, RewardOrderKind.EXIT} else 0.0,
                status="filled" if kind in {RewardOrderKind.INVENTORY, RewardOrderKind.EXIT} else "submitted",
                synthetic=True,
                created_ts=now,
                updated_ts=now,
            )
            if order.matched_size > 0:
                self._apply_fill_delta(market, order, order.matched_size, price)
            return order

        intent = OrderIntent(
            intent_id=str(uuid4()),
            candidate_id=f"reward-live-{market.market_slug}",
            mode=OrderMode.LIVE,
            market_slug=market.market_slug,
            token_id=market.yes_token_id,
            side=side,
            order_type=OrderType.LIMIT,
            size=size,
            limit_price=price,
            max_notional_usd=round(size * price, 6),
            ts=now,
            metadata={"reward_order_kind": kind.value},
        )
        report: ExecutionReport = self.broker.submit_limit_order(intent)
        order = RewardOrderRecord(
            kind=kind,
            side=side,
            order_id=str(report.metadata.get("live_order_id") or f"live-{kind.value}-{uuid4().hex[:8]}"),
            requested_size=size,
            price=price,
            matched_size=float(report.filled_size or 0.0),
            status=report.status.value if isinstance(report.status, OrderStatus) else str(report.status),
            synthetic=False,
            created_ts=now,
            updated_ts=now,
        )
        if order.matched_size > 0:
            self._apply_fill_delta(market, order, order.matched_size, float(report.avg_fill_price or price))
        return order

    def _cancel_order(self, order: RewardOrderRecord) -> None:
        if order.synthetic or self.config.dry_run:
            order.status = "canceled"
            return
        if self.broker.cancel_order(order.order_id):
            order.status = "canceled"

    def _cancel_quote_orders(self, market: RewardMarketRuntimeState) -> None:
        for attr in ("bid_order", "ask_order"):
            order = getattr(market, attr)
            if order is None:
                continue
            if order.status.lower() in {"filled", "canceled", "cancelled", "expired", "rejected"}:
                continue
            self._cancel_order(order)
        if market.reward_zone_started_ts is not None:
            market.reward_zone_elapsed_sec = round(
                market.reward_zone_elapsed_sec + (datetime.now(timezone.utc) - market.reward_zone_started_ts).total_seconds(),
                6,
            )
            market.reward_zone_started_ts = None

    def _submit_exit_if_needed(self, market: RewardMarketRuntimeState, now: datetime) -> None:
        if market.inventory_shares <= 1e-9:
            market.status = RewardExecutionState.CLOSED
            return
        if market.exit_order and market.exit_order.status.lower() not in {"filled", "canceled", "cancelled", "expired", "rejected"}:
            return
        exit_price = market.latest_mid_price or market.avg_inventory_cost or 0.5
        market.exit_order = self._submit_order(
            market=market,
            kind=RewardOrderKind.EXIT,
            side="SELL",
            price=max(min(exit_price, 0.99), 0.01),
            size=market.inventory_shares,
            now=now,
        )
        market.status = RewardExecutionState.EXITING if market.inventory_shares > 1e-9 else RewardExecutionState.CLOSED

    def _refresh_market_metrics(self, market: RewardMarketRuntimeState, now: datetime) -> None:
        candidate = market.last_candidate
        if candidate is None:
            return
        effective_reward_sec = market.reward_zone_elapsed_sec
        if market.reward_zone_started_ts is not None:
            effective_reward_sec += max((now - market.reward_zone_started_ts).total_seconds(), 0.0)
        market.reward_accrued_estimate_usdc = round(
            candidate.implied_reward_ev_daily_usdc * (effective_reward_sec / 86400.0),
            6,
        )
        if market.latest_mid_price is None:
            market.latest_mid_price = candidate.midpoint
        market.inventory_mtm_pnl_usdc = round(
            (float(market.latest_mid_price or 0.0) - market.avg_inventory_cost) * market.inventory_shares,
            6,
        )
        market.net_after_reward_usdc = round(
            market.reward_accrued_estimate_usdc + market.inventory_mtm_pnl_usdc + market.spread_realized_usdc,
            6,
        )
        market.capital_in_use_usdc = round(max(candidate.capital_in_use_usdc, market.inventory_shares * market.avg_inventory_cost), 6)

    def _global_halt(self, market_state_map: dict[str, RewardMarketRuntimeState]) -> tuple[bool, str | None]:
        total_net = sum(market.net_after_reward_usdc for market in market_state_map.values())
        if total_net <= -abs(self.config.max_daily_loss):
            return True, "daily_loss_limit"
        return False, None

    def _actual_reward_summary(self) -> dict[str, Any] | None:
        if self.reward_client is None:
            return None
        try:
            return self.reward_client.get_rewards_summary()
        except RewardClientError:
            return None

    def _build_state_report(
        self,
        market_state_map: dict[str, RewardMarketRuntimeState],
        selected: list[RewardMarketCandidate],
        halted: bool,
        halt_reason: str | None,
        actual_reward_summary: dict[str, Any] | None,
        now: datetime,
    ) -> RewardLiveStateReport:
        markets = list(market_state_map.values())
        return RewardLiveStateReport(
            generated_ts=now,
            mode="dry_run" if self.config.dry_run else "live",
            halted=halted,
            halt_reason=halt_reason,
            selected_market_count=len(selected),
            active_quote_market_count=sum(
                1
                for market in markets
                if market.status == RewardExecutionState.QUOTING
            ),
            total_capital_in_use_usdc=round(sum(market.capital_in_use_usdc for market in markets), 6),
            total_reward_accrued_estimate_usdc=round(sum(market.reward_accrued_estimate_usdc for market in markets), 6),
            total_inventory_mtm_pnl_usdc=round(sum(market.inventory_mtm_pnl_usdc for market in markets), 6),
            total_spread_realized_usdc=round(sum(market.spread_realized_usdc for market in markets), 6),
            total_net_after_reward_usdc=round(sum(market.net_after_reward_usdc for market in markets), 6),
            actual_reward_summary=actual_reward_summary,
            markets=sorted(markets, key=lambda market: market.net_after_reward_usdc, reverse=True),
        )

    def _build_pnl_report(self, state_report: RewardLiveStateReport) -> dict[str, Any]:
        per_market = []
        for market in state_report.markets:
            per_market.append(
                {
                    "event_slug": market.event_slug,
                    "market_slug": market.market_slug,
                    "status": market.status.value,
                    "inventory_shares": round(market.inventory_shares, 6),
                    "avg_inventory_cost": round(market.avg_inventory_cost, 6),
                    "capital_in_use_usdc": round(market.capital_in_use_usdc, 6),
                    "reward_accrued_estimate_usdc": round(market.reward_accrued_estimate_usdc, 6),
                    "inventory_mtm_pnl_usdc": round(market.inventory_mtm_pnl_usdc, 6),
                    "spread_realized_usdc": round(market.spread_realized_usdc, 6),
                    "net_after_reward_usdc": round(market.net_after_reward_usdc, 6),
                    "hours_in_reward_zone": round(
                        (
                            market.reward_zone_elapsed_sec
                            + (max((state_report.generated_ts - market.reward_zone_started_ts).total_seconds(), 0.0) if market.reward_zone_started_ts else 0.0)
                        ) / 3600.0,
                        6,
                    ),
                }
            )
        return {
            "generated_ts": state_report.generated_ts.isoformat(),
            "mode": state_report.mode,
            "summary": {
                "capital_in_use_usdc": state_report.total_capital_in_use_usdc,
                "reward_accrued_estimate_usdc": state_report.total_reward_accrued_estimate_usdc,
                "inventory_mtm_pnl_usdc": state_report.total_inventory_mtm_pnl_usdc,
                "spread_realized_usdc": state_report.total_spread_realized_usdc,
                "net_after_reward_usdc": state_report.total_net_after_reward_usdc,
                "active_quote_market_count": state_report.active_quote_market_count,
                "halted": state_report.halted,
                "halt_reason": state_report.halt_reason,
            },
            "actual_reward_summary": state_report.actual_reward_summary,
            "per_market": per_market,
        }


def build_reward_live_executor(
    *,
    gamma_host: str,
    clob_host: str,
    broker: LiveBroker,
    client: LiveWriteClient | None,
    reward_client: RewardClient | None,
    config: RewardLiveConfig,
) -> RewardLiveExecutor:
    return RewardLiveExecutor(
        broker=broker,
        client=client,
        reward_client=reward_client,
        selector=RewardMarketSelector(gamma_host=gamma_host, clob_host=clob_host),
        config=config,
    )
