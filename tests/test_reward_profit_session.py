from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.live.reward_profit_session import (
    RewardMarketStatus,
    RewardMarketState,
    RewardOrderManager,
    RewardProfitCandidate,
    RewardProfitConfig,
    RewardProfitSelector,
    RewardProfitSessionEngine,
    _sellable_token_size,
)
from src.live.client import LiveOpenOrder, LiveOrderStatus
from scripts.calibrate_reward_profit_model import build_calibration_report


class _FakeRewardClient:
    def __init__(self, user_earned_usd: float = 0.0):
        self.user_earned_usd = user_earned_usd

    def get_rewards_summary(self) -> dict:
        return {
            "epoch_id": "epoch-1",
            "user_earned_usd": self.user_earned_usd,
        }


class _CountingOrderManager(RewardOrderManager):
    def __init__(self):
        super().__init__(live=False)
        self.inventory_calls = 0
        self.quote_calls = 0
        self.cancel_calls: list[str | None] = []

    def build_inventory(self, market, candidate):
        self.inventory_calls += 1
        return candidate.quote_size, candidate.best_ask

    def ensure_quote_orders(self, market, candidate):
        self.quote_calls += 1
        return market.bid_order_id or "dry-bid-1", market.ask_order_id or "dry-ask-1"

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        return True


class _LiveLifecycleOrderManager(RewardOrderManager):
    def __init__(self):
        super().__init__(live=True)
        self.cancel_calls: list[str | None] = []
        self.quote_calls = 0
        self.statuses: dict[str, LiveOrderStatus] = {}
        self.submitted: list[tuple[str, str, float, float]] = []
        self.token_balances: dict[str, float] = {}
        self.open_orders: dict[str, list[LiveOpenOrder]] = {}
        self.all_open_orders: list[LiveOpenOrder] = []

    def ensure_quote_orders(self, market, candidate):
        self.quote_calls += 1
        bid_id = market.bid_order_id
        ask_id = market.ask_order_id
        allow_bid_with_inventory = bool(getattr(market, "allow_bid_with_inventory", False))
        sell_size = _sellable_token_size(
            market.inventory_shares,
            dust_shares=float(getattr(market, "inventory_dust_shares", 0.0001) or 0.0001),
            min_order_size_shares=float(getattr(market, "min_order_size_shares", 0.0) or 0.0),
        )
        has_tradeable_inventory = sell_size > 0.0
        if bid_id is None and (
            market.inventory_shares <= 1e-9 or allow_bid_with_inventory or not has_tradeable_inventory
        ):
            bid_id = f"live-bid-{self.quote_calls}"
            self.submitted.append(("bid", bid_id, candidate.quote_bid, candidate.quote_size))
            self.statuses.setdefault(
                bid_id,
                LiveOrderStatus(
                    order_id=bid_id,
                    status="open",
                    size_matched=0.0,
                    size_remaining=candidate.quote_size,
                    avg_price=None,
                ),
            )
        if bid_id is not None and has_tradeable_inventory and not allow_bid_with_inventory:
            self.cancel_order(bid_id)
            bid_id = None
        if ask_id is None and has_tradeable_inventory:
            ask_id = f"live-ask-{self.quote_calls}"
            ask_size = sell_size
            market.sell_cover_size = round(ask_size, 6)
            self.submitted.append(("ask", ask_id, candidate.quote_ask, ask_size))
            self.statuses.setdefault(
                ask_id,
                LiveOrderStatus(
                    order_id=ask_id,
                    status="open",
                    size_matched=0.0,
                    size_remaining=ask_size,
                    avg_price=None,
                ),
            )
        return bid_id, ask_id

    def get_order_status(self, order_id):
        return self.statuses.get(order_id)

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        return True

    def get_token_balance(self, token_id):
        if token_id not in self.token_balances:
            raise RuntimeError("no fake live balance")
        return self.token_balances.get(token_id, 0.0)

    def get_open_orders(self, token_id):
        return list(self.open_orders.get(token_id, []))

    def get_all_open_orders(self):
        return list(self.all_open_orders)


class _OrderVersionMismatchManager(RewardOrderManager):
    def __init__(self):
        super().__init__(live=True)
        self.quote_calls = 0
        self.cancel_calls: list[str | None] = []

    def ensure_quote_orders(self, market, candidate):
        self.quote_calls += 1
        market.last_order_error = "submit_order failed: PolyApiException[error_message={'error': 'order_version_mismatch'}]"
        return None, None

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        return True

    def get_token_balance(self, token_id):
        return 0.0

    def get_open_orders(self, token_id):
        return []


def _candidate(
    *,
    market_slug: str,
    event_slug: str = "event-1",
    capital: float = 100.0,
    quote_size: float = 100.0,
    best_bid: float = 0.35,
    best_ask: float = 0.40,
    reward_hour: float = 0.2,
    drawdown_hour: float = 0.05,
    spread_capture_hour: float = 0.0,
    volume_num: float = 10000.0,
    neg_risk: bool = False,
    tick_size: str | None = None,
) -> RewardProfitCandidate:
    midpoint = (best_bid + best_ask) / 2.0
    return RewardProfitCandidate(
        event_slug=event_slug,
        event_title=event_slug,
        market_slug=market_slug,
        question=market_slug,
        token_id=f"tok-{market_slug}",
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        quote_bid=best_bid,
        quote_ask=best_ask,
        quote_size=quote_size,
        capital_basis_usdc=capital,
        reward_daily_rate=24.0,
        rewards_max_spread_cents=3.5,
        volume_num=volume_num,
        neg_risk=neg_risk,
        tick_size=tick_size,
        expected_reward_per_hour_lower=reward_hour,
        expected_drawdown_cost_per_hour=drawdown_hour,
        reward_minus_drawdown_per_hour=round(reward_hour - drawdown_hour, 6),
        reward_per_dollar_inventory_per_hour=round(reward_hour / capital, 8),
        immediate_entry_cost_usdc=round((best_ask - best_bid) * quote_size, 6),
        immediate_entry_cost_pct=round((((best_ask - best_bid) * quote_size) / capital), 8) if capital > 0 else 0.0,
        break_even_hours=round((((best_ask - best_bid) * quote_size) / reward_hour), 6) if reward_hour > 0 else 999999.0,
        true_break_even_hours=round((((best_ask - best_bid) * quote_size) / max(reward_hour + spread_capture_hour - drawdown_hour, 1e-9)), 6) if (reward_hour + spread_capture_hour) > drawdown_hour else 999999.0,
        expected_spread_capture_per_hour=spread_capture_hour,
        expected_net_edge_per_hour=round(reward_hour + spread_capture_hour - drawdown_hour, 6),
        kelly_raw_fraction=0.0,
        kelly_position_shares=round(quote_size, 6),
        sizing_mode="reward_min_size",
        effective_quote_size=round(quote_size, 6),
        sizing_reason="KELLY_DISABLED_REWARD_MIN_SIZE",
        liquidity_factor=0.8,
        activity_factor=0.8,
    )


class RewardProfitSelectorTests(unittest.TestCase):
    def test_selector_respects_total_and_per_market_caps(self) -> None:
        selector = RewardProfitSelector()
        candidates = [
            _candidate(market_slug="a", event_slug="event-a", capital=180.0, reward_hour=0.5, drawdown_hour=0.1),
            _candidate(market_slug="b", event_slug="event-b", capital=120.0, reward_hour=0.4, drawdown_hour=0.1),
            _candidate(market_slug="c", event_slug="event-c", capital=90.0, reward_hour=0.3, drawdown_hour=0.05),
        ]

        selected = selector.select(
            candidates,
            capital_limit_usdc=300.0,
            per_market_cap_usdc=120.0,
            max_markets=2,
            max_markets_per_event=1,
        )

        self.assertEqual([item.market_slug for item in selected], ["b", "c"])
        self.assertLessEqual(sum(item.capital_basis_usdc for item in selected), 300.0)

    def test_selector_reports_selection_block_reasons(self) -> None:
        selector = RewardProfitSelector()
        selected, reasons = selector.select_with_reasons(
            [_candidate(market_slug="too-big", capital=80.0)],
            capital_limit_usdc=30.0,
            per_market_cap_usdc=20.0,
            max_markets=1,
            max_markets_per_event=1,
        )

        self.assertEqual(selected, [])
        self.assertEqual(reasons, {"SELECT_PER_MARKET_CAP": 1})


class RewardProfitSessionEngineTests(unittest.TestCase):
    def _engine(self, tmpdir: str, *, reward_client: _FakeRewardClient | None = None, order_manager: _CountingOrderManager | None = None, max_drawdown: float = 10.0) -> RewardProfitSessionEngine:
        config = RewardProfitConfig(
            out_dir=tmpdir,
            state_path=str(Path(tmpdir) / "reward_profit_state_latest.json"),
            pnl_path=str(Path(tmpdir) / "reward_profit_pnl_latest.json"),
            capital_limit_usdc=300.0,
            per_market_cap_usdc=120.0,
            max_markets=2,
            max_markets_per_event=1,
            max_drawdown_per_market=max_drawdown,
            max_daily_loss=10.0,
            max_entry_cost_usdc=10.0,
            max_entry_cost_pct=1.0,
            max_break_even_hours=100.0,
            entry_mode="inventory_first",
            live=False,
        )
        return RewardProfitSessionEngine(
            config,
            reward_client_factory=lambda dry_run: reward_client,
            order_manager=order_manager or _CountingOrderManager(),
            registry_provider=lambda cfg: {"events": []},
        )

    def test_pause_releases_capital(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir)
            cycle_ts = datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc)
            state = engine.run_cycle(scanned_candidates=[_candidate(market_slug="m1")], cycle_ts=cycle_ts)
            self.assertGreater(state.markets["m1"].capital_in_use_usdc, 0.0)

            state = engine.run_cycle(state, scanned_candidates=[], cycle_ts=cycle_ts + timedelta(minutes=1))
            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.PAUSED.value)
            self.assertEqual(state.markets["m1"].capital_in_use_usdc, 0.0)

    def test_reward_actual_and_estimate_are_tracked_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_client = _FakeRewardClient(user_earned_usd=5.0)
            engine = self._engine(tmpdir, reward_client=reward_client)
            cycle_ts = datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc)
            state = engine.run_cycle(scanned_candidates=[_candidate(market_slug="m1")], cycle_ts=cycle_ts)
            reward_client.user_earned_usd = 7.0
            state = engine.run_cycle(
                state,
                scanned_candidates=[_candidate(market_slug="m1")],
                cycle_ts=cycle_ts + timedelta(hours=1),
            )
            pnl = engine._build_pnl_report(state)
            market = pnl["markets"][0]

            self.assertGreater(market["reward_accrued_estimate_usdc"], 0.0)
            self.assertGreater(market["reward_accrued_actual_usdc"], 0.0)

    def test_mark_to_market_uses_bid_against_entry_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", quote_size=100.0, best_bid=0.35, best_ask=0.40, capital=40.0)],
                cycle_ts=datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc),
            )
            market = state.markets["m1"]
            self.assertAlmostEqual(market.inventory_mtm_pnl_usdc, -5.0, places=6)

    def test_profit_gate_rejects_low_net_reward_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _CountingOrderManager()
            engine = self._engine(tmpdir, order_manager=manager)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="weak", reward_hour=0.021, drawdown_hour=0.011)],
                cycle_ts=datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(state.markets, {})
            self.assertEqual(manager.inventory_calls, 0)

    def test_drawdown_limit_closes_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir, max_drawdown=3.0)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", quote_size=100.0, best_bid=0.35, best_ask=0.40, capital=40.0)],
                cycle_ts=datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc),
            )
            market = state.markets["m1"]
            self.assertEqual(market.status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(market.capital_in_use_usdc, 0.0)

    def test_restart_does_not_duplicate_inventory_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _CountingOrderManager()
            engine = self._engine(tmpdir, order_manager=manager)
            cycle_ts = datetime(2026, 4, 24, 0, 0, tzinfo=timezone.utc)
            state = engine.run_cycle(scanned_candidates=[_candidate(market_slug="m1")], cycle_ts=cycle_ts)
            pnl = engine._build_pnl_report(state)
            engine._write_reports(state, pnl)

            manager_2 = _CountingOrderManager()
            engine_2 = self._engine(tmpdir, order_manager=manager_2)
            state_2 = engine_2.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1")],
                cycle_ts=cycle_ts + timedelta(minutes=1),
            )
            self.assertEqual(manager.inventory_calls, 1)
            self.assertEqual(manager_2.inventory_calls, 0)
            self.assertEqual(state_2.markets["m1"].inventory_shares, state.markets["m1"].inventory_shares)

    def test_projected_net_at_horizon_gate_rejects_negative_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _CountingOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=10.0,
                max_daily_loss=10.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                projection_horizon_hours=1.0,
                min_projected_net_at_horizon_usdc=0.05,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(
                market_slug="bad",
                quote_size=100.0,
                best_bid=0.35,
                best_ask=0.40,
                capital=40.0,
                reward_hour=0.2,
                drawdown_hour=0.05,
            )
            state = engine.run_cycle(
                scanned_candidates=[cand],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(manager.inventory_calls, 0)

    def test_projected_net_at_horizon_gate_admits_strong_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=10.0,
                max_daily_loss=10.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                projection_horizon_hours=1.0,
                min_projected_net_at_horizon_usdc=0.05,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(
                market_slug="good",
                quote_size=100.0,
                best_bid=0.399,
                best_ask=0.40,
                capital=40.0,
                reward_hour=0.5,
                drawdown_hour=0.05,
            )
            state = engine.run_cycle(
                scanned_candidates=[cand],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            self.assertEqual(state.selected_market_slugs, ["good"])

    def test_max_true_break_even_gate_uses_net_of_drawdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _CountingOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=10.0,
                max_daily_loss=10.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                max_true_break_even_hours=2.0,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(
                market_slug="optimistic",
                quote_size=100.0,
                best_bid=0.35,
                best_ask=0.40,
                capital=40.0,
                reward_hour=0.6,
                drawdown_hour=0.5,
            )
            self.assertLess(cand.break_even_hours, 100.0)
            self.assertGreater(cand.true_break_even_hours, 2.0)
            state = engine.run_cycle(
                scanned_candidates=[cand],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(manager.inventory_calls, 0)

    def test_stagnation_halt_closes_market_with_no_fills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=100.0,
                max_daily_loss=100.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                max_quoting_hours_without_fills=0.5,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1")
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.QUOTING.value)
            state = engine.run_cycle(
                state,
                scanned_candidates=[cand],
                cycle_ts=cycle_ts + timedelta(hours=1),
            )
            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(state.markets["m1"].last_exit_reason, "STAGNATION_NO_FILLS")

    def test_dry_run_fill_simulation_realizes_spread_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=100.0,
                max_daily_loss=100.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", spread_capture_hour=1.25)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            state = engine.run_cycle(
                state,
                scanned_candidates=[cand],
                cycle_ts=cycle_ts + timedelta(hours=1),
            )
            market = state.markets["m1"]
            pnl = engine._build_pnl_report(state)

            self.assertGreater(market.simulated_bid_fill_shares, 0.0)
            self.assertGreater(market.simulated_ask_fill_shares, 0.0)
            self.assertAlmostEqual(market.simulated_spread_capture_usdc, 1.25, places=6)
            self.assertAlmostEqual(market.spread_realized_usdc, 1.25, places=6)
            self.assertEqual(pnl["markets"][0]["fill_simulation"]["simulated_spread_capture_usdc"], 1.25)

    def test_actual_reward_zero_streak_closes_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_client = _FakeRewardClient(user_earned_usd=0.0)
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=100.0,
                max_daily_loss=100.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                actual_reward_zero_cycle_limit=1,
                min_daily_reward_for_actual_gate_usdc=1.0,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: reward_client,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", reward_hour=0.5, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            state = engine.run_cycle(
                state,
                scanned_candidates=[cand],
                cycle_ts=cycle_ts + timedelta(hours=1),
            )

            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(state.markets["m1"].last_exit_reason, "ACTUAL_REWARD_ZERO_STREAK")

    def test_adverse_inventory_drift_closes_market_when_reward_does_not_cover_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=100.0,
                max_daily_loss=100.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                min_reward_per_dollar_inventory_per_hour=0.0,
                max_adverse_midpoint_move_cents_per_hour=0.20,
                min_inventory_risk_coverage_ratio=1.10,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            start = _candidate(
                market_slug="m1",
                quote_size=80.0,
                best_bid=0.495,
                best_ask=0.505,
                reward_hour=0.0275,
                drawdown_hour=0.0,
                spread_capture_hour=0.0,
            )
            drifted = _candidate(
                market_slug="m1",
                quote_size=80.0,
                best_bid=0.486,
                best_ask=0.496,
                reward_hour=0.0275,
                drawdown_hour=0.0,
                spread_capture_hour=0.0,
            )
            state = engine.run_cycle(scanned_candidates=[start], cycle_ts=cycle_ts)
            state = engine.run_cycle(
                state,
                scanned_candidates=[drifted],
                cycle_ts=cycle_ts + timedelta(hours=4),
            )
            market = state.markets["m1"]

            self.assertEqual(market.status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(market.last_exit_reason, "ADVERSE_INVENTORY_DRIFT")
            self.assertAlmostEqual(market.adverse_midpoint_move_usdc, 0.72, places=6)

    def test_spread_capture_lifts_candidate_above_horizon_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=10.0,
                max_daily_loss=10.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                min_reward_minus_drawdown_per_hour=0.0,
                projection_horizon_hours=1.0,
                min_projected_net_at_horizon_usdc=1.0,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            without_spread = _candidate(
                market_slug="no_spread",
                quote_size=100.0,
                best_bid=0.35,
                best_ask=0.40,
                capital=40.0,
                reward_hour=0.5,
                drawdown_hour=0.05,
                spread_capture_hour=0.0,
            )
            state = engine.run_cycle(
                scanned_candidates=[without_spread],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            self.assertEqual(state.selected_market_slugs, [])

            with_spread = _candidate(
                market_slug="with_spread",
                quote_size=100.0,
                best_bid=0.35,
                best_ask=0.40,
                capital=40.0,
                reward_hour=0.5,
                drawdown_hour=0.05,
                spread_capture_hour=6.0,
            )
            state = engine.run_cycle(
                scanned_candidates=[with_spread],
                cycle_ts=datetime(2026, 4, 24, 0, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(state.selected_market_slugs, ["with_spread"])

    def test_kelly_sizing_increases_quote_size_above_min(self) -> None:
        selector = RewardProfitSelector(
            use_kelly_sizing=True,
            kelly_fraction_scale=0.25,
            kelly_horizon_hours=1.0,
            max_drawdown_per_market=3.0,
            per_market_cap_usdc=120.0,
        )
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-1",
            "best_bid": 0.38,
            "best_ask": 0.40,
            "rewards_min_size": 10.0,
            "rewards_max_spread": 3.0,
            "clob_rewards": [{"rewardsDailyRate": 500.0}],
            "liquidity_num": 8000.0,
            "volume_num": 20000.0,
            "slug": "test-market",
            "question": "test?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        self.assertIsNotNone(cand)
        self.assertGreater(cand.kelly_raw_fraction, 0.0)
        self.assertGreater(cand.kelly_position_shares, 10.0)
        self.assertGreater(cand.quote_size, 10.0)
        self.assertAlmostEqual(cand.quote_size, cand.kelly_position_shares, places=1)

    def test_kelly_sizing_floors_at_min_reward_size(self) -> None:
        selector = RewardProfitSelector(
            use_kelly_sizing=True,
            kelly_fraction_scale=0.001,
            kelly_horizon_hours=0.1,
            max_drawdown_per_market=100.0,
            per_market_cap_usdc=120.0,
        )
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-2",
            "best_bid": 0.38,
            "best_ask": 0.40,
            "rewards_min_size": 50.0,
            "rewards_max_spread": 3.0,
            "clob_rewards": [{"rewardsDailyRate": 10.0}],
            "liquidity_num": 100.0,
            "volume_num": 100.0,
            "slug": "small-market",
            "question": "small?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        if cand is not None:
            self.assertGreaterEqual(cand.quote_size, 50.0)

    def test_kelly_off_keeps_rewards_min_size(self) -> None:
        selector = RewardProfitSelector(use_kelly_sizing=False, per_market_cap_usdc=120.0)
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-3",
            "best_bid": 0.38,
            "best_ask": 0.40,
            "rewards_min_size": 75.0,
            "rewards_max_spread": 3.0,
            "clob_rewards": [{"rewardsDailyRate": 500.0}],
            "liquidity_num": 8000.0,
            "volume_num": 20000.0,
            "slug": "base-market",
            "question": "base?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        self.assertIsNotNone(cand)
        self.assertAlmostEqual(cand.quote_size, 75.0, places=1)

    def test_quote_improvement_moves_bid_toward_midpoint(self) -> None:
        selector = RewardProfitSelector(
            use_kelly_sizing=False,
            quote_improvement_cents=0.1,
            max_quote_improvement_cost_usdc=0.25,
        )
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-mid",
            "best_bid": 0.462,
            "best_ask": 0.464,
            "rewards_min_size": 50.0,
            "rewards_max_spread": 4.5,
            "clob_rewards": [{"rewardsDailyRate": 211.0}],
            "liquidity_num": 8000.0,
            "volume_num": 20000.0,
            "slug": "midpoint-market",
            "question": "mid?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        self.assertIsNotNone(cand)
        self.assertAlmostEqual(cand.quote_bid, 0.463, places=6)
        self.assertAlmostEqual(cand.quote_ask, 0.464, places=6)
        self.assertAlmostEqual(cand.quote_improvement_cents, 0.1, places=6)
        self.assertAlmostEqual(cand.quote_improvement_cost_usdc, 0.05, places=6)
        self.assertEqual(cand.quote_improvement_reason, "BID_IMPROVED_POST_SELECTION")

    def test_quote_improvement_respects_cost_cap(self) -> None:
        selector = RewardProfitSelector(
            use_kelly_sizing=False,
            quote_improvement_cents=0.1,
            max_quote_improvement_cost_usdc=0.025,
        )
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-cap",
            "best_bid": 0.462,
            "best_ask": 0.464,
            "rewards_min_size": 50.0,
            "rewards_max_spread": 4.5,
            "clob_rewards": [{"rewardsDailyRate": 211.0}],
            "liquidity_num": 8000.0,
            "volume_num": 20000.0,
            "slug": "cap-market",
            "question": "cap?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        self.assertIsNotNone(cand)
        self.assertAlmostEqual(cand.quote_bid, 0.4625, places=6)
        self.assertAlmostEqual(cand.quote_improvement_cents, 0.05, places=6)
        self.assertAlmostEqual(cand.quote_improvement_cost_usdc, 0.025, places=6)

    def test_kelly_capital_respects_per_market_cap(self) -> None:
        selector = RewardProfitSelector(
            use_kelly_sizing=True,
            kelly_fraction_scale=1.0,
            kelly_horizon_hours=1.0,
            max_drawdown_per_market=0.01,
            per_market_cap_usdc=50.0,
        )
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-4",
            "best_bid": 0.49,
            "best_ask": 0.50,
            "rewards_min_size": 10.0,
            "rewards_max_spread": 3.0,
            "clob_rewards": [{"rewardsDailyRate": 1000.0}],
            "liquidity_num": 9000.0,
            "volume_num": 50000.0,
            "slug": "big-edge",
            "question": "big?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        if cand is not None:
            max_shares = 50.0 / 0.50
            self.assertLessEqual(cand.quote_size, max_shares + 1e-3)

    def test_kelly_raw_fraction_is_positive_for_good_market(self) -> None:
        selector = RewardProfitSelector(use_kelly_sizing=False)
        market = {
            "enable_orderbook": True,
            "fees_enabled": False,
            "is_binary_yes_no": True,
            "yes_token_id": "tok-5",
            "best_bid": 0.38,
            "best_ask": 0.40,
            "rewards_min_size": 10.0,
            "rewards_max_spread": 3.0,
            "clob_rewards": [{"rewardsDailyRate": 500.0}],
            "liquidity_num": 8000.0,
            "volume_num": 20000.0,
            "slug": "good-market",
            "question": "good?",
        }
        cand = selector._build_candidate(event_slug="ev", event_title="ev", market=market)
        self.assertIsNotNone(cand)
        self.assertGreater(cand.kelly_raw_fraction, 0.0)

    def test_reports_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1")],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            pnl = engine._build_pnl_report(state)
            engine._write_reports(state, pnl)
            self.assertTrue(Path(engine.config.state_path).exists())
            self.assertTrue(Path(engine.config.pnl_path).exists())
            payload = json.loads(Path(engine.config.pnl_path).read_text(encoding="utf-8"))
            self.assertIn("summary", payload)
            self.assertIn("model_vs_realized", payload["summary"])
            self.assertIn("verified_net_after_reward_and_cost_usdc", payload["summary"])
            self.assertIn("fill_simulation", payload["markets"][0])

    def test_calibration_downshifts_optimistic_maker_take_share(self) -> None:
        pnl_payload = {
            "session_id": "s1",
            "summary": {
                "cycle_index": 20,
                "net_after_reward_and_cost_usdc": -0.25,
                "verified_net_after_reward_and_cost_usdc": -0.25,
            },
            "markets": [
                {
                    "event_slug": "event-1",
                    "market_slug": "m1",
                    "hours_in_reward_zone": 2.0,
                    "simulated_spread_capture_usdc": 0.5,
                    "net_after_reward_and_cost_usdc": -0.25,
                    "selection_metrics": {
                        "expected_spread_capture_per_hour": 1.0,
                        "expected_net_edge_per_hour": 0.8,
                    },
                }
            ],
        }
        report = build_calibration_report(
            state_payload={"session_id": "s1"},
            pnl_payload=pnl_payload,
            current_maker_take_share=0.30,
            min_hours=1.0,
            min_net_usdc=0.0,
        )

        self.assertFalse(report["allow_live"])
        self.assertLess(report["recommended_maker_take_share"], 0.30)
        self.assertEqual(report["markets"][0]["reason"], "NEGATIVE_REALIZED_NET")

    def test_market_intel_filter_blocks_risky_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            intel_path = Path(tmpdir) / "market_intel.json"
            intel_path.write_text(
                json.dumps(
                    {
                        "markets": {
                            "m1": {
                                "risk_score": 0.95,
                                "blocked": False,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                capital_limit_usdc=300.0,
                per_market_cap_usdc=120.0,
                max_markets=2,
                max_markets_per_event=1,
                max_drawdown_per_market=100.0,
                max_daily_loss=100.0,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                enable_market_intel_filter=True,
                market_intel_path=str(intel_path),
                max_market_intel_risk_score=0.70,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", spread_capture_hour=1.0)],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )

            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(state.last_filter_reasons, {"MARKET_INTEL_RISK": 1})

    def test_target_hit_exit_uses_expected_gap_from_market_intel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            intel_path = Path(tmpdir) / "market_intel.json"
            intel_path.write_text(json.dumps({"markets": {"m1": {"expected_gap": 0.10}}}), encoding="utf-8")
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                max_drawdown_per_market=100.0,
                enable_market_intel_filter=True,
                market_intel_path=str(intel_path),
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", best_bid=0.49, best_ask=0.51, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts,
            )
            state = engine.run_cycle(
                state,
                scanned_candidates=[_candidate(market_slug="m1", best_bid=0.575, best_ask=0.595, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts + timedelta(hours=1),
            )

            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(state.markets["m1"].last_exit_reason, "TARGET_HIT")

    def test_volume_spike_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                max_drawdown_per_market=100.0,
                volume_spike_multiple=3.0,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", volume_num=1000.0, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts,
            )
            state = engine.run_cycle(
                state,
                scanned_candidates=[_candidate(market_slug="m1", volume_num=1100.0, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts + timedelta(hours=1),
            )
            state = engine.run_cycle(
                state,
                scanned_candidates=[_candidate(market_slug="m1", volume_num=1500.0, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts + timedelta(hours=2),
            )

            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(state.markets["m1"].last_exit_reason, "VOLUME_EXIT")

    def test_stale_thesis_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="inventory_first",
                max_drawdown_per_market=100.0,
                stale_thesis_hours=1.0,
                stale_thesis_max_price_change=0.02,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", best_bid=0.49, best_ask=0.51, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts,
            )
            state = engine.run_cycle(
                state,
                scanned_candidates=[_candidate(market_slug="m1", best_bid=0.495, best_ask=0.515, spread_capture_hour=1.0)],
                cycle_ts=cycle_ts + timedelta(hours=2),
            )

            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.CLOSED.value)
            self.assertEqual(state.markets["m1"].last_exit_reason, "STALE_THESIS")

    def test_maker_first_initial_cycle_does_not_cross_spread(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _CountingOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", quote_size=100.0, best_bid=0.35, best_ask=0.40)],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            market = state.markets["m1"]

            self.assertEqual(manager.inventory_calls, 0)
            self.assertEqual(market.inventory_shares, 0.0)
            self.assertIsNotNone(market.bid_order_id)
            self.assertIsNone(market.ask_order_id)
            self.assertEqual(market.inventory_mtm_pnl_usdc, 0.0)
            self.assertEqual(market.entry_spread_cost_usdc, 0.0)

    def test_maker_first_bid_fill_creates_inventory_before_ask_quote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(hours=1))
            market = state.markets["m1"]

            self.assertGreater(market.inventory_shares, 0.0)
            self.assertAlmostEqual(market.avg_inventory_cost, 0.35, places=6)
            self.assertIsNotNone(market.ask_order_id)
            self.assertGreater(market.simulated_bid_fill_shares, 0.0)
            self.assertEqual(market.simulated_ask_fill_shares, 0.0)
            self.assertEqual(market.inventory_mtm_pnl_usdc, 0.0)

    def test_cooldown_blocks_market_after_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                max_drawdown_per_market=0.1,
                entry_mode="inventory_first",
                exit_cooldown_minutes=45.0,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", quote_size=100.0, best_bid=0.35, best_ask=0.40)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.CLOSED.value)
            self.assertIsNotNone(state.markets["m1"].cooldown_until_ts)

            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(minutes=1))
            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(state.last_filter_reasons, {"COOLDOWN_AFTER_EXIT": 1})

    def test_not_selected_pause_enters_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir)
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1")
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            state = engine.run_cycle(state, scanned_candidates=[], cycle_ts=cycle_ts + timedelta(minutes=1))
            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.PAUSED.value)
            self.assertIsNotNone(state.markets["m1"].cooldown_until_ts)

            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(minutes=2))
            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(state.last_filter_reasons, {"COOLDOWN_AFTER_EXIT": 1})

    def test_repeat_exit_uses_longer_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                max_drawdown_per_market=0.1,
                entry_mode="inventory_first",
                exit_cooldown_minutes=45.0,
                repeat_exit_cooldown_minutes=90.0,
                max_reentries_per_market=1,
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", quote_size=100.0, best_bid=0.35, best_ask=0.40)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(minutes=46))
            market = state.markets["m1"]
            cooldown_until = datetime.fromisoformat(market.cooldown_until_ts or "")

            self.assertEqual(market.reentry_count, 1)
            self.assertEqual(market.status, RewardMarketStatus.CLOSED.value)
            self.assertAlmostEqual((cooldown_until - (cycle_ts + timedelta(minutes=46))).total_seconds(), 90 * 60, delta=2)

    def test_report_includes_cooldown_and_entry_spread_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                max_drawdown_per_market=0.1,
                entry_mode="inventory_first",
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", quote_size=100.0, best_bid=0.35, best_ask=0.40)],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            pnl = engine._build_pnl_report(state)

            self.assertEqual(pnl["summary"]["cooldown_market_count"], 1)
            self.assertAlmostEqual(pnl["summary"]["total_entry_spread_cost_usdc"], 5.0, places=6)
            self.assertIn("cooldown_until_ts", pnl["markets"][0])
            self.assertAlmostEqual(pnl["markets"][0]["total_entry_spread_cost_usdc"], 5.0, places=6)

    def test_live_run_cancels_open_orders_on_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                cancel_open_orders_on_finish=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            candidate = _candidate(market_slug="m1", spread_capture_hour=1.0)
            manager.token_balances[candidate.token_id] = 0.0
            state = engine.run_cycle(
                scanned_candidates=[candidate],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            bid_id = state.markets["m1"].bid_order_id
            manager.open_orders[candidate.token_id] = [
                LiveOpenOrder(
                    order_id=str(bid_id),
                    side="BUY",
                    price=candidate.quote_bid,
                    size=candidate.quote_size,
                    size_matched=0.0,
                    size_remaining=candidate.quote_size,
                    status="open",
                )
            ]

            self.assertEqual(bid_id, "live-bid-1")
            engine._cancel_open_orders_on_finish(state)

            self.assertIn("live-bid-1", manager.cancel_calls)
            self.assertIsNone(state.markets["m1"].bid_order_id)
            self.assertEqual(state.markets["m1"].status, RewardMarketStatus.PAUSED.value)
            self.assertEqual(state.markets["m1"].last_exit_reason, "RUN_FINISHED_CANCEL_OPEN_ORDERS")

    def test_live_bid_fill_updates_inventory_and_places_ask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            bid_id = state.markets["m1"].bid_order_id
            self.assertIsNotNone(bid_id)
            self.assertIsNone(state.markets["m1"].ask_order_id)

            manager.statuses[bid_id or ""] = LiveOrderStatus(
                order_id=bid_id or "",
                status="open",
                size_matched=6.0,
                size_remaining=94.0,
                avg_price=0.35,
            )
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=30))
            market = state.markets["m1"]

            self.assertAlmostEqual(market.inventory_shares, 6.0, places=6)
            self.assertAlmostEqual(market.avg_inventory_cost, 0.35, places=6)
            self.assertEqual(market.bid_filled_delta, 6.0)
            self.assertIsNotNone(market.ask_order_id)

    def test_live_synced_dust_balance_ignores_stale_bid_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                inventory_dust_shares=0.01,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 0.006811

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            bid_id = state.markets["m1"].bid_order_id
            self.assertIsNotNone(bid_id)

            manager.statuses[bid_id or ""] = LiveOrderStatus(
                order_id=bid_id or "",
                status="open",
                size_matched=50.0,
                size_remaining=0.0,
                avg_price=0.35,
            )
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=30))
            market = state.markets["m1"]

            self.assertEqual(market.inventory_shares, 0.0)
            self.assertEqual(market.avg_inventory_cost, 0.0)
            self.assertEqual(market.bid_filled_delta, 0.0)
            self.assertIsNone(market.ask_order_id)
            self.assertNotIn("ask", [side for side, *_ in manager.submitted])

    def test_live_ask_fill_realizes_spread(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            bid_id = state.markets["m1"].bid_order_id or ""
            manager.statuses[bid_id] = LiveOrderStatus(bid_id, "open", 6.0, 94.0, 0.35)
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=30))
            ask_id = state.markets["m1"].ask_order_id or ""

            manager.statuses[ask_id] = LiveOrderStatus(ask_id, "open", 2.0, 2.0, 0.40)
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=60))
            market = state.markets["m1"]

            self.assertAlmostEqual(market.inventory_shares, 4.0, places=6)
            self.assertAlmostEqual(market.spread_realized_usdc, 0.10, places=6)
            self.assertEqual(market.ask_filled_delta, 2.0)

    def test_live_stale_unfilled_bid_is_canceled_and_requoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                live_order_max_age_sec=120.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            old_bid_id = state.markets["m1"].bid_order_id

            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=121))
            market = state.markets["m1"]

            self.assertIn(old_bid_id, manager.cancel_calls)
            self.assertNotEqual(market.bid_order_id, old_bid_id)
            self.assertEqual(market.last_cancel_reason, "STALE_UNFILLED_BID")
            self.assertEqual(market.requote_count, 1)

    def test_live_price_move_requotes_without_duplicate_bid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                live_requote_price_move_cents=1.0,
                live_order_max_age_sec=0.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            old_bid_id = state.markets["m1"].bid_order_id
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=30))
            self.assertEqual(state.markets["m1"].bid_order_id, old_bid_id)
            self.assertEqual(len([row for row in manager.submitted if row[0] == "bid"]), 1)

            moved = _candidate(market_slug="m1", best_bid=0.34, best_ask=0.39, spread_capture_hour=1.0)
            state = engine.run_cycle(state, scanned_candidates=[moved], cycle_ts=cycle_ts + timedelta(seconds=60))
            market = state.markets["m1"]

            self.assertIn(old_bid_id, manager.cancel_calls)
            self.assertNotEqual(market.bid_order_id, old_bid_id)
            self.assertEqual(market.last_cancel_reason, "REQUOTE_BID_PRICE_MOVED")

    def test_no_fill_requote_limit_pauses_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                live_order_max_age_sec=120.0,
                max_no_fill_requotes=0,
                no_fill_cooldown_minutes=30.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            old_bid_id = state.markets["m1"].bid_order_id

            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=121))
            market = state.markets["m1"]

            self.assertIn(old_bid_id, manager.cancel_calls)
            self.assertEqual(market.status, RewardMarketStatus.PAUSED.value)
            self.assertEqual(market.last_exit_reason, "NO_FILL_REQUOTE_LIMIT")
            self.assertIsNotNone(market.cooldown_until_ts)

    def test_live_order_version_mismatch_pauses_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _OrderVersionMismatchManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                no_fill_cooldown_minutes=30.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)

            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", spread_capture_hour=1.0)],
                cycle_ts=cycle_ts,
            )
            market = state.markets["m1"]

            self.assertEqual(manager.quote_calls, 1)
            self.assertIsNone(market.bid_order_id)
            self.assertEqual(market.status, RewardMarketStatus.PAUSED.value)
            self.assertEqual(market.last_exit_reason, "ORDER_VERSION_MISMATCH")
            self.assertIsNotNone(market.cooldown_until_ts)

    def test_finish_keeps_ask_when_inventory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                cycles=2,
                interval_sec=0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cycle_ts = datetime(2026, 4, 24, tzinfo=timezone.utc)
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=cycle_ts)
            bid_id = state.markets["m1"].bid_order_id or ""
            manager.statuses[bid_id] = LiveOrderStatus(bid_id, "filled", 6.0, 0.0, 0.35)
            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=cycle_ts + timedelta(seconds=30))
            ask_id = state.markets["m1"].ask_order_id

            engine._cancel_open_orders_on_finish(state)
            market = state.markets["m1"]

            self.assertIsNone(market.bid_order_id)
            self.assertEqual(market.ask_order_id, ask_id)
            self.assertEqual(market.last_exit_reason, "RUN_FINISHED_KEEP_REDUCE_ONLY_ASK")
            self.assertEqual(market.finish_action, "PROTECT_INVENTORY_SELL_ONLY")
            self.assertEqual(market.protected_sell_order_id, ask_id)
            self.assertEqual(market.preserved_sell_count, 1)
            self.assertNotIn(ask_id, manager.cancel_calls)

    def test_live_sync_inventory_blocks_buy_and_places_sell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 12.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertEqual(market.inventory_mode, "sell_only")
            self.assertEqual(market.buy_block_reason, "INVENTORY_PRESENT")
            self.assertIsNone(market.bid_order_id)
            self.assertIsNotNone(market.ask_order_id)
            self.assertEqual([row[0] for row in manager.submitted], ["ask"])
            self.assertAlmostEqual(market.sell_cover_size, 11.9999, places=6)

    def test_live_sub_minimum_inventory_is_residue_not_sell_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                inventory_dust_shares=0.01,
                min_live_order_size_shares=5.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 0.01306

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertEqual(market.inventory_mode, "normal")
            self.assertIsNone(market.buy_block_reason)
            self.assertAlmostEqual(market.sell_cover_size, 0.0, places=6)
            self.assertEqual([row[0] for row in manager.submitted], ["bid"])
            self.assertIsNone(market.ask_order_id)

    def test_balanced_inventory_policy_treats_existing_inventory_as_selected_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                inventory_policy="balanced",
                max_inventory_shares_per_market=100.0,
                max_inventory_usdc_per_market=100.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", quote_size=10.0, best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 50.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertEqual(market.inventory_mode, "balanced")
            self.assertTrue(market.allow_bid_with_inventory)
            self.assertIsNone(market.buy_block_reason)
            self.assertIsNotNone(market.bid_order_id)
            self.assertIsNotNone(market.ask_order_id)
            self.assertEqual([row[0] for row in manager.submitted], ["bid", "ask"])

    def test_dry_run_does_not_call_live_account_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=False,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )

            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", spread_capture_hour=1.0)],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            pnl = engine._build_pnl_report(state)

            self.assertEqual(manager.quote_calls, 0)
            self.assertEqual(pnl["summary"]["account_sync_enabled"], False)
            self.assertEqual(pnl["summary"]["account_sync_mode"], "dry_run_disabled")

    def test_live_cancels_account_buy_orders_outside_selected_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 0.0
            manager.all_open_orders = [
                LiveOpenOrder("old-buy", "BUY", 0.41, 50.0, 0.0, 50.0, "open", token_id="tok-old"),
                LiveOpenOrder("selected-buy", "BUY", 0.35, 50.0, 0.0, 50.0, "open", token_id=cand.token_id),
                LiveOpenOrder("other-sell", "SELL", 0.50, 50.0, 0.0, 50.0, "open", token_id="tok-other"),
            ]

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            pnl = engine._build_pnl_report(state)

            self.assertIn("old-buy", manager.cancel_calls)
            self.assertNotIn("selected-buy", manager.cancel_calls)
            self.assertEqual(state.account_canceled_unselected_buy_count, 1)
            self.assertEqual(pnl["summary"]["account_canceled_unselected_buy_count"], 1)
            self.assertEqual(pnl["summary"]["account_open_sell_order_count"], 1)

    def test_live_sync_cancels_open_buy_when_inventory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 7.0
            manager.open_orders[cand.token_id] = [
                LiveOpenOrder("buy-1", "BUY", 0.35, 10.0, 0.0, 10.0, "open")
            ]

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertIn("buy-1", manager.cancel_calls)
            self.assertEqual(market.buy_block_reason, "INVENTORY_PRESENT_CANCEL_BUY")
            self.assertIsNone(market.bid_order_id)
            self.assertIsNotNone(market.ask_order_id)

    def test_live_sync_rebuilds_multiple_sells_as_one_cover_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 9.0
            manager.open_orders[cand.token_id] = [
                LiveOpenOrder("sell-1", "SELL", 0.40, 4.0, 0.0, 4.0, "open"),
                LiveOpenOrder("sell-2", "SELL", 0.40, 5.0, 0.0, 5.0, "open"),
            ]

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertIn("sell-1", manager.cancel_calls)
            self.assertIn("sell-2", manager.cancel_calls)
            self.assertIsNotNone(market.ask_order_id)
            self.assertEqual([row[0] for row in manager.submitted], ["ask"])
            self.assertAlmostEqual(market.sell_cover_size, 8.9999, places=6)

    def test_live_inventory_share_limit_blocks_new_buy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                max_inventory_shares_per_market=50.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", quote_size=60.0, capital=24.0, best_bid=0.35, best_ask=0.40)
            manager.token_balances[cand.token_id] = 0.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))

            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(state.last_filter_reasons["INVENTORY_LIMIT"], 1)
            self.assertEqual(manager.submitted, [])

    def test_live_inventory_usdc_limit_blocks_new_buy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
                max_inventory_usdc_per_market=10.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", quote_size=50.0, capital=20.0, best_bid=0.35, best_ask=0.40)
            manager.token_balances[cand.token_id] = 0.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))

            self.assertEqual(state.selected_market_slugs, [])
            self.assertEqual(state.last_filter_reasons["INVENTORY_LIMIT"], 1)
            self.assertEqual(manager.submitted, [])

    def test_report_includes_live_order_lifecycle_and_sizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                entry_mode="maker_first",
                live=True,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            state = engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1", spread_capture_hour=1.0)],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )
            pnl = engine._build_pnl_report(state)
            lifecycle = pnl["markets"][0]["live_order_lifecycle"]

            self.assertEqual(pnl["summary"]["open_bid_order_count"], 1)
            self.assertIn("open_bid_order_id", lifecycle)
            self.assertEqual(lifecycle["sizing_mode"], "reward_min_size")
            self.assertGreater(lifecycle["effective_quote_size"], 0.0)

    def test_report_includes_scan_diagnostics_funnel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                live=False,
            )
            valid_market = {
                "enable_orderbook": True,
                "fees_enabled": False,
                "is_binary_yes_no": True,
                "yes_token_id": "tok-yes",
                "best_bid": 0.35,
                "best_ask": 0.36,
                "rewards_min_size": 10.0,
                "clob_rewards": [{"rewardsDailyRate": 100.0}],
                "rewards_max_spread": 3.5,
                "volume_num": 10000.0,
                "liquidity_num": 10000.0,
                "slug": "valid",
                "question": "valid?",
            }
            registry = {
                "summary": {"events_seen": 3, "markets_seen": 50},
                "events": [
                    {
                        "slug": "event-1",
                        "title": "event 1",
                        "markets": [
                            valid_market,
                            {**valid_market, "slug": "fee-enabled", "fees_enabled": True},
                            {**valid_market, "slug": "no-book", "enable_orderbook": False},
                            {**valid_market, "slug": "no-reward", "clob_rewards": []},
                        ],
                    }
                ],
            }
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: registry,
            )

            state = engine.run_cycle(cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            pnl = engine._build_pnl_report(state)
            diagnostics = pnl["summary"]["scan_diagnostics"]

            self.assertEqual(diagnostics["raw_markets_seen"], 50)
            self.assertEqual(diagnostics["registry_markets"], 4)
            self.assertEqual(diagnostics["scored_candidates"], 2)
            self.assertEqual(diagnostics["eligible_candidates"], 2)
            self.assertEqual(diagnostics["selected_markets"], 1)
            self.assertEqual(diagnostics["candidate_prefilter_reasons"]["NO_ORDERBOOK"], 1)
            self.assertEqual(diagnostics["candidate_prefilter_reasons"]["NO_CLOB_REWARD_RATE"], 1)
            self.assertNotIn("FEES_ENABLED", diagnostics["candidate_prefilter_reasons"])

    def test_report_warns_when_scanned_candidate_count_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "s.json"),
                pnl_path=str(Path(tmpdir) / "p.json"),
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            state = engine._load_state()
            state.last_scan_diagnostics = {"raw_markets_seen": 2000}
            state.last_scanned_candidate_count = 16

            pnl = engine._build_pnl_report(state)

            self.assertIn("LOW_SCANNED_CANDIDATE_COUNT", pnl["summary"]["warnings"])

    def test_best_ev_quote_search_can_choose_more_than_fixed_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                quote_search_mode="best_ev",
                quote_improvement_cents=0.1,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
                registry_provider=lambda cfg: {"events": []},
            )
            market = RewardMarketState(
                event_slug="event-1",
                event_title="event-1",
                market_slug="m1",
                question="m1",
                token_id="tok-m1",
            )
            candidate = _candidate(
                market_slug="m1",
                quote_size=10.0,
                best_bid=0.35,
                best_ask=0.40,
                reward_hour=10.0,
                drawdown_hour=0.0,
                spread_capture_hour=10.0,
                tick_size="0.01",
            )

            priced = engine._candidate_with_execution_quote(market, candidate)

            self.assertGreater(priced.quote_bid, 0.351)
            self.assertLess(priced.quote_bid, candidate.best_ask)
            self.assertEqual(priced.quote_mode, "best_ev")

    def test_optimal_action_respects_account_open_buy_order_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            manager.all_open_orders = [
                LiveOpenOrder(
                    order_id="buy-1",
                    side="BUY",
                    price=0.35,
                    size=50.0,
                    size_matched=0.0,
                    size_remaining=50.0,
                    status="open",
                    token_id="tok-other",
                )
            ]
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                live=True,
                action_mode="optimal",
                quote_search_mode="best_ev",
                max_account_open_buy_orders=1,
                max_markets=1,
                max_entry_cost_usdc=100.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )

            state = engine._load_state()
            engine._sync_account_order_summary(state)
            reason = engine._candidate_filter_reason(
                _candidate(market_slug="m1", spread_capture_hour=1.0),
                state=state,
                now=datetime(2026, 4, 24, tzinfo=timezone.utc),
                horizon=1.0,
                max_true_be=0.0,
            )

            self.assertEqual(state.account_open_buy_order_count, 1)
            self.assertEqual(reason, "ACCOUNT_OPEN_BUY_COUNT")

    def test_disable_new_buys_prevents_live_bid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                live=True,
                action_mode="optimal",
                quote_search_mode="best_ev",
                disable_new_buys=True,
                max_entry_cost_usdc=100.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 0.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertEqual(market.action, "SKIP")
            self.assertEqual(market.risk_reject_reason, "NEW_BUYS_DISABLED")
            self.assertEqual(manager.submitted, [])
            self.assertIn("risk_reject=NEW_BUYS_DISABLED", market.decision_trace)

    def test_inventory_manager_only_places_reduce_only_ask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                live=True,
                action_mode="optimal",
                quote_search_mode="best_ev",
                inventory_manager_only=True,
                inventory_policy="balanced",
                max_entry_cost_usdc=100.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", best_bid=0.35, best_ask=0.40, spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 12.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertEqual(market.action, "PLACE_ASK")
            self.assertEqual(market.risk_reject_reason, "INVENTORY_MANAGER_ONLY")
            self.assertIsNone(market.bid_order_id)
            self.assertIsNotNone(market.ask_order_id)
            self.assertEqual([row[0] for row in manager.submitted], ["ask"])

    def test_strict_soft_fail_blocks_new_buy_after_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                live=True,
                action_mode="optimal",
                quote_search_mode="best_ev",
                profit_evidence_mode="strict",
                actual_reward_warmup_minutes=0.0,
                min_verified_net_window_usdc=0.01,
                min_actual_reward_window_usdc=0.05,
                max_entry_cost_usdc=100.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 0.0
            state = engine._load_state()
            state.markets["m1"] = RewardMarketState(
                event_slug=cand.event_slug,
                event_title=cand.event_title,
                market_slug=cand.market_slug,
                question=cand.question,
                token_id=cand.token_id,
                hours_in_reward_zone=1.0,
                net_after_reward_and_cost_usdc=-0.02,
            )

            state = engine.run_cycle(state, scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            market = state.markets["m1"]

            self.assertEqual(market.profit_gate_status, "SOFT_FAIL")
            self.assertEqual(market.action, "SKIP")
            self.assertEqual(market.kelly_size, 0.0)
            self.assertEqual(market.risk_reject_reason, "PROFIT_GATE_SOFT_FAIL")
            self.assertEqual(manager.submitted, [])

    def test_report_includes_decision_and_scale_gate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _LiveLifecycleOrderManager()
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                live=True,
                action_mode="optimal",
                quote_search_mode="best_ev",
                disable_new_buys=True,
                max_entry_cost_usdc=100.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=manager,
                registry_provider=lambda cfg: {"events": []},
            )
            cand = _candidate(market_slug="m1", spread_capture_hour=1.0)
            manager.token_balances[cand.token_id] = 0.0

            state = engine.run_cycle(scanned_candidates=[cand], cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc))
            pnl = engine._build_pnl_report(state)
            row = pnl["markets"][0]

            self.assertIn("decision_trace", row)
            self.assertIn("expected_verified_net", row)
            self.assertIn("risk_reject_reason", row)
            self.assertIn("scale_gate_status", row)
            self.assertIn("decision_trace", row["live_order_lifecycle"])
            self.assertEqual(pnl["summary"]["top_strategy_id"], "reward_mm")


if __name__ == "__main__":
    unittest.main()
