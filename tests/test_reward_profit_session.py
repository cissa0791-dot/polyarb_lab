from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.live.reward_profit_session import (
    RewardMarketStatus,
    RewardOrderManager,
    RewardProfitCandidate,
    RewardProfitConfig,
    RewardProfitSelector,
    RewardProfitSessionEngine,
)
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

    def build_inventory(self, market, candidate):
        self.inventory_calls += 1
        return candidate.quote_size, candidate.best_ask

    def ensure_quote_orders(self, market, candidate):
        self.quote_calls += 1
        return market.bid_order_id or "dry-bid-1", market.ask_order_id or "dry-ask-1"


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


if __name__ == "__main__":
    unittest.main()
