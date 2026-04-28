from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import ExecutionReport, OrderStatus
from src.live.reward_live_mm import (
    RewardExecutionState,
    RewardLiveConfig,
    RewardLiveExecutor,
    RewardLiveStateReport,
    RewardMarketCandidate,
    RewardMarketRuntimeState,
)


class _FakeSelector:
    def __init__(self, candidates: list[RewardMarketCandidate]) -> None:
        self._candidates = candidates

    def select(self, **kwargs):
        return list(self._candidates)


class _FakeBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, float]] = []

    def submit_limit_order(self, intent):
        self.calls.append((intent.market_slug, intent.side, float(intent.limit_price or 0.0), float(intent.size)))
        return ExecutionReport(
            intent_id=intent.intent_id,
            status=OrderStatus.SUBMITTED,
            filled_size=0.0,
            avg_fill_price=None,
            metadata={"live_order_id": f"fake-{len(self.calls)}"},
            ts=datetime.now(timezone.utc),
        )

    def cancel_order(self, order_id: str) -> bool:
        return True


class _FakeLiveClient:
    def __init__(self, statuses: dict[str, tuple[str, float, float, float | None]]) -> None:
        self.statuses = statuses

    def get_order_status(self, order_id: str):
        status, matched, remaining, avg_price = self.statuses[order_id]

        class _Status:
            def __init__(self, order_id, status, matched, remaining, avg_price):
                self.order_id = order_id
                self.status = status
                self.size_matched = matched
                self.size_remaining = remaining
                self.avg_price = avg_price

        return _Status(order_id, status, matched, remaining, avg_price)


class _FakeRewardClient:
    def get_rewards_summary(self):
        return {"user_earned_usd": 1.25, "dry_run": True}


def _candidate(
    *,
    market_slug: str = "market-1",
    event_slug: str = "event-1",
    midpoint: float = 0.45,
    quote_size: float = 20.0,
    quote_bid: float = 0.44,
    quote_ask: float = 0.46,
    reward_daily: float = 1.2,
    reward_hour: float = 0.05,
    drawdown_hour: float = 0.01,
) -> RewardMarketCandidate:
    return RewardMarketCandidate(
        event_slug=event_slug,
        market_slug=market_slug,
        question="Q?",
        yes_token_id=f"yes-{market_slug}",
        no_token_id=f"no-{market_slug}",
        best_bid=quote_bid,
        best_ask=quote_ask,
        midpoint=midpoint,
        neg_risk=False,
        reward_daily_rate=10.0,
        rewards_min_size=quote_size,
        rewards_max_spread=3.5,
        quote_bid=quote_bid,
        quote_ask=quote_ask,
        quote_size=quote_size,
        capital_in_use_usdc=quote_size * midpoint,
        implied_reward_ev_daily_usdc=reward_daily,
        expected_reward_per_hour=reward_hour,
        expected_drawdown_cost_per_hour=drawdown_hour,
        reward_minus_drawdown_per_hour=reward_hour - drawdown_hour,
        reward_per_dollar_inventory_per_day=0.02,
        modeled_spread_capture_ev_daily_usdc=0.4,
        modeled_cost_proxy_daily_usdc=0.2,
        selection_score=1.0,
        current_spread_cents=1.0,
    )


class RewardLiveDryRunTests(unittest.TestCase):
    def test_dry_run_builds_inventory_and_quotes(self) -> None:
        candidate = _candidate()
        executor = RewardLiveExecutor(
            broker=_FakeBroker(),
            client=None,
            reward_client=_FakeRewardClient(),
            selector=_FakeSelector([candidate]),
            config=RewardLiveConfig(dry_run=True, max_markets=1, max_markets_per_event=1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            pnl_path = Path(tmp) / "pnl.json"
            state_report, pnl_report = executor.run_cycle(
                event_slugs=[candidate.event_slug],
                state_path=state_path,
                pnl_path=pnl_path,
            )
            self.assertEqual(state_report.selected_market_count, 1)
            self.assertEqual(len(state_report.markets), 1)
            market = state_report.markets[0]
            self.assertEqual(market.status, RewardExecutionState.QUOTING)
            self.assertAlmostEqual(market.inventory_shares, candidate.quote_size)
            self.assertIsNotNone(market.bid_order)
            self.assertIsNotNone(market.ask_order)
            self.assertTrue(state_path.exists())
            self.assertTrue(pnl_path.exists())
            self.assertEqual(pnl_report["summary"]["active_quote_market_count"], 1)

    def test_profit_gates_reject_weak_candidate_before_orders(self) -> None:
        candidate = _candidate(
            reward_daily=0.1,
            reward_hour=0.015,
            drawdown_hour=0.01,
        )
        broker = _FakeBroker()
        executor = RewardLiveExecutor(
            broker=broker,
            client=None,
            reward_client=None,
            selector=_FakeSelector([candidate]),
            config=RewardLiveConfig(
                dry_run=True,
                min_reward_minus_drawdown_per_hour=0.02,
                min_implied_reward_ev_daily_usdc=0.25,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_report, pnl_report = executor.run_cycle(
                event_slugs=[candidate.event_slug],
                state_path=Path(tmp) / "state.json",
                pnl_path=Path(tmp) / "pnl.json",
            )

            self.assertEqual(state_report.selected_market_count, 0)
            self.assertEqual(state_report.active_quote_market_count, 0)
            self.assertEqual(pnl_report["summary"]["capital_in_use_usdc"], 0.0)
            self.assertEqual(broker.calls, [])

    def test_market_drawdown_triggers_exit(self) -> None:
        candidate = _candidate(midpoint=0.40, quote_bid=0.39, quote_ask=0.41)
        market = RewardMarketRuntimeState(
            event_slug=candidate.event_slug,
            market_slug=candidate.market_slug,
            question=candidate.question,
            yes_token_id=candidate.yes_token_id,
            no_token_id=candidate.no_token_id,
            status=RewardExecutionState.QUOTING,
            inventory_shares=20.0,
            avg_inventory_cost=0.55,
            latest_mid_price=0.40,
            last_candidate=candidate,
        )
        report = RewardLiveStateReport(
            generated_ts=datetime.now(timezone.utc),
            mode="dry_run",
            markets=[market],
        )
        executor = RewardLiveExecutor(
            broker=_FakeBroker(),
            client=None,
            reward_client=None,
            selector=_FakeSelector([candidate]),
            config=RewardLiveConfig(dry_run=True, max_drawdown_per_market=1.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            pnl_path = Path(tmp) / "pnl.json"
            state_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            state_report, _ = executor.run_cycle(
                event_slugs=[candidate.event_slug],
                state_path=state_path,
                pnl_path=pnl_path,
            )
            updated = state_report.markets[0]
            self.assertEqual(updated.status, RewardExecutionState.CLOSED)
            self.assertEqual(updated.last_halt_reason, "market_drawdown_limit")
            self.assertAlmostEqual(updated.inventory_shares, 0.0)


class RewardLiveRecoveryTests(unittest.TestCase):
    def test_live_recovery_applies_polled_inventory_fill(self) -> None:
        candidate = _candidate()
        market = RewardMarketRuntimeState(
            event_slug=candidate.event_slug,
            market_slug=candidate.market_slug,
            question=candidate.question,
            yes_token_id=candidate.yes_token_id,
            no_token_id=candidate.no_token_id,
            status=RewardExecutionState.SELECTED,
            inventory_shares=0.0,
            avg_inventory_cost=0.0,
            inventory_order={
                "kind": "inventory",
                "side": "BUY",
                "order_id": "live-order-1",
                "requested_size": 20.0,
                "price": 0.46,
                "matched_size": 0.0,
                "status": "live",
                "synthetic": False,
                "created_ts": datetime.now(timezone.utc).isoformat(),
                "updated_ts": datetime.now(timezone.utc).isoformat(),
            },
            last_candidate=candidate,
        )
        report = RewardLiveStateReport(
            generated_ts=datetime.now(timezone.utc),
            mode="live",
            markets=[market],
        )
        executor = RewardLiveExecutor(
            broker=_FakeBroker(),
            client=_FakeLiveClient({"live-order-1": ("matched", 20.0, 0.0, 0.46)}),  # type: ignore[arg-type]
            reward_client=None,
            selector=_FakeSelector([candidate]),
            config=RewardLiveConfig(dry_run=False, max_markets=1, max_markets_per_event=1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            pnl_path = Path(tmp) / "pnl.json"
            state_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            state_report, _ = executor.run_cycle(
                event_slugs=[candidate.event_slug],
                state_path=state_path,
                pnl_path=pnl_path,
            )
            updated = state_report.markets[0]
            self.assertGreaterEqual(updated.inventory_shares, 20.0)
            self.assertIn(updated.status, {RewardExecutionState.INVENTORY_BUILT, RewardExecutionState.QUOTING})


if __name__ == "__main__":
    unittest.main()
