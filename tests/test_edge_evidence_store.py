from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.live.reward_profit_session import (
    RewardOrderManager,
    RewardProfitCandidate,
    RewardProfitConfig,
    RewardProfitSessionEngine,
)


class _CountingOrderManager(RewardOrderManager):
    def __init__(self):
        super().__init__(live=False)

    def build_inventory(self, market, candidate):
        return candidate.quote_size, candidate.best_ask

    def ensure_quote_orders(self, market, candidate):
        return market.bid_order_id or "dry-bid-1", market.ask_order_id or "dry-ask-1"


def _candidate(market_slug: str) -> RewardProfitCandidate:
    return RewardProfitCandidate(
        event_slug="event-1",
        event_title="event-1",
        market_slug=market_slug,
        question=market_slug,
        token_id=f"tok-{market_slug}",
        best_bid=0.35,
        best_ask=0.40,
        midpoint=0.375,
        quote_bid=0.35,
        quote_ask=0.40,
        quote_size=10.0,
        capital_basis_usdc=4.0,
        reward_daily_rate=24.0,
        rewards_max_spread_cents=3.5,
        volume_num=10000.0,
        neg_risk=False,
        tick_size=None,
        expected_reward_per_hour_lower=0.2,
        expected_drawdown_cost_per_hour=0.05,
        reward_minus_drawdown_per_hour=0.15,
        reward_per_dollar_inventory_per_hour=0.05,
        immediate_entry_cost_usdc=0.5,
        immediate_entry_cost_pct=0.125,
        break_even_hours=2.5,
        true_break_even_hours=3.333333,
        expected_spread_capture_per_hour=0.0,
        expected_net_edge_per_hour=0.15,
        kelly_raw_fraction=0.0,
        kelly_position_shares=10.0,
        liquidity_factor=0.8,
        activity_factor=0.8,
    )


class EdgeEvidenceStoreTests(unittest.TestCase):
    def test_run_cycle_appends_market_observation_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_path = Path(tmpdir) / "live_edge_observations.jsonl"
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                edge_evidence_path=str(evidence_path),
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                min_reward_minus_drawdown_per_hour=0.0,
                min_reward_per_dollar_inventory_per_hour=0.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
            )

            engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1")],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )

            rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["row_type"], "cycle_summary")
            self.assertEqual(rows[0]["selected_market_count"], 1)
            self.assertEqual(rows[0]["active_quote_market_count"], 0)
            self.assertEqual(rows[0]["eligible_candidate_count"], 1)
            self.assertEqual(rows[0]["selected_market_slugs"], ["m1"])
            self.assertEqual(rows[1]["row_type"], "market_observation")
            self.assertEqual(rows[1]["market_slug"], "m1")
            self.assertIn("decision_trace", rows[1])
            self.assertIn("fill_rate_window", rows[1])

    def test_run_cycle_records_selected_orderbook_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "live_orderbook_snapshots.jsonl"
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                record_orderbook_snapshots=True,
                orderbook_snapshot_path=str(snapshot_path),
                orderbook_snapshot_max_markets=1,
                max_entry_cost_usdc=10.0,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                min_reward_minus_drawdown_per_hour=0.0,
                min_reward_per_dollar_inventory_per_hour=0.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
            )

            engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="m1"), _candidate(market_slug="m2")],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )

            rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["row_type"], "orderbook_snapshot")
            self.assertEqual(rows[0]["source"], "selected")
            self.assertIn(rows[0]["market_slug"], {"m1", "m2"})
            self.assertEqual(rows[0]["best_bid"], 0.35)
            self.assertIn("account_inventory_usdc", rows[0])

    def test_research_snapshot_records_filtered_near_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "live_orderbook_snapshots.jsonl"
            config = RewardProfitConfig(
                out_dir=tmpdir,
                state_path=str(Path(tmpdir) / "state.json"),
                pnl_path=str(Path(tmpdir) / "pnl.json"),
                record_orderbook_snapshots=True,
                orderbook_snapshot_path=str(snapshot_path),
                orderbook_snapshot_max_markets=5,
                orderbook_snapshot_include_filtered=True,
                orderbook_snapshot_filtered_max=5,
                max_entry_cost_usdc=0.1,
                max_entry_cost_pct=1.0,
                max_break_even_hours=100.0,
                min_reward_minus_drawdown_per_hour=0.0,
                min_reward_per_dollar_inventory_per_hour=0.0,
            )
            engine = RewardProfitSessionEngine(
                config,
                reward_client_factory=lambda dry_run: None,
                order_manager=_CountingOrderManager(),
            )

            engine.run_cycle(
                scanned_candidates=[_candidate(market_slug="filtered")],
                cycle_ts=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )

            rows = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["source"], "near_miss_filtered")
            self.assertEqual(rows[0]["filter_reason"], "ENTRY_COST_USDC")
            self.assertIn("total_score", rows[0])
            self.assertIn("expected_net_edge_per_hour", rows[0])


if __name__ == "__main__":
    unittest.main()
