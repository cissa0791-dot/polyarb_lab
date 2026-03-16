from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.config_runtime.models import PaperConfig
from src.domain.models import PositionState
from src.paper.exit_policy import evaluate_exit
from src.paper.ledger import Ledger


class PositionLifecycleTests(unittest.TestCase):
    def test_mark_to_market_tracks_unrealized_pnl(self) -> None:
        ledger = Ledger(cash=100.0)
        self.assertTrue(ledger.place_limit_order("buy-1", "token-a", "market-a", "BUY", 10.0, 0.40, candidate_id="cand-1", position_id="pos-1"))
        self.assertTrue(ledger.apply_fill("buy-1", 10.0, 0.40, fee_usd=0.10))

        mark = ledger.mark_position("pos-1", mark_price=0.50, ts=datetime.now(timezone.utc), source_bid=0.50, source_ask=0.51)

        self.assertIsNotNone(mark)
        assert mark is not None
        self.assertAlmostEqual(mark.remaining_entry_cost_usd, 4.1, places=6)
        self.assertAlmostEqual(mark.marked_value_usd, 5.0, places=6)
        self.assertAlmostEqual(mark.unrealized_pnl_usd, 0.9, places=6)
        snapshot = ledger.snapshot()
        self.assertAlmostEqual(snapshot.unrealized_pnl, 0.9, places=6)

    def test_take_profit_stop_loss_and_age_exit_signals(self) -> None:
        base_time = datetime.now(timezone.utc)
        ledger = Ledger(cash=100.0)
        self.assertTrue(ledger.place_limit_order("buy-2", "token-b", "market-b", "BUY", 5.0, 0.40, ts=base_time.isoformat(), candidate_id="cand-2", position_id="pos-2"))
        self.assertTrue(ledger.apply_fill("buy-2", 5.0, 0.40, ts=base_time.isoformat()))

        tp_mark = ledger.mark_position("pos-2", mark_price=0.70, ts=base_time + timedelta(seconds=5), source_bid=0.70, source_ask=0.71)
        assert tp_mark is not None
        tp_signal = evaluate_exit(tp_mark, PaperConfig(take_profit_usd=1.0, stop_loss_usd=10.0, max_holding_sec=1000.0))
        self.assertIsNotNone(tp_signal)
        assert tp_signal is not None
        self.assertEqual(tp_signal.reason_code, "TAKE_PROFIT")

        sl_mark = ledger.mark_position("pos-2", mark_price=0.10, ts=base_time + timedelta(seconds=6), source_bid=0.10, source_ask=0.11)
        assert sl_mark is not None
        sl_signal = evaluate_exit(sl_mark, PaperConfig(take_profit_usd=10.0, stop_loss_usd=1.0, max_holding_sec=1000.0))
        self.assertIsNotNone(sl_signal)
        assert sl_signal is not None
        self.assertEqual(sl_signal.reason_code, "STOP_LOSS")

        age_mark = ledger.mark_position("pos-2", mark_price=0.40, ts=base_time + timedelta(seconds=20), source_bid=0.40, source_ask=0.41)
        assert age_mark is not None
        age_signal = evaluate_exit(age_mark, PaperConfig(take_profit_usd=10.0, stop_loss_usd=10.0, max_holding_sec=10.0))
        self.assertIsNotNone(age_signal)
        assert age_signal is not None
        self.assertEqual(age_signal.reason_code, "MAX_HOLDING_AGE")

        force_signal = evaluate_exit(age_mark, PaperConfig(), force_reason="MANUAL_FORCE_FLATTEN")
        self.assertIsNotNone(force_signal)
        assert force_signal is not None
        self.assertTrue(force_signal.force_exit)
        self.assertEqual(force_signal.reason_code, "MANUAL_FORCE_FLATTEN")

    def test_position_close_lifecycle_and_trade_summary(self) -> None:
        ledger = Ledger(cash=100.0)
        self.assertTrue(ledger.place_limit_order("buy-3", "token-c", "market-c", "BUY", 5.0, 0.40, candidate_id="cand-3", position_id="pos-3"))
        self.assertTrue(ledger.apply_fill("buy-3", 5.0, 0.40, fee_usd=0.10))

        self.assertTrue(ledger.place_limit_order("sell-3a", "token-c", "market-c", "SELL", 2.0, 0.60, position_id="pos-3"))
        self.assertTrue(ledger.apply_fill("sell-3a", 2.0, 0.60, fee_usd=0.02))
        self.assertEqual(ledger.position_records["pos-3"].state, PositionState.PARTIALLY_REDUCED)
        self.assertAlmostEqual(ledger.position_records["pos-3"].remaining_shares, 3.0, places=6)

        self.assertTrue(ledger.place_limit_order("sell-3b", "token-c", "market-c", "SELL", 3.0, 0.65, position_id="pos-3"))
        self.assertTrue(ledger.apply_fill("sell-3b", 3.0, 0.65, fee_usd=0.03))
        ledger.set_position_state("pos-3", PositionState.CLOSED, reason_code="TAKE_PROFIT")
        trade_summary = ledger.build_trade_summary("pos-3")

        self.assertIsNotNone(trade_summary)
        assert trade_summary is not None
        self.assertEqual(trade_summary.state, PositionState.CLOSED)
        self.assertAlmostEqual(trade_summary.entry_cost_usd, 2.1, places=6)
        self.assertAlmostEqual(trade_summary.exit_proceeds_usd, 3.15, places=6)
        self.assertAlmostEqual(trade_summary.fees_paid_usd, 0.15, places=6)
        self.assertAlmostEqual(trade_summary.realized_pnl_usd, 1.0, places=6)
        self.assertAlmostEqual(ledger.cash, 101.0, places=6)


if __name__ == "__main__":
    unittest.main()
