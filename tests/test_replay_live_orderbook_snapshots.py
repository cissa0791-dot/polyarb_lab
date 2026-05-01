from __future__ import annotations

import unittest

from scripts.replay_live_orderbook_snapshots import build_replay_report
from src.backtest.replay import ReplayConfig


class ReplayLiveOrderbookSnapshotsTests(unittest.TestCase):
    def test_replays_snapshots_and_reports_adverse_selection(self) -> None:
        rows = [
            {
                "row_type": "orderbook_snapshot",
                "ts": "2026-05-01T00:00:00+00:00",
                "market_slug": "m1",
                "token_id": "tok-m1",
                "best_bid": 0.49,
                "best_ask": 0.51,
                "bids": [[0.49, 100]],
                "asks": [[0.51, 100]],
            },
            {
                "row_type": "orderbook_snapshot",
                "ts": "2026-05-01T00:00:30+00:00",
                "market_slug": "m1",
                "token_id": "tok-m1",
                "best_bid": 0.49,
                "best_ask": 0.49,
                "bids": [[0.49, 100]],
                "asks": [[0.49, 100]],
            },
            {
                "row_type": "orderbook_snapshot",
                "ts": "2026-05-01T00:01:00+00:00",
                "market_slug": "m1",
                "token_id": "tok-m1",
                "best_bid": 0.52,
                "best_ask": 0.54,
                "bids": [[0.52, 100]],
                "asks": [[0.54, 100]],
            },
        ]

        report = build_replay_report(rows, ReplayConfig(order_size=5.0, latency_ms=0))

        self.assertEqual(report["market_count"], 1)
        self.assertEqual(report["replayed_market_count"], 1)
        self.assertIn("adverse_selection_after_fill_usdc", report["markets"][0])
        self.assertIn(report["markets"][0]["suitability"], {"REWARD_MM_CANDIDATE", "REWARD_MM_WATCH", "INSUFFICIENT_DATA"})


if __name__ == "__main__":
    unittest.main()
