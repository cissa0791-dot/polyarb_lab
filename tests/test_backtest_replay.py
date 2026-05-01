from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.backtest.replay import ReplayConfig, load_saved_books, replay_from_saved_books, replay_quote_strategy


class BacktestReplayTests(unittest.TestCase):
    def test_loads_jsonl_book_snapshots(self) -> None:
        rows = [
            {"ts": "2026-01-01T00:00:01+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.50, "best_ask": 0.52},
            {"ts": "2026-01-01T00:00:00+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.49, "best_ask": 0.51},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "books.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            snapshots = load_saved_books(str(path))

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0].best_bid, 0.49)
        self.assertEqual(snapshots[1].best_ask, 0.52)

    def test_replay_simulates_passive_buy_and_sell_fills(self) -> None:
        rows = [
            {"ts": "2026-01-01T00:00:00+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.49, "best_ask": 0.51},
            {"ts": "2026-01-01T00:00:01+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.48, "best_ask": 0.49},
            {"ts": "2026-01-01T00:00:02+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.53, "best_ask": 0.55},
        ]
        snapshots = load_saved_books_from_rows(rows)

        result = replay_quote_strategy(
            snapshots,
            ReplayConfig(order_size=5.0, quote_bid_offset=0.0, quote_ask_offset=-0.04, latency_ms=0),
        )

        self.assertEqual(result.fill_count, 2)
        self.assertAlmostEqual(result.buy_fill_shares, 5.0)
        self.assertAlmostEqual(result.sell_fill_shares, 5.0)
        self.assertGreater(result.realized_pnl_usdc, 0.0)
        self.assertAlmostEqual(result.ending_inventory_shares, 0.0)

    def test_replay_from_saved_books_returns_serializable_summary(self) -> None:
        rows = [
            {"ts": "2026-01-01T00:00:00+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.49, "best_ask": 0.51},
            {"ts": "2026-01-01T00:00:01+00:00", "market_slug": "m", "token_id": "t", "best_bid": 0.48, "best_ask": 0.49},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "books.json"
            path.write_text(json.dumps({"snapshots": rows}), encoding="utf-8")

            payload = replay_from_saved_books(str(path), ReplayConfig(order_size=5.0))

        self.assertEqual(payload["snapshot_count"], 2)
        self.assertIn("net_pnl_usdc", payload)
        self.assertIsInstance(payload["fills"], list)


def load_saved_books_from_rows(rows):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "books.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return load_saved_books(str(path))


if __name__ == "__main__":
    unittest.main()
