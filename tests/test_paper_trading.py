from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.domain.models import OrderIntent, OrderMode, OrderStatus, OrderType
from src.paper.broker import PaperBroker
from src.paper.ledger import Ledger


class Level:
    def __init__(self, price: float, size: float):
        self.price = price
        self.size = size


class Book:
    def __init__(self, asks=None, bids=None):
        self.asks = asks or []
        self.bids = bids or []


class PaperTradingTests(unittest.TestCase):
    def test_limit_buy_fills_and_updates_cash(self) -> None:
        ledger = Ledger(cash=100.0)
        broker = PaperBroker(ledger)
        intent = OrderIntent(
            intent_id="ord-1",
            candidate_id="cand-1",
            mode=OrderMode.PAPER,
            market_slug="market-a",
            token_id="token-yes",
            side="BUY",
            order_type=OrderType.LIMIT,
            size=10.0,
            limit_price=0.40,
            max_notional_usd=4.0,
            ts=datetime.now(timezone.utc),
        )
        report = broker.submit_limit_order(intent, Book(asks=[Level(0.39, 10.0)]))

        self.assertEqual(report.status, OrderStatus.FILLED)
        self.assertAlmostEqual(ledger.cash, 96.1, places=6)
        self.assertAlmostEqual(ledger.frozen_cash, 0.0, places=6)
        self.assertAlmostEqual(ledger.positions["token-yes"].shares, 10.0, places=6)

    def test_partial_fill_cancels_remainder_and_releases_cash(self) -> None:
        ledger = Ledger(cash=100.0)
        broker = PaperBroker(ledger)
        intent = OrderIntent(
            intent_id="ord-2",
            candidate_id="cand-2",
            mode=OrderMode.PAPER,
            market_slug="market-a",
            token_id="token-no",
            side="BUY",
            order_type=OrderType.LIMIT,
            size=10.0,
            limit_price=0.50,
            max_notional_usd=5.0,
            ts=datetime.now(timezone.utc),
        )
        report = broker.submit_limit_order(intent, Book(asks=[Level(0.45, 4.0)]))

        self.assertEqual(report.status, OrderStatus.PARTIAL)
        self.assertAlmostEqual(report.filled_size, 4.0, places=6)
        self.assertAlmostEqual(ledger.frozen_cash, 0.0, places=6)
        self.assertAlmostEqual(ledger.cash, 98.2, places=6)

    def test_sell_realizes_pnl(self) -> None:
        ledger = Ledger(cash=100.0)
        self.assertTrue(ledger.place_limit_order("buy-1", "token-1", "market-a", "BUY", 5.0, 0.40))
        self.assertTrue(ledger.apply_fill("buy-1", 5.0, 0.40))

        self.assertTrue(ledger.place_limit_order("sell-1", "token-1", "market-a", "SELL", 3.0, 0.60))
        self.assertTrue(ledger.apply_fill("sell-1", 3.0, 0.60))

        self.assertAlmostEqual(ledger.realized_pnl, 0.6, places=6)
        self.assertAlmostEqual(ledger.cash, 99.8, places=6)


if __name__ == "__main__":
    unittest.main()
