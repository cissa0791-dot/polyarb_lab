from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.intelligence.trade_flow import (
    build_trade_flow_report,
    normalize_bbo_event,
    normalize_trade_event,
    write_trade_flow_report,
)


class MarketIntelligenceTradeFlowTests(unittest.TestCase):
    def test_normalize_trade_and_bbo_events(self) -> None:
        token_to_market = {"tok-1": "market-one"}

        trade = normalize_trade_event(
            {
                "event_type": "last_trade_price",
                "asset_id": "tok-1",
                "price": "0.44",
                "size": "12",
                "side": "buy",
                "timestamp": "2026-03-17T05:00:00Z",
            },
            token_to_market,
        )
        bbo = normalize_bbo_event(
            {
                "event_type": "best_bid_ask",
                "asset_id": "tok-1",
                "best_bid": "0.43",
                "best_ask": "0.45",
                "timestamp": "2026-03-17T05:00:01Z",
            },
            token_to_market,
        )

        assert trade is not None
        assert bbo is not None
        self.assertEqual(trade["market_slug"], "market-one")
        self.assertEqual(trade["side"], "BUY")
        self.assertEqual(bbo["spread"], 0.02)

    def test_build_trade_flow_report_computes_features(self) -> None:
        registry = {
            "summary": {"events_seen": 1, "markets_seen": 1},
            "events": [{"slug": "event-one", "title": "Event One", "markets": [{"slug": "market-one"}]}],
        }
        trade_events = [
            {
                "market_slug": "market-one",
                "side": "BUY",
                "size": 10.0,
                "observed_ts": "2026-03-17T05:00:00+00:00",
            },
            {
                "market_slug": "market-one",
                "side": "SELL",
                "size": 4.0,
                "observed_ts": "2026-03-17T05:00:00.500000+00:00",
            },
        ]
        bbo_events = [
            {
                "market_slug": "market-one",
                "spread": 0.03,
                "observed_ts": "2026-03-17T04:59:59+00:00",
            },
            {
                "market_slug": "market-one",
                "spread": 0.01,
                "observed_ts": "2026-03-17T05:00:01+00:00",
            },
        ]
        live_delta_events = [
            {
                "market_slug": "market-one",
                "observed_ts": "2026-03-17T05:00:02+00:00",
                "previous": {"best_bid_size": 10.0, "best_ask_size": 8.0},
                "current": {"best_bid_size": 12.0, "best_ask_size": 8.0},
            }
        ]

        report = build_trade_flow_report(
            registry=registry,
            trade_events=trade_events,
            bbo_events=bbo_events,
            live_delta_events=live_delta_events,
        )

        self.assertTrue(report["paper_only"])
        self.assertEqual(report["summary"]["trade_events_seen"], 2)
        self.assertEqual(report["top_markets"][0]["trade_count"], 2)
        self.assertEqual(report["top_markets"][0]["spread_compression_after_trades"], 2)
        self.assertEqual(report["top_markets"][0]["liquidity_refill_after_trades"], 2)

    def test_write_trade_flow_report_outputs_files(self) -> None:
        report = {"paper_only": True, "summary": {"trade_events_seen": 0}, "top_markets": [], "top_events": []}
        with tempfile.TemporaryDirectory() as tmp_dir:
            written = write_trade_flow_report(out_dir=Path(tmp_dir), report=report)
            self.assertTrue(written["report_path"].exists())
            self.assertTrue(written["latest_report_path"].exists())


if __name__ == "__main__":
    unittest.main()
