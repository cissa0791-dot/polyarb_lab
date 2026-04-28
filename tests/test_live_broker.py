"""Tests for src/live/broker.py — LiveBroker contract.

All tests are fully offline.  A _FakeLiveWriteClient replaces the real
LiveWriteClient so no credentials, network, or py_clob_client plumbing
is exercised.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import uuid4

from src.domain.models import ExecutionReport, OrderIntent, OrderMode, OrderStatus, OrderType
from src.live.broker import LiveBroker
from src.live.client import LiveClientError, LiveOrderResult, LiveWriteClient


# ---------------------------------------------------------------------------
# Fake LiveWriteClient (offline)
# ---------------------------------------------------------------------------

class _FakeLiveWriteClient:
    """Records calls; returns configurable LiveOrderResult or raises."""

    def __init__(
        self,
        size_matched: float = 0.0,
        order_id: str = "live-order-1",
        status: str = "live",
        raises: Exception | None = None,
    ):
        self.calls: list[tuple] = []
        self._size_matched = size_matched
        self._order_id = order_id
        self._status = status
        self._raises = raises

    def submit_order(self, *, token_id, side, price, size, neg_risk=False, tick_size=None):
        self.calls.append(("submit_order", token_id, side, price, size, neg_risk, tick_size))
        if self._raises:
            raise self._raises
        return LiveOrderResult(
            order_id=self._order_id,
            status=self._status,
            size_matched=self._size_matched,
        )


def _fake_write_client(**kwargs) -> _FakeLiveWriteClient:
    return _FakeLiveWriteClient(**kwargs)


def _broker(**kwargs) -> tuple[LiveBroker, _FakeLiveWriteClient]:
    fake = _fake_write_client(**kwargs)
    return LiveBroker(fake), fake  # type: ignore[arg-type]


def _intent(
    side: str = "BUY",
    size: float = 50.0,
    limit_price: float = 0.55,
    token_id: str = "tok-abc",
    metadata: dict | None = None,
) -> OrderIntent:
    return OrderIntent(
        intent_id=str(uuid4()),
        candidate_id="cand-1",
        mode=OrderMode.LIVE,
        market_slug="test-market",
        token_id=token_id,
        position_id=str(uuid4()),
        side=side,
        order_type=OrderType.LIMIT,
        size=size,
        limit_price=limit_price,
        max_notional_usd=size * limit_price,
        ts=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# submit_limit_order — return status
# ---------------------------------------------------------------------------

class TestLiveBrokerStatus(unittest.TestCase):

    def test_submitted_when_no_fill_at_submission(self) -> None:
        broker, _ = _broker(size_matched=0.0)
        report = broker.submit_limit_order(_intent(size=50.0))
        self.assertEqual(report.status, OrderStatus.SUBMITTED)

    def test_partial_when_partially_matched(self) -> None:
        broker, _ = _broker(size_matched=20.0)
        report = broker.submit_limit_order(_intent(size=50.0))
        self.assertEqual(report.status, OrderStatus.PARTIAL)

    def test_filled_when_fully_matched(self) -> None:
        broker, _ = _broker(size_matched=50.0)
        report = broker.submit_limit_order(_intent(size=50.0))
        self.assertEqual(report.status, OrderStatus.FILLED)

    def test_rejected_on_live_client_error(self) -> None:
        broker, _ = _broker(raises=LiveClientError("CLOB unavailable"))
        report = broker.submit_limit_order(_intent())
        self.assertEqual(report.status, OrderStatus.REJECTED)


# ---------------------------------------------------------------------------
# submit_limit_order — ExecutionReport fields
# ---------------------------------------------------------------------------

class TestLiveBrokerReportFields(unittest.TestCase):

    def test_returns_execution_report(self) -> None:
        broker, _ = _broker()
        report = broker.submit_limit_order(_intent())
        self.assertIsInstance(report, ExecutionReport)

    def test_intent_id_preserved(self) -> None:
        broker, _ = _broker()
        intent = _intent()
        report = broker.submit_limit_order(intent)
        self.assertEqual(report.intent_id, intent.intent_id)

    def test_position_id_preserved(self) -> None:
        broker, _ = _broker()
        intent = _intent()
        report = broker.submit_limit_order(intent)
        self.assertEqual(report.position_id, intent.position_id)

    def test_filled_size_from_result(self) -> None:
        broker, _ = _broker(size_matched=25.0)
        report = broker.submit_limit_order(_intent(size=50.0))
        self.assertAlmostEqual(report.filled_size, 25.0)

    def test_filled_size_zero_when_resting(self) -> None:
        broker, _ = _broker(size_matched=0.0)
        report = broker.submit_limit_order(_intent())
        self.assertEqual(report.filled_size, 0.0)

    def test_avg_fill_price_set_when_filled(self) -> None:
        broker, _ = _broker(size_matched=50.0)
        intent = _intent(limit_price=0.60)
        report = broker.submit_limit_order(intent)
        self.assertAlmostEqual(report.avg_fill_price, 0.60)

    def test_avg_fill_price_none_when_no_fill(self) -> None:
        broker, _ = _broker(size_matched=0.0)
        report = broker.submit_limit_order(_intent())
        self.assertIsNone(report.avg_fill_price)

    def test_live_order_id_in_metadata(self) -> None:
        broker, _ = _broker(order_id="clob-ord-99")
        report = broker.submit_limit_order(_intent())
        self.assertEqual(report.metadata["live_order_id"], "clob-ord-99")

    def test_live_status_in_metadata(self) -> None:
        broker, _ = _broker(status="matched")
        report = broker.submit_limit_order(_intent())
        self.assertEqual(report.metadata["live_status"], "matched")

    def test_rejected_metadata_contains_error(self) -> None:
        broker, _ = _broker(raises=LiveClientError("bad tick size"))
        report = broker.submit_limit_order(_intent())
        self.assertIn("bad tick size", report.metadata["error"])

    def test_ts_is_set(self) -> None:
        broker, _ = _broker()
        report = broker.submit_limit_order(_intent())
        self.assertIsInstance(report.ts, datetime)


# ---------------------------------------------------------------------------
# submit_limit_order — CLOB call arguments
# ---------------------------------------------------------------------------

class TestLiveBrokerCallArgs(unittest.TestCase):

    def test_calls_submit_order_once(self) -> None:
        broker, fake = _broker()
        broker.submit_limit_order(_intent())
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0][0], "submit_order")

    def test_passes_correct_token_id(self) -> None:
        broker, fake = _broker()
        broker.submit_limit_order(_intent(token_id="tok-xyz"))
        self.assertEqual(fake.calls[0][1], "tok-xyz")

    def test_passes_correct_side(self) -> None:
        broker, fake = _broker()
        broker.submit_limit_order(_intent(side="SELL"))
        self.assertEqual(fake.calls[0][2], "SELL")

    def test_passes_correct_price(self) -> None:
        broker, fake = _broker()
        broker.submit_limit_order(_intent(limit_price=0.72))
        self.assertAlmostEqual(fake.calls[0][3], 0.72)

    def test_passes_correct_size(self) -> None:
        broker, fake = _broker()
        broker.submit_limit_order(_intent(size=30.0))
        self.assertAlmostEqual(fake.calls[0][4], 30.0)

    def test_passes_market_signature_metadata(self) -> None:
        broker, fake = _broker()
        broker.submit_limit_order(_intent(metadata={"neg_risk": True, "tick_size": "0.01"}))
        self.assertEqual(fake.calls[0][5], True)
        self.assertEqual(fake.calls[0][6], "0.01")

    def test_no_call_on_rejected(self) -> None:
        # Even on error the call was made — just once
        broker, fake = _broker(raises=LiveClientError("x"))
        broker.submit_limit_order(_intent())
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
