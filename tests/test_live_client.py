"""Tests for src/live/client.py — write client construction and error contract.

All tests are fully offline.  A _FakeClient replaces ClobClient so no
credentials, network, or py_clob_client auth plumbing is exercised here.
build_authenticated_client is not called by any test; from_credentials is
covered in a separate integration/dry-run test (L10).
"""

from __future__ import annotations

import unittest

from src.live.client import (
    LiveClientError,
    LiveOrderResult,
    LiveOrderStatus,
    LiveWriteClient,
)


# ---------------------------------------------------------------------------
# Minimal fake ClobClient — no auth, no network
# ---------------------------------------------------------------------------

class _FakeOrder:
    """Minimal stand-in for the object returned by ClobClient.get_order."""
    def __init__(self, id="order-99", status="live", size_matched="25.0", size="50.0"):
        self.id = id
        self.status = status
        self.size_matched = size_matched
        self.size = size


class _FakeClient:
    """Records calls and returns configurable responses or raises on demand."""

    def __init__(
        self,
        submit_response: dict | None = None,
        order_response: object | None = None,
        raises: Exception | None = None,
        balance_response: object | None = None,
    ):
        self.calls: list[tuple] = []
        self._submit_response = submit_response or {
            "orderID": "fake-order-123",
            "status": "live",
            "size_matched": "0.0",
        }
        self._order_response = order_response or _FakeOrder()
        self._raises = raises
        self._balance_response = balance_response

    def get_balance_allowance(self, params):
        self.calls.append(("get_balance_allowance", params))
        if self._raises:
            raise self._raises
        return self._balance_response or {"balance": "0"}

    def create_and_post_order(self, order_args, options):
        self.calls.append(("create_and_post_order", order_args, options))
        if self._raises:
            raise self._raises
        return self._submit_response

    def get_tick_size(self, token_id):
        self.calls.append(("get_tick_size", token_id))
        return "0.01"

    def get_neg_risk(self, token_id):
        self.calls.append(("get_neg_risk", token_id))
        return False

    def cancel(self, order_id):
        self.calls.append(("cancel", order_id))
        if self._raises:
            raise self._raises
        return {}

    def get_order(self, order_id):
        self.calls.append(("get_order", order_id))
        if self._raises:
            raise self._raises
        return self._order_response


def _dry_client(**kwargs) -> LiveWriteClient:
    return LiveWriteClient(_FakeClient(**kwargs), dry_run=True)


def _live_client(**kwargs) -> LiveWriteClient:
    return LiveWriteClient(_FakeClient(**kwargs), dry_run=False)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLiveWriteClientConstruction(unittest.TestCase):

    def test_default_is_dry_run(self) -> None:
        client = LiveWriteClient(_FakeClient())
        self.assertTrue(client.dry_run)

    def test_dry_run_false_when_explicitly_set(self) -> None:
        client = LiveWriteClient(_FakeClient(), dry_run=False)
        self.assertFalse(client.dry_run)

    def test_client_stored(self) -> None:
        fake = _FakeClient()
        client = LiveWriteClient(fake)
        self.assertIs(client._client, fake)


# ---------------------------------------------------------------------------
# submit_order — dry_run
# ---------------------------------------------------------------------------

class TestSubmitOrderDryRun(unittest.TestCase):

    def setUp(self) -> None:
        self.fake = _FakeClient()
        self.client = LiveWriteClient(self.fake, dry_run=True)

    def test_returns_dry_run_sentinel(self) -> None:
        result = self.client.submit_order("tok", "BUY", 0.55, 50.0)
        self.assertIsInstance(result, LiveOrderResult)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.status, "dry_run")
        self.assertIsNone(result.order_id)

    def test_makes_no_network_call(self) -> None:
        self.client.submit_order("tok", "BUY", 0.55, 50.0)
        self.assertEqual(self.fake.calls, [])

    def test_size_matched_is_zero(self) -> None:
        result = self.client.submit_order("tok", "SELL", 0.45, 10.0)
        self.assertEqual(result.size_matched, 0.0)

    def test_result_is_frozen(self) -> None:
        result = self.client.submit_order("tok", "BUY", 0.55, 50.0)
        with self.assertRaises((AttributeError, TypeError)):
            result.status = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# submit_order — live mode
# ---------------------------------------------------------------------------

class TestSubmitOrderLive(unittest.TestCase):

    def test_calls_create_and_post_order(self) -> None:
        fake = _FakeClient()
        client = LiveWriteClient(fake, dry_run=False)
        client.submit_order("tok-abc", "BUY", 0.60, 30.0)
        self.assertEqual(fake.calls[-1][0], "create_and_post_order")

    def test_resolves_tick_size_when_missing(self) -> None:
        fake = _FakeClient()
        client = LiveWriteClient(fake, dry_run=False)
        client.submit_order("tok-abc", "BUY", 0.60, 30.0)
        self.assertIn(("get_tick_size", "tok-abc"), fake.calls)
        order_call = fake.calls[-1]
        self.assertEqual(order_call[2].tick_size, "0.01")

    def test_returns_normalized_result(self) -> None:
        fake = _FakeClient(submit_response={
            "orderID": "ord-xyz",
            "status": "matched",
            "size_matched": "30.0",
        })
        client = LiveWriteClient(fake, dry_run=False)
        result = client.submit_order("tok", "BUY", 0.60, 30.0)
        self.assertEqual(result.order_id, "ord-xyz")
        self.assertEqual(result.status, "matched")
        self.assertAlmostEqual(result.size_matched, 30.0)
        self.assertFalse(result.dry_run)

    def test_order_id_fallback_to_order_id_key(self) -> None:
        fake = _FakeClient(submit_response={"order_id": "alt-id", "status": "live"})
        client = LiveWriteClient(fake, dry_run=False)
        result = client.submit_order("tok", "BUY", 0.50, 10.0)
        self.assertEqual(result.order_id, "alt-id")

    def test_missing_size_matched_defaults_to_zero(self) -> None:
        fake = _FakeClient(submit_response={"orderID": "o1", "status": "live"})
        client = LiveWriteClient(fake, dry_run=False)
        result = client.submit_order("tok", "BUY", 0.50, 10.0)
        self.assertEqual(result.size_matched, 0.0)

    def test_normalizes_exception_to_live_client_error(self) -> None:
        fake = _FakeClient(raises=RuntimeError("CLOB unavailable"))
        client = LiveWriteClient(fake, dry_run=False)
        with self.assertRaises(LiveClientError) as ctx:
            client.submit_order("tok", "BUY", 0.50, 10.0)
        self.assertIn("submit_order failed", str(ctx.exception))

    def test_original_exception_is_chained(self) -> None:
        original = ValueError("bad tick size")
        fake = _FakeClient(raises=original)
        client = LiveWriteClient(fake, dry_run=False)
        try:
            client.submit_order("tok", "BUY", 0.50, 10.0)
            self.fail("expected LiveClientError")
        except LiveClientError as exc:
            self.assertIs(exc.__cause__, original)


# ---------------------------------------------------------------------------
# cancel_order — dry_run
# ---------------------------------------------------------------------------

class TestCancelOrderDryRun(unittest.TestCase):

    def setUp(self) -> None:
        self.fake = _FakeClient()
        self.client = LiveWriteClient(self.fake, dry_run=True)

    def test_returns_true(self) -> None:
        self.assertTrue(self.client.cancel_order("some-id"))

    def test_makes_no_network_call(self) -> None:
        self.client.cancel_order("some-id")
        self.assertEqual(self.fake.calls, [])


# ---------------------------------------------------------------------------
# cancel_order — live mode
# ---------------------------------------------------------------------------

class TestCancelOrderLive(unittest.TestCase):

    def test_calls_cancel_with_order_id_string(self) -> None:
        fake = _FakeClient()
        client = LiveWriteClient(fake, dry_run=False)
        client.cancel_order("ord-123")
        self.assertEqual(len(fake.calls), 1)
        call_name, call_arg = fake.calls[0][0], fake.calls[0][1]
        self.assertEqual(call_name, "cancel")
        self.assertEqual(call_arg, "ord-123")

    def test_returns_true_on_success(self) -> None:
        client = _live_client()
        self.assertTrue(client.cancel_order("ord-abc"))

    def test_normalizes_exception_to_live_client_error(self) -> None:
        fake = _FakeClient(raises=ConnectionError("timeout"))
        client = LiveWriteClient(fake, dry_run=False)
        with self.assertRaises(LiveClientError) as ctx:
            client.cancel_order("ord-xyz")
        self.assertIn("cancel_order failed", str(ctx.exception))
        self.assertIn("ord-xyz", str(ctx.exception))


# ---------------------------------------------------------------------------
# get_order_status — dry_run
# ---------------------------------------------------------------------------

class TestGetOrderStatusDryRun(unittest.TestCase):

    def test_raises_live_client_error_in_dry_run(self) -> None:
        client = _dry_client()
        with self.assertRaises(LiveClientError) as ctx:
            client.get_order_status("ord-1")
        self.assertIn("dry_run", str(ctx.exception))

    def test_makes_no_network_call(self) -> None:
        fake = _FakeClient()
        client = LiveWriteClient(fake, dry_run=True)
        try:
            client.get_order_status("ord-1")
        except LiveClientError:
            pass
        self.assertEqual(fake.calls, [])


# ---------------------------------------------------------------------------
# get_order_status — live mode
# ---------------------------------------------------------------------------

class TestGetOrderStatusLive(unittest.TestCase):

    def test_returns_normalized_status_from_object(self) -> None:
        fake = _FakeClient(order_response=_FakeOrder(
            id="ord-99", status="live", size_matched="25.0", size="50.0"
        ))
        client = LiveWriteClient(fake, dry_run=False)
        status = client.get_order_status("ord-99")
        self.assertIsInstance(status, LiveOrderStatus)
        self.assertEqual(status.order_id, "ord-99")
        self.assertEqual(status.status, "live")
        self.assertAlmostEqual(status.size_matched, 25.0)
        self.assertAlmostEqual(status.size_remaining, 25.0)

    def test_returns_normalized_status_from_dict(self) -> None:
        fake = _FakeClient(order_response={
            "id": "ord-77", "status": "matched",
            "size_matched": "50.0", "size": "50.0"
        })
        client = LiveWriteClient(fake, dry_run=False)
        status = client.get_order_status("ord-77")
        self.assertEqual(status.order_id, "ord-77")
        self.assertEqual(status.status, "matched")
        self.assertAlmostEqual(status.size_remaining, 0.0)

    def test_size_remaining_floored_at_zero(self) -> None:
        # size_matched > size shouldn't produce negative remaining
        fake = _FakeClient(order_response=_FakeOrder(
            size_matched="60.0", size="50.0"
        ))
        client = LiveWriteClient(fake, dry_run=False)
        status = client.get_order_status("x")
        self.assertGreaterEqual(status.size_remaining, 0.0)

    def test_normalizes_exception_to_live_client_error(self) -> None:
        fake = _FakeClient(raises=OSError("network error"))
        client = LiveWriteClient(fake, dry_run=False)
        with self.assertRaises(LiveClientError) as ctx:
            client.get_order_status("ord-fail")
        self.assertIn("get_order_status failed", str(ctx.exception))


# ---------------------------------------------------------------------------
# LiveClientError contract
# ---------------------------------------------------------------------------

class TestGetTokenBalance(unittest.TestCase):

    def test_balance_in_microunits_always_divided_by_1e6(self) -> None:
        # Polymarket returns balances in 1e6 units (e.g. 171_800_000 = 171.8 shares)
        fake = _FakeClient(balance_response={"balance": "171800000"})
        client = LiveWriteClient(fake, dry_run=False)
        result = client.get_token_balance("tok-1")
        self.assertAlmostEqual(result, 171.8, places=4)

    def test_small_balance_also_divided_by_1e6(self) -> None:
        # Previously the bug: balance <= 10_000 was NOT divided, returning raw int
        fake = _FakeClient(balance_response={"balance": "5000"})
        client = LiveWriteClient(fake, dry_run=False)
        result = client.get_token_balance("tok-1")
        self.assertAlmostEqual(result, 0.005, places=6)

    def test_zero_balance_returns_zero(self) -> None:
        fake = _FakeClient(balance_response={"balance": "0"})
        client = LiveWriteClient(fake, dry_run=False)
        self.assertEqual(client.get_token_balance("tok-1"), 0.0)

    def test_dry_run_returns_zero_without_api_call(self) -> None:
        fake = _FakeClient()
        client = LiveWriteClient(fake, dry_run=True)
        self.assertEqual(client.get_token_balance("tok-1"), 0.0)
        self.assertEqual(len(fake.calls), 0)

    def test_api_error_raises_live_client_error(self) -> None:
        fake = _FakeClient(raises=RuntimeError("network"))
        client = LiveWriteClient(fake, dry_run=False)
        with self.assertRaises(LiveClientError):
            client.get_token_balance("tok-1")


class TestLiveClientError(unittest.TestCase):

    def test_is_exception_subclass(self) -> None:
        self.assertTrue(issubclass(LiveClientError, Exception))

    def test_can_be_raised_with_message(self) -> None:
        with self.assertRaises(LiveClientError) as ctx:
            raise LiveClientError("something went wrong")
        self.assertIn("something went wrong", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
