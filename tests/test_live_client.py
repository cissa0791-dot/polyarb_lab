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
    LiveOpenOrder,
    LiveOrderResult,
    LiveOrderStatus,
    LiveWriteClient,
    clean_live_error_message,
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
        submit_raises: list[Exception | None] | None = None,
        neg_risk: bool = False,
        open_orders_response: object | None = None,
    ):
        self.calls: list[tuple] = []
        self._submit_response = submit_response or {
            "orderID": "fake-order-123",
            "status": "live",
            "size_matched": "0.0",
        }
        self._order_response = order_response or _FakeOrder()
        self._raises = raises
        self._submit_raises = list(submit_raises or [])
        self._neg_risk = neg_risk
        self._open_orders_response = open_orders_response if open_orders_response is not None else []

    def create_and_post_order(self, order_args, options):
        self.calls.append(("create_and_post_order", order_args, options))
        if self._submit_raises:
            exc = self._submit_raises.pop(0)
            if exc is not None:
                raise exc
        if self._raises:
            raise self._raises
        return self._submit_response

    def get_tick_size(self, token_id):
        self.calls.append(("get_tick_size", token_id))
        return "0.01"

    def get_neg_risk(self, token_id):
        self.calls.append(("get_neg_risk", token_id))
        return self._neg_risk

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

    def get_open_orders(self, *args):
        self.calls.append(("get_open_orders", args))
        if self._raises:
            raise self._raises
        return self._open_orders_response


class _FakeBuilder:
    def __init__(self, owner):
        self.owner = owner

    def create_order(self, order_args, options):
        self.owner.calls.append(("builder.create_order", order_args, options))
        return {"order_args": order_args, "options": options}


class _FakeBuilderClient(_FakeClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.builder = _FakeBuilder(self)

    def post_order(self, order):
        self.calls.append(("post_order", order))
        return self._submit_response


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

    def test_uses_clob_neg_risk_over_caller_hint(self) -> None:
        fake = _FakeClient(neg_risk=False)
        client = LiveWriteClient(fake, dry_run=False)
        client.submit_order("tok", "BUY", 0.50, 10.0, neg_risk=True)
        order_call = fake.calls[-1]
        self.assertEqual(order_call[0], "create_and_post_order")
        self.assertFalse(order_call[2].neg_risk)

    def test_retries_order_version_mismatch_with_opposite_neg_risk(self) -> None:
        fake = _FakeClient(
            submit_raises=[
                RuntimeError("PolyApiException[status_code=400, error_message={'error': 'order version mismatch'}]"),
                None,
            ],
            neg_risk=True,
        )
        client = LiveWriteClient(fake, dry_run=False)

        result = client.submit_order("tok", "BUY", 0.50, 10.0)

        order_calls = [call for call in fake.calls if call[0] == "create_and_post_order"]
        self.assertEqual(result.order_id, "fake-order-123")
        self.assertEqual(len(order_calls), 2)
        self.assertTrue(order_calls[0][2].neg_risk)
        self.assertFalse(order_calls[1][2].neg_risk)

    def test_retries_order_version_mismatch_with_underscore_error(self) -> None:
        fake = _FakeClient(
            submit_raises=[
                RuntimeError("PolyApiException[status_code=400, error_message={'error': 'order_version_mismatch'}]"),
                None,
            ],
            neg_risk=True,
        )
        client = LiveWriteClient(fake, dry_run=False)

        result = client.submit_order("tok", "BUY", 0.50, 10.0)

        order_calls = [call for call in fake.calls if call[0] == "create_and_post_order"]
        self.assertEqual(result.order_id, "fake-order-123")
        self.assertEqual(len(order_calls), 2)
        self.assertFalse(order_calls[1][2].neg_risk)

    def test_builder_path_preserves_false_neg_risk(self) -> None:
        fake = _FakeBuilderClient(neg_risk=False)
        client = LiveWriteClient(fake, dry_run=False)

        result = client.submit_order("tok", "BUY", 0.50, 10.0, neg_risk=True)

        builder_call = next(call for call in fake.calls if call[0] == "builder.create_order")
        self.assertEqual(result.order_id, "fake-order-123")
        self.assertFalse(builder_call[2].neg_risk)

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
# get_all_open_orders — live mode
# ---------------------------------------------------------------------------

class TestGetAllOpenOrdersLive(unittest.TestCase):

    def test_returns_normalized_account_open_orders_with_token_ids(self) -> None:
        fake = _FakeClient(open_orders_response=[
            {
                "id": "ord-1",
                "side": "BUY",
                "price": "0.41",
                "size": "50",
                "size_matched": "0",
                "asset_id": "tok-1",
                "status": "open",
            },
            {
                "orderID": "ord-2",
                "side": "SELL",
                "price": "0.50",
                "originalSize": "25",
                "sizeMatched": "5",
                "assetId": "tok-2",
            },
        ])
        client = LiveWriteClient(fake, dry_run=False)

        orders = client.get_all_open_orders()

        self.assertEqual(len(orders), 2)
        self.assertIsInstance(orders[0], LiveOpenOrder)
        self.assertEqual(orders[0].token_id, "tok-1")
        self.assertEqual(orders[0].side, "BUY")
        self.assertAlmostEqual(orders[1].size_remaining, 20.0)
        self.assertEqual(fake.calls[0], ("get_open_orders", ()))

    def test_rate_limit_html_body_is_sanitized(self) -> None:
        html = "<!doctype html><html><body>" + ("x" * 5000) + "</body></html>"
        fake = _FakeClient(
            raises=RuntimeError(
                "[py_clob_client_v2] request error status=429 "
                "url=https://clob.polymarket.com/balance-allowance "
                "body=Cloudflare error code: 1015 " + html
            )
        )
        client = LiveWriteClient(fake, dry_run=False)

        with self.assertRaises(LiveClientError) as ctx:
            client.get_all_open_orders()

        message = str(ctx.exception)
        self.assertIn("HTTP 429", message)
        self.assertIn("Cloudflare rate limit", message)
        self.assertNotIn("<html", message)
        self.assertLess(len(message), 700)


# ---------------------------------------------------------------------------
# LiveClientError contract
# ---------------------------------------------------------------------------

class TestLiveClientError(unittest.TestCase):

    def test_is_exception_subclass(self) -> None:
        self.assertTrue(issubclass(LiveClientError, Exception))

    def test_can_be_raised_with_message(self) -> None:
        with self.assertRaises(LiveClientError) as ctx:
            raise LiveClientError("something went wrong")
        self.assertIn("something went wrong", str(ctx.exception))

    def test_clean_live_error_message_trims_html(self) -> None:
        message = clean_live_error_message(
            "status=429 body=Cloudflare error code: 1015 <!doctype html><html>long</html>"
        )
        self.assertIn("HTTP 429", message)
        self.assertIn("Cloudflare rate limit", message)
        self.assertNotIn("<html", message)


if __name__ == "__main__":
    unittest.main()
