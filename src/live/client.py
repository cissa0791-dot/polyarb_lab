"""Authenticated live write client for the Polymarket CLOB.

Wraps the three py_clob_client write methods needed for live order
execution behind a clean internal interface:

    submit_order      → ClobClient.create_and_post_order
    cancel_order      → ClobClient.cancel
    get_order_status  → ClobClient.get_order

All py_clob_client exceptions are converted to LiveClientError so that
callers never need to import from py_clob_client directly.

When dry_run=True (the safe default) every mutating method is a no-op:
submit_order returns a sentinel result, cancel_order returns True, and
get_order_status raises LiveClientError because there are no real orders
to poll.  dry_run=True is the only mode available until the execution
router (L4) explicitly enables live submission.

Usage::

    from src.live.auth import load_live_credentials
    from src.live.client import LiveWriteClient

    creds = load_live_credentials()
    client = LiveWriteClient.from_credentials(
        creds, host="https://clob.polymarket.com", dry_run=True
    )
    result = client.submit_order(
        token_id="abc123", side="BUY", price=0.55, size=50.0
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions

from src.live.auth import LiveCredentials, build_authenticated_client


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class LiveClientError(Exception):
    """Raised when a live CLOB write operation fails or is called incorrectly."""


# ---------------------------------------------------------------------------
# Normalised result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveOrderResult:
    """Normalised result returned by submit_order.

    Attributes:
        order_id:     CLOB-assigned order ID string; None in dry_run mode.
        status:       CLOB status string ("live", "matched", "delayed", …)
                      or "dry_run" when dry_run=True.
        size_matched: Shares matched at submission time.  Resting GTC orders
                      typically show 0.0 here; poll get_order_status for fills.
        avg_price:    Actual average matched price from CLOB; None if not filled
                      at submission time or if the field is absent in the response.
        dry_run:      True when the result is a no-op sentinel.
    """
    order_id: str | None
    status: str
    size_matched: float
    avg_price: float | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class LiveOrderStatus:
    """Normalised status returned by get_order_status.

    Attributes:
        order_id:       CLOB order ID.
        status:         CLOB status string.
        size_matched:   Shares filled so far.
        size_remaining: Shares still open (total - matched, floored at 0).
        avg_price:      Actual average matched price from CLOB; None if the
                        order has no fills yet or the field is absent.
    """
    order_id: str
    status: str
    size_matched: float
    size_remaining: float
    avg_price: float | None = None


# ---------------------------------------------------------------------------
# Dry-run sentinels
# ---------------------------------------------------------------------------

_DRY_RUN_RESULT = LiveOrderResult(
    order_id=None,
    status="dry_run",
    size_matched=0.0,
    dry_run=True,
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LiveWriteClient:
    """Authenticated write client for the Polymarket CLOB.

    Construct via from_credentials (production) or pass a pre-built
    ClobClient directly (testing / injection).

    Attributes:
        dry_run: When True all mutating methods are no-ops.  Default: True.
    """

    def __init__(self, client: ClobClient, *, dry_run: bool = True) -> None:
        self._client = client
        self.dry_run = dry_run

    @classmethod
    def from_credentials(
        cls,
        creds: LiveCredentials,
        host: str,
        *,
        dry_run: bool = True,
        signature_type: int | None = None,
        funder: str | None = None,
    ) -> "LiveWriteClient":
        """Construct from validated LiveCredentials.

        Args:
            creds:          credentials from load_live_credentials().
            host:           CLOB base URL, e.g. "https://clob.polymarket.com".
            dry_run:        when True (default), no real orders are submitted.
            signature_type: 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE.
                            None defaults to EOA.  Most Polymarket web users
                            need POLY_PROXY (1) with their proxy wallet funder.
            funder:         Proxy/Gnosis wallet address (maker in orders).
                            None defaults to the EOA address of the private key.
        """
        return cls(
            build_authenticated_client(creds, host, signature_type=signature_type, funder=funder),
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        *,
        neg_risk: bool = False,
        tick_size: str | None = None,
        fee_rate_bps: int = 0,
    ) -> LiveOrderResult:
        """Submit a GTC limit order to the CLOB.

        In dry_run mode returns the DRY_RUN sentinel; no network call is made.

        Args:
            token_id:     Binary outcome token ID.
            side:         "BUY" or "SELL".
            price:        Limit price in (0, 1).
            size:         Order size in shares.
            neg_risk:     True for neg-risk (complementary) markets.
            tick_size:    Tick size string ("0.1", "0.01", "0.001", "0.0001").
                          None lets the CLOB client resolve it from the token.
            fee_rate_bps: Maker fee rate in basis points (default 0).

        Returns:
            LiveOrderResult with order_id, status, and size_matched.

        Raises:
            LiveClientError: on any CLOB or network error.
        """
        if self.dry_run:
            return _DRY_RUN_RESULT

        if tick_size is None:
            try:
                tick_size = self._client.get_tick_size(token_id)
            except Exception:
                tick_size = None
        if not neg_risk:
            try:
                neg_risk = bool(self._client.get_neg_risk(token_id))
            except Exception:
                neg_risk = False

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            fee_rate_bps=fee_rate_bps,
        )
        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )
        try:
            raw: dict[str, Any] = self._client.create_and_post_order(order_args, options)
        except Exception as exc:
            raise LiveClientError(
                f"submit_order failed for token={token_id!r} side={side} "
                f"price={price} size={size}: {exc}"
            ) from exc

        raw_avg = raw.get("avg_price") or raw.get("avgPrice")
        return LiveOrderResult(
            order_id=str(raw.get("orderID") or raw.get("order_id") or ""),
            status=str(raw.get("status", "unknown")),
            size_matched=float(raw.get("size_matched") or 0.0),
            avg_price=float(raw_avg) if raw_avg is not None else None,
        )

    # ------------------------------------------------------------------
    # Order cancellation
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a live order by its CLOB order ID.

        In dry_run mode returns True; no network call is made.

        Returns:
            True when the CLOB accepted the cancellation request.

        Raises:
            LiveClientError: on any CLOB or network error.
        """
        if self.dry_run:
            return True

        try:
            self._client.cancel(order_id)
            return True
        except Exception as exc:
            raise LiveClientError(
                f"cancel_order failed for order_id={order_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Order status polling
    # ------------------------------------------------------------------

    def get_order_status(self, order_id: str) -> LiveOrderStatus:
        """Poll the CLOB for the current fill status of an order.

        Not available in dry_run mode (there are no real orders to poll).

        Args:
            order_id: CLOB order ID returned by submit_order.

        Returns:
            LiveOrderStatus with current status and matched/remaining sizes.

        Raises:
            LiveClientError: in dry_run mode, or on any CLOB/network error.
        """
        if self.dry_run:
            raise LiveClientError(
                "get_order_status is not available in dry_run mode: "
                f"order_id={order_id!r}"
            )

        try:
            raw = self._client.get_order(order_id)
        except Exception as exc:
            raise LiveClientError(
                f"get_order_status failed for order_id={order_id!r}: {exc}"
            ) from exc

        try:
            size_matched = float(getattr(raw, "size_matched", None) or
                                 (raw.get("size_matched") if isinstance(raw, dict) else None) or
                                 0.0)
            total_size   = float(getattr(raw, "size", None) or
                                 getattr(raw, "original_size", None) or
                                 getattr(raw, "originalSize", None) or
                                 (raw.get("size") if isinstance(raw, dict) else None) or
                                 (raw.get("original_size") if isinstance(raw, dict) else None) or
                                 (raw.get("originalSize") if isinstance(raw, dict) else None) or
                                 0.0)
            size_remaining = max(0.0, total_size - size_matched)
            raw_id     = (getattr(raw, "id", None) or
                          getattr(raw, "orderID", None) or
                          (raw.get("id") if isinstance(raw, dict) else None) or
                          order_id)
            raw_status = (getattr(raw, "status", None) or
                          (raw.get("status") if isinstance(raw, dict) else None) or
                          "unknown")
            raw_avg = (getattr(raw, "avg_price", None) or
                       getattr(raw, "avgPrice", None) or
                       (raw.get("avg_price") if isinstance(raw, dict) else None) or
                       (raw.get("avgPrice") if isinstance(raw, dict) else None))
        except (TypeError, ValueError) as exc:
            raise LiveClientError(
                f"get_order_status: unexpected response shape for "
                f"order_id={order_id!r}: {exc}"
            ) from exc

        return LiveOrderStatus(
            order_id=str(raw_id),
            status=str(raw_status),
            size_matched=size_matched,
            size_remaining=size_remaining,
            avg_price=float(raw_avg) if raw_avg is not None else None,
        )
