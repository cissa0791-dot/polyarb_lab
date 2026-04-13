"""
auto_maker_loop — market_ws_client
polyarb_lab / research_lines / auto_maker_loop / modules

Thin public market websocket wrapper for Polymarket CLOB.

Captures only: book (best_bid_ask), last_trade_price, price_change,
tick_size_change events for the subscribed token IDs.
Persists to a JSONL event log (one JSON object per line).
Runs in a background daemon thread — never blocks the main loop.

Usage
-----
    from research_lines.auto_maker_loop.modules.market_ws_client import MarketWsClient

    client = MarketWsClient(token_ids=["<token_id>"], log_path=Path("market_ws.jsonl"))
    client.start()
    # ... main loop runs ...
    client.stop()
    print(f"market events received: {client.events_received}")

Event fields persisted (raw from exchange + injected):
    All raw fields from the Polymarket WS message, plus:
    _recv_ts   : ISO 8601 UTC timestamp at local receive time
    _event_type: normalized event type string

No auth required (market channel is public).
No web3.  No trading.  Read-only event capture.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Only persist these event types (empty set = persist all)
_CAPTURE_TYPES: set[str] = {
    "book",
    "last_trade_price",
    "price_change",
    "tick_size_change",
}


class MarketWsClient:
    """
    Background market websocket client for one or more token IDs.

    Thread-safe: start() and stop() may be called from any thread.
    """

    def __init__(
        self,
        token_ids: List[str],
        log_path: Path,
    ) -> None:
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        self.token_ids   = list(token_ids)
        self.log_path    = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.events_received: int  = 0
        self.trade_count: int      = 0   # count of last_trade_price events (fill proxy)
        self.last_best_bid: Optional[float] = None
        self.last_best_ask: Optional[float] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background listener thread. No-op if already running."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="market-ws", daemon=True
        )
        self._thread.start()
        logger.info(
            "market_ws_client: started  tokens=%d  log=%s",
            len(self.token_ids), self.log_path,
        )

    def stop(self, timeout: float = 6.0) -> None:
        """Signal stop and wait for background thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(
            "market_ws_client: stopped  events=%d  trades=%d  "
            "last_bid=%s  last_ask=%s",
            self.events_received, self.trade_count,
            self.last_best_bid, self.last_best_ask,
        )

    def summary(self) -> dict:
        """Return a lightweight snapshot dict for inclusion in cycle results."""
        return {
            "market_ws_events":     self.events_received,
            "market_ws_trades":     self.trade_count,
            "market_ws_last_bid":   self.last_best_bid,
            "market_ws_last_ask":   self.last_best_ask,
        }

    # ── Thread entry point ─────────────────────────────────────────────────

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_with_reconnect())
        except Exception as exc:
            logger.warning("market_ws_client: thread_main error: %s", exc)
        finally:
            loop.close()

    # ── Main async loop (reconnects on transient errors) ──────────────────

    async def _run_with_reconnect(self) -> None:
        retry_delay = 2.0
        while not self._stop_event.is_set():
            try:
                await self._connect_and_listen()
                retry_delay = 2.0
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "market_ws_client: connection error (%s) — retry in %.0fs",
                    exc, retry_delay,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0)

    async def _connect_and_listen(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect  # type: ignore[no-redef]

        async with connect(WS_MARKET_URL) as ws:
            logger.info("market_ws_client: connected")
            await ws.send(self._subscribe_msg())
            logger.info(
                "market_ws_client: subscribed  token_ids=%s",
                [t[:16] for t in self.token_ids],
            )

            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                self._handle(raw)

    # ── Subscribe message ──────────────────────────────────────────────────

    def _subscribe_msg(self) -> str:
        return json.dumps({
            "type":       "market",
            "assets_ids": self.token_ids,
        })

    # ── Event handler ──────────────────────────────────────────────────────

    def _handle(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        events: list[Any] = payload if isinstance(payload, list) else [payload]
        recv_ts = datetime.now(timezone.utc).isoformat()

        with self.log_path.open("a", encoding="utf-8") as fh:
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(
                    event.get("event_type") or event.get("type") or ""
                )
                if _CAPTURE_TYPES and event_type not in _CAPTURE_TYPES:
                    continue

                # Update in-memory top-of-book snapshot
                if event_type == "book":
                    bids = event.get("bids") or event.get("bid") or []
                    asks = event.get("asks") or event.get("ask") or []
                    if bids:
                        try:
                            self.last_best_bid = float(bids[0].get("price") or bids[0])
                        except (TypeError, ValueError, IndexError):
                            pass
                    if asks:
                        try:
                            self.last_best_ask = float(asks[0].get("price") or asks[0])
                        except (TypeError, ValueError, IndexError):
                            pass
                elif event_type == "last_trade_price":
                    self.trade_count += 1

                event["_recv_ts"]    = recv_ts
                event["_event_type"] = event_type
                fh.write(json.dumps(event) + "\n")
                self.events_received += 1
