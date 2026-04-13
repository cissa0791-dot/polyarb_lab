"""
auto_maker_loop — user_ws_client
polyarb_lab / research_lines / auto_maker_loop / modules

Thin authenticated user websocket wrapper for Polymarket CLOB.

Captures only: order_update, trade, status events.
Persists to a JSONL event log (one JSON object per line).
Runs in a background daemon thread — never blocks the main loop.

Usage
-----
    client = UserWsClient(api_key, api_secret, api_passphrase, log_path)
    client.start()
    # ... main loop runs ...
    client.stop()

Event fields persisted (raw from exchange + injected):
    All raw fields from the Polymarket WS message, plus:
    _recv_ts  : ISO 8601 UTC timestamp at local receive time
    _type     : top-level event type string

No web3.  No trading.  No order placement.  Read-only event capture.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

WS_USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

# Only persist these top-level event types (empty set = persist all)
_CAPTURE_TYPES: set[str] = set()  # empty = persist all; tighten once event type names confirmed


class UserWsClient:
    """
    Background user websocket client.

    Thread-safe: start() and stop() may be called from any thread.
    Log writes are append-only and do not hold any lock beyond the OS file lock.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        log_path: Path,
    ) -> None:
        self.api_key        = api_key
        self.api_secret     = api_secret
        self.api_passphrase = api_passphrase
        self.log_path       = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.events_received: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background listener thread. No-op if already running."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="user-ws", daemon=True
        )
        self._thread.start()
        logger.info("user_ws_client: started (log=%s)", self.log_path)

    def stop(self, timeout: float = 6.0) -> None:
        """Signal stop and wait for background thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(
            "user_ws_client: stopped (events_received=%d)", self.events_received
        )

    # ── Thread entry point ─────────────────────────────────────────────────

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_with_reconnect())
        except Exception as exc:
            logger.warning("user_ws_client: thread_main error: %s", exc)
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
                    "user_ws_client: connection error (%s) — retry in %.0fs", exc, retry_delay
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60.0)

    async def _connect_and_listen(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect  # type: ignore[no-redef]

        async with connect(WS_USER_URL) as ws:
            logger.info("user_ws_client: connected")
            await ws.send(self._subscribe_msg())
            logger.info("user_ws_client: subscribed")

            while not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                self._handle(raw)

    # ── Subscribe message ──────────────────────────────────────────────────

    def _subscribe_msg(self) -> str:
        return json.dumps({
            "type": "user",
            "auth": {
                "apiKey":     self.api_key,
                "secret":     self.api_secret,
                "passphrase": self.api_passphrase,
            },
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
                event_type = str(event.get("event_type") or event.get("type") or "")
                if _CAPTURE_TYPES and event_type not in _CAPTURE_TYPES:
                    continue
                event["_recv_ts"] = recv_ts
                event["_type"]    = event_type
                fh.write(json.dumps(event) + "\n")
                self.events_received += 1
