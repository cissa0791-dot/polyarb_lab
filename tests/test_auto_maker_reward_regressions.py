from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lines.auto_maker_loop.modules import reward_attribution_probe as rap
from research_lines.auto_maker_loop.modules import reward_aware_quote_planner_v2 as planner


def test_apply_zone_bid_rounds_up_into_zone() -> None:
    notes: list[str] = []
    bid = planner._apply_zone_bid(0.34, 0.3435, notes)

    assert bid == pytest.approx(0.35, abs=1e-9)
    assert notes == ["bid clamped from 0.3400 to zone_bid=0.3435"]


def test_apply_zone_ask_rounds_down_into_zone() -> None:
    notes: list[str] = []
    ask = planner._apply_zone_ask(0.40, 0.3885, notes)

    assert ask == pytest.approx(0.38, abs=1e-9)
    assert notes == ["ask clamped from 0.4000 to zone_ask=0.3885"]


class _FakeResponse:
    def __init__(self, url: str, body: dict[str, str]) -> None:
        self.status_code = 200
        self.request = SimpleNamespace(url=url)
        self.text = str(body)
        self._body = body

    def json(self) -> dict[str, str]:
        return self._body


class _FakeClient:
    def __init__(self, body: dict[str, str]) -> None:
        self.calls: list[dict[str, object]] = []
        self._body = body

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, headers=None, params=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
        )
        return _FakeResponse(url, self._body)


def test_fetch_user_total_uses_base_path_headers_only() -> None:
    fake_client = _FakeClient({"total_earnings": "12.5"})
    creds = SimpleNamespace(
        api_key="api-key",
        api_secret="c2VjcmV0",
        api_passphrase="passphrase",
        signature_type=2,
        funder="0x123",
        private_key="0x" + "11" * 32,
    )

    with patch("httpx.Client", return_value=fake_client), patch("builtins.print"):
        total, raw = rap.fetch_user_total(
            "https://clob.polymarket.com",
            creds,
            date="2026-03-29",
        )

    assert total == pytest.approx(12.5, abs=1e-9)
    assert raw == "{'total_earnings': '12.5'}"
    assert len(fake_client.calls) == 1

    call = fake_client.calls[0]
    assert call["url"] == "https://clob.polymarket.com/rewards/user/total"
    assert call["params"] is None
    assert call["timeout"] == 8
    assert set(call["headers"]) == {
        "CLOB-API-KEY",
        "CLOB-SIGNATURE",
        "CLOB-TIMESTAMP",
        "CLOB-PASSPHRASE",
    }
    assert "CLOB-SIGNATURE-TYPE" not in call["headers"]
    assert "POLY_ADDRESS" not in call["headers"]
