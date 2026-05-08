from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_live_market_microstructure_report import main
from src.live.market_microstructure_readiness import build_market_microstructure_report


NOW = datetime(2026, 5, 8, 1, 0, tzinfo=timezone.utc)
MARKET = "will-ivan-cepeda-castro-win-the-2026-colombian-presidential-election"


def _health(**target_overrides):
    target = {
        "market_slug": MARKET,
        "token_id": "123",
        "best_bid": 0.36,
        "best_ask": 0.38,
        "quote_bid": 0.36,
        "quote_ask": 0.38,
        "quote_size": 60.0,
        "tick_size": 0.01,
        "rewards_min_size": 50.0,
        "rewards_max_spread_cents": 4.5,
    }
    target.update(target_overrides)
    return {"generated_at_utc": NOW.isoformat(), "target_market": target}


def test_tick_aligned_quote_is_ready_and_report_only() -> None:
    report = build_market_microstructure_report(health_report=_health(), now=NOW)

    assert report["status"] == "MARKET_MICROSTRUCTURE_READY"
    assert report["blockers"] == []
    assert report["quote_bid"] == 0.36
    assert report["quote_ask"] == 0.38
    assert report["tick_size"] == 0.01
    assert report["can_submit_order"] is False


def test_missing_tick_size_blocks() -> None:
    report = build_market_microstructure_report(health_report=_health(tick_size=None), now=NOW)

    assert report["status"] == "MARKET_MICROSTRUCTURE_BLOCKED"
    assert "TICK_SIZE_MISSING" in report["blockers"]
    assert "QUOTE_BID_TICK_MISALIGNED" in report["blockers"]
    assert "QUOTE_ASK_TICK_MISALIGNED" in report["blockers"]


def test_misaligned_quote_blocks() -> None:
    report = build_market_microstructure_report(
        health_report=_health(quote_bid=0.3645, quote_ask=0.38, tick_size=0.01),
        now=NOW,
    )

    assert "QUOTE_BID_TICK_MISALIGNED" in report["blockers"]
    assert report["checks"]["quote_ask_tick_aligned"] is True


def test_inverted_quote_blocks() -> None:
    report = build_market_microstructure_report(
        health_report=_health(quote_bid=0.38, quote_ask=0.38),
        now=NOW,
    )

    assert "QUOTE_PRICE_INVERSION" in report["blockers"]


def test_out_of_bounds_quote_blocks() -> None:
    report = build_market_microstructure_report(
        health_report=_health(quote_bid=0.0, quote_ask=1.01),
        now=NOW,
    )

    assert "QUOTE_PRICE_OUT_OF_BOUNDS" in report["blockers"]


def test_tick_size_reader_fills_missing_tick() -> None:
    report = build_market_microstructure_report(
        health_report=_health(tick_size=None),
        tick_size_reader=lambda token_id: "0.01",
        now=NOW,
    )

    assert report["status"] == "MARKET_MICROSTRUCTURE_READY"
    assert report["tick_size_source"] == "CLOB_GET_TICK_SIZE"


def test_cli_writes_market_microstructure_report(tmp_path: Path) -> None:
    health = tmp_path / "health.json"
    health.write_text(json.dumps(_health()), encoding="utf-8")
    missing_candidate = tmp_path / "missing.json"
    out = tmp_path / "market_microstructure.json"

    rc = main(["--health-report", str(health), "--candidate-report", str(missing_candidate), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == "MARKET_MICROSTRUCTURE_READY"
    assert payload["can_submit_order"] is False
