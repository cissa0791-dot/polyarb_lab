from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.build_toxic_flow_report import main
from src.live.toxic_flow_detector import (
    build_toxic_flow_report,
    calc_imbalance,
    calc_price_momentum,
)


NOW = datetime(2026, 5, 8, 1, 0, tzinfo=timezone.utc)


def test_calc_imbalance_is_signed_and_normalized() -> None:
    assert calc_imbalance(bid_size=90, ask_size=10) == 0.8
    assert calc_imbalance(bid_size=10, ask_size=90) == -0.8
    assert calc_imbalance(bid_size=0, ask_size=0) is None


def test_calc_price_momentum_uses_lookback_midpoints() -> None:
    snapshots = [
        {"generated_at_utc": (NOW - timedelta(seconds=55)).isoformat(), "best_bid": 0.36, "best_ask": 0.38},
        {"generated_at_utc": (NOW - timedelta(seconds=5)).isoformat(), "best_bid": 0.38, "best_ask": 0.40},
    ]

    assert calc_price_momentum(snapshots, now=NOW, lookback_seconds=60) == 0.02


def test_extreme_orderbook_imbalance_blocks_as_adverse_selection_risk() -> None:
    report = build_toxic_flow_report(
        explicit={
            "market_slug": "ivan",
            "best_bid": 0.36,
            "best_ask": 0.37,
            "best_bid_size": 95,
            "best_ask_size": 5,
        },
        now=NOW,
    )

    assert report["status"] == "TOXIC_FLOW_BLOCKED"
    assert report["orderbook_imbalance"] == 0.9
    assert report["toxic_flow_detected"] is True
    assert "ADVERSE_SELECTION_RISK" in report["blockers"]
    assert report["can_submit_order"] is False


def test_volatility_lock_blocks_when_momentum_exceeds_spread_multiple() -> None:
    snapshots = [
        {"generated_at_utc": (NOW - timedelta(seconds=50)).isoformat(), "best_bid": 0.36, "best_ask": 0.37},
        {"generated_at_utc": (NOW - timedelta(seconds=1)).isoformat(), "best_bid": 0.39, "best_ask": 0.40},
    ]

    report = build_toxic_flow_report(
        explicit={
            "market_slug": "ivan",
            "best_bid": 0.39,
            "best_ask": 0.40,
            "best_bid_size": 50,
            "best_ask_size": 50,
            "quote_spread": 0.01,
        },
        snapshots=snapshots,
        now=NOW,
    )

    assert report["status"] == "TOXIC_FLOW_BLOCKED"
    assert report["volatility_lock"] is True
    assert "VOLATILITY_LOCK" in report["blockers"]


def test_balanced_book_produces_ready_fill_inputs_for_gate() -> None:
    report = build_toxic_flow_report(
        explicit={
            "market_slug": "ivan",
            "best_bid": 0.36,
            "best_ask": 0.37,
            "best_bid_size": 50,
            "best_ask_size": 50,
        },
        now=NOW,
    )

    assert report["status"] == "TOXIC_FLOW_READY"
    assert report["toxic_flow_detected"] is False
    assert report["fill_probability"] == 1.0
    assert report["adverse_selection_score"] == 0.0


def test_cli_writes_toxic_flow_report(tmp_path: Path) -> None:
    market_report = tmp_path / "market.json"
    out = tmp_path / "toxic.json"
    market_report.write_text(
        json.dumps(
            {
                "market_slug": "ivan",
                "best_bid": 0.36,
                "best_ask": 0.37,
                "best_bid_size": 50,
                "best_ask_size": 50,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--market-report", str(market_report), "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["status"] == "TOXIC_FLOW_READY"
