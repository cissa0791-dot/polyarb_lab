from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.build_fill_lifecycle_logger_report import main
from src.live.fill_lifecycle_logger import (
    append_fill_lifecycle_event,
    build_fill_lifecycle_event,
    read_fill_lifecycle_events,
    summarize_fill_lifecycle_events,
)


NOW = datetime(2026, 5, 9, 15, 0, tzinfo=timezone.utc)


def test_minimal_fill_event_records_cash_inventory_fee_and_reward_separation() -> None:
    event = build_fill_lifecycle_event(
        lifecycle_id="c-1",
        event_type="PARTIAL_FILL",
        market="m",
        side="BID_ONLY",
        order_id="0xabc",
        price=0.41,
        size=10,
        fill_quantity=5,
        inventory_delta_shares=5,
        cash_delta_usdc=-2.05,
        fees_usdc=0.01,
        realized_spread_pnl_usdc=0.0,
        expected_reward_usdc=0.5,
        pending_reward_usdc=0.5,
        timestamp=NOW,
    )

    assert event["fill_quantity"] == 5.0
    assert event["fill_qty"] == 5.0
    assert event["market_slug"] == "m"
    assert event["inventory_delta_shares"] == 5.0
    assert event["inventory_delta"] == 5.0
    assert event["cash_delta_usdc"] == -2.05
    assert event["fees_usdc"] == 0.01
    assert event["fee"] == 0.01
    assert event["realized_cash_pnl_usdc"] == -0.01
    assert event["reward_separation"]["expected_reward_is_forecast_only"] is True
    assert event["reward_separation"]["pending_reward_counted_as_confirmed_reward"] is False
    assert event["reward_separation"]["reward_count_is_actual_earned_reward"] is False
    assert event["estimated_reward_counted_as_realized_cash_pnl"] is False
    assert event["can_submit_order"] is False


def test_confirmed_reward_requires_source_before_entering_realized_cash_pnl() -> None:
    without_source = build_fill_lifecycle_event(
        lifecycle_id="no-source",
        event_type="FULL_FILL",
        market="m",
        side="BID_ONLY",
        order_id="0xabc",
        price=0.41,
        size=10,
        fill_quantity=10,
        inventory_delta_shares=10,
        cash_delta_usdc=-4.1,
        fees_usdc=0.01,
        realized_spread_pnl_usdc=0.2,
        confirmed_reward_usdc=0.4,
        timestamp=NOW,
    )
    with_source = build_fill_lifecycle_event(
        lifecycle_id="with-source",
        event_type="FULL_FILL",
        market="m",
        side="BID_ONLY",
        order_id="0xabc",
        price=0.41,
        size=10,
        fill_quantity=10,
        inventory_delta_shares=10,
        cash_delta_usdc=-4.1,
        fees_usdc=0.01,
        realized_spread_pnl_usdc=0.2,
        confirmed_reward_usdc=0.4,
        confirmed_reward_source="API_SETTLEMENT",
        timestamp=NOW,
    )

    assert without_source["realized_cash_pnl_usdc"] == 0.19
    assert with_source["realized_cash_pnl_usdc"] == 0.59


def test_summary_reconstructs_minimal_lifecycle_totals() -> None:
    events = [
        build_fill_lifecycle_event(lifecycle_id="zero", event_type="ZERO_FILL", market="m", side="BID_ONLY", order_id="0x1", price=0.38, size=5, fill_quantity=0, inventory_delta_shares=0, cash_delta_usdc=0, fees_usdc=0.0, realized_spread_pnl_usdc=0),
        build_fill_lifecycle_event(lifecycle_id="partial", event_type="PARTIAL_FILL", market="m", side="BID_ONLY", order_id="0x2", price=0.39, size=5, fill_quantity=2, inventory_delta_shares=2, cash_delta_usdc=-0.78, fees_usdc=0.01, realized_spread_pnl_usdc=0),
        build_fill_lifecycle_event(lifecycle_id="full", event_type="FULL_FILL", market="m", side="BID_ONLY", order_id="0x3", price=0.40, size=5, fill_quantity=5, inventory_delta_shares=5, cash_delta_usdc=-2.0, fees_usdc=0.01, realized_spread_pnl_usdc=0.1),
    ]
    summary = summarize_fill_lifecycle_events(events, now=NOW)

    assert summary["event_count"] == 3
    assert summary["fill_quantity"] == 7.0
    assert summary["inventory_delta_shares"] == 7.0
    assert summary["realized_cash_pnl_usdc"] == 0.08
    assert summary["estimated_reward_counted_as_realized_cash_pnl"] is False


def test_reader_tolerates_partial_jsonl_writes(tmp_path) -> None:
    log = tmp_path / "fill.jsonl"
    append_fill_lifecycle_event(
        log,
        build_fill_lifecycle_event(lifecycle_id="ok", event_type="ZERO_FILL", market="m", side="BID_ONLY", order_id="0x1", price=0.38, size=5, fill_quantity=0, inventory_delta_shares=0, cash_delta_usdc=0, fees_usdc=0, realized_spread_pnl_usdc=0),
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{partial\n")
        handle.write(json.dumps(["bad"]) + "\n")

    events, stats = read_fill_lifecycle_events(log)

    assert len(events) == 1
    assert stats["malformed_row_count"] == 1
    assert stats["non_object_row_count"] == 1


def test_cli_writes_fill_lifecycle_summary(tmp_path) -> None:
    log = tmp_path / "fill.jsonl"
    out = tmp_path / "summary.json"
    append_fill_lifecycle_event(
        log,
        build_fill_lifecycle_event(lifecycle_id="ok", event_type="ZERO_FILL", market="m", side="BID_ONLY", order_id="0x1", price=0.38, size=5, fill_quantity=0, inventory_delta_shares=0, cash_delta_usdc=0, fees_usdc=0, realized_spread_pnl_usdc=0),
    )

    rc = main(["--log", str(log), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == "FILL_LIFECYCLE_SUMMARY_READY"
    assert payload["event_count"] == 1
