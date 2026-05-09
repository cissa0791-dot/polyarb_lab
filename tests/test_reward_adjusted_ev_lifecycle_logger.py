from __future__ import annotations

import json
from datetime import datetime, timezone

from scripts.analyze_reward_adjusted_ev_lifecycle_log import main
from src.live.reward_adjusted_ev_lifecycle_logger import (
    append_lifecycle_event,
    build_lifecycle_event,
    read_lifecycle_events,
    summarize_lifecycle_events,
)


NOW = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)


def test_estimated_reward_never_enters_realized_cash_pnl() -> None:
    event = build_lifecycle_event(
        lifecycle_id="zero",
        event_type="QUOTE_CLOSED",
        market="m",
        side="BID_ONLY",
        price=0.38,
        size=50,
        expected_reward_usdc=1.25,
        pending_reward_usdc=1.25,
        realized_spread_pnl_usdc=0.0,
        fees_usdc=0.01,
        fill_result="ZERO_FILL",
        timestamp=NOW,
    )

    assert event["reward_accounting"]["expected_reward_is_forecast_only"] is True
    assert event["reward_accounting"]["pending_reward_counted_as_confirmed_reward"] is False
    assert event["pnl_accounting"]["realized_cash_pnl_usdc"] == -0.01
    assert event["pnl_accounting"]["estimated_reward_counted_as_realized_cash_pnl"] is False


def test_confirmed_reward_requires_explicit_source() -> None:
    event = build_lifecycle_event(
        lifecycle_id="confirmed",
        event_type="QUOTE_CLOSED",
        market="m",
        side="BID_ONLY",
        price=0.38,
        size=50,
        confirmed_reward_usdc=0.4,
        confirmed_reward_source="API_SETTLEMENT",
        realized_spread_pnl_usdc=0.2,
        fees_usdc=0.01,
        fill_result="FULL_FILL_RECONCILED",
        timestamp=NOW,
    )

    assert event["reward_accounting"]["confirmed_reward_counted_in_realized_cash_pnl"] is True
    assert event["pnl_accounting"]["realized_cash_pnl_usdc"] == 0.59


def test_summary_reconstructs_zero_partial_and_full_paths() -> None:
    events = [
        build_lifecycle_event(lifecycle_id="z", event_type="QUOTE_CLOSED", market="m", side="BID_ONLY", price=0.38, size=5, fill_result="ZERO_FILL"),
        build_lifecycle_event(lifecycle_id="p", event_type="QUOTE_CLOSED", market="m", side="BID_ONLY", price=0.39, size=5, fill_result="PARTIAL_FILL_RECONCILED"),
        build_lifecycle_event(lifecycle_id="f", event_type="QUOTE_CLOSED", market="m", side="BID_ONLY", price=0.40, size=5, fill_result="FULL_FILL_RECONCILED"),
    ]
    summary = summarize_lifecycle_events(events, now=NOW)

    assert summary["event_count"] == 3
    assert summary["lifecycle_count"] == 3
    assert summary["fill_result_counts"]["ZERO_FILL"] == 1
    assert summary["fill_result_counts"]["PARTIAL_FILL_RECONCILED"] == 1
    assert summary["fill_result_counts"]["FULL_FILL_RECONCILED"] == 1
    assert summary["cash_accounting"]["estimated_reward_counted_as_realized_cash_pnl"] is False


def test_reader_tolerates_malformed_and_non_object_rows(tmp_path) -> None:
    log = tmp_path / "events.jsonl"
    append_lifecycle_event(
        log,
        build_lifecycle_event(lifecycle_id="ok", event_type="QUOTE_CLOSED", market="m", side="BID_ONLY", price=0.38, size=5),
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{bad json\n")
        handle.write(json.dumps(["not", "object"]) + "\n")

    events, stats = read_lifecycle_events(log)

    assert len(events) == 1
    assert stats["malformed_row_count"] == 1
    assert stats["non_object_row_count"] == 1


def test_cli_analyzer_writes_summary(tmp_path) -> None:
    log = tmp_path / "events.jsonl"
    out = tmp_path / "summary.json"
    append_lifecycle_event(
        log,
        build_lifecycle_event(lifecycle_id="ok", event_type="QUOTE_CLOSED", market="m", side="BID_ONLY", price=0.38, size=5),
    )

    rc = main(["--log", str(log), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["status"] == "REWARD_ADJUSTED_EV_LIFECYCLE_SUMMARY_READY"
    assert payload["event_count"] == 1
