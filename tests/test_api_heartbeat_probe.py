from __future__ import annotations

from datetime import datetime, timezone

from src.live.api_heartbeat_probe import (
    CRITICAL_LATENCY,
    DEGRADED,
    DISCONNECTED,
    HEALTHY,
    HeartbeatProbeResult,
    build_api_heartbeat_report,
    classify_api_health,
)


NOW = datetime(2026, 5, 8, 2, 0, tzinfo=timezone.utc)


def _probe(*, latency_ms: float | None, ok: bool = True, status_code: int | None = 200, error_code: str | None = None):
    return HeartbeatProbeResult(
        url="https://clob.polymarket.com/ok",
        latency_ms=latency_ms,
        status_code=status_code,
        response_ok=ok,
        error_code=error_code,
        error_message=None,
    )


def test_classifies_latency_status_matrix() -> None:
    assert classify_api_health(probe=_probe(latency_ms=55.0)) == HEALTHY
    assert classify_api_health(probe=_probe(latency_ms=250.0)) == DEGRADED
    assert classify_api_health(probe=_probe(latency_ms=500.0)) == CRITICAL_LATENCY
    assert classify_api_health(probe=_probe(latency_ms=None, ok=False, status_code=None)) == DISCONNECTED


def test_healthy_probe_outputs_gate_network_fields_without_enabling_execution() -> None:
    report = build_api_heartbeat_report(probe=_probe(latency_ms=80.0), now=NOW)

    assert report["api_health_status"] == HEALTHY
    assert report["status"] == "API_HEARTBEAT_READY"
    assert report["is_within_safety_threshold"] is True
    assert report["heartbeat_ok"] is True
    assert report["latency_ok"] is True
    assert report["mass_cancel_ready"] is True
    assert report["http_425_window_active"] is False
    assert report["cancel_latency_ms"] == 80.0
    assert report["max_cancel_latency_ms"] == 500.0
    assert report["can_submit_order"] is False
    assert report["live_order_sent"] is False


def test_degraded_latency_warns_but_stays_inside_safety_threshold() -> None:
    report = build_api_heartbeat_report(probe=_probe(latency_ms=250.0), now=NOW)

    assert report["api_health_status"] == DEGRADED
    assert report["status"] == "API_HEARTBEAT_DEGRADED"
    assert report["degraded_warning"] is True
    assert report["is_within_safety_threshold"] is True
    assert report["heartbeat_ok"] is True
    assert report["blockers"] == []


def test_critical_latency_blocks_live_trading_and_gate_cancel_latency() -> None:
    report = build_api_heartbeat_report(probe=_probe(latency_ms=650.0), now=NOW)

    assert report["api_health_status"] == CRITICAL_LATENCY
    assert report["is_within_safety_threshold"] is False
    assert report["heartbeat_ok"] is False
    assert report["latency_ok"] is False
    assert report["cancel_latency_ms"] == 650.0
    assert "LATENCY_TOO_HIGH_FOR_LIVE_TRADING" in report["blockers"]


def test_disconnected_probe_increments_failures_and_triggers_mass_cancel_mode_after_three() -> None:
    previous = {"consecutive_failure_count": 2}
    report = build_api_heartbeat_report(
        probe=_probe(latency_ms=None, ok=False, status_code=None, error_code="TIMEOUT"),
        previous_report=previous,
        now=NOW,
    )

    assert report["api_health_status"] == DISCONNECTED
    assert report["consecutive_failure_count"] == 3
    assert report["mass_cancel_mode_recommended"] is True
    assert report["mass_cancel_ready"] is False
    assert "API_HEARTBEAT_DISCONNECTED" in report["blockers"]
    assert "MASS_CANCEL_MODE_REQUIRED" in report["blockers"]


def test_http_425_sets_restart_window_blocker() -> None:
    report = build_api_heartbeat_report(
        probe=_probe(latency_ms=100.0, ok=False, status_code=425, error_code="HTTP_425"),
        now=NOW,
    )

    assert report["api_health_status"] == DISCONNECTED
    assert report["http_425_window_active"] is True
    assert report["error_code"] == "HTTP_425_ENGINE_RESTART_WINDOW"
    assert "HTTP_425_ENGINE_RESTART_WINDOW" in report["blockers"]
