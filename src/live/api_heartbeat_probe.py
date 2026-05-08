from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import socket
import time


REPORT_SCHEMA_VERSION = "api_heartbeat_probe.v1"
REPORT_TYPE = "api_heartbeat_probe"

DEFAULT_HEARTBEAT_URL = "https://clob.polymarket.com/ok"
DEFAULT_TIMEOUT_SEC = 2.0
DEFAULT_HEALTHY_LATENCY_MS = 200.0
DEFAULT_CRITICAL_LATENCY_MS = 500.0
DEFAULT_MASS_CANCEL_FAILURE_THRESHOLD = 3

HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
CRITICAL_LATENCY = "CRITICAL_LATENCY"
DISCONNECTED = "DISCONNECTED"


@dataclass(frozen=True)
class HeartbeatProbeResult:
    url: str
    latency_ms: float | None
    status_code: int | None
    response_ok: bool
    error_code: str | None = None
    error_message: str | None = None


def run_http_heartbeat_probe(
    *,
    url: str = DEFAULT_HEARTBEAT_URL,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> HeartbeatProbeResult:
    """Measure a real HTTP round trip to the configured heartbeat endpoint."""

    started = time.perf_counter()
    request = Request(url, headers={"User-Agent": "polyarb-lab-api-heartbeat/1.0"})
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - explicit operator supplied URL
            response.read(64)
            latency_ms = (time.perf_counter() - started) * 1000.0
            status_code = int(response.getcode())
            return HeartbeatProbeResult(
                url=url,
                latency_ms=latency_ms,
                status_code=status_code,
                response_ok=200 <= status_code < 400,
            )
    except HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HeartbeatProbeResult(
            url=url,
            latency_ms=latency_ms,
            status_code=int(exc.code),
            response_ok=False,
            error_code=f"HTTP_{exc.code}",
            error_message=str(exc.reason),
        )
    except (TimeoutError, socket.timeout) as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HeartbeatProbeResult(
            url=url,
            latency_ms=latency_ms,
            status_code=None,
            response_ok=False,
            error_code="TIMEOUT",
            error_message=str(exc),
        )
    except (URLError, OSError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HeartbeatProbeResult(
            url=url,
            latency_ms=latency_ms,
            status_code=None,
            response_ok=False,
            error_code=exc.__class__.__name__,
            error_message=str(exc),
        )


def build_api_heartbeat_report(
    *,
    probe: HeartbeatProbeResult,
    previous_report: dict[str, Any] | None = None,
    now: datetime | None = None,
    healthy_latency_ms: float = DEFAULT_HEALTHY_LATENCY_MS,
    critical_latency_ms: float = DEFAULT_CRITICAL_LATENCY_MS,
    mass_cancel_failure_threshold: int = DEFAULT_MASS_CANCEL_FAILURE_THRESHOLD,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    previous_report = previous_report or {}
    api_health_status = classify_api_health(
        probe=probe,
        healthy_latency_ms=healthy_latency_ms,
        critical_latency_ms=critical_latency_ms,
    )
    is_within_safety_threshold = api_health_status in {HEALTHY, DEGRADED}
    consecutive_failures = _consecutive_failures(
        previous_report=previous_report,
        current_ok=is_within_safety_threshold,
    )
    http_425_window_active = probe.status_code == 425 or probe.error_code == "HTTP_425"
    mass_cancel_mode_recommended = consecutive_failures >= mass_cancel_failure_threshold
    blockers = _blockers(
        api_health_status=api_health_status,
        http_425_window_active=http_425_window_active,
        mass_cancel_mode_recommended=mass_cancel_mode_recommended,
    )
    generated_at = now.isoformat()
    latency_ms = _round(probe.latency_ms)

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "last_ping_time": generated_at,
        "read_only": True,
        "live_actions_enabled": False,
        "execution_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "heartbeat_url": probe.url,
        "timeout_sec": timeout_sec,
        "status": _report_status(api_health_status, blockers),
        "api_health_status": api_health_status,
        "status_code": probe.status_code,
        "latency_ms": latency_ms,
        "api_latency_ms": latency_ms,
        "cancel_latency_ms": latency_ms,
        "healthy_latency_threshold_ms": _round(healthy_latency_ms),
        "critical_latency_threshold_ms": _round(critical_latency_ms),
        "max_cancel_latency_ms": _round(critical_latency_ms),
        "is_within_safety_threshold": is_within_safety_threshold,
        "heartbeat_ok": is_within_safety_threshold,
        "latency_ok": is_within_safety_threshold,
        "degraded_warning": api_health_status == DEGRADED,
        "critical_latency": api_health_status == CRITICAL_LATENCY,
        "disconnected": api_health_status == DISCONNECTED,
        "http_425_window_active": http_425_window_active,
        "consecutive_failure_count": consecutive_failures,
        "mass_cancel_failure_threshold": mass_cancel_failure_threshold,
        "mass_cancel_mode_recommended": mass_cancel_mode_recommended,
        "mass_cancel_ready": not mass_cancel_mode_recommended,
        "market_suspended": False,
        "market_suspended_source": "NOT_CHECKED_BY_HEARTBEAT_PROBE",
        "high_velocity_toxic_flow": False,
        "error_code": _error_code(probe=probe, http_425_window_active=http_425_window_active),
        "error_message": probe.error_message,
        "blockers": blockers,
        "should_block_live_trading": bool(blockers),
        "one_line_verdict": _one_line_verdict(api_health_status, latency_ms, blockers),
    }


def classify_api_health(
    *,
    probe: HeartbeatProbeResult,
    healthy_latency_ms: float = DEFAULT_HEALTHY_LATENCY_MS,
    critical_latency_ms: float = DEFAULT_CRITICAL_LATENCY_MS,
) -> str:
    if not probe.response_ok or probe.latency_ms is None:
        return DISCONNECTED
    if probe.latency_ms < healthy_latency_ms:
        return HEALTHY
    if probe.latency_ms < critical_latency_ms:
        return DEGRADED
    return CRITICAL_LATENCY


def _consecutive_failures(*, previous_report: dict[str, Any], current_ok: bool) -> int:
    if current_ok:
        return 0
    previous = previous_report.get("consecutive_failure_count")
    try:
        return int(previous) + 1
    except (TypeError, ValueError):
        return 1


def _blockers(
    *,
    api_health_status: str,
    http_425_window_active: bool,
    mass_cancel_mode_recommended: bool,
) -> list[str]:
    blockers: list[str] = []
    if api_health_status == CRITICAL_LATENCY:
        blockers.append("LATENCY_TOO_HIGH_FOR_LIVE_TRADING")
    elif api_health_status == DISCONNECTED:
        blockers.append("API_HEARTBEAT_DISCONNECTED")
    if http_425_window_active:
        blockers.append("HTTP_425_ENGINE_RESTART_WINDOW")
    if mass_cancel_mode_recommended:
        blockers.append("MASS_CANCEL_MODE_REQUIRED")
    return blockers


def _report_status(api_health_status: str, blockers: list[str]) -> str:
    if blockers:
        return "API_HEARTBEAT_BLOCKED"
    if api_health_status == DEGRADED:
        return "API_HEARTBEAT_DEGRADED"
    return "API_HEARTBEAT_READY"


def _error_code(*, probe: HeartbeatProbeResult, http_425_window_active: bool) -> str | None:
    if http_425_window_active:
        return "HTTP_425_ENGINE_RESTART_WINDOW"
    return probe.error_code


def _one_line_verdict(api_health_status: str, latency_ms: float | None, blockers: list[str]) -> str:
    latency_text = "unknown" if latency_ms is None else f"{latency_ms}ms"
    if not blockers:
        return f"API_HEARTBEAT_{api_health_status}: latency={latency_text}; can_submit_order=false."
    return f"API_HEARTBEAT_BLOCKED: latency={latency_text}; {', '.join(blockers)}; can_submit_order=false."


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
