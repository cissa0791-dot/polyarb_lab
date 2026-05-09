from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


REPORT_SCHEMA_VERSION = "report_lineage_guard.v1"
REPORT_TYPE = "report_lineage_guard"

READY_STATUS = "POST_PROBE_REPORT_LINEAGE_READY"
BLOCKED_STATUS = "POST_PROBE_REPORT_LINEAGE_BLOCKED"


def build_report_lineage_guard(
    *,
    probe: dict[str, Any] | None = None,
    order_reconciliation: dict[str, Any] | None = None,
    market_microstructure: dict[str, Any] | None = None,
    planner: dict[str, Any] | None = None,
    fee_reconciliation: dict[str, Any] | None = None,
    toxic_flow: dict[str, Any] | None = None,
    reward_report: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    deposit_wallet: dict[str, Any] | None = None,
    heartbeat: dict[str, Any] | None = None,
    require_reward_report: bool = False,
    max_report_age_minutes: float = 10.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate that latest reports describe the same post-probe evidence chain."""

    now = now or datetime.now(timezone.utc)
    probe = probe or {}
    order_reconciliation = order_reconciliation or {}
    market_microstructure = market_microstructure or {}
    planner = planner or {}
    fee_reconciliation = fee_reconciliation or {}
    toxic_flow = toxic_flow or {}
    reward_report = reward_report or {}
    inventory_state = inventory_state or {}
    order_mutex = order_mutex or {}
    gate = gate or {}
    deposit_wallet = deposit_wallet or {}
    heartbeat = heartbeat or {}

    submit = probe.get("submit_result") if isinstance(probe.get("submit_result"), dict) else {}
    cancel = probe.get("cancel_result") if isinstance(probe.get("cancel_result"), dict) else {}
    order_id = submit.get("order_id") or cancel.get("order_id") or probe.get("order_id")
    reconciliation_order_id = order_reconciliation.get("order_id")
    probe_market = _market_slug_from_probe(probe)
    planner_market = _market_slug_from_planner(planner)
    expected_market = probe_market or planner_market
    market_sources = {
        "probe": probe_market,
        "market_microstructure": market_microstructure.get("market_slug"),
        "planner": planner_market,
        "fee_reconciliation": fee_reconciliation.get("market_slug"),
        "toxic_flow": toxic_flow.get("market_slug"),
        "gate": gate.get("target_market_slug"),
    }
    if reward_report:
        market_sources["reward_report"] = reward_report.get("market_slug")

    report_inputs = {
        "probe": probe,
        "order_reconciliation": order_reconciliation,
        "market_microstructure": market_microstructure,
        "planner": planner,
        "fee_reconciliation": fee_reconciliation,
        "toxic_flow": toxic_flow,
        "inventory_state": inventory_state,
        "order_mutex": order_mutex,
        "gate": gate,
        "deposit_wallet": deposit_wallet,
        "heartbeat": heartbeat,
    }
    if require_reward_report or reward_report:
        report_inputs["reward_report"] = reward_report

    source_timestamps = {name: _timestamp_text(payload) for name, payload in report_inputs.items()}
    completed_at = _probe_completed_at(probe)
    blockers: list[str] = []
    if not order_id:
        blockers.append("LINEAGE_PROBE_ORDER_ID_MISSING")
    if not reconciliation_order_id:
        blockers.append("LINEAGE_RECONCILIATION_ORDER_ID_MISSING")
    elif str(reconciliation_order_id) != str(order_id):
        blockers.append("LINEAGE_ORDER_ID_MISMATCH")
    if not expected_market:
        blockers.append("LINEAGE_MARKET_MISSING")
    for source, market in market_sources.items():
        if not market:
            blockers.append(f"LINEAGE_{source.upper()}_MARKET_MISSING")
        elif expected_market and str(market) != str(expected_market):
            blockers.append(f"LINEAGE_{source.upper()}_MARKET_MISMATCH")

    max_age = timedelta(minutes=max(0.0, float(max_report_age_minutes)))
    for source, payload in report_inputs.items():
        if not payload:
            blockers.append(f"LINEAGE_{source.upper()}_REPORT_MISSING")
            continue
        timestamp = _timestamp(payload)
        if timestamp is None:
            blockers.append(f"LINEAGE_{source.upper()}_TIMESTAMP_MISSING")
            continue
        if completed_at is not None and source != "probe" and timestamp < completed_at:
            blockers.append(f"LINEAGE_{source.upper()}_STALE_BEFORE_PROBE_COMPLETION")
        if now - timestamp > max_age:
            blockers.append(f"LINEAGE_{source.upper()}_REPORT_STALE")

    blockers = _unique(blockers)
    matched_order_id = str(order_id) if order_id and str(order_id) == str(reconciliation_order_id or "") else None
    matched_market_slug = expected_market if expected_market and all(
        not market or str(market) == str(expected_market) for market in market_sources.values()
    ) else None
    status = READY_STATUS if not blockers else BLOCKED_STATUS
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "lineage_status": status,
        "blockers": blockers,
        "lineage_blockers": blockers,
        "matched_order_id": matched_order_id,
        "matched_market_slug": matched_market_slug,
        "expected_market_slug": expected_market,
        "market_sources": market_sources,
        "probe_completed_at_utc": None if completed_at is None else completed_at.isoformat(),
        "max_report_age_minutes": max_report_age_minutes,
        "source_report_timestamps": source_timestamps,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
    }


def _market_slug_from_probe(probe: dict[str, Any]) -> str | None:
    target = probe.get("target") if isinstance(probe.get("target"), dict) else {}
    return _text(target.get("market_slug") or probe.get("market_slug") or probe.get("target_market_slug"))


def _market_slug_from_planner(planner: dict[str, Any]) -> str | None:
    plan = planner.get("recommended_plan") if isinstance(planner.get("recommended_plan"), dict) else {}
    selected = planner.get("selected_candidate") if isinstance(planner.get("selected_candidate"), dict) else {}
    return _text(
        plan.get("market_slug")
        or selected.get("market_slug")
        or planner.get("target_market_slug")
        or planner.get("market_slug")
    )


def _probe_completed_at(probe: dict[str, Any]) -> datetime | None:
    cancel = probe.get("cancel_result") if isinstance(probe.get("cancel_result"), dict) else {}
    hold = probe.get("hold_observation") if isinstance(probe.get("hold_observation"), dict) else {}
    for value in (
        cancel.get("cancel_response_at_utc"),
        hold.get("completed_at_utc"),
        probe.get("generated_at_utc"),
        probe.get("writer_generated_at_utc"),
    ):
        parsed = _timestamp_from_value(value)
        if parsed is not None:
            return parsed
    return None


def _timestamp(payload: dict[str, Any]) -> datetime | None:
    for key in (
        "generated_at_utc",
        "planner_snapshot_ts",
        "updated_at_utc",
        "fetched_at_utc",
        "timestamp_utc",
        "writer_generated_at_utc",
    ):
        parsed = _timestamp_from_value(payload.get(key))
        if parsed is not None:
            return parsed
    writer = payload.get("writer_metadata")
    if isinstance(writer, dict):
        return _timestamp_from_value(writer.get("generated_at_utc"))
    return None


def _timestamp_text(payload: dict[str, Any]) -> str | None:
    parsed = _timestamp(payload)
    return None if parsed is None else parsed.isoformat()


def _timestamp_from_value(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _text(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    return text or None


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out
