from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "full_market_probe_planner.v1"
REPORT_TYPE = "full_market_probe_planner"

READY_STATUS = "FULL_MARKET_PROBE_PLAN_READY"
NO_CANDIDATE_STATUS = "FULL_MARKET_PROBE_PLAN_NO_SAFE_CANDIDATE"
BLOCKED_STATUS = "FULL_MARKET_PROBE_PLAN_BLOCKED"


def build_full_market_probe_plan(
    *,
    candidates: list[dict[str, Any]] | None = None,
    max_live_risk_usdc: float,
    requested_probe_type: str = "B_LONG_OBSERVATION_STABILITY",
    stability_max_fill_probability: float = 0.30,
    scan_scope: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Rank multiple read-only probe candidates and select the cleanest one."""

    now = now or datetime.now(timezone.utc)
    scan_scope = scan_scope or {"declared_universe": "provided_candidates", "universe_complete": False}
    rows = candidates or []
    ranked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in rows:
        rejection_reasons = _rejection_reasons(
            candidate,
            max_live_risk_usdc=max_live_risk_usdc,
            requested_probe_type=requested_probe_type,
            stability_max_fill_probability=stability_max_fill_probability,
        )
        enriched = _enrich_candidate(candidate, rejection_reasons)
        if rejection_reasons:
            rejected.append(enriched)
        else:
            ranked.append(enriched)
    ranked.sort(key=lambda row: (row["score"], row.get("market_slug") or ""), reverse=True)
    selected = ranked[0] if ranked else None
    blockers: list[str] = []
    if not rows:
        blockers.append("NO_CANDIDATES_PROVIDED")
    status = BLOCKED_STATUS if blockers else READY_STATUS if selected else NO_CANDIDATE_STATUS
    planner_hash = _hash(
        {
            "selected_candidate": selected,
            "requested_probe_type": requested_probe_type,
            "max_live_risk_usdc": max_live_risk_usdc,
            "stability_max_fill_probability": stability_max_fill_probability,
            "scan_scope": scan_scope,
        }
    )
    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "requested_probe_type": requested_probe_type,
        "scan_scope": scan_scope,
        "is_global_optimum": scan_scope.get("universe_complete") is True,
        "ranked_candidates": ranked,
        "selected_candidate": selected,
        "rejected_candidates": rejected,
        "rejection_reasons": sorted({reason for row in rejected for reason in row.get("rejection_reasons", [])}),
        "planner_hash": planner_hash,
        "token_binding_required": selected is not None,
        "requires_new_token": selected is not None,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "blockers": blockers,
        "one_line_verdict": _one_line_verdict(status, selected, blockers),
    }


def _rejection_reasons(
    candidate: dict[str, Any],
    *,
    max_live_risk_usdc: float,
    requested_probe_type: str,
    stability_max_fill_probability: float,
) -> list[str]:
    reasons: list[str] = []
    if candidate.get("toxic_flow_status") not in {"TOXIC_FLOW_READY", "SAFE", None} or candidate.get("toxic_unsafe") is True:
        reasons.append("TOXIC_FLOW_UNSAFE")
    fill_probability = _first_float(candidate.get("fill_probability"), candidate.get("p_fill_300s"))
    if requested_probe_type == "B_LONG_OBSERVATION_STABILITY" and fill_probability is not None and fill_probability > stability_max_fill_probability:
        reasons.append("FILL_PROBABILITY_TOO_HIGH_FOR_B")
    if candidate.get("reward_min_size_check") is not True:
        reasons.append("REWARD_MIN_SIZE_FAIL")
    if candidate.get("fee_check") is False or candidate.get("fee_status") == "FEE_BLOCKER":
        reasons.append("FEE_ECONOMICS_NEGATIVE")
    if candidate.get("tick_size_check") is False:
        reasons.append("TICK_SIZE_INVALID")
    capital_required = _first_float(candidate.get("capital_required_usdc"))
    if capital_required is None:
        reasons.append("CAPITAL_REQUIRED_MISSING")
    elif capital_required > max_live_risk_usdc:
        reasons.append("CAPITAL_OVER_BUDGET")
    if not candidate.get("market_slug"):
        reasons.append("MARKET_SLUG_MISSING")
    return reasons


def _enrich_candidate(candidate: dict[str, Any], rejection_reasons: list[str]) -> dict[str, Any]:
    row = dict(candidate)
    fill_probability = _first_float(candidate.get("fill_probability"), candidate.get("p_fill_300s")) or 0.0
    capital_required = _first_float(candidate.get("capital_required_usdc")) or 0.0
    expected_reward = _first_float(candidate.get("expected_reward_usdc")) or 0.0
    toxic_score = _first_float(candidate.get("toxic_flow_score")) or 0.0
    score = expected_reward - (fill_probability * 0.5) - (toxic_score * 0.5) - (capital_required * 0.001)
    row["score"] = round(score, 6)
    row["rejection_reasons"] = rejection_reasons
    row["selected"] = False
    return row


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(parsed):
            return parsed
    return None


def _hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _one_line_verdict(status: str, selected: dict[str, Any] | None, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return f"FULL_MARKET_PROBE_PLAN_READY: selected {selected.get('market_slug')} without authorizing execution."
    if status == NO_CANDIDATE_STATUS:
        return "FULL_MARKET_PROBE_PLAN_NO_SAFE_CANDIDATE: all candidates were rejected; no token or execution authorized."
    return f"FULL_MARKET_PROBE_PLAN_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
