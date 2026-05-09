from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "b_low_fill_stability_probe_approval_package.v1"
REPORT_TYPE = "b_low_fill_stability_probe_approval_package"

READY_STATUS = "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_READY"
BLOCKED_STATUS = "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_BLOCKED"

PROBE_TYPE = "B_LONG_OBSERVATION_STABILITY"
APPROVAL_PACKAGE_ID = "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE"


def build_b_low_fill_stability_probe_approval_package(
    *,
    gate: dict[str, Any] | None,
    planner: dict[str, Any] | None,
    candidate_search: dict[str, Any] | None,
    second_probe_decision: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a governance-only package for the low-fill B stability probe.

    This package freezes the current planner evidence for manual review. It
    does not create a token, authorize execution, submit orders, cancel orders,
    or claim profitability.
    """

    now = now or datetime.now(timezone.utc)
    gate = gate or {}
    planner = planner or {}
    candidate_search = candidate_search or {}
    second_probe_decision = second_probe_decision or {}

    recommended = planner.get("recommended_plan") if isinstance(planner.get("recommended_plan"), dict) else {}
    best_candidate = (
        candidate_search.get("best_candidate") if isinstance(candidate_search.get("best_candidate"), dict) else {}
    )
    planner_hash = str(planner.get("planner_hash") or "").strip()
    quote_price = _first_float(recommended.get("quote_price"), recommended.get("quote_bid"))
    quote_size = _first_float(recommended.get("quote_size"))
    fill_probability = _first_float(recommended.get("fill_probability"))
    stability_max_fill_probability = _first_float(planner.get("stability_max_fill_probability"), recommended.get("stability_max_fill_probability"))
    max_live_risk_usdc = _first_float(planner.get("max_live_risk_usdc"), gate.get("max_live_risk_usdc"))
    hold_seconds = _first_int(planner.get("hold_seconds"))
    market_slug = recommended.get("market_slug") or planner.get("target_market_slug") or gate.get("target_market_slug")

    blockers = _blockers(
        gate=gate,
        planner=planner,
        candidate_search=candidate_search,
        second_probe_decision=second_probe_decision,
        recommended=recommended,
        best_candidate=best_candidate,
        planner_hash=planner_hash,
        quote_price=quote_price,
        quote_size=quote_size,
        fill_probability=fill_probability,
        stability_max_fill_probability=stability_max_fill_probability,
        hold_seconds=hold_seconds,
    )
    status = READY_STATUS if not blockers else BLOCKED_STATUS

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "approval_package_id": APPROVAL_PACKAGE_ID,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "status": status,
        "probe_type": PROBE_TYPE,
        "market_slug": market_slug,
        "selected_side": recommended.get("selected_side") or planner.get("selected_side") or "BID_ONLY",
        "quote_price": _round(quote_price),
        "quote_bid": _round(recommended.get("quote_bid")),
        "quote_ask": _round(recommended.get("quote_ask")),
        "quote_size": _round(quote_size),
        "fill_probability_proxy": _round(fill_probability),
        "fill_probability_is_model_estimate": True,
        "stability_max_fill_probability": _round(stability_max_fill_probability),
        "planner_hash": planner_hash or None,
        "hold_seconds": hold_seconds,
        "max_live_risk_usdc": _round(max_live_risk_usdc),
        "capital_required_usdc": _round(recommended.get("capital_required_usdc")),
        "expected_max_loss_if_filled": _round(recommended.get("expected_max_loss_if_filled")),
        "reward_scoring": {
            "reward_min_size": recommended.get("reward_min_size"),
            "reward_max_spread_cents": recommended.get("reward_max_spread_cents"),
            "reward_min_size_check": _nested_bool(recommended, "checks", "reward_min_size_check"),
            "reward_scoring_assert_preserved": True,
            "estimated_reward_counted_as_realized_pnl": False,
        },
        "planner_evidence": {
            "planner_status": planner.get("status"),
            "plan_classification": planner.get("plan_classification") or recommended.get("plan_classification"),
            "requested_probe_intent": planner.get("requested_probe_intent"),
            "token_binding_required": planner.get("token_binding_required") is True,
            "requires_new_token": planner.get("requires_new_token") is True,
            "requires_new_approval": planner.get("requires_new_approval") is True,
            "can_submit_order": planner.get("can_submit_order"),
            "live_order_sent": planner.get("live_order_sent"),
            "execution_authorized": planner.get("execution_authorized"),
            "recommended_plan_present": bool(recommended),
        },
        "candidate_search_evidence": {
            "search_status": candidate_search.get("status"),
            "safe_candidate_count": candidate_search.get("safe_candidate_count"),
            "best_candidate_matches_planner": _candidate_matches_plan(best_candidate, recommended),
            "can_submit_order": candidate_search.get("can_submit_order"),
            "live_order_sent": candidate_search.get("live_order_sent"),
        },
        "prelive_evidence": {
            "gate_status": gate.get("status"),
            "asserts_passed": gate.get("asserts_passed"),
            "asserts_failed": gate.get("asserts_failed"),
            "blockers": gate.get("blockers") if isinstance(gate.get("blockers"), list) else [],
            "can_submit_order": gate.get("can_submit_order"),
            "live_order_sent": gate.get("live_order_sent"),
        },
        "prior_decision_evidence": {
            "decision_status": second_probe_decision.get("status"),
            "recommended_option": second_probe_decision.get("recommended_option"),
            "present": bool(second_probe_decision),
        },
        "token_binding_required": True,
        "token_binding_fields": {
            "market_slug": market_slug,
            "selected_side": recommended.get("selected_side") or planner.get("selected_side") or "BID_ONLY",
            "quote_price": _round(quote_price),
            "quote_size": _round(quote_size),
            "max_live_risk_usdc": _round(max_live_risk_usdc),
            "hold_seconds": hold_seconds,
            "planner_hash": planner_hash or None,
        },
        "approval_boundary": {
            "token_ready": False,
            "token_created": False,
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "requires_new_operator_approval_before_token": True,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "rule": "This package freezes evidence for review only. It does not create a token or authorize execution.",
        },
        "success_criteria": {
            "primary": [
                "A future separately authorized BID_ONLY order remains observable for the configured long hold window.",
                "Heartbeat, order mutex, and open-order state remain consistent during the hold window.",
                "Exact cancel-by-order-id completes after the hold window.",
                "Final audit proves open_order_count=0 and inventory is clear or explicitly reconciled.",
            ],
            "fill_required_for_success": False,
            "secondary_if_fill_occurs": [
                "Any fill or partial fill is treated as reconciliation evidence, not profitability proof.",
                "Inventory, cash, fee, and reward separation reports must reconcile before any next action.",
            ],
        },
        "abort_conditions": [
            "heartbeat enters critical latency or disconnected status",
            "toxic-flow/adverse-selection report blocks current market",
            "order is invisible beyond visibility grace period",
            "unexpected fill cannot be reconciled immediately",
            "cancel-by-order-id is not confirmed",
            "final open_order_count is not zero",
            "final inventory is not clear or explicitly reconciled",
        ],
        "profitability_claimed": False,
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
        "pending_reward_counted_as_confirmed_reward": False,
        "token_created": False,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# B Low-Fill Stability Probe Approval Package",
        "",
        "## Boundary",
        f"- Status: {report.get('status')}",
        f"- Probe type: {report.get('probe_type')}",
        f"- Token ready: {(report.get('approval_boundary') or {}).get('token_ready')}",
        f"- Execution authorized: {report.get('execution_authorized')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        "",
        "## Frozen Plan",
        f"- Market: {report.get('market_slug')}",
        f"- Side: {report.get('selected_side')}",
        f"- Quote price: {report.get('quote_price')}",
        f"- Quote size: {report.get('quote_size')}",
        f"- Fill probability proxy: {report.get('fill_probability_proxy')}",
        f"- Planner hash: {report.get('planner_hash')}",
        f"- Hold seconds: {report.get('hold_seconds')}",
        "",
        "## Evidence",
        f"- Gate status: {(report.get('prelive_evidence') or {}).get('gate_status')}",
        f"- Gate assertions: {(report.get('prelive_evidence') or {}).get('asserts_passed')}/12",
        f"- Planner status: {(report.get('planner_evidence') or {}).get('planner_status')}",
        f"- Plan classification: {(report.get('planner_evidence') or {}).get('plan_classification')}",
        f"- Search status: {(report.get('candidate_search_evidence') or {}).get('search_status')}",
        f"- Search/planner match: {(report.get('candidate_search_evidence') or {}).get('best_candidate_matches_planner')}",
        "",
        "## Token Binding",
    ]
    binding = report.get("token_binding_fields") if isinstance(report.get("token_binding_fields"), dict) else {}
    for key, value in binding.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _blockers(
    *,
    gate: dict[str, Any],
    planner: dict[str, Any],
    candidate_search: dict[str, Any],
    second_probe_decision: dict[str, Any],
    recommended: dict[str, Any],
    best_candidate: dict[str, Any],
    planner_hash: str,
    quote_price: float | None,
    quote_size: float | None,
    fill_probability: float | None,
    stability_max_fill_probability: float | None,
    hold_seconds: int | None,
) -> list[str]:
    blockers: list[str] = []
    if gate.get("status") != "LIVE_READY_APPROVED" or _first_int(gate.get("asserts_passed")) != 12 or _first_int(gate.get("asserts_failed")) != 0:
        blockers.append("PRELIVE_12_OF_12_NOT_READY")
    if gate.get("can_submit_order") is not False:
        blockers.append("GATE_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if gate.get("live_order_sent") is not False:
        blockers.append("GATE_LIVE_ORDER_SENT_TRUE_UNEXPECTED")
    if planner.get("status") != "LIVE_PROBE_PLAN_READY":
        blockers.append("PLANNER_NOT_READY")
    if not recommended:
        blockers.append("PLANNER_RECOMMENDED_PLAN_MISSING")
    if (planner.get("plan_classification") or recommended.get("plan_classification")) != PROBE_TYPE:
        blockers.append("PLANNER_CLASSIFICATION_NOT_B_STABILITY")
    if planner.get("can_submit_order") is not False:
        blockers.append("PLANNER_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if planner.get("live_order_sent") is not False:
        blockers.append("PLANNER_LIVE_ORDER_SENT_TRUE_UNEXPECTED")
    if planner.get("execution_authorized") is not False:
        blockers.append("PLANNER_EXECUTION_AUTHORIZED_TRUE_UNEXPECTED")
    if planner.get("token_binding_required") is not True:
        blockers.append("PLANNER_TOKEN_BINDING_NOT_REQUIRED")
    if planner.get("requires_new_token") is not True:
        blockers.append("PLANNER_DOES_NOT_REQUIRE_NEW_TOKEN")
    if not planner_hash or len(planner_hash) != 64:
        blockers.append("PLANNER_HASH_MISSING_OR_INVALID")
    if quote_price is None:
        blockers.append("QUOTE_PRICE_MISSING")
    if quote_size is None:
        blockers.append("QUOTE_SIZE_MISSING")
    if hold_seconds is None or hold_seconds <= 0:
        blockers.append("HOLD_SECONDS_MISSING")
    if fill_probability is None:
        blockers.append("FILL_PROBABILITY_PROXY_MISSING")
    elif stability_max_fill_probability is not None and fill_probability > stability_max_fill_probability:
        blockers.append("FILL_PROBABILITY_TOO_HIGH_FOR_B_STABILITY")
    if candidate_search.get("status") != "LOW_FILL_STABILITY_CANDIDATE_SEARCH_READY":
        blockers.append("LOW_FILL_STABILITY_SEARCH_NOT_READY")
    if not _candidate_matches_plan(best_candidate, recommended):
        blockers.append("SEARCH_BEST_CANDIDATE_DOES_NOT_MATCH_PLANNER")
    if candidate_search.get("can_submit_order") is not False:
        blockers.append("SEARCH_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if candidate_search.get("live_order_sent") is not False:
        blockers.append("SEARCH_LIVE_ORDER_SENT_TRUE_UNEXPECTED")
    if second_probe_decision and second_probe_decision.get("recommended_option") != "B_LONG_OBSERVATION_BID_ONLY_STABILITY_PROBE":
        blockers.append("SECOND_PROBE_DECISION_NOT_B_STABILITY")
    return _unique(blockers)


def _candidate_matches_plan(candidate: dict[str, Any], recommended: dict[str, Any]) -> bool:
    if not candidate or not recommended:
        return False
    return (
        str(candidate.get("market_slug") or "") == str(recommended.get("market_slug") or "")
        and _same_float(candidate.get("quote_price") or candidate.get("quote_bid"), recommended.get("quote_price") or recommended.get("quote_bid"))
        and _same_float(candidate.get("quote_size"), recommended.get("quote_size"))
        and _same_float(candidate.get("fill_probability"), recommended.get("fill_probability"))
    )


def _nested_bool(payload: dict[str, Any], key: str, nested_key: str) -> bool:
    nested = payload.get(key) if isinstance(payload.get(key), dict) else {}
    return nested.get(nested_key) is True


def _same_float(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    parsed_left = _first_float(left)
    parsed_right = _first_float(right)
    if parsed_left is None or parsed_right is None:
        return False
    return abs(parsed_left - parsed_right) <= tolerance


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


def _first_int(value: Any) -> int | None:
    parsed = _first_float(value)
    return int(parsed) if parsed is not None else None


def _round(value: Any, digits: int = 6) -> float | None:
    parsed = _first_float(value)
    if parsed is None:
        return None
    return round(parsed, digits)


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == READY_STATUS:
        return (
            "B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_READY: evidence frozen for manual review only; "
            "token and execution remain unauthorized."
        )
    return f"B_LOW_FILL_STABILITY_PROBE_APPROVAL_PACKAGE_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}."
