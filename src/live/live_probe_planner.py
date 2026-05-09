from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any


REPORT_SCHEMA_VERSION = "live_probe_planner.v1"
REPORT_TYPE = "live_probe_planner"

PLAN_READY_STATUS = "LIVE_PROBE_PLAN_READY"
PLAN_BLOCKED_STATUS = "LIVE_PROBE_PLAN_BLOCKED"
NO_SAFE_CANDIDATE_STATUS = "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE"
NO_SAFE_CANDIDATE_FOR_STABILITY_STATUS = "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE_FOR_STABILITY"

PLANNER_NAME = "NEXT_LIVE_PROBE_PLANNER"
RECOMMENDED_MODE = "B_LONG_OBSERVATION_BID_ONLY_STABILITY_PROBE"
REJECTED_MODE = "MAKER_BOTH_SIDES_LIVE"
PROBE_INTENT_STABILITY = "B_LONG_OBSERVATION_STABILITY"
PROBE_INTENT_FILL_LIKELIHOOD = "C_FILL_LIKELIHOOD_RECONCILIATION"
PLAN_CLASSIFICATION_STABILITY = "B_LONG_OBSERVATION_STABILITY"
PLAN_RECLASSIFIED_TO_FILL_LIKELIHOOD = "PLAN_RECLASSIFIED_TO_FILL_LIKELIHOOD"
DEFAULT_SELECTED_SIDE = "BID_ONLY"
DEFAULT_HOLD_SECONDS = 300
DEFAULT_TOKEN_TTL_SECONDS = 600
DEFAULT_PLANNER_VALID_SECONDS = 120
DEFAULT_VISIBILITY_GRACE_PERIOD_MS = 2000.0
DEFAULT_MIN_PROBE_SIZE = 50.0
DEFAULT_STABILITY_MAX_FILL_PROBABILITY = 0.30


def build_live_probe_plan(
    *,
    gate: dict[str, Any],
    deposit_wallet: dict[str, Any],
    market_microstructure: dict[str, Any] | None = None,
    candidate_market_microstructures: list[dict[str, Any]] | None = None,
    toxic_flow: dict[str, Any],
    fee_reconciliation: dict[str, Any],
    inventory_state: dict[str, Any],
    order_mutex: dict[str, Any],
    network: dict[str, Any],
    target_market_slug: str | None = None,
    max_live_risk_usdc: float | None = None,
    min_probe_size: float = DEFAULT_MIN_PROBE_SIZE,
    hold_seconds: int = DEFAULT_HOLD_SECONDS,
    token_ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
    planner_valid_seconds: int = DEFAULT_PLANNER_VALID_SECONDS,
    visibility_grace_period_ms: float = DEFAULT_VISIBILITY_GRACE_PERIOD_MS,
    probe_intent: str = PROBE_INTENT_STABILITY,
    stability_max_fill_probability: float = DEFAULT_STABILITY_MAX_FILL_PROBABILITY,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only plan for the next single-side live probe.

    This is a planning artifact only. It must never create a token, authorize
    execution, place an order, cancel an order, or consume an approval.
    """

    now = now or datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=planner_valid_seconds)
    max_live_risk = _first_float(max_live_risk_usdc, gate.get("max_live_risk_usdc"))
    available_usdc = _first_float(deposit_wallet.get("available_usdc"))
    candidates = _candidate_reports(market_microstructure, candidate_market_microstructures)
    global_blockers = _global_blockers(
        gate=gate,
        deposit_wallet=deposit_wallet,
        inventory_state=inventory_state,
        order_mutex=order_mutex,
        network=network,
        available_usdc=available_usdc,
        max_live_risk=max_live_risk,
    )

    evaluated = [
        _evaluate_candidate(
            candidate=candidate,
            toxic_flow=_candidate_specific(candidate, "toxic_flow", toxic_flow),
            fee_reconciliation=_candidate_specific(candidate, "fee_reconciliation", fee_reconciliation),
            available_usdc=available_usdc,
            max_live_risk=max_live_risk,
            min_probe_size=min_probe_size,
            target_market_slug=target_market_slug,
            probe_intent=probe_intent,
            stability_max_fill_probability=stability_max_fill_probability,
        )
        for candidate in candidates
    ]
    safe_candidates = [item for item in evaluated if not item["blockers"]]
    recommended_plan = _choose_recommended_plan(safe_candidates)
    rejected_plans = [item for item in evaluated if item["blockers"]]
    reclassified_candidates = [
        item for item in rejected_plans if item.get("plan_classification") == PLAN_RECLASSIFIED_TO_FILL_LIKELIHOOD
    ]
    selected_side = DEFAULT_SELECTED_SIDE

    status = PLAN_BLOCKED_STATUS
    blockers = list(global_blockers)
    if not global_blockers and recommended_plan is None:
        if probe_intent == PROBE_INTENT_STABILITY and reclassified_candidates:
            status = NO_SAFE_CANDIDATE_FOR_STABILITY_STATUS
            blockers.append("NO_SAFE_CANDIDATE_FOR_STABILITY")
        else:
            status = NO_SAFE_CANDIDATE_STATUS
            blockers.append("NO_SAFE_CANDIDATE")
    elif not global_blockers and recommended_plan is not None:
        status = PLAN_READY_STATUS
    for item in rejected_plans:
        item["selection_result"] = "REJECTED"
    if recommended_plan is not None:
        recommended_plan["selection_result"] = "SELECTED"

    core = {
        "planner_name": PLANNER_NAME,
        "planner_snapshot_ts": now.isoformat(),
        "planner_expires_at": expires_at.isoformat(),
        "planner_valid_seconds": planner_valid_seconds,
        "recommended_plan": recommended_plan,
        "global_blockers": global_blockers,
        "max_live_risk_usdc": _round(max_live_risk),
        "hold_seconds": hold_seconds,
        "token_ttl_seconds": token_ttl_seconds,
        "visibility_grace_period_ms": _round(visibility_grace_period_ms),
        "probe_intent": probe_intent,
        "stability_max_fill_probability": _round(stability_max_fill_probability),
    }
    planner_hash = _stable_hash(core)

    report = {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "planner_name": PLANNER_NAME,
        "planner_snapshot_ts": now.isoformat(),
        "planner_expires_at": expires_at.isoformat(),
        "planner_valid_seconds": planner_valid_seconds,
        "planner_hash": planner_hash,
        "read_only": True,
        "global_market_scan_performed": False,
        "is_global_optimum": False,
        "optimization_scope": "CURRENT_APPROVED_MARKET_OR_PROVIDED_CANDIDATES_ONLY",
        "optimization_boundary": (
            "Planner evaluates refreshed local candidate reports only. It does not scan every Polymarket market, "
            "so the recommendation is not a global optimum claim."
        ),
        "requested_probe_intent": probe_intent,
        "stability_max_fill_probability": _round(stability_max_fill_probability),
        "plan_classification": recommended_plan.get("plan_classification") if recommended_plan else None,
        "classification_audit": [
            {
                "candidate_id": item.get("candidate_id"),
                "market_slug": item.get("market_slug"),
                "plan_classification": item.get("plan_classification"),
                "reclassified_probe_intent": item.get("reclassified_probe_intent"),
                "fill_probability": item.get("fill_probability"),
                "stability_max_fill_probability": item.get("stability_max_fill_probability"),
                "blockers": item.get("blockers") or [],
            }
            for item in evaluated
        ],
        "reclassified_candidates": [item.get("market_slug") or item.get("candidate_id") for item in reclassified_candidates],
        "recommended_next_mode": RECOMMENDED_MODE,
        "rejected_next_mode": REJECTED_MODE,
        "rejected_next_mode_reason": (
            "Cold-start live proving should isolate one side only; both-side live adds inventory and cancel/replace risk too early."
        ),
        "selected_side": selected_side,
        "candidate_plans": evaluated,
        "recommended_plan": recommended_plan,
        "rejected_plans": rejected_plans,
        "rejected_candidates": [item.get("market_slug") for item in rejected_plans],
        "rejection_reasons": {
            str(item.get("market_slug") or item.get("candidate_id")): item.get("blockers") or []
            for item in rejected_plans
        },
        "input_sources": {
            "gate_status": gate.get("status"),
            "deposit_wallet_status": deposit_wallet.get("status"),
            "market_candidate_count": len(candidates),
            "toxic_flow_status": toxic_flow.get("status"),
            "fee_reconciliation_status": fee_reconciliation.get("status"),
            "inventory_state": inventory_state.get("status"),
            "order_mutex_state": order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status"),
            "network_status": network.get("status"),
        },
        "market_snapshot": _market_snapshot(recommended_plan, evaluated),
        "orderbook_snapshot": _orderbook_snapshot(recommended_plan, toxic_flow),
        "reward_snapshot": _reward_snapshot(recommended_plan),
        "fee_snapshot": _fee_snapshot(recommended_plan, fee_reconciliation),
        "wallet_snapshot": {
            "status": deposit_wallet.get("status"),
            "available_usdc": _round(available_usdc),
            "max_live_risk_usdc": _round(max_live_risk),
        },
        "toxic_flow_snapshot": {
            "status": toxic_flow.get("status"),
            "blockers": toxic_flow.get("blockers") if isinstance(toxic_flow.get("blockers"), list) else [],
            "orderbook_imbalance": _round(toxic_flow.get("orderbook_imbalance")),
            "fill_probability": _round(toxic_flow.get("fill_probability")),
        },
        "max_live_risk_usdc": _round(max_live_risk),
        "capital_buffer_usdc": (
            _round((max_live_risk or 0.0) - (recommended_plan.get("capital_required_usdc") or 0.0))
            if recommended_plan and max_live_risk is not None
            else None
        ),
        "expected_max_loss_if_filled": recommended_plan.get("expected_max_loss_if_filled") if recommended_plan else None,
        "primary_success_criteria": (
            "One BID order is submitted under a separate fresh approval/token, remains observable for the configured hold window, "
            "then cancel-by-order-id returns final open_order_count=0 and inventory clear. Fill is secondary, not required."
        ),
        "abort_conditions": [
            "heartbeat degrades before or during hold window",
            "toxic-flow/adverse-selection report blocks current market",
            "open order remains invisible outside visibility grace period",
            "unexpected fill is detected and inventory reconciliation cannot prove quantity/cost",
            "cancel by order_id is not confirmed",
            "final open_order_count is not zero",
            "final inventory is not clear or explicitly reconciled",
        ],
        "token_binding_required": True,
        "token_binding_fields": [
            "market_slug",
            "selected_side",
            "quote_price",
            "quote_size",
            "max_live_risk_usdc",
            "hold_seconds",
            "planner_hash",
        ],
        "new_token_required": True,
        "requires_new_token": True,
        "requires_new_approval": True,
        "same_token_retry_allowed": False,
        "same_approval_retry_allowed": False,
        "execution_authorized": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "profitability_claimed": False,
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
        "pending_reward_counted_as_confirmed_reward": False,
        "validation": {
            "readiness_12_12": _prelive_ready(gate),
            "deployment_sync_ok": _gate_assertion_passed(gate, "DEPLOYMENT_SYNC_ASSERT"),
            "execution_authorized": False,
            "can_submit_order": False,
            "live_order_sent": False,
            "new_token_required": True,
            "same_token_retry_allowed": False,
            "same_approval_retry_allowed": False,
            "inventory_clear": inventory_state.get("status") == "INVENTORY_STATE_CLEAR",
            "order_mutex_clear": (order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status")) == "NO_ORDER",
            "heartbeat_ready": network.get("status") in {"API_HEARTBEAT_READY", "LIVE_NETWORK_READY"},
            "fee_check": recommended_plan.get("checks", {}).get("fee_check") if recommended_plan else False,
            "toxic_flow_check": recommended_plan.get("checks", {}).get("toxic_flow_check") if recommended_plan else False,
            "tick_size_check": recommended_plan.get("checks", {}).get("tick_size_check") if recommended_plan else False,
            "reward_min_size_check": recommended_plan.get("checks", {}).get("reward_min_size_check") if recommended_plan else False,
            "stability_fill_probability_check": (
                recommended_plan.get("checks", {}).get("stability_fill_probability_check") if recommended_plan else False
            ),
            "capital_required_within_allowed_risk": (
                recommended_plan.get("checks", {}).get("capital_required_within_allowed_risk") if recommended_plan else False
            ),
        },
        "status": status,
        "blockers": _unique(blockers),
        "one_line_verdict": _one_line_verdict(status, blockers),
    }
    return report


def markdown_report(report: dict[str, Any]) -> str:
    recommended = report.get("recommended_plan") if isinstance(report.get("recommended_plan"), dict) else {}
    lines = [
        "# Next Live Probe Planner",
        "",
        "## Boundary",
        f"- Status: {report.get('status')}",
        f"- Planner hash: {report.get('planner_hash')}",
        f"- Execution authorized: {report.get('execution_authorized')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        f"- Requires new approval: {report.get('requires_new_approval')}",
        f"- Requires new token: {report.get('requires_new_token')}",
        f"- Requested probe intent: {report.get('requested_probe_intent')}",
        f"- Stability max fill probability: {report.get('stability_max_fill_probability')}",
        "",
        "## Recommended Plan",
        f"- Mode: {report.get('recommended_next_mode')}",
        f"- Plan classification: {recommended.get('plan_classification')}",
        f"- Side: {report.get('selected_side')}",
        f"- Market: {recommended.get('market_slug')}",
        f"- Quote price: {recommended.get('quote_price')}",
        f"- Quote size: {recommended.get('quote_size')}",
        f"- Capital required USDC: {recommended.get('capital_required_usdc')}",
        f"- Expected max loss if filled: {recommended.get('expected_max_loss_if_filled')}",
        "",
        "## Scope",
        f"- Global market scan performed: {report.get('global_market_scan_performed')}",
        f"- Is global optimum: {report.get('is_global_optimum')}",
        f"- Optimization scope: {report.get('optimization_scope')}",
        "",
        "## Candidate Plans",
    ]
    for item in report.get("candidate_plans") or []:
        lines.append(
            f"- {item.get('market_slug') or item.get('candidate_id')}: "
            f"{item.get('selection_result')} - {', '.join(item.get('blockers') or ['PASS'])}"
        )
    lines.extend(["", "## Validation"])
    validation = report.get("validation") if isinstance(report.get("validation"), dict) else {}
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _candidate_reports(
    market_microstructure: dict[str, Any] | None,
    candidate_market_microstructures: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if candidate_market_microstructures:
        out.extend(item for item in candidate_market_microstructures if isinstance(item, dict))
    if market_microstructure:
        out.append(market_microstructure)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        marker = str(item.get("market_slug") or item.get("token_id") or len(seen))
        if marker in seen:
            continue
        unique.append(item)
        seen.add(marker)
    return unique


def _evaluate_candidate(
    *,
    candidate: dict[str, Any],
    toxic_flow: dict[str, Any],
    fee_reconciliation: dict[str, Any],
    available_usdc: float | None,
    max_live_risk: float | None,
    min_probe_size: float,
    target_market_slug: str | None,
    probe_intent: str,
    stability_max_fill_probability: float,
) -> dict[str, Any]:
    market_slug = _first_text(candidate.get("market_slug"))
    quote_bid = _first_float(candidate.get("quote_bid"), candidate.get("best_bid"))
    quote_ask = _first_float(candidate.get("quote_ask"), candidate.get("best_ask"))
    best_bid = _first_float(candidate.get("best_bid"), quote_bid)
    best_ask = _first_float(candidate.get("best_ask"), quote_ask)
    rewards_min_size = _first_float(candidate.get("rewards_min_size"))
    quote_size = _recommended_size(rewards_min_size=rewards_min_size, min_probe_size=min_probe_size)
    quote_price = quote_bid
    capital_required = _capital_required(quote_price, quote_size)
    expected_max_loss = capital_required
    fee_quote_bid = _first_float(fee_reconciliation.get("quote_bid"))
    fee_quote_size = _first_float(fee_reconciliation.get("quote_size"))
    fill_probability = _first_float(toxic_flow.get("fill_probability"))
    plan_classification = PLAN_CLASSIFICATION_STABILITY if probe_intent == PROBE_INTENT_STABILITY else probe_intent
    reclassified_probe_intent: str | None = None
    stability_fill_probability_check = True
    blockers: list[str] = []

    if target_market_slug and market_slug != target_market_slug:
        blockers.append("CANDIDATE_MARKET_DOES_NOT_MATCH_TARGET_SCOPE")
    if candidate.get("status") != "MARKET_MICROSTRUCTURE_READY":
        blockers.append("MARKET_MICROSTRUCTURE_NOT_READY")
    if not market_slug:
        blockers.append("MARKET_SLUG_MISSING")
    if quote_bid is None:
        blockers.append("QUOTE_BID_MISSING")
    if quote_ask is None:
        blockers.append("QUOTE_ASK_MISSING")
    if quote_bid is not None and quote_ask is not None and quote_bid >= quote_ask:
        blockers.append("QUOTE_PRICE_INVERSION")
    if candidate.get("checks") and not _candidate_check(candidate, "quote_bid_tick_aligned"):
        blockers.append("QUOTE_BID_TICK_MISALIGNED")
    if candidate.get("checks") and not _candidate_check(candidate, "quote_ask_tick_aligned"):
        blockers.append("QUOTE_ASK_TICK_MISALIGNED")
    if candidate.get("checks") and not _candidate_check(candidate, "price_bounds_ok"):
        blockers.append("QUOTE_PRICE_OUT_OF_BOUNDS")
    if rewards_min_size is None:
        blockers.append("REWARD_MIN_SIZE_MISSING")
    if quote_size is None:
        blockers.append("QUOTE_SIZE_MISSING")
    elif rewards_min_size is not None and quote_size < rewards_min_size:
        blockers.append("QUOTE_SIZE_BELOW_REWARD_MIN_SIZE")
    if capital_required is None:
        blockers.append("CAPITAL_REQUIRED_UNKNOWN")
    if available_usdc is None:
        blockers.append("AVAILABLE_USDC_MISSING")
    elif capital_required is not None and available_usdc < capital_required:
        blockers.append("AVAILABLE_USDC_BELOW_CAPITAL_REQUIRED")
    if max_live_risk is None:
        blockers.append("MAX_LIVE_RISK_USDC_MISSING")
    elif capital_required is not None and max_live_risk < capital_required:
        blockers.append("CAPITAL_REQUIRED_EXCEEDS_MAX_LIVE_RISK")
    if toxic_flow.get("market_slug") and market_slug and toxic_flow.get("market_slug") != market_slug:
        blockers.append("TOXIC_FLOW_REPORT_NOT_BOUND_TO_CANDIDATE")
    if toxic_flow.get("status") != "TOXIC_FLOW_READY" or toxic_flow.get("blockers"):
        blockers.append("TOXIC_FLOW_NOT_CLEAR")
    if toxic_flow.get("fill_probability_ok") is False:
        blockers.append("FILL_PROBABILITY_BELOW_MINIMUM")
    if probe_intent == PROBE_INTENT_STABILITY:
        if fill_probability is None:
            blockers.append("FILL_PROBABILITY_MISSING_FOR_STABILITY_PROBE")
            stability_fill_probability_check = False
        elif fill_probability > stability_max_fill_probability:
            blockers.append("FILL_PROBABILITY_TOO_HIGH_FOR_STABILITY_PROBE")
            plan_classification = PLAN_RECLASSIFIED_TO_FILL_LIKELIHOOD
            reclassified_probe_intent = PROBE_INTENT_FILL_LIKELIHOOD
            stability_fill_probability_check = False
    if fee_reconciliation.get("market_slug") and market_slug and fee_reconciliation.get("market_slug") != market_slug:
        blockers.append("FEE_REPORT_NOT_BOUND_TO_CANDIDATE")
    if fee_reconciliation.get("status") != "FEE_RECONCILIATION_READY" or fee_reconciliation.get("can_cover_fees") is not True:
        blockers.append("FEE_RECONCILIATION_NOT_READY")
    if not _fee_report_bound_to_plan(
        fee_reconciliation=fee_reconciliation,
        quote_price=quote_price,
        quote_size=quote_size,
        fee_quote_bid=fee_quote_bid,
        fee_quote_size=fee_quote_size,
    ):
        blockers.append("FEE_REPORT_NOT_BOUND_TO_PLAN_PRICE_SIZE")

    checks = {
        "tick_size_check": "QUOTE_BID_TICK_MISALIGNED" not in blockers
        and "QUOTE_ASK_TICK_MISALIGNED" not in blockers
        and "QUOTE_PRICE_OUT_OF_BOUNDS" not in blockers
        and "QUOTE_PRICE_INVERSION" not in blockers,
        "reward_min_size_check": rewards_min_size is not None and quote_size is not None and quote_size >= rewards_min_size,
        "toxic_flow_check": "TOXIC_FLOW_NOT_CLEAR" not in blockers
        and "FILL_PROBABILITY_BELOW_MINIMUM" not in blockers
        and "FILL_PROBABILITY_MISSING_FOR_STABILITY_PROBE" not in blockers
        and "FILL_PROBABILITY_TOO_HIGH_FOR_STABILITY_PROBE" not in blockers,
        "stability_fill_probability_check": stability_fill_probability_check,
        "fee_check": "FEE_RECONCILIATION_NOT_READY" not in blockers and "FEE_REPORT_NOT_BOUND_TO_PLAN_PRICE_SIZE" not in blockers,
        "capital_required_within_allowed_risk": max_live_risk is not None
        and capital_required is not None
        and capital_required <= max_live_risk,
    }
    return {
        "candidate_id": _candidate_id(market_slug, quote_price, quote_size),
        "market_slug": market_slug,
        "selected_side": DEFAULT_SELECTED_SIDE,
        "requested_probe_intent": probe_intent,
        "plan_classification": plan_classification,
        "reclassified_probe_intent": reclassified_probe_intent,
        "quote_price": _round(quote_price),
        "quote_bid": _round(quote_bid),
        "quote_ask": _round(quote_ask),
        "quote_size": _round(quote_size),
        "best_bid": _round(best_bid),
        "best_ask": _round(best_ask),
        "capital_required_usdc": _round(capital_required),
        "capital_buffer_usdc": _round((max_live_risk or 0.0) - (capital_required or 0.0)) if max_live_risk is not None else None,
        "expected_max_loss_if_filled": _round(expected_max_loss),
        "sizing_reason": _sizing_reason(rewards_min_size, min_probe_size),
        "reward_min_size": _round(rewards_min_size),
        "reward_max_spread_cents": _round(candidate.get("rewards_max_spread_cents")),
        "estimated_net_profit_usdc": _round(fee_reconciliation.get("estimated_net_profit_usdc")),
        "fill_probability": _round(fill_probability),
        "stability_max_fill_probability": _round(stability_max_fill_probability),
        "toxic_flow_status": toxic_flow.get("status"),
        "fee_reconciliation_status": fee_reconciliation.get("status"),
        "checks": checks,
        "blockers": _unique(blockers),
        "selection_result": "PENDING",
    }


def _global_blockers(
    *,
    gate: dict[str, Any],
    deposit_wallet: dict[str, Any],
    inventory_state: dict[str, Any],
    order_mutex: dict[str, Any],
    network: dict[str, Any],
    available_usdc: float | None,
    max_live_risk: float | None,
) -> list[str]:
    blockers: list[str] = []
    if not _prelive_ready(gate):
        blockers.append("PRELIVE_12_OF_12_NOT_READY")
    if gate.get("can_submit_order") is not False:
        blockers.append("GATE_CAN_SUBMIT_ORDER_TRUE_UNEXPECTED")
    if gate.get("live_order_sent") is not False:
        blockers.append("GATE_LIVE_ORDER_SENT_TRUE_UNEXPECTED")
    if deposit_wallet.get("status") != "DEPOSIT_WALLET_READY":
        blockers.append("DEPOSIT_WALLET_NOT_READY")
    if available_usdc is None:
        blockers.append("AVAILABLE_USDC_MISSING")
    if max_live_risk is None:
        blockers.append("MAX_LIVE_RISK_USDC_MISSING")
    if inventory_state.get("status") != "INVENTORY_STATE_CLEAR":
        blockers.append("INVENTORY_NOT_CLEAR")
    if (order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status")) != "NO_ORDER":
        blockers.append("ORDER_MUTEX_NOT_CLEAR")
    if network.get("status") not in {"API_HEARTBEAT_READY", "LIVE_NETWORK_READY"}:
        blockers.append("HEARTBEAT_NOT_READY")
    return _unique(blockers)


def _choose_recommended_plan(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item.get("capital_required_usdc") if item.get("capital_required_usdc") is not None else float("inf"),
            -(item.get("estimated_net_profit_usdc") or 0.0),
        ),
    )[0]


def _candidate_specific(candidate: dict[str, Any], key: str, fallback: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get(key)
    return value if isinstance(value, dict) else fallback


def _market_snapshot(recommended_plan: dict[str, Any] | None, evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    source = recommended_plan or (evaluated[0] if evaluated else {})
    return {
        "market_slug": source.get("market_slug"),
        "quote_bid": source.get("quote_bid"),
        "quote_ask": source.get("quote_ask"),
        "best_bid": source.get("best_bid"),
        "best_ask": source.get("best_ask"),
    }


def _orderbook_snapshot(recommended_plan: dict[str, Any] | None, toxic_flow: dict[str, Any]) -> dict[str, Any]:
    source = recommended_plan or {}
    return {
        "best_bid": source.get("best_bid"),
        "best_ask": source.get("best_ask"),
        "best_bid_size": _round(toxic_flow.get("best_bid_size")),
        "best_ask_size": _round(toxic_flow.get("best_ask_size")),
        "orderbook_imbalance": _round(toxic_flow.get("orderbook_imbalance")),
    }


def _reward_snapshot(recommended_plan: dict[str, Any] | None) -> dict[str, Any]:
    source = recommended_plan or {}
    return {
        "reward_min_size": source.get("reward_min_size"),
        "reward_max_spread_cents": source.get("reward_max_spread_cents"),
        "reward_counted_as_realized_pnl": False,
    }


def _fee_snapshot(recommended_plan: dict[str, Any] | None, fee_reconciliation: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": fee_reconciliation.get("status"),
        "can_cover_fees": fee_reconciliation.get("can_cover_fees"),
        "estimated_net_profit_usdc": _round(fee_reconciliation.get("estimated_net_profit_usdc")),
        "estimated_net_profit_counted_as_realized_cash_pnl": False,
        "bound_candidate_id": recommended_plan.get("candidate_id") if recommended_plan else None,
    }


def _prelive_ready(gate: dict[str, Any]) -> bool:
    return (
        gate.get("status") == "LIVE_READY_APPROVED"
        and _optional_int(gate.get("asserts_passed")) == 12
        and _optional_int(gate.get("asserts_failed")) == 0
        and not gate.get("blockers")
    )


def _gate_assertion_passed(gate: dict[str, Any], assert_id: str) -> bool:
    by_id = gate.get("asserts_by_id") if isinstance(gate.get("asserts_by_id"), dict) else {}
    item = by_id.get(assert_id) if isinstance(by_id.get(assert_id), dict) else {}
    return item.get("passed") is True


def _candidate_check(candidate: dict[str, Any], key: str) -> bool:
    checks = candidate.get("checks") if isinstance(candidate.get("checks"), dict) else {}
    return checks.get(key) is True


def _recommended_size(*, rewards_min_size: float | None, min_probe_size: float) -> float | None:
    candidates = [value for value in (rewards_min_size, min_probe_size) if value is not None and value > 0.0]
    return max(candidates) if candidates else None


def _sizing_reason(rewards_min_size: float | None, min_probe_size: float) -> str:
    if rewards_min_size is None:
        return "BLOCKED_REWARD_MIN_SIZE_UNKNOWN"
    if rewards_min_size >= min_probe_size:
        return "MIN_REWARD_ELIGIBLE_LOWEST_RISK_SIZE"
    return "MIN_PROBE_SIZE_ABOVE_REWARD_MIN_SIZE_BUFFER"


def _fee_report_bound_to_plan(
    *,
    fee_reconciliation: dict[str, Any],
    quote_price: float | None,
    quote_size: float | None,
    fee_quote_bid: float | None,
    fee_quote_size: float | None,
) -> bool:
    if fee_reconciliation.get("status") != "FEE_RECONCILIATION_READY":
        return False
    if quote_price is None or quote_size is None:
        return False
    if fee_quote_bid is None or fee_quote_size is None:
        return False
    return abs(fee_quote_bid - quote_price) < 1e-9 and abs(fee_quote_size - quote_size) < 1e-9


def _candidate_id(market_slug: str | None, quote_price: float | None, quote_size: float | None) -> str:
    return _stable_hash(
        {"market_slug": market_slug, "side": DEFAULT_SELECTED_SIDE, "quote_price": quote_price, "quote_size": quote_size}
    )[:16]


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capital_required(price: float | None, size: float | None) -> float | None:
    if price is None or size is None:
        return None
    return price * size


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(parsed):
            return parsed
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value in {None, ""}:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


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
    if status == PLAN_READY_STATUS:
        return "LIVE_PROBE_PLAN_READY: next-probe plan is ready for review only; execution remains unauthorized."
    if status == NO_SAFE_CANDIDATE_FOR_STABILITY_STATUS:
        return (
            "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE_FOR_STABILITY: candidate fill probability is too high for a "
            "stability probe; reclassify or wait for a lower-fill setup. No token or live order authorized."
        )
    if status == NO_SAFE_CANDIDATE_STATUS:
        return "LIVE_PROBE_PLAN_NO_SAFE_CANDIDATE: no candidate passed the read-only planner checks; no token or live order authorized."
    return f"LIVE_PROBE_PLAN_BLOCKED: {', '.join(blockers) or 'UNKNOWN'}; no token or live order authorized."
