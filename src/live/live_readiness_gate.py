from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


REPORT_SCHEMA_VERSION = "live_readiness_gate.v1"
REPORT_TYPE = "live_readiness_gate"
LIVE_READY_APPROVED = "LIVE_READY_APPROVED"
LIVE_NOT_READY = "LIVE_NOT_READY"

DEFAULT_MAX_REPORT_AGE_MINUTES = 2.0
DEFAULT_MAX_LIVE_RISK_USDC = 300.0
DEFAULT_FEE_BUFFER_USDC = 5.0
DEFAULT_CANCEL_BUFFER_USDC = 5.0

ASSERTION_ORDER = [
    "EXECUTION_ISOLATION_ASSERT",
    "DEPLOYMENT_SYNC_ASSERT",
    "AUTH_SCOPE_ASSERT",
    "DEPOSIT_WALLET_BALANCE_ASSERT",
    "ORDER_MUTEX_ASSERT",
    "CANCEL_HEARTBEAT_ASSERT",
    "TICK_SIZE_PRICE_ASSERT",
    "REWARD_SCORING_ASSERT",
    "FILL_ADVERSE_SELECTION_ASSERT",
    "INVENTORY_STATE_ASSERT",
    "FEE_RECONCILIATION_ASSERT",
    "FINAL_PHYSICAL_ASSERT",
]


def build_live_readiness_gate(
    *,
    health: dict[str, Any] | None = None,
    execution_system: dict[str, Any] | None = None,
    profit_gate: dict[str, Any] | None = None,
    approval: dict[str, Any] | None = None,
    auth_readiness: dict[str, Any] | None = None,
    deposit_wallet: dict[str, Any] | None = None,
    deployment: dict[str, Any] | None = None,
    network: dict[str, Any] | None = None,
    order_mutex: dict[str, Any] | None = None,
    market_microstructure: dict[str, Any] | None = None,
    toxic_flow: dict[str, Any] | None = None,
    inventory_state: dict[str, Any] | None = None,
    fee_reconciliation: dict[str, Any] | None = None,
    final_physical: dict[str, Any] | None = None,
    kill_switch: dict[str, Any] | None = None,
    target_market_slug: str | None = None,
    approved_action_scope: str = "FIRST_CYCLE_BOOTSTRAP_EVIDENCE_GENERATION_ONLY",
    max_live_risk_usdc: float = DEFAULT_MAX_LIVE_RISK_USDC,
    fee_buffer_usdc: float = DEFAULT_FEE_BUFFER_USDC,
    cancel_buffer_usdc: float = DEFAULT_CANCEL_BUFFER_USDC,
    max_report_age_minutes: float = DEFAULT_MAX_REPORT_AGE_MINUTES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only live-readiness gate report.

    This module is intentionally non-executing. It never enables order
    submission; execution code must treat this as a hard precondition, not as
    an order path.
    """

    health = health or {}
    execution_system = execution_system or {}
    profit_gate = profit_gate or {}
    approval = approval or {}
    auth_readiness = auth_readiness or {}
    deposit_wallet = deposit_wallet or {}
    deployment = deployment or {}
    network = network or {}
    order_mutex = order_mutex or {}
    toxic_flow = toxic_flow or {}
    inventory_state = inventory_state or {}
    fee_reconciliation = fee_reconciliation or {}
    final_physical = final_physical or {}
    kill_switch = kill_switch or {}
    market = _market_context(
        health=health,
        profit_gate=profit_gate,
        market_microstructure=market_microstructure or {},
        toxic_flow=toxic_flow,
    )
    now = now or datetime.now(timezone.utc)
    target_market_slug = target_market_slug or _target_market_slug(health=health, profit_gate=profit_gate, market=market)

    assertions = [
        _execution_isolation_assertion(execution_system, now=now, max_report_age_minutes=max_report_age_minutes),
        _deployment_sync_assertion(deployment, now=now, max_report_age_minutes=max_report_age_minutes),
        _auth_scope_assertion(
            approval,
            auth_readiness,
            target_market_slug=target_market_slug,
            approved_action_scope=approved_action_scope,
            now=now,
            max_report_age_minutes=max_report_age_minutes,
        ),
        _deposit_wallet_balance_assertion(
            deposit_wallet,
            max_live_risk_usdc=max_live_risk_usdc,
            fee_buffer_usdc=fee_buffer_usdc,
            cancel_buffer_usdc=cancel_buffer_usdc,
            now=now,
            max_report_age_minutes=max_report_age_minutes,
        ),
        _order_mutex_assertion(order_mutex),
        _cancel_heartbeat_assertion(network, now=now, max_report_age_minutes=max_report_age_minutes),
        _tick_size_price_assertion(market),
        _reward_scoring_assertion(market, approved_action_scope=approved_action_scope),
        _fill_adverse_selection_assertion(market, network),
        _inventory_state_assertion(
            health=health,
            profit_gate=profit_gate,
            inventory_state=inventory_state,
            now=now,
            max_report_age_minutes=max_report_age_minutes,
        ),
        _fee_reconciliation_assertion(fee_reconciliation, now=now, max_report_age_minutes=max_report_age_minutes),
        _final_physical_assertion(
            health=health,
            network=network,
            final_physical=final_physical,
            kill_switch=kill_switch,
            now=now,
            max_report_age_minutes=max_report_age_minutes,
        ),
    ]
    failed = [item for item in assertions if not item["passed"]]
    blockers = _unique(str(item["blocking_reason"]) for item in failed if item.get("blocking_reason"))
    status = LIVE_READY_APPROVED if not blockers else LIVE_NOT_READY
    asserts_by_id = {item["assert_id"]: item for item in assertions}

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "read_only": True,
        "live_actions_enabled": False,
        "execution_enabled": False,
        "can_submit_order": False,
        "live_order_sent": False,
        "live_decision_binding": False,
        "status": status,
        "target_market_slug": target_market_slug,
        "max_live_risk_usdc": _round(max_live_risk_usdc),
        "required_deposit_wallet_buffer_usdc": _round(
            max_live_risk_usdc + fee_buffer_usdc + cancel_buffer_usdc
        ),
        "assertions": assertions,
        "asserts_by_id": asserts_by_id,
        "asserts_passed": sum(1 for item in assertions if item["passed"]),
        "asserts_failed": len(failed),
        "blockers": blockers,
        "can_submit_order_gate_reason": (
            "REPORT_ONLY_LIVE_READINESS_GATE_NOT_EXECUTION_PATH"
            if status == LIVE_READY_APPROVED
            else "LIVE_EXECUTION_BLOCKED_BY_MICROSTRUCTURE_READINESS_GATE"
        ),
        "one_line_verdict": _one_line_verdict(status, blockers),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Live Readiness Gate v1",
        "",
        "## Safety",
        f"- Status: {report.get('status')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live actions enabled: {report.get('live_actions_enabled')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        f"- Target market: {report.get('target_market_slug')}",
        f"- Max live risk USDC: {report.get('max_live_risk_usdc')}",
        "",
        "## Assertions",
    ]
    for item in report.get("assertions") or []:
        status = "PASS" if item.get("passed") else "FAIL"
        lines.append(f"- {item.get('assert_id')}: {status} - {item.get('reason')}")
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _execution_isolation_assertion(
    execution_system: dict[str, Any],
    *,
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    fields_false = {
        "can_submit_order": execution_system.get("can_submit_order") is False,
        "execution_enabled": execution_system.get("execution_enabled") is False,
        "live_actions_enabled": execution_system.get("live_actions_enabled") is False,
        "live_order_sent": execution_system.get("live_order_sent") is False,
    }
    present = bool(execution_system)
    fresh = _is_fresh(execution_system, now=now, max_report_age_minutes=max_report_age_minutes)
    status_ready = execution_system.get("status") == "EXECUTION_ISOLATION_READY"
    process_scan_clear = execution_system.get("single_writer_ok") is True
    passed = present and all(fields_false.values()) and fresh and status_ready and process_scan_clear
    reason = "EXECUTION_SYSTEM_IS_HARD_DISABLED_AND_ISOLATED" if passed else "EXECUTION_SYSTEM_NOT_HARD_DISABLED_OR_ISOLATED"
    return _assertion(
        assert_id="EXECUTION_ISOLATION_ASSERT",
        category="execution_isolation",
        passed=passed,
        reason=reason if present else "EXECUTION_SYSTEM_REPORT_MISSING",
        blocking_reason=None if passed else "EXECUTION_ISOLATION_NOT_PROVEN",
        details={
            "required_false_fields": fields_false,
            "report_fresh": fresh,
            "status_ready": status_ready,
            "single_writer_ok": process_scan_clear,
            "suspicious_process_count": execution_system.get("suspicious_process_count"),
            "execution_blockers": execution_system.get("blockers") or [],
        },
    )


def _deployment_sync_assertion(
    deployment: dict[str, Any],
    *,
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    fresh = _is_fresh(deployment, now=now, max_report_age_minutes=max_report_age_minutes)
    checks = {
        "head_matches_approved": deployment.get("head_matches_approved") is True,
        "critical_checksums_match": deployment.get("critical_checksums_match") is True,
        "unreviewed_changes_absent": deployment.get("unreviewed_changes_present") is False,
        "report_fresh": fresh,
    }
    passed = bool(deployment) and all(checks.values())
    return _assertion(
        assert_id="DEPLOYMENT_SYNC_ASSERT",
        category="deployment",
        passed=passed,
        reason="DEPLOYMENT_SYNC_PROVEN" if passed else "DEPLOYMENT_SYNC_NOT_PROVEN",
        blocking_reason=None if passed else "DEPLOYMENT_SYNC_NOT_PROVEN",
        details=checks,
    )


def _auth_scope_assertion(
    approval: dict[str, Any],
    auth_readiness: dict[str, Any],
    *,
    target_market_slug: str | None,
    approved_action_scope: str,
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    if auth_readiness:
        checks = {
            "auth_report_present": True,
            "auth_scope_ready": auth_readiness.get("status") == "AUTH_SCOPE_READY",
            "level_2_auth_ok": auth_readiness.get("level_2_auth_ok") is True,
            "is_signing_enabled": auth_readiness.get("is_signing_enabled") is True,
            "configured_key_is_trading_key": auth_readiness.get("configured_key_is_trading_key") is True,
            "configured_key_is_not_readonly": auth_readiness.get("configured_key_is_readonly_key") is False,
            "funder_matches_deposit_wallet": auth_readiness.get("funder_matches_deposit_wallet") is True,
            "deposit_wallet_balance_sufficient": auth_readiness.get("deposit_wallet_balance_sufficient") is True,
            "rate_limit_clear": auth_readiness.get("rate_limit_degraded") is not True,
            "report_fresh": _is_fresh(auth_readiness, now=now, max_report_age_minutes=max_report_age_minutes),
        }
        passed = all(checks.values())
        return _assertion(
            assert_id="AUTH_SCOPE_ASSERT",
            category="authorization",
            passed=passed,
            reason="AUTH_SCOPE_AND_KEY_PERMISSIONS_PROVEN" if passed else "AUTH_SCOPE_OR_KEY_PERMISSIONS_NOT_PROVEN",
            blocking_reason=None if passed else "AUTH_SCOPE_NOT_PROVEN",
            details={
                **checks,
                "auth_readiness_status": auth_readiness.get("status"),
                "abnormal_restrictions": auth_readiness.get("abnormal_restrictions") or [],
                "auth_blockers": auth_readiness.get("blockers") or [],
            },
        )

    approved_market = str(
        approval.get("approved_market_slug")
        or approval.get("market_slug")
        or approval.get("canary_market_slug")
        or ""
    )
    scopes = approval.get("approved_action_scopes") or approval.get("scope") or []
    if isinstance(scopes, str):
        scopes = [scopes]
    checks = {
        "approval_present": bool(approval),
        "market_scope_matches": bool(target_market_slug) and approved_market == target_market_slug,
        "action_scope_matches": approved_action_scope in set(str(item) for item in scopes),
        "approval_not_consumed": approval.get("consumed") is not True,
        "report_fresh": _is_fresh(approval, now=now, max_report_age_minutes=max_report_age_minutes),
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="AUTH_SCOPE_ASSERT",
        category="authorization",
        passed=passed,
        reason="AUTH_SCOPE_AND_APPROVAL_PROVEN" if passed else "AUTH_SCOPE_OR_APPROVAL_NOT_PROVEN",
        blocking_reason=None if passed else "AUTH_SCOPE_NOT_PROVEN",
        details={**checks, "approved_market_slug": approved_market, "target_market_slug": target_market_slug},
    )


def _deposit_wallet_balance_assertion(
    deposit_wallet: dict[str, Any],
    *,
    max_live_risk_usdc: float,
    fee_buffer_usdc: float,
    cancel_buffer_usdc: float,
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    required = max_live_risk_usdc + fee_buffer_usdc + cancel_buffer_usdc
    available = _first_float(
        deposit_wallet.get("available_usdc"),
        deposit_wallet.get("available_balance_usdc"),
        deposit_wallet.get("deposit_wallet_available_usdc"),
        (deposit_wallet.get("balances") or {}).get("available_usdc")
        if isinstance(deposit_wallet.get("balances"), dict)
        else None,
        (deposit_wallet.get("deposit_wallet") or {}).get("available_usdc")
        if isinstance(deposit_wallet.get("deposit_wallet"), dict)
        else None,
    )
    wallet_type = str(deposit_wallet.get("wallet_type") or "").upper()
    source_ok = wallet_type == "DEPOSIT_WALLET"
    fresh = _is_fresh(deposit_wallet, now=now, max_report_age_minutes=max_report_age_minutes)
    checks = {
        "report_present": bool(deposit_wallet),
        "read_only": deposit_wallet.get("read_only") is True,
        "source_is_deposit_wallet": source_ok,
        "balance_present": available is not None,
        "balance_covers_required_buffer": available is not None and available >= required,
        "report_fresh": fresh,
        "no_adapter_error": not bool(deposit_wallet.get("error") or deposit_wallet.get("errors")),
    }
    passed = all(checks.values())
    blocker = None
    if not passed:
        blocker = _deposit_wallet_blocker(checks)
    return _assertion(
        assert_id="DEPOSIT_WALLET_BALANCE_ASSERT",
        category="deposit_wallet",
        passed=passed,
        reason="DEPOSIT_WALLET_BALANCE_READY" if passed else blocker,
        blocking_reason=None if passed else blocker,
        details={
            **checks,
            "wallet_type": wallet_type or None,
            "available_usdc": _round(available),
            "required_usdc": _round(required),
            "max_live_risk_usdc": _round(max_live_risk_usdc),
            "fee_buffer_usdc": _round(fee_buffer_usdc),
            "cancel_buffer_usdc": _round(cancel_buffer_usdc),
        },
    )


def _order_mutex_assertion(order_mutex: dict[str, Any]) -> dict[str, Any]:
    state = str(order_mutex.get("order_mutex_state") or order_mutex.get("live_order_status") or "")
    checks = {
        "mutex_report_present": bool(order_mutex),
        "state_is_no_order": state == "NO_ORDER",
        "state_valid": state in {"NO_ORDER", "PLACE_IN_FLIGHT", "LIVE_ORDER_OPEN", "CANCEL_IN_FLIGHT"},
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="ORDER_MUTEX_ASSERT",
        category="order_mutex",
        passed=passed,
        reason="ORDER_MUTEX_CLEAR" if passed else "ORDER_MUTEX_NOT_CLEAR",
        blocking_reason=None if passed else "ORDER_MUTEX_NOT_CLEAR",
        details={**checks, "order_mutex_state": state or None},
    )


def _cancel_heartbeat_assertion(
    network: dict[str, Any],
    *,
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    cancel_latency_ms = _first_float(network.get("cancel_latency_ms"), network.get("p95_cancel_latency_ms"))
    max_cancel_latency_ms = _first_float(network.get("max_cancel_latency_ms"), 1000.0)
    checks = {
        "network_report_present": bool(network),
        "heartbeat_ok": network.get("heartbeat_ok") is True,
        "mass_cancel_ready": network.get("mass_cancel_ready") is True,
        "http_425_window_clear": network.get("http_425_window_active") is False,
        "cancel_latency_present": cancel_latency_ms is not None,
        "cancel_latency_ok": cancel_latency_ms is not None and cancel_latency_ms <= max_cancel_latency_ms,
        "report_fresh": _is_fresh(network, now=now, max_report_age_minutes=max_report_age_minutes),
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="CANCEL_HEARTBEAT_ASSERT",
        category="cancel_heartbeat",
        passed=passed,
        reason="CANCEL_AND_HEARTBEAT_READY" if passed else "CANCEL_OR_HEARTBEAT_NOT_PROVEN",
        blocking_reason=None if passed else "CANCEL_HEARTBEAT_NOT_PROVEN",
        details={**checks, "cancel_latency_ms": _round(cancel_latency_ms), "max_cancel_latency_ms": _round(max_cancel_latency_ms)},
    )


def _tick_size_price_assertion(market: dict[str, Any]) -> dict[str, Any]:
    tick = _first_float(market.get("tick_size"), market.get("minimum_tick_size"))
    bid = _first_float(market.get("quote_bid"), market.get("best_bid"))
    ask = _first_float(market.get("quote_ask"), market.get("best_ask"))
    checks = {
        "market_present": bool(market),
        "tick_size_present": tick is not None and tick > 0,
        "quote_bid_present": bid is not None,
        "quote_ask_present": ask is not None,
        "quote_bid_below_quote_ask": bid is not None and ask is not None and bid < ask,
        "quote_bid_tick_aligned": _price_tick_aligned(bid, tick),
        "quote_ask_tick_aligned": _price_tick_aligned(ask, tick),
        "price_bounds_ok": _price_in_bounds(bid) and _price_in_bounds(ask),
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="TICK_SIZE_PRICE_ASSERT",
        category="market_microstructure",
        passed=passed,
        reason="TICK_SIZE_AND_QUOTE_SANITY_READY" if passed else "TICK_SIZE_OR_QUOTE_SANITY_NOT_PROVEN",
        blocking_reason=None if passed else "TICK_SIZE_PRICE_SANITY_NOT_PROVEN",
        details={**checks, "tick_size": _round(tick), "quote_bid": _round(bid), "quote_ask": _round(ask)},
    )


def _reward_scoring_assertion(market: dict[str, Any], *, approved_action_scope: str) -> dict[str, Any]:
    reward_size_required = approved_action_scope != "C_FILL_LIKELIHOOD_RECONCILIATION"
    min_size = _first_float(market.get("rewards_min_size"), market.get("reward_min_size"))
    max_spread = _first_float(market.get("rewards_max_spread_cents"), market.get("reward_max_spread_cents"))
    quote_size = _first_float(market.get("quote_size"), market.get("planned_quote_size"))
    bid = _first_float(market.get("quote_bid"), market.get("best_bid"))
    ask = _first_float(market.get("quote_ask"), market.get("best_ask"))
    spread_cents = None if bid is None or ask is None else (ask - bid) * 100.0
    checks = {
        "reward_metadata_present": min_size is not None and max_spread is not None,
        "quote_size_present": quote_size is not None,
        "quote_size_meets_reward_min": quote_size is not None and min_size is not None and quote_size >= min_size,
        "quote_size_reward_min_required_for_scope": reward_size_required,
        "spread_present": spread_cents is not None,
        "spread_inside_reward_band": spread_cents is not None and max_spread is not None and spread_cents <= max_spread,
    }
    passed = (
        all(checks.values())
        if reward_size_required
        else (
            checks["reward_metadata_present"]
            and checks["quote_size_present"]
            and checks["spread_present"]
            and checks["spread_inside_reward_band"]
        )
    )
    return _assertion(
        assert_id="REWARD_SCORING_ASSERT",
        category="reward_scoring",
        passed=passed,
        reason=(
            "REWARD_SCORING_INFORMATIONAL_FOR_C_FILL_RECONCILIATION"
            if passed and not reward_size_required
            else "REWARD_SCORING_BAND_PROVEN"
            if passed
            else "REWARD_SCORING_BAND_NOT_PROVEN"
        ),
        blocking_reason=None if passed else "REWARD_SCORING_BAND_NOT_PROVEN",
        details={
            **checks,
            "quote_size": _round(quote_size),
            "rewards_min_size": _round(min_size),
            "spread_cents": _round(spread_cents),
            "rewards_max_spread_cents": _round(max_spread),
        },
    )


def _fill_adverse_selection_assertion(market: dict[str, Any], network: dict[str, Any]) -> dict[str, Any]:
    fill_probability = _first_float(market.get("fill_probability"), market.get("maker_fill_probability"))
    min_fill_probability = _first_float(market.get("min_fill_probability"), 0.05)
    adverse_selection_score = _first_float(market.get("adverse_selection_score"))
    toxic_flow_detected = market.get("toxic_flow_detected") is True
    high_velocity_toxic_flow = (
        network.get("high_velocity_toxic_flow") is True
        or market.get("high_velocity_toxic_flow") is True
        or market.get("volatility_lock") is True
    )
    checks = {
        "fill_probability_present": fill_probability is not None,
        "fill_probability_ok": fill_probability is not None and fill_probability >= min_fill_probability,
        "toxic_flow_clear": not toxic_flow_detected,
        "adverse_selection_score_present": adverse_selection_score is not None,
        "orderbook_velocity_clear": not high_velocity_toxic_flow,
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="FILL_ADVERSE_SELECTION_ASSERT",
        category="fill_probability",
        passed=passed,
        reason="FILL_AND_ADVERSE_SELECTION_READY" if passed else "FILL_OR_ADVERSE_SELECTION_NOT_PROVEN",
        blocking_reason=None if passed else "FILL_ADVERSE_SELECTION_NOT_PROVEN",
        details={
            **checks,
            "fill_probability": _round(fill_probability),
            "min_fill_probability": _round(min_fill_probability),
            "adverse_selection_score": _round(adverse_selection_score),
            "toxic_flow_detected": toxic_flow_detected,
            "high_velocity_toxic_flow": high_velocity_toxic_flow,
        },
    )


def _inventory_state_assertion(
    *,
    health: dict[str, Any],
    profit_gate: dict[str, Any],
    inventory_state: dict[str, Any],
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    if inventory_state:
        token_balance = _first_float(inventory_state.get("token_balance_shares"))
        open_order_count = _first_float(inventory_state.get("open_order_count"), inventory_state.get("token_open_order_count"))
        inventory_status = str(inventory_state.get("current_inventory_status") or inventory_state.get("inventory_status") or "").upper()
        open_order_status = str(inventory_state.get("open_order_status") or "").upper()
        checks = {
            "inventory_report_present": True,
            "inventory_report_ready": inventory_state.get("status") == "INVENTORY_STATE_CLEAR",
            "report_fresh": _is_fresh(inventory_state, now=now, max_report_age_minutes=max_report_age_minutes),
            "token_balance_flat_or_dust": token_balance is not None and abs(token_balance) <= 0.001,
            "open_order_count_zero": open_order_count is not None and open_order_count == 0,
            "inventory_flat": inventory_status in {"FLAT", "ECONOMICALLY_CLOSED_WITH_DUST"},
            "open_order_clear": open_order_status == "NO_OPEN_ORDER",
            "partial_fill_unresolved_false": inventory_state.get("partial_fill_unresolved") is not True,
        }
        passed = all(checks.values())
        return _assertion(
            assert_id="INVENTORY_STATE_ASSERT",
            category="inventory",
            passed=passed,
            reason="INVENTORY_AND_OPEN_ORDERS_CLEAR" if passed else "INVENTORY_OR_OPEN_ORDER_STATE_NOT_CLEAR",
            blocking_reason=None if passed else "INVENTORY_STATE_NOT_CLEAR",
            details={
                **checks,
                "token_balance_shares": _round(token_balance),
                "open_order_count": _round(open_order_count),
                "current_inventory_status": inventory_status or None,
                "open_order_status": open_order_status or None,
                "inventory_blockers": inventory_state.get("blockers") or [],
            },
        )

    checks_payload = health.get("checks") if isinstance(health.get("checks"), dict) else {}
    target_account = checks_payload.get("target_account_state") if isinstance(checks_payload.get("target_account_state"), dict) else {}
    account_orders = checks_payload.get("account_open_orders") if isinstance(checks_payload.get("account_open_orders"), dict) else {}
    token_balance = _first_float(target_account.get("token_balance_shares"), profit_gate.get("position_shares"))
    open_order_count = _first_float(target_account.get("token_open_order_count"), account_orders.get("open_order_count"))
    inventory_status = str(profit_gate.get("current_inventory_status") or "").upper()
    open_order_status = str(profit_gate.get("open_order_status") or "").upper()
    checks = {
        "account_state_present": bool(target_account) or bool(profit_gate),
        "token_balance_flat_or_dust": token_balance is not None and abs(token_balance) <= 0.001,
        "open_order_count_zero": open_order_count is not None and open_order_count == 0,
        "profit_gate_inventory_flat": inventory_status in {"FLAT", "ECONOMICALLY_CLOSED_WITH_DUST"},
        "profit_gate_open_order_clear": open_order_status == "NO_OPEN_ORDER",
        "partial_fill_unresolved_false": profit_gate.get("partial_fill_unresolved") is not True,
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="INVENTORY_STATE_ASSERT",
        category="inventory",
        passed=passed,
        reason="INVENTORY_AND_OPEN_ORDERS_CLEAR" if passed else "INVENTORY_OR_OPEN_ORDER_STATE_NOT_CLEAR",
        blocking_reason=None if passed else "INVENTORY_STATE_NOT_CLEAR",
        details={
            **checks,
            "token_balance_shares": _round(token_balance),
            "open_order_count": _round(open_order_count),
            "current_inventory_status": inventory_status or None,
            "open_order_status": open_order_status or None,
        },
    )


def _fee_reconciliation_assertion(
    fee_reconciliation: dict[str, Any],
    *,
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    checks = {
        "fee_report_present": bool(fee_reconciliation),
        "fee_reconciliation_ready": fee_reconciliation.get("status") == "FEE_RECONCILIATION_READY",
        "maker_fee_model_present": fee_reconciliation.get("maker_fee_model_present") is True,
        "taker_fee_model_present": fee_reconciliation.get("taker_fee_model_present") is True,
        "projected_fee_known": fee_reconciliation.get("projected_fee_unknown") is False,
        "can_cover_fees": fee_reconciliation.get("can_cover_fees") is True,
        "reward_payout_mismatch_clear": fee_reconciliation.get("reward_payout_mismatch") is False,
        "report_fresh": _is_fresh(fee_reconciliation, now=now, max_report_age_minutes=max_report_age_minutes),
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="FEE_RECONCILIATION_ASSERT",
        category="fee_reconciliation",
        passed=passed,
        reason="FEES_AND_REWARD_RECONCILIATION_READY" if passed else "FEES_OR_REWARD_RECONCILIATION_NOT_PROVEN",
        blocking_reason=None if passed else "FEE_RECONCILIATION_NOT_PROVEN",
        details=checks,
    )


def _final_physical_assertion(
    *,
    health: dict[str, Any],
    network: dict[str, Any],
    final_physical: dict[str, Any],
    kill_switch: dict[str, Any],
    now: datetime,
    max_report_age_minutes: float,
) -> dict[str, Any]:
    if final_physical:
        physical_checks = final_physical.get("checks") if isinstance(final_physical.get("checks"), dict) else {}
        checks = {
            "final_physical_report_present": True,
            "final_physical_ready": final_physical.get("status") == "FINAL_PHYSICAL_READY",
            "report_fresh": _is_fresh(final_physical, now=now, max_report_age_minutes=max_report_age_minutes),
            "disk_free_pct_ok": physical_checks.get("disk_free_pct_ok") is True,
            "clock_skew_ok": physical_checks.get("clock_skew_ok") is True,
            "latency_ok": physical_checks.get("latency_ok") is True,
            "heartbeat_ok": physical_checks.get("heartbeat_ok") is True,
            "kill_switch_clear": physical_checks.get("kill_switch_clear") is True,
            "market_not_suspended": physical_checks.get("market_not_suspended") is True,
        }
        passed = all(checks.values())
        return _assertion(
            assert_id="FINAL_PHYSICAL_ASSERT",
            category="final_physical",
            passed=passed,
            reason="FINAL_PHYSICAL_PRECONDITIONS_READY" if passed else "FINAL_PHYSICAL_PRECONDITIONS_NOT_PROVEN",
            blocking_reason=None if passed else "FINAL_PHYSICAL_PRECONDITIONS_NOT_PROVEN",
            details={
                **checks,
                "final_physical_blockers": final_physical.get("blockers") or [],
            },
        )

    checks = {
        "live_api_health_healthy": health.get("healthy") is True and str(health.get("status") or "").upper() == "HEALTHY",
        "health_report_fresh": _is_fresh(health, now=now, max_report_age_minutes=max_report_age_minutes),
        "account_state_fresh": health.get("stale_process_possible") is not True,
        "latency_ok": network.get("latency_ok") is True,
        "heartbeat_ok": network.get("heartbeat_ok") is True,
        "kill_switch_clear": kill_switch.get("kill_switch_active") is False,
        "market_not_suspended": network.get("market_suspended") is False,
    }
    passed = all(checks.values())
    return _assertion(
        assert_id="FINAL_PHYSICAL_ASSERT",
        category="final_physical",
        passed=passed,
        reason="FINAL_PHYSICAL_PRECONDITIONS_READY" if passed else "FINAL_PHYSICAL_PRECONDITIONS_NOT_PROVEN",
        blocking_reason=None if passed else "FINAL_PHYSICAL_PRECONDITIONS_NOT_PROVEN",
        details=checks,
    )


def _assertion(
    *,
    assert_id: str,
    category: str,
    passed: bool,
    reason: str | None,
    blocking_reason: str | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "assert_id": assert_id,
        "category": category,
        "passed": bool(passed),
        "reason": reason,
        "blocking_reason": blocking_reason,
        "details": details,
    }


def _deposit_wallet_blocker(checks: dict[str, bool]) -> str:
    if not checks.get("report_present"):
        return "DEPOSIT_WALLET_REPORT_MISSING"
    if not checks.get("read_only"):
        return "DEPOSIT_WALLET_ADAPTER_NOT_READ_ONLY"
    if not checks.get("source_is_deposit_wallet"):
        return "DEPOSIT_WALLET_SOURCE_NOT_DEPOSIT_WALLET"
    if not checks.get("balance_present"):
        return "DEPOSIT_WALLET_BALANCE_MISSING"
    if not checks.get("balance_covers_required_buffer"):
        return "DEPOSIT_WALLET_BALANCE_BELOW_REQUIRED"
    if not checks.get("report_fresh"):
        return "DEPOSIT_WALLET_BALANCE_NOT_FRESH"
    return "DEPOSIT_WALLET_ADAPTER_ERROR"


def _market_context(
    *,
    health: dict[str, Any],
    profit_gate: dict[str, Any],
    market_microstructure: dict[str, Any],
    toxic_flow: dict[str, Any],
) -> dict[str, Any]:
    target_market = health.get("target_market") if isinstance(health.get("target_market"), dict) else {}
    checks = health.get("checks") if isinstance(health.get("checks"), dict) else {}
    checked_target = checks.get("target_market") if isinstance(checks.get("target_market"), dict) else {}
    merged: dict[str, Any] = {}
    for source in (target_market, checked_target, profit_gate.get("context") or {}, market_microstructure, toxic_flow):
        if isinstance(source, dict):
            merged.update({key: value for key, value in source.items() if value is not None})
    return merged


def _target_market_slug(*, health: dict[str, Any], profit_gate: dict[str, Any], market: dict[str, Any]) -> str | None:
    target = health.get("target_market") if isinstance(health.get("target_market"), dict) else {}
    return (
        target.get("market_slug")
        or market.get("market_slug")
        or profit_gate.get("canary_market_slug")
        or profit_gate.get("market_slug")
    )


def _is_fresh(report: dict[str, Any], *, now: datetime, max_report_age_minutes: float) -> bool:
    if not report:
        return False
    if report.get("stale_process_possible") is True:
        return False
    ts = _first_timestamp(
        report.get("generated_at_utc"),
        report.get("generated_ts"),
        report.get("timestamp_utc"),
        report.get("as_of_utc"),
    )
    if ts is None:
        return False
    return now - ts <= timedelta(minutes=max_report_age_minutes)


def _first_timestamp(*values: Any) -> datetime | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _price_tick_aligned(price: float | None, tick: float | None) -> bool:
    if price is None or tick is None or tick <= 0:
        return False
    units = price / tick
    return abs(units - round(units)) <= 1e-8


def _price_in_bounds(price: float | None) -> bool:
    return price is not None and 0.0 < price < 1.0


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _unique(items: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _one_line_verdict(status: str, blockers: list[str]) -> str:
    if status == LIVE_READY_APPROVED:
        return "LIVE_READY_APPROVED_REPORT_ONLY: all microstructure asserts passed; can_submit_order remains false here."
    preview = ", ".join(blockers[:5])
    suffix = "" if len(blockers) <= 5 else f" (+{len(blockers) - 5} more)"
    return f"LIVE_NOT_READY: {preview}{suffix}; can_submit_order=false."
