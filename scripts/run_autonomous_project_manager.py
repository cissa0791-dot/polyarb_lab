from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HUMAN_APPROVAL_RISK_LIMIT_USDC = 50.0
CANARY_PER_MARKET_CAP_USDC = 20.0
MICRO_PROBE_RISK_USDC = 20.0
CANARY_STOP_LOSS_USDC = 1.0
MICRO_PROBE_REPLAY_TOTAL_FLOOR_USDC = -0.10
MICRO_PROBE_REPLAY_AVG_FLOOR_USDC = -0.0025
LIVE_PROBE_ROOT = ROOT / "data" / "reports" / "live_probe_runs"
DEFAULT_LIVE_PNL = ROOT / "data" / "reports" / "auto_trade_profit_pnl_latest.json"
LIVE_RATE_LIMIT_COOLDOWN_MINUTES = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous research/live canary decision manager.")
    parser.add_argument(
        "--summary",
        default=str(ROOT / "data" / "reports" / "research_pipeline_summary_latest.json"),
        help="Research pipeline summary to evaluate.",
    )
    parser.add_argument(
        "--live-pnl",
        default=str(DEFAULT_LIVE_PNL),
        help="Latest live/canary PnL report, if one exists.",
    )
    parser.add_argument(
        "--profit-drivers",
        default=str(ROOT / "data" / "reports" / "research_profit_drivers_latest.json"),
        help="Latest research profit-driver report.",
    )
    parser.add_argument(
        "--replay",
        default=str(ROOT / "data" / "reports" / "research_replay_report_latest.json"),
        help="Latest orderbook replay report.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "reports" / "autonomous_decision_latest.json"),
        help="Where to write the autonomous decision report.",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "print-live-command", "execute-live-canary"],
        default="evaluate",
        help="evaluate writes the decision only; execute-live-canary runs the gated live command when allowed.",
    )
    parser.add_argument("--max-live-risk-usdc", type=float, default=40.0)
    parser.add_argument("--live-cycles", type=int, default=60)
    parser.add_argument("--live-interval-sec", type=int, default=30)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_autonomous_decision(
    *,
    research_summary: dict[str, Any],
    live_pnl: dict[str, Any] | None = None,
    profit_drivers: dict[str, Any] | None = None,
    replay_report: dict[str, Any] | None = None,
    max_live_risk_usdc: float = 40.0,
    live_cycles: int = 60,
    live_interval_sec: int = 30,
) -> dict[str, Any]:
    live_pnl = live_pnl or {}
    profit_drivers = profit_drivers or {}
    replay_report = replay_report or {}
    requested_risk = max(0.0, float(max_live_risk_usdc or 0.0))
    requires_human_approval = requested_risk > HUMAN_APPROVAL_RISK_LIMIT_USDC
    live_health = _live_health(live_pnl, max_live_risk_usdc=min(requested_risk, HUMAN_APPROVAL_RISK_LIMIT_USDC))
    replay_health = _replay_health(replay_report)
    profit_quality = _profit_driver_quality(profit_drivers, research_summary)
    research_policy = _research_policy(research_summary, profit_quality, replay_health)
    summary_blockers = list(research_summary.get("live_ready_blockers") or [])
    live_canary_eligible_count = int(_float(research_summary.get("live_canary_eligible_count")))
    dry_run_focus_count = int(_float(research_summary.get("dry_run_focus_count")))
    scale_recommendation = str(research_summary.get("scale_recommendation") or "DO_NOT_SCALE")
    live_blockers = _live_data_blockers(
        live_canary_eligible_count=live_canary_eligible_count,
        profit_quality=profit_quality,
        replay_health=replay_health,
    )

    decision = {
        "report_type": "autonomous_project_manager_decision",
        "decision": "WAIT",
        "reason": "no actionable research summary yet",
        "target_live_markets": 0,
        "max_live_risk_usdc": 0.0,
        "requires_human_approval": requires_human_approval,
        "can_execute_live": False,
        "live_command": [],
        "stop_rules": {
            "verified_live_net_stop_usdc": -CANARY_STOP_LOSS_USDC,
            "order_reject_count_stop": 1,
            "max_humanless_live_risk_usdc": HUMAN_APPROVAL_RISK_LIMIT_USDC,
        },
        "inputs": {
            "scale_recommendation": scale_recommendation,
            "live_canary_eligible_count": live_canary_eligible_count,
            "dry_run_focus_count": dry_run_focus_count,
            "live_ready_blockers": summary_blockers,
            "live_health": live_health,
            "profit_driver_quality": profit_quality,
            "replay": replay_health,
        },
        "research_policy": research_policy,
    }

    if requires_human_approval:
        decision.update(
            {
                "decision": "HOLD_HUMAN_APPROVAL_REQUIRED",
                "reason": "requested live risk exceeds autonomous approval limit",
            }
        )
        return decision

    if live_health["pause_required"]:
        decision.update(
            {
                "decision": "PAUSE_LIVE",
                "reason": live_health["pause_reason"],
            }
        )
        return decision

    if bool(research_summary.get("contaminated_state_detected")):
        decision.update(
            {
                "decision": "HOLD_NO_LIVE",
                "reason": "research state contamination detected",
            }
        )
        return decision

    if bool(research_summary.get("partial")):
        decision.update(
            {
                "decision": "RUN_RESEARCH",
                "reason": "latest research is partial; rerun or merge complete data first",
            }
        )
        return decision

    if summary_blockers or live_blockers:
        if _micro_live_probe_allowed(
            research_summary=research_summary,
            profit_quality=profit_quality,
            replay_health=replay_health,
            summary_blockers=summary_blockers,
            live_blockers=live_blockers,
            dry_run_focus_count=dry_run_focus_count,
        ):
            live_risk = min(requested_risk, MICRO_PROBE_RISK_USDC, CANARY_PER_MARKET_CAP_USDC)
            live_paths = _live_run_paths("micro_probe", research_summary)
            command = _live_canary_command(
                target_live_markets=1,
                live_risk_usdc=live_risk,
                cycles=live_cycles,
                interval_sec=live_interval_sec,
                live_paths=live_paths,
            )
            decision.update(
                {
                    "decision": "START_MICRO_LIVE_PROBE",
                    "reason": "dry-run edge is positive but actual reward requires a capped live probe",
                    "target_live_markets": 1,
                    "max_live_risk_usdc": live_risk,
                    "can_execute_live": True,
                    "live_command": command,
                    "fresh_live_state": True,
                    "live_state_path": str(live_paths["state"]),
                    "live_pnl_path": str(live_paths["pnl"]),
                    "live_edge_evidence_path": str(live_paths["edge_evidence"]),
                    "live_orderbook_snapshot_path": str(live_paths["orderbook_snapshots"]),
                }
            )
            return decision
        blockers = summary_blockers + live_blockers
        decision.update(
            {
                "decision": "RUN_RESEARCH",
                "reason": "live readiness blockers remain: " + ",".join(blockers),
            }
        )
        return decision

    target_live_markets = 0
    if scale_recommendation == "ALLOW_SCALE_TO_2_MARKETS" and live_canary_eligible_count >= 2 and live_health["healthy_live_seen"]:
        target_live_markets = 2
        next_decision = "SCALE_LIVE_TO_2_MARKETS"
        reason = "two confirmed candidates and existing live canary is healthy"
    elif scale_recommendation in {"ALLOW_CANARY_ONLY", "ALLOW_SCALE_TO_2_MARKETS"} and live_canary_eligible_count >= 1:
        target_live_markets = 1
        next_decision = "START_LIVE_CANARY"
        reason = "confirmed candidate available for autonomous canary"
    elif dry_run_focus_count > 0:
        decision.update(
            {
                "decision": "RUN_RESEARCH",
                "reason": "dry-run focus candidates exist but live gate is not satisfied",
            }
        )
        return decision
    else:
        decision.update(
            {
                "decision": "WAIT",
                "reason": "no dry-run focus or live-ready candidates",
            }
        )
        return decision

    live_risk = min(requested_risk, CANARY_PER_MARKET_CAP_USDC * target_live_markets)
    live_paths = _live_run_paths(next_decision.lower(), research_summary)
    command = _live_canary_command(
        target_live_markets=target_live_markets,
        live_risk_usdc=live_risk,
        cycles=live_cycles,
        interval_sec=live_interval_sec,
        live_paths=live_paths,
    )
    decision.update(
        {
            "decision": next_decision,
            "reason": reason,
            "target_live_markets": target_live_markets,
            "max_live_risk_usdc": live_risk,
            "can_execute_live": True,
            "live_command": command,
            "fresh_live_state": True,
            "live_state_path": str(live_paths["state"]),
            "live_pnl_path": str(live_paths["pnl"]),
            "live_edge_evidence_path": str(live_paths["edge_evidence"]),
            "live_orderbook_snapshot_path": str(live_paths["orderbook_snapshots"]),
        }
    )
    return decision


def _live_health(live_pnl: dict[str, Any], *, max_live_risk_usdc: float) -> dict[str, Any]:
    summary = live_pnl.get("summary") if isinstance(live_pnl.get("summary"), dict) else {}
    markets = live_pnl.get("markets") if isinstance(live_pnl.get("markets"), list) else []
    mode = str(summary.get("mode") or "").upper()
    verified_net = _float(summary.get("verified_net_after_reward_and_cost_usdc"))
    account_inventory = _float(summary.get("account_inventory_usdc"))
    account_open_buy = _float(summary.get("account_open_buy_usdc"))
    open_buy_count = int(_float(summary.get("account_open_buy_order_count")))
    order_reject_count = sum(int(_float(market.get("order_reject_count"))) for market in markets if isinstance(market, dict))
    account_sync_error = str(summary.get("account_order_sync_error") or "")
    market_errors = [
        str(market.get("last_order_error") or "")
        for market in markets
        if isinstance(market, dict) and market.get("last_order_error")
    ]
    rate_limit_detected = _is_live_rate_limit_error(account_sync_error) or any(
        _is_live_rate_limit_error(error) for error in market_errors
    )
    generated_ts = _parse_utc_ts(summary.get("generated_ts") or live_pnl.get("generated_ts"))
    rate_limit_cooldown_until = (
        generated_ts + timedelta(minutes=LIVE_RATE_LIMIT_COOLDOWN_MINUTES)
        if generated_ts is not None and rate_limit_detected
        else None
    )
    rate_limit_cooldown_active = (
        rate_limit_detected
        if rate_limit_cooldown_until is None
        else datetime.now(timezone.utc) < rate_limit_cooldown_until
    )

    pause_reason = None
    if mode == "LIVE" and rate_limit_detected and rate_limit_cooldown_active:
        pause_reason = "CLOB_RATE_LIMIT"
    elif mode == "LIVE" and verified_net <= -CANARY_STOP_LOSS_USDC:
        pause_reason = "VERIFIED_LIVE_NET_STOP"
    elif mode == "LIVE" and order_reject_count > 0:
        pause_reason = "ORDER_REJECT_STOP"
    elif mode == "LIVE" and account_inventory > max_live_risk_usdc + 1e-9:
        pause_reason = "INVENTORY_CAP_STOP"
    elif mode == "LIVE" and account_open_buy > max_live_risk_usdc + 1e-9:
        pause_reason = "OPEN_BUY_CAP_STOP"

    return {
        "mode": mode or "UNKNOWN",
        "healthy_live_seen": mode == "LIVE" and pause_reason is None and verified_net >= 0.0,
        "pause_required": pause_reason is not None,
        "pause_reason": pause_reason,
        "verified_net_after_cost_usdc": round(verified_net, 6),
        "account_inventory_usdc": round(account_inventory, 6),
        "account_open_buy_usdc": round(account_open_buy, 6),
        "account_open_buy_order_count": open_buy_count,
        "order_reject_count": order_reject_count,
        "account_order_sync_error": account_sync_error or None,
        "rate_limit_detected": rate_limit_detected,
        "rate_limit_cooldown_active": rate_limit_cooldown_active,
        "rate_limit_cooldown_minutes": LIVE_RATE_LIMIT_COOLDOWN_MINUTES if rate_limit_detected else 0,
        "rate_limit_cooldown_until_ts": rate_limit_cooldown_until.isoformat()
        if rate_limit_cooldown_until is not None
        else None,
    }


def _replay_health(replay_report: dict[str, Any]) -> dict[str, Any]:
    total_net = _float(replay_report.get("total_net_pnl_usdc"))
    replayed_count = int(_float(replay_report.get("replayed_market_count")))
    profitable_count = int(_float(replay_report.get("profitable_market_count")))
    market_count = int(_float(replay_report.get("market_count")))
    materially_negative = replayed_count > 0 and total_net < -0.01
    return {
        "market_count": market_count,
        "replayed_market_count": replayed_count,
        "profitable_market_count": profitable_count,
        "total_net_pnl_usdc": round(total_net, 6),
        "materially_negative": materially_negative,
    }


def _profit_driver_quality(
    profit_drivers: dict[str, Any],
    research_summary: dict[str, Any],
) -> dict[str, Any]:
    candidates = profit_drivers.get("live_canary_candidates")
    if not isinstance(candidates, list):
        candidates = []
    drivers = profit_drivers.get("top_profit_drivers")
    if not isinstance(drivers, list):
        drivers = profit_drivers.get("markets")
    if not isinstance(drivers, list):
        drivers = []
    totals = profit_drivers.get("totals") if isinstance(profit_drivers.get("totals"), dict) else {}
    if not totals and isinstance(profit_drivers.get("profit_totals"), dict):
        totals = profit_drivers["profit_totals"]

    confirmed_candidate_count = sum(1 for row in candidates if _is_confirmed_profit_row(row))
    simulated_candidate_count = sum(1 for row in drivers if _is_simulated_profit_row(row))
    positive_driver_count = sum(
        1
        for row in drivers
        if _float(
            row.get(
                "verified_net_usdc",
                row.get("latest_verified_net_usdc", row.get("verified_net_window_usdc")),
            )
        )
        > 0
    )
    confirmed_summary_count = int(_float(research_summary.get("live_canary_eligible_count")))
    actual_reward_total = _float(totals.get("actual_reward_usdc", research_summary.get("actual_reward_total")))
    realized_spread_total = _float(totals.get("realized_spread_usdc", research_summary.get("realized_spread_total")))
    simulated_only = False
    if positive_driver_count > 0 and confirmed_candidate_count <= 0:
        simulated_only = simulated_candidate_count > 0 or actual_reward_total <= 0.0
    if confirmed_summary_count > 0 and confirmed_candidate_count <= 0 and profit_drivers:
        simulated_only = True

    return {
        "confirmed_candidate_count": max(confirmed_candidate_count, confirmed_summary_count if not profit_drivers else 0),
        "profit_driver_candidate_count": len(candidates),
        "positive_driver_count": positive_driver_count,
        "simulated_positive_driver_count": simulated_candidate_count,
        "actual_reward_total_usdc": round(actual_reward_total, 6),
        "realized_spread_total_usdc": round(realized_spread_total, 6),
        "simulated_only": simulated_only,
        "run_id": profit_drivers.get("run_id"),
        "expected_run_id": research_summary.get("run_id"),
        "run_id_matches": _run_id_matches(profit_drivers.get("run_id"), research_summary.get("run_id")),
    }


def _is_confirmed_profit_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    quality = str(row.get("profit_quality") or row.get("evidence_source") or "").upper()
    if quality in {"CONFIRMED", "ACTUAL_REWARD", "REALIZED_SPREAD", "ACTUAL_REWARD_CONFIRMED", "CONFIRMED_SPREAD"}:
        return True
    if _float(row.get("actual_reward_usdc", row.get("actual_reward_window_usdc"))) > 0:
        return True
    if _float(row.get("realized_spread_usdc", row.get("spread_realized_usdc"))) > 0 and not _is_simulated_profit_row(row):
        return True
    return False


def _is_simulated_profit_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    quality = str(row.get("profit_quality") or row.get("evidence_source") or "").upper()
    if "SIMULATED" in quality:
        return True
    return bool(row.get("simulated_fill"))


def _live_data_blockers(
    *,
    live_canary_eligible_count: int,
    profit_quality: dict[str, Any],
    replay_health: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if replay_health["materially_negative"]:
        blockers.append("REPLAY_NET_NEGATIVE")
    if profit_quality["expected_run_id"] and not profit_quality["run_id_matches"]:
        blockers.append("PROFIT_DRIVER_RUN_MISMATCH")
    if live_canary_eligible_count > 0 and profit_quality["simulated_only"]:
        blockers.append("SIMULATED_ONLY_PROFIT_DRIVER")
    if live_canary_eligible_count > 0 and profit_quality["confirmed_candidate_count"] <= 0:
        blockers.append("NO_CONFIRMED_PROFIT_DRIVER")
    return blockers


def _micro_live_probe_allowed(
    *,
    research_summary: dict[str, Any],
    profit_quality: dict[str, Any],
    replay_health: dict[str, Any],
    summary_blockers: list[str],
    live_blockers: list[str],
    dry_run_focus_count: int,
) -> bool:
    allowed_summary_blockers = {
        "NO_LIVE_CANARY_ELIGIBLE",
        "SIMULATED_PROFIT_ONLY",
        "NO_ACTUAL_REWARD_CONFIRMED",
        "REPLAY_NOT_CONFIRMED",
    }
    allowed_live_blockers = {"REPLAY_NET_NEGATIVE"}
    if any(blocker not in allowed_summary_blockers for blocker in summary_blockers):
        return False
    if any(blocker not in allowed_live_blockers for blocker in live_blockers):
        return False
    if dry_run_focus_count <= 0:
        return False
    positive_driver_count = int(_float(profit_quality.get("positive_driver_count")))
    simulated_profitable_count = int(_float(research_summary.get("simulated_profitable_market_count")))
    if positive_driver_count <= 0 and _float(research_summary.get("verified_net_total")) <= 0.0:
        return False
    if max(positive_driver_count, simulated_profitable_count) < 2:
        return False
    if int(_float(research_summary.get("actual_reward_confirmed_market_count"))) > 0:
        return False
    replayed_count = max(1, int(_float(replay_health.get("replayed_market_count"))))
    replay_total = _float(replay_health.get("total_net_pnl_usdc"))
    replay_avg = replay_total / replayed_count
    if replay_total < MICRO_PROBE_REPLAY_TOTAL_FLOOR_USDC:
        return False
    if replay_avg < MICRO_PROBE_REPLAY_AVG_FLOOR_USDC:
        return False
    return True


def _run_id_matches(profit_run_id: Any, summary_run_id: Any) -> bool:
    if not summary_run_id:
        return True
    return bool(profit_run_id) and str(profit_run_id) == str(summary_run_id)


def _research_policy(
    research_summary: dict[str, Any],
    profit_quality: dict[str, Any],
    replay_health: dict[str, Any],
) -> dict[str, Any]:
    focus_count = int(_float(research_summary.get("dry_run_focus_count")))
    selected_positive = profit_quality["positive_driver_count"] > 0 or _float(research_summary.get("verified_net_total")) > 0
    single_market_positive = int(_float(research_summary.get("profitable_market_count"))) <= 1 and selected_positive
    should_broaden = focus_count > 0 and single_market_positive
    if replay_health["materially_negative"]:
        should_broaden = True
    return {
        "next_research_mode": "BROADEN_DRY_RUN_ONLY" if should_broaden else "STANDARD_RESEARCH",
        "recommended_max_selected_markets": 3,
        "recommended_dry_run_per_market_cap_usdc": 80.0 if should_broaden else 40.0,
        "recommended_cycles": 120 if should_broaden else 240,
        "recommended_interval_sec": 20 if should_broaden else 30,
        "live_cap_unchanged_usdc": CANARY_PER_MARKET_CAP_USDC,
        "reason": "single-market or replay-blocked evidence; broaden dry-run only"
        if should_broaden
        else "standard evidence collection",
    }


def _live_run_paths(kind: str, research_summary: dict[str, Any]) -> dict[str, Path]:
    raw_run_id = str(research_summary.get("run_id") or "").strip()
    if not raw_run_id:
        raw_run_id = datetime.now(timezone.utc).strftime("manual-%Y%m%dT%H%M%SZ")
    safe_run_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in raw_run_id)
    safe_kind = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in kind.strip() or "live_probe")
    run_dir = LIVE_PROBE_ROOT / f"{safe_run_id}-{safe_kind}"
    return {
        "dir": run_dir,
        "state": run_dir / "auto_trade_state.json",
        "pnl": run_dir / "auto_trade_pnl.json",
        "edge_evidence": run_dir / "live_edge_observations.jsonl",
        "orderbook_snapshots": run_dir / "live_orderbook_snapshots.jsonl",
    }


def _is_live_rate_limit_error(value: Any) -> bool:
    text = str(value or "").lower()
    if not text:
        return False
    markers = (
        "cloudflare",
        "error code: 1015",
        " 1015",
        "rate limited",
        "rate-limit",
        "too many requests",
        "http 429",
        "status code 429",
    )
    return any(marker in text for marker in markers)


def _parse_utc_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def latest_live_probe_pnl(root: Path = LIVE_PROBE_ROOT) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.glob("*/auto_trade_pnl.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_live_pnl_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested != DEFAULT_LIVE_PNL:
        return requested
    probe = latest_live_probe_pnl()
    if probe is None:
        return requested
    if not requested.exists() or probe.stat().st_mtime >= requested.stat().st_mtime:
        return probe
    return requested


def _live_canary_command(
    *,
    target_live_markets: int,
    live_risk_usdc: float,
    cycles: int,
    interval_sec: int,
    live_paths: dict[str, Path],
) -> list[str]:
    per_market_cap = min(CANARY_PER_MARKET_CAP_USDC, live_risk_usdc / max(1, target_live_markets))
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_auto_trade_profit.py"),
        "--live",
        "--cycles",
        str(max(1, int(cycles))),
        "--interval-sec",
        str(max(1, int(interval_sec))),
        "--capital",
        f"{live_risk_usdc:.2f}",
        "--per-market-cap",
        f"{per_market_cap:.2f}",
        "--max-markets",
        str(max(1, int(target_live_markets))),
        "--max-total-inventory-usdc",
        f"{live_risk_usdc:.2f}",
        "--max-total-open-buy-usdc",
        f"{live_risk_usdc:.2f}",
        "--max-account-open-buy-orders",
        str(max(1, int(target_live_markets))),
        "--strategy-set",
        "inventory_manager,reward_mm",
        "--action-mode",
        "optimal",
        "--quote-search-mode",
        "best_ev",
        "--scale-mode",
        "evidence_gated",
        "--risk-profile",
        "canary",
        "--profit-evidence-mode",
        "strict",
        "--max-verified-drawdown-usdc",
        f"{CANARY_STOP_LOSS_USDC:.2f}",
        "--max-daily-loss",
        f"{CANARY_STOP_LOSS_USDC:.2f}",
        "--max-order-rejects-per-hour",
        "1",
        "--state-path",
        str(live_paths["state"]),
        "--pnl-path",
        str(live_paths["pnl"]),
        "--reset-state",
        "--edge-evidence-path",
        str(live_paths["edge_evidence"]),
        "--record-orderbook-snapshots",
        "--orderbook-snapshot-path",
        str(live_paths["orderbook_snapshots"]),
        "--orderbook-snapshot-max-markets",
        "20",
        "--target-inventory-usdc-per-market",
        f"{per_market_cap / 2.0:.2f}",
        "--max-inventory-usdc-per-market",
        f"{per_market_cap:.2f}",
        "--min-live-order-size-shares",
        "5",
        "--inventory-policy",
        "auto",
        "--enable-evidence-market-filter",
        "--evidence-market-intel-path",
        str(ROOT / "data" / "reports" / "research_evidence_market_intel_latest.json"),
        "--enable-market-intel-filter",
        "--market-intel-path",
        str(ROOT / "data" / "reports" / "research_market_intel_latest.json"),
        "--event-limit",
        "1000",
        "--market-limit",
        "2000",
        "--verbose",
    ]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    live_pnl_path = resolve_live_pnl_path(args.live_pnl)
    decision = build_autonomous_decision(
        research_summary=load_json(args.summary),
        live_pnl=load_json(live_pnl_path),
        profit_drivers=load_json(args.profit_drivers),
        replay_report=load_json(args.replay),
        max_live_risk_usdc=args.max_live_risk_usdc,
        live_cycles=args.live_cycles,
        live_interval_sec=args.live_interval_sec,
    )
    decision.setdefault("inputs", {})["live_pnl_path"] = str(live_pnl_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: decision[key] for key in ("decision", "reason", "target_live_markets", "max_live_risk_usdc", "can_execute_live")}, indent=2))
    if args.mode == "print-live-command" and decision["live_command"]:
        print("\n" + " ".join(decision["live_command"]))
    if args.mode == "execute-live-canary":
        if not decision["can_execute_live"]:
            raise SystemExit(f"live execution blocked: {decision['reason']}")
        live_dir = decision.get("live_state_path")
        if live_dir:
            Path(str(live_dir)).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_polymarket_auth.py")], cwd=str(ROOT), check=True)
        subprocess.run(decision["live_command"], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
