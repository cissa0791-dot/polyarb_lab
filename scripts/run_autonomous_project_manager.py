from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HUMAN_APPROVAL_RISK_LIMIT_USDC = 50.0
CANARY_PER_MARKET_CAP_USDC = 20.0
CANARY_STOP_LOSS_USDC = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous research/live canary decision manager.")
    parser.add_argument(
        "--summary",
        default=str(ROOT / "data" / "reports" / "research_pipeline_summary_latest.json"),
        help="Research pipeline summary to evaluate.",
    )
    parser.add_argument(
        "--live-pnl",
        default=str(ROOT / "data" / "reports" / "auto_trade_profit_pnl_latest.json"),
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
    command = _live_canary_command(
        target_live_markets=target_live_markets,
        live_risk_usdc=live_risk,
        cycles=live_cycles,
        interval_sec=live_interval_sec,
    )
    decision.update(
        {
            "decision": next_decision,
            "reason": reason,
            "target_live_markets": target_live_markets,
            "max_live_risk_usdc": live_risk,
            "can_execute_live": True,
            "live_command": command,
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

    pause_reason = None
    if mode == "LIVE" and verified_net <= -CANARY_STOP_LOSS_USDC:
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
        "live_cap_unchanged_usdc": CANARY_PER_MARKET_CAP_USDC,
        "reason": "single-market or replay-blocked evidence; broaden dry-run only"
        if should_broaden
        else "standard evidence collection",
    }


def _live_canary_command(*, target_live_markets: int, live_risk_usdc: float, cycles: int, interval_sec: int) -> list[str]:
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
        "--target-inventory-usdc-per-market",
        f"{per_market_cap / 2.0:.2f}",
        "--max-inventory-usdc-per-market",
        f"{per_market_cap:.2f}",
        "--min-live-order-size-shares",
        "5",
        "--inventory-policy",
        "balanced",
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
    decision = build_autonomous_decision(
        research_summary=load_json(args.summary),
        live_pnl=load_json(args.live_pnl),
        profit_drivers=load_json(args.profit_drivers),
        replay_report=load_json(args.replay),
        max_live_risk_usdc=args.max_live_risk_usdc,
        live_cycles=args.live_cycles,
        live_interval_sec=args.live_interval_sec,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: decision[key] for key in ("decision", "reason", "target_live_markets", "max_live_risk_usdc", "can_execute_live")}, indent=2))
    if args.mode == "print-live-command" and decision["live_command"]:
        print("\n" + " ".join(decision["live_command"]))
    if args.mode == "execute-live-canary":
        if not decision["can_execute_live"]:
            raise SystemExit(f"live execution blocked: {decision['reason']}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_polymarket_auth.py")], cwd=str(ROOT), check=True)
        subprocess.run(decision["live_command"], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
