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
    max_live_risk_usdc: float = 40.0,
    live_cycles: int = 60,
    live_interval_sec: int = 30,
) -> dict[str, Any]:
    live_pnl = live_pnl or {}
    requested_risk = max(0.0, float(max_live_risk_usdc or 0.0))
    requires_human_approval = requested_risk > HUMAN_APPROVAL_RISK_LIMIT_USDC
    live_health = _live_health(live_pnl, max_live_risk_usdc=min(requested_risk, HUMAN_APPROVAL_RISK_LIMIT_USDC))
    summary_blockers = list(research_summary.get("live_ready_blockers") or [])
    live_canary_eligible_count = int(_float(research_summary.get("live_canary_eligible_count")))
    dry_run_focus_count = int(_float(research_summary.get("dry_run_focus_count")))
    scale_recommendation = str(research_summary.get("scale_recommendation") or "DO_NOT_SCALE")

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
        },
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

    if summary_blockers:
        decision.update(
            {
                "decision": "RUN_RESEARCH",
                "reason": "live readiness blockers remain: " + ",".join(summary_blockers),
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
        subprocess.run(decision["live_command"], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
