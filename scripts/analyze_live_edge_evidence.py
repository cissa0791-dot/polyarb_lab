from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze live edge evidence and build market filter intel.")
    parser.add_argument(
        "--evidence",
        type=str,
        default=str(ROOT / "data" / "reports" / "live_edge_observations.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "reports" / "live_edge_summary_latest.json"),
    )
    parser.add_argument(
        "--market-intel-out",
        type=str,
        default=str(ROOT / "data" / "reports" / "evidence_market_intel_latest.json"),
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_live_edge_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    observations = [row for row in rows if row.get("row_type") == "market_observation"]
    by_market: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for row in observations:
        slug = str(row.get("market_slug") or "")
        if not slug:
            continue
        by_market[slug] = row
        bucket = counts.setdefault(slug, {"paused": 0, "not_selected": 0, "rows": 0})
        bucket["rows"] += 1
        status = str(row.get("status") or "").upper()
        cancel_reason = str(row.get("last_cancel_reason") or "").upper()
        if status == "PAUSED":
            bucket["paused"] += 1
        if "NOT_SELECTED" in cancel_reason:
            bucket["not_selected"] += 1

    market_rows: list[dict[str, Any]] = []
    market_intel: dict[str, Any] = {}
    actual_reward_total = 0.0
    realized_spread_total = 0.0
    simulated_spread_total = 0.0
    verified_net_total = 0.0
    confirmed_verified_net_total = 0.0
    fill_rate_values: list[float] = []
    order_reject_count = 0
    simulated_profitable_market_count = 0
    actual_reward_confirmed_market_count = 0
    evidence_source_counts: dict[str, int] = {}

    for slug, row in by_market.items():
        actual_reward = _float(row.get("actual_reward_usdc"))
        raw_spread = _float(row.get("spread_realized_usdc"))
        evidence_source = str(row.get("evidence_source") or "UNKNOWN")
        simulated_fill = _truthy(row.get("simulated_fill")) or evidence_source == "DRY_RUN_SIMULATED"
        simulated_spread = _float(row.get("simulated_spread_usdc"))
        if simulated_fill and simulated_spread <= 0.0:
            simulated_spread = max(0.0, raw_spread)
        confirmed_spread = 0.0 if simulated_fill else raw_spread
        verified_net = _float(row.get("verified_net_window_usdc"), _float(row.get("net_after_reward_and_cost_usdc")))
        fill_rate = _fill_rate(row)
        confirmed_fill_rate = 0.0 if simulated_fill else fill_rate
        replay_confirmed = _truthy(row.get("replay_confirmed"))
        rejects = int(_float(row.get("order_reject_count")))
        churn = counts.get(slug, {})
        paused_count = int(churn.get("paused", 0))
        not_selected_count = int(churn.get("not_selected", 0))

        actual_reward_total += actual_reward
        realized_spread_total += confirmed_spread
        simulated_spread_total += simulated_spread
        verified_net_total += verified_net
        if not simulated_fill:
            confirmed_verified_net_total += verified_net
        fill_rate_values.append(confirmed_fill_rate)
        order_reject_count += rejects
        evidence_source_counts[evidence_source] = evidence_source_counts.get(evidence_source, 0) + 1
        if actual_reward > 0.0:
            actual_reward_confirmed_market_count += 1

        evidence_status = "NO_EVIDENCE"
        reasons: list[str] = []
        blocked = False
        allow = False
        has_confirmed_positive = actual_reward > 0.0 or confirmed_spread > 0.0 or confirmed_fill_rate > 0.0
        has_simulated_positive = simulated_fill and verified_net > 0.0 and (
            simulated_spread > 0.0 or raw_spread > 0.0 or fill_rate > 0.0
        )
        has_real_negative = has_confirmed_positive and verified_net < 0.0
        if has_simulated_positive and not has_confirmed_positive:
            reasons.append("SIMULATED_PROFIT_ONLY")
            reasons.append("NO_ACTUAL_REWARD_CONFIRMED")
        if verified_net <= 0.0 and actual_reward == 0.0 and confirmed_spread == 0.0 and confirmed_fill_rate == 0.0:
            reasons.append("NO_REAL_EDGE_EVIDENCE_YET")
        if verified_net <= -0.05 and actual_reward == 0.0:
            blocked = True
            evidence_status = "BLACKLIST_CANDIDATE"
            reasons.append("VERIFIED_NET_WINDOW_BELOW_NEG_0_05")
        if rejects > 0:
            blocked = True
            evidence_status = "BLACKLIST_CANDIDATE"
            reasons.append("ORDER_REJECTS_PRESENT")
        if paused_count >= 3 or not_selected_count >= 3:
            reasons.append("REPEATED_COOLDOWN_OR_NOT_SELECTED_CHURN")
            if has_real_negative:
                blocked = True
                evidence_status = "BLACKLIST_CANDIDATE"
                reasons.append("CHURN_WITH_REAL_NEGATIVE_EVIDENCE")
        if verified_net > 0.01 and confirmed_fill_rate > 0.0 and (actual_reward > 0.0 or confirmed_spread > 0.0):
            evidence_status = "WHITELIST_CANDIDATE"
            allow = True
            blocked = False
            reasons.append("VERIFIED_POSITIVE_WITH_REAL_EVIDENCE")
        elif not blocked and has_confirmed_positive:
            evidence_status = "NO_EVIDENCE"
            reasons.append("REAL_ACTIVITY_BUT_NOT_ENOUGH_FOR_WHITELIST")

        risk_score = _risk_score(blocked=blocked, verified_net=verified_net, rejects=rejects, fill_rate=confirmed_fill_rate)
        market_summary = {
            "market_slug": slug,
            "event_slug": row.get("event_slug"),
            "token_id": row.get("token_id"),
            "status": row.get("status"),
            "actual_reward_usdc": round(actual_reward, 6),
            "realized_spread_usdc": round(confirmed_spread, 6),
            "raw_realized_spread_usdc": round(raw_spread, 6),
            "simulated_spread_usdc": round(simulated_spread, 6),
            "verified_net_usdc": round(verified_net, 6),
            "fill_rate": round(confirmed_fill_rate, 6),
            "raw_fill_rate": round(fill_rate, 6),
            "order_reject_count": rejects,
            "evidence_source": evidence_source,
            "simulated_fill": simulated_fill,
            "replay_confirmed": replay_confirmed,
            "evidence_status": evidence_status,
            "evidence_reasons": reasons,
        }
        market_rows.append(market_summary)
        market_intel[slug] = {
            "blocked": blocked,
            "allow": allow,
            "risk_score": risk_score,
            "evidence_status": evidence_status,
            "evidence_reasons": reasons,
            "verified_net_window_usdc": round(verified_net, 6),
            "actual_reward_window_usdc": round(actual_reward, 6),
            "realized_spread_window_usdc": round(confirmed_spread, 6),
            "raw_realized_spread_window_usdc": round(raw_spread, 6),
            "simulated_spread_window_usdc": round(simulated_spread, 6),
            "fill_rate_window": round(confirmed_fill_rate, 6),
            "raw_fill_rate_window": round(fill_rate, 6),
            "evidence_source": evidence_source,
            "simulated_fill": simulated_fill,
            "replay_confirmed": replay_confirmed,
            "order_reject_count": rejects,
            "paused_count": paused_count,
            "not_selected_count": not_selected_count,
        }
        if has_simulated_positive and not has_confirmed_positive:
            simulated_profitable_market_count += 1

    best_markets = sorted(market_rows, key=lambda row: row["verified_net_usdc"], reverse=True)[:10]
    worst_markets = sorted(market_rows, key=lambda row: row["verified_net_usdc"])[:10]
    blacklist_candidates = [row for row in market_rows if row["evidence_status"] == "BLACKLIST_CANDIDATE"]
    whitelist_candidates = [row for row in market_rows if row["evidence_status"] == "WHITELIST_CANDIDATE"]
    fill_rate = sum(fill_rate_values) / len(fill_rate_values) if fill_rate_values else 0.0
    profitable_market_count = sum(
        1
        for row in market_rows
        if row["verified_net_usdc"] > 0.0 and (row["actual_reward_usdc"] > 0.0 or row["realized_spread_usdc"] > 0.0 or row["fill_rate"] > 0.0)
    )
    scale_recommendation = _scale_recommendation(
        verified_net_total=confirmed_verified_net_total,
        profitable_market_count=profitable_market_count,
        fill_rate=fill_rate,
        order_reject_count=order_reject_count,
    )

    return {
        "report_type": "live_edge_summary",
        "market_count": len(market_rows),
        "profitable_market_count": profitable_market_count,
        "simulated_profitable_market_count": simulated_profitable_market_count,
        "actual_reward_confirmed_market_count": actual_reward_confirmed_market_count,
        "actual_reward_total": round(actual_reward_total, 6),
        "realized_spread_total": round(realized_spread_total, 6),
        "simulated_spread_total": round(simulated_spread_total, 6),
        "verified_net_total": round(verified_net_total, 6),
        "confirmed_verified_net_total": round(confirmed_verified_net_total, 6),
        "fill_rate": round(fill_rate, 6),
        "order_reject_count": order_reject_count,
        "evidence_source_counts": dict(sorted(evidence_source_counts.items())),
        "best_markets": best_markets,
        "worst_markets": worst_markets,
        "blacklist_candidates": blacklist_candidates,
        "whitelist_candidates": whitelist_candidates,
        "scale_recommendation": scale_recommendation,
        "market_intel": {
            "report_type": "evidence_market_intel",
            "markets": market_intel,
        },
    }


def _scale_recommendation(
    *,
    verified_net_total: float,
    profitable_market_count: int,
    fill_rate: float,
    order_reject_count: int,
) -> str:
    if order_reject_count > 0 or verified_net_total <= -0.05:
        return "DO_NOT_SCALE"
    if profitable_market_count >= 2 and fill_rate > 0.0:
        return "ALLOW_SCALE_TO_2_MARKETS"
    if profitable_market_count >= 1 and fill_rate > 0.0:
        return "ALLOW_CANARY_ONLY"
    if verified_net_total > -0.05:
        return "ALLOW_DRY_RUN_FOCUS"
    return "DO_NOT_SCALE"


def _fill_rate(row: dict[str, Any]) -> float:
    explicit = row.get("fill_rate_window")
    if explicit is not None:
        return max(0.0, _float(explicit))
    filled = _float(row.get("bid_order_filled_size")) + _float(row.get("ask_order_filled_size"))
    return 1.0 if filled > 0.0 else 0.0


def _risk_score(*, blocked: bool, verified_net: float, rejects: int, fill_rate: float) -> float:
    if blocked or rejects > 0:
        return 1.0
    score = 0.25
    if verified_net < 0.0:
        score += min(0.5, abs(verified_net) * 4.0)
    if fill_rate <= 0.0:
        score += 0.1
    return round(min(0.94, score), 6)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def main() -> None:
    args = parse_args()
    summary = build_live_edge_summary(load_jsonl(args.evidence))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    intel_path = Path(args.market_intel_out)
    intel_path.parent.mkdir(parents=True, exist_ok=True)
    intel_path.write_text(
        json.dumps(summary["market_intel"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ("market_count", "profitable_market_count", "verified_net_total", "scale_recommendation")}, indent=2))


if __name__ == "__main__":
    main()
