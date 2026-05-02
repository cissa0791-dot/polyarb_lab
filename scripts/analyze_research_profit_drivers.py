from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain research profit drivers and blockers from evidence rows.")
    parser.add_argument(
        "--evidence",
        default=str(ROOT / "data" / "reports" / "research_edge_observations_latest.jsonl"),
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--markdown-out", default=None, help="Optional Markdown report path.")
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def analyze_profit_drivers(rows: Iterable[dict[str, Any]], *, top: int = 10) -> dict[str, Any]:
    market_rows = [row for row in rows if row.get("row_type") == "market_observation"]
    cycle_rows = [row for row in rows if row.get("row_type") == "cycle_summary"]
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in market_rows:
        slug = str(row.get("market_slug") or "")
        cycle = _int(row.get("cycle_index"))
        if slug:
            by_market[slug].append(row)
        if cycle is not None:
            by_cycle[cycle].append(row)

    cycle_curve = _cycle_curve(cycle_rows, by_cycle)
    latest_cycle = max([row["cycle_index"] for row in cycle_curve], default=None)
    latest_cycle_summary = next((row for row in reversed(cycle_curve) if row["cycle_index"] == latest_cycle), {})
    market_summaries = [_summarize_market(slug, observations) for slug, observations in by_market.items()]
    market_summaries.sort(key=lambda row: (row["latest_verified_net_usdc"], row["max_fill_rate"]), reverse=True)

    live_ready = [row for row in market_summaries if row["recommended_bucket"] == "LIVE_CANARY_CANDIDATE"]
    focus = [row for row in market_summaries if row["recommended_bucket"] == "DRY_RUN_FOCUS"]
    avoid = [row for row in market_summaries if row["recommended_bucket"] == "AVOID"]

    totals = _latest_totals(cycle_rows, by_cycle)
    return {
        "report_type": "research_profit_drivers",
        "row_count": len(market_rows) + len(cycle_rows),
        "market_observation_count": len(market_rows),
        "cycle_summary_count": len(cycle_rows),
        "market_count": len(market_summaries),
        "latest_cycle": latest_cycle,
        "latest_selected_markets": latest_cycle_summary.get("selected_market_count"),
        "latest_active_quote_markets": latest_cycle_summary.get("active_quote_market_count"),
        "latest_eligible_candidates": latest_cycle_summary.get("eligible_candidate_count"),
        "latest_selection_reasons": latest_cycle_summary.get("last_selection_reasons"),
        "latest_filter_reasons": latest_cycle_summary.get("last_filter_reasons"),
        "latest_selection_blockers": _selection_blockers(latest_cycle_summary),
        "profit_totals": totals,
        "selection_curve": cycle_curve[-top:],
        "top_profit_drivers": market_summaries[:top],
        "live_canary_candidates": live_ready[:top],
        "dry_run_focus": focus[:top],
        "avoid": avoid[:top],
        "strategy_actions": _strategy_actions(
            totals=totals,
            live_ready_count=len(live_ready),
            focus_count=len(focus),
            avoid_count=len(avoid),
            latest_selection_reasons=latest_cycle_summary.get("last_selection_reasons") or {},
            latest_filter_reasons=latest_cycle_summary.get("last_filter_reasons") or {},
            latest_selection_blockers=_selection_blockers(latest_cycle_summary),
        ),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Research Profit Drivers",
        "",
        f"- Latest cycle: {report.get('latest_cycle')}",
        f"- Markets observed: {report.get('market_count')}",
        f"- Selected / active / eligible: {report.get('latest_selected_markets')} / "
        f"{report.get('latest_active_quote_markets')} / {report.get('latest_eligible_candidates')}",
        f"- Profit totals: {_compact(report.get('profit_totals'))}",
        f"- Latest selection reasons: {_compact(report.get('latest_selection_reasons'))}",
        f"- Latest filter reasons: {_compact(report.get('latest_filter_reasons'))}",
        f"- Latest selection blockers: {_compact(report.get('latest_selection_blockers'))}",
        "",
        "## Strategy Actions",
        "",
    ]
    for action in report.get("strategy_actions") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Top Profit Drivers", ""])
    for row in report.get("top_profit_drivers") or []:
        lines.append(
            f"- {row.get('market_slug')} | {row.get('recommended_bucket')} | "
            f"verified={row.get('latest_verified_net_usdc')} | fill={row.get('max_fill_rate')} | "
            f"quality={row.get('profit_quality')}"
        )
    lines.extend(["", "## Avoid", ""])
    for row in report.get("avoid") or []:
        lines.append(
            f"- {row.get('market_slug')} | verified={row.get('latest_verified_net_usdc')} | "
            f"risk={_compact(row.get('risk_reject_reasons'))}"
        )
    return "\n".join(lines) + "\n"


def _cycle_curve(cycle_rows: list[dict[str, Any]], by_cycle: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if cycle_rows:
        result = []
        for row in cycle_rows:
            cycle = _int(row.get("cycle_index"))
            if cycle is None:
                continue
            result.append(
                {
                    "cycle_index": cycle,
                    "selected_market_count": _coalesce(row.get("selected_market_count"), 0),
                    "active_quote_market_count": _coalesce(row.get("active_quote_market_count"), 0),
                    "eligible_candidate_count": row.get("eligible_candidate_count"),
                    "last_selection_reasons": row.get("last_selection_reasons"),
                    "last_filter_reasons": row.get("last_filter_reasons"),
                    "scan_diagnostics": row.get("scan_diagnostics"),
                    "verified_net_after_cost_usdc": row.get("verified_net_after_reward_and_cost_usdc"),
                    "modeled_net_after_cost_usdc": row.get("net_after_reward_and_cost_usdc"),
                }
            )
        return sorted(result, key=lambda row: row["cycle_index"])

    result = []
    for cycle, rows in sorted(by_cycle.items()):
        result.append(
            {
                "cycle_index": cycle,
                "selected_market_count": len(rows),
                "active_quote_market_count": sum(1 for row in rows if str(row.get("status") or "").upper() == "QUOTING"),
                "eligible_candidate_count": None,
                "last_selection_reasons": None,
                "last_filter_reasons": None,
                "verified_net_after_cost_usdc": round(sum(_float(row.get("verified_net_window_usdc")) for row in rows), 6),
                "modeled_net_after_cost_usdc": round(
                    sum(_float(row.get("net_after_reward_and_cost_usdc")) for row in rows), 6
                ),
            }
        )
    return result


def _selection_blockers(cycle_summary: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = cycle_summary.get("scan_diagnostics") if isinstance(cycle_summary, dict) else {}
    if not isinstance(diagnostics, dict):
        return []
    blockers = diagnostics.get("selection_blocked_candidates")
    if not isinstance(blockers, list):
        return []
    result: list[dict[str, Any]] = []
    for row in blockers[:10]:
        if isinstance(row, dict):
            result.append(row)
    return result


def _latest_totals(cycle_rows: list[dict[str, Any]], by_cycle: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    valid_cycle_rows = [row for row in cycle_rows if _int(row.get("cycle_index")) is not None]
    if valid_cycle_rows:
        latest = max(valid_cycle_rows, key=lambda row: _int(row.get("cycle_index")) or -1)
        return {
            "modeled_net_after_cost_usdc": _round(latest.get("net_after_reward_and_cost_usdc")),
            "verified_net_after_cost_usdc": _round(latest.get("verified_net_after_reward_and_cost_usdc")),
            "reward_estimate_usdc": _round(latest.get("reward_accrued_estimate_usdc")),
            "actual_reward_usdc": _round(latest.get("reward_accrued_actual_usdc")),
            "spread_realized_usdc": _round(latest.get("spread_realized_usdc")),
            "cost_proxy_usdc": _round(latest.get("cost_proxy_usdc")),
        }
    if not by_cycle:
        return {
            "modeled_net_after_cost_usdc": 0.0,
            "verified_net_after_cost_usdc": 0.0,
            "reward_estimate_usdc": 0.0,
            "actual_reward_usdc": 0.0,
            "spread_realized_usdc": 0.0,
            "cost_proxy_usdc": 0.0,
        }
    latest_cycle = max(by_cycle)
    rows = by_cycle[latest_cycle]
    return {
        "modeled_net_after_cost_usdc": round(sum(_float(row.get("net_after_reward_and_cost_usdc")) for row in rows), 6),
        "verified_net_after_cost_usdc": round(sum(_float(row.get("verified_net_window_usdc")) for row in rows), 6),
        "reward_estimate_usdc": round(sum(_float(row.get("reward_estimate_usdc")) for row in rows), 6),
        "actual_reward_usdc": round(sum(_float(row.get("actual_reward_usdc")) for row in rows), 6),
        "spread_realized_usdc": round(sum(_float(row.get("spread_realized_usdc")) for row in rows), 6),
        "cost_proxy_usdc": 0.0,
    }


def _summarize_market(slug: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: _int(row.get("cycle_index")) or -1)
    latest = rows[-1]
    verified_values = [_float(row.get("verified_net_window_usdc")) for row in rows]
    fill_values = [_float(row.get("fill_rate_window")) for row in rows]
    risk_reasons = Counter(str(row.get("risk_reject_reason") or "") for row in rows if row.get("risk_reject_reason"))
    cancel_reasons = Counter(str(row.get("last_cancel_reason") or "") for row in rows if row.get("last_cancel_reason"))
    actual_reward = _float(latest.get("actual_reward_usdc"))
    spread = _float(latest.get("spread_realized_usdc"))
    simulated = _truthy(latest.get("simulated_fill")) or str(latest.get("evidence_source") or "") == "DRY_RUN_SIMULATED"
    latest_verified = _float(latest.get("verified_net_window_usdc"))
    max_fill = max(fill_values, default=0.0)
    quality = _profit_quality(
        latest_verified=latest_verified,
        actual_reward=actual_reward,
        spread=spread,
        max_fill=max_fill,
        simulated=simulated,
    )
    return {
        "market_slug": slug,
        "first_cycle": _int(rows[0].get("cycle_index")),
        "last_cycle": _int(latest.get("cycle_index")),
        "observation_count": len(rows),
        "latest_status": latest.get("status"),
        "latest_action": latest.get("action"),
        "latest_strategy_id": latest.get("strategy_id"),
        "latest_verified_net_usdc": round(latest_verified, 6),
        "max_verified_net_usdc": round(max(verified_values, default=0.0), 6),
        "min_verified_net_usdc": round(min(verified_values, default=0.0), 6),
        "latest_fill_rate": round(_float(latest.get("fill_rate_window")), 6),
        "max_fill_rate": round(max_fill, 6),
        "bid_filled_shares": round(_float(latest.get("bid_order_filled_size")), 6),
        "ask_filled_shares": round(_float(latest.get("ask_order_filled_size")), 6),
        "actual_reward_usdc": round(actual_reward, 6),
        "spread_realized_usdc": round(spread, 6),
        "evidence_source": latest.get("evidence_source"),
        "simulated_fill": simulated,
        "profit_quality": quality,
        "risk_reject_reasons": dict(risk_reasons),
        "cancel_reasons": dict(cancel_reasons),
        "latest_decision_trace": list(latest.get("decision_trace") or []),
        "recommended_bucket": _bucket_for_market(
            latest_verified=latest_verified,
            actual_reward=actual_reward,
            spread=spread,
            max_fill=max_fill,
            simulated=simulated,
            risk_reject_count=sum(risk_reasons.values()),
        ),
    }


def _profit_quality(
    *, latest_verified: float, actual_reward: float, spread: float, max_fill: float, simulated: bool
) -> str:
    if actual_reward > 0.0:
        return "ACTUAL_REWARD_CONFIRMED"
    if latest_verified > 0.0 and spread > 0.0 and not simulated:
        return "CONFIRMED_SPREAD"
    if latest_verified > 0.0 and (spread > 0.0 or max_fill > 0.0):
        return "SIMULATED_ONLY"
    if latest_verified < 0.0:
        return "NEGATIVE"
    return "NO_EDGE"


def _bucket_for_market(
    *,
    latest_verified: float,
    actual_reward: float,
    spread: float,
    max_fill: float,
    simulated: bool,
    risk_reject_count: int,
) -> str:
    if risk_reject_count > 0 or latest_verified < -0.02:
        return "AVOID"
    if latest_verified > 0.01 and (actual_reward > 0.0 or (spread > 0.0 and not simulated)):
        return "LIVE_CANARY_CANDIDATE"
    if latest_verified > 0.0 or max_fill > 0.0:
        return "DRY_RUN_FOCUS"
    return "WAIT"


def _strategy_actions(
    *,
    totals: dict[str, Any],
    live_ready_count: int,
    focus_count: int,
    avoid_count: int,
    latest_selection_reasons: dict[str, Any],
    latest_filter_reasons: dict[str, Any],
    latest_selection_blockers: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    actual_reward = _float(totals.get("actual_reward_usdc"))
    verified_net = _float(totals.get("verified_net_after_cost_usdc"))
    if live_ready_count == 0:
        actions.append("Keep research/shadow mode until confirmed live evidence exists; simulated profit is not enough.")
    if verified_net > 0.0 and actual_reward <= 0.0:
        actions.append("Treat current edge as spread/fill evidence; keep reward estimates as ranking only.")
    if latest_selection_reasons:
        actions.append(f"Selection pressure is dominated by {_top_counter_key(latest_selection_reasons)}.")
    if latest_selection_blockers:
        top_blocker = latest_selection_blockers[0]
        reason = top_blocker.get("reason")
        slug = top_blocker.get("market_slug")
        cap = top_blocker.get("capital_basis_usdc")
        if reason == "SELECT_PER_MARKET_CAP":
            actions.append(
                f"Top blocked market {slug} needs about ${cap} capital basis; keep live cap unchanged, "
                "but consider a higher dry-run-only cap if we want to study that tier."
            )
        elif reason == "SELECT_ZERO_SIZE_REJECT":
            actions.append(f"Top blocked market {slug} sized to zero after risk checks; keep rotating past it.")
        else:
            actions.append(f"Top blocked market {slug} was skipped by {reason}; inspect blocker list before scaling.")
    if latest_filter_reasons:
        actions.append(f"Candidate pool is dominated by filter {_top_counter_key(latest_filter_reasons)}.")
    if focus_count > 0:
        actions.append("Prioritize dry-run focus markets with positive fill/spread before adding new broad scans.")
    if avoid_count > 0:
        actions.append("Blacklist or down-rank avoid markets before the next long run.")
    return actions


def _top_counter_key(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "none"
    key, value = max(mapping.items(), key=lambda item: _float(item[1]))
    return f"{key}:{value}"


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _round(value: Any) -> float:
    return round(_float(value), 6)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze_profit_drivers(load_jsonl(Path(args.evidence)), top=max(1, int(args.top)))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
