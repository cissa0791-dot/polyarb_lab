from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.show_research_status import build_status, load_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze research run performance and live-readiness.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "reports"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser.parse_args(argv)


def analyze_performance(
    *,
    out_dir: Path,
    run_dir: Path | None = None,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    status = build_status(out_dir, proc_root)
    selected_run_dir = run_dir or (Path(status["current_run_dir"]) if status.get("current_run_dir") else None)
    if selected_run_dir is None:
        selected_run_dir = _latest_run_dir(out_dir)
    if selected_run_dir is None:
        return {
            "report_type": "research_performance",
            "generated_ts": _utc_now(),
            "run_dir": None,
            "error": "NO_RESEARCH_RUN_FOUND",
        }

    pnl = load_json(selected_run_dir / "research_auto_trade_pnl_latest.json")
    state = load_json(selected_run_dir / "research_auto_trade_state_latest.json")
    pipeline = load_json(selected_run_dir / "research_pipeline_summary_latest.json")
    replay = load_json(selected_run_dir / "research_replay_report_latest.json")
    current = status.get("current_run") if isinstance(status.get("current_run"), dict) else {}
    summary = pnl.get("summary") if isinstance(pnl.get("summary"), dict) else {}

    cycle_index = _int(_coalesce(summary.get("cycle_index"), state.get("cycle_index"), current.get("cycle_index"))) or 0
    cycles_requested = _int(_coalesce(current.get("cycles_requested"), pipeline.get("scan_cycles_requested")))
    interval_sec = _int(_coalesce(current.get("interval_sec"), pipeline.get("scan_interval_sec")))

    duration_hours, duration_source = _duration_hours(
        state=state,
        pnl=pnl,
        cycle_index=cycle_index,
        interval_sec=interval_sec,
    )
    verified = _float(summary.get("verified_net_after_reward_and_cost_usdc"))
    modeled = _float(summary.get("net_after_reward_and_cost_usdc"))
    capital_limit = _float(summary.get("capital_limit_usdc"))
    capital_used = _float(summary.get("capital_in_use_usdc"))
    actual_reward = _float(summary.get("reward_accrued_actual_usdc"))
    estimated_reward = _float(summary.get("reward_accrued_estimate_usdc"))
    spread = _float(summary.get("spread_realized_usdc"))
    cost = _float(summary.get("cost_proxy_usdc"))
    replay_net = _float(replay.get("total_net_pnl_usdc"))
    active = _int(summary.get("active_quote_market_count")) or 0
    selected = _int(
        summary.get("scan_diagnostics", {}).get("selected_markets")
        if isinstance(summary.get("scan_diagnostics"), dict)
        else current.get("selected_markets")
    )

    blockers = _live_blockers(
        summary=summary,
        pipeline=pipeline,
        replay=replay,
        verified=verified,
        actual_reward=actual_reward,
        active=active,
    )
    live_ready = not blockers and _int(pipeline.get("live_canary_eligible_count")) not in (None, 0)
    performance = {
        "report_type": "research_performance",
        "generated_ts": _utc_now(),
        "run_dir": str(selected_run_dir),
        "run_id": selected_run_dir.name,
        "completed": bool(pipeline),
        "partial": pipeline.get("partial") if pipeline else None,
        "cycle_index": cycle_index,
        "cycles_requested": cycles_requested,
        "progress_pct": round(cycle_index / cycles_requested * 100.0, 4) if cycles_requested else None,
        "interval_sec": interval_sec,
        "duration_hours": _round(duration_hours),
        "duration_source": duration_source,
        "selected_markets": selected,
        "active_quote_market_count": active,
        "eligible_candidates": summary.get("last_eligible_candidate_count"),
        "capital_limit_usdc": _round(capital_limit),
        "capital_in_use_usdc": _round(capital_used),
        "verified_net_after_cost_usdc": _round(verified),
        "modeled_net_after_cost_usdc": _round(modeled),
        "spread_realized_usdc": _round(spread),
        "estimated_reward_usdc": _round(estimated_reward),
        "actual_reward_usdc": _round(actual_reward),
        "cost_proxy_usdc": _round(cost),
        "replay_total_net_pnl_usdc": _round(replay_net),
        "verified_usdc_per_hour": _rate(verified, duration_hours),
        "modeled_usdc_per_hour": _rate(modeled, duration_hours),
        "spread_usdc_per_hour": _rate(spread, duration_hours),
        "estimated_reward_usdc_per_hour": _rate(estimated_reward, duration_hours),
        "actual_reward_usdc_per_hour": _rate(actual_reward, duration_hours),
        "verified_roi_on_cap_pct": _pct(verified, capital_limit),
        "verified_roi_on_cap_pct_per_hour": _pct_per_hour(verified, capital_limit, duration_hours),
        "verified_roi_on_used_cap_pct": _pct(verified, capital_used),
        "verified_roi_on_used_cap_pct_per_hour": _pct_per_hour(verified, capital_used, duration_hours),
        "estimated_reward_share_of_modeled_net": _ratio(estimated_reward, modeled),
        "spread_share_of_verified_net": _ratio(spread, verified),
        "latest_selection_reasons": summary.get("last_selection_reasons"),
        "latest_filter_reasons": summary.get("last_filter_reasons"),
        "live_ready": live_ready,
        "why_not_live": blockers,
        "scale_recommendation": pipeline.get("scale_recommendation") if pipeline else None,
        "live_canary_eligible_count": pipeline.get("live_canary_eligible_count") if pipeline else None,
        "recommended_next_step": _recommended_next_step(live_ready=live_ready, blockers=blockers, completed=bool(pipeline)),
    }
    return performance


def build_markdown_report(report: dict[str, Any]) -> str:
    if report.get("error"):
        return f"# Research Performance\n\n- Error: {report.get('error')}\n"
    lines = [
        "# Research Performance",
        "",
        f"- Run: {report.get('run_id')}",
        f"- Completed: {report.get('completed')} / partial={report.get('partial')}",
        f"- Cycle: {report.get('cycle_index')} / {report.get('cycles_requested')} ({report.get('progress_pct')}%)",
        f"- Selected / active / eligible: {report.get('selected_markets')} / "
        f"{report.get('active_quote_market_count')} / {report.get('eligible_candidates')}",
        f"- Duration hours: {report.get('duration_hours')} ({report.get('duration_source')})",
        f"- Verified net after cost: {report.get('verified_net_after_cost_usdc')} USDC",
        f"- Modeled net after cost: {report.get('modeled_net_after_cost_usdc')} USDC",
        f"- Verified per hour: {report.get('verified_usdc_per_hour')} USDC/h",
        f"- Modeled per hour: {report.get('modeled_usdc_per_hour')} USDC/h",
        f"- ROI on cap: {report.get('verified_roi_on_cap_pct')}% total, "
        f"{report.get('verified_roi_on_cap_pct_per_hour')}%/h",
        f"- ROI on used capital: {report.get('verified_roi_on_used_cap_pct')}% total, "
        f"{report.get('verified_roi_on_used_cap_pct_per_hour')}%/h",
        f"- Spread / estimated reward / actual reward: {report.get('spread_realized_usdc')} / "
        f"{report.get('estimated_reward_usdc')} / {report.get('actual_reward_usdc')} USDC",
        f"- Replay net: {report.get('replay_total_net_pnl_usdc')} USDC",
        f"- Live ready: {report.get('live_ready')}",
        f"- Why not live: {json.dumps(report.get('why_not_live'), ensure_ascii=False)}",
        f"- Next step: {report.get('recommended_next_step')}",
        "",
    ]
    return "\n".join(lines)


def _live_blockers(
    *,
    summary: dict[str, Any],
    pipeline: dict[str, Any],
    replay: dict[str, Any],
    verified: float,
    actual_reward: float,
    active: int,
) -> list[str]:
    blockers: list[str] = []
    for blocker in pipeline.get("live_ready_blockers") or []:
        if blocker not in blockers:
            blockers.append(str(blocker))
    if str(summary.get("mode") or "").upper() == "DRY_RUN" and "DRY_RUN_MODE" not in blockers:
        blockers.append("DRY_RUN_MODE")
    if verified <= 0 and "NON_POSITIVE_VERIFIED_NET" not in blockers:
        blockers.append("NON_POSITIVE_VERIFIED_NET")
    if actual_reward <= 0 and "NO_ACTUAL_REWARD_CONFIRMED" not in blockers:
        blockers.append("NO_ACTUAL_REWARD_CONFIRMED")
    if active <= 0 and "NO_ACTIVE_QUOTES" not in blockers:
        blockers.append("NO_ACTIVE_QUOTES")
    if replay and _float(replay.get("total_net_pnl_usdc")) < 0 and "REPLAY_NEGATIVE" not in blockers:
        blockers.append("REPLAY_NEGATIVE")
    return blockers


def _recommended_next_step(*, live_ready: bool, blockers: list[str], completed: bool) -> str:
    if live_ready:
        return "allow autonomous capped micro live probe"
    if not completed:
        return "let active dry-run finish; do not switch live mid-run"
    if "REPLAY_NEGATIVE" in blockers:
        return "continue dry-run/sweep; improve replay-confirmed fill realism before live"
    if "NO_ACTUAL_REWARD_CONFIRMED" in blockers:
        return "run focused reward-first dry-run or capped isolated live probe only if manager gate allows"
    return "continue dry-run focus and parameter sweep"


def _duration_hours(
    *,
    state: dict[str, Any],
    pnl: dict[str, Any],
    cycle_index: int,
    interval_sec: int | None,
) -> tuple[float, str]:
    started = _parse_ts(state.get("started_ts"))
    updated = _parse_ts(state.get("updated_ts")) or _parse_ts(pnl.get("generated_ts"))
    if started and updated and updated > started:
        return max(0.0, (updated - started).total_seconds() / 3600.0), "state_wall_clock"
    if interval_sec is not None and cycle_index > 0:
        return max(0.0, cycle_index * interval_sec / 3600.0), "cycle_interval_nominal"
    return 0.0, "unknown"


def _latest_run_dir(out_dir: Path) -> Path | None:
    runs_dir = out_dir / "research_runs"
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _rate(numerator: float, hours: float) -> float | None:
    if hours <= 0:
        return None
    return round(numerator / hours, 6)


def _pct(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100.0, 6)


def _pct_per_hour(numerator: float, denominator: float, hours: float) -> float | None:
    pct = _pct(numerator, denominator)
    if pct is None or hours <= 0:
        return None
    return round(pct / hours, 6)


def _ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return round(numerator / denominator, 6)


def main() -> None:
    args = parse_args()
    report = analyze_performance(
        out_dir=Path(args.out_dir),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        proc_root=Path(args.proc_root),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
