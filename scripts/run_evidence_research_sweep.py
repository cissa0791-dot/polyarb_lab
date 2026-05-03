from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ResearchArm:
    name: str
    event_limit: int
    market_limit: int
    max_selected_markets: int
    per_market_cap_usdc: float
    min_reward_minus_drawdown_per_hour: float
    min_projected_net_at_horizon_usdc: float
    max_true_break_even_hours: float
    snapshot_max_markets: int = 60
    snapshot_filtered_max: int = 40


DEFAULT_ARMS: tuple[ResearchArm, ...] = (
    ResearchArm("baseline_2000_sel3_cap80", 1000, 2000, 3, 80.0, 0.0, 0.0, 2.5),
    ResearchArm("highraw_3000_sel3_cap80", 1500, 3000, 3, 80.0, 0.0, 0.0, 2.5),
    ResearchArm("highraw_3000_sel5_cap80", 1500, 3000, 5, 80.0, 0.0, 0.0, 2.5),
    ResearchArm("cap120_sel3", 1000, 2000, 3, 120.0, 0.0, 0.0, 2.5),
    ResearchArm("cap120_sel5", 1000, 2000, 5, 120.0, 0.0, 0.0, 2.5),
    ResearchArm("stricter_quality", 1000, 2000, 3, 80.0, 0.001, 0.001, 2.0),
    ResearchArm("loose_break_even_4h", 1000, 2000, 3, 80.0, 0.0, 0.0, 4.0),
    ResearchArm("fast_break_even_1h", 1000, 2000, 3, 80.0, 0.0, 0.0, 1.0),
    ResearchArm("raw2500_cap60_sel3", 1250, 2500, 3, 60.0, 0.0, 0.0, 2.5),
    ResearchArm("raw3000_cap40_sel5", 1500, 3000, 5, 40.0, 0.0, 0.0, 2.5),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an isolated multi-arm dry-run evidence sweep.")
    parser.add_argument("--out-root", default=str(ROOT / "data" / "reports_parallel" / "arm_sweep"))
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--cycles", type=int, default=36)
    parser.add_argument("--interval-sec", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--arm", action="append", default=None, help="Run only named default arm(s).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def selected_arms(names: list[str] | None) -> list[ResearchArm]:
    arms = list(DEFAULT_ARMS)
    if not names:
        return arms
    requested = set(names)
    known = {arm.name for arm in arms}
    missing = sorted(requested - known)
    if missing:
        raise SystemExit(f"unknown research arm(s): {', '.join(missing)}")
    return [arm for arm in arms if arm.name in requested]


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    sweep_id = _safe_id(args.sweep_id or datetime.now(timezone.utc).strftime("sweep-%Y%m%dT%H%M%SZ"))
    out_root = Path(args.out_root)
    sweep_dir = out_root / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    arms = selected_arms(args.arm)
    manifest = {
        "report_type": "research_arm_sweep_manifest",
        "sweep_id": sweep_id,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "cycles": max(1, int(args.cycles)),
        "interval_sec": max(0, int(args.interval_sec)),
        "max_concurrency": max(1, int(args.max_concurrency)),
        "arms": [asdict(arm) for arm in arms],
    }
    _write_json(sweep_dir / "sweep_manifest.json", manifest)
    results = _run_arms(
        arms=arms,
        sweep_dir=sweep_dir,
        logs_dir=logs_dir,
        cycles=manifest["cycles"],
        interval_sec=manifest["interval_sec"],
        max_concurrency=manifest["max_concurrency"],
        verbose=bool(args.verbose),
    )
    summary = build_sweep_summary(sweep_id=sweep_id, sweep_dir=sweep_dir, results=results)
    _write_json(sweep_dir / "sweep_summary.json", summary)
    _write_markdown_report(sweep_dir / "sweep_report.md", summary)
    latest = out_root / "sweep_summary_latest.json"
    _write_json(latest, summary)
    return summary


def _run_arms(
    *,
    arms: list[ResearchArm],
    sweep_dir: Path,
    logs_dir: Path,
    cycles: int,
    interval_sec: int,
    max_concurrency: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    pending = list(arms)
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < max_concurrency:
            arm = pending.pop(0)
            started = _start_arm(
                arm=arm,
                sweep_dir=sweep_dir,
                logs_dir=logs_dir,
                cycles=cycles,
                interval_sec=interval_sec,
                verbose=verbose,
            )
            running.append(started)
            _write_json(sweep_dir / "sweep_status_latest.json", _status_payload(pending, running, finished))
        time.sleep(2)
        still_running: list[dict[str, Any]] = []
        for item in running:
            return_code = item["process"].poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["return_code"] = return_code
            item["finished_ts"] = datetime.now(timezone.utc).isoformat()
            item.pop("process")
            finished.append(item)
            _write_json(sweep_dir / "sweep_status_latest.json", _status_payload(pending, still_running, finished))
        running = still_running
    return finished


def _start_arm(
    *,
    arm: ResearchArm,
    sweep_dir: Path,
    logs_dir: Path,
    cycles: int,
    interval_sec: int,
    verbose: bool,
) -> dict[str, Any]:
    arm_dir = sweep_dir / arm.name
    run_id = _safe_id(arm.name)
    log_path = logs_dir / f"{arm.name}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_evidence_research_pipeline.py"),
        "--cycles",
        str(cycles),
        "--interval-sec",
        str(interval_sec),
        "--event-limit",
        str(arm.event_limit),
        "--market-limit",
        str(arm.market_limit),
        "--snapshot-max-markets",
        str(arm.snapshot_max_markets),
        "--snapshot-filtered-max",
        str(arm.snapshot_filtered_max),
        "--max-selected-markets",
        str(arm.max_selected_markets),
        "--research-per-market-cap-usdc",
        str(arm.per_market_cap_usdc),
        "--research-min-reward-minus-drawdown-per-hour",
        str(arm.min_reward_minus_drawdown_per_hour),
        "--research-min-projected-net-at-horizon-usdc",
        str(arm.min_projected_net_at_horizon_usdc),
        "--research-max-true-break-even-hours",
        str(arm.max_true_break_even_hours),
        "--out-dir",
        str(arm_dir),
        "--run-id",
        run_id,
        "--no-archive-existing",
    ]
    if verbose:
        command.append("--verbose")
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "arm": asdict(arm),
        "arm_name": arm.name,
        "arm_dir": str(arm_dir),
        "run_id": run_id,
        "log_path": str(log_path),
        "command": command,
        "pid": process.pid,
        "started_ts": datetime.now(timezone.utc).isoformat(),
        "process": process,
    }


def build_sweep_summary(*, sweep_id: str, sweep_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        arm_dir = Path(result["arm_dir"])
        run_id = str(result["run_id"])
        run_dir = arm_dir / "research_runs" / run_id
        pipeline = _load_json(run_dir / "research_pipeline_summary_latest.json")
        pnl = _load_json(run_dir / "research_auto_trade_pnl_latest.json")
        replay = _load_json(run_dir / "research_replay_report_latest.json")
        pnl_summary = pnl.get("summary") if isinstance(pnl.get("summary"), dict) else {}
        scan = pnl_summary.get("scan_diagnostics") if isinstance(pnl_summary.get("scan_diagnostics"), dict) else {}
        verified = _float(pnl_summary.get("verified_net_after_reward_and_cost_usdc"))
        modeled = _float(pnl_summary.get("net_after_reward_and_cost_usdc"))
        replay_net = _float(replay.get("total_net_pnl_usdc"))
        active = _float(pnl_summary.get("active_quote_market_count"))
        selected = _float(scan.get("selected_markets", pipeline.get("max_selected_markets_requested")))
        partial = bool(pipeline.get("partial"))
        score = _arm_score(
            verified=verified,
            modeled=modeled,
            replay_net=replay_net,
            active=active,
            selected=selected,
            partial=partial,
            return_code=int(result.get("return_code") or 0),
        )
        rows.append(
            {
                "arm_name": result["arm_name"],
                "return_code": result.get("return_code"),
                "arm_dir": str(arm_dir),
                "run_dir": str(run_dir),
                "verified_net_after_cost_usdc": round(verified, 6),
                "modeled_net_after_cost_usdc": round(modeled, 6),
                "replay_total_net_pnl_usdc": round(replay_net, 6),
                "selected_markets": int(selected),
                "active_quote_market_count": int(active),
                "eligible_candidates": pnl_summary.get("last_eligible_candidate_count"),
                "profitability_score": round(score, 6),
                "scale_recommendation": pipeline.get("scale_recommendation"),
                "partial": partial,
                "selection_reasons": pnl_summary.get("last_selection_reasons"),
                "filter_reasons": pnl_summary.get("last_filter_reasons"),
            }
        )
    rows.sort(key=lambda row: row["profitability_score"], reverse=True)
    return {
        "report_type": "research_arm_sweep_summary",
        "sweep_id": sweep_id,
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep_dir),
        "arm_count": len(rows),
        "best_arm": rows[0] if rows else None,
        "arms": rows,
    }


def _arm_score(
    *,
    verified: float,
    modeled: float,
    replay_net: float,
    active: float,
    selected: float,
    partial: bool,
    return_code: int,
) -> float:
    score = verified
    score += min(0.05, max(0.0, modeled) * 0.10)
    score += max(0.0, active) * 0.002
    score += max(0.0, selected) * 0.001
    if replay_net < 0:
        score += replay_net * 0.5
    else:
        score += min(0.02, replay_net * 0.2)
    if partial:
        score -= 0.05
    if return_code != 0:
        score -= 0.10
    return score


def _status_payload(
    pending: list[ResearchArm],
    running: list[dict[str, Any]],
    finished: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "pending": [arm.name for arm in pending],
        "running": [
            {"arm_name": item["arm_name"], "pid": item["pid"], "log_path": item["log_path"]}
            for item in running
        ],
        "finished": [
            {"arm_name": item["arm_name"], "return_code": item.get("return_code")}
            for item in finished
        ],
    }


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Research Arm Sweep", ""]
    best = summary.get("best_arm") if isinstance(summary.get("best_arm"), dict) else None
    if best:
        lines.append(f"- Best arm: {best.get('arm_name')}")
        lines.append(f"- Best score: {best.get('profitability_score')}")
        lines.append(f"- Best verified net: {best.get('verified_net_after_cost_usdc')}")
        lines.append("")
    lines.append("| Rank | Arm | Score | Verified | Replay | Selected | Active | Eligible |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for index, row in enumerate(summary.get("arms") or [], start=1):
        lines.append(
            "| "
            f"{index} | {row.get('arm_name')} | {row.get('profitability_score')} | "
            f"{row.get('verified_net_after_cost_usdc')} | {row.get('replay_total_net_pnl_usdc')} | "
            f"{row.get('selected_markets')} | {row.get('active_quote_market_count')} | "
            f"{row.get('eligible_candidates')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip())
    return safe.strip("-_") or datetime.now(timezone.utc).strftime("sweep-%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    summary = run_sweep(parse_args())
    best = summary.get("best_arm") or {}
    print(
        json.dumps(
            {
                "sweep_id": summary["sweep_id"],
                "arm_count": summary["arm_count"],
                "best_arm": best.get("arm_name"),
                "best_score": best.get("profitability_score"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
