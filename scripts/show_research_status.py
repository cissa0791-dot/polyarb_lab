from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show active and latest autonomous research status.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "reports"))
    parser.add_argument("--proc-root", default="/proc")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def active_research_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    if not proc_root.is_dir():
        return processes
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if not any(arg.endswith("run_evidence_research_pipeline.py") for arg in argv):
            continue
        processes.append(
            {
                "pid": int(entry.name),
                "run_id": _arg_value(argv, "--run-id"),
                "out_dir": _arg_value(argv, "--out-dir"),
                "cycles": _arg_value(argv, "--cycles"),
                "interval_sec": _arg_value(argv, "--interval-sec"),
                "argv": argv,
            }
        )
    return sorted(processes, key=lambda row: int(row["pid"]))


def pick_current_run_dir(out_dir: Path, active: list[dict[str, Any]]) -> Path | None:
    runs_dir = out_dir / "research_runs"
    for process in reversed(active):
        run_id = process.get("run_id")
        process_out = Path(str(process.get("out_dir") or out_dir))
        if run_id:
            run_dir = process_out / "research_runs" / str(run_id)
            if run_dir.exists():
                return run_dir
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_status(out_dir: Path, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    active = active_research_processes(proc_root)
    run_dir = pick_current_run_dir(out_dir, active)
    root_summary = load_json(out_dir / "research_pipeline_summary_latest.json")
    status: dict[str, Any] = {
        "active_research_process_count": len(active),
        "active_research_processes": [
            {
                "pid": process["pid"],
                "run_id": process.get("run_id"),
                "cycles": process.get("cycles"),
                "interval_sec": process.get("interval_sec"),
            }
            for process in active
        ],
        "current_run_dir": str(run_dir) if run_dir else None,
        "root_latest_run_id": root_summary.get("run_id"),
        "root_latest_scale_recommendation": root_summary.get("scale_recommendation"),
    }
    if run_dir:
        status["current_run"] = _run_dir_status(run_dir)
    else:
        status["current_run"] = {}
    return status


def _run_dir_status(run_dir: Path) -> dict[str, Any]:
    pnl = load_json(run_dir / "research_auto_trade_pnl_latest.json")
    summary = pnl.get("summary") if isinstance(pnl.get("summary"), dict) else {}
    pipeline_summary = load_json(run_dir / "research_pipeline_summary_latest.json")
    evidence = run_dir / "research_edge_observations_latest.jsonl"
    snapshots = run_dir / "research_orderbook_snapshots_latest.jsonl"
    return {
        "run_id": run_dir.name,
        "completed": bool(pipeline_summary),
        "partial": pipeline_summary.get("partial") if pipeline_summary else None,
        "scale_recommendation": pipeline_summary.get("scale_recommendation") if pipeline_summary else None,
        "cycle_index": summary.get("cycle_index"),
        "selected_markets": summary.get("scan_diagnostics", {}).get("selected_markets")
        if isinstance(summary.get("scan_diagnostics"), dict)
        else None,
        "active_quote_market_count": summary.get("active_quote_market_count"),
        "eligible_candidates": summary.get("last_eligible_candidate_count"),
        "last_selection_reasons": summary.get("last_selection_reasons"),
        "last_filter_reasons": summary.get("last_filter_reasons"),
        "verified_net_after_cost_usdc": summary.get("verified_net_after_reward_and_cost_usdc"),
        "modeled_net_after_cost_usdc": summary.get("net_after_reward_and_cost_usdc"),
        "bid_filled_shares": summary.get("bid_order_filled_shares"),
        "ask_filled_shares": summary.get("ask_order_filled_shares"),
        "evidence_rows": count_lines(evidence),
        "snapshot_rows": count_lines(snapshots),
        "evidence_bytes": evidence.stat().st_size if evidence.exists() else 0,
        "snapshot_bytes": snapshots.stat().st_size if snapshots.exists() else 0,
    }


def print_status(status: dict[str, Any]) -> None:
    print("== active research ==")
    print(f"processes: {status['active_research_process_count']}")
    for process in status["active_research_processes"]:
        print(
            " - "
            f"pid={process.get('pid')} "
            f"run_id={process.get('run_id')} "
            f"cycles={process.get('cycles')} "
            f"interval={process.get('interval_sec')}"
        )
    print(f"current_run_dir: {status.get('current_run_dir')}")

    current = status.get("current_run") if isinstance(status.get("current_run"), dict) else {}
    if current:
        print()
        print("== current run ==")
        for key in (
            "run_id",
            "completed",
            "partial",
            "scale_recommendation",
            "cycle_index",
            "selected_markets",
            "active_quote_market_count",
            "eligible_candidates",
            "verified_net_after_cost_usdc",
            "modeled_net_after_cost_usdc",
            "bid_filled_shares",
            "ask_filled_shares",
            "evidence_rows",
            "snapshot_rows",
            "evidence_bytes",
            "snapshot_bytes",
        ):
            print(f"{key}: {current.get(key)}")
        print(f"last_selection_reasons: {current.get('last_selection_reasons')}")
        print(f"last_filter_reasons: {current.get('last_filter_reasons')}")

    print()
    print("== root latest ==")
    print(f"root_latest_run_id: {status.get('root_latest_run_id')}")
    print(f"root_latest_scale_recommendation: {status.get('root_latest_scale_recommendation')}")


def _arg_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(argv):
        return None
    return argv[value_index]


def main() -> None:
    args = parse_args()
    status = build_status(Path(args.out_dir), Path(args.proc_root))
    print_status(status)


if __name__ == "__main__":
    main()
