from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.show_research_status import build_status, load_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a concise research run report.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "reports"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def build_report(*, out_dir: Path, run_dir: Path | None = None, proc_root: Path = Path("/proc")) -> str:
    status = build_status(out_dir, proc_root)
    selected_run_dir = run_dir or (Path(status["current_run_dir"]) if status.get("current_run_dir") else None)
    current = status.get("current_run") if isinstance(status.get("current_run"), dict) else {}
    pipeline_summary = load_json(selected_run_dir / "research_pipeline_summary_latest.json") if selected_run_dir else {}
    market_intel = load_json(selected_run_dir / "research_market_intel_latest.json") if selected_run_dir else {}
    profit_drivers = load_json(selected_run_dir / "research_profit_drivers_latest.json") if selected_run_dir else {}
    decision = load_json(out_dir / "autonomous_decision_latest.json")

    lines = [
        "# Research Run Report",
        "",
        f"- Active research processes: {status.get('active_research_process_count')}",
        f"- Current run: {selected_run_dir.name if selected_run_dir else 'none'}",
        f"- Current run dir: {selected_run_dir or 'none'}",
    ]
    if current:
        lines.extend(
            [
                f"- Completed: {current.get('completed')}",
                f"- Cycle: {current.get('cycle_index')}",
                f"- Selected / active / eligible: {current.get('selected_markets')} / "
                f"{current.get('active_quote_market_count')} / {current.get('eligible_candidates')}",
                f"- Verified net after cost: {current.get('verified_net_after_cost_usdc')}",
                f"- Modeled net after cost: {current.get('modeled_net_after_cost_usdc')}",
                f"- Filled shares bid/ask: {current.get('bid_filled_shares')} / {current.get('ask_filled_shares')}",
                f"- Evidence rows / snapshot rows: {current.get('evidence_rows')} / {current.get('snapshot_rows')}",
                f"- Selection reasons: {_compact_json(current.get('last_selection_reasons'))}",
                f"- Filter reasons: {_compact_json(current.get('last_filter_reasons'))}",
            ]
        )

    if pipeline_summary:
        lines.extend(
            [
                "",
                "## Gate Summary",
                "",
                f"- Scale recommendation: {pipeline_summary.get('scale_recommendation')}",
                f"- Live canary eligible: {pipeline_summary.get('live_canary_eligible_count')}",
                f"- Dry-run focus: {pipeline_summary.get('dry_run_focus_count')}",
                f"- Blacklist: {pipeline_summary.get('blacklist_count')}",
                f"- Partial: {pipeline_summary.get('partial')} ({pipeline_summary.get('partial_reason')})",
                f"- Live blockers: {_compact_json(pipeline_summary.get('live_ready_blockers'))}",
            ]
        )

    if decision:
        lines.extend(
            [
                "",
                "## Autonomous Decision",
                "",
                f"- Decision: {decision.get('decision')}",
                f"- Reason: {decision.get('reason')}",
                f"- Target live markets: {decision.get('target_live_markets')}",
                f"- Max live risk USDC: {decision.get('max_live_risk_usdc')}",
                f"- Can execute live: {decision.get('can_execute_live')}",
            ]
        )

    if profit_drivers:
        lines.extend(["", "## Profit Drivers", ""])
        lines.append(f"- Profit totals: {_compact_json(profit_drivers.get('profit_totals'))}")
        lines.append(f"- Latest selection reasons: {_compact_json(profit_drivers.get('latest_selection_reasons'))}")
        lines.append(f"- Latest filter reasons: {_compact_json(profit_drivers.get('latest_filter_reasons'))}")
        blockers = profit_drivers.get("latest_selection_blockers") or []
        if blockers:
            lines.append("- Latest selection blockers:")
            for blocker in blockers[:5]:
                if not isinstance(blocker, dict):
                    continue
                lines.append(
                    f"  - {blocker.get('market_slug')} | {blocker.get('reason')} | "
                    f"capital={blocker.get('capital_basis_usdc')} | score={blocker.get('total_score')}"
                )
        for action in profit_drivers.get("strategy_actions") or []:
            lines.append(f"- {action}")

    markets = market_intel.get("markets") if isinstance(market_intel.get("markets"), list) else []
    if markets:
        lines.extend(["", "## Top Markets", ""])
        for row in markets[:10]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('market_slug')} | {row.get('recommended_action')} | "
                f"score={row.get('research_score')} | {row.get('reason')}"
            )

    lines.extend(["", "## Operator Notes", ""])
    if status.get("active_research_process_count"):
        lines.append("- Research is still running; do not interpret root latest files as final until the run completes.")
    elif pipeline_summary and pipeline_summary.get("live_canary_eligible_count", 0):
        lines.append("- Research is complete and has live candidates; autonomous manager can decide canary under hard caps.")
    elif pipeline_summary:
        lines.append("- Research is complete but live gate is blocked; continue dry-run focus or adjust strategy filters.")
    else:
        lines.append("- No completed pipeline summary is available for this run yet.")

    return "\n".join(lines) + "\n"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = parse_args()
    report = build_report(
        out_dir=Path(args.out_dir),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        proc_root=Path(args.proc_root),
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
