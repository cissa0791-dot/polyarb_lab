from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_research_profit_drivers import analyze_profit_drivers, build_markdown_report, load_jsonl
from scripts.build_research_run_report import build_report
from scripts.show_research_status import build_status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh active or latest research run reports.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "reports"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args(argv)


def refresh_reports(
    *,
    out_dir: Path,
    run_dir: Path | None = None,
    proc_root: Path = Path("/proc"),
    top: int = 10,
) -> dict[str, Any]:
    status = build_status(out_dir, proc_root)
    selected_run_dir = run_dir or (Path(status["current_run_dir"]) if status.get("current_run_dir") else None)
    if selected_run_dir is None:
        return {"refreshed": False, "reason": "NO_RESEARCH_RUN", "run_dir": None}

    evidence_path = selected_run_dir / "research_edge_observations_latest.jsonl"
    if not evidence_path.exists():
        return {"refreshed": False, "reason": "NO_EVIDENCE_FILE", "run_dir": str(selected_run_dir)}

    profit_report = analyze_profit_drivers(load_jsonl(evidence_path), top=max(1, int(top)))
    profit_json = selected_run_dir / "research_profit_drivers_latest.json"
    profit_md = selected_run_dir / "research_profit_drivers.md"
    run_report = selected_run_dir / "research_run_report.md"

    profit_json.write_text(json.dumps(profit_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profit_md.write_text(build_markdown_report(profit_report), encoding="utf-8")
    run_report.write_text(build_report(out_dir=out_dir, run_dir=selected_run_dir, proc_root=proc_root), encoding="utf-8")
    return {
        "refreshed": True,
        "run_dir": str(selected_run_dir),
        "latest_cycle": profit_report.get("latest_cycle"),
        "market_count": profit_report.get("market_count"),
        "latest_selected_markets": profit_report.get("latest_selected_markets"),
        "latest_active_quote_markets": profit_report.get("latest_active_quote_markets"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = refresh_reports(
        out_dir=Path(args.out_dir),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        proc_root=Path(args.proc_root),
        top=args.top,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
