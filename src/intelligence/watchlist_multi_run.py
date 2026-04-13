from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_multi_run_stability_report(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def _avg(key: str) -> float:
        if not run_summaries:
            return 0.0
        return round(sum(float(run.get(key, 0.0) or 0.0) for run in run_summaries) / len(run_summaries), 6)

    def _count_positive(key: str) -> int:
        return sum(1 for run in run_summaries if float(run.get(key, 0.0) or 0.0) > 0.0)

    positive_market_gap_runs = _count_positive("market_signal_gap")
    positive_event_gap_runs = _count_positive("event_signal_gap")

    return {
        "report_type": "market_intelligence_watchlist_multi_run_stability",
        "generated_ts": datetime.now(timezone.utc).isoformat(),
        "paper_only": True,
        "primary_ranking_level": "event",
        "summary": {
            "run_count": len(run_summaries),
            "total_validation_forward_events": int(sum(int(run.get("validation_forward_events", 0) or 0) for run in run_summaries)),
            "total_markets_ranked": int(sum(int(run.get("markets_ranked", 0) or 0) for run in run_summaries)),
            "total_events_ranked": int(sum(int(run.get("events_ranked", 0) or 0) for run in run_summaries)),
            "avg_top5_market_forward_signal_score": _avg("top5_market_avg_forward_signal_score"),
            "avg_lower_market_forward_signal_score": _avg("lower_market_avg_forward_signal_score"),
            "avg_top3_event_forward_signal_score": _avg("top3_event_avg_forward_signal_score"),
            "avg_lower_event_forward_signal_score": _avg("lower_event_avg_forward_signal_score"),
            "avg_market_signal_gap": _avg("market_signal_gap"),
            "avg_event_signal_gap": _avg("event_signal_gap"),
            "positive_market_signal_gap_runs": positive_market_gap_runs,
            "positive_event_signal_gap_runs": positive_event_gap_runs,
            "market_signal_gap_positive_rate": round((positive_market_gap_runs / len(run_summaries)), 6) if run_summaries else 0.0,
            "event_signal_gap_positive_rate": round((positive_event_gap_runs / len(run_summaries)), 6) if run_summaries else 0.0,
            "ranking_edge_stays_positive": bool(run_summaries) and positive_market_gap_runs == len(run_summaries),
            "event_ranking_edge_stays_positive": bool(run_summaries) and positive_event_gap_runs == len(run_summaries),
        },
        "runs": run_summaries,
    }


def write_multi_run_stability_report(*, out_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"market_intelligence_watchlist_multi_run_{stamp}.json"
    latest_report_path = out_dir / "market_intelligence_watchlist_multi_run_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    report_path.write_text(payload, encoding="utf-8")
    latest_report_path.write_text(payload, encoding="utf-8")
    return {
        "report_path": report_path,
        "latest_report_path": latest_report_path,
    }
