from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_live_edge_evidence import load_jsonl as load_evidence_jsonl


CommandRunner = Callable[[list[str]], None]


class PipelineInterrupted(RuntimeError):
    """Raised when the long dry-run scan is interrupted but partial outputs can be built."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dry-run evidence research pipeline.")
    parser.add_argument("--live", action="store_true", help="Rejected: this pipeline is dry-run only.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short smoke-test run: 10 cycles with 5 second intervals.",
    )
    parser.add_argument("--cycles", type=int, default=120)
    parser.add_argument("--interval-sec", type=int, default=30)
    parser.add_argument("--event-limit", type=int, default=1000)
    parser.add_argument("--market-limit", type=int, default=2000)
    parser.add_argument("--snapshot-max-markets", type=int, default=80)
    parser.add_argument("--snapshot-filtered-max", type=int, default=60)
    parser.add_argument("--max-selected-markets", type=int, default=3)
    parser.add_argument(
        "--research-per-market-cap-usdc",
        type=float,
        default=40.0,
        help="Dry-run research cap per selected market. This does not change live canary caps.",
    )
    parser.add_argument(
        "--research-min-reward-minus-drawdown-per-hour",
        type=float,
        default=0.0,
        help="Dry-run research filter only. Live/canary commands keep their own stricter defaults.",
    )
    parser.add_argument(
        "--research-min-projected-net-at-horizon-usdc",
        type=float,
        default=0.0,
        help="Dry-run research filter only. Lower values broaden observation without placing live orders.",
    )
    parser.add_argument(
        "--research-max-true-break-even-hours",
        type=float,
        default=2.5,
        help="Dry-run research filter only. Live/canary commands keep their own stricter defaults.",
    )
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "reports"))
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable identifier for this research run. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip the scan and rebuild reports from existing evidence/snapshot files.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Directory containing existing research files for --merge-only. Defaults to latest files in --out-dir.",
    )
    parser.add_argument(
        "--no-archive-existing",
        action="store_true",
        help="Do not copy existing latest research files into the run archive before updating latest.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.live:
        parser.error("run_evidence_research_pipeline.py is dry-run only; --live is not allowed")
    if args.quick:
        args.cycles = 10
        args.interval_sec = 5
    return args


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_pipeline(args: argparse.Namespace, *, command_runner: CommandRunner | None = None) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_quick = bool(getattr(args, "quick", False))
    prefix = "research_quick" if is_quick else "research"
    run_id = _normalise_run_id(getattr(args, "run_id", None), quick=is_quick)
    run_dir = out_dir / "research_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_paths = _research_paths(out_dir, prefix=prefix)
    run_paths = _research_paths(run_dir, prefix=prefix)
    source_dir_arg = getattr(args, "source_dir", None)
    if bool(getattr(args, "merge_only", False)):
        source_paths = _research_paths(Path(source_dir_arg), prefix=prefix) if source_dir_arg else latest_paths
        _copy_existing_inputs_for_merge(source_paths, run_paths)
    else:
        _clean_run_outputs(run_paths)
        if not bool(getattr(args, "no_archive_existing", False)):
            _archive_latest_outputs(latest_paths, run_dir / "archive_before_latest_update")

    runner = command_runner or _default_command_runner
    partial = False
    partial_reason = None
    if not bool(getattr(args, "merge_only", False)):
        try:
            runner(_scan_command(args, run_paths))
        except (KeyboardInterrupt, PipelineInterrupted):
            partial = True
            partial_reason = "SCAN_INTERRUPTED"
            print(
                "Scan interrupted; building partial evidence/replay reports from data collected so far.",
                file=sys.stderr,
            )
    runner(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_live_edge_evidence.py"),
            "--evidence",
            str(run_paths["evidence"]),
            "--out",
            str(run_paths["evidence_summary"]),
            "--market-intel-out",
            str(run_paths["evidence_market_intel"]),
        ]
    )
    runner(
        [
            sys.executable,
            str(ROOT / "scripts" / "replay_live_orderbook_snapshots.py"),
            "--snapshots",
            str(run_paths["snapshots"]),
            "--out",
            str(run_paths["replay"]),
        ]
    )

    evidence_summary = load_json(run_paths["evidence_summary"])
    replay_report = load_json(run_paths["replay"])
    evidence_rows = load_evidence_jsonl(str(run_paths["evidence"]))
    arb_rows = [row for row in evidence_rows if row.get("row_type") == "arb_opportunity"]
    contamination = _detect_contaminated_state(evidence_rows)
    outputs = build_research_outputs(
        evidence_summary=evidence_summary,
        replay_report=replay_report,
        arb_rows=arb_rows,
        paths={key: str(path) for key, path in run_paths.items()},
    )
    outputs["summary"]["partial"] = partial
    outputs["summary"]["partial_reason"] = partial_reason
    outputs["summary"]["scan_cycles_requested"] = max(1, int(args.cycles))
    outputs["summary"]["scan_interval_sec"] = max(0, int(args.interval_sec))
    outputs["summary"]["max_selected_markets_requested"] = max(1, int(getattr(args, "max_selected_markets", 3)))
    outputs["summary"]["research_per_market_cap_usdc"] = max(
        1.0,
        float(getattr(args, "research_per_market_cap_usdc", 40.0)),
    )
    outputs["summary"]["quick"] = is_quick
    outputs["summary"]["merge_only"] = bool(getattr(args, "merge_only", False))
    outputs["summary"]["source_dir"] = str(Path(source_dir_arg)) if source_dir_arg else None
    outputs["summary"]["run_id"] = run_id
    outputs["summary"]["run_dir"] = str(run_dir)
    outputs["summary"]["latest_prefix"] = prefix
    outputs["summary"]["fresh_state"] = True
    outputs["summary"]["state_path"] = str(run_paths["state"])
    outputs["summary"]["pnl_path"] = str(run_paths["pnl"])
    outputs["summary"]["contaminated_state_detected"] = bool(contamination["contaminated_state_detected"])
    outputs["summary"]["contamination_reason"] = contamination["contamination_reason"]
    outputs["summary"]["initial_filled_shares"] = contamination["initial_filled_shares"]
    if contamination["contaminated_state_detected"]:
        outputs["summary"]["scale_recommendation"] = "DO_NOT_SCALE"

    _write_json(run_paths["pipeline_summary"], outputs["summary"])
    _write_json(run_paths["market_intel"], outputs["market_intel"])
    _write_json(run_paths["whitelist"], outputs["whitelist"])
    _write_json(run_paths["blacklist"], outputs["blacklist"])
    _copy_outputs_to_latest(run_paths, latest_paths)
    return outputs["summary"]


def _research_paths(base_dir: Path, *, prefix: str) -> dict[str, Path]:
    return {
        "evidence": base_dir / f"{prefix}_edge_observations_latest.jsonl",
        "snapshots": base_dir / f"{prefix}_orderbook_snapshots_latest.jsonl",
        "evidence_summary": base_dir / f"{prefix}_live_edge_summary_latest.json",
        "evidence_market_intel": base_dir / f"{prefix}_evidence_market_intel_latest.json",
        "replay": base_dir / f"{prefix}_replay_report_latest.json",
        "pipeline_summary": base_dir / f"{prefix}_pipeline_summary_latest.json",
        "market_intel": base_dir / f"{prefix}_market_intel_latest.json",
        "whitelist": base_dir / f"{prefix}_whitelist_latest.json",
        "blacklist": base_dir / f"{prefix}_blacklist_latest.json",
        "state": base_dir / f"{prefix}_auto_trade_state_latest.json",
        "pnl": base_dir / f"{prefix}_auto_trade_pnl_latest.json",
    }


def _normalise_run_id(value: str | None, *, quick: bool) -> str:
    raw = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in raw.strip())
    safe = safe.strip("-_") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"quick-{safe}" if quick and not safe.startswith("quick-") else safe


def _clean_run_outputs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if path.exists():
            path.unlink()


def _archive_latest_outputs(paths: dict[str, Path], archive_dir: Path) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, archive_dir / path.name)


def _copy_existing_inputs_for_merge(source_paths: dict[str, Path], run_paths: dict[str, Path]) -> None:
    for key in ("evidence", "snapshots", "state", "pnl"):
        source = source_paths[key]
        destination = run_paths[key]
        if not source.exists():
            continue
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_outputs_to_latest(run_paths: dict[str, Path], latest_paths: dict[str, Path]) -> None:
    for key, source in run_paths.items():
        if not source.exists():
            continue
        destination = latest_paths[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _scan_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    max_selected_markets = max(1, int(getattr(args, "max_selected_markets", 3)))
    per_market_cap_usdc = max(1.0, float(getattr(args, "research_per_market_cap_usdc", 40.0)))
    capital_usdc = per_market_cap_usdc * max_selected_markets
    capital_arg = _number_arg(capital_usdc)
    per_market_cap_arg = _number_arg(per_market_cap_usdc)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_auto_trade_profit.py"),
        "--dry-run",
        "--cycles",
        str(max(1, int(args.cycles))),
        "--interval-sec",
        str(max(0, int(args.interval_sec))),
        "--capital",
        capital_arg,
        "--per-market-cap",
        per_market_cap_arg,
        "--max-markets",
        str(max_selected_markets),
        "--strategy-set",
        "inventory_manager,reward_mm,arb_scanner",
        "--action-mode",
        "optimal",
        "--quote-search-mode",
        "best_ev",
        "--scale-mode",
        "evidence_gated",
        "--risk-profile",
        "canary",
        "--target-inventory-usdc-per-market",
        "20",
        "--max-inventory-usdc-per-market",
        per_market_cap_arg,
        "--max-total-inventory-usdc",
        capital_arg,
        "--max-total-open-buy-usdc",
        capital_arg,
        "--max-account-open-buy-orders",
        str(max_selected_markets),
        "--min-reward-minus-drawdown-per-hour",
        str(float(getattr(args, "research_min_reward_minus_drawdown_per_hour", 0.0))),
        "--min-projected-net-at-horizon-usdc",
        str(float(getattr(args, "research_min_projected_net_at_horizon_usdc", 0.0))),
        "--max-true-break-even-hours",
        str(float(getattr(args, "research_max_true_break_even_hours", 2.5))),
        "--min-live-order-size-shares",
        "5",
        "--inventory-policy",
        "balanced",
        "--inventory-dust-shares",
        "0.01",
        "--profit-evidence-mode",
        "strict",
        "--actual-reward-warmup-minutes",
        "120",
        "--state-path",
        str(paths["state"]),
        "--pnl-path",
        str(paths["pnl"]),
        "--reset-state",
        "--edge-evidence-path",
        str(paths["evidence"]),
        "--record-orderbook-snapshots",
        "--orderbook-snapshot-path",
        str(paths["snapshots"]),
        "--orderbook-snapshot-max-markets",
        str(max(1, int(args.snapshot_max_markets))),
        "--orderbook-snapshot-include-filtered",
        "--orderbook-snapshot-filtered-max",
        str(max(0, int(args.snapshot_filtered_max))),
        "--orderbook-snapshot-min-score",
        "0.0",
        "--event-limit",
        str(max(1, int(args.event_limit))),
        "--market-limit",
        str(max(1, int(args.market_limit))),
    ]
    if bool(getattr(args, "verbose", False)):
        command.append("--verbose")
    return command


def build_research_outputs(
    *,
    evidence_summary: dict[str, Any],
    replay_report: dict[str, Any],
    arb_rows: Iterable[dict[str, Any]],
    paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    evidence_markets = dict(((evidence_summary.get("market_intel") or {}).get("markets") or {}))
    evidence_rows = _evidence_rows_by_market(evidence_summary)
    replay_markets = {
        str(row.get("market_slug")): row
        for row in list(replay_report.get("markets") or [])
        if isinstance(row, dict) and row.get("market_slug")
    }
    arb_status = _arb_status_by_market(arb_rows)
    all_slugs = sorted(set(evidence_markets) | set(evidence_rows) | set(replay_markets) | set(arb_status))

    rows: list[dict[str, Any]] = []
    for slug in all_slugs:
        evidence = evidence_markets.get(slug) or {}
        evidence_row = evidence_rows.get(slug) or {}
        replay = replay_markets.get(slug) or {}
        evidence_status = str(evidence.get("evidence_status") or evidence_row.get("evidence_status") or "NO_EVIDENCE")
        replay_suitability = str(replay.get("suitability") or "NO_EVIDENCE")
        arb = arb_status.get(slug) or "ARB_SKIP"
        actual_reward = _float(evidence.get("actual_reward_window_usdc", evidence_row.get("actual_reward_usdc", 0.0)))
        simulated_fill = _truthy(evidence.get("simulated_fill", evidence_row.get("simulated_fill")))
        simulated_spread = _float(
            evidence.get("simulated_spread_window_usdc", evidence_row.get("simulated_spread_usdc", 0.0))
        )
        replay_confirmed = _truthy(replay.get("replay_confirmed")) or replay_suitability == "REWARD_MM_CANDIDATE"
        action, reason = _recommended_action(
            evidence_status=evidence_status,
            replay_suitability=replay_suitability,
            arb_status=arb,
            simulated_fill=simulated_fill,
            actual_reward_usdc=actual_reward,
            replay_confirmed=replay_confirmed,
        )
        row = {
            "market_slug": slug,
            "status": action,
            "research_score": _research_score(evidence, replay, evidence_status, replay_suitability, arb),
            "evidence_status": evidence_status,
            "replay_suitability": replay_suitability,
            "arb_status": arb,
            "recommended_action": action,
            "reason": reason,
            "verified_net_window_usdc": evidence.get("verified_net_window_usdc", evidence_row.get("verified_net_usdc", 0.0)),
            "actual_reward_window_usdc": round(actual_reward, 6),
            "fill_rate_window": evidence.get("fill_rate_window", evidence_row.get("fill_rate", 0.0)),
            "evidence_source": evidence.get("evidence_source", evidence_row.get("evidence_source", "UNKNOWN")),
            "simulated_fill": simulated_fill,
            "simulated_spread_window_usdc": round(simulated_spread, 6),
            "replay_confirmed": replay_confirmed,
            "net_pnl_usdc": replay.get("net_pnl_usdc", 0.0),
            "simulated_fill_count": replay.get("simulated_fill_count", replay.get("fill_count", 0)),
        }
        rows.append(row)

    rows.sort(key=lambda item: (item["recommended_action"] == "LIVE_CANARY_ELIGIBLE", item["research_score"]), reverse=True)
    whitelist = [row for row in rows if row["recommended_action"] == "LIVE_CANARY_ELIGIBLE"]
    blacklist = [row for row in rows if row["recommended_action"] == "BLACKLIST"]
    focus = [row for row in rows if row["recommended_action"] == "DRY_RUN_FOCUS"]
    positive_dry_run = [row for row in focus if _float(row.get("verified_net_window_usdc")) > 0.0]
    blocked_high_score = [
        row
        for row in rows
        if row["recommended_action"] in {"DRY_RUN_FOCUS", "OBSERVE_MORE"}
        and _float(row.get("research_score")) >= 0.25
    ]
    avoid = [row for row in rows if row["recommended_action"] in {"BLACKLIST", "IGNORE"}]
    scale_recommendation = _clamped_scale_recommendation(
        requested=str(evidence_summary.get("scale_recommendation") or "DO_NOT_SCALE"),
        live_canary_eligible_count=len(whitelist),
        dry_run_focus_count=len(focus),
    )
    simulated_profitable_market_count = int(
        evidence_summary.get(
            "simulated_profitable_market_count",
            sum(1 for row in rows if row["simulated_fill"] and _float(row["verified_net_window_usdc"]) > 0.0),
        )
        or 0
    )
    replay_confirmed_market_count = sum(1 for row in rows if row["replay_confirmed"])
    actual_reward_confirmed_market_count = int(evidence_summary.get("actual_reward_confirmed_market_count") or 0)
    summary = {
        "report_type": "research_pipeline_summary",
        "market_count": len(rows),
        "live_canary_eligible_count": len(whitelist),
        "dry_run_focus_count": len(focus),
        "blacklist_count": len(blacklist),
        "scale_recommendation": scale_recommendation,
        "profitable_market_count": evidence_summary.get("profitable_market_count", 0),
        "simulated_profitable_market_count": simulated_profitable_market_count,
        "replay_confirmed_market_count": replay_confirmed_market_count,
        "actual_reward_confirmed_market_count": actual_reward_confirmed_market_count,
        "live_ready_blockers": _live_ready_blockers(
            rows=rows,
            whitelist_count=len(whitelist),
            replay_confirmed_market_count=replay_confirmed_market_count,
            actual_reward_confirmed_market_count=actual_reward_confirmed_market_count,
        ),
        "replayed_market_count": replay_report.get("replayed_market_count", 0),
        "top_focus_markets": focus[:10],
        "positive_dry_run_market_count": len(positive_dry_run),
        "blocked_high_score_market_count": len(blocked_high_score),
        "avoid_market_count": len(avoid),
        "positive_dry_run_markets": positive_dry_run[:10],
        "blocked_high_score_markets": blocked_high_score[:10],
        "avoid_markets": avoid[:10],
        "output_paths": paths or {},
        "partial": False,
        "partial_reason": None,
        "fresh_state": True,
        "state_path": (paths or {}).get("state"),
        "pnl_path": (paths or {}).get("pnl"),
        "contaminated_state_detected": False,
        "contamination_reason": None,
    }
    return {
        "summary": summary,
        "market_intel": {
            "report_type": "research_market_intel",
            "markets": rows,
            "by_market": {row["market_slug"]: row for row in rows},
            "positive_dry_run_markets": positive_dry_run,
            "blocked_high_score_markets": blocked_high_score,
            "avoid_markets": avoid,
        },
        "whitelist": {"report_type": "research_whitelist", "markets": whitelist},
        "blacklist": {"report_type": "research_blacklist", "markets": blacklist},
    }


def _clamped_scale_recommendation(
    *,
    requested: str,
    live_canary_eligible_count: int,
    dry_run_focus_count: int,
) -> str:
    order = {
        "DO_NOT_SCALE": 0,
        "ALLOW_DRY_RUN_FOCUS": 1,
        "ALLOW_CANARY_ONLY": 2,
        "ALLOW_SCALE_TO_2_MARKETS": 3,
    }
    by_rank = {rank: value for value, rank in order.items()}
    requested_rank = order.get(requested, 0)
    if live_canary_eligible_count <= 0:
        max_allowed = "ALLOW_DRY_RUN_FOCUS" if dry_run_focus_count > 0 else "DO_NOT_SCALE"
    elif live_canary_eligible_count == 1:
        max_allowed = "ALLOW_CANARY_ONLY"
    else:
        max_allowed = "ALLOW_SCALE_TO_2_MARKETS"
    return by_rank[min(requested_rank, order[max_allowed])]


def _evidence_rows_by_market(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for section in ("best_markets", "worst_markets", "blacklist_candidates", "whitelist_candidates"):
        for row in summary.get(section) or []:
            if isinstance(row, dict) and row.get("market_slug"):
                rows[str(row["market_slug"])] = row
    return rows


def _arb_status_by_market(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    status_by_market: dict[str, str] = {}
    for row in rows:
        status = str(row.get("status") or "ARB_SKIP")
        for leg in row.get("required_legs") or []:
            if not isinstance(leg, dict) or not leg.get("market_slug"):
                continue
            slug = str(leg["market_slug"])
            if status == "ARB_CANDIDATE" or slug not in status_by_market:
                status_by_market[slug] = status
    return status_by_market


def _recommended_action(
    *,
    evidence_status: str,
    replay_suitability: str,
    arb_status: str,
    simulated_fill: bool,
    actual_reward_usdc: float,
    replay_confirmed: bool,
) -> tuple[str, str]:
    if evidence_status == "BLACKLIST_CANDIDATE" or replay_suitability == "AVOID_ADVERSE_SELECTION":
        return "BLACKLIST", "bad evidence or adverse replay"
    if replay_suitability == "AVOID_TOO_THIN":
        return "IGNORE", "orderbook too thin for reward MM replay"
    confirmed_positive_evidence = evidence_status == "WHITELIST_CANDIDATE" and (
        not simulated_fill or actual_reward_usdc > 0.0
    )
    if confirmed_positive_evidence and replay_suitability == "REWARD_MM_CANDIDATE" and replay_confirmed:
        return "LIVE_CANARY_ELIGIBLE", "real positive evidence and replay candidate"
    if simulated_fill and evidence_status != "BLACKLIST_CANDIDATE":
        return "DRY_RUN_FOCUS", "simulated edge only; needs confirmed evidence before live"
    if evidence_status == "WHITELIST_CANDIDATE":
        return "DRY_RUN_FOCUS", "positive evidence needs replay confirmation before live"
    if replay_suitability in {"REWARD_MM_CANDIDATE", "REWARD_MM_WATCH"} or arb_status == "ARB_CANDIDATE":
        return "DRY_RUN_FOCUS", "needs more dry-run observation before live"
    if replay_suitability == "AVOID_NO_FILL":
        return "OBSERVE_MORE", "snapshots did not produce simulated fills"
    return "OBSERVE_MORE", "no reliable edge evidence yet"


def _live_ready_blockers(
    *,
    rows: list[dict[str, Any]],
    whitelist_count: int,
    replay_confirmed_market_count: int,
    actual_reward_confirmed_market_count: int,
) -> list[str]:
    if whitelist_count > 0:
        return []
    blockers: list[str] = ["NO_LIVE_CANARY_ELIGIBLE"]
    if any(row["simulated_fill"] and _float(row["verified_net_window_usdc"]) > 0.0 for row in rows):
        blockers.append("SIMULATED_PROFIT_ONLY")
    if replay_confirmed_market_count <= 0:
        blockers.append("REPLAY_NOT_CONFIRMED")
    if actual_reward_confirmed_market_count <= 0:
        blockers.append("NO_ACTUAL_REWARD_CONFIRMED")
    return blockers


def _research_score(
    evidence: dict[str, Any],
    replay: dict[str, Any],
    evidence_status: str,
    replay_suitability: str,
    arb_status: str,
) -> float:
    score = _float(evidence.get("verified_net_window_usdc")) + _float(replay.get("net_pnl_usdc"))
    if evidence_status == "WHITELIST_CANDIDATE":
        score += 1.0
    if evidence_status == "BLACKLIST_CANDIDATE":
        score -= 1.0
    if replay_suitability == "REWARD_MM_CANDIDATE":
        score += 0.5
    elif replay_suitability.startswith("AVOID_"):
        score -= 0.5
    if arb_status == "ARB_CANDIDATE":
        score += 0.25
    return round(score, 6)


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


def _number_arg(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _detect_contaminated_state(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    initial_filled = 0.0
    initial_spread = 0.0
    initial_reward_estimate = 0.0
    for row in rows:
        if row.get("row_type") != "market_observation":
            continue
        cycle_index = int(_float(row.get("cycle_index")))
        if cycle_index != 1:
            continue
        initial_filled += max(0.0, _float(row.get("bid_order_filled_size")))
        initial_filled += max(0.0, _float(row.get("ask_order_filled_size")))
        initial_spread += max(0.0, _float(row.get("spread_realized_usdc")))
        initial_reward_estimate += max(0.0, _float(row.get("reward_estimate_usdc")))

    reasons: list[str] = []
    if initial_filled > 5.0:
        reasons.append("CYCLE_1_FILLED_SHARES_GT_5")
    if initial_spread > 0.01:
        reasons.append("CYCLE_1_SPREAD_GT_0_01")
    if initial_reward_estimate > 0.01:
        reasons.append("CYCLE_1_REWARD_ESTIMATE_GT_0_01")
    return {
        "contaminated_state_detected": bool(reasons),
        "contamination_reason": ",".join(reasons) if reasons else None,
        "initial_filled_shares": round(initial_filled, 6),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_command_runner(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=str(ROOT))
    try:
        return_code = process.wait()
    except KeyboardInterrupt as exc:
        _terminate_process(process)
        raise PipelineInterrupted() from exc
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    summary = run_pipeline(parse_args())
    print(
        json.dumps(
            {
                "market_count": summary["market_count"],
                "live_canary_eligible_count": summary["live_canary_eligible_count"],
                "dry_run_focus_count": summary["dry_run_focus_count"],
                "blacklist_count": summary["blacklist_count"],
                "scale_recommendation": summary["scale_recommendation"],
                "partial": summary.get("partial", False),
                "partial_reason": summary.get("partial_reason"),
                "run_id": summary.get("run_id"),
                "run_dir": summary.get("run_dir"),
                "live_ready_blockers": summary.get("live_ready_blockers", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
