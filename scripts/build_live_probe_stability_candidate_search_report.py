from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.live_probe_stability_candidate_search import (  # noqa: E402
    DEFAULT_ESTIMATED_GAS_COSTS_USDC,
    DEFAULT_MIN_FILL_PROBABILITY,
    DEFAULT_MIN_PROBE_SIZE,
    DEFAULT_STABILITY_MAX_FILL_PROBABILITY,
    REPORT_SCHEMA_VERSION,
    SEARCH_READY,
    build_stability_candidate_search,
)
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_MARKET_REPORT = DEFAULT_REPORTS_DIR / "live_market_microstructure_latest.json"
DEFAULT_DEPOSIT_WALLET_REPORT = DEFAULT_REPORTS_DIR / "deposit_wallet_readonly_latest.json"
DEFAULT_TOXIC_FLOW_REPORT = DEFAULT_REPORTS_DIR / "toxic_flow_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "live_probe_stability_candidate_search_latest.json"
DEFAULT_BEST_OUT = DEFAULT_REPORTS_DIR / "live_probe_stability_candidate_best_latest.json"
DEFAULT_MD_OUT = DEFAULT_REPORTS_DIR / "live_probe_stability_candidate_search_latest.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only low-fill stability candidate search report for the next live probe."
    )
    parser.add_argument("--market-report", default=str(DEFAULT_MARKET_REPORT))
    parser.add_argument("--deposit-wallet-report", default=str(DEFAULT_DEPOSIT_WALLET_REPORT))
    parser.add_argument("--toxic-flow-report", default=str(DEFAULT_TOXIC_FLOW_REPORT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--best-out", default=str(DEFAULT_BEST_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--max-live-risk-usdc", type=float, required=True)
    parser.add_argument("--stability-max-fill-probability", type=float, default=DEFAULT_STABILITY_MAX_FILL_PROBABILITY)
    parser.add_argument("--min-fill-probability", type=float, default=DEFAULT_MIN_FILL_PROBABILITY)
    parser.add_argument("--min-probe-size", type=float, default=DEFAULT_MIN_PROBE_SIZE)
    parser.add_argument("--estimated-gas-costs-usdc", type=float, default=DEFAULT_ESTIMATED_GAS_COSTS_USDC)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    market_report_path = Path(args.market_report)
    deposit_wallet_path = Path(args.deposit_wallet_report)
    toxic_flow_path = Path(args.toxic_flow_report)
    report = build_stability_candidate_search(
        market_microstructure=_load_json(market_report_path),
        deposit_wallet=_load_json(deposit_wallet_path),
        toxic_flow=_load_json(toxic_flow_path),
        max_live_risk_usdc=args.max_live_risk_usdc,
        stability_max_fill_probability=args.stability_max_fill_probability,
        min_fill_probability=args.min_fill_probability,
        min_probe_size=args.min_probe_size,
        estimated_gas_costs_usdc=args.estimated_gas_costs_usdc,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=[market_report_path, deposit_wallet_path, toxic_flow_path],
        source_files=[Path(__file__), ROOT / "src" / "live" / "live_probe_stability_candidate_search.py"],
        root=ROOT,
    )
    return report


def markdown_report(report: dict[str, Any]) -> str:
    best = report.get("best_candidate") if isinstance(report.get("best_candidate"), dict) else {}
    lines = [
        "# Live Probe Stability Candidate Search",
        "",
        "## Boundary",
        f"- Status: {report.get('status')}",
        f"- Search mode: {report.get('search_mode')}",
        f"- Can submit order: {report.get('can_submit_order')}",
        f"- Live order sent: {report.get('live_order_sent')}",
        f"- Execution authorized: {report.get('execution_authorized')}",
        f"- Token created: {report.get('token_created')}",
        "",
        "## Selected Candidate",
        f"- Market: {best.get('market_slug')}",
        f"- Side: {best.get('selected_side')}",
        f"- Quote bid: {best.get('quote_bid')}",
        f"- Quote ask: {best.get('quote_ask')}",
        f"- Quote size: {best.get('quote_size')}",
        f"- Fill probability: {best.get('fill_probability')}",
        f"- Estimated net profit USDC: {best.get('estimated_net_profit_usdc')}",
        "",
        "## Rejected Candidates",
    ]
    for item in report.get("rejected_candidates") or []:
        lines.append(
            f"- {item.get('candidate_id')}: {', '.join(item.get('blockers') or ['PASS'])}"
        )
    lines.extend(["", "## Blockers"])
    blockers = report.get("blockers") or []
    lines.extend([f"- {blocker}" for blocker in blockers] if blockers else ["- None"])
    lines.extend(["", f"Verdict: {report.get('one_line_verdict')}"])
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _best_candidate_payload(report: dict[str, Any]) -> dict[str, Any]:
    best = report.get("best_candidate")
    if isinstance(best, dict):
        return best
    return {
        "status": report.get("status"),
        "report_type": "live_probe_stability_candidate_best",
        "market_slug": report.get("market_slug"),
        "blockers": report.get("blockers") or ["NO_LOW_FILL_STABILITY_CANDIDATE"],
        "best_candidate": None,
        "can_submit_order": False,
        "live_order_sent": False,
        "execution_authorized": False,
        "token_created": False,
        "source_search_report": str(Path(DEFAULT_OUT)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    _write_json(Path(args.out), report)
    _write_json(Path(args.best_out), _best_candidate_payload(report))
    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == SEARCH_READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
