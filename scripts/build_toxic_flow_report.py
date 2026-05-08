from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.report_metadata import attach_writer_metadata  # noqa: E402
from src.live.toxic_flow_detector import (  # noqa: E402
    DEFAULT_IMBALANCE_THRESHOLD,
    DEFAULT_LOOKBACK_SECONDS,
    DEFAULT_MIN_FILL_PROBABILITY,
    DEFAULT_VOLATILITY_SPREAD_MULTIPLIER,
    READY,
    REPORT_SCHEMA_VERSION,
    build_toxic_flow_report,
)


DEFAULT_REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_MARKET_REPORT = DEFAULT_REPORTS_DIR / "live_market_microstructure_latest.json"
DEFAULT_OUT = DEFAULT_REPORTS_DIR / "toxic_flow_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only toxic flow / adverse selection report.")
    parser.add_argument("--market-report", default=str(DEFAULT_MARKET_REPORT))
    parser.add_argument("--snapshots-jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--market-slug")
    parser.add_argument("--best-bid", type=float)
    parser.add_argument("--best-ask", type=float)
    parser.add_argument("--best-bid-size", type=float)
    parser.add_argument("--best-ask-size", type=float)
    parser.add_argument("--quote-spread", type=float)
    parser.add_argument("--imbalance-threshold", type=float, default=DEFAULT_IMBALANCE_THRESHOLD)
    parser.add_argument("--lookback-sec", type=float, default=DEFAULT_LOOKBACK_SECONDS)
    parser.add_argument("--volatility-spread-multiplier", type=float, default=DEFAULT_VOLATILITY_SPREAD_MULTIPLIER)
    parser.add_argument("--min-fill-probability", type=float, default=DEFAULT_MIN_FILL_PROBABILITY)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    explicit = {
        "market_slug": args.market_slug,
        "best_bid": args.best_bid,
        "best_ask": args.best_ask,
        "best_bid_size": args.best_bid_size,
        "best_ask_size": args.best_ask_size,
        "quote_spread": args.quote_spread,
    }
    snapshots = _load_jsonl(Path(args.snapshots_jsonl)) if args.snapshots_jsonl else []
    report = build_toxic_flow_report(
        market_microstructure=_load_json(Path(args.market_report)),
        explicit=explicit,
        snapshots=snapshots,
        imbalance_threshold=args.imbalance_threshold,
        lookback_seconds=args.lookback_sec,
        volatility_spread_multiplier=args.volatility_spread_multiplier,
        min_fill_probability=args.min_fill_probability,
    )
    inputs = [Path(args.market_report)]
    if args.snapshots_jsonl:
        inputs.append(Path(args.snapshots_jsonl))
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        input_reports_used=inputs,
        source_files=[Path(__file__), ROOT / "src" / "live" / "toxic_flow_detector.py"],
        root=ROOT,
    )
    return report


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            nested = payload.get("orderbook_snapshot")
            if isinstance(nested, dict):
                out.append({**nested, "generated_at_utc": payload.get("generated_at_utc") or payload.get("timestamp_utc")})
            else:
                out.append(payload)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.get("status") == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
