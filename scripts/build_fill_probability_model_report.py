from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.fill_probability_model import REPORT_SCHEMA_VERSION, estimate_fill_probability  # noqa: E402
from src.live.report_metadata import attach_writer_metadata  # noqa: E402


DEFAULT_OUT = ROOT / "data" / "reports" / "fill_probability_model_latest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fill probability model v1 proxy report.")
    parser.add_argument("--distance-to-best", type=float, required=True)
    parser.add_argument("--queue-ahead-size", type=float, default=0.0)
    parser.add_argument("--same-price-depth", type=float, default=0.0)
    parser.add_argument("--recent-trade-rate", type=float, default=0.0)
    parser.add_argument("--cancel-velocity", type=float, default=0.0)
    parser.add_argument("--spread", type=float, default=0.01)
    parser.add_argument("--depth-imbalance", type=float, default=0.0)
    parser.add_argument("--quote-lifetime-seconds", type=float, default=300.0)
    parser.add_argument("--volatility", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = estimate_fill_probability(
        distance_to_best=args.distance_to_best,
        queue_ahead_size=args.queue_ahead_size,
        same_price_depth=args.same_price_depth,
        recent_trade_rate=args.recent_trade_rate,
        cancel_velocity=args.cancel_velocity,
        spread=args.spread,
        depth_imbalance=args.depth_imbalance,
        quote_lifetime_seconds=args.quote_lifetime_seconds,
        volatility=args.volatility,
    )
    attach_writer_metadata(
        report,
        writer_script=Path(__file__),
        report_schema_version=REPORT_SCHEMA_VERSION,
        source_files=[Path(__file__), ROOT / "src" / "live" / "fill_probability_model.py"],
        root=ROOT,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
