from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEFAULT_LOG = ROOT / "data" / "reports" / "reward_correlation_log.json"
_DEFAULT_OUT  = ROOT / "data" / "reports" / "reward_correlation_analysis_latest.json"


def analyze(
    log: list[dict[str, Any]],
    *,
    min_entries: int = 3,
) -> dict[str, Any]:
    if not log:
        return {"status": "NO_DATA", "entries": 0, "recommended_calibration_factor": 1.0, "markets": []}

    # 过滤掉 estimated=0 的记录（首次实盘无对比基准）
    valid = [e for e in log if float(e.get("estimated_since_last_actual_usdc") or 0.0) > 1e-9]

    if len(valid) < min_entries:
        return {
            "status": "INSUFFICIENT_DATA",
            "entries": len(valid),
            "min_required": min_entries,
            "recommended_calibration_factor": 1.0,
            "markets": [],
        }

    # 按市场分组统计
    by_market: dict[str, list[dict[str, Any]]] = {}
    for entry in valid:
        slug = str(entry.get("market_slug") or "unknown")
        by_market.setdefault(slug, []).append(entry)

    market_rows = []
    all_ratios: list[float] = []
    for slug, entries in sorted(by_market.items()):
        total_actual    = sum(float(e.get("actual_delta_usdc") or 0.0) for e in entries)
        total_estimated = sum(float(e.get("estimated_since_last_actual_usdc") or 0.0) for e in entries)
        ratio           = round(total_actual / total_estimated, 6) if total_estimated > 0.0 else 0.0
        all_ratios.append(ratio)
        market_rows.append({
            "market_slug":        slug,
            "data_points":        len(entries),
            "total_actual_usdc":  round(total_actual, 6),
            "total_estimated_usdc": round(total_estimated, 6),
            "actual_vs_estimate_ratio": ratio,
            "interpretation": _interpret(ratio),
        })

    overall_actual    = sum(float(e.get("actual_delta_usdc") or 0.0) for e in valid)
    overall_estimated = sum(float(e.get("estimated_since_last_actual_usdc") or 0.0) for e in valid)
    overall_ratio     = round(overall_actual / overall_estimated, 6) if overall_estimated > 0.0 else 0.0

    # 推荐校准因子 = overall ratio，但保守地向 1.0 回归 30%
    raw_factor   = overall_ratio
    conservative = round(raw_factor * 0.7 + 1.0 * 0.3, 6)   # 30% regression to 1.0
    recommended  = round(max(0.1, min(2.0, conservative)), 6)

    return {
        "status": "OK",
        "entries": len(valid),
        "overall_actual_usdc":    round(overall_actual, 6),
        "overall_estimated_usdc": round(overall_estimated, 6),
        "overall_actual_vs_estimate_ratio": overall_ratio,
        "recommended_calibration_factor": recommended,
        "interpretation": _interpret(overall_ratio),
        "how_to_apply": (
            f"Set reward_calibration_factor: {recommended} in config/auto_trade_profit.yaml "
            f"or pass --reward-calibration-factor {recommended} to the runner."
        ),
        "markets": sorted(market_rows, key=lambda r: r["actual_vs_estimate_ratio"]),
    }


def _interpret(ratio: float) -> str:
    if ratio == 0.0:
        return "NO_ACTUAL_DATA"
    if ratio < 0.25:
        return "SEVERELY_OVERESTIMATED (model is 4x+ too high)"
    if ratio < 0.5:
        return "OVERESTIMATED (model is 2-4x too high)"
    if ratio < 0.8:
        return "SLIGHTLY_HIGH (model is 25-100% too high)"
    if ratio <= 1.2:
        return "ACCURATE"
    if ratio <= 2.0:
        return "UNDERESTIMATED (model is too conservative)"
    return "SEVERELY_UNDERESTIMATED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze reward estimate vs actual to calibrate the model.")
    p.add_argument("--log",  type=Path, default=_DEFAULT_LOG,
                   help="Path to reward_correlation_log.json")
    p.add_argument("--out",  type=Path, default=_DEFAULT_OUT,
                   help="Where to write the analysis JSON")
    p.add_argument("--min-entries", type=int, default=3,
                   help="Minimum valid data points required to recommend a factor")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log.exists():
        print(f"Log file not found: {args.log}")
        print("Run at least one live session with actual reward data first.")
        return

    log: list[dict[str, Any]] = json.loads(args.log.read_text(encoding="utf-8"))
    report = analyze(log, min_entries=args.min_entries)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("REWARD CORRELATION ANALYSIS")
    print(f"Data points        : {report['entries']}")
    print(f"Status             : {report['status']}")
    if report["status"] == "OK":
        print(f"Overall ratio      : {report['overall_actual_vs_estimate_ratio']:.4f}  ({report['interpretation']})")
        print(f"Actual total       : ${report['overall_actual_usdc']:.4f}")
        print(f"Estimated total    : ${report['overall_estimated_usdc']:.4f}")
        print(f"Recommended factor : {report['recommended_calibration_factor']}")
        print(f"Action             : {report['how_to_apply']}")
        print()
        print("Per-market breakdown:")
        for m in report["markets"]:
            print(f"  {m['market_slug'][:55]:55s} ratio={m['actual_vs_estimate_ratio']:.3f}  {m['interpretation']}")
    print(f"Report             : {args.out.resolve()}")


if __name__ == "__main__":
    main()
