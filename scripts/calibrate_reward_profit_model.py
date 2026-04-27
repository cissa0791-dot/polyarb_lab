from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_calibration_report(
    *,
    state_payload: dict[str, Any],
    pnl_payload: dict[str, Any],
    current_maker_take_share: float = 0.30,
    min_hours: float = 1.0,
    min_net_usdc: float = 0.0,
) -> dict[str, Any]:
    markets = list(pnl_payload.get("markets") or [])
    calibrated_markets: list[dict[str, Any]] = []
    recommended_values: list[float] = []

    for market in markets:
        metrics = dict(market.get("selection_metrics") or {})
        hours = float(market.get("hours_in_reward_zone") or 0.0)
        expected_net_edge_per_hour = float(metrics.get("expected_net_edge_per_hour") or 0.0)
        expected_spread_per_hour = float(metrics.get("expected_spread_capture_per_hour") or 0.0)
        expected_model_net = expected_net_edge_per_hour * hours
        expected_spread = expected_spread_per_hour * hours
        realized_spread = float(
            market.get("simulated_spread_capture_usdc")
            or market.get("spread_realized_usdc")
            or 0.0
        )
        realized_net = float(market.get("verified_net_after_reward_and_cost_usdc") or market.get("net_after_reward_and_cost_usdc") or 0.0)

        if hours <= 0.0:
            spread_ratio = 0.0
            recommended = current_maker_take_share * 0.5
            reason = "INSUFFICIENT_RUNTIME"
        elif expected_spread <= 0.0:
            spread_ratio = 0.0
            recommended = current_maker_take_share * 0.5
            reason = "NO_EXPECTED_SPREAD_EVIDENCE"
        else:
            spread_ratio = _clamp(realized_spread / expected_spread, 0.05, 1.0)
            recommended = current_maker_take_share * spread_ratio
            reason = "OK" if realized_net >= min_net_usdc else "NEGATIVE_REALIZED_NET"

        recommended = round(_clamp(recommended, 0.01, current_maker_take_share), 6)
        recommended_values.append(recommended)
        calibrated_markets.append(
            {
                "market_slug": market.get("market_slug"),
                "event_slug": market.get("event_slug"),
                "hours_in_reward_zone": round(hours, 6),
                "expected_net_edge_per_hour": round(expected_net_edge_per_hour, 6),
                "expected_model_net_usdc": round(expected_model_net, 6),
                "realized_net_after_cost_usdc": round(realized_net, 6),
                "expected_spread_capture_usdc": round(expected_spread, 6),
                "realized_or_simulated_spread_usdc": round(realized_spread, 6),
                "spread_realization_ratio": round(spread_ratio, 6),
                "recommended_maker_take_share": recommended,
                "reason": reason,
            }
        )

    summary = dict(pnl_payload.get("summary") or {})
    total_hours = sum(float(market.get("hours_in_reward_zone") or 0.0) for market in markets)
    total_modeled_net = float(summary.get("net_after_reward_and_cost_usdc") or 0.0)
    total_net = float(summary.get("verified_net_after_reward_and_cost_usdc") or total_modeled_net)
    recommended_maker_take_share = min(recommended_values) if recommended_values else round(current_maker_take_share * 0.5, 6)
    allow_live = bool(
        markets
        and total_hours >= min_hours
        and total_net >= min_net_usdc
        and all(item["reason"] == "OK" for item in calibrated_markets)
    )

    # 奖励实测 vs 估算比较
    total_reward_estimated = float(summary.get("reward_accrued_estimate_usdc") or 0.0)
    total_reward_actual    = float(summary.get("reward_accrued_actual_usdc") or 0.0)
    reward_ratio = round(total_reward_actual / total_reward_estimated, 6) if total_reward_estimated > 1e-9 else None
    reward_calibration_note = (
        "NO_ACTUAL_DATA — reward ratio requires live session with completed epoch"
        if reward_ratio is None
        else (
            f"actual/estimated = {reward_ratio:.3f} — "
            + ("model OVERESTIMATES reward" if reward_ratio < 0.8 else
               "model UNDERESTIMATES reward" if reward_ratio > 1.2 else
               "model reward estimate is ACCURATE")
        )
    )

    return {
        "report_type": "reward_profit_model_calibration",
        "session_id": state_payload.get("session_id") or pnl_payload.get("session_id"),
        "source_cycle_index": summary.get("cycle_index"),
        "current_maker_take_share": round(current_maker_take_share, 6),
        "recommended_maker_take_share": round(recommended_maker_take_share, 6),
        "allow_live": allow_live,
        "summary": {
            "market_count": len(markets),
            "total_hours_in_reward_zone": round(total_hours, 6),
            "total_net_after_cost_usdc": round(total_net, 6),
            "total_modeled_net_after_cost_usdc": round(total_modeled_net, 6),
            "min_required_hours": round(min_hours, 6),
            "min_required_net_usdc": round(min_net_usdc, 6),
        },
        "reward_calibration": {
            "total_reward_estimated_usdc": round(total_reward_estimated, 6),
            "total_reward_actual_usdc": round(total_reward_actual, 6),
            "actual_vs_estimate_ratio": reward_ratio,
            "note": reward_calibration_note,
            "run_track_reward_correlation": "python scripts/track_reward_correlation.py",
        },
        "markets": sorted(
            calibrated_markets,
            key=lambda row: (row["reason"] != "OK", row["recommended_maker_take_share"], str(row["market_slug"])),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate reward-profit maker assumptions from dry-run/live reports.")
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "data" / "reports" / "auto_trade_profit_state_latest.json",
    )
    parser.add_argument(
        "--pnl",
        type=Path,
        default=ROOT / "data" / "reports" / "auto_trade_profit_pnl_latest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "reports" / "reward_profit_calibration_latest.json",
    )
    parser.add_argument("--current-maker-take-share", type=float, default=0.30)
    parser.add_argument("--min-hours", type=float, default=1.0)
    parser.add_argument("--min-net-usdc", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_payload = _load_json(args.state)
    pnl_payload = _load_json(args.pnl)
    report = build_calibration_report(
        state_payload=state_payload,
        pnl_payload=pnl_payload,
        current_maker_take_share=args.current_maker_take_share,
        min_hours=args.min_hours,
        min_net_usdc=args.min_net_usdc,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("REWARD PROFIT CALIBRATION")
    print(f"Markets:                    {report['summary']['market_count']}")
    print(f"Total reward-zone hours:    {report['summary']['total_hours_in_reward_zone']:.4f}")
    print(f"Total net after cost:       ${report['summary']['total_net_after_cost_usdc']:.4f}")
    print(f"Recommended maker share:    {report['recommended_maker_take_share']:.4f}")
    print(f"Allow live:                 {report['allow_live']}")
    print(f"Report:                     {args.out.resolve()}")


if __name__ == "__main__":
    main()
