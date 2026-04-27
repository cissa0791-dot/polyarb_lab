from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.reward_profit_session import RewardProfitConfig, RewardProfitSessionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict reward-profit auto-trade session.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Submit real orders.")
    mode.add_argument("--dry-run", action="store_true", help="No real orders. Default.")
    parser.add_argument("--capital", type=float, default=300.0)
    parser.add_argument("--per-market-cap", type=float, default=120.0)
    parser.add_argument("--max-markets", type=int, default=2)
    parser.add_argument("--max-markets-per-event", type=int, default=1)
    parser.add_argument("--max-drawdown-per-market", type=float, default=20.0)
    parser.add_argument("--max-drawdown-pct-of-capital", type=float, default=0.10)
    parser.add_argument("--max-daily-loss", type=float, default=10.0)
    parser.add_argument("--max-entry-cost-usdc", type=float, default=0.0)
    parser.add_argument("--max-entry-cost-pct", type=float, default=0.0)
    parser.add_argument("--max-break-even-hours", type=float, default=0.0)
    parser.add_argument("--max-true-break-even-hours", type=float, default=1.25)
    parser.add_argument("--min-reward-minus-drawdown-per-hour", type=float, default=0.02)
    parser.add_argument("--min-reward-per-dollar-inventory-per-hour", type=float, default=0.0005)
    parser.add_argument("--projection-horizon-hours", type=float, default=1.0)
    parser.add_argument("--min-projected-net-at-horizon-usdc", type=float, default=0.05)
    parser.add_argument("--max-quoting-hours-without-fills", type=float, default=1.0)
    parser.add_argument("--reward-share-floor", type=float, default=0.005)
    parser.add_argument("--reward-share-ceiling", type=float, default=0.02)
    parser.add_argument("--reward-share-quality-weight", type=float, default=0.015)
    parser.add_argument("--maker-take-share", type=float, default=0.30)
    parser.add_argument("--drawdown-factor-per-day", type=float, default=0.30)
    parser.add_argument("--reward-calibration-factor", type=float, default=1.0)
    parser.add_argument("--use-kelly-sizing", action="store_true")
    parser.add_argument("--kelly-fraction-scale", type=float, default=0.25)
    parser.add_argument("--kelly-horizon-hours", type=float, default=1.0)
    parser.add_argument("--entry-mode", choices=["maker_first", "inventory_first"], default="maker_first")
    parser.add_argument("--exit-cooldown-minutes", type=float, default=45.0)
    parser.add_argument("--repeat-exit-cooldown-minutes", type=float, default=90.0)
    parser.add_argument("--max-reentries-per-market", type=int, default=1)
    parser.add_argument("--disable-dry-run-fill-simulation", action="store_true")
    parser.add_argument("--actual-reward-zero-cycle-limit", type=int, default=0)
    parser.add_argument("--min-actual-reward-delta-usdc", type=float, default=0.0)
    parser.add_argument("--min-daily-reward-for-actual-gate-usdc", type=float, default=1.0)
    parser.add_argument("--max-adverse-midpoint-move-cents-per-hour", type=float, default=0.25)
    parser.add_argument("--min-inventory-risk-coverage-ratio", type=float, default=1.10)
    parser.add_argument("--enable-market-intel-filter", action="store_true")
    parser.add_argument("--market-intel-path", type=str, default=str(ROOT / "data" / "reports" / "market_intel_latest.json"))
    parser.add_argument("--max-market-intel-risk-score", type=float, default=0.70)
    parser.add_argument("--exit-target-capture-ratio", type=float, default=0.85)
    parser.add_argument("--volume-spike-multiple", type=float, default=3.0)
    parser.add_argument("--stale-thesis-hours", type=float, default=24.0)
    parser.add_argument("--stale-thesis-max-price-change", type=float, default=0.02)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--market-limit", type=int, default=400)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-sec", type=int, default=0)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--keep-open-orders", action="store_true", help="Do not cancel live open orders when the run exits.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RewardProfitConfig(
        out_dir=str(ROOT / "data" / "reports"),
        state_path=str(ROOT / "data" / "reports" / "auto_trade_profit_state_latest.json"),
        pnl_path=str(ROOT / "data" / "reports" / "auto_trade_profit_pnl_latest.json"),
        capital_limit_usdc=args.capital,
        per_market_cap_usdc=args.per_market_cap,
        max_markets=args.max_markets,
        max_markets_per_event=args.max_markets_per_event,
        max_drawdown_per_market=args.max_drawdown_per_market,
        max_drawdown_pct_of_capital=args.max_drawdown_pct_of_capital,
        max_daily_loss=args.max_daily_loss,
        max_entry_cost_usdc=args.max_entry_cost_usdc,
        max_entry_cost_pct=args.max_entry_cost_pct,
        max_break_even_hours=args.max_break_even_hours,
        max_true_break_even_hours=args.max_true_break_even_hours,
        min_reward_minus_drawdown_per_hour=args.min_reward_minus_drawdown_per_hour,
        min_reward_per_dollar_inventory_per_hour=args.min_reward_per_dollar_inventory_per_hour,
        projection_horizon_hours=args.projection_horizon_hours,
        min_projected_net_at_horizon_usdc=args.min_projected_net_at_horizon_usdc,
        max_quoting_hours_without_fills=args.max_quoting_hours_without_fills,
        reward_share_floor=args.reward_share_floor,
        reward_share_ceiling=args.reward_share_ceiling,
        reward_share_quality_weight=args.reward_share_quality_weight,
        maker_take_share=args.maker_take_share,
        drawdown_factor_per_day=args.drawdown_factor_per_day,
        reward_calibration_factor=args.reward_calibration_factor,
        use_kelly_sizing=args.use_kelly_sizing,
        kelly_fraction_scale=args.kelly_fraction_scale,
        kelly_horizon_hours=args.kelly_horizon_hours,
        entry_mode=args.entry_mode,
        exit_cooldown_minutes=args.exit_cooldown_minutes,
        repeat_exit_cooldown_minutes=args.repeat_exit_cooldown_minutes,
        max_reentries_per_market=args.max_reentries_per_market,
        dry_run_fill_simulation=not args.disable_dry_run_fill_simulation,
        actual_reward_zero_cycle_limit=args.actual_reward_zero_cycle_limit,
        min_actual_reward_delta_usdc=args.min_actual_reward_delta_usdc,
        min_daily_reward_for_actual_gate_usdc=args.min_daily_reward_for_actual_gate_usdc,
        max_adverse_midpoint_move_cents_per_hour=args.max_adverse_midpoint_move_cents_per_hour,
        min_inventory_risk_coverage_ratio=args.min_inventory_risk_coverage_ratio,
        enable_market_intel_filter=args.enable_market_intel_filter,
        market_intel_path=args.market_intel_path,
        max_market_intel_risk_score=args.max_market_intel_risk_score,
        exit_target_capture_ratio=args.exit_target_capture_ratio,
        volume_spike_multiple=args.volume_spike_multiple,
        stale_thesis_hours=args.stale_thesis_hours,
        stale_thesis_max_price_change=args.stale_thesis_max_price_change,
        event_limit=args.event_limit,
        market_limit=args.market_limit,
        cycles=args.cycles,
        interval_sec=args.interval_sec,
        live=bool(args.live),
        verbose=bool(args.verbose),
        show_progress=not args.no_progress,
        cancel_open_orders_on_finish=not args.keep_open_orders,
    )
    if args.reset_state:
        for path_str in (config.state_path, config.pnl_path):
            path = Path(path_str)
            if path.exists():
                path.unlink()

    state, pnl_report = RewardProfitSessionEngine(config).run()
    summary = pnl_report["summary"]
    print("AUTO TRADE PROFIT")
    print(f"Mode:                 {summary['mode']}")
    print(f"Cycle:                {summary['cycle_index']}")
    print(f"Selected markets:     {len(state.selected_market_slugs)}")
    print(f"Active quote markets: {summary['active_quote_market_count']}")
    print(f"Capital in use:       ${summary['capital_in_use_usdc']:.4f}")
    print(f"Modeled net after cost:  ${summary['net_after_reward_and_cost_usdc']:.4f}")
    print(f"Verified net after cost: ${summary['verified_net_after_reward_and_cost_usdc']:.4f}")
    print(f"Halted:               {summary['halted']}")
    print(f"State:                {Path(config.state_path).resolve()}")
    print(f"PnL:                  {Path(config.pnl_path).resolve()}")
    if args.verbose:
        print(json.dumps(pnl_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
