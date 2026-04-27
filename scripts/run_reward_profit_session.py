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
    parser = argparse.ArgumentParser(description="Run reward profit session with hard capital limits.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Enable live order submission and reward API polling.")
    mode.add_argument("--dry-run", action="store_true", help="Run without live orders. Default mode.")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--gamma-host", default="https://gamma-api.polymarket.com")
    parser.add_argument("--out-dir", default="data/reports")
    parser.add_argument("--state-path", default="data/reports/reward_profit_state_latest.json")
    parser.add_argument("--pnl-path", default="data/reports/reward_profit_pnl_latest.json")
    parser.add_argument("--capital", type=float, default=300.0)
    parser.add_argument("--per-market-cap", type=float, default=120.0)
    parser.add_argument("--max-markets", type=int, default=2)
    parser.add_argument("--max-markets-per-event", type=int, default=1)
    parser.add_argument("--max-drawdown-per-market", type=float, default=3.0)
    parser.add_argument("--max-daily-loss", type=float, default=10.0)
    parser.add_argument("--max-entry-cost-usdc", type=float, default=0.5)
    parser.add_argument("--max-entry-cost-pct", type=float, default=0.02)
    parser.add_argument("--max-break-even-hours", type=float, default=0.5)
    parser.add_argument("--min-reward-minus-drawdown-per-hour", type=float, default=0.02)
    parser.add_argument("--min-reward-per-dollar-inventory-per-hour", type=float, default=0.0005)
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--market-limit", type=int, default=400)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-sec", type=int, default=0)
    parser.add_argument("--reset-state", action="store_true", help="Delete latest state/pnl before starting a new session.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RewardProfitConfig(
        settings_path=args.settings,
        gamma_host=args.gamma_host,
        out_dir=args.out_dir,
        state_path=args.state_path,
        pnl_path=args.pnl_path,
        capital_limit_usdc=args.capital,
        per_market_cap_usdc=args.per_market_cap,
        max_markets=args.max_markets,
        max_markets_per_event=args.max_markets_per_event,
        max_drawdown_per_market=args.max_drawdown_per_market,
        max_daily_loss=args.max_daily_loss,
        max_entry_cost_usdc=args.max_entry_cost_usdc,
        max_entry_cost_pct=args.max_entry_cost_pct,
        max_break_even_hours=args.max_break_even_hours,
        min_reward_minus_drawdown_per_hour=args.min_reward_minus_drawdown_per_hour,
        min_reward_per_dollar_inventory_per_hour=args.min_reward_per_dollar_inventory_per_hour,
        event_limit=args.event_limit,
        market_limit=args.market_limit,
        cycles=args.cycles,
        interval_sec=args.interval_sec,
        live=bool(args.live),
        verbose=bool(args.verbose),
    )
    if args.reset_state:
        for path_str in (config.state_path, config.pnl_path):
            path = Path(path_str)
            if path.exists():
                path.unlink()
    engine = RewardProfitSessionEngine(config)
    state, pnl_report = engine.run()

    summary = pnl_report["summary"]
    print("")
    print("REWARD PROFIT SESSION")
    print(f"Mode:                 {summary['mode']}")
    print(f"Cycle:                {summary['cycle_index']}")
    print(f"Selected markets:     {len(state.selected_market_slugs)}")
    print(f"Active quote markets: {summary['active_quote_market_count']}")
    print(f"Capital in use:       ${summary['capital_in_use_usdc']:.4f}")
    print(f"Reward est:           ${summary['reward_accrued_estimate_usdc']:.4f}")
    print(f"Reward actual:        ${summary['reward_accrued_actual_usdc']:.4f}")
    print(f"Inventory MTM PnL:    ${summary['inventory_mtm_pnl_usdc']:.4f}")
    print(f"Spread realized:      ${summary['spread_realized_usdc']:.4f}")
    print(f"Net after reward:     ${summary['net_after_reward_usdc']:.4f}")
    print(f"Net after cost:       ${summary['net_after_reward_and_cost_usdc']:.4f}")
    print(f"Halted:               {summary['halted']}")
    print(f"State:                {Path(config.state_path).resolve()}")
    print(f"PnL:                  {Path(config.pnl_path).resolve()}")
    if args.verbose:
        print(json.dumps(pnl_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
