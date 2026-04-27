from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live.auth import load_live_credentials
from src.live.broker import LiveBroker
from src.live.client import LiveWriteClient
from src.live.reward_live_mm import (
    DEFAULT_REWARD_EVENTS,
    RewardLiveConfig,
    build_reward_live_executor,
)
from src.live.rewards import RewardClient

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated reward live MM loop")
    parser.add_argument("--events", nargs="+", default=DEFAULT_REWARD_EVENTS)
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--max-capital-per-market", type=float, default=1200.0)
    parser.add_argument("--max-markets", type=int, default=6)
    parser.add_argument("--max-markets-per-event", type=int, default=2)
    parser.add_argument("--cooldown-sessions", type=int, default=4)
    parser.add_argument("--max-drawdown-per-market", type=float, default=5.0)
    parser.add_argument("--max-daily-loss", type=float, default=25.0)
    parser.add_argument("--min-reward-minus-drawdown-per-hour", type=float, default=0.02)
    parser.add_argument("--min-reward-per-dollar-inventory-per-day", type=float, default=0.01)
    parser.add_argument("--min-implied-reward-ev-daily-usdc", type=float, default=0.25)
    parser.add_argument("--max-current-spread-cents", type=float, default=4.0)
    parser.add_argument("--event-fetch-limit", type=int, default=1500)
    parser.add_argument("--market-fetch-limit", type=int, default=3000)
    parser.add_argument("--replace-drift-cents", type=float, default=0.5)
    parser.add_argument("--gamma-host", default="https://gamma-api.polymarket.com")
    parser.add_argument("--clob-host", default="https://clob.polymarket.com")
    parser.add_argument("--state-path", default=str(ROOT / "data" / "reports" / "reward_live_state_latest.json"))
    parser.add_argument("--pnl-path", default=str(ROOT / "data" / "reports" / "reward_live_pnl_latest.json"))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--live", action="store_true", help="Enable real order submission")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run mode (default)")
    parser.add_argument("--signature-type", type=int, default=None)
    parser.add_argument("--funder", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = not args.live or args.dry_run

    if dry_run:
        write_client = LiveWriteClient(client=None, dry_run=True)  # type: ignore[arg-type]
        reward_client = RewardClient(address="dry_run", dry_run=True)
    else:
        creds = load_live_credentials()
        write_client = LiveWriteClient.from_credentials(
            creds,
            host=args.clob_host,
            dry_run=False,
            signature_type=args.signature_type,
            funder=args.funder,
        )
        reward_client = RewardClient.from_credentials(
            creds,
            host=args.clob_host,
            dry_run=False,
        )

    broker = LiveBroker(write_client)
    config = RewardLiveConfig(
        capital=args.capital,
        max_capital_per_market=args.max_capital_per_market,
        max_markets=args.max_markets,
        max_markets_per_event=args.max_markets_per_event,
        cooldown_sessions=args.cooldown_sessions,
        max_drawdown_per_market=args.max_drawdown_per_market,
        max_daily_loss=args.max_daily_loss,
        min_reward_minus_drawdown_per_hour=args.min_reward_minus_drawdown_per_hour,
        min_reward_per_dollar_inventory_per_day=args.min_reward_per_dollar_inventory_per_day,
        min_implied_reward_ev_daily_usdc=args.min_implied_reward_ev_daily_usdc,
        max_current_spread_cents=args.max_current_spread_cents,
        event_fetch_limit=args.event_fetch_limit,
        market_fetch_limit=args.market_fetch_limit,
        replace_drift_cents=args.replace_drift_cents,
        dry_run=dry_run,
    )
    executor = build_reward_live_executor(
        gamma_host=args.gamma_host,
        clob_host=args.clob_host,
        broker=broker,
        client=None if dry_run else write_client,
        reward_client=reward_client,
        config=config,
    )
    state_path = Path(args.state_path)
    pnl_path = Path(args.pnl_path)

    while True:
        state_report, pnl_report = executor.run_cycle(
            event_slugs=args.events,
            state_path=state_path,
            pnl_path=pnl_path,
        )
        print("\nREWARD LIVE MM")
        print(f"Mode: {'DRY_RUN' if dry_run else 'LIVE'}")
        print(f"Selected markets:      {state_report.selected_market_count}")
        print(f"Active quote markets:  {state_report.active_quote_market_count}")
        print(f"Capital in use:        ${state_report.total_capital_in_use_usdc:.4f}")
        print(f"Reward accrued est:    ${state_report.total_reward_accrued_estimate_usdc:.4f}")
        print(f"Inventory MTM PnL:     ${state_report.total_inventory_mtm_pnl_usdc:.4f}")
        print(f"Spread realized:       ${state_report.total_spread_realized_usdc:.4f}")
        print(f"Net after reward:      ${state_report.total_net_after_reward_usdc:.4f}")
        if state_report.halted:
            print(f"Halted:                {state_report.halt_reason}")
        print(f"State: {state_path}")
        print(f"PnL:   {pnl_path}")
        if not args.loop:
            break
        time.sleep(max(args.interval_sec, 1))


if __name__ == "__main__":
    main()
