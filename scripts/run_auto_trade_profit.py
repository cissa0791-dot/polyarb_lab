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
    parser.add_argument("--kelly-multiplier", type=float, default=None)
    parser.add_argument("--kelly-horizon-hours", type=float, default=1.0)
    parser.add_argument("--action-mode", choices=["legacy", "optimal"], default="legacy")
    parser.add_argument("--quote-search-mode", choices=["legacy", "best_ev"], default="legacy")
    parser.add_argument("--target-inventory-usdc-per-market", type=float, default=0.0)
    parser.add_argument("--max-total-inventory-usdc", type=float, default=0.0)
    parser.add_argument("--max-total-open-buy-usdc", type=float, default=0.0)
    parser.add_argument("--max-account-open-buy-orders", type=int, default=0)
    parser.add_argument("--min-action-edge-usdc", type=float, default=0.0)
    parser.add_argument("--profit-evidence-mode", choices=["off", "strict"], default="off")
    parser.add_argument("--actual-reward-warmup-minutes", type=float, default=0.0)
    parser.add_argument("--strategy-set", type=str, default="reward_mm,inventory_manager")
    parser.add_argument("--scale-mode", choices=["off", "evidence_gated"], default="evidence_gated")
    parser.add_argument("--risk-profile", choices=["standard", "canary"], default="standard")
    parser.add_argument("--max-verified-drawdown-usdc", type=float, default=0.0)
    parser.add_argument("--min-actual-reward-window-usdc", type=float, default=0.0)
    parser.add_argument("--min-verified-net-window-usdc", type=float, default=0.0)
    parser.add_argument("--disable-new-buys", action="store_true")
    parser.add_argument("--inventory-manager-only", action="store_true")
    parser.add_argument("--arb-scan-only", action="store_true", help="Run the arbitrage / relative-value scanner without placing or canceling orders.")
    parser.add_argument("--arb-scan-min-edge", type=float, default=0.02)
    parser.add_argument(
        "--edge-evidence-path",
        type=str,
        default=str(ROOT / "data" / "reports" / "live_edge_observations.jsonl"),
    )
    parser.add_argument("--record-orderbook-snapshots", action="store_true")
    parser.add_argument(
        "--orderbook-snapshot-path",
        type=str,
        default=str(ROOT / "data" / "reports" / "live_orderbook_snapshots.jsonl"),
    )
    parser.add_argument("--orderbook-snapshot-max-markets", type=int, default=20)
    parser.add_argument("--orderbook-snapshot-include-filtered", action="store_true")
    parser.add_argument("--orderbook-snapshot-filtered-max", type=int, default=60)
    parser.add_argument("--orderbook-snapshot-min-score", type=float, default=0.0)
    parser.add_argument("--enable-evidence-market-filter", action="store_true")
    parser.add_argument(
        "--evidence-market-intel-path",
        type=str,
        default=str(ROOT / "data" / "reports" / "evidence_market_intel_latest.json"),
    )
    parser.add_argument("--max-order-rejects-per-hour", type=int, default=0)
    parser.add_argument("--quote-refresh-mode", choices=["cycle", "event_driven"], default="cycle")
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
    parser.add_argument("--live-order-max-age-sec", type=float, default=120.0)
    parser.add_argument("--live-requote-price-move-cents", type=float, default=1.0)
    parser.add_argument("--quote-mode", choices=["passive", "midpoint", "adaptive", "urgent"], default="adaptive")
    parser.add_argument(
        "--quote-improvement-cents",
        type=float,
        default=0.1,
        help="Improve maker bid by this many cents, without crossing the ask. Example: 0.1 posts 46.3c when bid/ask are 46.2c/46.4c.",
    )
    parser.add_argument("--max-quote-improvement-cents", type=float, default=0.3)
    parser.add_argument(
        "--max-quote-improvement-cost-usdc",
        type=float,
        default=0.25,
        help="Maximum extra inventory cost from quote improvement per market.",
    )
    parser.add_argument("--max-no-fill-requotes", type=int, default=2)
    parser.add_argument("--no-fill-cooldown-minutes", type=float, default=30.0)
    parser.add_argument("--max-inventory-shares-per-market", type=float, default=None)
    parser.add_argument("--max-inventory-usdc-per-market", type=float, default=None)
    parser.add_argument("--inventory-dust-shares", type=float, default=0.0001)
    parser.add_argument(
        "--min-live-order-size-shares",
        type=float,
        default=5.0,
        help="Minimum live order size in shares. Balances below this after dust are treated as non-tradeable residue.",
    )
    parser.add_argument(
        "--inventory-policy",
        choices=["sell_only", "balanced"],
        default="sell_only",
        help="Live inventory handling. sell_only blocks new buys while inventory exists; balanced allows two-sided quoting until inventory limits are hit.",
    )
    parser.add_argument("--event-limit", type=int, default=200)
    parser.add_argument("--market-limit", type=int, default=400)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-sec", type=int, default=0)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--keep-open-orders", action="store_true", help="Do not cancel live open orders when the run exits.")
    parser.add_argument("--exclude-market-slug", action="append", dest="excluded_market_slugs", default=[], metavar="SLUG", help="Skip this market slug (repeatable).")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kelly_fraction_scale = args.kelly_multiplier if args.kelly_multiplier is not None else args.kelly_fraction_scale
    if args.risk_profile == "canary" and args.kelly_multiplier is None:
        kelly_fraction_scale = min(kelly_fraction_scale, 0.10)
    max_inventory_shares = (
        float(args.max_inventory_shares_per_market)
        if args.max_inventory_shares_per_market is not None
        else (50.0 if args.live else 0.0)
    )
    max_inventory_usdc = (
        float(args.max_inventory_usdc_per_market)
        if args.max_inventory_usdc_per_market is not None
        else (min(float(args.per_market_cap), 25.0) if args.live else 0.0)
    )
    optimal_live = bool(args.live and args.action_mode == "optimal")
    target_inventory_usdc = args.target_inventory_usdc_per_market or (40.0 if optimal_live else 0.0)
    max_total_inventory_usdc = args.max_total_inventory_usdc or (160.0 if optimal_live else 0.0)
    max_total_open_buy_usdc = args.max_total_open_buy_usdc or (80.0 if optimal_live else 0.0)
    max_account_open_buy_orders = args.max_account_open_buy_orders or (2 if optimal_live else 0)
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
        kelly_fraction_scale=kelly_fraction_scale,
        kelly_horizon_hours=args.kelly_horizon_hours,
        action_mode=args.action_mode,
        quote_search_mode=args.quote_search_mode,
        target_inventory_usdc_per_market=target_inventory_usdc,
        max_total_inventory_usdc=max_total_inventory_usdc,
        max_total_open_buy_usdc=max_total_open_buy_usdc,
        max_account_open_buy_orders=max_account_open_buy_orders,
        min_action_edge_usdc=args.min_action_edge_usdc,
        profit_evidence_mode=args.profit_evidence_mode,
        actual_reward_warmup_minutes=args.actual_reward_warmup_minutes,
        strategy_set=args.strategy_set,
        scale_mode=args.scale_mode,
        risk_profile=args.risk_profile,
        max_verified_drawdown_usdc=args.max_verified_drawdown_usdc,
        min_actual_reward_window_usdc=args.min_actual_reward_window_usdc,
        min_verified_net_window_usdc=args.min_verified_net_window_usdc,
        disable_new_buys=args.disable_new_buys,
        inventory_manager_only=args.inventory_manager_only,
        arb_scan_only=args.arb_scan_only,
        arb_scan_min_edge_usdc=args.arb_scan_min_edge,
        edge_evidence_path=args.edge_evidence_path,
        record_orderbook_snapshots=args.record_orderbook_snapshots,
        orderbook_snapshot_path=args.orderbook_snapshot_path,
        orderbook_snapshot_max_markets=args.orderbook_snapshot_max_markets,
        orderbook_snapshot_include_filtered=args.orderbook_snapshot_include_filtered,
        orderbook_snapshot_filtered_max=args.orderbook_snapshot_filtered_max,
        orderbook_snapshot_min_score=args.orderbook_snapshot_min_score,
        enable_evidence_market_filter=args.enable_evidence_market_filter,
        evidence_market_intel_path=args.evidence_market_intel_path,
        max_order_rejects_per_hour=args.max_order_rejects_per_hour,
        quote_refresh_mode=args.quote_refresh_mode,
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
        live_order_max_age_sec=args.live_order_max_age_sec,
        live_requote_price_move_cents=args.live_requote_price_move_cents,
        quote_mode=args.quote_mode,
        quote_improvement_cents=args.quote_improvement_cents,
        max_quote_improvement_cents=args.max_quote_improvement_cents,
        max_quote_improvement_cost_usdc=args.max_quote_improvement_cost_usdc,
        max_no_fill_requotes=args.max_no_fill_requotes,
        no_fill_cooldown_minutes=args.no_fill_cooldown_minutes,
        max_inventory_shares_per_market=max_inventory_shares,
        max_inventory_usdc_per_market=max_inventory_usdc,
        inventory_dust_shares=args.inventory_dust_shares,
        min_live_order_size_shares=args.min_live_order_size_shares,
        inventory_policy=args.inventory_policy,
        event_limit=args.event_limit,
        market_limit=args.market_limit,
        cycles=args.cycles,
        interval_sec=args.interval_sec,
        live=bool(args.live),
        verbose=bool(args.verbose),
        show_progress=not args.no_progress,
        cancel_open_orders_on_finish=not args.keep_open_orders,
        excluded_market_slugs=list(args.excluded_market_slugs),
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
