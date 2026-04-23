"""
Forward-test runner — paper simulation with win-rate and return-rate summary.

Runs N cycles of paper trading against live markets, then queries the SQLite
database to compute trade-level win rate and return rate.

Usage:
    python scripts/run_forward_test.py --cycles 200 --cash 5000 --sleep 3
    python scripts/run_forward_test.py --cycles 500 --cash 10000 --sleep 2
"""
from __future__ import annotations

import argparse
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.runner import ResearchRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper forward-test with win-rate and return-rate report.")
    parser.add_argument("--cycles",   type=int,   default=200,    help="Number of scan cycles (default: 200)")
    parser.add_argument("--cash",     type=float, default=5000.0, help="Starting paper cash in USD (default: 5000)")
    parser.add_argument("--sleep",    type=float, default=2.0,    help="Seconds between cycles (default: 2)")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--constraints", default="config/constraints.yaml")
    return parser.parse_args()


def print_progress(cycle: int, total: int, summary) -> None:
    elapsed = (summary.ended_ts - summary.started_ts).total_seconds()
    bar_filled = int(40 * cycle / total)
    bar = "█" * bar_filled + "░" * (40 - bar_filled)
    print(
        f"\r[{bar}] {cycle}/{total}  "
        f"cand={summary.candidates_generated}  "
        f"orders={summary.paper_orders_created}  "
        f"rpnl={summary.realized_pnl:+.4f}  "
        f"t={elapsed:.0f}s",
        end="",
        flush=True,
    )


def print_final_report(db_path: Path, starting_cash: float, session_start: datetime) -> None:
    if not db_path.exists():
        print("\n[warn] Database not found — no trades recorded yet.")
        return

    db = sqlite3.connect(str(db_path))
    cur = db.cursor()
    start_str = session_start.strftime("%Y-%m-%d %H:%M:%S")

    # All closed trades in this session
    cur.execute(
        """
        SELECT realized_pnl_usd, entry_cost_usd, holding_duration_sec, market_slug
        FROM trade_summaries
        WHERE closed_ts >= ?
        ORDER BY closed_ts
        """,
        (start_str,),
    )
    rows = cur.fetchall()
    db.close()

    total_trades = len(rows)
    if total_trades == 0:
        print("\n\n" + "=" * 60)
        print("  没有完成的交易记录。")
        print("  可能需要更多轮次让仓位完整开关。")
        print("  建议增加 --cycles 数量后重试。")
        print("=" * 60)
        return

    winning     = [r for r in rows if r[0] > 0]
    losing      = [r for r in rows if r[0] < 0]
    breakeven   = [r for r in rows if r[0] == 0]

    total_pnl   = sum(r[0] for r in rows)
    gross_profit = sum(r[0] for r in winning)
    gross_loss   = abs(sum(r[0] for r in losing))
    win_rate     = len(winning) / total_trades * 100
    return_rate  = total_pnl / starting_cash * 100
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win      = gross_profit / len(winning) if winning else 0.0
    avg_loss     = gross_loss / len(losing) if losing else 0.0
    avg_hold_sec = sum(r[2] for r in rows) / total_trades

    # Per-market breakdown (top 10 by PnL)
    market_pnl: dict[str, float] = {}
    for pnl, _, _, slug in rows:
        market_pnl[slug] = market_pnl.get(slug, 0.0) + pnl
    top_markets = sorted(market_pnl.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n\n" + "=" * 60)
    print("  FORWARD TEST 结果报告")
    print("=" * 60)
    print(f"  起始资金      : ${starting_cash:,.2f}")
    print(f"  已实现盈亏    : ${total_pnl:+,.4f}")
    print(f"  最终资金(估)  : ${starting_cash + total_pnl:,.2f}")
    print(f"  回报率        : {return_rate:+.4f}%")
    print("-" * 60)
    print(f"  总交易次数    : {total_trades}")
    print(f"  盈利次数      : {len(winning)}  ({win_rate:.1f}%)")
    print(f"  亏损次数      : {len(losing)}")
    print(f"  平手次数      : {len(breakeven)}")
    print(f"  胜率          : {win_rate:.2f}%")
    print("-" * 60)
    print(f"  平均盈利/笔   : ${avg_win:+.4f}")
    print(f"  平均亏损/笔   : ${avg_loss:-.4f}")
    print(f"  盈亏比        : {profit_factor:.3f}")
    print(f"  平均持仓时间  : {avg_hold_sec:.1f} 秒 ({avg_hold_sec/60:.1f} 分钟)")
    if top_markets:
        print("-" * 60)
        print("  市场盈亏 TOP 10:")
        for slug, pnl in top_markets:
            bar = "▲" if pnl >= 0 else "▼"
            print(f"    {bar} {slug[:48]:<48}  ${pnl:+.4f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    runner = ResearchRunner(
        settings_path=args.settings,
        constraints_path=args.constraints,
    )
    runner.config.paper.starting_cash = args.cash
    runner.config.market_data.market_limit = getattr(runner.config.market_data, "market_limit", 200)

    db_path = ROOT / "data" / "processed" / "paper.db"
    session_start = datetime.now(timezone.utc)

    # Allow Ctrl+C to exit gracefully and still print the report
    stopped_early = False

    def _handle_interrupt(sig, frame):
        nonlocal stopped_early
        stopped_early = True
        print("\n\n[Ctrl+C 检测到 — 正在生成报告...]")

    signal.signal(signal.SIGINT, _handle_interrupt)

    print(f"  起始资金: ${args.cash:,.0f}")
    print(f"  计划轮次: {args.cycles}  间隔: {args.sleep}s")
    print(f"  数据库  : {db_path}")
    print(f"  开始时间: {session_start.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

    last_summary = None
    for i in range(1, args.cycles + 1):
        if stopped_early:
            break
        try:
            summary = runner.run_once()
            last_summary = summary
            print_progress(i, args.cycles, summary)
            if i < args.cycles and args.sleep > 0 and not stopped_early:
                time.sleep(args.sleep)
        except Exception as exc:
            print(f"\n[cycle {i} error] {exc}", flush=True)

    completed = (last_summary is not None)
    if not completed:
        print("\n没有完成任何轮次。")
        return

    runner.store.close()
    runner.opportunity_store.engine.dispose()

    print_final_report(db_path, args.cash, session_start)


if __name__ == "__main__":
    main()
