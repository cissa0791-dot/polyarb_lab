from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REPORT_SCHEMA_VERSION = "profitability_proof.v1"
REPORT_TYPE = "profitability_proof"

NO_PROOF_STATUS = "NO_PROFITABILITY_PROOF_YET"
INSUFFICIENT_STATUS = "PROFITABILITY_EVIDENCE_INSUFFICIENT"
CONFIRMED_STATUS = "PROFITABILITY_CONFIRMED"
FAILED_STATUS = "PROFITABILITY_FAILED"


def build_profitability_proof(
    *,
    lifecycle_summary: dict[str, Any] | None = None,
    min_sample_count: int = 30,
    max_drawdown_usdc: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    lifecycle_summary = lifecycle_summary or {}
    event_count = int(lifecycle_summary.get("event_count") or 0)
    cash = lifecycle_summary.get("cash_accounting") if isinstance(lifecycle_summary.get("cash_accounting"), dict) else {}
    forecast = lifecycle_summary.get("forecast_accounting") if isinstance(lifecycle_summary.get("forecast_accounting"), dict) else {}
    realized_spread = float(cash.get("realized_spread_pnl_usdc") or lifecycle_summary.get("realized_spread_pnl_usdc") or 0.0)
    confirmed_reward = float(cash.get("confirmed_reward_usdc") or 0.0)
    fees = float(cash.get("fees_usdc") or lifecycle_summary.get("fees_usdc") or 0.0)
    adverse_selection_loss = float(cash.get("adverse_selection_loss_usdc") or 0.0)
    inventory_markdown = float(cash.get("inventory_markdown_usdc") or 0.0)
    net_realized = float(cash.get("realized_cash_pnl_usdc") or lifecycle_summary.get("realized_cash_pnl_usdc") or 0.0)
    pending_reward = float(forecast.get("pending_reward_usdc") or lifecycle_summary.get("pending_reward_usdc") or 0.0)
    unconfirmed_reward_mixed = (
        cash.get("estimated_reward_counted_as_realized_cash_pnl") is True
        or forecast.get("pending_reward_counted_as_confirmed_reward") is True
    )
    drawdown = float(cash.get("drawdown_usdc") or 0.0)

    blockers: list[str] = []
    if event_count <= 0:
        status = NO_PROOF_STATUS
        blockers.append("NO_REAL_EVIDENCE_EVENTS")
    elif unconfirmed_reward_mixed:
        status = FAILED_STATUS
        blockers.append("UNCONFIRMED_REWARD_MIXED_INTO_CASH_PNL")
    elif event_count < min_sample_count:
        status = INSUFFICIENT_STATUS
        blockers.append("SAMPLE_SIZE_TOO_SMALL")
    elif net_realized <= 0:
        status = FAILED_STATUS
        blockers.append("NET_REALIZED_PNL_NOT_POSITIVE")
    elif max_drawdown_usdc is not None and drawdown > max_drawdown_usdc:
        status = FAILED_STATUS
        blockers.append("DRAWDOWN_EXCEEDS_LIMIT")
    else:
        status = CONFIRMED_STATUS

    return {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "status": status,
        "sample_count": event_count,
        "min_sample_count": min_sample_count,
        "realized_spread_pnl_usdc": round(realized_spread, 6),
        "confirmed_reward_usdc": round(confirmed_reward, 6),
        "fees_usdc": round(fees, 6),
        "adverse_selection_loss_usdc": round(adverse_selection_loss, 6),
        "inventory_markdown_usdc": round(inventory_markdown, 6),
        "net_realized_pnl_usdc": round(net_realized, 6),
        "pending_reward_usdc": round(pending_reward, 6),
        "pending_reward_counted_as_profit": False,
        "unconfirmed_reward_mixed": unconfirmed_reward_mixed,
        "drawdown_usdc": round(drawdown, 6),
        "profitability_claimed": status == CONFIRMED_STATUS,
        "blockers": blockers,
        "one_line_verdict": _verdict(status, blockers),
    }


def _verdict(status: str, blockers: list[str]) -> str:
    if status == CONFIRMED_STATUS:
        return "PROFITABILITY_CONFIRMED: enough confirmed cash evidence supports positive net realized PnL."
    return f"{status}: {', '.join(blockers) or 'not enough confirmed evidence'}."
