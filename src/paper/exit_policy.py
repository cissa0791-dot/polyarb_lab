from __future__ import annotations

from datetime import datetime, timezone

from src.config_runtime.models import PaperConfig
from src.domain.models import ExitSignal, PositionMark


def evaluate_exit(mark: PositionMark, config: PaperConfig, force_reason: str | None = None) -> ExitSignal | None:
    now = datetime.now(timezone.utc)
    if force_reason:
        return ExitSignal(
            position_id=mark.position_id,
            candidate_id=mark.candidate_id,
            symbol=mark.symbol,
            market_slug=mark.market_slug,
            reason_code=force_reason,
            force_exit=True,
            expected_exit_price=mark.mark_price,
            expected_unrealized_pnl_usd=mark.unrealized_pnl_usd,
            age_sec=mark.age_sec,
            ts=now,
        )

    if config.edge_decay_bid_delta > 0 and (mark.mark_price - mark.avg_entry_price) >= config.edge_decay_bid_delta:
        return ExitSignal(
            position_id=mark.position_id,
            candidate_id=mark.candidate_id,
            symbol=mark.symbol,
            market_slug=mark.market_slug,
            reason_code="EDGE_DECAY",
            expected_exit_price=mark.mark_price,
            expected_unrealized_pnl_usd=mark.unrealized_pnl_usd,
            age_sec=mark.age_sec,
            ts=now,
        )

    if config.take_profit_usd > 0 and mark.unrealized_pnl_usd >= config.take_profit_usd:
        return ExitSignal(
            position_id=mark.position_id,
            candidate_id=mark.candidate_id,
            symbol=mark.symbol,
            market_slug=mark.market_slug,
            reason_code="TAKE_PROFIT",
            expected_exit_price=mark.mark_price,
            expected_unrealized_pnl_usd=mark.unrealized_pnl_usd,
            age_sec=mark.age_sec,
            ts=now,
        )

    if config.stop_loss_usd > 0 and mark.unrealized_pnl_usd <= -abs(config.stop_loss_usd):
        return ExitSignal(
            position_id=mark.position_id,
            candidate_id=mark.candidate_id,
            symbol=mark.symbol,
            market_slug=mark.market_slug,
            reason_code="STOP_LOSS",
            expected_exit_price=mark.mark_price,
            expected_unrealized_pnl_usd=mark.unrealized_pnl_usd,
            age_sec=mark.age_sec,
            ts=now,
        )

    if config.max_holding_sec > 0 and mark.age_sec >= config.max_holding_sec:
        return ExitSignal(
            position_id=mark.position_id,
            candidate_id=mark.candidate_id,
            symbol=mark.symbol,
            market_slug=mark.market_slug,
            reason_code="MAX_HOLDING_AGE",
            expected_exit_price=mark.mark_price,
            expected_unrealized_pnl_usd=mark.unrealized_pnl_usd,
            age_sec=mark.age_sec,
            ts=now,
        )

    return None
