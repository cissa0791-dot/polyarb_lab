from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from src.domain.models import AccountSnapshot, PositionMark, PositionSnapshot, PositionState, TradeSummary


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass
class Fill:
    order_id: str
    position_id: str | None
    candidate_id: str | None
    symbol: str
    market_slug: str
    side: str
    shares: float
    price: float
    fee_usd: float
    ts: str


@dataclass
class PaperPosition:
    symbol: str
    market_slug: str
    shares: float = 0.0
    avg_price: float = 0.0


@dataclass
class PaperPositionRecord:
    position_id: str
    candidate_id: str | None
    symbol: str
    market_slug: str
    opened_ts: str
    state: PositionState = PositionState.OPENED
    closed_ts: str | None = None
    close_reason: str | None = None
    total_entry_shares: float = 0.0
    total_exit_shares: float = 0.0
    entry_notional_usd: float = 0.0
    exit_notional_usd: float = 0.0
    total_entry_fees_usd: float = 0.0
    remaining_entry_fee_usd: float = 0.0
    total_exit_fees_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    last_mark_price: float | None = None
    last_marked_value_usd: float = 0.0
    last_unrealized_pnl_usd: float = 0.0
    peak_unrealized_pnl_usd: float = 0.0
    first_adverse_ts: str | None = None
    last_mark_ts: str | None = None
    repricing_event_count: int = 0
    edge_decay_candidate_count: int = 0

    @property
    def remaining_shares(self) -> float:
        return max(0.0, self.total_entry_shares - self.total_exit_shares)

    @property
    def avg_entry_price(self) -> float:
        if self.total_entry_shares <= 1e-9:
            return 0.0
        return self.entry_notional_usd / self.total_entry_shares

    @property
    def entry_cost_usd(self) -> float:
        return self.entry_notional_usd + self.total_entry_fees_usd

    @property
    def exit_proceeds_usd(self) -> float:
        return self.exit_notional_usd

    @property
    def total_fees_usd(self) -> float:
        return self.total_entry_fees_usd + self.total_exit_fees_usd

    @property
    def remaining_entry_cost_usd(self) -> float:
        return (self.remaining_shares * self.avg_entry_price) + self.remaining_entry_fee_usd

    @property
    def is_open(self) -> bool:
        return self.remaining_shares > 1e-9


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    market_slug: str
    side: str
    shares: float
    limit_price: float
    created_ts: str
    candidate_id: str | None = None
    position_id: str | None = None
    reserved_cash: float = 0.0
    filled_shares: float = 0.0

    @property
    def remaining_shares(self) -> float:
        return max(0.0, self.shares - self.filled_shares)


@dataclass
class Ledger:
    cash: float = 10000.0
    frozen_cash: float = 0.0
    realized_pnl: float = 0.0
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    position_records: Dict[str, PaperPositionRecord] = field(default_factory=dict)
    orders: Dict[str, PaperOrder] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)

    @property
    def available_cash(self) -> float:
        return self.cash - self.frozen_cash

    def place_limit_order(
        self,
        order_id: str,
        symbol: str,
        market_slug: str,
        side: str,
        shares: float,
        limit_price: float,
        ts: str | None = None,
        candidate_id: str | None = None,
        position_id: str | None = None,
    ) -> bool:
        if shares <= 0 or limit_price <= 0:
            return False

        side_name = side.upper()
        if side_name == "BUY":
            reserve = shares * limit_price
            if reserve > self.available_cash + 1e-9:
                return False
        elif side_name == "SELL":
            reserve = 0.0
            resolved_position = self._resolve_position_record(position_id=position_id, symbol=symbol)
            if resolved_position is None or resolved_position.remaining_shares + 1e-9 < shares:
                return False
            position_id = resolved_position.position_id
        else:
            return False

        if side_name == "BUY":
            self.frozen_cash += reserve

        self.orders[order_id] = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            market_slug=market_slug,
            side=side_name,
            shares=shares,
            limit_price=limit_price,
            created_ts=ts or _now_iso(),
            candidate_id=candidate_id,
            position_id=position_id or (order_id if side_name == "BUY" else position_id),
            reserved_cash=reserve,
        )
        return True

    def apply_fill(self, order_id: str, shares: float, price: float, fee_usd: float = 0.0, ts: str | None = None) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            return False
        if shares <= 0 or price <= 0 or shares > order.remaining_shares + 1e-9:
            return False

        fill_ts = ts or _now_iso()
        side_name = order.side.upper()
        position_id = order.position_id or order_id

        if side_name == "BUY":
            reserved_for_fill = order.limit_price * shares
            actual_cost = price * shares + fee_usd
            self.frozen_cash = max(0.0, self.frozen_cash - reserved_for_fill)
            self.cash -= actual_cost
            self._apply_buy_fill(position_id, order, shares, price, fee_usd, fill_ts)
        else:
            proceeds = price * shares - fee_usd
            self.cash += proceeds
            self._apply_sell_fill(position_id, order, shares, price, fee_usd, fill_ts)

        order.filled_shares += shares
        if side_name == "BUY":
            order.reserved_cash = max(0.0, order.limit_price * order.remaining_shares)

        self.fills.append(
            Fill(
                order_id=order_id,
                position_id=position_id,
                candidate_id=order.candidate_id,
                symbol=order.symbol,
                market_slug=order.market_slug,
                side=order.side,
                shares=shares,
                price=price,
                fee_usd=fee_usd,
                ts=fill_ts,
            )
        )
        return True

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.remaining_shares <= 1e-9:
            return False
        if order.side == "BUY" and order.reserved_cash > 0:
            self.frozen_cash = max(0.0, self.frozen_cash - order.reserved_cash)
            order.reserved_cash = 0.0
        order.filled_shares = order.shares
        return True

    def mark_position(
        self,
        position_id: str,
        mark_price: float,
        ts: datetime | None = None,
        source_bid: float | None = None,
        source_ask: float | None = None,
        method: str = "best_bid",
    ) -> PositionMark | None:
        position = self.position_records.get(position_id)
        if position is None or not position.is_open or mark_price < 0:
            return None

        mark_ts = ts or datetime.now(timezone.utc)
        previous_mark_price = position.last_mark_price
        marked_value = position.remaining_shares * mark_price
        unrealized = marked_value - position.remaining_entry_cost_usd
        if previous_mark_price is not None and abs(mark_price - previous_mark_price) > 1e-9:
            position.repricing_event_count += 1
        position.last_mark_price = mark_price
        position.last_marked_value_usd = marked_value
        position.last_unrealized_pnl_usd = unrealized
        if unrealized > position.peak_unrealized_pnl_usd:
            position.peak_unrealized_pnl_usd = unrealized
        if unrealized < 0 and position.first_adverse_ts is None:
            position.first_adverse_ts = mark_ts.isoformat()
        position.last_mark_ts = mark_ts.isoformat()

        return PositionMark(
            position_id=position.position_id,
            candidate_id=position.candidate_id,
            symbol=position.symbol,
            market_slug=position.market_slug,
            state=position.state,
            shares=round(position.remaining_shares, 6),
            avg_entry_price=round(position.avg_entry_price, 6),
            source_bid=source_bid,
            source_ask=source_ask,
            mark_price=round(mark_price, 6),
            mark_method=method,
            marked_value_usd=round(marked_value, 6),
            remaining_entry_cost_usd=round(position.remaining_entry_cost_usd, 6),
            unrealized_pnl_usd=round(unrealized, 6),
            age_sec=round((mark_ts - _parse_ts(position.opened_ts)).total_seconds(), 6),
            ts=mark_ts,
            metadata={"close_reason": position.close_reason},
        )

    def set_position_state(self, position_id: str, state: PositionState, reason_code: str | None = None, ts: datetime | None = None) -> bool:
        position = self.position_records.get(position_id)
        if position is None:
            return False
        position.state = state
        position.close_reason = reason_code or position.close_reason
        if state in {PositionState.CLOSED, PositionState.FORCE_CLOSED, PositionState.EXPIRED}:
            position.closed_ts = (ts or datetime.now(timezone.utc)).isoformat()
        return True

    def build_trade_summary(self, position_id: str) -> TradeSummary | None:
        position = self.position_records.get(position_id)
        if position is None or position.is_open or position.closed_ts is None:
            return None
        opened_ts = _parse_ts(position.opened_ts)
        closed_ts = _parse_ts(position.closed_ts)
        return TradeSummary(
            position_id=position.position_id,
            candidate_id=position.candidate_id,
            symbol=position.symbol,
            market_slug=position.market_slug,
            state=position.state,
            entry_cost_usd=round(position.entry_cost_usd, 6),
            exit_proceeds_usd=round(position.exit_proceeds_usd, 6),
            fees_paid_usd=round(position.total_fees_usd, 6),
            realized_pnl_usd=round(position.realized_pnl_usd, 6),
            holding_duration_sec=round((closed_ts - opened_ts).total_seconds(), 6),
            opened_ts=opened_ts,
            closed_ts=closed_ts,
            metadata={"close_reason": position.close_reason},
        )

    def get_open_positions(self) -> list[PaperPositionRecord]:
        return [position for position in self.position_records.values() if position.is_open]

    def snapshot(self, ts: datetime | None = None) -> AccountSnapshot:
        snapshot_ts = ts or datetime.now(timezone.utc)
        active_positions = self.get_open_positions()
        exposure_by_market: Dict[str, float] = {}
        unrealized_pnl = 0.0
        for position in active_positions:
            exposure_by_market[position.market_slug] = exposure_by_market.get(position.market_slug, 0.0) + position.remaining_entry_cost_usd
            unrealized_pnl += position.last_unrealized_pnl_usd

        return AccountSnapshot(
            cash=round(self.cash, 6),
            frozen_cash=round(self.frozen_cash, 6),
            realized_pnl=round(self.realized_pnl, 6),
            unrealized_pnl=round(unrealized_pnl, 6),
            daily_pnl=round(self.realized_pnl + unrealized_pnl, 6),
            consecutive_losses=0,
            positions=[
                PositionSnapshot(
                    position_id=position.position_id,
                    candidate_id=position.candidate_id,
                    symbol=position.symbol,
                    market_slug=position.market_slug,
                    shares=round(position.remaining_shares, 6),
                    avg_price=round(position.avg_entry_price, 6),
                    state=position.state,
                    opened_ts=_parse_ts(position.opened_ts),
                    closed_ts=_parse_ts(position.closed_ts) if position.closed_ts else None,
                    entry_cost_usd=round(position.entry_cost_usd, 6),
                    exit_proceeds_usd=round(position.exit_proceeds_usd, 6),
                    fees_paid_usd=round(position.total_fees_usd, 6),
                    realized_pnl=round(position.realized_pnl_usd, 6),
                    unrealized_pnl=round(position.last_unrealized_pnl_usd, 6),
                    marked_value_usd=round(position.last_marked_value_usd, 6),
                )
                for position in active_positions
            ],
            exposure_by_market={key: round(value, 6) for key, value in exposure_by_market.items()},
            open_positions=len(active_positions),
            ts=snapshot_ts,
        )

    def _apply_buy_fill(self, position_id: str, order: PaperOrder, shares: float, price: float, fee_usd: float, fill_ts: str) -> None:
        position = self.position_records.get(position_id)
        if position is None:
            position = PaperPositionRecord(
                position_id=position_id,
                candidate_id=order.candidate_id,
                symbol=order.symbol,
                market_slug=order.market_slug,
                opened_ts=fill_ts,
            )
            self.position_records[position_id] = position

        position.total_entry_shares += shares
        position.entry_notional_usd += price * shares
        position.total_entry_fees_usd += fee_usd
        position.remaining_entry_fee_usd += fee_usd
        position.state = PositionState.OPENED
        self._rebuild_symbol_position(order.symbol)

    def _apply_sell_fill(self, position_id: str, order: PaperOrder, shares: float, price: float, fee_usd: float, fill_ts: str) -> None:
        position = self._resolve_position_record(position_id=position_id, symbol=order.symbol)
        if position is None or position.remaining_shares + 1e-9 < shares:
            raise ValueError(f"Cannot sell {shares} shares of {order.symbol}; position too small")

        remaining_before = position.remaining_shares
        allocated_entry_fee = position.remaining_entry_fee_usd * (shares / remaining_before) if remaining_before > 1e-9 else 0.0
        realized_increment = ((price - position.avg_entry_price) * shares) - fee_usd - allocated_entry_fee

        position.total_exit_shares += shares
        position.exit_notional_usd += price * shares
        position.total_exit_fees_usd += fee_usd
        position.remaining_entry_fee_usd = max(0.0, position.remaining_entry_fee_usd - allocated_entry_fee)
        position.realized_pnl_usd += realized_increment
        self.realized_pnl += realized_increment

        if position.remaining_shares <= 1e-9:
            position.state = PositionState.CLOSED
            position.closed_ts = fill_ts
        else:
            position.state = PositionState.PARTIALLY_REDUCED

        self._rebuild_symbol_position(order.symbol)

    def _resolve_position_record(self, position_id: str | None, symbol: str) -> PaperPositionRecord | None:
        if position_id:
            position = self.position_records.get(position_id)
            if position is not None and position.is_open:
                return position
        for position in self.position_records.values():
            if position.symbol == symbol and position.is_open:
                return position
        return None

    def _rebuild_symbol_position(self, symbol: str) -> None:
        open_positions = [
            position
            for position in self.position_records.values()
            if position.symbol == symbol and position.is_open
        ]
        if not open_positions:
            self.positions.pop(symbol, None)
            return

        total_shares = sum(position.remaining_shares for position in open_positions)
        total_cost = sum(position.remaining_shares * position.avg_entry_price for position in open_positions)
        market_slug = open_positions[0].market_slug
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            market_slug=market_slug,
            shares=total_shares,
            avg_price=(total_cost / total_shares) if total_shares > 1e-9 else 0.0,
        )
