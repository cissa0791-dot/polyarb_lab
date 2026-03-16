from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.models import RejectionReason


TOUCH_WINDOW_CENTS = 0.01
INTEGRITY_FAILURE = "integrity_failure"
FEASIBILITY_FAILURE = "feasibility_failure"

_INTEGRITY_REASON_CODES = {
    RejectionReason.INVALID_ORDERBOOK.value,
    RejectionReason.CROSSED_BOOK.value,
    RejectionReason.NON_MONOTONIC_BOOK.value,
    RejectionReason.MALFORMED_PRICE_LEVEL.value,
    RejectionReason.MISSING_ORDERBOOK.value,
    RejectionReason.ORDERBOOK_FETCH_FAILED.value,
}
_FEASIBILITY_REASON_CODES = {
    RejectionReason.EMPTY_ASKS.value,
    RejectionReason.EMPTY_BIDS.value,
    RejectionReason.NO_TOUCH_DEPTH.value,
}


@dataclass(frozen=True)
class OrderBookValidationResult:
    passed: bool
    reason_code: str | None
    problem_stage: str
    validation_rule: str | None
    token_id: str
    required_action: str
    raw_bids_count: int
    raw_asks_count: int
    normalized_bids_count: int
    normalized_asks_count: int
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    total_depth_near_touch: float
    malformed_bid_levels: int = 0
    malformed_ask_levels: int = 0
    non_positive_bid_levels: int = 0
    non_positive_ask_levels: int = 0
    raw_bids_monotonic: bool | None = None
    raw_asks_monotonic: bool | None = None
    details: dict[str, Any] | None = None

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "required_action": self.required_action,
            "failure_class": orderbook_failure_class(self.reason_code),
            "raw_bids_count": self.raw_bids_count,
            "raw_asks_count": self.raw_asks_count,
            "normalized_bids_count": self.normalized_bids_count,
            "normalized_asks_count": self.normalized_asks_count,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "total_depth_near_touch": self.total_depth_near_touch,
            "malformed_bid_levels": self.malformed_bid_levels,
            "malformed_ask_levels": self.malformed_ask_levels,
            "non_positive_bid_levels": self.non_positive_bid_levels,
            "non_positive_ask_levels": self.non_positive_ask_levels,
            "raw_bids_monotonic": self.raw_bids_monotonic,
            "raw_asks_monotonic": self.raw_asks_monotonic,
            "validation_rule": self.validation_rule,
            "problem_stage": self.problem_stage,
            "details": self.details or {},
        }


def orderbook_failure_class(reason_code: str | None) -> str | None:
    if reason_code in _INTEGRITY_REASON_CODES:
        return INTEGRITY_FAILURE
    if reason_code in _FEASIBILITY_REASON_CODES:
        return FEASIBILITY_FAILURE
    return None


def build_fetch_failure_validation(token_id: str, error: Exception) -> OrderBookValidationResult:
    message = str(error)
    reason = RejectionReason.MISSING_ORDERBOOK.value if "No orderbook exists" in message else RejectionReason.ORDERBOOK_FETCH_FAILED.value
    return OrderBookValidationResult(
        passed=False,
        reason_code=reason,
        problem_stage="fetch",
        validation_rule="fetch_book_failed",
        token_id=str(token_id),
        required_action="UNKNOWN",
        raw_bids_count=0,
        raw_asks_count=0,
        normalized_bids_count=0,
        normalized_asks_count=0,
        best_bid=None,
        best_ask=None,
        spread=None,
        total_depth_near_touch=0.0,
        details={"error": message},
    )


def validate_orderbook(book: Any, required_action: str = "BUY") -> OrderBookValidationResult:
    metadata = getattr(book, "metadata", {}) or {}
    bids = list(getattr(book, "bids", []) or [])
    asks = list(getattr(book, "asks", []) or [])
    normalized_bids_count = len(bids)
    normalized_asks_count = len(asks)
    best_bid = float(bids[0].price) if bids else None
    best_ask = float(asks[0].price) if asks else None
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    action = str(required_action).upper()
    relevant_levels = asks if action == "BUY" else bids
    total_depth_near_touch = round(_depth_near_touch(relevant_levels, action), 6)

    raw_bids_count = int(metadata.get("raw_bids_count", normalized_bids_count))
    raw_asks_count = int(metadata.get("raw_asks_count", normalized_asks_count))
    malformed_bid_levels = int(metadata.get("malformed_bid_levels", 0))
    malformed_ask_levels = int(metadata.get("malformed_ask_levels", 0))
    non_positive_bid_levels = int(metadata.get("non_positive_bid_levels", 0))
    non_positive_ask_levels = int(metadata.get("non_positive_ask_levels", 0))
    raw_bids_monotonic = metadata.get("raw_bids_monotonic")
    raw_asks_monotonic = metadata.get("raw_asks_monotonic")

    if action == "BUY":
        side_raw_count = raw_asks_count
        side_normalized_count = normalized_asks_count
        side_malformed = malformed_ask_levels
        side_non_positive = non_positive_ask_levels
        empty_reason = RejectionReason.EMPTY_ASKS.value
        empty_rule = "required_ask_side_empty"
    else:
        side_raw_count = raw_bids_count
        side_normalized_count = normalized_bids_count
        side_malformed = malformed_bid_levels
        side_non_positive = non_positive_bid_levels
        empty_reason = RejectionReason.EMPTY_BIDS.value
        empty_rule = "required_bid_side_empty"

    if side_normalized_count == 0:
        if side_malformed > 0:
            return _failed_result(
                book=book,
                required_action=action,
                reason_code=RejectionReason.MALFORMED_PRICE_LEVEL.value,
                problem_stage="parse",
                validation_rule="all_required_side_levels_malformed",
                total_depth_near_touch=total_depth_near_touch,
            )
        if side_raw_count > 0 and side_non_positive == side_raw_count:
            return _failed_result(
                book=book,
                required_action=action,
                reason_code=RejectionReason.NO_TOUCH_DEPTH.value,
                problem_stage="validate",
                validation_rule="all_required_side_levels_non_positive",
                total_depth_near_touch=total_depth_near_touch,
            )
        return _failed_result(
            book=book,
            required_action=action,
            reason_code=empty_reason,
            problem_stage="validate",
            validation_rule=empty_rule,
            total_depth_near_touch=total_depth_near_touch,
        )

    if not _is_monotonic(bids, descending=True) or not _is_monotonic(asks, descending=False):
        return _failed_result(
            book=book,
            required_action=action,
            reason_code=RejectionReason.NON_MONOTONIC_BOOK.value,
            problem_stage="normalize",
            validation_rule="normalized_levels_not_monotonic",
            total_depth_near_touch=total_depth_near_touch,
        )

    if best_bid is not None and best_ask is not None and best_bid > best_ask + 1e-9:
        return _failed_result(
            book=book,
            required_action=action,
            reason_code=RejectionReason.CROSSED_BOOK.value,
            problem_stage="validate",
            validation_rule="best_bid_exceeds_best_ask",
            total_depth_near_touch=total_depth_near_touch,
        )

    if total_depth_near_touch <= 1e-9:
        return _failed_result(
            book=book,
            required_action=action,
            reason_code=RejectionReason.NO_TOUCH_DEPTH.value,
            problem_stage="validate",
            validation_rule="zero_depth_near_touch",
            total_depth_near_touch=total_depth_near_touch,
        )

    return _result(
        book=book,
        required_action=action,
        reason_code=None,
        problem_stage="validate",
        validation_rule=None,
        total_depth_near_touch=total_depth_near_touch,
        passed=True,
    )


def _depth_near_touch(levels: list[Any], action: str) -> float:
    if not levels:
        return 0.0
    touch_price = float(levels[0].price)
    depth = 0.0
    for level in levels:
        price = float(level.price)
        size = float(level.size)
        if action == "BUY":
            if price > touch_price + TOUCH_WINDOW_CENTS + 1e-12:
                break
        else:
            if price < touch_price - TOUCH_WINDOW_CENTS - 1e-12:
                break
        depth += price * size
    return depth


def _is_monotonic(levels: list[Any], descending: bool) -> bool:
    if len(levels) < 2:
        return True
    previous = float(levels[0].price)
    for level in levels[1:]:
        price = float(level.price)
        if descending and price > previous + 1e-12:
            return False
        if not descending and price < previous - 1e-12:
            return False
        previous = price
    return True


def _failed_result(
    book: Any,
    required_action: str,
    reason_code: str,
    problem_stage: str,
    validation_rule: str,
    total_depth_near_touch: float,
) -> OrderBookValidationResult:
    return _result(
        book=book,
        required_action=required_action,
        reason_code=reason_code,
        problem_stage=problem_stage,
        validation_rule=validation_rule,
        total_depth_near_touch=total_depth_near_touch,
        passed=False,
    )


def _result(
    book: Any,
    required_action: str,
    reason_code: str | None,
    problem_stage: str,
    validation_rule: str | None,
    total_depth_near_touch: float,
    passed: bool,
) -> OrderBookValidationResult:
    metadata = getattr(book, "metadata", {}) or {}
    bids = list(getattr(book, "bids", []) or [])
    asks = list(getattr(book, "asks", []) or [])
    best_bid = float(bids[0].price) if bids else None
    best_ask = float(asks[0].price) if asks else None
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    return OrderBookValidationResult(
        passed=passed,
        reason_code=reason_code,
        problem_stage=problem_stage,
        validation_rule=validation_rule,
        token_id=str(getattr(book, "token_id", metadata.get("requested_token_id", ""))),
        required_action=required_action,
        raw_bids_count=int(metadata.get("raw_bids_count", len(bids))),
        raw_asks_count=int(metadata.get("raw_asks_count", len(asks))),
        normalized_bids_count=len(bids),
        normalized_asks_count=len(asks),
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        total_depth_near_touch=total_depth_near_touch,
        malformed_bid_levels=int(metadata.get("malformed_bid_levels", 0)),
        malformed_ask_levels=int(metadata.get("malformed_ask_levels", 0)),
        non_positive_bid_levels=int(metadata.get("non_positive_bid_levels", 0)),
        non_positive_ask_levels=int(metadata.get("non_positive_ask_levels", 0)),
        raw_bids_monotonic=metadata.get("raw_bids_monotonic"),
        raw_asks_monotonic=metadata.get("raw_asks_monotonic"),
        details={
            "requested_token_id": metadata.get("requested_token_id"),
            "response_asset_id": metadata.get("response_asset_id"),
        },
    )
