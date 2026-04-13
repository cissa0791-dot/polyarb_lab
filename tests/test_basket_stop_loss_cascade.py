"""Tests for basket-level STOP_LOSS cascade.

Mirrors the structure of test_basket_edge_decay.py.

Invariants:
1. One leg fires STOP_LOSS  → all remaining open basket legs are flagged for cascade exit.
2. Non-SL closed siblings (e.g. MAX_HOLDING_AGE) must NOT trigger a SL cascade.
3. Unrelated baskets are unaffected by a STOP_LOSS cascade in a different basket.
4. The triggering leg itself is excluded from the sibling scan.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.domain.models import PositionState
from src.paper.ledger import Ledger, PaperPositionRecord
from src.runtime.runner import _has_basket_sl_exit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc).isoformat()


def _make_record(
    position_id: str,
    candidate_id: str | None,
    *,
    close_reason: str | None = None,
    remaining_shares: float = 10.0,
) -> PaperPositionRecord:
    rec = PaperPositionRecord(
        position_id=position_id,
        candidate_id=candidate_id,
        symbol=f"sym-{position_id}",
        market_slug=f"market-{position_id}",
        opened_ts=_NOW,
        state=PositionState.OPENED if close_reason is None else PositionState.CLOSED,
        close_reason=close_reason,
        total_entry_shares=remaining_shares,
        total_exit_shares=0.0 if close_reason is None else remaining_shares,
    )
    return rec


def _records(*recs: PaperPositionRecord) -> dict:
    return {r.position_id: r for r in recs}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class BasketStopLossCascadeTests(unittest.TestCase):

    def test_sl_closed_sibling_triggers_cascade_for_open_leg(self) -> None:
        """One leg closed via STOP_LOSS → sibling open leg should cascade."""
        records = _records(
            _make_record("pos-A1", "basket-A", close_reason="STOP_LOSS", remaining_shares=5.0),
            _make_record("pos-A2", "basket-A"),  # still open
        )
        result = _has_basket_sl_exit(records, "basket-A", "pos-A2")
        self.assertTrue(result, "Open sibling should detect cascade trigger from SL-closed leg")

    def test_non_sl_closed_sibling_does_not_trigger_cascade(self) -> None:
        """Sibling closed via MAX_HOLDING_AGE must NOT trigger a SL cascade."""
        records = _records(
            _make_record("pos-B1", "basket-B", close_reason="MAX_HOLDING_AGE", remaining_shares=5.0),
            _make_record("pos-B2", "basket-B"),  # still open
        )
        result = _has_basket_sl_exit(records, "basket-B", "pos-B2")
        self.assertFalse(result, "MHA-closed sibling must not trigger SL cascade")

    def test_edge_decay_closed_sibling_does_not_trigger_sl_cascade(self) -> None:
        """Sibling closed via EDGE_DECAY must NOT trigger a SL cascade."""
        records = _records(
            _make_record("pos-C1", "basket-C", close_reason="EDGE_DECAY", remaining_shares=5.0),
            _make_record("pos-C2", "basket-C"),
        )
        result = _has_basket_sl_exit(records, "basket-C", "pos-C2")
        self.assertFalse(result, "ED-closed sibling must not trigger SL cascade")

    def test_no_closed_sibling_does_not_trigger_cascade(self) -> None:
        """All legs still open → no cascade."""
        records = _records(
            _make_record("pos-D1", "basket-D"),
            _make_record("pos-D2", "basket-D"),
        )
        result = _has_basket_sl_exit(records, "basket-D", "pos-D2")
        self.assertFalse(result)

    def test_self_closed_sl_does_not_trigger_own_cascade(self) -> None:
        """The triggering leg itself is excluded from the sibling scan."""
        records = _records(
            _make_record("pos-E1", "basket-E", close_reason="STOP_LOSS", remaining_shares=5.0),
        )
        result = _has_basket_sl_exit(records, "basket-E", "pos-E1")
        self.assertFalse(result, "Self should be excluded from sibling scan")

    def test_unrelated_basket_unaffected(self) -> None:
        """STOP_LOSS in basket-F must not cascade into basket-G."""
        records = _records(
            _make_record("pos-F1", "basket-F", close_reason="STOP_LOSS", remaining_shares=5.0),
            _make_record("pos-G1", "basket-G"),
            _make_record("pos-G2", "basket-G"),
        )
        self.assertFalse(_has_basket_sl_exit(records, "basket-G", "pos-G1"))
        self.assertFalse(_has_basket_sl_exit(records, "basket-G", "pos-G2"))

    def test_none_candidate_id_never_cascades(self) -> None:
        """Positions without a candidate_id must not cascade."""
        records = _records(
            _make_record("pos-H1", None, close_reason="STOP_LOSS", remaining_shares=5.0),
            _make_record("pos-H2", None),
        )
        result = _has_basket_sl_exit(records, None, "pos-H2")
        self.assertFalse(result, "None candidate_id must return False immediately")

    def test_ledger_close_reason_readable_for_sl_cascade(self) -> None:
        """set_position_state persists close_reason so _has_basket_sl_exit can read it."""
        ledger = Ledger(cash=1000.0)

        ledger.place_limit_order("o1", "tok-1", "mkt-1", "BUY", 10.0, 0.40, candidate_id="bsk-1", position_id="p1")
        ledger.apply_fill("o1", 10.0, 0.40)
        ledger.place_limit_order("o2", "tok-2", "mkt-2", "BUY", 10.0, 0.40, candidate_id="bsk-1", position_id="p2")
        ledger.apply_fill("o2", 10.0, 0.40)

        # Simulate p1 exit-filled and closed via STOP_LOSS
        ledger.place_limit_order("o1x", "tok-1", "mkt-1", "SELL", 10.0, 0.35, position_id="p1")
        ledger.apply_fill("o1x", 10.0, 0.35)
        ledger.set_position_state("p1", PositionState.CLOSED, reason_code="STOP_LOSS")

        self.assertTrue(ledger.position_records["p2"].is_open)
        result = _has_basket_sl_exit(ledger.position_records, "bsk-1", "p2")
        self.assertTrue(result, "Ledger close_reason='STOP_LOSS' on p1 must trigger cascade for p2")
