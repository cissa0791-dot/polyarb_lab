"""Deterministic tests for basket-level EDGE_DECAY cascade.

Covers three invariants:
1. One leg fires EDGE_DECAY  → all remaining open basket legs are flagged for cascade exit.
2. Already-closed legs (non-ED reason) are safely skipped — they do not trigger a cascade.
3. Unrelated baskets are unaffected by a cascade in a different basket.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.domain.models import PositionState
from src.paper.ledger import Ledger, PaperPositionRecord
from src.config_runtime.models import PaperConfig
from src.runtime.runner import (
    _basket_exit_confirmed,
    _basket_idle_release_eligible,
    _build_basket_audit,
    _has_basket_ed_exit,
    _has_basket_idle_release,
)


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

class BasketEdgeDecayCascadeTests(unittest.TestCase):

    # ── Scenario 1 ──────────────────────────────────────────────────────────
    def test_ed_closed_sibling_triggers_cascade_for_open_leg(self) -> None:
        """One leg closed via EDGE_DECAY → sibling open leg should cascade."""
        records = _records(
            _make_record("pos-A1", "basket-A", close_reason="EDGE_DECAY", remaining_shares=5.0),
            _make_record("pos-A2", "basket-A"),  # still open
        )
        # pos-A2 is open; pos-A1 closed with EDGE_DECAY
        result = _has_basket_ed_exit(records, "basket-A", "pos-A2")
        self.assertTrue(result, "Open sibling should detect cascade trigger from ED-closed leg")

    # ── Scenario 2 ──────────────────────────────────────────────────────────
    def test_non_ed_closed_sibling_does_not_trigger_cascade(self) -> None:
        """Sibling closed via MAX_HOLDING_AGE must NOT trigger an ED cascade."""
        records = _records(
            _make_record("pos-B1", "basket-B", close_reason="MAX_HOLDING_AGE", remaining_shares=5.0),
            _make_record("pos-B2", "basket-B"),  # still open
        )
        result = _has_basket_ed_exit(records, "basket-B", "pos-B2")
        self.assertFalse(result, "MHA-closed sibling must not trigger ED cascade")

    def test_no_closed_sibling_does_not_trigger_cascade(self) -> None:
        """All legs still open → no cascade."""
        records = _records(
            _make_record("pos-C1", "basket-C"),
            _make_record("pos-C2", "basket-C"),
        )
        result = _has_basket_ed_exit(records, "basket-C", "pos-C2")
        self.assertFalse(result)

    def test_self_closed_ed_does_not_trigger_own_cascade(self) -> None:
        """The triggering leg itself is excluded from the sibling scan."""
        records = _records(
            _make_record("pos-D1", "basket-D", close_reason="EDGE_DECAY", remaining_shares=5.0),
        )
        # Querying for pos-D1 itself — should return False (no *other* sibling)
        result = _has_basket_ed_exit(records, "basket-D", "pos-D1")
        self.assertFalse(result, "Self should be excluded from sibling scan")

    # ── Scenario 3 ──────────────────────────────────────────────────────────
    def test_unrelated_basket_unaffected(self) -> None:
        """EDGE_DECAY in basket-E must not cascade into basket-F."""
        records = _records(
            _make_record("pos-E1", "basket-E", close_reason="EDGE_DECAY", remaining_shares=5.0),
            _make_record("pos-F1", "basket-F"),  # different basket, still open
            _make_record("pos-F2", "basket-F"),
        )
        result_f1 = _has_basket_ed_exit(records, "basket-F", "pos-F1")
        result_f2 = _has_basket_ed_exit(records, "basket-F", "pos-F2")
        self.assertFalse(result_f1, "basket-F pos-F1 must not be affected by basket-E ED exit")
        self.assertFalse(result_f2, "basket-F pos-F2 must not be affected by basket-E ED exit")

    def test_mixed_basket_ed_and_mha_cascades_correctly(self) -> None:
        """Basket with both ED and MHA closed legs — cascade fires because ED leg exists."""
        records = _records(
            _make_record("pos-G1", "basket-G", close_reason="MAX_HOLDING_AGE", remaining_shares=5.0),
            _make_record("pos-G2", "basket-G", close_reason="EDGE_DECAY", remaining_shares=5.0),
            _make_record("pos-G3", "basket-G"),  # open
        )
        result = _has_basket_ed_exit(records, "basket-G", "pos-G3")
        self.assertTrue(result, "Basket with at least one ED-closed leg must cascade")

    def test_none_candidate_id_never_cascades(self) -> None:
        """Positions without a candidate_id (no basket membership) must not cascade."""
        records = _records(
            _make_record("pos-H1", None, close_reason="EDGE_DECAY", remaining_shares=5.0),
            _make_record("pos-H2", None),
        )
        result = _has_basket_ed_exit(records, None, "pos-H2")
        self.assertFalse(result, "None candidate_id must return False immediately")

    # ── Integration: Ledger round-trip ──────────────────────────────────────
    def test_ledger_close_reason_readable_for_cascade_check(self) -> None:
        """set_position_state persists close_reason so _has_basket_ed_exit can read it."""
        ledger = Ledger(cash=1000.0)

        # Open two legs of the same basket
        ledger.place_limit_order("o1", "tok-1", "mkt-1", "BUY", 10.0, 0.40, candidate_id="bsk-1", position_id="p1")
        ledger.apply_fill("o1", 10.0, 0.40)
        ledger.place_limit_order("o2", "tok-2", "mkt-2", "BUY", 10.0, 0.40, candidate_id="bsk-1", position_id="p2")
        ledger.apply_fill("o2", 10.0, 0.40)

        # Simulate leg p1 being exit-filled and closed via EDGE_DECAY
        ledger.place_limit_order("o1x", "tok-1", "mkt-1", "SELL", 10.0, 0.45, position_id="p1")
        ledger.apply_fill("o1x", 10.0, 0.45)
        ledger.set_position_state("p1", PositionState.CLOSED, reason_code="EDGE_DECAY")

        # p2 is still open — cascade check should fire
        self.assertTrue(ledger.position_records["p2"].is_open)
        result = _has_basket_ed_exit(ledger.position_records, "bsk-1", "p2")
        self.assertTrue(result, "Ledger close_reason='EDGE_DECAY' on p1 must trigger cascade for p2")

        # After closing p2 as well, no further cascade needed (no open siblings)
        ledger.place_limit_order("o2x", "tok-2", "mkt-2", "SELL", 10.0, 0.45, position_id="p2")
        ledger.apply_fill("o2x", 10.0, 0.45)
        ledger.set_position_state("p2", PositionState.CLOSED, reason_code="EDGE_DECAY")

        # Both closed — querying for a hypothetical p3 of same basket still returns True
        # (there IS an ED sibling), but is_open guard in runner prevents re-processing
        self.assertFalse(ledger.position_records["p1"].is_open)
        self.assertFalse(ledger.position_records["p2"].is_open)


# ---------------------------------------------------------------------------
# Basket audit tests
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_OPENED_TS = _BASE_TS.isoformat()


def _make_closed_record(
    position_id: str,
    candidate_id: str,
    *,
    close_reason: str,
    realized_pnl: float,
    peak_unrealized: float = 0.0,
    first_adverse_ts: str | None = None,
    closed_ts: str | None = None,
) -> PaperPositionRecord:
    shares = 10.0
    rec = PaperPositionRecord(
        position_id=position_id,
        candidate_id=candidate_id,
        symbol=f"sym-{position_id}",
        market_slug=f"mkt-{position_id}",
        opened_ts=_OPENED_TS,
        state=PositionState.CLOSED,
        close_reason=close_reason,
        closed_ts=closed_ts or _BASE_TS.isoformat(),
        total_entry_shares=shares,
        total_exit_shares=shares,
    )
    # Manually set audit fields since we bypass mark_position().
    rec.realized_pnl_usd = realized_pnl
    rec.peak_unrealized_pnl_usd = peak_unrealized
    rec.first_adverse_ts = first_adverse_ts
    return rec


def _make_open_record(
    position_id: str,
    candidate_id: str,
    *,
    last_unrealized: float,
    peak_unrealized: float = 0.0,
    first_adverse_ts: str | None = None,
) -> PaperPositionRecord:
    shares = 10.0
    rec = PaperPositionRecord(
        position_id=position_id,
        candidate_id=candidate_id,
        symbol=f"sym-{position_id}",
        market_slug=f"mkt-{position_id}",
        opened_ts=_OPENED_TS,
        state=PositionState.OPENED,
        total_entry_shares=shares,
        total_exit_shares=0.0,
    )
    rec.last_unrealized_pnl_usd = last_unrealized
    rec.peak_unrealized_pnl_usd = peak_unrealized
    rec.first_adverse_ts = first_adverse_ts
    return rec


class BasketAuditTests(unittest.TestCase):

    # ── MHA_only basket ──────────────────────────────────────────────────────
    def test_mha_only_classification(self) -> None:
        """Basket with no ED exits → MHA_only."""
        r1 = _make_closed_record("p1", "bsk", close_reason="MAX_HOLDING_AGE", realized_pnl=-0.05)
        r2 = _make_closed_record("p2", "bsk", close_reason="MAX_HOLDING_AGE", realized_pnl=-0.03)
        records = {r1.position_id: r1, r2.position_id: r2}
        result = _build_basket_audit(records, "bsk", _BASE_TS)
        audit = result["basket_audit"]
        self.assertEqual(audit["exit_path_classification"], "MHA_only")
        self.assertIsNone(audit["trigger_leg_id"])

    # ── dominant_leg_candidate ──────────────────────────────────────────────
    def test_dominant_leg_candidate_when_trigger_is_dominant_loser(self) -> None:
        """ED trigger leg is the dominant loser → dominant_leg_candidate."""
        trigger_ts = (_BASE_TS + timedelta(seconds=60)).isoformat()
        r1 = _make_closed_record(
            "p1", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.20,
            closed_ts=trigger_ts,
        )
        r2 = _make_closed_record(
            "p2", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.02,
            closed_ts=(_BASE_TS + timedelta(seconds=120)).isoformat(),
        )
        records = {r1.position_id: r1, r2.position_id: r2}
        result = _build_basket_audit(records, "bsk", _BASE_TS + timedelta(seconds=130))
        audit = result["basket_audit"]
        self.assertEqual(audit["exit_path_classification"], "dominant_leg_candidate")
        self.assertEqual(audit["trigger_leg_id"], "p1")

    # ── minor_leg_candidate ─────────────────────────────────────────────────
    def test_minor_leg_candidate_when_trigger_is_not_dominant(self) -> None:
        """Trigger leg has small loss; another leg is the dominant loser → minor_leg_candidate."""
        trigger_ts = (_BASE_TS + timedelta(seconds=30)).isoformat()
        r1 = _make_closed_record(
            "p1", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.01,
            closed_ts=trigger_ts,
        )
        r2 = _make_closed_record(
            "p2", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.50,
            closed_ts=(_BASE_TS + timedelta(seconds=60)).isoformat(),
        )
        records = {r1.position_id: r1, r2.position_id: r2}
        result = _build_basket_audit(records, "bsk", _BASE_TS + timedelta(seconds=70))
        audit = result["basket_audit"]
        self.assertEqual(audit["exit_path_classification"], "minor_leg_candidate")
        self.assertEqual(audit["trigger_leg_id"], "p1")

    # ── aggregate_basket_deterioration ──────────────────────────────────────
    def test_aggregate_basket_deterioration_when_loss_distributed(self) -> None:
        """Loss spread across multiple legs equally → aggregate_basket_deterioration."""
        trigger_ts = (_BASE_TS + timedelta(seconds=30)).isoformat()
        r1 = _make_closed_record(
            "p1", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.10,
            closed_ts=trigger_ts,
        )
        r2 = _make_closed_record(
            "p2", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.12,
            closed_ts=(_BASE_TS + timedelta(seconds=60)).isoformat(),
        )
        r3 = _make_closed_record(
            "p3", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.11,
            closed_ts=(_BASE_TS + timedelta(seconds=90)).isoformat(),
        )
        records = {r.position_id: r for r in [r1, r2, r3]}
        result = _build_basket_audit(records, "bsk", _BASE_TS + timedelta(seconds=100))
        audit = result["basket_audit"]
        self.assertEqual(audit["exit_path_classification"], "aggregate_basket_deterioration")

    # ── basket_unrealized_pnl sums correctly ─────────────────────────────────
    def test_basket_unrealized_pnl_sums_realized_and_open(self) -> None:
        """Closed legs use realized_pnl; open legs use last_unrealized_pnl."""
        r_closed = _make_closed_record(
            "p1", "bsk2", close_reason="EDGE_DECAY", realized_pnl=-0.10,
            closed_ts=_BASE_TS.isoformat(),
        )
        r_open = _make_open_record("p2", "bsk2", last_unrealized=-0.05)
        records = {r_closed.position_id: r_closed, r_open.position_id: r_open}
        result = _build_basket_audit(records, "bsk2", _BASE_TS)
        audit = result["basket_audit"]
        self.assertAlmostEqual(audit["basket_unrealized_pnl"], -0.15, places=5)

    # ── peak tracking and drawdown ───────────────────────────────────────────
    def test_peak_and_drawdown_fields(self) -> None:
        """basket_peak_to_current_drawdown = peak - current."""
        r1 = _make_closed_record(
            "p1", "bsk3", close_reason="EDGE_DECAY", realized_pnl=-0.05,
            peak_unrealized=0.08,
            closed_ts=_BASE_TS.isoformat(),
        )
        r2 = _make_closed_record(
            "p2", "bsk3", close_reason="EDGE_DECAY", realized_pnl=-0.03,
            peak_unrealized=0.06,
            closed_ts=_BASE_TS.isoformat(),
        )
        records = {r1.position_id: r1, r2.position_id: r2}
        result = _build_basket_audit(records, "bsk3", _BASE_TS)
        audit = result["basket_audit"]
        self.assertAlmostEqual(audit["basket_peak_unrealized_pnl"], 0.14, places=5)
        expected_drawdown = round(0.14 - (-0.08), 6)
        self.assertAlmostEqual(audit["basket_peak_to_current_drawdown"], expected_drawdown, places=5)

    # ── time_since_first_adverse_state ───────────────────────────────────────
    def test_time_since_first_adverse_state_computed(self) -> None:
        """Earliest first_adverse_ts across legs is used to compute elapsed time."""
        adverse_ts_early = (_BASE_TS + timedelta(seconds=30)).isoformat()
        adverse_ts_late = (_BASE_TS + timedelta(seconds=90)).isoformat()
        r1 = _make_closed_record(
            "p1", "bsk4", close_reason="EDGE_DECAY", realized_pnl=-0.05,
            first_adverse_ts=adverse_ts_early,
            closed_ts=_BASE_TS.isoformat(),
        )
        r2 = _make_closed_record(
            "p2", "bsk4", close_reason="EDGE_DECAY", realized_pnl=-0.03,
            first_adverse_ts=adverse_ts_late,
            closed_ts=_BASE_TS.isoformat(),
        )
        records = {r1.position_id: r1, r2.position_id: r2}
        now = _BASE_TS + timedelta(seconds=150)
        result = _build_basket_audit(records, "bsk4", now)
        audit = result["basket_audit"]
        self.assertAlmostEqual(audit["time_since_first_adverse_state"], 120.0, places=1)

    def test_no_adverse_state_returns_minus_one(self) -> None:
        """If no leg ever went adverse, time_since_first_adverse_state == -1.0."""
        r1 = _make_closed_record(
            "p1", "bsk5", close_reason="MAX_HOLDING_AGE", realized_pnl=0.02,
            closed_ts=_BASE_TS.isoformat(),
        )
        records = {r1.position_id: r1}
        result = _build_basket_audit(records, "bsk5", _BASE_TS)
        audit = result["basket_audit"]
        self.assertEqual(audit["time_since_first_adverse_state"], -1.0)

    # ── none candidate_id ────────────────────────────────────────────────────
    def test_none_candidate_id_returns_null_audit(self) -> None:
        """No candidate_id → basket_audit: None."""
        records: dict = {}
        result = _build_basket_audit(records, None, _BASE_TS)
        self.assertIsNone(result["basket_audit"])

    # ── peak_unrealized maintained by ledger mark_position ──────────────────
    def test_ledger_mark_position_tracks_peak_and_adverse(self) -> None:
        """mark_position updates peak_unrealized_pnl_usd and first_adverse_ts correctly."""
        ledger = Ledger(cash=1000.0)
        ledger.place_limit_order("o1", "tok-1", "mkt-1", "BUY", 10.0, 0.40, candidate_id="bsk-z", position_id="pz1")
        ledger.apply_fill("o1", 10.0, 0.40)

        pos = ledger.position_records["pz1"]
        now = datetime.now(timezone.utc)

        # First mark: bid > avg_entry → positive unrealized.
        ledger.mark_position("pz1", mark_price=0.45, ts=now)
        self.assertGreater(pos.peak_unrealized_pnl_usd, 0.0)
        self.assertIsNone(pos.first_adverse_ts)

        # Second mark: bid < avg_entry → negative unrealized; adverse recorded.
        ledger.mark_position("pz1", mark_price=0.35, ts=now + timedelta(seconds=10))
        self.assertIsNotNone(pos.first_adverse_ts)

        # Peak should not decrease.
        peak_after_positive = pos.peak_unrealized_pnl_usd
        ledger.mark_position("pz1", mark_price=0.30, ts=now + timedelta(seconds=20))
        self.assertAlmostEqual(pos.peak_unrealized_pnl_usd, peak_after_positive, places=6)

        # Better mark → peak updates.
        ledger.mark_position("pz1", mark_price=0.60, ts=now + timedelta(seconds=30))
        self.assertGreater(pos.peak_unrealized_pnl_usd, peak_after_positive)


# ---------------------------------------------------------------------------
# Basket exit gate tests
# ---------------------------------------------------------------------------

def _gate_config(
    dominance: float = 0.40,
    drawdown: float = 0.05,
    floor: float = -0.10,
) -> PaperConfig:
    return PaperConfig(
        basket_dominance_threshold=dominance,
        basket_drawdown_exit_threshold=drawdown,
        basket_unrealized_pnl_floor=floor,
    )


def _disabled_gate_config() -> PaperConfig:
    return PaperConfig(basket_dominance_threshold=0.0)


def _records_from_list(*recs: PaperPositionRecord) -> dict:
    return {r.position_id: r for r in recs}


class BasketExitGateTests(unittest.TestCase):

    # ── Gate disabled ────────────────────────────────────────────────────────

    def test_gate_disabled_always_confirms(self) -> None:
        """basket_dominance_threshold=0.0 → gate disabled → always True."""
        r1 = _make_open_record("p1", "bsk", last_unrealized=-0.02)
        r2 = _make_open_record("p2", "bsk", last_unrealized=-0.50)
        records = _records_from_list(r1, r2)
        # Even though p1 is the minor loser, gate is disabled → confirm.
        result = _basket_exit_confirmed(records, "bsk", "p1", _disabled_gate_config())
        self.assertTrue(result)

    def test_gate_disabled_with_none_candidate_id(self) -> None:
        result = _basket_exit_confirmed({}, None, "p1", _disabled_gate_config())
        self.assertTrue(result)

    # ── Path A: trigger is dominant loser ────────────────────────────────────

    def test_path_a_trigger_is_dominant_loser_confirms(self) -> None:
        """Trigger leg IS the dominant loss leg → Path A confirms."""
        r_trigger = _make_open_record("p1", "bsk", last_unrealized=-0.50)
        r_other = _make_open_record("p2", "bsk", last_unrealized=-0.05)
        records = _records_from_list(r_trigger, r_other)
        result = _basket_exit_confirmed(records, "bsk", "p1", _gate_config())
        self.assertTrue(result)

    def test_path_a_dominant_share_meets_threshold_confirms(self) -> None:
        """Trigger leg is not the dominant loser but dominant_loss_leg_share >= threshold."""
        # p2 is the dominant loser with 80% share; threshold is 0.40 → passes.
        r_trigger = _make_open_record("p1", "bsk", last_unrealized=-0.10)
        r_dominant = _make_open_record("p2", "bsk", last_unrealized=-0.40)
        records = _records_from_list(r_trigger, r_dominant)
        # dominant_loss_leg_share = 0.40 / 0.50 = 0.80 >= 0.40 → True.
        result = _basket_exit_confirmed(records, "bsk", "p1", _gate_config())
        self.assertTrue(result)

    def test_path_a_minor_trigger_below_threshold_blocked(self) -> None:
        """Distributed loss: trigger leg is clearly minor, no single leg dominates → gate blocks.

        Uses differentiated losses so trigger (p0) is not the dominant loser.
        Path B disabled (drawdown=0.0, floor=0.0) to isolate Path A behaviour.
        """
        # Trigger p0 has the smallest loss; p2 has the largest but still < 40% share.
        legs = [
            _make_open_record("p0", "bsk", last_unrealized=-0.08),  # trigger
            _make_open_record("p1", "bsk", last_unrealized=-0.11),
            _make_open_record("p2", "bsk", last_unrealized=-0.12),  # dominant @ 0.12/0.50 = 0.24
            _make_open_record("p3", "bsk", last_unrealized=-0.09),
            _make_open_record("p4", "bsk", last_unrealized=-0.10),
        ]
        records = _records_from_list(*legs)
        # dominant_loss_leg_share = 0.12/0.50 = 0.24 < 0.40 → Path A fails.
        # trigger (p0) != dominant (p2) → trigger_is_dominant = False.
        # Path B disabled: drawdown=0.0, floor=0.0.
        cfg = _gate_config(dominance=0.40, drawdown=0.0, floor=0.0)
        result = _basket_exit_confirmed(records, "bsk", "p0", cfg)
        self.assertFalse(result)

    def test_path_a_two_leg_winner_trigger_confirms(self) -> None:
        """Trigger leg has positive PnL (no negative legs); basket has no loss → Path A confirms."""
        r_winner = _make_open_record("p1", "bsk", last_unrealized=0.05)
        r_loser = _make_open_record("p2", "bsk", last_unrealized=-0.02)
        records = _records_from_list(r_winner, r_loser)
        # total_loss > 0 because p2 is negative. dominant is p2, not p1.
        # dominant_loss_leg_share = 0.02/0.02 = 1.0 >= 0.40 → Path A confirms.
        result = _basket_exit_confirmed(records, "bsk", "p1", _gate_config())
        self.assertTrue(result)

    def test_path_a_no_negative_legs_confirms(self) -> None:
        """All legs profitable → total_loss == 0 → always confirm (basket fine)."""
        r1 = _make_open_record("p1", "bsk", last_unrealized=0.05)
        r2 = _make_open_record("p2", "bsk", last_unrealized=0.03)
        records = _records_from_list(r1, r2)
        result = _basket_exit_confirmed(records, "bsk", "p1", _gate_config())
        self.assertTrue(result)

    # ── Path B: drawdown override ─────────────────────────────────────────────

    def test_path_b_drawdown_override_confirms_minor_trigger(self) -> None:
        """Minor trigger blocked by Path A, but basket drawdown >= threshold → Path B confirms."""
        legs = [_make_open_record(f"p{i}", "bsk", last_unrealized=-0.10, peak_unrealized=0.05)
                for i in range(5)]
        records = _records_from_list(*legs)
        # basket_pnl = -0.50; basket_peak = 0.25; drawdown = 0.75 >= 0.05 → Path B confirms.
        result = _basket_exit_confirmed(records, "bsk", "p0", _gate_config())
        self.assertTrue(result)

    def test_path_b_drawdown_disabled_when_zero(self) -> None:
        """basket_drawdown_exit_threshold=0.0 → Path B drawdown check not used.

        Even with a large drawdown, gate blocks when threshold is zero because
        only Path A applies — and no single leg dominates.
        """
        legs = [
            _make_open_record("p0", "bsk", last_unrealized=-0.08, peak_unrealized=0.05),
            _make_open_record("p1", "bsk", last_unrealized=-0.11, peak_unrealized=0.05),
            _make_open_record("p2", "bsk", last_unrealized=-0.12, peak_unrealized=0.05),
            _make_open_record("p3", "bsk", last_unrealized=-0.09, peak_unrealized=0.05),
            _make_open_record("p4", "bsk", last_unrealized=-0.10, peak_unrealized=0.05),
        ]
        records = _records_from_list(*legs)
        # basket_drawdown = 0.25 + 0.50 = 0.75 but threshold=0.0 → disabled.
        # floor=0.0 → disabled.
        # Path A: dominant_loss_leg_share = 0.12/0.50 = 0.24 < 0.40 AND trigger (p0) != dominant (p2) → False.
        cfg = PaperConfig(
            basket_dominance_threshold=0.40,
            basket_drawdown_exit_threshold=0.0,
            basket_unrealized_pnl_floor=0.0,
        )
        result = _basket_exit_confirmed(records, "bsk", "p0", cfg)
        self.assertFalse(result)

    # ── Path B: floor override ────────────────────────────────────────────────

    def test_path_b_floor_override_confirms_minor_trigger(self) -> None:
        """Minor trigger blocked by Path A, but basket_pnl <= floor → Path B confirms."""
        legs = [_make_open_record(f"p{i}", "bsk", last_unrealized=-0.03) for i in range(5)]
        records = _records_from_list(*legs)
        # basket_pnl = -0.15; floor = -0.10; -0.15 <= -0.10 → Path B confirms.
        result = _basket_exit_confirmed(records, "bsk", "p0", _gate_config(floor=-0.10))
        self.assertTrue(result)

    def test_path_b_floor_not_reached_does_not_confirm(self) -> None:
        """basket_pnl above floor, Path A fails, drawdown disabled → gate blocks."""
        legs = [
            _make_open_record("p0", "bsk", last_unrealized=-0.008),  # trigger, smallest
            _make_open_record("p1", "bsk", last_unrealized=-0.011),
            _make_open_record("p2", "bsk", last_unrealized=-0.012),  # dominant @ 0.012/0.050 = 0.24
            _make_open_record("p3", "bsk", last_unrealized=-0.009),
            _make_open_record("p4", "bsk", last_unrealized=-0.010),
        ]
        records = _records_from_list(*legs)
        # basket_pnl = -0.050 > floor (-0.10) → floor check fails.
        # drawdown disabled (0.0) to prevent basket_peak=0 artifact.
        # Path A: 0.012/0.050 = 0.24 < 0.40 AND trigger != dominant → False → blocked.
        cfg = _gate_config(dominance=0.40, drawdown=0.0, floor=-0.10)
        result = _basket_exit_confirmed(records, "bsk", "p0", cfg)
        self.assertFalse(result)

    def test_path_b_floor_disabled_when_zero(self) -> None:
        """basket_unrealized_pnl_floor=0.0 → Path B floor check not used.

        Even with basket_pnl well below zero, gate blocks because floor is disabled
        and Path A has no dominant leg.
        """
        legs = [
            _make_open_record("p0", "bsk", last_unrealized=-0.08),
            _make_open_record("p1", "bsk", last_unrealized=-0.11),
            _make_open_record("p2", "bsk", last_unrealized=-0.12),
            _make_open_record("p3", "bsk", last_unrealized=-0.09),
            _make_open_record("p4", "bsk", last_unrealized=-0.10),
        ]
        records = _records_from_list(*legs)
        # basket_pnl = -0.50; floor=0.0 → disabled; drawdown=0.0 → disabled.
        # Path A: 0.24 < 0.40 AND trigger (p0) != dominant (p2) → blocked.
        cfg = PaperConfig(
            basket_dominance_threshold=0.40,
            basket_drawdown_exit_threshold=0.0,
            basket_unrealized_pnl_floor=0.0,
        )
        result = _basket_exit_confirmed(records, "bsk", "p0", cfg)
        self.assertFalse(result)

    # ── Cascade safety: suppressed trigger blocks cascade ────────────────────

    def test_suppressed_trigger_leaves_no_ed_close_so_cascade_never_fires(self) -> None:
        """When gate blocks a trigger, no close_reason='EDGE_DECAY' is set.
        _has_basket_ed_exit must return False, preventing cascade for all siblings.

        Uses a 5-leg distributed basket where no single leg dominates (< 40% share)
        so the gate blocks the minor trigger.  A 2-leg basket with a massive dominant
        loser would confirm (correct behaviour — see test_confirmed_trigger_allows_cascade).
        """
        # Minor trigger p0; p2 is largest but still < 40% share of distributed loss.
        legs = [
            _make_open_record("p0", "bsk2", last_unrealized=-0.08),  # trigger
            _make_open_record("p1", "bsk2", last_unrealized=-0.11),
            _make_open_record("p2", "bsk2", last_unrealized=-0.12),
            _make_open_record("p3", "bsk2", last_unrealized=-0.09),
            _make_open_record("p4", "bsk2", last_unrealized=-0.10),
        ]
        records = _records_from_list(*legs)

        # Gate blocks p0 (minor trigger): Path A fails (0.24 < 0.40), Path B disabled.
        cfg = _gate_config(dominance=0.40, drawdown=0.0, floor=0.0)
        confirmed = _basket_exit_confirmed(records, "bsk2", "p0", cfg)
        self.assertFalse(confirmed)

        # Because gate blocked it, no position is ever closed with EDGE_DECAY.
        # _has_basket_ed_exit must return False for all siblings.
        for sibling_id in ["p1", "p2", "p3", "p4"]:
            cascade_fires = _has_basket_ed_exit(records, "bsk2", sibling_id)
            self.assertFalse(cascade_fires, f"Cascade must not fire for {sibling_id} when trigger was suppressed")

    def test_confirmed_trigger_allows_cascade_for_sibling(self) -> None:
        """When gate confirms trigger (dominant leg), closing it allows cascade for siblings."""
        r1 = _make_open_record("p1", "bsk", last_unrealized=-0.50)  # dominant
        r2 = _make_open_record("p2", "bsk", last_unrealized=-0.05)
        records = _records_from_list(r1, r2)

        confirmed = _basket_exit_confirmed(records, "bsk", "p1", _gate_config())
        self.assertTrue(confirmed)

        # Simulate the runner closing p1 with EDGE_DECAY (gate passed).
        closed_r1 = _make_closed_record(
            "p1", "bsk", close_reason="EDGE_DECAY", realized_pnl=-0.50,
            closed_ts=_BASE_TS.isoformat(),
        )
        records["p1"] = closed_r1

        # Now cascade check for p2 must fire.
        cascade_fires = _has_basket_ed_exit(records, "bsk", "p2")
        self.assertTrue(cascade_fires, "Cascade must fire after a confirmed EDGE_DECAY close")

    # ── None candidate_id with active gate ───────────────────────────────────

    def test_none_candidate_id_active_gate_confirms(self) -> None:
        """candidate_id=None with active gate → confirm (no basket context available)."""
        result = _basket_exit_confirmed({}, None, "p1", _gate_config())
        self.assertTrue(result)

    # ── Empty basket with active gate ────────────────────────────────────────

    def test_empty_basket_active_gate_confirms(self) -> None:
        """No legs found for candidate_id → confirm (safe fallback)."""
        result = _basket_exit_confirmed({}, "nonexistent-basket", "p1", _gate_config())
        self.assertTrue(result)


class IdleHoldReleaseTests(unittest.TestCase):

    def test_idle_release_eligible_for_clearly_inert_basket(self) -> None:
        cfg = PaperConfig(
            max_holding_sec=300.0,
            idle_hold_release_check_sec=180.0,
            idle_hold_release_max_repricing_events=1,
            idle_hold_release_max_abs_unrealized_pnl=0.01,
            idle_hold_release_max_drawdown=0.01,
        )
        r1 = _make_open_record("p1", "idle-bsk", last_unrealized=0.004, peak_unrealized=0.004)
        r2 = _make_open_record("p2", "idle-bsk", last_unrealized=-0.003, peak_unrealized=0.0)
        r1.opened_ts = (_BASE_TS - timedelta(seconds=240)).isoformat()
        r2.opened_ts = (_BASE_TS - timedelta(seconds=240)).isoformat()
        r1.repricing_event_count = 1
        r2.repricing_event_count = 0
        records = _records_from_list(r1, r2)

        self.assertTrue(_basket_idle_release_eligible(records, "idle-bsk", cfg, _BASE_TS))

    def test_idle_release_blocks_weak_signal_basket(self) -> None:
        cfg = PaperConfig(
            max_holding_sec=300.0,
            idle_hold_release_check_sec=180.0,
            idle_hold_release_max_repricing_events=1,
            idle_hold_release_max_abs_unrealized_pnl=0.01,
            idle_hold_release_max_drawdown=0.01,
        )
        r1 = _make_open_record("p1", "weak-bsk", last_unrealized=0.004, peak_unrealized=0.004)
        r2 = _make_open_record("p2", "weak-bsk", last_unrealized=-0.003, peak_unrealized=0.0)
        r1.opened_ts = (_BASE_TS - timedelta(seconds=240)).isoformat()
        r2.opened_ts = (_BASE_TS - timedelta(seconds=240)).isoformat()
        r1.edge_decay_candidate_count = 1
        records = _records_from_list(r1, r2)

        self.assertFalse(_basket_idle_release_eligible(records, "weak-bsk", cfg, _BASE_TS))

    def test_idle_release_blocks_after_adverse_state(self) -> None:
        cfg = PaperConfig(
            max_holding_sec=300.0,
            idle_hold_release_check_sec=180.0,
            idle_hold_release_max_repricing_events=1,
            idle_hold_release_max_abs_unrealized_pnl=0.01,
            idle_hold_release_max_drawdown=0.01,
        )
        r1 = _make_open_record("p1", "adv-bsk", last_unrealized=0.002, peak_unrealized=0.002)
        r2 = _make_open_record("p2", "adv-bsk", last_unrealized=-0.001, peak_unrealized=0.0)
        r1.opened_ts = (_BASE_TS - timedelta(seconds=240)).isoformat()
        r2.opened_ts = (_BASE_TS - timedelta(seconds=240)).isoformat()
        r2.first_adverse_ts = (_BASE_TS - timedelta(seconds=60)).isoformat()
        records = _records_from_list(r1, r2)

        self.assertFalse(_basket_idle_release_eligible(records, "adv-bsk", cfg, _BASE_TS))

    def test_idle_release_cascade_only_follows_same_reason(self) -> None:
        idle_closed = _make_closed_record(
            "p1",
            "idle-bsk",
            close_reason="IDLE_HOLD_RELEASE",
            realized_pnl=0.0,
            closed_ts=_BASE_TS.isoformat(),
        )
        open_sibling = _make_open_record("p2", "idle-bsk", last_unrealized=0.0)
        records = _records_from_list(idle_closed, open_sibling)

        self.assertTrue(_has_basket_idle_release(records, "idle-bsk", "p2"))
        self.assertFalse(_has_basket_ed_exit(records, "idle-bsk", "p2"))


if __name__ == "__main__":
    unittest.main()
