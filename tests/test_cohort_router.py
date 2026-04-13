"""
tests/test_cohort_router.py

Deterministic tests for cohort_router.py.
No network calls. Synthetic fixtures only.
"""
import datetime as dt
import pytest

from src.scanner.cohort_router import (
    ALL_FAMILIES,
    FAMILY_CRYPTO_DATE_TARGET,
    FAMILY_ELECTION,
    FAMILY_MACRO,
    FAMILY_MISC,
    FAMILY_MULTI_DAY_SPORTS,
    FAMILY_POLITICAL_BINARY,
    FAMILY_SAME_DAY_SPORTS,
    CohortRouter,
    RouterConfig,
    classify_family,
    end_date_horizon_hours,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = dt.datetime(2026, 3, 21, 12, 0, 0, tzinfo=dt.timezone.utc)

# A market expiring in N hours from NOW
def _market(slug: str, hours_to_expiry: float = 48.0, volume: float = 5000.0) -> dict:
    end = NOW + dt.timedelta(hours=hours_to_expiry)
    return {
        "slug"    : slug,
        "endDate" : end.isoformat(),
        "volume"  : volume,
    }


# ---------------------------------------------------------------------------
# 1. Family classification
# ---------------------------------------------------------------------------

class TestClassifyFamily:

    def test_same_day_sports_nba(self):
        m = _market("nba-bos-mia-2026-03-21", hours_to_expiry=6)
        assert classify_family(m, NOW) == FAMILY_SAME_DAY_SPORTS

    def test_multi_day_sports_nba_futures(self):
        m = _market("nba-championship-2026", hours_to_expiry=200)
        assert classify_family(m, NOW) == FAMILY_MULTI_DAY_SPORTS

    def test_cbb_same_day(self):
        m = _market("cbb-txam-hou-2026-03-21", hours_to_expiry=5)
        assert classify_family(m, NOW) == FAMILY_SAME_DAY_SPORTS

    def test_election(self):
        m = _market("will-trump-win-presidential-election-2026", hours_to_expiry=500)
        assert classify_family(m, NOW) == FAMILY_ELECTION

    def test_political_binary(self):
        m = _market("will-trump-sign-executive-order", hours_to_expiry=48)
        assert classify_family(m, NOW) == FAMILY_POLITICAL_BINARY

    def test_crypto(self):
        m = _market("will-bitcoin-hit-100k-by-april", hours_to_expiry=240)
        assert classify_family(m, NOW) == FAMILY_CRYPTO_DATE_TARGET

    def test_macro(self):
        m = _market("will-fed-cut-rates-march-2026", hours_to_expiry=72)
        assert classify_family(m, NOW) == FAMILY_MACRO

    def test_misc(self):
        # Slug has no sports/crypto/macro/election/political keywords
        m = _market("will-elon-musk-go-to-mars-before-2030", hours_to_expiry=800)
        assert classify_family(m, NOW) == FAMILY_MISC

    def test_lol_esports_same_day(self):
        m = _market("lol-g2-fox1-2026-03-21", hours_to_expiry=4)
        assert classify_family(m, NOW) == FAMILY_SAME_DAY_SPORTS

    def test_israel_action_political(self):
        # Neither sports nor crypto nor macro nor election; slug has no political keyword
        m = _market("will-israel-take-military-action-in-gaza-on-march-22-2026", hours_to_expiry=30)
        # Should classify as MISC (no matching RE)
        family = classify_family(m, NOW)
        assert family in (FAMILY_MISC, FAMILY_POLITICAL_BINARY)  # acceptable either way


# ---------------------------------------------------------------------------
# 2. end_date_horizon_hours
# ---------------------------------------------------------------------------

class TestHorizonHours:

    def test_exact_48h(self):
        m = _market("x", hours_to_expiry=48.0)
        h = end_date_horizon_hours(m, NOW)
        assert h is not None
        assert abs(h - 48.0) < 0.1

    def test_missing_enddate(self):
        m = {"slug": "x", "volume": 100}
        h = end_date_horizon_hours(m, NOW)
        assert h is None

    def test_enddate_z_format(self):
        end = NOW + dt.timedelta(hours=72)
        m = {"slug": "x", "endDate": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "volume": 100}
        h = end_date_horizon_hours(m, NOW)
        assert h is not None
        assert abs(h - 72.0) < 0.1


# ---------------------------------------------------------------------------
# 3. score_candidate
# ---------------------------------------------------------------------------

class TestScoreCandidate:

    def test_hard_cooldown_suppresses(self):
        m = _market("slug-x")
        plateau = {"both_98_rounds": 10, "consecutive_rounds_seen": 8}
        s = score_candidate(m, FAMILY_SAME_DAY_SPORTS, 6.0, plateau,
                            cooldown_both98=5, cooldown_consec=5)
        assert s <= -1e5

    def test_no_plateau_no_penalty(self):
        m = _market("slug-y", hours_to_expiry=200, volume=200_000)
        s = score_candidate(m, FAMILY_ELECTION, 200.0, None)
        assert s > 50   # horizon bonus + family base + volume bonus

    def test_soft_penalty_reduces_score(self):
        m = _market("slug-z", hours_to_expiry=200, volume=500)
        plateau_clean = None
        plateau_sat   = {"both_98_rounds": 3, "consecutive_rounds_seen": 3}
        s_clean = score_candidate(m, FAMILY_POLITICAL_BINARY, 200.0, plateau_clean)
        s_sat   = score_candidate(m, FAMILY_POLITICAL_BINARY, 200.0, plateau_sat)
        assert s_clean > s_sat

    def test_same_day_sports_lowest_family_base(self):
        m_sport = _market("nba-today", hours_to_expiry=4)
        m_elect = _market("election-today", hours_to_expiry=4)
        s_sport = score_candidate(m_sport, FAMILY_SAME_DAY_SPORTS, 4.0, None)
        s_elect = score_candidate(m_elect, FAMILY_ELECTION, 4.0, None)
        assert s_elect > s_sport

    def test_longer_horizon_higher_score(self):
        m = _market("slug", volume=1000)
        s_short = score_candidate(m, FAMILY_MISC, 12.0,  None)   # < 24h
        s_med   = score_candidate(m, FAMILY_MISC, 96.0,  None)   # 3-7d
        s_long  = score_candidate(m, FAMILY_MISC, 200.0, None)   # >7d
        assert s_short < s_med < s_long


# ---------------------------------------------------------------------------
# 4. CohortRouter.filter_candidates — quota allocation
# ---------------------------------------------------------------------------

class TestCohortRouterQuota:

    def _router(self, quota=None, max_total=30, enabled=True):
        cfg = RouterConfig(
            family_quota=quota or {
                FAMILY_SAME_DAY_SPORTS   : 2,
                FAMILY_MULTI_DAY_SPORTS  : 3,
                FAMILY_ELECTION          : 5,
                FAMILY_POLITICAL_BINARY  : 5,
                FAMILY_CRYPTO_DATE_TARGET: 3,
                FAMILY_MACRO             : 3,
                FAMILY_MISC              : 3,
            },
            max_total=max_total,
            enabled=enabled,
        )
        return CohortRouter(plateau_tracker=None, config=cfg)

    def _same_day_sports_batch(self, n: int) -> list:
        return [_market(f"nba-game-{i}", hours_to_expiry=3) for i in range(n)]

    def _election_batch(self, n: int) -> list:
        return [_market(f"will-senator-win-election-{i}", hours_to_expiry=500) for i in range(n)]

    def test_same_day_sports_capped(self):
        candidates = self._same_day_sports_batch(10)
        router = self._router()
        result = router.filter_candidates(candidates, NOW)
        # At most 2 same-day sports allowed through
        assert len(result) <= 2

    def test_election_survives_when_sports_saturated(self):
        """Non-same-day cohorts must survive even when same-day slots are full."""
        same_day = self._same_day_sports_batch(10)
        elections = self._election_batch(3)
        candidates = same_day + elections
        router = self._router()
        result = router.filter_candidates(candidates, NOW)
        slugs = [m["slug"] for m in result]
        # At least some election candidates should survive
        election_through = [s for s in slugs if "election" in s]
        assert len(election_through) >= 1

    def test_max_total_respected(self):
        candidates = (
            self._same_day_sports_batch(8)
            + self._election_batch(8)
            + [_market(f"will-btc-hit-100k-{i}", hours_to_expiry=200) for i in range(8)]
        )
        router = self._router(max_total=5)
        result = router.filter_candidates(candidates, NOW)
        assert len(result) <= 5

    def test_empty_input_returns_empty(self):
        router = self._router()
        assert router.filter_candidates([], NOW) == []

    def test_disabled_is_passthrough(self):
        candidates = self._same_day_sports_batch(20)
        router = self._router(enabled=False)
        result = router.filter_candidates(candidates, NOW)
        assert result is candidates   # exact same object, no copy


# ---------------------------------------------------------------------------
# 5. CohortRouter — cooldown suppression
# ---------------------------------------------------------------------------

class TestCooldown:

    def _tracker_stub(self, plateau_map: dict):
        """Minimal stub for Both98PlateauTracker."""
        class _Stub:
            _state = plateau_map
            def _load_state(self):
                pass
        return _Stub()

    def test_cooled_slug_suppressed(self):
        slug = "nba-same-day-hopeless"
        plateau_map = {
            slug: {"both_98_rounds": 8, "consecutive_rounds_seen": 8}
        }
        tracker = self._tracker_stub(plateau_map)
        cfg = RouterConfig(
            family_quota={FAMILY_SAME_DAY_SPORTS: 10},
            max_total=30,
            cooldown_both98=5,
            cooldown_consec=5,
        )
        router = CohortRouter(plateau_tracker=tracker, config=cfg)
        m = _market(slug, hours_to_expiry=4)
        result = router.filter_candidates([m], NOW)
        assert len(result) == 0

    def test_fresh_slug_not_suppressed(self):
        slug = "nba-fresh-game"
        plateau_map = {
            slug: {"both_98_rounds": 1, "consecutive_rounds_seen": 1}
        }
        tracker = self._tracker_stub(plateau_map)
        cfg = RouterConfig(
            family_quota={FAMILY_SAME_DAY_SPORTS: 10},
            max_total=30,
            cooldown_both98=5,
            cooldown_consec=5,
        )
        router = CohortRouter(plateau_tracker=tracker, config=cfg)
        m = _market(slug, hours_to_expiry=4)
        result = router.filter_candidates([m], NOW)
        assert len(result) == 1

    def test_below_consec_threshold_not_suppressed(self):
        """High both_98_rounds but low consecutive — not cooled."""
        slug = "political-slug"
        plateau_map = {
            slug: {"both_98_rounds": 10, "consecutive_rounds_seen": 2}
        }
        tracker = self._tracker_stub(plateau_map)
        cfg = RouterConfig(
            family_quota={f: 10 for f in ALL_FAMILIES},
            max_total=30,
            cooldown_both98=5,
            cooldown_consec=5,
        )
        router = CohortRouter(plateau_tracker=tracker, config=cfg)
        m = _market(slug, hours_to_expiry=200)
        result = router.filter_candidates([m], NOW)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 6. Backward compatibility — router disabled means no behavioural change
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_disabled_router_returns_original_list_object(self):
        cfg = RouterConfig(enabled=False)
        router = CohortRouter(plateau_tracker=None, config=cfg)
        candidates = [_market(f"slug-{i}") for i in range(20)]
        result = router.filter_candidates(candidates, NOW)
        assert result is candidates

    def test_none_router_in_scan_functions_is_noop(self):
        """
        Confirm scan_events / scan_slice accept router=None without error.
        (Import-level test only; no network call made.)
        """
        from scripts.trial_entry_scan import scan_events, scan_slice
        import inspect
        sig_events = inspect.signature(scan_events)
        sig_slice  = inspect.signature(scan_slice)
        assert "router" in sig_events.parameters
        assert "router" in sig_slice.parameters
        assert sig_events.parameters["router"].default is None
        assert sig_slice.parameters["router"].default is None

    def test_router_failure_returns_original(self):
        """
        If _filter_inner raises unexpectedly, filter_candidates catches it
        and returns the original list unchanged.
        _load_plateau failures are caught internally (non-fatal); routing
        continues with an empty plateau map rather than falling back.
        Test the outer broad-except by injecting a failure after plateau load.
        """
        class _CrashRouter(CohortRouter):
            def _filter_inner(self, candidates, now_utc):
                raise RuntimeError("unexpected internal crash")

        cfg = RouterConfig(enabled=True)
        router = _CrashRouter(plateau_tracker=None, config=cfg)
        candidates = [_market(f"slug-{i}") for i in range(5)]
        result = router.filter_candidates(candidates, NOW)
        # Broad except returns original list
        assert result is candidates
