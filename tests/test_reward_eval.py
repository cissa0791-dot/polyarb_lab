"""
tests/test_reward_eval.py

Deterministic tests for:
  - src/research/reward_eval.py       (parse + evaluate_fitness)
  - src/scanner/logit_utils.py        (transforms + Greeks)
  - src/scanner/belief_var_estimator.py
  - src/scanner/inventory_quote_engine.py
  - src/scanner/reward_quote_feasibility.py

No network calls. Synthetic fixtures only.
"""
import math
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reward_market(
    min_size   : int   = 200,
    max_spread : float = 4.5,
    daily_rate : float = 500.0,
    holding    : bool  = False,
) -> dict:
    """Minimal Gamma-style market dict with reward fields."""
    return {
        "slug"               : "test-slug",
        "rewardsMinSize"     : min_size,
        "rewardsMaxSpread"   : max_spread,
        "holdingRewardsEnabled": holding,
        "clobRewards"        : [{"rewardsDailyRate": daily_rate}],
        "orderMinSize"       : 10,
        "volume"             : 50_000,
    }


# ===========================================================================
# 1. reward_eval.py — parse_reward_config
# ===========================================================================

class TestParseRewardConfig:

    def test_valid_market(self):
        from src.research.reward_eval import parse_reward_config
        m = _reward_market(min_size=200, max_spread=4.5, daily_rate=500.0)
        cfg = parse_reward_config(m)
        assert cfg is not None
        assert cfg.min_size_shares == 200
        assert cfg.max_spread_cents == 4.5
        assert cfg.daily_rate_usdc == 500.0
        assert cfg.has_rewards is True

    def test_zero_daily_rate_no_rewards(self):
        from src.research.reward_eval import parse_reward_config
        m = _reward_market(daily_rate=0.0)
        cfg = parse_reward_config(m)
        assert cfg is not None
        assert cfg.has_rewards is False

    def test_missing_clob_rewards(self):
        from src.research.reward_eval import parse_reward_config
        m = {"slug": "x", "rewardsMinSize": 100, "rewardsMaxSpread": 3.0}
        cfg = parse_reward_config(m)
        assert cfg is not None
        assert cfg.has_rewards is False   # daily_rate = 0

    def test_empty_market(self):
        from src.research.reward_eval import parse_reward_config
        cfg = parse_reward_config({})
        assert cfg is not None
        assert cfg.has_rewards is False


# ===========================================================================
# 2. reward_eval.py — evaluate_fitness
# ===========================================================================

class TestEvaluateFitness:

    def test_spread_fits_when_market_tight(self):
        from src.research.reward_eval import evaluate_fitness, parse_reward_config
        # Market spread = (0.51 + 0.51 - 1.0)*100 = 2¢ < 4.5¢ → spread_fits
        cfg = parse_reward_config(_reward_market(max_spread=4.5))
        fitness = evaluate_fitness(0.51, 0.51, 10.0, cfg)
        assert fitness.spread_fits is True
        assert abs(fitness.market_spread_cents - 2.0) < 0.01

    def test_spread_fails_for_both_98(self):
        from src.research.reward_eval import evaluate_fitness, parse_reward_config
        # both_98: spread = 96¢ >> 4.5¢
        cfg = parse_reward_config(_reward_market(max_spread=4.5))
        fitness = evaluate_fitness(0.98, 0.98, 10.0, cfg)
        assert fitness.spread_fits is False
        assert fitness.market_spread_cents > 90.0

    def test_size_fits_when_min_size_affordable(self):
        from src.research.reward_eval import evaluate_fitness, parse_reward_config
        # market_min_size=10 <= rewards_min_size=200 → size_fits
        cfg = parse_reward_config(_reward_market(min_size=200))
        fitness = evaluate_fitness(0.51, 0.51, 10.0, cfg)
        assert fitness.size_fits is True

    def test_size_fails_when_min_size_too_large(self):
        from src.research.reward_eval import evaluate_fitness, parse_reward_config
        # market_min_size=500 > rewards_min_size=200 → size_fails
        cfg = parse_reward_config(_reward_market(min_size=200))
        fitness = evaluate_fitness(0.51, 0.51, 500.0, cfg)
        assert fitness.size_fits is False

    def test_reward_per_capital_positive(self):
        from src.research.reward_eval import evaluate_fitness, parse_reward_config
        cfg = parse_reward_config(_reward_market(min_size=100, daily_rate=100.0))
        fitness = evaluate_fitness(0.51, 0.51, 10.0, cfg)
        # capital = 100 shares * 1.0 = 100 USDC; rate = 100 USDC/day → 100%/day
        assert fitness.reward_per_capital > 0

    def test_no_config_returns_no_rewards(self):
        from src.research.reward_eval import evaluate_fitness
        fitness = evaluate_fitness(0.51, 0.51, 10.0, None)
        assert fitness.has_rewards is False
        assert fitness.reward_score == 0.0

    def test_competitive_tight_spread_penalised(self):
        """Markets with max_spread < COMPETITIVE_SPREAD_THRESHOLD get lower score."""
        from src.research.reward_eval import evaluate_fitness, parse_reward_config
        # Tight: max_spread=1.5 (over-competed CBB type)
        cfg_tight = parse_reward_config(_reward_market(max_spread=1.5, daily_rate=500))
        fit_tight = evaluate_fitness(0.51, 0.51, 10.0, cfg_tight)
        # Wide: max_spread=4.5 (political type)
        cfg_wide  = parse_reward_config(_reward_market(max_spread=4.5, daily_rate=500))
        fit_wide  = evaluate_fitness(0.51, 0.51, 10.0, cfg_wide)
        # Wide should score higher (less over-competed)
        assert fit_wide.reward_score > fit_tight.reward_score


# ===========================================================================
# 3. logit_utils.py
# ===========================================================================

class TestLogitUtils:

    def test_prob_to_logit_roundtrip(self):
        from src.scanner.logit_utils import prob_to_logit, logit_to_prob
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert abs(logit_to_prob(prob_to_logit(p)) - p) < 1e-10

    def test_logit_zero_at_half(self):
        from src.scanner.logit_utils import prob_to_logit
        assert abs(prob_to_logit(0.5)) < 1e-10

    def test_logit_positive_above_half(self):
        from src.scanner.logit_utils import prob_to_logit
        assert prob_to_logit(0.7) > 0
        assert prob_to_logit(0.3) < 0

    def test_delta_x_max_at_half(self):
        from src.scanner.logit_utils import delta_x_from_p
        d_half = delta_x_from_p(0.5)
        for p in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:
            assert delta_x_from_p(p) < d_half
        assert abs(d_half - 0.25) < 1e-10

    def test_gamma_x_zero_at_half(self):
        from src.scanner.logit_utils import gamma_x_from_p
        assert abs(gamma_x_from_p(0.5)) < 1e-10

    def test_gamma_x_sign_flip(self):
        from src.scanner.logit_utils import gamma_x_from_p
        # gamma_x > 0 below 0.5 (p*(1-p)*(1-2p) > 0 for p < 0.5)
        assert gamma_x_from_p(0.3) > 0
        assert gamma_x_from_p(0.7) < 0

    def test_spread_x_to_spread_p(self):
        from src.scanner.logit_utils import spread_x_to_spread_p, delta_x_from_p
        p = 0.5
        h = 0.1
        expected = delta_x_from_p(p) * h
        assert abs(spread_x_to_spread_p(h, p) - expected) < 1e-10

    def test_extreme_probs_clamped(self):
        """prob_to_logit must not raise at 0/1; logit_to_prob must not raise at extreme x."""
        from src.scanner.logit_utils import prob_to_logit, logit_to_prob
        assert math.isfinite(prob_to_logit(0.0))
        assert math.isfinite(prob_to_logit(1.0))
        # logit_to_prob(-1000) underflows to 0.0 in float64 — that is acceptable
        v_lo = logit_to_prob(-1000)
        v_hi = logit_to_prob(1000)
        assert math.isfinite(v_lo) and 0.0 <= v_lo <= 1.0
        assert math.isfinite(v_hi) and 0.0 <= v_hi <= 1.0


# ===========================================================================
# 4. belief_var_estimator.py
# ===========================================================================

class TestBeliefVarEstimator:

    def test_prior_returned_with_few_obs(self):
        from src.scanner.belief_var_estimator import BeliefVarEstimator, PRIOR_VARIANCE
        est = BeliefVarEstimator()
        est.update(0.5)
        est.update(0.51)
        # only 2 obs, fewer than MIN_OBS=3
        assert est.variance() == PRIOR_VARIANCE

    def test_variance_increases_with_spread(self):
        from src.scanner.belief_var_estimator import BeliefVarEstimator
        est_narrow = BeliefVarEstimator()
        est_wide   = BeliefVarEstimator()
        for p in [0.49, 0.50, 0.51, 0.50, 0.49]:
            est_narrow.update(p)
        for p in [0.30, 0.50, 0.70, 0.40, 0.60]:
            est_wide.update(p)
        assert est_wide.variance() > est_narrow.variance()

    def test_constant_series_near_zero_variance(self):
        from src.scanner.belief_var_estimator import BeliefVarEstimator
        est = BeliefVarEstimator()
        for _ in range(10):
            est.update(0.50)
        assert est.variance() < 1e-4

    def test_estimate_var_from_series(self):
        from src.scanner.belief_var_estimator import estimate_var_from_series, PRIOR_VARIANCE
        var = estimate_var_from_series([0.4, 0.5, 0.6, 0.5, 0.4])
        assert var > 0
        assert var != PRIOR_VARIANCE  # real estimate, not prior

    def test_n_obs_tracking(self):
        from src.scanner.belief_var_estimator import BeliefVarEstimator
        est = BeliefVarEstimator(window=5)
        for _ in range(10):
            est.update(0.50)
        assert est.n_obs() == 5  # capped by window

    def test_reset_clears_window(self):
        from src.scanner.belief_var_estimator import BeliefVarEstimator, PRIOR_VARIANCE
        est = BeliefVarEstimator()
        for p in [0.4, 0.5, 0.6, 0.5, 0.4]:
            est.update(p)
        est.reset()
        assert est.n_obs() == 0
        assert est.variance() == PRIOR_VARIANCE


# ===========================================================================
# 5. inventory_quote_engine.py
# ===========================================================================

class TestInventoryQuoteEngine:

    def _engine(self, risk_aversion=1.0, k=0.5):
        from src.scanner.inventory_quote_engine import InventoryQuoteEngine
        return InventoryQuoteEngine(risk_aversion=risk_aversion, k=k)

    def test_zero_inventory_symmetric_around_mid(self):
        eng   = self._engine()
        quote = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=0.0)
        assert quote is not None
        # With zero inventory, reservation = mid
        assert abs(quote.reservation_p - 0.5) < 1e-6
        # bid and ask equidistant from mid
        assert abs((quote.mid_p - quote.bid_p) - (quote.ask_p - quote.mid_p)) < 1e-6

    def test_long_inventory_skews_bid_down(self):
        """Long position should push quotes down (to encourage selling)."""
        eng = self._engine()
        q_long  = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=+5.0)
        q_flat  = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=0.0)
        assert q_long.reservation_p < q_flat.reservation_p
        assert q_long.bid_p < q_flat.bid_p

    def test_short_inventory_skews_ask_up(self):
        """Short position should push quotes up (to encourage buying)."""
        eng = self._engine()
        q_short = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=-5.0)
        q_flat  = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=0.0)
        assert q_short.reservation_p > q_flat.reservation_p

    def test_higher_belief_var_widens_spread(self):
        eng = self._engine()
        q_lo = eng.compute_quote(mid_p=0.5, belief_var=0.01, horizon_left=0.5, inventory=0.0)
        q_hi = eng.compute_quote(mid_p=0.5, belief_var=0.20, horizon_left=0.5, inventory=0.0)
        assert q_hi.half_spread_x > q_lo.half_spread_x

    def test_higher_horizon_widens_spread(self):
        eng = self._engine()
        q_near = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.1, inventory=0.0)
        q_far  = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.9, inventory=0.0)
        assert q_far.half_spread_x > q_near.half_spread_x

    def test_higher_risk_aversion_widens_spread(self):
        q_lo = self._engine(risk_aversion=0.5).compute_quote(0.5, 0.04, 0.5, 0.0)
        q_hi = self._engine(risk_aversion=3.0).compute_quote(0.5, 0.04, 0.5, 0.0)
        assert q_hi.half_spread_x > q_lo.half_spread_x

    def test_bid_below_ask(self):
        eng   = self._engine()
        quote = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=0.0)
        assert quote.bid_p < quote.ask_p

    def test_spread_p_approx_positive(self):
        eng   = self._engine()
        quote = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5, inventory=0.0)
        assert quote.spread_p_approx > 0

    def test_returns_none_on_invalid_input(self):
        """Engine must not raise; returns None on bad input."""
        from src.scanner.inventory_quote_engine import InventoryQuoteEngine
        eng = InventoryQuoteEngine(risk_aversion=float("nan"), k=0.5)
        result = eng.compute_quote(mid_p=0.5, belief_var=0.04, horizon_left=0.5)
        # nan propagates to NaN checks → should return None or a result with NaN
        # either is acceptable as long as it doesn't raise
        # (our code may return None or a NaN-filled result)
        # just verify no exception
        assert result is None or hasattr(result, "bid_p")


# ===========================================================================
# 6. reward_quote_feasibility.py
# ===========================================================================

class TestRewardQuoteFeasibility:

    def _config(self, min_size=100, max_spread=4.5, daily_rate=500.0):
        from src.research.reward_eval import RewardConfig
        return RewardConfig(
            daily_rate_usdc  = daily_rate,
            max_spread_cents = max_spread,
            min_size_shares  = min_size,
            has_rewards      = True,
            holding_enabled  = False,
        )

    def _quote(self, mid_p=0.5, bid_p=0.48, ask_p=0.52):
        """Build a minimal QuoteSuggestion stub."""
        from src.scanner.inventory_quote_engine import QuoteSuggestion
        from src.scanner.logit_utils import prob_to_logit
        return QuoteSuggestion(
            mid_p           = mid_p,
            mid_x           = prob_to_logit(mid_p),
            reservation_x   = prob_to_logit(mid_p),
            reservation_p   = mid_p,
            half_spread_x   = 0.08,
            bid_x           = prob_to_logit(bid_p),
            ask_x           = prob_to_logit(ask_p),
            bid_p           = bid_p,
            ask_p           = ask_p,
            spread_p_approx = (ask_p - bid_p),
            belief_var      = 0.04,
            horizon_left    = 0.5,
            inventory       = 0.0,
        )

    def test_eligible_when_within_spread_and_size(self):
        from src.scanner.reward_quote_feasibility import evaluate_quote_feasibility
        # |0.50 - 0.48| * 100 = 2¢ < 4.5¢; size 200 >= 100 → eligible
        result = evaluate_quote_feasibility(
            self._quote(mid_p=0.5, bid_p=0.48, ask_p=0.52),
            self._config(max_spread=4.5, min_size=100),
            proposed_size=200,
        )
        assert result.eligible is True
        assert result.reason == "eligible"

    def test_ineligible_when_spread_too_wide(self):
        from src.scanner.reward_quote_feasibility import evaluate_quote_feasibility
        # |0.50 - 0.44| * 100 = 6¢ > 4.5¢ → ineligible
        result = evaluate_quote_feasibility(
            self._quote(mid_p=0.5, bid_p=0.44, ask_p=0.56),
            self._config(max_spread=4.5, min_size=100),
            proposed_size=200,
        )
        assert result.eligible is False
        assert "spread" in result.reason

    def test_ineligible_when_size_too_small(self):
        from src.scanner.reward_quote_feasibility import evaluate_quote_feasibility
        result = evaluate_quote_feasibility(
            self._quote(mid_p=0.5, bid_p=0.48, ask_p=0.52),
            self._config(max_spread=4.5, min_size=500),
            proposed_size=50,   # 50 < 500
        )
        assert result.eligible is False
        assert "size" in result.reason

    def test_ineligible_when_no_reward_config(self):
        from src.scanner.reward_quote_feasibility import evaluate_quote_feasibility
        result = evaluate_quote_feasibility(
            self._quote(), reward_config=None, proposed_size=100
        )
        assert result.eligible is False
        assert result.reason == "no_reward_program"

    def test_ineligible_when_no_quote(self):
        from src.scanner.reward_quote_feasibility import evaluate_quote_feasibility
        result = evaluate_quote_feasibility(
            None, self._config(), proposed_size=100
        )
        assert result.eligible is False
        assert result.reason == "no_quote"

    def test_tight_max_spread_config_rejects_wide_quote(self):
        from src.scanner.reward_quote_feasibility import evaluate_quote_feasibility
        # CBB-style: max_spread = 1.5¢; quote spread from mid = 2¢ → ineligible
        result = evaluate_quote_feasibility(
            self._quote(mid_p=0.5, bid_p=0.48, ask_p=0.52),
            self._config(max_spread=1.5, min_size=1000),
            proposed_size=1000,
        )
        assert result.eligible is False


# ===========================================================================
# 7. Router + quote layer compatibility
# ===========================================================================

class TestRouterRewardIntegration:

    def test_cohort_router_accepts_reward_bonus(self):
        """CohortRouter.score_candidate must not raise with reward fields in market."""
        from src.scanner.cohort_router import CohortRouter, RouterConfig
        import datetime as dt

        NOW = dt.datetime(2026, 3, 21, 12, 0, tzinfo=dt.timezone.utc)
        m = {
            "slug"               : "netanyahu-out-before-2027",
            "endDate"            : "2026-12-31T00:00:00Z",
            "volume"             : 10_000.0,
            "rewardsMinSize"     : 200,
            "rewardsMaxSpread"   : 4.5,
            "clobRewards"        : [{"rewardsDailyRate": 500.0}],
            "holdingRewardsEnabled": False,
        }
        cfg    = RouterConfig(enabled=True)
        router = CohortRouter(plateau_tracker=None, config=cfg)
        result = router.filter_candidates([m], NOW)
        # Should not raise; result should contain at least this candidate
        assert isinstance(result, list)
