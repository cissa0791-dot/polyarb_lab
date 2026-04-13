"""
information_testing_research_line — registry
polyarb_lab / research_line / active

Schema and item registry for information items under test.
Items are paper/research-only. No execution path from this file.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


Category = Literal[
    "market_observation",
    "external_script_tool_idea",
    "trader_wallet_behavior_idea",
    "theoretical_academic_idea",
    "signal_hypothesis",
    "market_relation_hypothesis",
    "execution_hypothesis",
    "ranking_scoring_hypothesis",
]

FinalClassification = Literal[
    "keep",
    "downgrade",
    "reject",
    "park",
    "escalate",
    "pending",
]


@dataclass
class InformationItem:
    item_id: str                    # e.g. INFO-001
    source: str                     # where the idea came from
    category: Category
    raw_claim: str                  # original idea or information, unmodified
    codified_form: str              # rule / score / metric / experiment logic
    test_method: str                # how it will be evaluated (paper/research only)
    expected_value: str             # what improvement it might bring
    possible_failure_mode: str      # how it may fail or mislead
    evidence_standard: str          # what counts as supporting evidence
    result: Optional[str] = None    # what happened after testing (None = untested)
    final_classification: FinalClassification = "pending"
    notes: str = ""


# ---------------------------------------------------------------------------
# Item registry — initial population 2026-03-21
# ---------------------------------------------------------------------------

REGISTRY: list[InformationItem] = [

    InformationItem(
        item_id="INFO-001",
        source="polyrec (txbabaxyz/polyrec) — Phase 0 external toolchain inventory",
        category="external_script_tool_idea",
        raw_claim=(
            "Combining Binance real-time price feed + Chainlink oracle + Polymarket CLOB "
            "in a single dashboard may reveal timing or price-discovery dynamics that "
            "CLOB-only scanning misses. polyrec implements 70+ indicators across this combined feed."
        ),
        codified_form=(
            "Metric: cross-feed lag — time delta between Binance price move and CLOB midpoint update "
            "on the same underlying. "
            "Rule: if |binance_delta| > threshold AND clob_midpoint has not moved within lag_window, "
            "tag as 'update_lag' event. "
            "Indicator catalog: log which of 70+ polyrec indicators have no equivalent in "
            "src/intelligence/ and score each by potential information uplift."
        ),
        test_method=(
            "Run polyrec dash.py in streaming mode (isolated exploration dir). "
            "Collect 30 min of raw CSV output. "
            "Compare Binance price timestamps vs CLOB mid timestamps for same event. "
            "Measure update_lag event frequency and magnitude. "
            "Paper only — no order submission."
        ),
        expected_value=(
            "If lag > 1–2 seconds with measurable magnitude, cross-feed aggregation is a "
            "candidate for improving entry timing in external-reference strategy family. "
            "Indicator catalog reduces blind spots in src/intelligence/ modules."
        ),
        possible_failure_mode=(
            "BTC 15-min market scope is too narrow to generalize. "
            "Lag may be a data artifact of WebSocket subscription order, not a real market effect. "
            "If Chainlink feed lags Binance by design, oracle adds noise rather than signal."
        ),
        evidence_standard=(
            "Minimum: consistent update_lag > 1s measured in ≥10 events across a 30-min window. "
            "Indicator catalog: at least 3 indicators with no equivalent in baseline intelligence modules."
        ),
        result=None,
        final_classification="pending",
        notes="polyrec exact name confirmed. No secrets required. Python 3.10+ only dep.",
    ),

    InformationItem(
        item_id="INFO-002",
        source="collectmarkets2 proxy (pselamy/polymarket-insider-tracker) — Phase 0 inventory",
        category="trader_wallet_behavior_idea",
        raw_claim=(
            "Wallet-level clustering and pattern detection (early entry, abnormal sizing, "
            "correlated wallets) on Polymarket may act as a pre-filter for candidate market selection. "
            "Markets where 'informed' wallets are active may have higher signal quality than "
            "randomly selected markets."
        ),
        codified_form=(
            "Rule: classify wallets into tiers (uninformed / neutral / informed) based on: "
            "entry timing relative to market open, sizing relative to book depth, "
            "win rate over last N markets. "
            "Score: wallet_intelligence_score = w1*timing_rank + w2*size_rank + w3*win_rate. "
            "Market tag: if top-tier wallets are active in a market, tag market as wallet_informed=True. "
            "Candidate filter: prefer wallet_informed markets when multiple candidates score equally."
        ),
        test_method=(
            "Deploy collectmarkets2 Docker stack in isolated environment. "
            "Run against 5–10 known public Polymarket wallets with documented histories. "
            "Validate: does wallet tier assignment correlate with known outcomes? "
            "Compare market outcomes for wallet_informed=True vs False markets. "
            "Paper only — monitoring pipeline only."
        ),
        expected_value=(
            "If wallet_informed tag correlates with market resolution quality, "
            "candidate ranking gains an independent signal dimension. "
            "Reduces noise in candidate list by deprioritizing markets with only uninformed flow."
        ),
        possible_failure_mode=(
            "Insider tracker produces high false-positive rate. "
            "Docker/PostgreSQL overhead makes it impractical for local use. "
            "Wallet tiers may reflect risk tolerance rather than information. "
            "Exact repo identity (collectmarkets2 = insider-tracker) unconfirmed."
        ),
        evidence_standard=(
            "Minimum: wallet tier assignment on ≥20 wallets with ≥3 markets each. "
            "wallet_informed=True markets must resolve at higher rate (>55%) in correct direction. "
            "Docker stack must run stably for ≥1 hour without restart."
        ),
        result=None,
        final_classification="pending",
        notes="Gated on Docker availability. Identity unconfirmed. High potential but heavyweight.",
    ),

    InformationItem(
        item_id="INFO-003",
        source="scanner_v3 design — outcomePrices pre-filter (2026-03-20)",
        category="market_observation",
        raw_claim=(
            "Markets with p_yes in 0.05–0.95 have structurally different fill realism than tail "
            "markets (p_yes < 0.05 or > 0.95). "
            "The current pre-filter excludes tails to reduce CLOB calls, but the boundary values "
            "(0.05, 0.95) were chosen heuristically. "
            "The optimal boundary may differ by market category or liquidity tier."
        ),
        codified_form=(
            "Metric: fill_realism_score = (best_ask_depth / min_order_size) across markets "
            "grouped by p_yes decile. "
            "Experiment: for each decile bucket [0.00–0.10], [0.10–0.20], ... [0.90–1.00], "
            "measure mean ask_depth and HIT rate. "
            "Rule candidate: optimal_p_yes_band = decile range where "
            "mean_ask_depth > 2× floor AND hit_rate > 0."
        ),
        test_method=(
            "Run observation-only diagnostic: trial_entry_scan.py --mode events "
            "--events-limit 50 --min-edge 0.01. "
            "Log p_yes and ask_depth for all markets (pre and post filter). "
            "Group by decile. Compute fill_realism_score per bucket. "
            "Paper only — diagnostic mode only, never triggers escalation."
        ),
        expected_value=(
            "May tighten the pre-filter band (e.g. 0.08–0.92 better than 0.05–0.95) "
            "and reduce CLOB calls further. "
            "May reveal that some 0.05–0.10 markets have usable depth and are excluded unnecessarily."
        ),
        possible_failure_mode=(
            "Sample size too small per decile bucket across a single 50-event run. "
            "p_yes distribution shifts over time with market lifecycle. "
            "Optimizing on a single day's data may overfit to a snapshot."
        ),
        evidence_standard=(
            "Minimum: ≥30 markets per decile bucket across ≥3 independent diagnostic runs. "
            "Fill_realism_score must be measurably different across deciles (not flat). "
            "Proposed new band must not reduce hit rate vs current 0.05–0.95 band."
        ),
        result=None,
        final_classification="pending",
        notes=(
            "Uses existing scanner tooling. No new dependencies. "
            "Low risk — diagnostic mode only. Quick to test."
        ),
    ),

    InformationItem(
        item_id="INFO-004",
        source="scanner_v3 design — structural slug filter (2026-03-20)",
        category="market_observation",
        raw_claim=(
            "Structural derivative slugs (spread, handicap, total-pts, game-N) are excluded "
            "by the slug pre-filter because they are derivative families. "
            "The claim is that standalone markets have higher information density than structural "
            "derivatives — but this has not been validated empirically. "
            "If true, the structural slug filter improves candidate quality beyond just "
            "reducing CLOB call volume."
        ),
        codified_form=(
            "Metric: for each excluded slug pattern, measure: "
            "(1) ask_depth before exclusion, (2) p_yes range, (3) CAD if book was fetched. "
            "Rule: tag each excluded market as EXCLUDED_STRUCTURAL_SLUG with the matched pattern. "
            "Compare: mean CAD and ask_depth of slug-excluded markets vs slug-passed markets "
            "within the same event batch. "
            "Hypothesis score: if slug-excluded markets have lower mean CAD AND lower mean depth, "
            "the filter is confirmed as quality-improving not just volume-reducing."
        ),
        test_method=(
            "Modify diagnostic copy of scanner to log slug-excluded markets with their "
            "outcomePrices before exclusion. "
            "Run 3 diagnostic passes. Compare excluded vs included market distributions. "
            "Do not modify production scanner_v3."
        ),
        expected_value=(
            "Confirm that structural slug filter is a quality filter, not just volume reduction. "
            "May reveal slug patterns that should be added to or removed from the filter."
        ),
        possible_failure_mode=(
            "Some structural slug markets may have genuine CAD — excluding them loses edge. "
            "Slug pattern matching may be over-broad in some categories (e.g. 'game1' may be standalone)."
        ),
        evidence_standard=(
            "Minimum: ≥50 slug-excluded markets logged across ≥3 passes. "
            "Mean CAD of excluded < mean CAD of included, OR "
            "mean ask_depth of excluded < minimum viable depth threshold. "
            "At least one of these conditions must hold."
        ),
        result=None,
        final_classification="pending",
        notes="Scanner is FROZEN. Test must use diagnostic copy, never modify production scanner_v3.",
    ),

    InformationItem(
        item_id="INFO-005",
        source="mlmodelpoly proxy (NavnoorBawa/polymarket-prediction-system) — Phase 0 inventory",
        category="theoretical_academic_idea",
        raw_claim=(
            "An XGBoost + LightGBM ensemble trained on Polymarket market data may produce "
            "better-calibrated fair-value estimates than the analytic GBM first-touch model "
            "currently used in the BTC external-reference branch. "
            "ML approach may capture non-Gaussian tail behavior in prediction markets."
        ),
        codified_form=(
            "Experiment: run mlmodelpoly main.py on 5–10 markets. "
            "For each market, record: ml_signal (Yes/No), confidence_score, kelly_size. "
            "Compare to baseline: run same markets through GBM first-touch model. "
            "Calibration metric: |ml_implied_prob - market_mid| vs |gbm_implied_prob - market_mid|. "
            "Tag: if ml_implied_prob consistently closer to eventual resolution, "
            "ensemble approach has calibration lift."
        ),
        test_method=(
            "Run mlmodelpoly as batch research tool (python main.py 10). "
            "Log output signals to results/INFO-005_ml_signals.json. "
            "Backfill resolved markets to check signal direction vs resolution. "
            "Paper only — signals never connected to execution layer."
        ),
        expected_value=(
            "If ensemble calibration measurably better than GBM in non-extreme regimes, "
            "it is a candidate to complement or replace the analytic model in external-ref branch "
            "when BTC market inventory returns."
        ),
        possible_failure_mode=(
            "Training data source unknown — model may be trained on a stale or thin dataset. "
            "Kelly outputs could be misread as trade instructions. "
            "Exact repo identity (mlmodelpoly = NavnoorBawa repo) unconfirmed. "
            "Feature engineering may duplicate what baseline src/beliefs/ already implements."
        ),
        evidence_standard=(
            "Minimum: ≥20 markets evaluated. "
            "Calibration gap (|ml_implied - resolution| < |gbm_implied - resolution|) "
            "in ≥60% of resolved markets. "
            "Must be confirmed on resolved markets only — pending markets excluded from evaluation."
        ),
        result=None,
        final_classification="pending",
        notes=(
            "Repo identity unconfirmed. Zero execution risk. "
            "Cannot promote to external-ref branch until BTC inventory returns."
        ),
    ),

    InformationItem(
        item_id="INFO-006",
        source="system family 1 (reward-aware maker) — family classification 2026-03-20",
        category="signal_hypothesis",
        raw_claim=(
            "Maker reward rate per filled order can be modeled as a dynamic minimum edge floor. "
            "Markets where post-fill reward rate is below operating cost should be excluded. "
            "Currently the reward rate is treated as binary (program active / inactive) rather "
            "than as a continuous per-market signal."
        ),
        codified_form=(
            "Metric: reward_floor_per_market = (expected_reward_rate × fill_probability) - fee_cost. "
            "Rule: exclude candidate if reward_floor_per_market < 0. "
            "Score: rank candidates by reward_floor_per_market as a secondary sort key "
            "after primary edge criterion. "
            "Dynamic: recalculate each monitoring round using current epoch reward rate."
        ),
        test_method=(
            "Implement reward_floor metric in paper mode as a scoring annotation only. "
            "Run 5 paper monitoring rounds. "
            "Log reward_floor_per_market for each candidate. "
            "Observe: do high reward_floor candidates resolve more favorably than low ones? "
            "Paper only — no live execution."
        ),
        expected_value=(
            "If reward_floor is a meaningful discriminator, candidate ranking improves. "
            "Reduces wasted attention on markets where structural edge is reward-dependent "
            "but reward rate is too thin."
        ),
        possible_failure_mode=(
            "Reward rate from /rewards/epoch currently returns 405 (REWARDS_API_UNVERIFIED). "
            "Without a reliable reward rate signal, the metric cannot be computed accurately. "
            "May create false confidence in markets that appear positive only due to reward assumptions."
        ),
        evidence_standard=(
            "Blocked until REWARDS_API_UNVERIFIED is resolved. "
            "Once API is available: ≥20 candidates scored across ≥5 rounds. "
            "High reward_floor quartile must show better candidate quality than low quartile."
        ),
        result=None,
        final_classification="park",
        notes=(
            "PARKED: blocked by REWARDS_API_UNVERIFIED (405 on /rewards/epoch). "
            "Resume when rewards API is confirmed."
        ),
    ),

    InformationItem(
        item_id="INFO-007",
        source="polyterminal whale tracker — Phase 0 inventory",
        category="trader_wallet_behavior_idea",
        raw_claim=(
            "Large single-order imbalances visible in CLOB depth asymmetry (ask >> bid or bid >> ask "
            "in dollar terms) may predict short-term midpoint direction. "
            "polyterminal's whale tracker screens surface these imbalances. "
            "If the signal is real, CLOB depth asymmetry is a ranking input for entry timing."
        ),
        codified_form=(
            "Metric: depth_asymmetry_ratio = ask_depth_1ct / bid_depth_1ct. "
            "Rule: tag market as whale_bid if ratio < 0.5 (ask thin, bid heavy); "
            "whale_ask if ratio > 2.0 (bid thin, ask heavy). "
            "Hypothesis: whale_bid markets trend toward YES; whale_ask trend toward NO "
            "within 1–5 minute window. "
            "Test: compare midpoint direction 5 min after whale tag vs direction of 0 fill."
        ),
        test_method=(
            "Extend CLOB ingest (src/ingest/clob.py) to log depth_asymmetry_ratio per market pull. "
            "Tag whale events in paper mode. "
            "Track midpoint 5 min post-tag. "
            "Compare directional accuracy of whale_bid/ask tags vs baseline (random). "
            "Paper only — no execution."
        ),
        expected_value=(
            "If depth asymmetry predicts direction at >55% accuracy, "
            "it improves entry timing in maker and single-market strategies."
        ),
        possible_failure_mode=(
            "Depth asymmetry is a lagging signal — large orders may already reflect "
            "a move that has happened. "
            "Thin markets have high depth_asymmetry_ratio by default (not signal). "
            "Whale behavior on small Polymarket books differs from traditional market microstructure."
        ),
        evidence_standard=(
            "Minimum: ≥50 whale events tagged. "
            "Directional accuracy > 55% measured on resolved midpoint direction at 5 min. "
            "Must exclude markets with < $10 depth (structural thin book, not whale signal)."
        ),
        result=None,
        final_classification="pending",
        notes="Requires CLOB ingest extension. Low-cost annotation. No new deps.",
    ),

    InformationItem(
        item_id="INFO-008",
        source="system family 2 (cross-market logical-constraint) — family classification",
        category="market_relation_hypothesis",
        raw_claim=(
            "In political and sports constraint sets (A + B = 1.0), one leg type "
            "(e.g. favorite vs underdog, or Democrat vs Republican) may persistently misprice "
            "in a consistent direction across multiple constraint sets. "
            "If systematic, this is a structural bias that the constraint scanner can exploit "
            "more reliably than searching for random gaps."
        ),
        codified_form=(
            "Metric: per-leg mispricing direction — for each constraint set where CAD > 0, "
            "record which leg is overpriced (leg_A or leg_B) and tag by category "
            "(favorite/underdog, incumbent/challenger, home/away). "
            "Aggregate: across N sets, compute directional_bias_rate = "
            "count(leg_A_overpriced) / count(all_biased_sets). "
            "Threshold: if directional_bias_rate > 0.65 for a category, tag as systematic_bias."
        ),
        test_method=(
            "Run constraint scanner (discover_cross_market_constraints.py) across "
            "political and NBA constraint families. "
            "Log leg-level mispricing direction for each set with |CAD| > 0.02. "
            "Accumulate over ≥5 scan sessions. "
            "Compute directional_bias_rate per category. "
            "Paper only — observation only."
        ),
        expected_value=(
            "If systematic bias confirmed, constraint scanner can prioritize the consistently "
            "overpriced leg as the sell side, improving candidate construction quality. "
            "Reduces scan noise from symmetric random gaps."
        ),
        possible_failure_mode=(
            "Polymarket bettors may correct systematic bias quickly once observed. "
            "Sample size per constraint set category may be too small to measure reliably. "
            "Constraint sets may not have enough sets per category for statistical confidence."
        ),
        evidence_standard=(
            "Minimum: ≥15 constraint sets with |CAD| > 0.02 per category. "
            "Directional_bias_rate > 0.65 in ≥1 category. "
            "Must hold across ≥3 independent scan sessions (not a single snapshot)."
        ),
        result=None,
        final_classification="pending",
        notes="Uses existing constraint scanner. Observation-only extension. No new code needed.",
    ),

    InformationItem(
        item_id="INFO-009",
        source="4coinsbot proxy (dev-protocol/polymarket-arbitrage-bot) — Phase 0 inventory",
        category="execution_hypothesis",
        raw_claim=(
            "In 15-min binary prediction markets (BTC/ETH/SOL/XRP UP/DOWN), "
            "a sudden price spike away from fair value (dump detection) followed by a "
            "counter-position (hedge timing) may be a learnable and repeatable signal pattern. "
            "4coinsbot implements this as a TypeScript dump-and-hedge strategy."
        ),
        codified_form=(
            "Signal: dump_signal = (current_mid - rolling_mid_5m) / rolling_std_5m. "
            "If dump_signal > 2.0: price has spiked up anomalously; counter = YES sell. "
            "If dump_signal < -2.0: price has dropped anomalously; counter = NO sell. "
            "Timing rule: enter counter position within 30s of dump_signal trigger. "
            "Exit: at market close or when mid reverts to rolling_mid ± 0.5 std."
        ),
        test_method=(
            "Review 4coinsbot src/dumpHedgeTrader.ts logic in read-only mode. "
            "Translate signal logic to Python pseudocode. "
            "Backtest on polyrec CSV output (if available) or Gamma historical data. "
            "Paper only — never set PRODUCTION=true, never provide PRIVATE_KEY."
        ),
        expected_value=(
            "If dump-hedge signal backtests positively on 15-min binary markets, "
            "it provides an executable sample for a new strategy family "
            "not present in current baseline."
        ),
        possible_failure_mode=(
            "15-min binary markets have very thin books and high spread — "
            "signal may not survive transaction costs. "
            "Dump events may be informed flow rather than noise. "
            "TypeScript barrier limits direct reuse; translation may introduce errors. "
            "Exact repo identity (4coinsbot = dev-protocol repo) unconfirmed."
        ),
        evidence_standard=(
            "Minimum: signal logic fully extracted from TypeScript source. "
            "Backtest on ≥50 historical market periods. "
            "Edge after fees > 0 in ≥60% of periods. "
            "Must not rely on thin-book periods where fills are unrealistic."
        ),
        result=None,
        final_classification="pending",
        notes=(
            "Repo identity unconfirmed. TypeScript read-only. "
            "Gated on polyrec CSV data or alternative historical data source."
        ),
    ),

    InformationItem(
        item_id="INFO-010",
        source="BTC external-reference branch diagnostic — 2026-03-20",
        category="signal_hypothesis",
        raw_claim=(
            "The GBM first-touch model produces structurally small absolute CAD "
            "at extreme reference distances (>100% away from threshold) regardless of ratio, "
            "because probability clips to ~2% floor. "
            "A floor-adjusted CAD — normalizing raw CAD by the model's probability floor — "
            "may be a better gate than raw absolute CAD at extreme distances."
        ),
        codified_form=(
            "Current gate: |CAD| >= threshold (e.g. 0.03). "
            "Proposed floor-adjusted metric: "
            "adjusted_CAD = CAD / max(model_prob, floor_prob_at_distance). "
            "Rule: use adjusted_CAD as primary gate when distance_pct > 80%. "
            "At moderate distances (< 80%), keep raw CAD gate unchanged. "
            "This preserves strict threshold at moderate distances while "
            "allowing model to compete at extreme distances."
        ),
        test_method=(
            "Run BTC external-ref explain diagnostic on December 31 2026 market. "
            "Log: raw CAD, model_prob, distance_pct, floor_prob_at_distance. "
            "Compute adjusted_CAD. "
            "Compare: does adjusted_CAD > threshold while raw CAD < threshold? "
            "Paper only — research diagnostic only."
        ),
        expected_value=(
            "If floor-adjusted CAD captures genuine edge at extreme distances "
            "that raw CAD misses, the external-ref branch may produce more candidates "
            "when BTC inventory is present. "
            "Reduces the 'always zero candidates at extreme distance' structural artifact."
        ),
        possible_failure_mode=(
            "Adjusted metric may inflate apparent edge at extreme distances "
            "where the market is genuinely offering no real edge. "
            "The model floor may be accurate — adjusting it away may introduce false positives. "
            "BTC inventory must return before this can be tested in live conditions."
        ),
        evidence_standard=(
            "Minimum: adjusted_CAD computed on ≥5 BTC markets at >80% distance. "
            "At least one market where adjusted_CAD > threshold AND raw CAD < threshold. "
            "Must be confirmed on eventually-resolved markets to assess direction accuracy."
        ),
        result=None,
        final_classification="pending",
        notes=(
            "Blocked until BTC market inventory returns (new short/medium horizon listings). "
            "December 31 2026 market is the best current audit target."
        ),
    ),
]


def get_by_id(item_id: str) -> InformationItem:
    for item in REGISTRY:
        if item.item_id == item_id:
            return item
    raise KeyError(f"No item with id {item_id!r}")


def get_by_category(category: Category) -> list[InformationItem]:
    return [item for item in REGISTRY if item.category == category]


def get_by_classification(classification: FinalClassification) -> list[InformationItem]:
    return [item for item in REGISTRY if item.final_classification == classification]
