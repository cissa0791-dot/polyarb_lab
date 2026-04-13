from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.config_runtime.models import OpportunityConfig
from src.domain.models import RejectionReason
from src.opportunity.models import CandidateLeg, ExecutableCandidate, QualificationDecision, QualificationPassReason, RankedOpportunity, RawCandidate


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class FillEstimate:
    side: str
    required_shares: float
    filled_shares: float
    best_price: float | None
    vwap_price: float | None
    notional_usd: float
    available_shares: float
    available_notional_usd: float
    spread_cents: float | None

    @property
    def is_complete(self) -> bool:
        return self.filled_shares + 1e-9 >= self.required_shares


class VWAPCalculator:
    def estimate_buy(self, levels: list, shares: float, spread_cents: float | None = None) -> FillEstimate:
        return self._estimate(levels, shares, side="BUY", spread_cents=spread_cents)

    def estimate_sell(self, levels: list, shares: float, spread_cents: float | None = None) -> FillEstimate:
        return self._estimate(levels, shares, side="SELL", spread_cents=spread_cents)

    def _estimate(self, levels: list, shares: float, side: str, spread_cents: float | None) -> FillEstimate:
        required = max(0.0, float(shares))
        remaining = required
        filled = 0.0
        notional = 0.0
        available_shares = 0.0
        available_notional = 0.0
        best_price: float | None = None

        for level in levels:
            price = float(level.price)
            size = float(level.size)
            if price <= 0 or size <= 0:
                continue
            if best_price is None:
                best_price = price
            available_shares += size
            available_notional += size * price
            if remaining <= 1e-9:
                continue
            take = min(remaining, size)
            filled += take
            notional += take * price
            remaining -= take

        vwap = (notional / filled) if filled > 1e-9 else None
        return FillEstimate(
            side=side,
            required_shares=required,
            filled_shares=filled,
            best_price=best_price,
            vwap_price=vwap,
            notional_usd=notional,
            available_shares=available_shares,
            available_notional_usd=available_notional,
            spread_cents=spread_cents,
        )


class DepthAnalyzer:
    def __init__(self, calculator: VWAPCalculator):
        self.calculator = calculator

    def check_absolute_depth(
        self,
        legs: list[tuple[CandidateLeg, object]],
        min_depth_usd: float,
    ) -> tuple[bool, list[str]]:
        """Return (all_pass, failing_token_ids).

        Each leg must have at least *min_depth_usd* of dollar notional available
        on the correct side of the book (asks for BUY, bids for SELL).  A leg
        that fails contributes its token_id to the second element of the tuple.
        """
        if min_depth_usd <= 0.0:
            return True, []
        failing: list[str] = []
        for leg, book in legs:
            levels = self._levels_for_action(leg.action, book)
            available_notional = sum(
                float(level.price) * float(level.size)
                for level in levels
                if float(getattr(level, "price", 0)) > 0 and float(getattr(level, "size", 0)) > 0
            )
            if available_notional < min_depth_usd:
                failing.append(leg.token_id)
        return len(failing) == 0, failing

    def analyze(self, legs: list[tuple[CandidateLeg, object]]) -> dict[str, float]:
        if not legs:
            return {
                "required_shares": 0.0,
                "available_shares": 0.0,
                "required_depth_usd": 0.0,
                "available_depth_usd": 0.0,
            }

        required_shares = min(max(0.0, leg.required_shares) for leg, _book in legs)
        available_shares = min(self._summarize_levels(leg.action, book).available_shares for leg, book in legs)
        required_depth_usd = 0.0
        available_depth_usd = 0.0

        for leg, book in legs:
            spread = self._spread(book)
            fill_required = self._estimate(leg.action, self._levels_for_action(leg.action, book), required_shares, spread)
            required_depth_usd += fill_required.notional_usd
            if available_shares > 1e-9:
                fill_available = self._estimate(leg.action, self._levels_for_action(leg.action, book), available_shares, spread)
                available_depth_usd += fill_available.notional_usd

        return {
            "required_shares": round(required_shares, 6),
            "available_shares": round(available_shares, 6),
            "required_depth_usd": round(required_depth_usd, 6),
            "available_depth_usd": round(available_depth_usd, 6),
        }

    def _summarize_levels(self, action: str, book) -> FillEstimate:
        spread = self._spread(book)
        levels = self._levels_for_action(action, book)
        return self._estimate(action, levels, 0.0, spread)

    def _estimate(self, action: str, levels: list, shares: float, spread_cents: float | None) -> FillEstimate:
        if action.upper() == "BUY":
            return self.calculator.estimate_buy(levels, shares, spread_cents=spread_cents)
        return self.calculator.estimate_sell(levels, shares, spread_cents=spread_cents)

    def _levels_for_action(self, action: str, book) -> list:
        return getattr(book, "asks" if action.upper() == "BUY" else "bids", [])

    def _spread(self, book) -> float | None:
        bids = getattr(book, "bids", [])
        asks = getattr(book, "asks", [])
        if not bids or not asks:
            return None
        return max(0.0, float(asks[0].price) - float(bids[0].price))


class PartialFillRiskEstimator:
    def __init__(self, min_depth_multiple: float):
        self.min_depth_multiple = max(1.0, float(min_depth_multiple))

    def score(self, required_shares: float, available_shares: float) -> float:
        if required_shares <= 1e-9:
            return 1.0
        coverage = available_shares / required_shares
        if coverage >= self.min_depth_multiple:
            return 0.0
        normalized = (coverage - 1.0) / max(self.min_depth_multiple - 1.0, 1.0)
        return round(1.0 - _clamp(normalized), 6)


class AtomicityRiskScorer:
    def score(self, fill_estimates: list[FillEstimate], partial_fill_risk: float) -> float:
        if not fill_estimates:
            return 1.0

        best_prices = [estimate.best_price for estimate in fill_estimates if estimate.best_price]
        vwaps = [estimate.vwap_price for estimate in fill_estimates if estimate.vwap_price]
        if not best_prices or not vwaps:
            return 1.0

        slippage_ratios = [
            max(0.0, (float(estimate.vwap_price or 0.0) - float(estimate.best_price or 0.0)) / max(float(estimate.best_price or 0.0), 1e-9))
            for estimate in fill_estimates
        ]
        availability = [estimate.available_shares for estimate in fill_estimates]
        imbalance = (max(availability) - min(availability)) / max(max(availability), 1e-9)
        slippage_risk = _clamp(max(slippage_ratios) * 10.0)
        score = (0.55 * partial_fill_risk) + (0.25 * imbalance) + (0.20 * slippage_risk)
        return round(_clamp(score), 6)


class ExecutionFeasibilityEvaluator:
    def __init__(self, config: OpportunityConfig):
        self.config = config
        self.calculator = VWAPCalculator()
        self.depth_analyzer = DepthAnalyzer(self.calculator)
        self.partial_fill_risk = PartialFillRiskEstimator(config.min_depth_multiple)
        self.atomicity_risk = AtomicityRiskScorer()

    def qualify(self, raw_candidate: RawCandidate, books_by_token: dict[str, object]) -> QualificationDecision:
        reasons: list[str] = []
        leg_fills: list[FillEstimate] = []
        qualified_legs: list[CandidateLeg] = []
        evaluation_ts = datetime.now(timezone.utc)

        paired_legs: list[tuple[CandidateLeg, object]] = []
        for leg in raw_candidate.legs:
            book = books_by_token.get(leg.token_id)
            if book is None:
                reasons.append(RejectionReason.MISSING_ORDERBOOK.value)
                continue
            paired_legs.append((leg, book))

            spread = self.depth_analyzer._spread(book)
            if spread is not None and spread > self.config.max_spread_cents:
                reasons.append(RejectionReason.SPREAD_TOO_WIDE.value)

            is_maker = (leg.metadata or {}).get("maker_first", False)
            if is_maker:
                levels = getattr(book, "bids", [])
                fill = self.calculator.estimate_sell(levels, leg.required_shares, spread)
            else:
                levels = getattr(book, "asks" if leg.action.upper() == "BUY" else "bids", [])
                fill = self.calculator.estimate_buy(levels, leg.required_shares, spread) if leg.action.upper() == "BUY" else self.calculator.estimate_sell(levels, leg.required_shares, spread)
            leg_fills.append(fill)
            qualified_legs.append(
                leg.model_copy(
                    update={
                        "best_price": fill.best_price,
                        "vwap_price": fill.vwap_price,
                        "spread_cents": spread,
                        "available_shares": round(fill.available_shares, 6),
                        "available_notional_usd": round(fill.available_notional_usd, 6),
                    }
                )
            )
            if not fill.is_complete:
                reasons.append(RejectionReason.INSUFFICIENT_DEPTH.value)

        # Absolute depth floor — checked once all legs have been paired.
        # Must run before VWAP/edge arithmetic so that thin-book candidates
        # are rejected with an explicit reason rather than producing misleading
        # edge numbers computed against an effectively empty book.
        if self.config.min_absolute_leg_depth_usd > 0.0:
            abs_ok, failing_tokens = self.depth_analyzer.check_absolute_depth(
                paired_legs, self.config.min_absolute_leg_depth_usd
            )
            if not abs_ok:
                reasons.append(RejectionReason.ABSOLUTE_DEPTH_BELOW_FLOOR.value)

        # Single-leg concentration ceiling — rejects baskets where one outcome leg
        # dominates notional allocation.  Must run after all leg fills are computed
        # so that best_price reflects the live book top-of-book bid.
        if self.config.max_single_leg_bid < 1.0 and leg_fills:
            max_leg_bid = max(fill.best_price or 0.0 for fill in leg_fills)
            if max_leg_bid > self.config.max_single_leg_bid:
                reasons.append(RejectionReason.SINGLE_LEG_CONCENTRATION.value)

        if len(leg_fills) != len(raw_candidate.legs):
            decision = QualificationDecision(
                raw_candidate=raw_candidate,
                passed=False,
                reason_codes=sorted(set(reasons or [RejectionReason.MISSING_ORDERBOOK.value])),
                metadata={"status": "missing_books"},
                ts=evaluation_ts,
            )
            return decision

        depth_metrics = self.depth_analyzer.analyze(paired_legs)
        pair_vwap = sum(float(fill.vwap_price or 0.0) for fill in leg_fills)
        payout_per_share = raw_candidate.expected_payout / raw_candidate.target_shares if raw_candidate.target_shares > 1e-9 else 0.0
        expected_gross_edge_cents = round(max(0.0, payout_per_share - pair_vwap), 6)
        expected_fee_usd = round(raw_candidate.target_shares * self.config.fee_buffer_cents, 6)
        expected_slippage_usd = round(raw_candidate.target_shares * self.config.slippage_buffer_cents, 6)
        expected_gross_profit = round(raw_candidate.target_shares * expected_gross_edge_cents, 6)
        expected_net_profit = round(expected_gross_profit - expected_fee_usd - expected_slippage_usd, 6)
        partial_fill_risk = self.partial_fill_risk.score(depth_metrics["required_shares"], depth_metrics["available_shares"])
        non_atomic_risk = self.atomicity_risk.score(leg_fills, partial_fill_risk)

        if expected_gross_edge_cents - self.config.fee_buffer_cents - self.config.slippage_buffer_cents < self.config.min_edge_cents:
            reasons.append(RejectionReason.EDGE_BELOW_THRESHOLD.value)
        if expected_net_profit < self.config.min_net_profit_usd:
            reasons.append(RejectionReason.NET_PROFIT_BELOW_THRESHOLD.value)
        if depth_metrics["available_depth_usd"] + 1e-9 < depth_metrics["required_depth_usd"]:
            reasons.append(RejectionReason.INSUFFICIENT_DEPTH.value)
        if partial_fill_risk > self.config.max_partial_fill_risk:
            reasons.append(RejectionReason.PARTIAL_FILL_RISK_TOO_HIGH.value)
        if non_atomic_risk > self.config.max_non_atomic_risk:
            reasons.append(RejectionReason.NON_ATOMIC_RISK_TOO_HIGH.value)

        metadata = {
            "pair_vwap": round(pair_vwap, 6),
            "expected_gross_edge_cents": expected_gross_edge_cents,
            "expected_net_edge_cents": round(expected_gross_edge_cents - self.config.fee_buffer_cents - self.config.slippage_buffer_cents, 6),
            "expected_gross_profit_usd": expected_gross_profit,
            "expected_net_profit_usd": expected_net_profit,
            "required_depth_usd": depth_metrics["required_depth_usd"],
            "available_depth_usd": depth_metrics["available_depth_usd"],
            "required_shares": depth_metrics["required_shares"],
            "available_shares": depth_metrics["available_shares"],
            "partial_fill_risk_score": partial_fill_risk,
            "non_atomic_execution_risk_score": non_atomic_risk,
            "legs": [leg.model_dump(mode="json") for leg in qualified_legs],
        }

        if reasons:
            return QualificationDecision(
                raw_candidate=raw_candidate,
                passed=False,
                reason_codes=sorted(set(reasons)),
                metadata=metadata,
                ts=evaluation_ts,
            )

        # Build explicit pass-gate codes so auditors can see *which* gates
        # this candidate cleared, not just that it passed overall.
        net_edge_cents = round(
            expected_gross_edge_cents - self.config.fee_buffer_cents - self.config.slippage_buffer_cents, 6
        )
        pass_reasons: list[str] = []
        if net_edge_cents >= self.config.min_edge_cents:
            pass_reasons.append(QualificationPassReason.EDGE_SUFFICIENT)
        if expected_net_profit >= self.config.min_net_profit_usd:
            pass_reasons.append(QualificationPassReason.NET_PROFIT_SUFFICIENT)
        if depth_metrics["available_depth_usd"] + 1e-9 >= depth_metrics["required_depth_usd"]:
            pass_reasons.append(QualificationPassReason.DEPTH_SUFFICIENT)
        if partial_fill_risk <= self.config.max_partial_fill_risk:
            pass_reasons.append(QualificationPassReason.PARTIAL_FILL_RISK_OK)
        if non_atomic_risk <= self.config.max_non_atomic_risk:
            pass_reasons.append(QualificationPassReason.NON_ATOMIC_RISK_OK)
        if self.config.min_absolute_leg_depth_usd > 0.0:
            pass_reasons.append(QualificationPassReason.ABSOLUTE_DEPTH_OK)
        if self.config.max_single_leg_bid < 1.0:
            pass_reasons.append(QualificationPassReason.SINGLE_LEG_CONCENTRATION_OK)

        executable = ExecutableCandidate(
            strategy_id=raw_candidate.strategy_id,
            strategy_family=raw_candidate.strategy_family,
            execution_mode=raw_candidate.execution_mode,
            research_only=raw_candidate.research_only,
            candidate_id=raw_candidate.candidate_id,
            kind=raw_candidate.kind,
            market_slugs=raw_candidate.market_slugs,
            gross_edge_cents=expected_gross_edge_cents,
            fee_estimate_cents=self.config.fee_buffer_cents,
            slippage_estimate_cents=self.config.slippage_buffer_cents,
            expected_payout=raw_candidate.expected_payout,
            target_notional_usd=raw_candidate.target_notional_usd,
            estimated_depth_usd=depth_metrics["available_depth_usd"],
            score=0.0,
            repeatable=False,
            book_age_ms=0,
            estimated_net_profit_usd=expected_net_profit,
            metadata={**raw_candidate.metadata, "detection_name": raw_candidate.detection_name},
            ts=raw_candidate.ts,
            qualification_reason_codes=pass_reasons,
            qualification_metadata=metadata,
            expected_gross_profit_usd=expected_gross_profit,
            expected_fee_usd=expected_fee_usd,
            expected_slippage_usd=expected_slippage_usd,
            pair_vwap=round(pair_vwap, 6),
            required_depth_usd=depth_metrics["required_depth_usd"],
            available_depth_usd=depth_metrics["available_depth_usd"],
            required_shares=depth_metrics["required_shares"],
            available_shares=depth_metrics["available_shares"],
            partial_fill_risk_score=partial_fill_risk,
            non_atomic_execution_risk_score=non_atomic_risk,
            legs=qualified_legs,
        )
        return QualificationDecision(
            raw_candidate=raw_candidate,
            passed=True,
            pass_reason_codes=pass_reasons,
            executable_candidate=executable,
            metadata=metadata,
            ts=evaluation_ts,
        )


class OpportunityRanker:
    def __init__(self, config: OpportunityConfig):
        self.config = config

    def rank(self, candidate: ExecutableCandidate) -> RankedOpportunity:
        net_edge_component = _clamp(candidate.net_edge_cents / max(self.config.min_edge_cents * 2.0, 1e-6))
        profit_component = _clamp(candidate.estimated_net_profit_usd / max(self.config.min_net_profit_usd * 2.0, 1e-6))
        depth_component = _clamp(candidate.available_depth_usd / max(candidate.required_depth_usd, 1e-6))
        risk_component = 1.0 - max(candidate.partial_fill_risk_score, candidate.non_atomic_execution_risk_score)
        quality_score = round(
            100.0
            * ((0.35 * net_edge_component) + (0.25 * profit_component) + (0.20 * depth_component) + (0.20 * risk_component)),
            6,
        )
        capital_efficiency = round(candidate.estimated_net_profit_usd / max(candidate.target_notional_usd, 1e-6), 6)

        payload = candidate.model_dump(mode="python")
        payload.update(
            {
                "score": quality_score,
                "ranking_score": quality_score,
                "strategy_tag": f"{candidate.strategy_family.value}:{candidate.strategy_id}",
                "sizing_hint_usd": candidate.target_notional_usd,
                "sizing_hint_shares": candidate.required_shares,
                "capital_efficiency": capital_efficiency,
                "expected_profit_usd": candidate.estimated_net_profit_usd,
                "quality_score": quality_score,
            }
        )
        return RankedOpportunity.model_validate(payload)
