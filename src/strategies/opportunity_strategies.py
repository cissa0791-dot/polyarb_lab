from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.core.models import MarketPair
from src.opportunity.models import CandidateLeg, RawCandidate, StrategyFamily
from src.scanner.cross_market import (
    analyze_leq_constraint,
    analyze_leq_constraint_execution_gross,
    analyze_leq_constraint_gross,
)
from src.scanner.maker_rewarded_mm import analyze_maker_rewarded_market
from src.scanner.neg_risk import analyze_neg_risk_rebalancing_event
from src.scanner.political_binary import (
    analyze_political_implication_pair,
    analyze_political_mutex_pair,
)
from src.scanner.single_market import analyze_yes_no_pair, analyze_yes_no_touch_pair
from src.strategies.base import BaseOpportunityStrategy


class SingleMarketMispricingStrategy(BaseOpportunityStrategy):
    strategy_id = "single_market_sum_under_1"
    strategy_family = StrategyFamily.SINGLE_MARKET_MISPRICING

    def detect_with_audit(self, pair: MarketPair, yes_book, no_book, max_notional: float, total_buffer_cents: float) -> tuple[RawCandidate | None, dict | None]:
        analysis = analyze_yes_no_pair(
            pair,
            yes_book,
            no_book,
            max_notional=max_notional,
            total_buffer_cents=total_buffer_cents,
        )
        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
            }

        shares = float(opportunity.details.get("shares", 0.0))
        yes_best_ask = float(yes_book.asks[0].price) if getattr(yes_book, "asks", []) else None
        no_best_ask = float(no_book.asks[0].price) if getattr(no_book, "asks", []) else None

        return RawCandidate(
            strategy_id=self.strategy_id,
            strategy_family=self.strategy_family,
            candidate_id=str(uuid4()),
            kind=opportunity.kind,
            detection_name=opportunity.name,
            market_slugs=[pair.market_slug],
            gross_edge_cents=opportunity.edge_cents,
            expected_payout=opportunity.est_payout,
            target_notional_usd=opportunity.notional,
            target_shares=shares,
            gross_profit_usd=opportunity.gross_profit,
            est_fill_cost_usd=opportunity.est_fill_cost,
            execution_mode="paper_eligible",
            research_only=False,
            legs=[
                CandidateLeg(
                    token_id=pair.yes_token_id,
                    market_slug=pair.market_slug,
                    action="BUY",
                    side="YES",
                    required_shares=shares,
                    best_price=yes_best_ask,
                    vwap_price=float(opportunity.details.get("yes_vwap", 0.0)),
                ),
                CandidateLeg(
                    token_id=pair.no_token_id,
                    market_slug=pair.market_slug,
                    action="BUY",
                    side="NO",
                    required_shares=shares,
                    best_price=no_best_ask,
                    vwap_price=float(opportunity.details.get("no_vwap", 0.0)),
                ),
            ],
            metadata={
                **opportunity.details,
                "question": pair.question,
                "strategy_family": self.strategy_family.value,
            },
            ts=opportunity.ts,
        ), None

    def detect(self, pair: MarketPair, yes_book, no_book, max_notional: float, total_buffer_cents: float) -> RawCandidate | None:
        raw_candidate, _ = self.detect_with_audit(
            pair,
            yes_book,
            no_book,
            max_notional=max_notional,
            total_buffer_cents=total_buffer_cents,
        )
        return raw_candidate


class SingleMarketTouchMispricingStrategy(BaseOpportunityStrategy):
    strategy_id = "single_market_touch_under_1"
    strategy_family = StrategyFamily.SINGLE_MARKET_TOUCH_MISPRICING

    def detect_with_audit(self, pair: MarketPair, yes_book, no_book, max_notional: float, total_buffer_cents: float) -> tuple[RawCandidate | None, dict | None]:
        analysis = analyze_yes_no_touch_pair(
            pair,
            yes_book,
            no_book,
            max_notional=max_notional,
            total_buffer_cents=total_buffer_cents,
        )
        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
            }

        shares = float(opportunity.details.get("shares", 0.0))
        yes_best_ask = float(yes_book.asks[0].price) if getattr(yes_book, "asks", []) else None
        no_best_ask = float(no_book.asks[0].price) if getattr(no_book, "asks", []) else None

        return RawCandidate(
            strategy_id=self.strategy_id,
            strategy_family=self.strategy_family,
            candidate_id=str(uuid4()),
            kind=opportunity.kind,
            detection_name=opportunity.name,
            market_slugs=[pair.market_slug],
            gross_edge_cents=opportunity.edge_cents,
            expected_payout=opportunity.est_payout,
            target_notional_usd=opportunity.notional,
            target_shares=shares,
            gross_profit_usd=opportunity.gross_profit,
            est_fill_cost_usd=opportunity.est_fill_cost,
            execution_mode="paper_eligible",
            research_only=False,
            legs=[
                CandidateLeg(
                    token_id=pair.yes_token_id,
                    market_slug=pair.market_slug,
                    action="BUY",
                    side="YES",
                    required_shares=shares,
                    best_price=yes_best_ask,
                    vwap_price=yes_best_ask,
                ),
                CandidateLeg(
                    token_id=pair.no_token_id,
                    market_slug=pair.market_slug,
                    action="BUY",
                    side="NO",
                    required_shares=shares,
                    best_price=no_best_ask,
                    vwap_price=no_best_ask,
                ),
            ],
            metadata={
                **opportunity.details,
                "question": pair.question,
                "strategy_family": self.strategy_family.value,
            },
            ts=opportunity.ts,
        ), None

    def detect(self, pair: MarketPair, yes_book, no_book, max_notional: float, total_buffer_cents: float) -> RawCandidate | None:
        raw_candidate, _ = self.detect_with_audit(
            pair,
            yes_book,
            no_book,
            max_notional=max_notional,
            total_buffer_cents=total_buffer_cents,
        )
        return raw_candidate


class CrossMarketConstraintStrategy(BaseOpportunityStrategy):
    strategy_id = "cross_market_leq"
    strategy_family = StrategyFamily.CROSS_MARKET_CONSTRAINT

    def _analyze_relation(
        self,
        rule: dict,
        lhs_relation_book,
        rhs_relation_book,
        total_buffer_cents: float,
    ) -> dict[str, object]:
        return analyze_leq_constraint(
            rule["name"],
            lhs_relation_book,
            rhs_relation_book,
            rule["lhs"]["market_slug"],
            rule["rhs"]["market_slug"],
            total_buffer_cents,
        )

    def detect_with_audit(
        self,
        rule: dict,
        lhs_relation_book,
        rhs_relation_book,
        lhs_execution_token: str,
        lhs_execution_side: str,
        lhs_execution_book,
        rhs_execution_token: str,
        rhs_execution_side: str,
        rhs_execution_book,
        max_notional: float,
        total_buffer_cents: float,
    ) -> tuple[RawCandidate | None, dict | None]:
        analysis = self._analyze_relation(
            rule,
            lhs_relation_book,
            rhs_relation_book,
            total_buffer_cents,
        )
        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
                "lhs_execution_side": lhs_execution_side,
                "rhs_execution_side": rhs_execution_side,
            }

        return self._build_raw_candidate(
            opportunity,
            rule,
            lhs_execution_token,
            lhs_execution_side,
            lhs_execution_book,
            rhs_execution_token,
            rhs_execution_side,
            rhs_execution_book,
            max_notional,
        ), None

    def _build_raw_candidate(
        self,
        opportunity,
        rule: dict,
        lhs_execution_token: str,
        lhs_execution_side: str,
        lhs_execution_book,
        rhs_execution_token: str,
        rhs_execution_side: str,
        rhs_execution_book,
        max_notional: float,
    ) -> RawCandidate:
        lhs_execution_best_ask = float(lhs_execution_book.asks[0].price) if getattr(lhs_execution_book, "asks", []) else None
        rhs_execution_best_ask = float(rhs_execution_book.asks[0].price) if getattr(rhs_execution_book, "asks", []) else None
        if lhs_execution_best_ask and rhs_execution_best_ask:
            lhs_budget_shares = (max_notional / 2.0) / lhs_execution_best_ask
            rhs_budget_shares = (max_notional / 2.0) / rhs_execution_best_ask
            target_shares = max(0.0, min(lhs_budget_shares, rhs_budget_shares))
            pair_cost = (lhs_execution_best_ask + rhs_execution_best_ask) * target_shares
            guaranteed_payout = target_shares
            executable_edge = max(0.0, 1.0 - lhs_execution_best_ask - rhs_execution_best_ask)
        else:
            target_shares = 0.0
            pair_cost = 0.0
            guaranteed_payout = 0.0
            executable_edge = 0.0

        return RawCandidate(
            strategy_id=self.strategy_id,
            strategy_family=self.strategy_family,
            candidate_id=str(uuid4()),
            kind=opportunity.kind,
            detection_name=opportunity.name,
            market_slugs=[rule["lhs"]["market_slug"], rule["rhs"]["market_slug"]],
            gross_edge_cents=opportunity.edge_cents,
            expected_payout=round(guaranteed_payout, 6),
            target_notional_usd=round(pair_cost, 6) if pair_cost > 0 else round(max_notional, 6),
            target_shares=round(target_shares, 6),
            gross_profit_usd=round(max(0.0, guaranteed_payout - pair_cost), 6),
            est_fill_cost_usd=round(pair_cost, 6),
            execution_mode="research_only",
            research_only=True,
            legs=[
                CandidateLeg(
                    token_id=lhs_execution_token,
                    market_slug=rule["lhs"]["market_slug"],
                    action="BUY",
                    side=lhs_execution_side,
                    required_shares=round(target_shares, 6),
                    best_price=lhs_execution_best_ask,
                ),
                CandidateLeg(
                    token_id=rhs_execution_token,
                    market_slug=rule["rhs"]["market_slug"],
                    action="BUY",
                    side=rhs_execution_side,
                    required_shares=round(target_shares, 6),
                    best_price=rhs_execution_best_ask,
                ),
            ],
            metadata={
                **opportunity.details,
                "strategy_family": self.strategy_family.value,
                "constraint_name": rule["name"],
                "constraint_relation": "implication_pair_cost_under_1",
                "lhs_relation_side": str(rule.get("lhs", {}).get("side", "YES")).upper(),
                "rhs_relation_side": str(rule.get("rhs", {}).get("side", "YES")).upper(),
                "execution_pair_best_ask_cost": round((lhs_execution_best_ask or 0.0) + (rhs_execution_best_ask or 0.0), 6),
                "execution_best_ask_edge_cents": round(executable_edge, 6),
                "lhs_execution_side": lhs_execution_side,
                "rhs_execution_side": rhs_execution_side,
            },
            ts=opportunity.ts,
        )

    def detect(
        self,
        rule: dict,
        lhs_relation_book,
        rhs_relation_book,
        lhs_execution_token: str,
        lhs_execution_side: str,
        lhs_execution_book,
        rhs_execution_token: str,
        rhs_execution_side: str,
        rhs_execution_book,
        max_notional: float,
        total_buffer_cents: float,
    ) -> RawCandidate | None:
        raw_candidate, _ = self.detect_with_audit(
            rule,
            lhs_relation_book,
            rhs_relation_book,
            lhs_execution_token,
            lhs_execution_side,
            lhs_execution_book,
            rhs_execution_token,
            rhs_execution_side,
            rhs_execution_book,
            max_notional,
            total_buffer_cents,
        )
        return raw_candidate


class CrossMarketGrossConstraintStrategy(CrossMarketConstraintStrategy):
    strategy_id = "cross_market_leq_gross"
    strategy_family = StrategyFamily.CROSS_MARKET_GROSS_CONSTRAINT

    def _analyze_relation(
        self,
        rule: dict,
        lhs_relation_book,
        rhs_relation_book,
        total_buffer_cents: float,
    ) -> dict[str, object]:
        return analyze_leq_constraint_gross(
            rule["name"],
            lhs_relation_book,
            rhs_relation_book,
            rule["lhs"]["market_slug"],
            rule["rhs"]["market_slug"],
            total_buffer_cents,
        )


class CrossMarketExecutionGrossConstraintStrategy(CrossMarketConstraintStrategy):
    strategy_id = "cross_market_execution_gross"
    strategy_family = StrategyFamily.CROSS_MARKET_EXECUTION_GROSS_CONSTRAINT

    def detect_with_audit(
        self,
        rule: dict,
        lhs_relation_book,
        rhs_relation_book,
        lhs_execution_token: str,
        lhs_execution_side: str,
        lhs_execution_book,
        rhs_execution_token: str,
        rhs_execution_side: str,
        rhs_execution_book,
        max_notional: float,
        total_buffer_cents: float,
    ) -> tuple[RawCandidate | None, dict | None]:
        analysis = analyze_leq_constraint_execution_gross(
            rule["name"],
            lhs_relation_book,
            rhs_relation_book,
            lhs_execution_book,
            rhs_execution_book,
            rule["lhs"]["market_slug"],
            rule["rhs"]["market_slug"],
            total_buffer_cents,
        )
        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
                "lhs_execution_side": lhs_execution_side,
                "rhs_execution_side": rhs_execution_side,
            }

        return self._build_raw_candidate(
            opportunity,
            rule,
            lhs_execution_token,
            lhs_execution_side,
            lhs_execution_book,
            rhs_execution_token,
            rhs_execution_side,
            rhs_execution_book,
            max_notional,
        ), None


class NegRiskRebalancingStrategy(BaseOpportunityStrategy):
    strategy_id = "neg_risk_rebalancing_v1"
    strategy_family = StrategyFamily.NEG_RISK_REBALANCING

    def detect_with_audit(
        self,
        event_group: dict,
        books_by_token: dict[str, object],
        max_notional: float,
    ) -> tuple[RawCandidate | None, dict | None]:
        analysis = analyze_neg_risk_rebalancing_event(
            event_group,
            books_by_token,
            max_notional=max_notional,
        )
        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
            }

        shares = float(opportunity.details.get("shares", 0.0))
        legs: list[CandidateLeg] = []
        for leg in list(opportunity.details.get("legs") or []):
            token_id = str(leg.get("yes_token_id") or "")
            market_slug = str(leg.get("market_slug") or "")
            maker_bid = leg.get("maker_yes_bid")
            legs.append(
                CandidateLeg(
                    token_id=token_id,
                    market_slug=market_slug,
                    action="BUY",
                    side="YES",
                    required_shares=shares,
                    best_price=float(maker_bid) if maker_bid is not None else None,
                    metadata={
                        "maker_first": True,
                        "maker_yes_bid": maker_bid,
                        "maker_yes_bid_size": leg.get("maker_yes_bid_size"),
                    },
                )
            )

        return RawCandidate(
            strategy_id=self.strategy_id,
            strategy_family=self.strategy_family,
            candidate_id=str(uuid4()),
            kind=opportunity.kind,
            detection_name=opportunity.name,
            market_slugs=[str(leg.market_slug) for leg in legs],
            gross_edge_cents=opportunity.edge_cents,
            expected_payout=opportunity.est_payout,
            target_notional_usd=opportunity.notional,
            target_shares=shares,
            gross_profit_usd=opportunity.gross_profit,
            est_fill_cost_usd=opportunity.est_fill_cost,
            execution_mode="paper_eligible",
            research_only=False,
            legs=legs,
            metadata={
                **opportunity.details,
                "strategy_family": self.strategy_family.value,
            },
            ts=opportunity.ts,
        ), None

    def detect(
        self,
        event_group: dict,
        books_by_token: dict[str, object],
        max_notional: float,
    ) -> RawCandidate | None:
        raw_candidate, _ = self.detect_with_audit(
            event_group,
            books_by_token,
            max_notional=max_notional,
        )
        return raw_candidate


class PoliticalBinaryConstraintPaperStrategy(CrossMarketConstraintStrategy):
    strategy_id = "political_binary_constraint_paper_v1"
    strategy_family = StrategyFamily.POLITICAL_BINARY_CONSTRAINT_PAPER

    def detect_with_audit(
        self,
        rule: dict,
        lhs_relation_book,
        rhs_relation_book,
        lhs_execution_token: str,
        lhs_execution_side: str,
        lhs_execution_book,
        rhs_execution_token: str,
        rhs_execution_side: str,
        rhs_execution_book,
        max_notional: float,
        total_buffer_cents: float,
    ) -> tuple[RawCandidate | None, dict | None]:
        relation_type = str(rule.get("relation_type") or "")
        if relation_type in {"nominee_family_mutex", "winner_family_mutex"}:
            analysis = analyze_political_mutex_pair(
                rule,
                lhs_relation_book,
                rhs_relation_book,
                lhs_execution_book,
                rhs_execution_book,
                total_buffer_cents,
            )
        elif relation_type in {"time_monotone_implication", "numeric_monotone_implication", "combo_decomposition"}:
            analysis = analyze_political_implication_pair(
                rule,
                lhs_relation_book,
                rhs_relation_book,
                lhs_execution_book,
                rhs_execution_book,
                total_buffer_cents,
            )
        else:
            return None, {
                "failure_stage": "pre_candidate_relation",
                "failure_reason": "UNSUPPORTED_POLITICAL_RELATION_TYPE",
                "constraint_name": rule.get("name"),
                "relation_type": relation_type,
                "strategy_family": self.strategy_family.value,
            }

        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
            }

        return self._build_raw_candidate(
            opportunity,
            rule,
            lhs_execution_token,
            lhs_execution_side,
            lhs_execution_book,
            rhs_execution_token,
            rhs_execution_side,
            rhs_execution_book,
            max_notional,
        ), None

    def _build_raw_candidate(
        self,
        opportunity,
        rule: dict,
        lhs_execution_token: str,
        lhs_execution_side: str,
        lhs_execution_book,
        rhs_execution_token: str,
        rhs_execution_side: str,
        rhs_execution_book,
        max_notional: float,
    ) -> RawCandidate:
        lhs_execution_best_ask = float(lhs_execution_book.asks[0].price) if getattr(lhs_execution_book, "asks", []) else None
        rhs_execution_best_ask = float(rhs_execution_book.asks[0].price) if getattr(rhs_execution_book, "asks", []) else None
        if lhs_execution_best_ask and rhs_execution_best_ask:
            lhs_budget_shares = (max_notional / 2.0) / lhs_execution_best_ask
            rhs_budget_shares = (max_notional / 2.0) / rhs_execution_best_ask
            target_shares = max(0.0, min(lhs_budget_shares, rhs_budget_shares))
            pair_cost = (lhs_execution_best_ask + rhs_execution_best_ask) * target_shares
            guaranteed_payout = target_shares
            executable_edge = max(0.0, 1.0 - lhs_execution_best_ask - rhs_execution_best_ask)
        else:
            target_shares = 0.0
            pair_cost = 0.0
            guaranteed_payout = 0.0
            executable_edge = 0.0

        return RawCandidate(
            strategy_id=self.strategy_id,
            strategy_family=self.strategy_family,
            candidate_id=str(uuid4()),
            kind=opportunity.kind,
            detection_name=opportunity.name,
            market_slugs=[rule["lhs"]["market_slug"], rule["rhs"]["market_slug"]],
            gross_edge_cents=opportunity.edge_cents,
            expected_payout=round(guaranteed_payout, 6),
            target_notional_usd=round(pair_cost, 6) if pair_cost > 0 else round(max_notional, 6),
            target_shares=round(target_shares, 6),
            gross_profit_usd=round(max(0.0, guaranteed_payout - pair_cost), 6),
            est_fill_cost_usd=round(pair_cost, 6),
            execution_mode="paper_eligible",
            research_only=False,
            legs=[
                CandidateLeg(
                    token_id=lhs_execution_token,
                    market_slug=rule["lhs"]["market_slug"],
                    action="BUY",
                    side=lhs_execution_side,
                    required_shares=round(target_shares, 6),
                    best_price=lhs_execution_best_ask,
                ),
                CandidateLeg(
                    token_id=rhs_execution_token,
                    market_slug=rule["rhs"]["market_slug"],
                    action="BUY",
                    side=rhs_execution_side,
                    required_shares=round(target_shares, 6),
                    best_price=rhs_execution_best_ask,
                ),
            ],
            metadata={
                **opportunity.details,
                "strategy_family": self.strategy_family.value,
                "constraint_name": rule["name"],
                "constraint_relation": rule.get("assertion"),
                "constraint_tier": rule.get("tier"),
                "constraint_model_id": rule.get("constraint_model_id"),
                "relation_type": rule.get("relation_type"),
                "lhs_relation_side": str(rule.get("lhs", {}).get("side", "YES")).upper(),
                "rhs_relation_side": str(rule.get("rhs", {}).get("side", "YES")).upper(),
                "lhs_execution_side": lhs_execution_side,
                "rhs_execution_side": rhs_execution_side,
                "notes": rule.get("notes"),
                "preconditions": rule.get("preconditions", {}),
            },
            ts=opportunity.ts,
        )


class MakerRewardedEventMMStrategy(BaseOpportunityStrategy):
    strategy_id = "maker_rewarded_event_mm_v1"
    strategy_family = StrategyFamily.MAKER_REWARDED_EVENT_MM_V1

    def detect_with_audit(self, event_group: dict, market: dict) -> tuple[RawCandidate | None, dict | None]:
        analysis = analyze_maker_rewarded_market(
            event_group=event_group,
            market=market,
        )
        opportunity = analysis["opportunity"]
        if opportunity is None:
            audit = analysis["audit"] or {}
            return None, {
                **audit,
                "strategy_family": self.strategy_family.value,
            }

        details = dict(opportunity.details)
        quote_size = float(details.get("quote_size", 0.0))
        quote_bid = details.get("quote_bid")
        quote_ask = details.get("quote_ask")
        market_slug = str(details.get("market_slug") or market.get("market_slug") or "")

        return RawCandidate(
            strategy_id=self.strategy_id,
            strategy_family=self.strategy_family,
            candidate_id=str(uuid4()),
            kind=opportunity.kind,
            detection_name=opportunity.name,
            market_slugs=[market_slug],
            gross_edge_cents=opportunity.edge_cents,
            expected_payout=opportunity.est_payout,
            target_notional_usd=opportunity.notional,
            target_shares=quote_size,
            gross_profit_usd=opportunity.gross_profit,
            est_fill_cost_usd=opportunity.est_fill_cost,
            execution_mode="research_only",
            research_only=True,
            legs=[
                CandidateLeg(
                    token_id=str(market.get("yes_token_id") or ""),
                    market_slug=market_slug,
                    action="MAKE",
                    side="BOTH",
                    required_shares=quote_size,
                    best_price=float(quote_bid) if quote_bid is not None else None,
                    metadata={
                        "quote_bid": quote_bid,
                        "quote_ask": quote_ask,
                        "maker_only": True,
                        "paper_only": True,
                    },
                )
            ],
            metadata={
                **details,
                "strategy_family": self.strategy_family.value,
            },
            ts=opportunity.ts,
        ), None

    def detect(self, event_group: dict, market: dict) -> RawCandidate | None:
        raw_candidate, _ = self.detect_with_audit(event_group, market)
        return raw_candidate


class RebalancingStrategy(BaseOpportunityStrategy):
    strategy_id = "rebalancing"
    strategy_family = StrategyFamily.REBALANCING

    def detect(self, *args, **kwargs) -> RawCandidate | None:
        return None


class ExternalBeliefStrategy(BaseOpportunityStrategy):
    strategy_id = "external_belief"
    strategy_family = StrategyFamily.EXTERNAL_BELIEF

    def detect(self, *args, **kwargs) -> RawCandidate | None:
        return None
