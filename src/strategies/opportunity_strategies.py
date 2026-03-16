from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.core.models import MarketPair
from src.opportunity.models import CandidateLeg, RawCandidate, StrategyFamily
from src.scanner.cross_market import scan_leq_constraint
from src.scanner.single_market import scan_yes_no_pair
from src.strategies.base import BaseOpportunityStrategy


class SingleMarketMispricingStrategy(BaseOpportunityStrategy):
    strategy_id = "single_market_sum_under_1"
    strategy_family = StrategyFamily.SINGLE_MARKET_MISPRICING

    def detect(self, pair: MarketPair, yes_book, no_book, max_notional: float, total_buffer_cents: float) -> RawCandidate | None:
        opportunity = scan_yes_no_pair(pair, yes_book, no_book, max_notional=max_notional, total_buffer_cents=total_buffer_cents)
        if opportunity is None:
            return None

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
        )


class CrossMarketConstraintStrategy(BaseOpportunityStrategy):
    strategy_id = "cross_market_leq"
    strategy_family = StrategyFamily.CROSS_MARKET_CONSTRAINT

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
        opportunity = scan_leq_constraint(
            rule["name"],
            lhs_relation_book,
            rhs_relation_book,
            rule["lhs"]["market_slug"],
            rule["rhs"]["market_slug"],
            total_buffer_cents,
        )
        if opportunity is None:
            return None

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
