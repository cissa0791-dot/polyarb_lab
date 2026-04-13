from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


POLITICAL_BINARY_MODEL_ID = "political_binary_constraint_paper_v1"
POLITICAL_BINARY_STRATEGY_FAMILY = "political_binary_constraint_paper"

_TRADEABLE_RELATION_TYPES = {
    "nominee_family_mutex",
    "winner_family_mutex",
    "time_monotone_implication",
    "numeric_monotone_implication",
    "combo_decomposition",
}
_MUTEX_RELATION_TYPES = {
    "nominee_family_mutex",
    "winner_family_mutex",
}
_IMPLICATION_RELATION_TYPES = {
    "time_monotone_implication",
    "numeric_monotone_implication",
    "combo_decomposition",
}


class PoliticalMarketRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_slug: str
    side: str = "YES"


class PoliticalConstraintRule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relation_id: str
    tier: str
    relation_type: str
    lhs_market: PoliticalMarketRef
    rhs_market: PoliticalMarketRef
    preconditions: dict[str, Any] = Field(default_factory=dict)
    assertion: str
    trade_enabled: bool = False
    notes: str | None = None


class PoliticalConstraintRuleSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_id: str = POLITICAL_BINARY_MODEL_ID
    rules: list[PoliticalConstraintRule] = Field(default_factory=list)


def build_constraint_scan_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if not _looks_like_political_rule_payload(payload):
        return payload

    ruleset = PoliticalConstraintRuleSet.model_validate(payload)
    approved_slugs: set[str] = set()
    normalized_rules: list[dict[str, Any]] = []

    for rule in ruleset.rules:
        approved_slugs.add(rule.lhs_market.market_slug)
        approved_slugs.add(rule.rhs_market.market_slug)

        if not rule.trade_enabled or str(rule.tier).upper() != "A":
            continue
        if rule.relation_type not in _TRADEABLE_RELATION_TYPES:
            raise ValueError(
                f"Unsupported political relation_type '{rule.relation_type}' in {rule.relation_id}"
            )

        normalized_rule = {
            "name": rule.relation_id,
            "relation": "political_binary",
            "relation_type": rule.relation_type,
            "strategy_family": POLITICAL_BINARY_STRATEGY_FAMILY,
            "lhs": {
                "market_slug": rule.lhs_market.market_slug,
                "side": str(rule.lhs_market.side).upper(),
            },
            "rhs": {
                "market_slug": rule.rhs_market.market_slug,
                "side": str(rule.rhs_market.side).upper(),
            },
            "lhs_execution": {
                "side": "NO",
            },
            "rhs_execution": {
                "side": "NO" if rule.relation_type in _MUTEX_RELATION_TYPES else "YES",
            },
            "tier": rule.tier,
            "assertion": rule.assertion,
            "trade_enabled": rule.trade_enabled,
            "notes": rule.notes,
            "preconditions": rule.preconditions,
            "constraint_model_id": ruleset.model_id,
        }
        normalized_rules.append(normalized_rule)

    return {
        "single_market": False,
        "cross_market": normalized_rules,
        "approved_market_slugs": sorted(approved_slugs),
        "constraint_model_id": ruleset.model_id,
        "constraint_domain": "political_binary",
        "approved_rule_count": len(normalized_rules),
        "rule_count": len(ruleset.rules),
    }


def _looks_like_political_rule_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("model_id") == POLITICAL_BINARY_MODEL_ID:
        return True
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        return False
    first = rules[0]
    if not isinstance(first, dict):
        return False
    required_keys = {"relation_id", "tier", "relation_type", "lhs_market", "rhs_market", "assertion"}
    return required_keys.issubset(first)

