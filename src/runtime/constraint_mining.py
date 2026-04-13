from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from src.core.models import OrderBook
from src.core.normalize import build_yes_no_pairs
from src.ingest.clob import ReadOnlyClob


_NBA_FINALS_RE = re.compile(r"^will-the-(?P<subject>.+)-win-the-2026-nba-finals$")
_NBA_CONF_RE = re.compile(r"^will-the-(?P<subject>.+)-win-the-nba-(?P<conference>eastern|western)-conference-finals$")
_WORLD_CUP_WIN_RE = re.compile(r"^will-(?P<subject>.+)-win-the-2026-fifa-world-cup(?:-[0-9]+)?$")
_WORLD_CUP_QUALIFY_RE = re.compile(r"^will-(?P<subject>.+)-qualify-for-the-2026-fifa-world-cup$")
_ELECTION_RE = re.compile(r"^will-(?P<subject>.+)-win-the-2028-us-presidential-election$")
_NOMINATION_RE = re.compile(r"^will-(?P<subject>.+)-win-the-2028-(?P<party>democratic|republican)-presidential-nomination(?:-[0-9]+)?$")
_NHL_CHAMPION_RE = re.compile(r"^will-the-(?P<subject>.+)-win-the-2026-nhl-stanley-cup$")
_NHL_PLAYOFF_RE = re.compile(r"^will-the-(?P<subject>.+)-make-the-nhl-playoffs$")
_NBA_PLAYOFF_TEAM_RE = re.compile(r"^will-the-(?P<subject>.+)-make-the-nba-playoffs(?:-[0-9]+)?$")


@dataclass(frozen=True)
class DiscoveredConstraint:
    name: str
    lhs_market_slug: str
    rhs_market_slug: str
    relation: str
    discovery_rule: str
    lhs_question: str | None = None
    rhs_question: str | None = None
    lhs_relation_ask: float | None = None
    rhs_relation_ask: float | None = None
    relation_gap: float | None = None
    edge_after_buffer: float | None = None
    current_relation_rank: int | None = None
    lhs_execution_ask: float | None = None
    rhs_execution_ask: float | None = None
    execution_pair_best_ask_cost: float | None = None
    execution_best_ask_edge_cents: float | None = None
    current_execution_rank: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "lhs": {
                "market_slug": self.lhs_market_slug,
                "side": "YES",
            },
            "rhs": {
                "market_slug": self.rhs_market_slug,
                "side": "YES",
            },
            "relation": self.relation,
            "discovery_rule": self.discovery_rule,
        }
        metadata: dict[str, Any] = {}
        if self.lhs_question or self.rhs_question:
            metadata["lhs_question"] = self.lhs_question
            metadata["rhs_question"] = self.rhs_question
        if self.current_relation_rank is not None:
            metadata["current_relation_rank"] = self.current_relation_rank
        if self.current_execution_rank is not None:
            metadata["current_execution_rank"] = self.current_execution_rank
        if self.lhs_relation_ask is not None:
            metadata["lhs_relation_ask"] = self.lhs_relation_ask
        if self.rhs_relation_ask is not None:
            metadata["rhs_relation_ask"] = self.rhs_relation_ask
        if self.relation_gap is not None:
            metadata["relation_gap"] = self.relation_gap
        if self.edge_after_buffer is not None:
            metadata["edge_after_buffer"] = self.edge_after_buffer
        if self.lhs_execution_ask is not None:
            metadata["lhs_execution_ask"] = self.lhs_execution_ask
        if self.rhs_execution_ask is not None:
            metadata["rhs_execution_ask"] = self.rhs_execution_ask
        if self.execution_pair_best_ask_cost is not None:
            metadata["execution_pair_best_ask_cost"] = self.execution_pair_best_ask_cost
        if self.execution_best_ask_edge_cents is not None:
            metadata["execution_best_ask_edge_cents"] = self.execution_best_ask_edge_cents
        if metadata:
            payload["metadata"] = metadata
        return payload


def discover_cross_market_constraints(markets: list[dict[str, Any]]) -> list[DiscoveredConstraint]:
    by_slug = {
        str(market.get("slug") or "").strip(): market
        for market in markets
        if str(market.get("slug") or "").strip()
    }

    constraints: list[DiscoveredConstraint] = []
    constraints.extend(_discover_nba_finals_vs_conference(by_slug))
    constraints.extend(_discover_world_cup_qualify_relations(by_slug))
    constraints.extend(_discover_election_nomination_relations(by_slug))
    constraints.extend(_discover_nhl_champion_vs_playoffs(by_slug))
    constraints.extend(_discover_nba_finals_vs_playoffs(by_slug))

    deduped: dict[tuple[str, str, str], DiscoveredConstraint] = {}
    for constraint in constraints:
        deduped[(constraint.lhs_market_slug, constraint.rhs_market_slug, constraint.discovery_rule)] = constraint
    return sorted(deduped.values(), key=lambda item: (item.discovery_rule, item.name))


def build_constraints_document(
    markets: list[dict[str, Any]],
    *,
    single_market_enabled: bool = True,
    constraints: list[DiscoveredConstraint] | None = None,
) -> dict[str, Any]:
    constraints = list(constraints) if constraints is not None else discover_cross_market_constraints(markets)
    return {
        "single_market": {
            "enabled": bool(single_market_enabled),
        },
        "cross_market": [constraint.to_payload() for constraint in constraints],
    }


def rank_constraints_by_current_relation(
    constraints: list[DiscoveredConstraint],
    markets: list[dict[str, Any]],
    *,
    clob_host: str,
    total_buffer_cents: float,
) -> list[DiscoveredConstraint]:
    scored = _score_constraints(
        constraints,
        markets,
        clob_host=clob_host,
        total_buffer_cents=total_buffer_cents,
    )
    ranked = sorted(scored, key=_constraint_rank_key)
    return [
        replace(constraint, current_relation_rank=index)
        for index, constraint in enumerate(ranked, start=1)
    ]


def rank_constraints_by_current_execution(
    constraints: list[DiscoveredConstraint],
    markets: list[dict[str, Any]],
    *,
    clob_host: str,
    total_buffer_cents: float,
) -> list[DiscoveredConstraint]:
    scored = _score_constraints(
        constraints,
        markets,
        clob_host=clob_host,
        total_buffer_cents=total_buffer_cents,
    )
    ranked = sorted(scored, key=_constraint_execution_rank_key)
    return [
        replace(constraint, current_execution_rank=index)
        for index, constraint in enumerate(ranked, start=1)
    ]


def _score_constraints(
    constraints: list[DiscoveredConstraint],
    markets: list[dict[str, Any]],
    *,
    clob_host: str,
    total_buffer_cents: float,
) -> list[DiscoveredConstraint]:
    if not constraints:
        return []

    pairs_by_slug = {
        pair.market_slug: pair
        for pair in build_yes_no_pairs(markets)
    }
    token_ids = sorted(
        {
            token_id
            for constraint in constraints
            for token_id in (
                getattr(pairs_by_slug.get(constraint.lhs_market_slug), "yes_token_id", None),
                getattr(pairs_by_slug.get(constraint.rhs_market_slug), "yes_token_id", None),
                getattr(pairs_by_slug.get(constraint.lhs_market_slug), "no_token_id", None),
                getattr(pairs_by_slug.get(constraint.rhs_market_slug), "yes_token_id", None),
            )
            if token_id
        }
    )
    books_by_token = _load_books_by_token(clob_host, token_ids)

    return [
        _score_constraint(
            constraint,
            lhs_relation_book=books_by_token.get(getattr(pairs_by_slug.get(constraint.lhs_market_slug), "yes_token_id", "")),
            rhs_relation_book=books_by_token.get(getattr(pairs_by_slug.get(constraint.rhs_market_slug), "yes_token_id", "")),
            lhs_execution_book=books_by_token.get(getattr(pairs_by_slug.get(constraint.lhs_market_slug), "no_token_id", "")),
            rhs_execution_book=books_by_token.get(getattr(pairs_by_slug.get(constraint.rhs_market_slug), "yes_token_id", "")),
            total_buffer_cents=total_buffer_cents,
        )
        for constraint in constraints
    ]


def limit_constraints(constraints: list[DiscoveredConstraint], max_constraints: int | None) -> list[DiscoveredConstraint]:
    if max_constraints is None or max_constraints <= 0:
        return list(constraints)
    return list(constraints[:max_constraints])


def _discover_nba_finals_vs_conference(by_slug: dict[str, dict[str, Any]]) -> list[DiscoveredConstraint]:
    finals_by_subject: dict[str, dict[str, Any]] = {}
    conference_by_subject: dict[str, tuple[str, dict[str, Any]]] = {}

    for slug, market in by_slug.items():
        finals_match = _NBA_FINALS_RE.match(slug)
        if finals_match:
            finals_by_subject[finals_match.group("subject")] = market
            continue
        conf_match = _NBA_CONF_RE.match(slug)
        if conf_match:
            conference_by_subject[conf_match.group("subject")] = (conf_match.group("conference"), market)

    constraints: list[DiscoveredConstraint] = []
    for subject, lhs_market in finals_by_subject.items():
        rhs_match = conference_by_subject.get(subject)
        if rhs_match is None:
            continue
        conference, rhs_market = rhs_match
        constraints.append(
            DiscoveredConstraint(
                name=f"{subject}_{conference}_nba_finals_implies_conference",
                lhs_market_slug=str(lhs_market["slug"]),
                rhs_market_slug=str(rhs_market["slug"]),
                relation="leq",
                discovery_rule="nba_finals_implies_conference",
                lhs_question=_question(lhs_market),
                rhs_question=_question(rhs_market),
            )
        )
    return constraints


def _discover_world_cup_qualify_relations(by_slug: dict[str, dict[str, Any]]) -> list[DiscoveredConstraint]:
    win_by_subject: dict[str, dict[str, Any]] = {}
    qualify_by_subject: dict[str, dict[str, Any]] = {}

    for slug, market in by_slug.items():
        win_match = _WORLD_CUP_WIN_RE.match(slug)
        if win_match:
            win_by_subject[win_match.group("subject")] = market
            continue
        qualify_match = _WORLD_CUP_QUALIFY_RE.match(slug)
        if qualify_match:
            qualify_by_subject[qualify_match.group("subject")] = market

    constraints: list[DiscoveredConstraint] = []
    for subject, lhs_market in win_by_subject.items():
        rhs_market = qualify_by_subject.get(subject)
        if rhs_market is None:
            continue
        constraints.append(
            DiscoveredConstraint(
                name=f"{subject}_world_cup_win_implies_qualify",
                lhs_market_slug=str(lhs_market["slug"]),
                rhs_market_slug=str(rhs_market["slug"]),
                relation="leq",
                discovery_rule="world_cup_win_implies_qualify",
                lhs_question=_question(lhs_market),
                rhs_question=_question(rhs_market),
            )
        )
    return constraints


def _discover_election_nomination_relations(by_slug: dict[str, dict[str, Any]]) -> list[DiscoveredConstraint]:
    election_by_subject: dict[str, dict[str, Any]] = {}
    nomination_by_subject: dict[str, tuple[str, dict[str, Any]]] = {}

    for slug, market in by_slug.items():
        election_match = _ELECTION_RE.match(slug)
        if election_match:
            election_by_subject[election_match.group("subject")] = market
            continue
        nomination_match = _NOMINATION_RE.match(slug)
        if nomination_match:
            nomination_by_subject[nomination_match.group("subject")] = (nomination_match.group("party"), market)

    constraints: list[DiscoveredConstraint] = []
    for subject, lhs_market in election_by_subject.items():
        rhs_match = nomination_by_subject.get(subject)
        if rhs_match is None:
            continue
        party, rhs_market = rhs_match
        constraints.append(
            DiscoveredConstraint(
                name=f"{subject}_{party}_election_implies_nomination",
                lhs_market_slug=str(lhs_market["slug"]),
                rhs_market_slug=str(rhs_market["slug"]),
                relation="leq",
                discovery_rule="election_implies_party_nomination",
                lhs_question=_question(lhs_market),
                rhs_question=_question(rhs_market),
            )
        )
    return constraints


def _discover_nba_finals_vs_playoffs(by_slug: dict[str, dict[str, Any]]) -> list[DiscoveredConstraint]:
    """
    NBA Finals win implies team made the playoffs.
    LHS: will-the-{subject}-win-the-2026-nba-finals
    RHS: will-the-{subject}-make-the-nba-playoffs[-NNN]
    """
    finals_by_subject: dict[str, dict[str, Any]] = {}
    playoff_by_subject: dict[str, dict[str, Any]] = {}

    for slug, market in by_slug.items():
        finals_match = _NBA_FINALS_RE.match(slug)
        if finals_match:
            finals_by_subject[finals_match.group("subject")] = market
            continue
        playoff_match = _NBA_PLAYOFF_TEAM_RE.match(slug)
        if playoff_match:
            playoff_by_subject[playoff_match.group("subject")] = market

    constraints: list[DiscoveredConstraint] = []
    for subject, lhs_market in finals_by_subject.items():
        rhs_market = playoff_by_subject.get(subject)
        if rhs_market is None:
            continue
        constraints.append(
            DiscoveredConstraint(
                name=f"{subject}_nba_finals_implies_playoffs",
                lhs_market_slug=str(lhs_market["slug"]),
                rhs_market_slug=str(rhs_market["slug"]),
                relation="leq",
                discovery_rule="nba_finals_implies_playoffs",
                lhs_question=_question(lhs_market),
                rhs_question=_question(rhs_market),
            )
        )
    return constraints


def _discover_nhl_champion_vs_playoffs(by_slug: dict[str, dict[str, Any]]) -> list[DiscoveredConstraint]:
    champion_by_subject: dict[str, dict[str, Any]] = {}
    playoff_by_subject: dict[str, dict[str, Any]] = {}

    for slug, market in by_slug.items():
        champ_match = _NHL_CHAMPION_RE.match(slug)
        if champ_match:
            champion_by_subject[champ_match.group("subject")] = market
            continue
        playoff_match = _NHL_PLAYOFF_RE.match(slug)
        if playoff_match:
            playoff_by_subject[playoff_match.group("subject")] = market

    constraints: list[DiscoveredConstraint] = []
    for subject, lhs_market in champion_by_subject.items():
        rhs_market = playoff_by_subject.get(subject)
        if rhs_market is None:
            continue
        constraints.append(
            DiscoveredConstraint(
                name=f"{subject}_nhl_champion_implies_playoffs",
                lhs_market_slug=str(lhs_market["slug"]),
                rhs_market_slug=str(rhs_market["slug"]),
                relation="leq",
                discovery_rule="nhl_champion_implies_playoffs",
                lhs_question=_question(lhs_market),
                rhs_question=_question(rhs_market),
            )
        )
    return constraints


def _question(market: dict[str, Any]) -> str | None:
    value = market.get("question")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_books_by_token(clob_host: str, token_ids: list[str]) -> dict[str, OrderBook]:
    if not token_ids:
        return {}
    clob = ReadOnlyClob(clob_host)
    books: dict[str, OrderBook] = {}
    try:
        for book in clob.get_books(token_ids):
            books[str(book.token_id)] = book
        return books
    except Exception:
        for token_id in token_ids:
            try:
                books[token_id] = clob.get_book(token_id)
            except Exception:
                continue
        return books


def _score_constraint(
    constraint: DiscoveredConstraint,
    *,
    lhs_relation_book: OrderBook | None,
    rhs_relation_book: OrderBook | None,
    lhs_execution_book: OrderBook | None,
    rhs_execution_book: OrderBook | None,
    total_buffer_cents: float,
) -> DiscoveredConstraint:
    lhs_relation_ask = _best_ask(lhs_relation_book)
    rhs_relation_ask = _best_ask(rhs_relation_book)
    lhs_execution_ask = _best_ask(lhs_execution_book)
    rhs_execution_ask = _best_ask(rhs_execution_book)
    relation_gap = None
    edge_after_buffer = None
    if lhs_relation_ask is not None and rhs_relation_ask is not None:
        relation_gap = round(lhs_relation_ask - rhs_relation_ask, 6)
        edge_after_buffer = round(relation_gap - float(total_buffer_cents), 6)
    execution_pair_best_ask_cost = None
    execution_best_ask_edge_cents = None
    if lhs_execution_ask is not None and rhs_execution_ask is not None:
        execution_pair_best_ask_cost = round(lhs_execution_ask + rhs_execution_ask, 6)
        execution_best_ask_edge_cents = round(1.0 - execution_pair_best_ask_cost, 6)
    return replace(
        constraint,
        lhs_relation_ask=lhs_relation_ask,
        rhs_relation_ask=rhs_relation_ask,
        relation_gap=relation_gap,
        edge_after_buffer=edge_after_buffer,
        lhs_execution_ask=lhs_execution_ask,
        rhs_execution_ask=rhs_execution_ask,
        execution_pair_best_ask_cost=execution_pair_best_ask_cost,
        execution_best_ask_edge_cents=execution_best_ask_edge_cents,
    )


def _best_ask(book: OrderBook | None) -> float | None:
    if book is None or not book.asks:
        return None
    return float(book.asks[0].price)


def _constraint_rank_key(constraint: DiscoveredConstraint) -> tuple[int, float, str]:
    if constraint.edge_after_buffer is None:
        return (1, 0.0, constraint.name)
    return (0, -constraint.edge_after_buffer, constraint.name)


def _constraint_execution_rank_key(constraint: DiscoveredConstraint) -> tuple[int, float, float, str]:
    if constraint.execution_best_ask_edge_cents is None:
        return (1, 0.0, 0.0, constraint.name)
    relation_gap = constraint.relation_gap or 0.0
    return (0, -constraint.execution_best_ask_edge_cents, -relation_gap, constraint.name)
