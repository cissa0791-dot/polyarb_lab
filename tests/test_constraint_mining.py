from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.core.models import BookLevel, OrderBook
from src.runtime.constraint_mining import (
    build_constraints_document,
    discover_cross_market_constraints,
    limit_constraints,
    rank_constraints_by_current_execution,
    rank_constraints_by_current_relation,
)


def _market(
    slug: str,
    question: str | None = None,
    *,
    yes_token_id: str | None = None,
    no_token_id: str | None = None,
) -> dict[str, object]:
    return {
        "slug": slug,
        "question": question or slug,
        "outcomes": ["YES", "NO"] if yes_token_id and no_token_id else [],
        "clobTokenIds": [yes_token_id, no_token_id] if yes_token_id and no_token_id else [],
    }


def _book(token_id: str, best_ask: float | None) -> OrderBook:
    asks = [] if best_ask is None else [BookLevel(price=best_ask, size=100.0)]
    return OrderBook(
        token_id=token_id,
        bids=[],
        asks=asks,
        ts=datetime.now(timezone.utc),
        metadata={},
    )


_FAKE_BOOKS = {
    "celtics-finals-yes": _book("celtics-finals-yes", 0.06),
    "celtics-finals-no": _book("celtics-finals-no", 0.99),
    "celtics-conf-yes": _book("celtics-conf-yes", 0.03),
    "celtics-conf-no": _book("celtics-conf-no", 0.98),
    "italy-win-yes": _book("italy-win-yes", 0.02),
    "italy-win-no": _book("italy-win-no", 0.96),
    "italy-qualify-yes": _book("italy-qualify-yes", 0.03),
    "italy-qualify-no": _book("italy-qualify-no", 0.97),
}


class _FakeClob:
    def __init__(self, host: str):
        self.host = host

    def get_books(self, token_ids: list[str]) -> list[OrderBook]:
        return [_FAKE_BOOKS[token_id] for token_id in token_ids if token_id in _FAKE_BOOKS]


class ConstraintMiningTests(unittest.TestCase):
    def test_discovers_supported_relation_families(self) -> None:
        markets = [
            _market("will-the-boston-celtics-win-the-2026-nba-finals"),
            _market("will-the-boston-celtics-win-the-nba-eastern-conference-finals"),
            _market("will-italy-win-the-2026-fifa-world-cup-813"),
            _market("will-italy-qualify-for-the-2026-fifa-world-cup"),
            _market("will-gavin-newsom-win-the-2028-us-presidential-election"),
            _market("will-gavin-newsom-win-the-2028-democratic-presidential-nomination-568"),
        ]

        discovered = discover_cross_market_constraints(markets)

        self.assertEqual(len(discovered), 3)
        names = {item.name for item in discovered}
        self.assertIn("boston-celtics_eastern_nba_finals_implies_conference", names)
        self.assertIn("italy_world_cup_win_implies_qualify", names)
        self.assertIn("gavin-newsom_democratic_election_implies_nomination", names)

    def test_build_constraints_document_preserves_single_market_toggle(self) -> None:
        markets = [
            _market("will-the-boston-celtics-win-the-2026-nba-finals"),
            _market("will-the-boston-celtics-win-the-nba-eastern-conference-finals"),
        ]

        document = build_constraints_document(markets, single_market_enabled=False)

        self.assertEqual(document["single_market"]["enabled"], False)
        self.assertEqual(len(document["cross_market"]), 1)
        self.assertEqual(document["cross_market"][0]["relation"], "leq")
        self.assertEqual(document["cross_market"][0]["discovery_rule"], "nba_finals_implies_conference")

    @patch("src.runtime.constraint_mining.ReadOnlyClob", _FakeClob)
    def test_rank_constraints_by_current_relation_prefers_best_current_gap(self) -> None:
        markets = [
            _market(
                "will-the-boston-celtics-win-the-2026-nba-finals",
                yes_token_id="celtics-finals-yes",
                no_token_id="celtics-finals-no",
            ),
            _market(
                "will-the-boston-celtics-win-the-nba-eastern-conference-finals",
                yes_token_id="celtics-conf-yes",
                no_token_id="celtics-conf-no",
            ),
            _market(
                "will-italy-win-the-2026-fifa-world-cup-813",
                yes_token_id="italy-win-yes",
                no_token_id="italy-win-no",
            ),
            _market(
                "will-italy-qualify-for-the-2026-fifa-world-cup",
                yes_token_id="italy-qualify-yes",
                no_token_id="italy-qualify-no",
            ),
        ]

        discovered = discover_cross_market_constraints(markets)
        ranked = rank_constraints_by_current_relation(
            discovered,
            markets,
            clob_host="https://clob.polymarket.com",
            total_buffer_cents=0.02,
        )

        self.assertEqual(ranked[0].name, "boston-celtics_eastern_nba_finals_implies_conference")
        self.assertEqual(ranked[0].current_relation_rank, 1)
        self.assertEqual(ranked[0].relation_gap, 0.03)
        self.assertEqual(ranked[0].edge_after_buffer, 0.01)
        self.assertEqual(ranked[1].name, "italy_world_cup_win_implies_qualify")
        self.assertEqual(ranked[1].current_relation_rank, 2)
        self.assertEqual(ranked[1].edge_after_buffer, -0.03)

        limited = limit_constraints(ranked, 1)
        self.assertEqual(len(limited), 1)
        payload = build_constraints_document(markets, single_market_enabled=False, constraints=limited)
        self.assertEqual(payload["cross_market"][0]["metadata"]["current_relation_rank"], 1)
        self.assertEqual(payload["cross_market"][0]["metadata"]["edge_after_buffer"], 0.01)
        self.assertEqual(payload["cross_market"][0]["metadata"]["execution_best_ask_edge_cents"], -0.02)

    @patch("src.runtime.constraint_mining.ReadOnlyClob", _FakeClob)
    def test_rank_constraints_by_current_execution_prefers_best_execution_edge(self) -> None:
        markets = [
            _market(
                "will-the-boston-celtics-win-the-2026-nba-finals",
                yes_token_id="celtics-finals-yes",
                no_token_id="celtics-finals-no",
            ),
            _market(
                "will-the-boston-celtics-win-the-nba-eastern-conference-finals",
                yes_token_id="celtics-conf-yes",
                no_token_id="celtics-conf-no",
            ),
            _market(
                "will-italy-win-the-2026-fifa-world-cup-813",
                yes_token_id="italy-win-yes",
                no_token_id="italy-win-no",
            ),
            _market(
                "will-italy-qualify-for-the-2026-fifa-world-cup",
                yes_token_id="italy-qualify-yes",
                no_token_id="italy-qualify-no",
            ),
        ]

        discovered = discover_cross_market_constraints(markets)
        ranked = rank_constraints_by_current_execution(
            discovered,
            markets,
            clob_host="https://clob.polymarket.com",
            total_buffer_cents=0.02,
        )

        self.assertEqual(ranked[0].name, "italy_world_cup_win_implies_qualify")
        self.assertEqual(ranked[0].current_execution_rank, 1)
        self.assertEqual(ranked[0].execution_pair_best_ask_cost, 0.99)
        self.assertEqual(ranked[0].execution_best_ask_edge_cents, 0.01)
        self.assertEqual(ranked[1].name, "boston-celtics_eastern_nba_finals_implies_conference")
        self.assertEqual(ranked[1].current_execution_rank, 2)
        self.assertEqual(ranked[1].execution_best_ask_edge_cents, -0.02)

        payload = build_constraints_document(markets, single_market_enabled=False, constraints=ranked[:1])
        self.assertEqual(payload["cross_market"][0]["metadata"]["current_execution_rank"], 1)
        self.assertEqual(payload["cross_market"][0]["metadata"]["execution_pair_best_ask_cost"], 0.99)


if __name__ == "__main__":
    unittest.main()
