from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from src.ingest import gamma


class MarketSlicingTests(unittest.TestCase):
    def test_near_term_liquidity_core_filters_across_pages_with_stable_order(self) -> None:
        now = datetime(2026, 3, 16, tzinfo=timezone.utc)
        market_slice = {
            "name": "near_term_liquidity_core",
            "max_days_to_end": 90,
            "min_liquidity_num": 2500,
        }

        def market(market_id: str, slug: str, *, days_to_end: int, liquidity: float) -> dict:
            return {
                "id": market_id,
                "slug": slug,
                "endDate": (now + timedelta(days=days_to_end)).isoformat().replace("+00:00", "Z"),
                "liquidityNum": liquidity,
                "active": True,
                "closed": False,
            }

        pages = {
            0: [
                market("1", "accept-first", days_to_end=10, liquidity=3000),
                market("2", "reject-far", days_to_end=120, liquidity=5000),
                market("3", "reject-thin", days_to_end=20, liquidity=1000),
            ],
            3: [
                market("1", "accept-first", days_to_end=10, liquidity=3000),
                market("4", "accept-second", days_to_end=30, liquidity=2600),
                market("5", "accept-third", days_to_end=60, liquidity=6000),
            ],
        }

        with patch.object(gamma, "DEFAULT_MARKETS_PAGE_SIZE", 3):
            with patch.object(gamma, "_fetch_markets_page", side_effect=lambda host, limit, offset: pages.get(offset, [])):
                results = gamma.fetch_markets_with_slice(
                    "https://gamma-api.polymarket.com",
                    limit=3,
                    market_slice=market_slice,
                    now=now,
                )

        self.assertEqual([item["slug"] for item in results], ["accept-first", "accept-second", "accept-third"])
        self.assertEqual(len({item["id"] for item in results}), 3)

    def test_liquidity_core_ignores_end_date_and_filters_only_on_liquidity(self) -> None:
        now = datetime(2026, 3, 16, tzinfo=timezone.utc)
        market_slice = {
            "name": "liquidity_core",
            "min_liquidity_num": 2500,
        }

        def market(market_id: str, slug: str, *, days_to_end: int, liquidity: float) -> dict:
            return {
                "id": market_id,
                "slug": slug,
                "endDate": (now + timedelta(days=days_to_end)).isoformat().replace("+00:00", "Z"),
                "liquidityNum": liquidity,
                "active": True,
                "closed": False,
            }

        pages = {
            0: [
                market("1", "accept-far-liquid", days_to_end=365, liquidity=3000),
                market("2", "reject-thin-near", days_to_end=5, liquidity=1000),
                market("3", "accept-near-liquid", days_to_end=20, liquidity=2600),
            ],
        }

        with patch.object(gamma, "DEFAULT_MARKETS_PAGE_SIZE", 3):
            with patch.object(gamma, "_fetch_markets_page", side_effect=lambda host, limit, offset: pages.get(offset, [])):
                results = gamma.fetch_markets_with_slice(
                    "https://gamma-api.polymarket.com",
                    limit=3,
                    market_slice=market_slice,
                    now=now,
                )

        self.assertEqual([item["slug"] for item in results], ["accept-far-liquid", "accept-near-liquid"])


if __name__ == "__main__":
    unittest.main()
