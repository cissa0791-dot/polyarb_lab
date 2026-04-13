from __future__ import annotations

import unittest
from unittest.mock import patch

from src.ingest.gamma import fetch_markets, fetch_markets_from_events


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class GammaPaginationTests(unittest.TestCase):
    def test_fetch_markets_uses_offset_pagination_and_deduplicates(self) -> None:
        pages = {
            0: [{"id": idx, "slug": f"market-{idx}"} for idx in range(200)],
            200: (
                [{"id": idx, "slug": f"market-{idx}"} for idx in range(150, 350)]
                + [{"slug": "slug-only-market"}]
            ),
            400: [{"id": idx, "slug": f"market-{idx}"} for idx in range(350, 600)],
        }
        offsets_seen: list[int] = []

        def fake_get(url, params, timeout):
            self.assertEqual(url, "https://gamma-api.polymarket.com/markets")
            self.assertEqual(timeout, 20)
            offsets_seen.append(int(params["offset"]))
            return _FakeResponse(pages.get(int(params["offset"]), []))

        with patch("src.ingest.gamma.httpx.get", side_effect=fake_get):
            markets = fetch_markets("https://gamma-api.polymarket.com", limit=500)

        self.assertEqual(offsets_seen, [0, 200, 400])
        self.assertEqual(len(markets), 500)
        identities = []
        for market in markets:
            if market.get("id") is not None:
                identities.append(f"id:{market['id']}")
            else:
                identities.append(f"slug:{market['slug']}")
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(identities[0], "id:0")
        self.assertEqual(identities[199], "id:199")
        self.assertEqual(identities[200], "id:200")
        self.assertIn("slug:slug-only-market", identities)
        self.assertEqual(identities[-1], "id:498")

    def test_fetch_markets_uses_slug_fallback_and_preserves_first_seen_order(self) -> None:
        pages = {
            0: [{"slug": "alpha"}, {"id": 2, "slug": "beta"}],
            200: [{"slug": "alpha"}, {"slug": "gamma"}, {"id": 2, "slug": "beta-duplicate"}],
        }

        def fake_get(url, params, timeout):
            return _FakeResponse(pages.get(int(params["offset"]), []))

        with patch("src.ingest.gamma.httpx.get", side_effect=fake_get):
            markets = fetch_markets("https://gamma-api.polymarket.com", limit=10)

        self.assertEqual(
            markets,
            [
                {"slug": "alpha"},
                {"id": 2, "slug": "beta"},
            ],
        )

    def test_fetch_markets_returns_empty_for_non_positive_limit(self) -> None:
        with patch("src.ingest.gamma.httpx.get") as mocked_get:
            self.assertEqual(fetch_markets("https://gamma-api.polymarket.com", limit=0), [])
            mocked_get.assert_not_called()

    def test_fetch_markets_from_events_flattens_and_deduplicates(self) -> None:
        pages = {
            0: [
                {
                    "id": "evt-1",
                    "slug": "event-1",
                    "title": "Event 1",
                    "markets": [
                        {"id": "m-1", "slug": "market-1"},
                        {"id": "m-2", "slug": "market-2"},
                    ],
                },
                {
                    "id": "evt-2",
                    "slug": "event-2",
                    "title": "Event 2",
                    "markets": [
                        {"id": "m-2", "slug": "market-2-duplicate"},
                        {"slug": "slug-only-market"},
                    ],
                },
            ],
            100: [],
        }
        offsets_seen: list[int] = []

        def fake_get(url, params, timeout):
            self.assertEqual(url, "https://gamma-api.polymarket.com/events")
            offsets_seen.append(int(params["offset"]))
            return _FakeResponse(pages.get(int(params["offset"]), []))

        with patch("src.ingest.gamma.httpx.get", side_effect=fake_get):
            markets = fetch_markets_from_events("https://gamma-api.polymarket.com", limit=10)

        self.assertEqual(offsets_seen, [0])
        self.assertEqual(
            markets,
            [
                {
                    "id": "m-1",
                    "slug": "market-1",
                    "eventSlug": "event-1",
                    "eventTitle": "Event 1",
                    "eventId": "evt-1",
                },
                {
                    "id": "m-2",
                    "slug": "market-2",
                    "eventSlug": "event-1",
                    "eventTitle": "Event 1",
                    "eventId": "evt-1",
                },
                {
                    "slug": "slug-only-market",
                    "eventSlug": "event-2",
                    "eventTitle": "Event 2",
                    "eventId": "evt-2",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
