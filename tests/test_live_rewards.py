"""Unit tests for src/live/rewards.py.

Tests run entirely offline — no network calls, no credentials required.
Real-network tests live in scripts/verify_api_integration.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.live.rewards import (
    EpochInfo,
    RewardClient,
    RewardClientError,
    UserRewardSummary,
    _DRY_EPOCH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(dry_run: bool = True) -> RewardClient:
    """RewardClient with a fixed wallet address — no private key needed."""
    return RewardClient(
        address="0xDeaDbeefdEAdbeefdEadbEEFdeadbeEFdEaDbeeF",
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Dry-run mode tests
# ---------------------------------------------------------------------------

class TestDryRunMode(unittest.TestCase):

    def test_get_current_epoch_returns_sentinel(self) -> None:
        client = _make_client(dry_run=True)
        epoch = client.get_current_epoch()
        self.assertIsInstance(epoch, EpochInfo)
        self.assertEqual(epoch.epoch_id, "dry_run")

    def test_get_user_rewards_returns_sentinel(self) -> None:
        client = _make_client(dry_run=True)
        summary = client.get_user_rewards()
        self.assertIsInstance(summary, UserRewardSummary)
        self.assertEqual(summary.epoch_id, "dry_run")
        self.assertEqual(summary.rewards_earned_usd, 0.0)

    def test_get_rewards_summary_dry_run_flag_true(self) -> None:
        client = _make_client(dry_run=True)
        result = client.get_rewards_summary()
        self.assertTrue(result["dry_run"])

    def test_get_rewards_summary_has_expected_keys(self) -> None:
        client = _make_client(dry_run=True)
        result = client.get_rewards_summary()
        for key in (
            "wallet_address", "epoch_id", "epoch_start", "epoch_end",
            "epoch_total_pool_usd", "user_earned_usd", "user_maker_vol_usd",
            "user_share_pct", "dry_run",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_dry_run_makes_no_network_calls(self) -> None:
        client = _make_client(dry_run=True)
        with patch("httpx.get") as mock_get:
            client.get_current_epoch()
            client.get_user_rewards()
            client.get_rewards_summary()
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# from_credentials constructor
# ---------------------------------------------------------------------------

class TestFromCredentials(unittest.TestCase):

    def _make_creds(self) -> MagicMock:
        creds = MagicMock()
        # Valid 32-byte private key (test only — not a real key).
        creds.private_key = "0x" + "aa" * 32
        return creds

    def test_derives_address_from_private_key(self) -> None:
        creds = self._make_creds()
        client = RewardClient.from_credentials(creds, dry_run=True)
        self.assertTrue(client.address.startswith("0x"))
        self.assertEqual(len(client.address), 42)

    def test_private_key_without_0x_prefix_accepted(self) -> None:
        creds = self._make_creds()
        creds.private_key = "aa" * 32   # no 0x prefix
        client = RewardClient.from_credentials(creds, dry_run=True)
        self.assertTrue(client.address.startswith("0x"))

    def test_invalid_private_key_raises_reward_client_error(self) -> None:
        creds = self._make_creds()
        creds.private_key = "not_a_valid_key"
        with self.assertRaises(RewardClientError):
            RewardClient.from_credentials(creds, dry_run=True)


# ---------------------------------------------------------------------------
# Live mode — network calls mocked
# ---------------------------------------------------------------------------

class TestLiveEpochFetch(unittest.TestCase):

    def _mock_epoch_response(self) -> dict:
        return {
            "epochNumber": "42",
            "startDate": "2024-01-01T00:00:00Z",
            "endDate": "2024-01-07T00:00:00Z",
            "totalRewards": "50000.0",
        }

    def test_get_current_epoch_parses_response(self) -> None:
        client = _make_client(dry_run=False)
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_epoch_response()
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_resp):
            epoch = client.get_current_epoch()

        self.assertEqual(epoch.epoch_id, "42")
        self.assertEqual(epoch.start_date, "2024-01-01T00:00:00Z")
        self.assertEqual(epoch.total_rewards_usd, 50000.0)

    def test_get_current_epoch_nested_data_key(self) -> None:
        """API may wrap payload under 'data'."""
        client = _make_client(dry_run=False)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": self._mock_epoch_response()}
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.get", return_value=mock_resp):
            epoch = client.get_current_epoch()

        self.assertEqual(epoch.epoch_id, "42")

    def test_get_current_epoch_network_error_raises(self) -> None:
        client = _make_client(dry_run=False)
        with patch("httpx.get", side_effect=Exception("connection refused")):
            with self.assertRaises(RewardClientError):
                client.get_current_epoch()


class TestLiveUserRewards(unittest.TestCase):

    def _mock_epoch_resp(self) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {
            "epochNumber": "42",
            "startDate": "2024-01-01T00:00:00Z",
            "endDate": "2024-01-07T00:00:00Z",
            "totalRewards": "50000.0",
        }
        resp.raise_for_status.return_value = None
        resp.status_code = 200
        return resp

    def _mock_user_resp(self) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {
            "rewardsEarned": "12.50",
            "makerVolume": "3200.00",
            "rewardShare": "0.025",
        }
        resp.raise_for_status.return_value = None
        resp.status_code = 200
        return resp

    def test_get_user_rewards_parses_response(self) -> None:
        client = _make_client(dry_run=False)
        with patch("httpx.get", side_effect=[
            self._mock_epoch_resp(), self._mock_user_resp()
        ]):
            summary = client.get_user_rewards()

        self.assertAlmostEqual(summary.rewards_earned_usd, 12.5)
        self.assertAlmostEqual(summary.maker_volume_usd, 3200.0)
        self.assertAlmostEqual(summary.reward_share_pct, 0.025)

    def test_get_user_rewards_with_explicit_epoch_skips_epoch_fetch(self) -> None:
        client = _make_client(dry_run=False)
        with patch("httpx.get", return_value=self._mock_user_resp()) as mock_get:
            summary = client.get_user_rewards(epoch_id="42")

        # Should call exactly once (user stats only).
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(summary.epoch_id, "42")

    def test_get_user_rewards_falls_back_on_404(self) -> None:
        client = _make_client(dry_run=False)

        epoch_resp = self._mock_epoch_resp()

        # First user-rewards call returns 404; fallback should be attempted.
        not_found = MagicMock()
        not_found.status_code = 404
        not_found.raise_for_status.return_value = None

        fallback_resp = self._mock_user_resp()

        with patch("httpx.get", side_effect=[epoch_resp, not_found, fallback_resp]):
            summary = client.get_user_rewards()

        self.assertIsInstance(summary, UserRewardSummary)

    def test_get_rewards_summary_combines_epoch_and_user(self) -> None:
        client = _make_client(dry_run=False)
        with patch("httpx.get", side_effect=[
            self._mock_epoch_resp(), self._mock_user_resp()
        ]):
            result = client.get_rewards_summary()

        self.assertEqual(result["epoch_id"], "42")
        self.assertAlmostEqual(result["user_earned_usd"], 12.5)
        self.assertFalse(result["dry_run"])


# ---------------------------------------------------------------------------
# Parse helpers — tolerant of missing/alternate field names
# ---------------------------------------------------------------------------

class TestParseEpoch(unittest.TestCase):

    def test_epoch_id_fallback_fields(self) -> None:
        client = _make_client()
        epoch = client._parse_epoch({"id": "99", "startDate": "", "endDate": ""})
        self.assertEqual(epoch.epoch_id, "99")

    def test_epoch_id_ultimate_fallback(self) -> None:
        client = _make_client()
        epoch = client._parse_epoch({})
        self.assertEqual(epoch.epoch_id, "unknown")

    def test_total_rewards_alternate_field(self) -> None:
        client = _make_client()
        epoch = client._parse_epoch({"epochNumber": "1", "total_rewards": "999"})
        self.assertAlmostEqual(epoch.total_rewards_usd, 999.0)


class TestParseUserSummary(unittest.TestCase):

    def test_alternate_field_names(self) -> None:
        client = _make_client()
        summary = client._parse_user_summary(
            {"rewards_earned": "5.5", "maker_volume": "100.0", "reward_share": "0.1"},
            epoch_id="1",
        )
        self.assertAlmostEqual(summary.rewards_earned_usd, 5.5)

    def test_missing_fields_default_to_zero(self) -> None:
        client = _make_client()
        summary = client._parse_user_summary({}, epoch_id="1")
        self.assertEqual(summary.rewards_earned_usd, 0.0)
        self.assertEqual(summary.maker_volume_usd, 0.0)

    def test_address_preserved(self) -> None:
        client = _make_client()
        summary = client._parse_user_summary({}, epoch_id="1")
        self.assertEqual(summary.address, client.address)


if __name__ == "__main__":
    unittest.main()
