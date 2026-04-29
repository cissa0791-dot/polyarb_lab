"""Tests for src/live/auth.py — credential loading and redaction contract.

These tests never touch the network, never construct a real ClobClient,
and never require actual Polymarket credentials.  build_authenticated_client
is excluded from unit tests because it calls the py_clob_client Signer
constructor (which validates the key format); that path belongs in a
separate integration/dry-run test.
"""

from __future__ import annotations

import unittest
from importlib import metadata
from unittest.mock import patch

from src.live.auth import CredentialError, LiveCredentials, assert_clob_v2_available, load_live_credentials


_ALL_VARS = [
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
    "POLYMARKET_CHAIN_ID",
]

# A realistic-looking but entirely fake set of credentials.
_VALID_ENV: dict[str, str] = {
    "POLYMARKET_PRIVATE_KEY":    "0x" + "a" * 64,
    "POLYMARKET_API_KEY":        "test-api-key-abcd",
    "POLYMARKET_API_SECRET":     "test-api-secret-xyz",
    "POLYMARKET_API_PASSPHRASE": "test-passphrase-123",
    "POLYMARKET_CHAIN_ID":       "137",
}


def _set_env(env: dict[str, str]) -> None:
    """Remove all credential vars then set only those in env."""
    import os
    for var in _ALL_VARS:
        os.environ.pop(var, None)
    for var, val in env.items():
        os.environ[var] = val


def _clear_env() -> None:
    import os
    for var in _ALL_VARS:
        os.environ.pop(var, None)


class TestLoadLiveCredentials(unittest.TestCase):

    def setUp(self) -> None:
        _clear_env()

    def tearDown(self) -> None:
        _clear_env()

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_load_returns_credentials_with_correct_values(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertIsInstance(creds, LiveCredentials)
        self.assertEqual(creds.chain_id, 137)
        self.assertEqual(creds.api_key, _VALID_ENV["POLYMARKET_API_KEY"])

    def test_load_mainnet_chain_id(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertEqual(creds.chain_id, 137)

    def test_load_testnet_chain_id(self) -> None:
        env = {**_VALID_ENV, "POLYMARKET_CHAIN_ID": "80002"}
        _set_env(env)
        creds = load_live_credentials()
        self.assertEqual(creds.chain_id, 80002)

    def test_credentials_are_frozen(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        with self.assertRaises((AttributeError, TypeError)):
            creds.api_key = "mutated"  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Missing variables
    # ------------------------------------------------------------------

    def test_raises_when_all_vars_missing(self) -> None:
        _set_env({})
        with self.assertRaises(CredentialError) as ctx:
            load_live_credentials()
        msg = str(ctx.exception)
        for var in _ALL_VARS:
            self.assertIn(var, msg)

    def test_error_message_lists_missing_var_names(self) -> None:
        env = {k: v for k, v in _VALID_ENV.items() if k != "POLYMARKET_API_SECRET"}
        _set_env(env)
        with self.assertRaises(CredentialError) as ctx:
            load_live_credentials()
        self.assertIn("POLYMARKET_API_SECRET", str(ctx.exception))

    def test_raises_on_each_single_missing_var(self) -> None:
        for missing_var in _ALL_VARS:
            with self.subTest(missing=missing_var):
                env = {k: v for k, v in _VALID_ENV.items() if k != missing_var}
                _set_env(env)
                with self.assertRaises(CredentialError):
                    load_live_credentials()

    def test_raises_on_empty_string_value(self) -> None:
        env = {**_VALID_ENV, "POLYMARKET_API_KEY": ""}
        _set_env(env)
        with self.assertRaises(CredentialError) as ctx:
            load_live_credentials()
        self.assertIn("POLYMARKET_API_KEY", str(ctx.exception))

    # ------------------------------------------------------------------
    # chain_id validation
    # ------------------------------------------------------------------

    def test_raises_on_non_integer_chain_id(self) -> None:
        env = {**_VALID_ENV, "POLYMARKET_CHAIN_ID": "mainnet"}
        _set_env(env)
        with self.assertRaises(CredentialError) as ctx:
            load_live_credentials()
        self.assertIn("POLYMARKET_CHAIN_ID", str(ctx.exception))

    def test_raises_on_float_chain_id(self) -> None:
        env = {**_VALID_ENV, "POLYMARKET_CHAIN_ID": "137.0"}
        _set_env(env)
        with self.assertRaises(CredentialError):
            load_live_credentials()

    # ------------------------------------------------------------------
    # Redaction contract — secrets must not appear in repr / str
    # ------------------------------------------------------------------

    def test_repr_does_not_expose_private_key(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertNotIn(_VALID_ENV["POLYMARKET_PRIVATE_KEY"], repr(creds))

    def test_repr_does_not_expose_api_secret(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertNotIn(_VALID_ENV["POLYMARKET_API_SECRET"], repr(creds))

    def test_repr_does_not_expose_api_passphrase(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertNotIn(_VALID_ENV["POLYMARKET_API_PASSPHRASE"], repr(creds))

    def test_str_matches_repr(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertEqual(str(creds), repr(creds))

    def test_repr_contains_chain_id(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        self.assertIn("137", repr(creds))

    def test_repr_contains_api_key_prefix_only(self) -> None:
        _set_env(_VALID_ENV)
        creds = load_live_credentials()
        r = repr(creds)
        # prefix visible, full key not
        self.assertIn("test", r)
        self.assertNotIn(_VALID_ENV["POLYMARKET_API_KEY"], r)

    def test_credential_error_is_exception(self) -> None:
        self.assertTrue(issubclass(CredentialError, Exception))

    def test_clob_v2_guard_reports_install_command(self) -> None:
        with patch("src.live.auth.metadata.version", side_effect=metadata.PackageNotFoundError):
            with self.assertRaises(CredentialError) as ctx:
                assert_clob_v2_available()

        message = str(ctx.exception)
        self.assertIn("py-clob-client-v2", message)
        self.assertIn("order_version_mismatch", message)


if __name__ == "__main__":
    unittest.main()
