"""Live-execution credential scaffold.

Loads the five secrets required for authenticated access to the Polymarket
CLOB from environment variables only.  Secrets are never written to logs,
repr strings, or config files.

Environment variables required before live execution can be enabled:

    POLYMARKET_PRIVATE_KEY      EVM private key for signing orders (hex, with
                                or without leading 0x)
    POLYMARKET_API_KEY          Polymarket CLOB Level-2 API key
    POLYMARKET_API_SECRET       Polymarket CLOB Level-2 API secret
    POLYMARKET_API_PASSPHRASE   Polymarket CLOB Level-2 API passphrase
    POLYMARKET_CHAIN_ID         Integer chain ID  (137 = Polygon mainnet,
                                80002 = Amoy testnet)

Usage::

    from src.live.auth import load_live_credentials, build_authenticated_client
    creds = load_live_credentials()            # raises CredentialError if any var is missing
    client = build_authenticated_client(creds, host="https://clob.polymarket.com")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata

from src.live.clob_compat import ApiCreds, ClobClient


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class CredentialError(Exception):
    """Raised when required live-execution credentials are absent or invalid."""


def assert_clob_v2_available() -> None:
    """Fail fast when live trading would use the pre-V2 CLOB SDK.

    Polymarket CLOB V2 uses a new EIP-712 exchange domain.  The old
    ``py-clob-client`` package can still authenticate Level-2 requests, but
    orders signed with it are rejected by the matching engine with
    ``order_version_mismatch``.  The V2 wheel uses the ``py_clob_client_v2``
    import namespace, so package metadata is the reliable runtime guard.
    """
    try:
        metadata.version("py-clob-client-v2")
    except metadata.PackageNotFoundError as exc:
        raise CredentialError(
            "Polymarket live trading now requires py-clob-client-v2>=1.0.0. "
            "Install it on the VPS with:\n"
            "  python -m pip install -U py-clob-client-v2\n\n"
            "The old py-clob-client package can pass auth but will reject live "
            "orders with order_version_mismatch."
        ) from exc


# ---------------------------------------------------------------------------
# Credential descriptor table
# ---------------------------------------------------------------------------

_REQUIRED_ENV_VARS: dict[str, str] = {
    "POLYMARKET_PRIVATE_KEY":    "EVM private key for signing orders",
    "POLYMARKET_API_KEY":        "Polymarket CLOB Level-2 API key",
    "POLYMARKET_API_SECRET":     "Polymarket CLOB Level-2 API secret",
    "POLYMARKET_API_PASSPHRASE": "Polymarket CLOB Level-2 API passphrase",
    "POLYMARKET_CHAIN_ID":       "EVM chain ID (137 = Polygon mainnet, 80002 = Amoy testnet)",
}


# ---------------------------------------------------------------------------
# Credential container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class LiveCredentials:
    """Immutable holder for live-execution credentials.

    repr and str are intentionally redacted — only the first four characters
    of api_key and the chain_id are visible.  private_key, api_secret, and
    api_passphrase are never exposed.
    """

    private_key: str
    api_key: str
    api_secret: str
    api_passphrase: str
    chain_id: int

    def __repr__(self) -> str:
        prefix = self.api_key[:4] if len(self.api_key) >= 4 else self.api_key
        return f"LiveCredentials(api_key={prefix}****, chain_id={self.chain_id})"

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_live_credentials() -> LiveCredentials:
    """Load and validate live-execution credentials from environment variables.

    Raises:
        CredentialError: if any required variable is absent or if
                         POLYMARKET_CHAIN_ID is not a valid integer.

    Returns:
        LiveCredentials: validated, immutable credential object.
    """
    missing = [
        f"  {var}  ({desc})"
        for var, desc in _REQUIRED_ENV_VARS.items()
        if not os.environ.get(var)
    ]
    if missing:
        raise CredentialError(
            "Missing required live-execution environment variables:\n"
            + "\n".join(missing)
            + "\n\nSet these variables before enabling live execution."
        )

    raw_chain_id = os.environ["POLYMARKET_CHAIN_ID"]
    try:
        chain_id = int(raw_chain_id)
    except ValueError:
        raise CredentialError(
            f"POLYMARKET_CHAIN_ID must be an integer, got: {raw_chain_id!r}"
        )

    return LiveCredentials(
        private_key=os.environ["POLYMARKET_PRIVATE_KEY"],
        api_key=os.environ["POLYMARKET_API_KEY"],
        api_secret=os.environ["POLYMARKET_API_SECRET"],
        api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
        chain_id=chain_id,
    )


def build_authenticated_client(
    creds: LiveCredentials,
    host: str,
    *,
    signature_type: int | None = None,
    funder: str | None = None,
) -> ClobClient:
    """Construct a fully-authenticated Level-2 ClobClient from credentials.

    Level-2 provides access to all CLOB endpoints including order placement
    and cancellation.  This function does not perform any network call;
    authentication headers are attached per-request by the client library.

    Args:
        creds:          validated LiveCredentials from load_live_credentials().
        host:           CLOB base URL, e.g. "https://clob.polymarket.com".
        signature_type: EIP-712 signature scheme for order signing.
                        0 = EOA (raw wallet, default when None).
                        1 = POLY_PROXY (Polymarket proxy wallet — most web users).
                        2 = POLY_GNOSIS_SAFE (Gnosis Safe multisig).
                        If None, py_clob_client defaults to EOA (0).
        funder:         Address that holds the funds (maker address in orders).
                        For proxy / Gnosis wallets this is the smart-contract
                        wallet address shown on polymarket.com, NOT the EOA.
                        If None, defaults to the EOA address of the private key.

    Returns:
        Authenticated ClobClient ready for order submission.
    """
    assert_clob_v2_available()
    return ClobClient(
        host=host,
        chain_id=creds.chain_id,
        key=creds.private_key,
        creds=ApiCreds(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
        ),
        signature_type=signature_type,
        funder=funder,
    )
