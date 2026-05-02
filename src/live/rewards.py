"""Polymarket liquidity reward retrieval client.

Fetches reward epoch metadata (public) and user-specific reward stats
(requires wallet address derived from the live private key).

The wallet address is derived from POLYMARKET_PRIVATE_KEY using eth_account,
which is already a transitive dependency of py_clob_client.

Two modes:
  dry_run=True   Returns placeholder data; no network calls made.
  dry_run=False  Makes real HTTP requests to the Polymarket CLOB rewards API.

Usage::

    from src.live.auth import load_live_credentials
    from src.live.rewards import RewardClient

    creds = load_live_credentials()
    client = RewardClient.from_credentials(creds, dry_run=False)
    epoch   = client.get_current_epoch()
    summary = client.get_user_rewards()
    print(summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from eth_account import Account

from src.live.auth import LiveCredentials

logger = logging.getLogger("polyarb.live.rewards")

_CLOB_HOST = "https://clob.polymarket.com"
_DATA_API_HOST = "https://data-api.polymarket.com"
_TIMEOUT   = 15.0   # seconds


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class RewardClientError(Exception):
    """Raised when a reward API call fails."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpochInfo:
    """Current reward epoch details.

    Attributes:
        epoch_id:      Epoch identifier (integer or string from API).
        start_date:    ISO-8601 start timestamp string.
        end_date:      ISO-8601 end timestamp string.
        total_rewards: Total USDC rewards for the epoch (may be 0 if unknown).
        raw:           Full raw API payload for inspection.
    """
    epoch_id: str
    start_date: str
    end_date: str
    total_rewards_usd: float
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class UserRewardSummary:
    """User-specific reward stats for the current epoch.

    Attributes:
        address:          Wallet address queried.
        epoch_id:         Epoch this summary belongs to.
        rewards_earned:   USDC earned so far this epoch.
        maker_volume_usd: Maker volume contributing to rewards.
        reward_share_pct: User's share of the epoch pool (0–100).
        raw:              Full raw API payload for inspection.
    """
    address: str
    epoch_id: str
    rewards_earned_usd: float
    maker_volume_usd: float
    reward_share_pct: float
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------------------
# Dry-run sentinels
# ---------------------------------------------------------------------------

_DRY_EPOCH = EpochInfo(
    epoch_id="dry_run",
    start_date="N/A",
    end_date="N/A",
    total_rewards_usd=0.0,
)

def _dry_user_summary(address: str) -> UserRewardSummary:
    return UserRewardSummary(
        address=address,
        epoch_id="dry_run",
        rewards_earned_usd=0.0,
        maker_volume_usd=0.0,
        reward_share_pct=0.0,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RewardClient:
    """Fetches Polymarket liquidity reward data.

    Construct via from_credentials (production) or directly (testing).

    Attributes:
        address:  Wallet address derived from the private key.
        dry_run:  When True all methods return placeholder data.
    """

    def __init__(
        self,
        address: str,
        *,
        host: str = _CLOB_HOST,
        dry_run: bool = True,
    ) -> None:
        self.address = address
        self._host   = host.rstrip("/")
        self.dry_run = dry_run

    @classmethod
    def from_credentials(
        cls,
        creds: LiveCredentials,
        *,
        host: str = _CLOB_HOST,
        dry_run: bool = True,
    ) -> "RewardClient":
        """Construct from validated LiveCredentials.

        Derives the wallet address from the private key (no network call).

        Args:
            creds:   credentials from load_live_credentials().
            host:    CLOB base URL override (default: production).
            dry_run: when True (default), no network calls are made.
        """
        pk = creds.private_key
        if not pk.startswith("0x"):
            pk = "0x" + pk
        try:
            account = Account.from_key(pk)
        except Exception as exc:
            raise RewardClientError(
                f"Cannot derive wallet address from POLYMARKET_PRIVATE_KEY: {exc}"
            ) from exc
        return cls(account.address, host=host, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_epoch(self) -> EpochInfo:
        """Fetch the current reward epoch from the CLOB API.

        Returns:
            EpochInfo with epoch dates and total reward pool size.

        Raises:
            RewardClientError: on network or parse error.
        """
        if self.dry_run:
            logger.debug("get_current_epoch: dry_run mode, returning sentinel")
            return _DRY_EPOCH

        url = f"{self._host}/rewards/epoch"
        try:
            resp = httpx.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 405:
                raise RewardClientError(
                    f"get_current_epoch HTTP 405: rewards endpoint unavailable at {url}"
                ) from exc
            raise RewardClientError(
                f"get_current_epoch HTTP {status}: {url}"
            ) from exc
        except Exception as exc:
            raise RewardClientError(f"get_current_epoch failed: {exc}") from exc

        return self._parse_epoch(data)

    def get_user_rewards(self, epoch_id: str | None = None) -> UserRewardSummary:
        """Fetch this user's reward stats for a given epoch.

        If epoch_id is None, queries the current epoch first.

        Args:
            epoch_id: Specific epoch to query. If None, uses current epoch.

        Returns:
            UserRewardSummary with earned rewards and maker volume.

        Raises:
            RewardClientError: on network or parse error.
        """
        if self.dry_run:
            logger.debug("get_user_rewards: dry_run mode, returning sentinel")
            return _dry_user_summary(self.address)

        # Resolve epoch if not provided.
        resolved_epoch = epoch_id
        if resolved_epoch is None:
            epoch_info = self.get_current_epoch()
            resolved_epoch = epoch_info.epoch_id

        # Try the CLOB rewards endpoint with the user's address.
        # Polymarket exposes per-address reward stats publicly (no auth required).
        url = f"{self._host}/rewards/epoch/{resolved_epoch}/stats"
        params: dict[str, str] = {"address": self.address}

        try:
            resp = httpx.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 404:
                # Endpoint not found for this epoch — try alternate path.
                return self._fetch_rewards_alternate(resolved_epoch)
            resp.raise_for_status()
            data = resp.json()
        except RewardClientError:
            raise
        except httpx.HTTPStatusError as exc:
            raise RewardClientError(
                f"get_user_rewards HTTP {exc.response.status_code} for "
                f"address={self.address} epoch={resolved_epoch}"
            ) from exc
        except Exception as exc:
            raise RewardClientError(f"get_user_rewards failed: {exc}") from exc

        return self._parse_user_summary(data, resolved_epoch)

    def get_rewards_summary(self) -> dict[str, Any]:
        """High-level combined summary: epoch info + user stats.

        Returns a plain dict suitable for logging or display.

        Raises:
            RewardClientError: if either underlying call fails.
        """
        try:
            epoch = self.get_current_epoch()
            summary = self.get_user_rewards(epoch_id=epoch.epoch_id)
            source = "clob_rewards"
            clob_error = None
        except RewardClientError as exc:
            clob_error = str(exc)
            try:
                summary = self._fetch_activity_yield_summary()
            except RewardClientError as fallback_exc:
                raise RewardClientError(
                    f"get_rewards_summary failed via clob ({clob_error}) and activity fallback ({fallback_exc})"
                ) from fallback_exc
            epoch = EpochInfo(
                epoch_id=summary.epoch_id,
                start_date="N/A",
                end_date="N/A",
                total_rewards_usd=0.0,
                raw=summary.raw,
            )
            source = "data_api_activity_yield"

        result = {
            "wallet_address":      self.address,
            "epoch_id":            epoch.epoch_id,
            "epoch_start":         epoch.start_date,
            "epoch_end":           epoch.end_date,
            "epoch_total_pool_usd": epoch.total_rewards_usd,
            "user_earned_usd":     summary.rewards_earned_usd,
            "user_maker_vol_usd":  summary.maker_volume_usd,
            "user_share_pct":      summary.reward_share_pct,
            "dry_run":             self.dry_run,
            "source":              source,
        }
        if clob_error:
            result["clob_rewards_error"] = clob_error
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_activity_yield_summary(self, *, limit: int = 500, max_pages: int = 10) -> UserRewardSummary:
        """Fallback actual reward source using data-api activity YIELD rows.

        The CLOB rewards epoch endpoint has returned HTTP 405 in production.
        Activity YIELD rows are cumulative enough for session delta tracking:
        the session stores a baseline on first read and only attributes later
        increases as actual reward.
        """
        rows: list[dict[str, Any]] = []
        for page in range(max(1, max_pages)):
            params = {"user": self.address, "limit": str(limit), "offset": str(page * limit)}
            url = f"{_DATA_API_HOST}/activity"
            try:
                resp = httpx.get(url, params=params, timeout=_TIMEOUT)
                resp.raise_for_status()
                payload = resp.json()
            except httpx.HTTPStatusError as exc:
                raise RewardClientError(
                    f"_fetch_activity_yield_summary HTTP {exc.response.status_code}"
                ) from exc
            except Exception as exc:
                raise RewardClientError(f"_fetch_activity_yield_summary failed: {exc}") from exc

            page_rows = payload
            if isinstance(payload, dict):
                page_rows = payload.get("data") or payload.get("activity") or []
            if not isinstance(page_rows, list) or not page_rows:
                break
            rows.extend(dict(row) for row in page_rows if isinstance(row, dict))
            if len(page_rows) < limit:
                break

        yield_rows = [row for row in rows if str(row.get("type") or "").upper() == "YIELD"]
        earned = 0.0
        for row in yield_rows:
            try:
                earned += float(row.get("usdcSize") or row.get("size") or 0.0)
            except (TypeError, ValueError):
                continue
        return UserRewardSummary(
            address=self.address,
            epoch_id="data_api_activity_yield",
            rewards_earned_usd=earned,
            maker_volume_usd=0.0,
            reward_share_pct=0.0,
            raw={
                "source": "data_api_activity_yield",
                "activity_rows_scanned": len(rows),
                "yield_rows": len(yield_rows),
                "sample_yield_rows": yield_rows[:20],
            },
        )

    def _fetch_rewards_alternate(self, epoch_id: str) -> UserRewardSummary:
        """Fallback: query top-level /rewards with address filter."""
        url = f"{self._host}/rewards"
        params: dict[str, str] = {"address": self.address}
        try:
            resp = httpx.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RewardClientError(
                f"_fetch_rewards_alternate HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise RewardClientError(f"_fetch_rewards_alternate failed: {exc}") from exc

        # data may be a list of reward entries or a single dict.
        if isinstance(data, list):
            data = data[0] if data else {}

        return self._parse_user_summary(data, epoch_id)

    @staticmethod
    def _parse_epoch(data: dict[str, Any]) -> EpochInfo:
        """Parse epoch payload; tolerant of missing fields."""
        # Epoch API may nest under 'data' or return flat.
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # epoch_id: string/int field — `or`-chaining here is safe (we want the
        # first truthy non-empty string; 0 is not a valid epoch id).
        epoch_id = str(
            data.get("epochNumber") or data.get("epoch_id") or data.get("id") or "unknown"
        )
        # total_rewards_usd: numeric — use key-presence to preserve 0.0.
        def _float_field(d: dict, *keys: str) -> float:
            for k in keys:
                if k in d:
                    return float(d[k])
            return 0.0

        return EpochInfo(
            epoch_id=epoch_id,
            start_date=str(data.get("startDate") or data.get("start_date") or ""),
            end_date=str(data.get("endDate") or data.get("end_date") or ""),
            total_rewards_usd=_float_field(data, "totalRewards", "total_rewards"),
            raw=data,
        )

    def _parse_user_summary(
        self, data: dict[str, Any], epoch_id: str
    ) -> UserRewardSummary:
        """Parse user reward payload; tolerant of missing fields."""
        if "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # Key-presence extraction: do NOT use `or`-chaining — 0.0 is a valid
        # reward value and `0 or fallback` coerces it to the fallback.
        def _float_field(d: dict, *keys: str) -> float:
            for k in keys:
                if k in d:
                    return float(d[k])
            return 0.0

        return UserRewardSummary(
            address=self.address,
            epoch_id=epoch_id,
            rewards_earned_usd=_float_field(data, "rewardsEarned", "rewards_earned"),
            maker_volume_usd=_float_field(data, "makerVolume", "maker_volume"),
            reward_share_pct=_float_field(data, "rewardShare", "reward_share"),
            raw=data,
        )
