"""
dome_market_lookup — external observability sidecar
polyarb_lab / src / external

Read-only market metadata lookup via Dome API.
Used for token alignment verification only.
Does NOT participate in order placement, cancellation, reward logic,
scoring logic, or any live trading path.

Interface:
    lookup(market_slug: str) -> dict

Returns a plain dict:
    {
        "title":        str | None,
        "market_slug":  str | None,
        "condition_id": str | None,
        "side_a_id":    str | None,   # YES token ID
        "side_b_id":    str | None,   # NO token ID
        "status":       str | None,
        "source":       "sdk" | "requests" | "error",
        "error":        str | None,   # set on failure, None on success
    }

Resolution order:
    1. dome_api_sdk.DomeClient  (if installed)
    2. requests fallback        (always available)
    3. error dict               (never raises)

Requires env var:
    DOME_API_KEY — Dome API bearer token

Usage:
    from src.external.dome_market_lookup import lookup
    meta = lookup("will-the-next-prime-minister-of-hungary-be-pter-magyar")
    print(meta["condition_id"])
"""
from __future__ import annotations

import os
from typing import Optional


DOME_BASE_URL = "https://api.domeapi.io/v1/polymarket/markets"

_EMPTY: dict = {
    "title":        None,
    "market_slug":  None,
    "condition_id": None,
    "side_a_id":    None,
    "side_b_id":    None,
    "status":       None,
    "source":       "error",
    "error":        None,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(obj) -> Optional[dict]:
    """Convert any SDK model / dict / object to a plain dict. Never raises."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {"raw_repr": repr(obj)}


def _extract(market_dict: dict) -> dict:
    """Pull the five target fields out of a normalized market dict."""
    side_a = _normalize(market_dict.get("side_a")) or {}
    side_b = _normalize(market_dict.get("side_b")) or {}
    return {
        "title":        market_dict.get("title"),
        "market_slug":  market_dict.get("market_slug"),
        "condition_id": market_dict.get("condition_id"),
        "side_a_id":    side_a.get("id"),
        "side_b_id":    side_b.get("id"),
        "status":       market_dict.get("status"),
        "source":       "unknown",
        "error":        None,
    }


# ---------------------------------------------------------------------------
# SDK path
# ---------------------------------------------------------------------------

def _lookup_sdk(api_key: str, market_slug: str) -> Optional[dict]:
    """
    Attempt lookup via dome_api_sdk.DomeClient.
    Returns extracted dict on success, None if SDK unavailable or call fails.
    """
    try:
        from dome_api_sdk import DomeClient
    except ImportError:
        return None

    try:
        dome = DomeClient({"api_key": api_key})
        resp = dome.polymarket.markets.get_markets({
            "market_slug": [market_slug],
            "limit": 1,
        })

        if hasattr(resp, "markets"):
            market_list = resp.markets
        else:
            resp_dict = _normalize(resp) or {}
            market_list = resp_dict.get("markets") or []

        if not market_list:
            return None

        first = _normalize(market_list[0])
        if not first:
            return None

        result = _extract(first)
        result["source"] = "sdk"
        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Requests fallback
# ---------------------------------------------------------------------------

def _lookup_requests(api_key: str, market_slug: str) -> Optional[dict]:
    """
    Attempt lookup via raw HTTP requests.
    Returns extracted dict on success, None on any failure.
    """
    try:
        import requests as _requests
        headers = {"Authorization": f"Bearer {api_key}"}
        params  = {"market_slug": market_slug, "limit": 1}
        r = _requests.get(DOME_BASE_URL, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        body = r.json()

        # Unwrap response envelope — several possible shapes
        if isinstance(body, dict):
            if isinstance(body.get("markets"), list):
                market_list = body["markets"]
            elif isinstance(body.get("data"), dict) and isinstance(
                body["data"].get("markets"), list
            ):
                market_list = body["data"]["markets"]
            elif isinstance(body.get("data"), list):
                market_list = body["data"]
            elif isinstance(body.get("results"), list):
                market_list = body["results"]
            elif isinstance(body.get("market_slug"), str):
                market_list = [body]
            else:
                market_list = []
        elif isinstance(body, list):
            market_list = body
        else:
            market_list = []

        if not market_list:
            return None

        result = _extract(market_list[0])
        result["source"] = "requests"
        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def lookup(market_slug: str) -> dict:
    """
    Look up market metadata from Dome for the given market_slug.

    Returns a plain dict with title / market_slug / condition_id /
    side_a_id / side_b_id / status / source / error.

    Never raises.  On any failure, returns the _EMPTY template with
    source="error" and error=<message>.

    Requires DOME_API_KEY in env.  Returns error dict immediately if absent.
    """
    api_key = os.environ.get("DOME_API_KEY")
    if not api_key:
        result = dict(_EMPTY)
        result["error"] = "DOME_API_KEY not set in env"
        return result

    # SDK path first (richer type information)
    result = _lookup_sdk(api_key, market_slug)
    if result is not None:
        return result

    # Requests fallback
    result = _lookup_requests(api_key, market_slug)
    if result is not None:
        return result

    # Both paths failed
    out = dict(_EMPTY)
    out["error"] = "both SDK and requests paths failed for slug=" + market_slug
    return out
