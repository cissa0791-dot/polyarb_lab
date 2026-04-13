import os
import json
import requests

SLUG = "will-gavin-newsom-win-the-2028-us-presidential-election"
BASE_URL = "https://api.domeapi.io/v1/polymarket/markets"


def print_first_market(obj, label):
    markets = None

    if isinstance(obj, dict):
        if "markets" in obj and isinstance(obj["markets"], list):
            markets = obj["markets"]
        elif isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("markets"), list):
            markets = obj["data"]["markets"]
        elif isinstance(obj.get("data"), list):
            markets = obj["data"]
        elif isinstance(obj.get("results"), list):
            markets = obj["results"]
        elif isinstance(obj.get("items"), list):
            markets = obj["items"]
        elif isinstance(obj.get("market_slug"), str):
            markets = [obj]

    if markets is None:
        print(f"[{label}] could not find markets list in response")
        print(json.dumps(obj, indent=2)[:2000])
        return

    print(f"[{label}] market_count = {len(markets)}")
    if not markets:
        return

    m = markets[0]
    side_a = m.get("side_a") or {}
    side_b = m.get("side_b") or {}

    print(f"[{label}] title        = {m.get('title')}")
    print(f"[{label}] market_slug  = {m.get('market_slug')}")
    print(f"[{label}] condition_id = {m.get('condition_id')}")
    print(f"[{label}] side_a.id    = {side_a.get('id')}")
    print(f"[{label}] side_b.id    = {side_b.get('id')}")
    print(f"[{label}] status       = {m.get('status')}")


def test_requests():
    api_key = os.environ["DOME_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"market_slug": SLUG, "limit": 10}

    r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    print("[requests] status =", r.status_code)
    r.raise_for_status()
    data = r.json()
    print_first_market(data, "requests")


def normalize_obj(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
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


def test_sdk():
    try:
        from dome_api_sdk import DomeClient
    except Exception as e:
        print("[sdk] dome-api-sdk not installed or import failed")
        print("[sdk] install command: pip install dome-api-sdk")
        print("[sdk] import error:", repr(e))
        return

    api_key = os.environ["DOME_API_KEY"]
    dome = DomeClient({"api_key": api_key})

    resp = dome.polymarket.markets.get_markets({
        "market_slug": ["will-gavin-newsom-win-the-2028-us-presidential-election"],
        "limit": 10
    })

    print("[sdk] raw_type =", type(resp).__name__)

    if hasattr(resp, "markets"):
        market_list = resp.markets
    else:
        resp_dict = normalize_obj(resp)
        market_list = resp_dict.get("markets", [])

    print(f"[sdk] market_count = {len(market_list)}")
    if not market_list:
        print("[sdk] empty markets list")
        print("[sdk] raw =", normalize_obj(resp))
        return

    first_market = normalize_obj(market_list[0])
    print("[sdk] first_market_raw =", first_market)

    side_a = normalize_obj(first_market.get("side_a")) if isinstance(first_market, dict) else {}
    side_b = normalize_obj(first_market.get("side_b")) if isinstance(first_market, dict) else {}

    print(f"[sdk] title        = {first_market.get('title')}")
    print(f"[sdk] market_slug  = {first_market.get('market_slug')}")
    print(f"[sdk] condition_id = {first_market.get('condition_id')}")
    print(f"[sdk] side_a.id    = {side_a.get('id') if isinstance(side_a, dict) else None}")
    print(f"[sdk] side_b.id    = {side_b.get('id') if isinstance(side_b, dict) else None}")
    print(f"[sdk] status       = {first_market.get('status')}")


def main():
    if "DOME_API_KEY" not in os.environ:
        raise RuntimeError("Missing DOME_API_KEY environment variable")

    print("=== Dome market lookup test ===")
    print("slug =", SLUG)

    try:
        test_requests()
    except Exception as e:
        print("[requests] FAILED:", repr(e))

    print()

    try:
        test_sdk()
    except Exception as e:
        print("[sdk] FAILED:", repr(e))


if __name__ == "__main__":
    main()
