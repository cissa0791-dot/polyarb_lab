import httpx
import json

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MIN_EDGE = 0.03
TARGET_USD = 10.0

resp = httpx.get(
    f"{GAMMA}/markets",
    params={
        "limit": 100,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
    },
    timeout=20,
)
markets = resp.json()
if not isinstance(markets, list):
    markets = markets.get("markets", [])

hits = []

for m in markets:
    raw_ids = m.get("clobTokenIds") or []
    if isinstance(raw_ids, str):
        try:
            raw_ids = json.loads(raw_ids)
        except Exception:
            continue

    if len(raw_ids) < 2:
        continue

    slug = str(m.get("slug") or "?")
    min_size = float(m.get("orderMinSize") or 5)
    yes_id, no_id = str(raw_ids[0]), str(raw_ids[1])

    try:
        yb = httpx.get(f"{CLOB}/book", params={"token_id": yes_id}, timeout=5).json()
        nb = httpx.get(f"{CLOB}/book", params={"token_id": no_id}, timeout=5).json()
    except Exception:
        continue

    ya = yb.get("asks") or []
    na = nb.get("asks") or []
    if not ya or not na:
        continue

    yes_ask = float(ya[0]["price"])
    no_ask = float(na[0]["price"])

    if yes_ask >= 0.98 and no_ask >= 0.98:
        continue

    edge = 1.0 - yes_ask - no_ask
    if edge >= MIN_EDGE:
        buy_side = "YES" if yes_ask <= no_ask else "NO"
        buy_token = yes_id if buy_side == "YES" else no_id
        buy_ask = yes_ask if buy_side == "YES" else no_ask
        shares = max(int(min_size), int(TARGET_USD / buy_ask))

        hits.append((edge, slug, buy_side, buy_token, buy_ask, shares, min_size))
        print(
            f"HIT  {slug[:55]}  YES={yes_ask} NO={no_ask}  "
            f"edge={edge*100:.2f}c  buy={buy_side}@{buy_ask}  "
            f"{shares}sh=${shares*buy_ask:.2f}"
        )

if not hits:
    print("No qualifying entry found.")
else:
    best = sorted(hits, reverse=True)[0]
    e, s, side, tok, ask, sh, ms = best
    print("\nBEST ENTRY:")
    print(f"  --token {tok}")
    print(f"  --ask {ask}")
    print(f"  --min-size {sh}")
    print(f"  --market-slug {s}")