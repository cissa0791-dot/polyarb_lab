import os
import json
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
TOKEN_ID = "94192784911459194325909253314484842244405314804074606736702592885535642919725"

PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY") or os.getenv("PRIVATE_KEY")
FUNDER = os.getenv("POLYMARKET_FUNDER")

client = ClobClient(
    host=HOST,
    chain_id=CHAIN_ID,
    key=PRIVATE_KEY,
    signature_type=2,
    funder=FUNDER,
)
client.set_api_creds(client.create_or_derive_api_creds())

checks = [
    ("COLLATERAL", BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)),
    ("CONDITIONAL", BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=TOKEN_ID, signature_type=2)),
]

for name, params in checks:
    print(f"=== {name} ===")
    try:
        client.update_balance_allowance(params=params)
    except Exception as e:
        print("update_balance_allowance_error:", repr(e))
    try:
        res = client.get_balance_allowance(params=params)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print("get_balance_allowance_error:", repr(e))
