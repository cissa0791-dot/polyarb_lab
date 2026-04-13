import os
import sys
sys.path.insert(0, ".")
from py_clob_client.client import ClobClient

pk = os.environ["POLYMARKET_PRIVATE_KEY"]
cfg_key = os.environ.get("POLYMARKET_API_KEY", "")
cfg_sec = os.environ.get("POLYMARKET_API_SECRET", "")
cfg_pp = os.environ.get("POLYMARKET_API_PASSPHRASE", "")

c = ClobClient(host="https://clob.polymarket.com", chain_id=137, key=pk)
d = c.derive_api_key()

print("key_match=" + str(d.api_key == cfg_key))
print("secret_match=" + str(d.api_secret == cfg_sec))
print("passphrase_match=" + str(d.api_passphrase == cfg_pp))
print("sig_type=" + os.environ.get("POLYMARKET_SIGNATURE_TYPE", ""))
print("funder=" + os.environ.get("POLYMARKET_FUNDER", ""))
print("chain_id=" + os.environ.get("POLYMARKET_CHAIN_ID", ""))