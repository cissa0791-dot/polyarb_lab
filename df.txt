import os
import sys
sys.path.insert(0, ".")
from py_clob_client.client import ClobClient

pk = os.environ["POLYMARKET_PRIVATE_KEY"]
c = ClobClient(host="https://clob.polymarket.com", chain_id=137, key=pk)
d = c.derive_api_key()

print("DERIVED_API_KEY=" + d.api_key)
print("DERIVED_API_SECRET=" + d.api_secret)
print("DERIVED_API_PASSPHRASE=" + d.api_passphrase)