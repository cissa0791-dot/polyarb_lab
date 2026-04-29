# Polymarket CLOB V2 Live Notes

## Incident: auth OK, live order rejected

Observed symptoms:

- `scripts/check_polymarket_auth.py` showed:
  - `CLOB SDK: V2 OK`
  - `Credentials: OK`
  - `API key pairing: OK`
  - `Level-2 auth: OK`
- Live order submission failed before the V2 fixes with:
  - `order_version_mismatch`
  - `ModuleNotFoundError: No module named 'py_clob_client'`
  - `OrderBuilder object has no attribute create_order`

Root cause:

- Polymarket moved live trading to CLOB V2.
- The old `py-clob-client` package can still pass Level-2 auth, but orders signed by it are rejected by the matching engine with `order_version_mismatch`.
- The V2 package is named `py-clob-client-v2`, but its import namespace is `py_clob_client_v2`, not `py_clob_client`.
- The V2 order builder API changed:
  - old: `builder.create_order(...)`
  - new: `builder.build_order(...)`
- The V2 `OrderArgs` constructor no longer accepts `fee_rate_bps`.

Correct fixes in this repo:

- `requirements.txt` must use `py-clob-client-v2>=1.0.0`.
- CLOB imports must go through `src/live/clob_compat.py`.
- Live startup must call `assert_clob_v2_available()` before creating authenticated live clients.
- `LiveWriteClient._create_and_post_order()` must support both builder APIs:
  - `build_order` for V2
  - `create_order` for legacy fallback/tests
- `OrderArgs` construction must tolerate V2 by omitting `fee_rate_bps` when unsupported.

Do not misdiagnose this as:

- bad IP / VPS region
- bad API key when `API key pairing` and `Level-2 auth` are OK
- wrong `POLYMARKET_SIGNATURE_TYPE` once auth is OK
- wrong `POLYMARKET_FUNDER` when auth and signer/funder display are correct

Required VPS check before any live run:

```bash
cd ~/polyarb_lab
source venv/bin/activate
source ~/.polymarket_env
python scripts/check_polymarket_auth.py
```

Required healthy output:

```text
CLOB SDK:           V2 OK
Credentials:        OK
API key pairing:    OK
Level-2 auth:       OK
```

If `CLOB SDK` fails:

```bash
python -m pip install -U -r requirements.txt
python -m pip install -U py-clob-client-v2
```

Regression tests to run after touching live auth/client/order code:

```bash
python -m pytest tests/test_live_auth.py tests/test_live_client.py tests/test_reward_profit_session.py tests/test_reward_live_mm.py tests/test_auto_maker_reward_regressions.py tests/test_market_intel_tools.py tests/test_live_broker.py -q
```

