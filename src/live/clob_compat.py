"""Compatibility imports for Polymarket CLOB Python clients.

The V2 package uses the ``py_clob_client_v2`` import namespace, while the
legacy package used ``py_clob_client``.  Live trading requires V2, but tests
and read-only utilities can still import this module in environments where only
the legacy package exists.
"""

from __future__ import annotations

try:
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import (
        ApiCreds,
        AssetType,
        BalanceAllowanceParams,
        BookParams,
        CreateOrderOptions,
        OpenOrderParams,
        OrderArgs,
        OrderMarketCancelParams,
        OrderPayload,
        PartialCreateOrderOptions,
    )
    USING_CLOB_V2 = True
except ModuleNotFoundError:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        AssetType,
        BalanceAllowanceParams,
        BookParams,
        CreateOrderOptions,
        OpenOrderParams,
        OrderArgs,
        OrderMarketCancelParams,
        OrderPayload,
        PartialCreateOrderOptions,
    )
    USING_CLOB_V2 = False
