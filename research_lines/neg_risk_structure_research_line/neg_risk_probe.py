"""
neg_risk_probe.py — Neg-Risk Path Readiness Diagnostic
neg_risk_structure_research_line / polyarb_lab / research utility

Purpose:
    Given a wallet context and one or more token IDs, report the readiness
    of both the ordinary CLOB path and the neg-risk CLOB path.

    This is a diagnostic utility only.
    No orders are submitted. No state is mutated. No mainline files are touched.

Checks performed:
    CHECK_01  API health (GET /ok)
    CHECK_02  Wallet address derivation (L1 — from private key only)
    CHECK_03  Token neg-risk classification (L0 — GET /neg-risk per token)
    CHECK_04  Order book fetch (L0 — GET /book per token)
    CHECK_05  Ordinary collateral (USDC) allowance (L2 — GET /balance-allowance)
    CHECK_06  Conditional token allowance per token (L2 — GET /balance-allowance)
    CHECK_07  Neg-risk exchange collateral allowance (L2 — same endpoint, sig_type context)
    CHECK_08  Known on-chain contract addresses (informational — from py-clob-client config)
    CHECK_09  Path classification per token

Inputs (environment variables):
    POLYMARKET_PRIVATE_KEY       required — wallet private key (0x...)
    POLYMARKET_API_KEY           optional — enables L2 allowance checks
    POLYMARKET_API_SECRET        optional
    POLYMARKET_API_PASSPHRASE    optional
    POLYMARKET_SIGNATURE_TYPE    optional — 0 (EOA), 1 (gnosis), 2 (poly proxy). default: 0
    POLYMARKET_CHAIN_ID          optional — 137 (mainnet) or 80002 (amoy). default: 137

Inputs (CLI positional):
    token_ids — one or more CLOB token IDs (hex strings or decimal)

Path classifications:
    ordinary_ready   — ordinary exchange: collateral approved, order book live, not neg-risk token
    neg_risk_ready   — neg-risk exchange: collateral approved, order book live, IS neg-risk token
    partial          — some required checks pass, not all
    unknown          — L2 auth unavailable or all checks failed

Run:
    py -3 research_lines/neg_risk_structure_research_line/neg_risk_probe.py <token_id> [<token_id> ...]
    py -3 research_lines/neg_risk_structure_research_line/neg_risk_probe.py --help

Example:
    py -3 research_lines/neg_risk_structure_research_line/neg_risk_probe.py \
        21742633143463906290569050155826241533067272736897614950488156847949938836455 \
        52114319501245915516055106046884209969926127482827954674443846427813813222426
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import httpx

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
    from py_clob_client.config import get_contract_config
    PY_CLOB_AVAILABLE = True
except ImportError:
    PY_CLOB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
CHAIN_ID_MAINNET = 137
CHAIN_ID_AMOY = 80002

# From py-clob-client config.py (canonical source — do not edit)
KNOWN_CONTRACTS: dict[int, dict[str, str]] = {
    CHAIN_ID_MAINNET: {
        "ordinary_exchange":   "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
        "neg_risk_exchange":   "0xC5d563A36AE78145C45a50134d48A1215220f80a",
        "collateral_usdc":     "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "conditional_tokens":  "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
        # neg_risk_adapter is a separate contract not in py-clob-client config.
        # Source: https://github.com/Polymarket/neg-risk-ctf-adapter/blob/main/addresses.json
        # Verify the actual address from that file before approving anything on-chain.
        "neg_risk_adapter":    "VERIFY_FROM_neg-risk-ctf-adapter/addresses.json",
    },
    CHAIN_ID_AMOY: {
        "ordinary_exchange":   "0xdFE02Eb6733538f8Ea35D585af8DE5958AD99E40",
        "neg_risk_exchange":   "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
        "collateral_usdc":     "0x9c4e1703476e875070ee25b56a58b008cfb8fa78",
        "conditional_tokens":  "0x69308FB512518e39F9b16112fA8d994F4e2Bf8bB",
        "neg_risk_adapter":    "VERIFY_FROM_neg-risk-ctf-adapter/addresses.json",
    },
}

# Check result states
PASS   = "PASS"
FAIL   = "FAIL"
SKIP   = "SKIP"       # check not applicable or not yet reached
WARN   = "WARN"       # partial/ambiguous result
NO_L2  = "NO_L2"      # requires L2 auth not available
ERROR  = "ERROR"      # unexpected exception

PathClass = Literal["ordinary_ready", "neg_risk_ready", "partial", "unknown"]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    check_id: str
    name: str
    state: str          # PASS | FAIL | SKIP | WARN | NO_L2 | ERROR
    value: Any = None   # the raw value returned (balance, bool, dict excerpt, etc.)
    note: str = ""


@dataclass
class TokenProbeResult:
    token_id: str
    checks: list[CheckResult] = field(default_factory=list)
    path_class: Optional[PathClass] = None
    path_reason: str = ""

    def get(self, check_id: str) -> Optional[CheckResult]:
        for c in self.checks:
            if c.check_id == check_id:
                return c
        return None


@dataclass
class ProbeSession:
    run_at: datetime
    chain_id: int
    wallet_address: Optional[str]
    clob_host: str
    global_checks: list[CheckResult] = field(default_factory=list)
    token_results: list[TokenProbeResult] = field(default_factory=list)
    auth_level: int = 0   # 0, 1, or 2

    def get_global(self, check_id: str) -> Optional[CheckResult]:
        for c in self.global_checks:
            if c.check_id == check_id:
                return c
        return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

W = 72

def _line(char: str = "─") -> str:
    return char * W


def _state_icon(state: str) -> str:
    return {
        PASS:  "[PASS]",
        FAIL:  "[FAIL]",
        SKIP:  "[SKIP]",
        WARN:  "[WARN]",
        NO_L2: "[NO_L2]",
        ERROR: "[ERROR]",
    }.get(state, f"[{state}]")


def _print_check(c: CheckResult, indent: int = 2) -> None:
    pad = " " * indent
    icon = _state_icon(c.state)
    label = f"{icon} {c.check_id} {c.name}"
    print(f"{pad}{label}")
    if c.value is not None:
        val_str = str(c.value)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        for line in textwrap.wrap(val_str, width=W - indent - 8):
            print(f"{pad}        {line}")
    if c.note:
        for line in textwrap.wrap(c.note, width=W - indent - 8):
            print(f"{pad}        NOTE: {line}")


def _print_section(title: str) -> None:
    print()
    print(_line("─"))
    print(f"  {title}")
    print(_line("─"))


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def check_api_health(clob_host: str) -> CheckResult:
    """CHECK_01: GET /ok"""
    try:
        resp = httpx.get(f"{clob_host}/ok", timeout=10)
        if resp.status_code == 200:
            return CheckResult("CHECK_01", "API health (GET /ok)", PASS, resp.text.strip())
        return CheckResult(
            "CHECK_01", "API health (GET /ok)", FAIL,
            note=f"HTTP {resp.status_code}"
        )
    except Exception as exc:
        return CheckResult("CHECK_01", "API health (GET /ok)", ERROR, note=str(exc))


def check_wallet_address(client: Optional["ClobClient"]) -> CheckResult:
    """CHECK_02: Wallet address derivation (L1)"""
    if client is None:
        return CheckResult(
            "CHECK_02", "Wallet address (L1)", NO_L2,
            note="ClobClient not initialized — POLYMARKET_PRIVATE_KEY not set"
        )
    try:
        addr = client.get_address()
        if addr:
            return CheckResult("CHECK_02", "Wallet address (L1)", PASS, addr)
        return CheckResult(
            "CHECK_02", "Wallet address (L1)", FAIL,
            note="get_address() returned None"
        )
    except Exception as exc:
        return CheckResult("CHECK_02", "Wallet address (L1)", ERROR, note=str(exc))


def check_neg_risk_flag(
    token_id: str,
    clob_host: str,
    client: Optional["ClobClient"],
) -> CheckResult:
    """CHECK_03: Token neg-risk classification (L0 — GET /neg-risk)"""
    # Try via client cache first, then direct httpx (L0, no auth needed)
    try:
        if client is not None:
            result = client.get_neg_risk(token_id)
        else:
            resp = httpx.get(
                f"{clob_host}/neg-risk",
                params={"token_id": token_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("neg_risk")

        if result is True:
            return CheckResult(
                "CHECK_03", "Token neg-risk flag", PASS, True,
                note="Token is in a neg-risk market"
            )
        elif result is False:
            return CheckResult(
                "CHECK_03", "Token neg-risk flag", PASS, False,
                note="Token is in an ordinary (non-neg-risk) market"
            )
        else:
            return CheckResult(
                "CHECK_03", "Token neg-risk flag", WARN, result,
                note="Unexpected response shape from /neg-risk"
            )
    except Exception as exc:
        return CheckResult("CHECK_03", "Token neg-risk flag", ERROR, note=str(exc))


def check_order_book(
    token_id: str,
    clob_host: str,
    client: Optional["ClobClient"],
) -> CheckResult:
    """CHECK_04: Order book fetch (L0)"""
    try:
        if client is not None:
            book = client.get_order_book(token_id)
            bids = getattr(book, "bids", []) or []
            asks = getattr(book, "asks", []) or []
            neg_risk = getattr(book, "neg_risk", None)
            summary = {
                "bids": len(bids),
                "asks": len(asks),
                "neg_risk": neg_risk,
                "best_bid": float(bids[0].price) if bids else None,
                "best_ask": float(asks[0].price) if asks else None,
            }
        else:
            resp = httpx.get(
                f"{clob_host}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            summary = {
                "bids": len(bids),
                "asks": len(asks),
                "neg_risk": data.get("neg_risk"),
                "best_bid": float(bids[0]["price"]) if bids else None,
                "best_ask": float(asks[0]["price"]) if asks else None,
            }

        if summary["asks"] == 0 and summary["bids"] == 0:
            return CheckResult(
                "CHECK_04", "Order book fetch", WARN, summary,
                note="Book returned successfully but both sides empty (thin/closed market)"
            )
        return CheckResult("CHECK_04", "Order book fetch", PASS, summary)

    except Exception as exc:
        return CheckResult("CHECK_04", "Order book fetch", FAIL, note=str(exc))


def check_collateral_allowance_ordinary(
    client: Optional["ClobClient"],
    sig_type: int,
) -> CheckResult:
    """CHECK_05: Ordinary exchange USDC (collateral) allowance (L2)"""
    if client is None or not PY_CLOB_AVAILABLE:
        return CheckResult(
            "CHECK_05", "Ordinary exchange collateral allowance (L2)", NO_L2,
            note="Requires ClobClient with private key"
        )
    try:
        client.assert_level_2_auth()
    except Exception:
        return CheckResult(
            "CHECK_05", "Ordinary exchange collateral allowance (L2)", NO_L2,
            note="L2 auth not available — set POLYMARKET_API_KEY/SECRET/PASSPHRASE"
        )

    try:
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=sig_type,
        )
        result = client.get_balance_allowance(params)
        allowance = result.get("allowance") if isinstance(result, dict) else None
        balance = result.get("balance") if isinstance(result, dict) else None

        state = PASS if (allowance is not None and float(allowance) > 0) else WARN
        note = (
            "Ordinary exchange collateral approved" if state == PASS
            else "Allowance is zero — ordinary exchange not approved or USDC not deposited"
        )
        return CheckResult(
            "CHECK_05", "Ordinary exchange collateral allowance (L2)", state,
            {"allowance": allowance, "balance": balance},
            note=note,
        )
    except Exception as exc:
        return CheckResult(
            "CHECK_05", "Ordinary exchange collateral allowance (L2)", ERROR, note=str(exc)
        )


def check_conditional_allowance(
    token_id: str,
    client: Optional["ClobClient"],
    sig_type: int,
) -> CheckResult:
    """CHECK_06: Conditional token allowance for a specific token (L2)"""
    if client is None or not PY_CLOB_AVAILABLE:
        return CheckResult(
            "CHECK_06", "Conditional token allowance (L2)", NO_L2,
            note="Requires ClobClient with private key"
        )
    try:
        client.assert_level_2_auth()
    except Exception:
        return CheckResult(
            "CHECK_06", "Conditional token allowance (L2)", NO_L2,
            note="L2 auth not available — set POLYMARKET_API_KEY/SECRET/PASSPHRASE"
        )

    try:
        params = BalanceAllowanceParams(
            asset_type=AssetType.CONDITIONAL,
            token_id=token_id,
            signature_type=sig_type,
        )
        result = client.get_balance_allowance(params)
        allowance = result.get("allowance") if isinstance(result, dict) else None
        balance = result.get("balance") if isinstance(result, dict) else None

        state = PASS if (allowance is not None and float(allowance) > 0) else WARN
        note = (
            "Conditional token exchange approved" if state == PASS
            else "Conditional allowance is zero — conditional token exchange not approved"
        )
        return CheckResult(
            "CHECK_06", "Conditional token allowance (L2)", state,
            {"allowance": allowance, "balance": balance, "token_id": token_id[:24] + "..."},
            note=note,
        )
    except Exception as exc:
        return CheckResult(
            "CHECK_06", "Conditional token allowance (L2)", ERROR, note=str(exc)
        )


def check_neg_risk_exchange_allowance(
    client: Optional["ClobClient"],
    sig_type: int,
    clob_host: str,
) -> CheckResult:
    """
    CHECK_07: Neg-risk exchange collateral allowance.

    The py-clob-client's get_balance_allowance checks the ordinary exchange path by default.
    For neg-risk, the underlying exchange address differs
    (0xC5d563A36AE78145C45a50134d48A1215220f80a on mainnet).

    The CLOB API may route to the correct exchange based on the sig_type and endpoint context.
    We attempt the same call and compare result structure.
    If the result shows the same allowance as CHECK_05, the API may not distinguish paths.
    If different, the neg-risk exchange allowance is independently confirmed.
    """
    if client is None or not PY_CLOB_AVAILABLE:
        return CheckResult(
            "CHECK_07", "Neg-risk exchange collateral allowance (L2)", NO_L2,
            note="Requires ClobClient with private key"
        )
    try:
        client.assert_level_2_auth()
    except Exception:
        return CheckResult(
            "CHECK_07", "Neg-risk exchange collateral allowance (L2)", NO_L2,
            note="L2 auth not available"
        )

    try:
        # The CLOB balance-allowance endpoint routes based on the signed request context.
        # We call with COLLATERAL to probe collateral readiness for the authenticated account.
        # The neg-risk exchange is separate on-chain, so if allowance == 0 here while
        # ordinary path works, the neg-risk exchange has NOT been separately approved.
        # The API does not expose a neg_risk=true parameter on this endpoint.
        # This result should be compared with CHECK_05 by the operator.
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=sig_type,
        )
        result = client.get_balance_allowance(params)
        allowance = result.get("allowance") if isinstance(result, dict) else None
        balance = result.get("balance") if isinstance(result, dict) else None

        # We cannot distinguish which exchange this allowance is for via the CLOB API alone.
        # Flag this explicitly.
        return CheckResult(
            "CHECK_07", "Neg-risk exchange collateral allowance (L2)", WARN,
            {"allowance": allowance, "balance": balance},
            note=(
                "CLOB API returns allowance for the authenticated account context. "
                "Cannot confirm whether this covers the neg-risk exchange specifically "
                "(0xC5d563A36AE78145C45a50134d48A1215220f80a on mainnet) "
                "without an on-chain query. "
                "If this allowance differs from CHECK_05, the API is returning different state. "
                "To fully verify: call update_balance_allowance or check on-chain via web3."
            ),
        )
    except Exception as exc:
        return CheckResult(
            "CHECK_07", "Neg-risk exchange collateral allowance (L2)", ERROR, note=str(exc)
        )


def check_neg_risk_adapter_allowance(
    wallet_address: Optional[str],
    chain_id: int,
) -> CheckResult:
    """
    CHECK_08b: Neg-risk adapter allowance.

    The neg-risk adapter is a separate contract from the neg-risk exchange.
    Allowance for it cannot be checked via the CLOB API — it requires a direct
    on-chain call to the USDC contract: allowance(wallet, negRiskAdapter).

    This probe reports the known adapter address and states what is needed.
    Actual allowance value cannot be determined without web3.
    """
    contracts = KNOWN_CONTRACTS.get(chain_id, {})
    adapter_addr = contracts.get("neg_risk_adapter", "UNKNOWN")

    return CheckResult(
        "CHECK_08b", "Neg-risk adapter allowance", SKIP,
        {"neg_risk_adapter_address": adapter_addr},
        note=(
            "Cannot verify neg-risk adapter USDC allowance via CLOB API. "
            "Requires on-chain call: USDC.allowance(wallet, negRiskAdapter). "
            f"Adapter address on chain_id={chain_id}: {adapter_addr}. "
            "Verify address from: "
            "https://github.com/Polymarket/neg-risk-ctf-adapter/blob/main/addresses.json"
        ),
    )


def report_contract_addresses(chain_id: int) -> CheckResult:
    """CHECK_08: Known on-chain contract addresses (informational)"""
    contracts = KNOWN_CONTRACTS.get(chain_id, {})
    return CheckResult(
        "CHECK_08", "Known contract addresses (informational)", PASS,
        contracts,
        note=(
            f"Addresses from py-clob-client config.py (chain_id={chain_id}). "
            "neg_risk_adapter must be verified from neg-risk-ctf-adapter repo."
        ),
    )


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

def classify_path(token_result: TokenProbeResult) -> tuple[PathClass, str]:
    """
    Classify the token's path readiness based on all check results.

    ordinary_ready:
        - CHECK_03 PASS with neg_risk=False
        - CHECK_04 PASS (book live)
        - CHECK_05 PASS (collateral approved)

    neg_risk_ready:
        - CHECK_03 PASS with neg_risk=True
        - CHECK_04 PASS (book live)
        - CHECK_05 PASS (collateral approved, may cover neg-risk exchange)
        - Note: CHECK_07 and CHECK_08b cannot fully verify without web3

    partial:
        - Some checks PASS, at least one required check WARN/FAIL/NO_L2

    unknown:
        - Critical checks ERROR or all FAIL
    """
    c03 = token_result.get("CHECK_03")
    c04 = token_result.get("CHECK_04")
    c05 = token_result.get("CHECK_05")

    is_neg_risk: Optional[bool] = None
    if c03 and c03.state == PASS and isinstance(c03.value, bool):
        is_neg_risk = c03.value

    book_ok = c04 is not None and c04.state in (PASS, WARN)
    collateral_ok = c05 is not None and c05.state == PASS

    if is_neg_risk is None:
        return "unknown", "Could not determine token type (CHECK_03 failed or errored)"

    if is_neg_risk is False:
        # Ordinary path
        if collateral_ok and book_ok:
            return "ordinary_ready", (
                "Token is ordinary (non-neg-risk). "
                "Collateral approved. Order book live."
            )
        reasons = []
        if not book_ok:
            reasons.append("order book fetch failed or empty")
        if not collateral_ok:
            reasons.append(
                "collateral allowance zero or L2 auth unavailable — "
                "run update_balance_allowance or check POLYMARKET_API_KEY"
            )
        return "partial", "Ordinary path: " + "; ".join(reasons)

    else:
        # Neg-risk path
        c07 = token_result.get("CHECK_07")
        neg_risk_allowance_attempted = c07 is not None and c07.state in (PASS, WARN)

        if collateral_ok and book_ok:
            return "neg_risk_ready", (
                "Token is neg-risk. "
                "Collateral approved via CLOB API. Order book live. "
                "CAVEAT: neg-risk exchange and adapter allowances cannot be fully "
                "verified without on-chain query (web3). Treat as partial until "
                "confirmed on-chain."
            )
        reasons = []
        if not book_ok:
            reasons.append("order book fetch failed or empty")
        if not collateral_ok:
            reasons.append(
                "collateral allowance zero or L2 auth unavailable"
            )
        return "partial", "Neg-risk path: " + "; ".join(reasons)


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------

def run_probe(
    token_ids: list[str],
    clob_host: str = CLOB_HOST,
    chain_id: int = CHAIN_ID_MAINNET,
    private_key: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_passphrase: Optional[str] = None,
    sig_type: int = 0,
) -> ProbeSession:
    """Run the full probe and return a ProbeSession with all results."""

    run_at = datetime.now(timezone.utc)
    session = ProbeSession(run_at=run_at, chain_id=chain_id, wallet_address=None, clob_host=clob_host)

    # Build client
    client: Optional[ClobClient] = None
    if PY_CLOB_AVAILABLE and private_key:
        try:
            creds: Optional[ApiCreds] = None
            if api_key and api_secret and api_passphrase:
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )
            client = ClobClient(
                host=clob_host,
                chain_id=chain_id,
                key=private_key,
                creds=creds,
            )
            session.auth_level = 2 if creds else 1
        except Exception as exc:
            session.global_checks.append(
                CheckResult("CLIENT_INIT", "ClobClient initialization", ERROR, note=str(exc))
            )

    # Global checks
    session.global_checks.append(check_api_health(clob_host))
    c02 = check_wallet_address(client)
    session.global_checks.append(c02)
    if c02.state == PASS:
        session.wallet_address = str(c02.value)

    session.global_checks.append(report_contract_addresses(chain_id))
    session.global_checks.append(check_neg_risk_adapter_allowance(session.wallet_address, chain_id))

    # Ordinary exchange collateral allowance (shared across all tokens — check once)
    session.global_checks.append(check_collateral_allowance_ordinary(client, sig_type))

    # Neg-risk exchange allowance attempt (once, applies to all neg-risk tokens)
    session.global_checks.append(check_neg_risk_exchange_allowance(client, sig_type, clob_host))

    # Per-token checks
    for token_id in token_ids:
        tr = TokenProbeResult(token_id=token_id)

        tr.checks.append(check_neg_risk_flag(token_id, clob_host, client))
        tr.checks.append(check_order_book(token_id, clob_host, client))
        tr.checks.append(check_conditional_allowance(token_id, client, sig_type))

        # Copy global allowance results into token result for path classification
        c05_global = session.get_global("CHECK_05")
        if c05_global:
            tr.checks.append(c05_global)
        c07_global = session.get_global("CHECK_07")
        if c07_global:
            tr.checks.append(c07_global)

        path, reason = classify_path(tr)
        tr.path_class = path
        tr.path_reason = reason

        session.token_results.append(tr)

    return session


# ---------------------------------------------------------------------------
# Output printer
# ---------------------------------------------------------------------------

def print_session(session: ProbeSession) -> None:
    print()
    print(_line("═"))
    print("  NEG-RISK PROBE — Readiness Diagnostic")
    print(f"  Run at    : {session.run_at.isoformat()}")
    print(f"  Chain ID  : {session.chain_id}")
    print(f"  Auth level: {session.auth_level} "
          f"({'L2 (full)' if session.auth_level == 2 else 'L1 (key only)' if session.auth_level == 1 else 'L0 (none)'})")
    if session.wallet_address:
        print(f"  Wallet    : {session.wallet_address}")
    print(_line("═"))

    _print_section("GLOBAL CHECKS")
    for c in session.global_checks:
        _print_check(c)

    for tr in session.token_results:
        short_id = tr.token_id[:24] + "..." if len(tr.token_id) > 24 else tr.token_id
        _print_section(f"TOKEN: {short_id}")
        print(f"  Full token_id: {tr.token_id}")
        print()
        for c in tr.checks:
            # Skip re-printing the global checks (they were shown above)
            if c.check_id in ("CHECK_05", "CHECK_07"):
                continue
            _print_check(c)

        print()
        path_icons = {
            "ordinary_ready": "[ORDINARY_READY]",
            "neg_risk_ready": "[NEG_RISK_READY]",
            "partial":        "[PARTIAL       ]",
            "unknown":        "[UNKNOWN       ]",
        }
        icon = path_icons.get(tr.path_class or "unknown", "[?]")
        print(f"  PATH CLASSIFICATION: {icon}")
        for line in textwrap.wrap(tr.path_reason, width=W - 4):
            print(f"    {line}")

    _print_section("SUMMARY")
    for tr in session.token_results:
        short_id = tr.token_id[:20] + "..."
        path = tr.path_class or "unknown"
        print(f"  {short_id}  →  {path.upper()}")

    print()
    print("  NOTES:")
    print("  - No orders submitted. No state mutated.")
    print("  - neg-risk adapter allowance requires on-chain verification (web3).")
    print("  - neg_risk_ready path has caveat: see CHECK_07 and CHECK_08b above.")
    print("  - To approve neg-risk exchange: call client.update_balance_allowance()")
    print("    with appropriate params (triggers on-chain approval via CLOB relay).")
    print(_line("═"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_env() -> dict[str, Any]:
    return {
        "private_key":     os.environ.get("POLYMARKET_PRIVATE_KEY"),
        "api_key":         os.environ.get("POLYMARKET_API_KEY"),
        "api_secret":      os.environ.get("POLYMARKET_API_SECRET"),
        "api_passphrase":  os.environ.get("POLYMARKET_API_PASSPHRASE"),
        "sig_type":        int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0")),
        "chain_id":        int(os.environ.get("POLYMARKET_CHAIN_ID", str(CHAIN_ID_MAINNET))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            neg_risk_probe — Neg-Risk Path Readiness Diagnostic
            Utility/research only. No orders submitted.

            Checks ordinary vs neg-risk CLOB path readiness for given token IDs.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "token_ids",
        nargs="+",
        help="One or more CLOB token IDs to probe",
    )
    parser.add_argument(
        "--clob-host",
        default=CLOB_HOST,
        help=f"CLOB API host (default: {CLOB_HOST})",
    )
    parser.add_argument(
        "--chain-id",
        type=int,
        default=None,
        help=f"Chain ID: 137 (mainnet) or 80002 (amoy). Default: from env or {CHAIN_ID_MAINNET}",
    )

    args = parser.parse_args()
    env = _load_env()
    chain_id = args.chain_id if args.chain_id is not None else env["chain_id"]

    if not PY_CLOB_AVAILABLE:
        print("WARNING: py_clob_client not installed. Auth-dependent checks will be skipped.")
        print("         Install: pip install py-clob-client")
        print()

    session = run_probe(
        token_ids=args.token_ids,
        clob_host=args.clob_host,
        chain_id=chain_id,
        private_key=env["private_key"],
        api_key=env["api_key"],
        api_secret=env["api_secret"],
        api_passphrase=env["api_passphrase"],
        sig_type=env["sig_type"],
    )
    print_session(session)


if __name__ == "__main__":
    main()
