"""Controlled live execution wrapper with preflight safety gates.

Wraps run_fill_validation.py with hard limits and fail-closed behavior.
No order is submitted unless ALL preflight gates pass.

Gates (all fail-closed):
  1. Required env vars present
  2. pending live orders <= max_pending  (default 0)
  3. unclosed positions  <= max_open_positions (default 5)
  4. reconcile_audit exit 0 (CLEAN) before first run
  5. max_runs_per_invocation hard cap (default 1)

After each run:
  - reconcile_audit is re-run; non-zero exit aborts remaining runs

Usage:
    python scripts/run_controlled_live.py
    python scripts/run_controlled_live.py --max-runs 1 --max-open-positions 5
    python scripts/run_controlled_live.py --dry-run
    python scripts/run_controlled_live.py --token 0xABC --ask 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_SEP    = "=" * 60

_REQUIRED_ENV = [
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
    "POLYMARKET_CHAIN_ID",
    "POLYMARKET_SIGNATURE_TYPE",
    "POLYMARKET_FUNDER",
]

_audit_log: list[dict] = []


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET} {msg}")
def _fail(msg: str) -> None: print(f"  {_RED}✗{_RESET} {msg}")
def _info(msg: str) -> None: print(f"  {_YELLOW}·{_RESET} {msg}")


def _log(event: str, **payload) -> None:
    entry = {"ts": _now(), "event": event, **payload}
    _audit_log.append(entry)
    # machine-readable line to stderr so stdout stays human-readable
    print(json.dumps(entry), file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n{_BOLD}{title}{_RESET}")
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Preflight gates
# ---------------------------------------------------------------------------

def _check_env() -> bool:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k, "").strip()]
    if missing:
        for k in missing:
            _fail(f"env var missing: {k}")
        _log("preflight_fail", gate="env_vars", missing=missing)
        return False
    _ok(f"all {len(_REQUIRED_ENV)} required env vars present")
    _log("preflight_pass", gate="env_vars")
    return True


def _count_pending_live_orders(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT COUNT(*) FROM order_intents oi
            JOIN (
                SELECT intent_id, MAX(id) AS mid
                FROM execution_reports GROUP BY intent_id
            ) L ON oi.intent_id = L.intent_id
            JOIN execution_reports er ON er.id = L.mid
            WHERE oi.mode = 'live'
              AND er.status IN ('submitted', 'partial')
        """)).scalar() or 0


def _count_unclosed_positions(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT COUNT(DISTINCT position_id) FROM position_events
            WHERE event_type = 'position_opened'
              AND position_id NOT IN (
                SELECT position_id FROM position_events
                WHERE event_type IN (
                    'position_closed', 'position_expired', 'position_force_closed'
                )
              )
        """)).scalar() or 0


def _check_live_state(engine, max_pending: int, max_open: int) -> bool:
    pending = _count_pending_live_orders(engine)
    unclosed = _count_unclosed_positions(engine)

    _info(f"pending live orders: {pending}  (limit {max_pending})")
    _info(f"unclosed positions:  {unclosed}  (limit {max_open})")
    _log("live_state", pending_live_orders=pending, unclosed_positions=unclosed,
         max_pending=max_pending, max_open_positions=max_open)

    ok = True
    if pending > max_pending:
        _fail(f"pending live orders {pending} exceeds limit {max_pending} — abort")
        _log("preflight_fail", gate="max_pending", pending=pending, limit=max_pending)
        ok = False
    else:
        _ok(f"pending live orders within limit ({pending} <= {max_pending})")

    if unclosed > max_open:
        _fail(f"unclosed positions {unclosed} exceeds limit {max_open} — abort")
        _log("preflight_fail", gate="max_open_positions", unclosed=unclosed, limit=max_open)
        ok = False
    else:
        _ok(f"unclosed positions within limit ({unclosed} <= {max_open})")

    return ok


def _run_reconcile_audit(db_url: str, label: str) -> bool:
    result = subprocess.run(
        [sys.executable, "scripts/reconcile_audit.py", "--db", db_url],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    clean = result.returncode == 0
    if clean:
        _ok(f"reconcile_audit CLEAN [{label}]")
        _log("reconcile_audit", label=label, clean=True)
    else:
        _fail(f"reconcile_audit NOT CLEAN [{label}] — abort")
        _log("reconcile_audit", label=label, clean=False, returncode=result.returncode)
    return clean


# ---------------------------------------------------------------------------
# Run one fill_validation pass
# ---------------------------------------------------------------------------

def _run_fill_validation(db_url: str, dry_run: bool, extra_args: list[str]) -> bool:
    cmd = [sys.executable, "scripts/run_fill_validation.py", "--db", db_url]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(extra_args)
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    success = result.returncode == 0
    _log("fill_validation", success=success, returncode=result.returncode,
         dry_run=dry_run)
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled live execution wrapper")
    parser.add_argument("--db",                 default="sqlite:///data/processed/paper.db")
    parser.add_argument("--max-runs",           type=int,   default=1,
                        help="Max fill_validation runs per invocation (default 1)")
    parser.add_argument("--max-open-positions", type=int,   default=5,
                        help="Abort if unclosed positions >= this (default 5)")
    parser.add_argument("--max-pending",        type=int,   default=0,
                        help="Abort if pending live orders > this (default 0)")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Pass --dry-run to run_fill_validation (discover only)")
    # pass-through args for run_fill_validation
    parser.add_argument("--token",       default=None)
    parser.add_argument("--ask",         type=float, default=None)
    parser.add_argument("--market-slug", default="manual")
    parser.add_argument("--min-size",    type=float, default=5.0)
    args = parser.parse_args()

    passthrough: list[str] = []
    if args.token:
        passthrough += ["--token", args.token]
        if args.ask is None:
            print(f"{_RED}ERROR: --ask required when --token is set{_RESET}")
            sys.exit(2)
        passthrough += ["--ask", str(args.ask)]
        passthrough += ["--market-slug", args.market_slug]
        passthrough += ["--min-size", str(args.min_size)]

    print(f"\n{_BOLD}{_SEP}")
    print("Controlled Live Execution Wrapper")
    print(f"{_SEP}{_RESET}")
    print(f"  db:                 {args.db}")
    print(f"  max_runs:           {args.max_runs}")
    print(f"  max_open_positions: {args.max_open_positions}")
    print(f"  max_pending:        {args.max_pending}")
    print(f"  dry_run:            {args.dry_run}")
    _log("start", db=args.db, max_runs=args.max_runs,
         max_open_positions=args.max_open_positions,
         max_pending=args.max_pending, dry_run=args.dry_run)

    engine = create_engine(args.db, future=True)

    # ------------------------------------------------------------------
    # Preflight block — ALL must pass
    # ------------------------------------------------------------------
    _section("PREFLIGHT CHECKS")

    gates_ok = True
    gates_ok &= _check_env()
    gates_ok &= _check_live_state(engine, args.max_pending, args.max_open_positions)

    if not gates_ok:
        print(f"\n{_RED}{_BOLD}PREFLIGHT FAILED — no orders submitted.{_RESET}\n")
        _log("abort", reason="preflight_gates_failed")
        _dump_audit_log()
        sys.exit(1)

    _section("PRE-RUN RECONCILE AUDIT")
    if not _run_reconcile_audit(args.db, "pre-run"):
        print(f"\n{_RED}{_BOLD}PRE-RUN AUDIT NOT CLEAN — no orders submitted.{_RESET}\n")
        _log("abort", reason="pre_run_audit_not_clean")
        _dump_audit_log()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------
    for run_num in range(1, args.max_runs + 1):
        _section(f"RUN {run_num} / {args.max_runs}")

        # Re-check live state before each subsequent run
        if run_num > 1:
            _info("Re-checking live state before subsequent run")
            if not _check_live_state(engine, args.max_pending, args.max_open_positions):
                _fail(f"Live state limits exceeded before run {run_num} — stopping")
                _log("abort", reason="live_state_limit_exceeded_mid_run", run=run_num)
                _dump_audit_log()
                sys.exit(1)

        success = _run_fill_validation(args.db, args.dry_run, passthrough)
        if not success:
            _fail(f"run_fill_validation failed (run {run_num})")
        else:
            _ok(f"run_fill_validation completed (run {run_num})")

        _section(f"POST-RUN {run_num} RECONCILE AUDIT")
        audit_clean = _run_reconcile_audit(args.db, f"post-run-{run_num}")
        if not audit_clean:
            _fail("Post-run audit not clean — halting remaining runs")
            _log("abort", reason="post_run_audit_not_clean", run=run_num)
            _dump_audit_log()
            sys.exit(1)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n{_SEP}")
    print(f"{_GREEN}{_BOLD}CONTROLLED RUN COMPLETE — all gates passed.{_RESET}")
    print(_SEP)
    _log("complete", runs_executed=args.max_runs)
    _dump_audit_log()
    sys.exit(0)


def _dump_audit_log() -> None:
    print(f"\n{_YELLOW}--- structured audit log ({len(_audit_log)} events) written to stderr ---{_RESET}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        _log("abort", reason="unhandled_exception", error=str(exc))
        print(f"\n{_RED}{_BOLD}FATAL: unhandled exception — {exc}{_RESET}", file=sys.stderr)
        _dump_audit_log()
        sys.exit(1)
