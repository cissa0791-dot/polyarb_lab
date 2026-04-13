"""
run_unbalanced_leg_pilot
polyarb_lab / research_lines / auto_maker_loop

Thin single-market unbalanced-leg pilot.

Action space
------------
    BOOTSTRAP_BUY
    PLACE_SELL
    HOLD
    CANCEL_REPRICE

This pilot reuses the proven live components:
  - inventory bootstrap BUY path
  - repaired target metadata resolution
  - existing open SELL lookup
  - official auth/sign/place/cancel stack
  - market/user websocket observability
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_FILE_DIR = Path(__file__).resolve().parent
_LAB_ROOT = _FILE_DIR.parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("unbalanced_leg_pilot")

DATA_DIR = Path("data/research/auto_maker_loop")
RUNS_JSONL = DATA_DIR / "unbalanced_leg_pilot_runs.jsonl"
COMPETITIVE_QUEUE_LABELS = {"AT_TOP_OR_AHEAD", "NEAR_TOP", "INSIDE_SPREAD"}


def snapshot_state(
    args: argparse.Namespace,
    client: Any,
    creds: Any,
    cycle_num: int,
) -> dict[str, Any]:
    from research_lines.auto_maker_loop.modules.inventory_bootstrap import (
        check_needs_bootstrap,
    )
    from research_lines.auto_maker_loop.modules.market_ws_client import MarketWsClient
    from research_lines.auto_maker_loop.modules.user_ws_client import UserWsClient
    from research_lines.auto_maker_loop.run_hungary_alignment_test import (
        _fetch_existing_sell,
        _parse_user_log,
        _queue_label,
        _resolve_target_metadata,
    )
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        _get_order_fill_info,
        SURVIVOR_DATA,
        fetch_midpoint,
    )

    def _sanitize_tag(text: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in text.lower()
        ).strip("_")
        return safe[:64] or "target"

    def _book_fallback(token_id: str) -> tuple[Optional[float], Optional[float]]:
        try:
            book = client.get_order_book(token_id)
            bids = getattr(book, "bids", []) or []
            asks = getattr(book, "asks", []) or []
            best_bid = None
            best_ask = None
            if bids:
                raw_bid = bids[0].get("price") if isinstance(bids[0], dict) else getattr(bids[0], "price", None)
                best_bid = float(raw_bid) if raw_bid is not None else None
            if asks:
                raw_ask = asks[0].get("price") if isinstance(asks[0], dict) else getattr(asks[0], "price", None)
                best_ask = float(raw_ask) if raw_ask is not None else None
            return best_bid, best_ask
        except Exception as exc:
            logger.warning("book fallback failed token=%s: %s", token_id[:20], exc)
            return None, None

    def _raw_to_dict(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        out: dict[str, Any] = {}
        for key in (
            "id",
            "order_id",
            "status",
            "price",
            "size",
            "original_size",
            "size_matched",
            "asset_id",
            "token_id",
            "created_at",
            "createdAt",
            "timestamp",
            "time",
            "updatedAt",
        ):
            val = getattr(raw, key, None)
            if val is not None:
                out[key] = val
        return out

    def _parse_age_sec(raw: dict[str, Any]) -> Optional[float]:
        for key in (
            "created_at",
            "createdAt",
            "timestamp",
            "time",
            "updatedAt",
        ):
            val = raw.get(key)
            if val in (None, ""):
                continue
            try:
                num = float(val)
                if num > 1e11:
                    num /= 1000.0
                dt = datetime.fromtimestamp(num, tz=timezone.utc)
                return round((datetime.now(timezone.utc) - dt).total_seconds(), 2)
            except (TypeError, ValueError, OSError):
                pass
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return round((datetime.now(timezone.utc) - dt).total_seconds(), 2)
            except Exception:
                pass
        return None

    target_slug, target_data, metadata_source = _resolve_target_metadata(
        args.target.strip(),
        SURVIVOR_DATA,
    )

    token_id = str(target_data["token_id"])
    fallback_min_size = float(target_data["fallback_min_size"])
    price_ref = target_data.get("yes_price_ref")
    target_tag = _sanitize_tag(args.target or target_slug)
    invocation_id = str(
        getattr(args, "invocation_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    episode_id = str(getattr(args, "episode_id", f"{invocation_id}_c{cycle_num}"))
    pass_index = int(getattr(args, "pass_index", 1) or 1)
    pass_label = str(getattr(args, "pass_label", "FIRST_PASS"))
    pass_tag = _sanitize_tag(pass_label)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    market_log = DATA_DIR / f"unbalanced_pilot_market_{target_tag}_{pass_tag}_{ts_label}.jsonl"
    user_log = DATA_DIR / f"unbalanced_pilot_user_{target_tag}_{pass_tag}_{ts_label}.jsonl"

    market_ws = MarketWsClient(token_ids=[token_id], log_path=market_log)
    market_ws.start()

    user_ws: Optional[UserWsClient] = None
    if getattr(creds, "api_key", ""):
        user_ws = UserWsClient(
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
            log_path=user_log,
        )
        user_ws.start()

    time.sleep(max(args.observe_seconds, 0.0))

    midpoint, midpoint_source = fetch_midpoint(client, token_id, price_ref=price_ref)
    best_bid = market_ws.last_best_bid
    best_ask = market_ws.last_best_ask
    if best_bid is None or best_ask is None:
        fb_bid, fb_ask = _book_fallback(token_id)
        best_bid = best_bid if best_bid is not None else fb_bid
        best_ask = best_ask if best_ask is not None else fb_ask

    balance_shares = float(
        check_needs_bootstrap(client, token_id, required=fallback_min_size)
    )
    shortfall = round(max(0.0, fallback_min_size - balance_shares), 4)
    has_inventory = balance_shares > 0.0
    open_sell = _fetch_existing_sell(client, token_id)
    open_sell_exists = open_sell is not None
    open_sell_price = float(open_sell["price"]) if open_sell else None
    open_sell_size = float(open_sell["size"]) if open_sell else None
    open_sell_order_id = str(open_sell["order_id"]) if open_sell else ""

    raw_order = {}
    last_order_status: Optional[str] = None
    order_age_sec: Optional[float] = None
    fill_info: Optional[dict[str, Any]] = None
    if open_sell_exists:
        try:
            raw_order = _raw_to_dict(client.get_order(open_sell_order_id))
        except Exception as exc:
            logger.warning("get_order failed order_id=%s: %s", open_sell_order_id[:16], exc)
        fill_info = _get_order_fill_info(client, open_sell_order_id)
        last_order_status = (
            fill_info.get("status")
            if fill_info else str(raw_order.get("status") or "").upper() or None
        )
        order_age_sec = _parse_age_sec(raw_order)

    price_drift_cents: Optional[float] = None
    queue_label = "NO_OPEN_SELL"
    if open_sell_price is not None and best_ask is not None:
        price_drift_cents = round((open_sell_price - best_ask) * 100.0, 2)
        queue_label = _queue_label(price_drift_cents)

    recent_market_trade_count = int(market_ws.trade_count)
    sparse_flow = recent_market_trade_count == 0
    reduce_only = open_sell_exists or balance_shares >= fallback_min_size
    if balance_shares <= 0.0:
        exposure_bucket = "NONE"
    elif balance_shares < fallback_min_size:
        exposure_bucket = "PARTIAL"
    elif balance_shares == fallback_min_size:
        exposure_bucket = "READY"
    else:
        exposure_bucket = "EXCESS"

    inventory_state_digest = {
        "balance_shares": round(balance_shares, 4),
        "fallback_min_size": fallback_min_size,
        "shortfall": shortfall,
        "has_inventory": has_inventory,
        "can_sell": balance_shares >= fallback_min_size,
        "can_bootstrap": shortfall > 0.0,
        "reduce_only": reduce_only,
        "exposure_bucket": exposure_bucket,
    }
    market_state_digest = {
        "target_slug": target_slug,
        "token_id": token_id,
        "side": "SELL",
        "midpoint": midpoint,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "open_sell_exists": open_sell_exists,
        "open_sell_order_id": open_sell_order_id or None,
        "open_sell_price": open_sell_price,
        "open_sell_size": open_sell_size,
        "queue_label": queue_label,
        "recent_market_trade_count": recent_market_trade_count,
        "sparse_flow": sparse_flow,
        "price_drift_cents": price_drift_cents,
        "last_order_status": last_order_status,
        "order_age_sec": order_age_sec,
    }

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle_num": cycle_num,
        "invocation_id": invocation_id,
        "episode_id": episode_id,
        "pass_index": pass_index,
        "pass_label": pass_label,
        "args": args,
        "client": client,
        "creds": creds,
        "target_slug": target_slug,
        "target_tag": target_tag,
        "target_data": target_data,
        "metadata_source": metadata_source,
        "price_ref": price_ref,
        "midpoint_source": midpoint_source,
        "market_ws": market_ws,
        "user_ws": user_ws,
        "market_log": str(market_log),
        "user_log": str(user_log),
        "user_log_path": user_log,
        "inventory_state_digest": inventory_state_digest,
        "market_state_digest": market_state_digest,
        "pre_order_state": {
            "open_sell": open_sell,
            "raw_order": raw_order,
            "fill_info": fill_info,
            "inventory_before": round(balance_shares, 4),
        },
        "post_order_state": None,
        "user_event_digest": _parse_user_log(user_log, open_sell_order_id or None),
    }


def decide_action(state: dict[str, Any]) -> dict[str, str]:
    inv = state["inventory_state_digest"]
    market = state["market_state_digest"]
    dead_book_block = (
        not market["open_sell_exists"]
        and inv["balance_shares"] < inv["fallback_min_size"]
        and market["best_bid"] is not None
        and float(market["best_bid"]) <= 0.01
        and market["best_ask"] is not None
        and float(market["best_ask"]) >= 0.99
        and bool(market["sparse_flow"])
        and int(market["recent_market_trade_count"]) == 0
    )

    if dead_book_block:
        return {
            "chosen_action": "HOLD",
            "action_reason": (
                "bootstrap blocked by dead-book economic-interest gate: "
                f"best_bid={float(market['best_bid']):.4f} "
                f"best_ask={float(market['best_ask']):.4f} "
                f"sparse_flow={bool(market['sparse_flow'])} "
                f"recent_market_trade_count={int(market['recent_market_trade_count'])}"
            ),
            "audit_label": "BOOTSTRAP_BLOCKED_DEAD_BOOK",
        }

    if inv["balance_shares"] < inv["fallback_min_size"]:
        return {
            "chosen_action": "BOOTSTRAP_BUY",
            "action_reason": (
                f"balance_shares={inv['balance_shares']:.4f} < "
                f"fallback_min_size={inv['fallback_min_size']:.4f}"
            ),
            "audit_label": None,
        }

    if inv["balance_shares"] >= inv["fallback_min_size"] and not market["open_sell_exists"]:
        return {
            "chosen_action": "PLACE_SELL",
            "action_reason": "inventory ready and no open SELL exists",
            "audit_label": None,
        }

    if market["open_sell_exists"] and market["queue_label"] in COMPETITIVE_QUEUE_LABELS:
        return {
            "chosen_action": "HOLD",
            "action_reason": f"open SELL remains competitive ({market['queue_label']})",
            "audit_label": None,
        }

    if market["open_sell_exists"] and market["queue_label"] == "BEHIND":
        age_ok = (market["order_age_sec"] or 0.0) >= 60.0
        drift_ok = (market["price_drift_cents"] or 0.0) >= 1.0
        sparse_block = bool(market["sparse_flow"])
        if age_ok and drift_ok and not sparse_block:
            return {
                "chosen_action": "CANCEL_REPRICE",
                "action_reason": (
                    f"queue_label=BEHIND age={market['order_age_sec']:.2f}s "
                    f"drift={market['price_drift_cents']:.2f}c sparse_flow=False"
                ),
                "audit_label": None,
            }
        hold_reasons: list[str] = []
        if not age_ok:
            hold_reasons.append(f"order_age_sec={market['order_age_sec'] or 0.0:.2f}<60")
        if not drift_ok:
            hold_reasons.append(
                f"price_drift_cents={market['price_drift_cents'] or 0.0:.2f}<1.00"
            )
        if sparse_block:
            hold_reasons.append("sparse_flow_block_active")
        return {
            "chosen_action": "HOLD",
            "action_reason": "BEHIND but cancel gate blocked: " + ", ".join(hold_reasons),
            "audit_label": None,
        }

    return {
        "chosen_action": "HOLD",
        "action_reason": "default safe hold path",
        "audit_label": None,
    }


def execute_action(
    action: dict[str, str],
    state: dict[str, Any],
) -> dict[str, Any]:
    from research_lines.auto_maker_loop.modules.inventory_bootstrap import run_bootstrap
    from research_lines.auto_maker_loop.run_hungary_alignment_test import (
        _fetch_existing_sell,
        _parse_user_log,
    )
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        _cancel_order,
        _check_sell_inventory,
        _get_order_fill_info,
        _place_order,
        fetch_midpoint,
    )

    def _raw_to_dict(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        out: dict[str, Any] = {}
        for key in (
            "id",
            "order_id",
            "status",
            "price",
            "size",
            "original_size",
            "size_matched",
            "asset_id",
            "token_id",
            "created_at",
            "createdAt",
            "timestamp",
            "time",
            "updatedAt",
        ):
            val = getattr(raw, key, None)
            if val is not None:
                out[key] = val
        return out

    def _parse_age_sec(raw: dict[str, Any]) -> Optional[float]:
        for key in (
            "created_at",
            "createdAt",
            "timestamp",
            "time",
            "updatedAt",
        ):
            val = raw.get(key)
            if val in (None, ""):
                continue
            try:
                num = float(val)
                if num > 1e11:
                    num /= 1000.0
                dt = datetime.fromtimestamp(num, tz=timezone.utc)
                return round((datetime.now(timezone.utc) - dt).total_seconds(), 2)
            except (TypeError, ValueError, OSError):
                pass
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return round((datetime.now(timezone.utc) - dt).total_seconds(), 2)
            except Exception:
                pass
        return None

    args = state["args"]
    client = state["client"]
    creds = state["creds"]
    target_data = state["target_data"]
    token_id = str(target_data["token_id"])
    fallback_min_size = float(target_data["fallback_min_size"])
    market = state["market_state_digest"]
    pre_order = state["pre_order_state"]

    action_started = time.monotonic()
    result: dict[str, Any] = {
        "chosen_action": action["chosen_action"],
        "action_reason": action["action_reason"],
        "cancel_reason": None,
        "error": None,
        "pre_order_state": pre_order,
        "post_order_state": None,
        "bootstrap_result": None,
        "order_action_result": None,
    }

    tracked_order_id: Optional[str] = market.get("open_sell_order_id")
    tracked_post_order_id: Optional[str] = tracked_order_id

    try:
        if action["chosen_action"] == "BOOTSTRAP_BUY":
            boot = run_bootstrap(
                client=client,
                token_id=token_id,
                creds=creds,
                required_shares=fallback_min_size,
                dry_run=not args.live,
                price_ref=state["price_ref"],
            )
            result["bootstrap_result"] = boot
            tracked_order_id = boot.get("order_id")
            tracked_post_order_id = tracked_order_id

        elif action["chosen_action"] == "PLACE_SELL":
            if args.ask_price is not None:
                ask_price = round(args.ask_price, 4)
            elif market["best_ask"] is not None:
                ask_price = round(float(market["best_ask"]), 4)
            else:
                mid, _ = fetch_midpoint(client, token_id, price_ref=state["price_ref"])
                if mid is None:
                    raise RuntimeError("midpoint unavailable for PLACE_SELL")
                ask_price = round(min(0.99, mid + 0.015), 4)
            ask_size = fallback_min_size
            order_result = {
                "side": "SELL",
                "price": ask_price,
                "size": ask_size,
                "dry_run": not args.live,
            }
            if args.live:
                order_id, err = _place_order(client, token_id, ask_price, ask_size, "SELL")
                order_result["order_id"] = order_id
                order_result["error"] = err
                if not order_id:
                    raise RuntimeError(f"SELL placement failed: {err}")
                tracked_post_order_id = order_id
            result["order_action_result"] = order_result

        elif action["chosen_action"] == "HOLD":
            time.sleep(max(args.hold_observe_seconds, 0.0))

        elif action["chosen_action"] == "CANCEL_REPRICE":
            old_order = pre_order.get("open_sell") or {}
            old_order_id = str(old_order.get("order_id") or market.get("open_sell_order_id") or "")
            old_price = float(old_order.get("price") or market.get("open_sell_price") or 0.0)
            if not old_order_id:
                raise RuntimeError("CANCEL_REPRICE requested but no open SELL order_id found")

            if market["best_ask"] is not None:
                new_price = round(float(market["best_ask"]), 4)
            else:
                new_price = round(max(0.01, old_price - 0.01), 4)

            result["cancel_reason"] = action["action_reason"]
            order_result = {
                "cancel_order_id": old_order_id,
                "old_price": old_price,
                "new_price": new_price,
                "size": float(old_order.get("size") or fallback_min_size),
                "dry_run": not args.live,
            }
            if args.live:
                cancel_ok = _cancel_order(client, old_order_id)
                order_result["cancel_ok"] = cancel_ok
                if not cancel_ok:
                    raise RuntimeError(f"cancel failed for order_id={old_order_id}")
                new_order_id, err = _place_order(
                    client,
                    token_id,
                    new_price,
                    float(old_order.get("size") or fallback_min_size),
                    "SELL",
                )
                order_result["new_order_id"] = new_order_id
                order_result["error"] = err
                if not new_order_id:
                    raise RuntimeError(f"reprice SELL placement failed: {err}")
                tracked_post_order_id = new_order_id
            result["order_action_result"] = order_result

        time.sleep(max(args.post_action_wait_seconds, 0.0))

    except Exception as exc:
        result["error"] = str(exc)

    finally:
        inventory_after = _check_sell_inventory(
            client,
            token_id,
            required_shares=fallback_min_size,
        )
        open_sell_after = _fetch_existing_sell(client, token_id)
        open_sell_after_id = (
            str(open_sell_after["order_id"]) if open_sell_after else tracked_post_order_id
        )
        raw_post_order = {}
        post_fill_info = None
        post_order_age_sec = None
        post_order_status = None
        if open_sell_after_id:
            try:
                raw_post_order = _raw_to_dict(client.get_order(open_sell_after_id))
            except Exception as exc:
                logger.warning("post get_order failed order_id=%s: %s", open_sell_after_id[:16], exc)
            post_fill_info = _get_order_fill_info(client, open_sell_after_id)
            post_order_age_sec = _parse_age_sec(raw_post_order)
            post_order_status = (
                post_fill_info.get("status")
                if post_fill_info else str(raw_post_order.get("status") or "").upper() or None
            )

        best_bid_after = state["market_ws"].last_best_bid
        best_ask_after = state["market_ws"].last_best_ask
        price_drift_after = None
        queue_label_after = "NO_OPEN_SELL"
        if open_sell_after and best_ask_after is not None:
            price_drift_after = round(
                (float(open_sell_after["price"]) - best_ask_after) * 100.0, 2
            )
            from research_lines.auto_maker_loop.run_hungary_alignment_test import _queue_label

            queue_label_after = _queue_label(price_drift_after)

        if state["user_ws"] is not None:
            state["user_ws"].stop()
        state["market_ws"].stop()

        tracked_for_user = open_sell_after_id or tracked_order_id
        state["user_event_digest"] = _parse_user_log(
            state["user_log_path"],
            tracked_for_user,
        )
        result["post_order_state"] = {
            "open_sell": open_sell_after,
            "raw_order": raw_post_order,
            "fill_info": post_fill_info,
            "inventory_after": inventory_after,
            "market_state_after": {
                "best_bid": best_bid_after,
                "best_ask": best_ask_after,
                "recent_market_trade_count": int(state["market_ws"].trade_count),
                "sparse_flow": int(state["market_ws"].trade_count) == 0,
                "queue_label": queue_label_after,
                "price_drift_cents": price_drift_after,
                "last_order_status": post_order_status,
                "order_age_sec": post_order_age_sec,
            },
        }
        result["hold_duration_sec"] = round(
            post_order_age_sec
            if post_order_age_sec is not None
            else market.get("order_age_sec") or 0.0,
            2,
        )
        result["cycle_elapsed_sec"] = round(time.monotonic() - action_started, 2)

    return result


def print_summary_and_log(
    state: dict[str, Any],
    action: dict[str, str],
    result: dict[str, Any],
) -> None:
    def _first_non_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _inventory_shares(value: Any) -> Optional[float]:
        if isinstance(value, dict):
            return _to_float(value.get("balance_shares"))
        return _to_float(value)

    post_market_state = ((result.get("post_order_state") or {}).get("market_state_after")) or {}
    pre_order_state = dict(result.get("pre_order_state") or {})
    post_order_state = dict(result.get("post_order_state") or {})
    pre_order_state.setdefault(
        "inventory_before",
        round(float(state["inventory_state_digest"]["balance_shares"]), 4),
    )
    pre_raw_order = pre_order_state.get("raw_order") or {}
    post_raw_order = post_order_state.get("raw_order") or {}
    pre_fill_info = pre_order_state.get("fill_info") or {}
    post_fill_info = post_order_state.get("fill_info") or {}

    queue_label = post_market_state.get(
        "queue_label",
        state["market_state_digest"]["queue_label"],
    )
    best_bid = post_market_state.get(
        "best_bid",
        state["market_state_digest"]["best_bid"],
    )
    best_ask = post_market_state.get(
        "best_ask",
        state["market_state_digest"]["best_ask"],
    )
    midpoint = state["market_state_digest"]["midpoint"]
    price_drift_cents = post_market_state.get(
        "price_drift_cents",
        state["market_state_digest"]["price_drift_cents"],
    )
    raw_order_status = _first_non_none(
        post_raw_order.get("status"),
        pre_raw_order.get("status"),
        post_fill_info.get("status"),
        pre_fill_info.get("status"),
    )
    size_matched = _first_non_none(
        _to_float(post_raw_order.get("size_matched")),
        _to_float(post_fill_info.get("size_matched")),
        _to_float(post_fill_info.get("filled_size")),
        _to_float(pre_raw_order.get("size_matched")),
        _to_float(pre_fill_info.get("size_matched")),
        _to_float(pre_fill_info.get("filled_size")),
    )
    inventory_before = _inventory_shares(
        _first_non_none(
            pre_order_state.get("inventory_before"),
            state["inventory_state_digest"]["balance_shares"],
        )
    )
    inventory_after = _inventory_shares(
        _first_non_none(
            post_order_state.get("inventory_after"),
            inventory_before,
        )
    )
    fallback_min_size = _to_float(state["inventory_state_digest"].get("fallback_min_size")) or 0.0
    open_sell_after = post_order_state.get("open_sell") or {}
    open_sell_exists = bool(open_sell_after)
    raw_status_upper = str(raw_order_status or "").upper()
    live_sell_status = raw_status_upper in {"LIVE", "OPEN", "PLACED"}
    residual_inventory_class: Optional[str] = None
    residual_inventory_reason: Optional[str] = None

    if inventory_after is not None and inventory_after > 0.0:
        if open_sell_exists and live_sell_status:
            residual_inventory_class = "STRANDED_LIVE_SELL"
            residual_inventory_reason = (
                f"inventory_after={inventory_after:.4f} "
                f"open_sell_status={raw_status_upper or 'UNKNOWN'} "
                f"size_matched={float(size_matched or 0.0):.4f}"
            )
        elif inventory_after >= fallback_min_size:
            residual_inventory_class = "FULL_STRANDED_POSITION"
            residual_inventory_reason = (
                f"inventory_after={inventory_after:.4f} >= "
                f"fallback_min_size={fallback_min_size:.4f} with no live sell"
            )
        elif inventory_after <= 1.0:
            residual_inventory_class = "DUST_LEFTOVER"
            residual_inventory_reason = (
                f"inventory_after={inventory_after:.4f} <= dust_threshold=1.0000"
            )
        else:
            residual_inventory_class = "PARTIAL_LEFTOVER"
            residual_inventory_reason = (
                f"inventory_after={inventory_after:.4f} < "
                f"fallback_min_size={fallback_min_size:.4f} with no live sell"
            )

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle_num": state["cycle_num"],
        "invocation_id": state["invocation_id"],
        "episode_id": state["episode_id"],
        "pass_index": state["pass_index"],
        "pass_label": state["pass_label"],
        "target": state["target_slug"],
        "target_slug": state["target_slug"],
        "metadata_source": state["metadata_source"],
        "chosen_action": action["chosen_action"],
        "action_reason": action["action_reason"],
        "audit_label": action.get("audit_label"),
        "queue_label": queue_label,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": midpoint,
        "price_drift_cents": price_drift_cents,
        "inventory_state_digest": state["inventory_state_digest"],
        "market_state_digest": state["market_state_digest"],
        "pre_order_state": pre_order_state,
        "post_order_state": post_order_state,
        "bootstrap_result": result.get("bootstrap_result"),
        "order_action_result": result.get("order_action_result"),
        "market_trade_count": post_market_state.get(
            "recent_market_trade_count",
            state["market_ws"].trade_count,
        ),
        "sparse_flow": post_market_state.get(
            "sparse_flow",
            state["market_state_digest"]["sparse_flow"],
        ),
        "hold_duration_sec": result.get("hold_duration_sec"),
        "cycle_elapsed_sec": result.get("cycle_elapsed_sec"),
        "cancel_reason": result.get("cancel_reason"),
        "raw_order.status": raw_order_status,
        "size_matched": size_matched,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "residual_class": residual_inventory_class,
        "residual_reason": residual_inventory_reason,
        "user_event_digest": state.get("user_event_digest"),
        "market_log": state["market_log"],
        "user_log": state["user_log"],
        "midpoint_source": state["midpoint_source"],
        "error": result.get("error"),
    }
    summary = _normalize_persisted_row(summary)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    print("\n" + "=" * 64)
    print("  UNBALANCED LEG PILOT")
    print("=" * 64)
    print(f"  invocation_id       : {summary['invocation_id']}")
    print(f"  episode_id          : {summary['episode_id']}")
    print(f"  pass_label          : {summary['pass_label']}")
    print(f"  pass_index          : {summary['pass_index']}")
    print(f"  target_slug         : {summary['target_slug']}")
    print(f"  chosen_action       : {summary['chosen_action']}")
    print(f"  action_reason       : {summary['action_reason']}")
    if summary["audit_label"]:
        print(f"  audit_label         : {summary['audit_label']}")
    print(f"  queue_label         : {summary['queue_label']}")
    print(f"  best_bid            : {summary['best_bid']}")
    print(f"  best_ask            : {summary['best_ask']}")
    print(f"  midpoint            : {summary['midpoint']}")
    print(f"  price_drift_cents   : {summary['price_drift_cents']}")
    print(f"  market_trade_count  : {summary['market_trade_count']}")
    print(f"  sparse_flow         : {summary['sparse_flow']}")
    print(f"  hold_duration_sec   : {summary['hold_duration_sec']}")
    print(f"  raw_order.status    : {summary['raw_order.status']}")
    print(f"  size_matched        : {summary['size_matched']}")
    print(f"  inventory_before    : {summary['inventory_before']}")
    print(f"  inventory_after     : {summary['inventory_after']}")
    if summary["residual_class"]:
        print(f"  residual_class      : {summary['residual_class']}")
        print(f"  residual_reason     : {summary['residual_reason']}")
    if summary["cancel_reason"]:
        print(f"  cancel_reason       : {summary['cancel_reason']}")
    if summary["error"]:
        print(f"  error               : {summary['error']}")
    print(f"  inventory_digest    : {json.dumps(summary['inventory_state_digest'])}")
    print(f"  market_digest       : {json.dumps(summary['market_state_digest'])}")
    print(f"  pre_order_state     : {json.dumps(summary['pre_order_state'], default=str)}")
    print(f"  post_order_state    : {json.dumps(summary['post_order_state'], default=str)}")
    print(f"  user_event_digest   : {json.dumps(summary['user_event_digest'], default=str)}")
    print(f"  market_log          : {summary['market_log']}")
    print(f"  user_log            : {summary['user_log']}")
    print(f"  runs_jsonl          : {RUNS_JSONL}")
    return summary


def _stop_snapshot_streams(state: dict[str, Any]) -> None:
    user_ws = state.get("user_ws")
    if user_ws is not None:
        try:
            user_ws.stop()
        except Exception as exc:
            logger.warning("user_ws stop failed during follow-through skip: %s", exc)

    market_ws = state.get("market_ws")
    if market_ws is not None:
        try:
            market_ws.stop()
        except Exception as exc:
            logger.warning("market_ws stop failed during follow-through skip: %s", exc)


def _normalize_persisted_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if out.get("residual_class") is None and "residual_inventory_class" in out:
        out["residual_class"] = out.get("residual_inventory_class")
    if out.get("residual_reason") is None and "residual_inventory_reason" in out:
        out["residual_reason"] = out.get("residual_inventory_reason")

    out.pop("residual_inventory_class", None)
    out.pop("residual_inventory_reason", None)

    out.setdefault("terminal_audit_label", None)
    out.setdefault("terminal_audit_reason", None)
    out.setdefault("terminal_policy_action", None)
    out.setdefault("terminal_cancel_order_id", None)
    out.setdefault("terminal_cancel_ok", None)
    return out


def _latest_terminal_row_for_target(target_slug: str) -> Optional[dict[str, Any]]:
    if not RUNS_JSONL.exists():
        return None

    try:
        lines = RUNS_JSONL.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        logger.warning("failed to read runs_jsonl for parked gate: %s", exc)
        return None

    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("target_slug") or row.get("target") or "") != target_slug:
            continue
        if row.get("terminal_audit_label"):
            return _normalize_persisted_row(row)
    return None


def _target_history_digest(target_slug: str) -> dict[str, Any]:
    digest = {
        "latest_row": None,
        "has_terminal_audit": False,
        "has_residual_class": False,
    }
    if not RUNS_JSONL.exists():
        return digest

    try:
        lines = RUNS_JSONL.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        logger.warning("failed to read runs_jsonl for target history digest: %s", exc)
        return digest

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = _normalize_persisted_row(row)
        if str(row.get("target_slug") or row.get("target") or "") != target_slug:
            continue
        digest["latest_row"] = row
        if row.get("terminal_audit_label"):
            digest["has_terminal_audit"] = True
        if row.get("residual_class"):
            digest["has_residual_class"] = True
    return digest


def _maybe_apply_inactive_inventory_release_classifier(
    state: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    target_slug = str(state.get("target_slug") or "")
    latest_terminal = _latest_terminal_row_for_target(target_slug)
    if not latest_terminal:
        return False

    prior_terminal_label = str(latest_terminal.get("terminal_audit_label") or "")
    if prior_terminal_label not in {
        "PARKED_STRANDED_POSITION",
        "LEGACY_UNGOVERNED_INVENTORY",
    }:
        return False

    current_inventory = _to_float(state["inventory_state_digest"].get("balance_shares")) or 0.0
    if current_inventory <= 0.0:
        return False

    market = state["market_state_digest"]
    open_sell_exists = bool(market.get("open_sell_exists"))
    best_bid = _to_float(market.get("best_bid"))
    best_ask = _to_float(market.get("best_ask"))
    market_trade_count = int(market.get("recent_market_trade_count") or 0)
    sparse_flow = bool(market.get("sparse_flow"))
    release_eligible = (
        not open_sell_exists
        and best_bid is not None
        and best_bid > 0.01
        and best_ask is not None
        and best_ask < 0.99
        and market_trade_count > 0
        and not sparse_flow
    )

    prior_terminal_policy_action = latest_terminal.get("terminal_policy_action")
    if release_eligible:
        terminal_label = "INACTIVE_INVENTORY_RELEASE_ELIGIBLE"
        terminal_policy_action = "NO_ACTION_RELEASE_ELIGIBLE"
        terminal_reason = (
            f"prior_terminal_label={prior_terminal_label} "
            f"current_inventory={current_inventory:.4f} "
            f"open_sell_exists={open_sell_exists} "
            f"best_bid={best_bid:.4f} best_ask={best_ask:.4f} "
            f"market_trade_count={market_trade_count} sparse_flow=False"
        )
    else:
        terminal_label = prior_terminal_label
        terminal_policy_action = prior_terminal_policy_action
        terminal_reason = (
            f"inactive hold: prior_terminal_label={prior_terminal_label} "
            f"current_inventory={current_inventory:.4f} "
            f"open_sell_exists={open_sell_exists} "
            f"best_bid={best_bid} best_ask={best_ask} "
            f"market_trade_count={market_trade_count} sparse_flow={sparse_flow}"
        )

    terminal_row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle_num": state.get("cycle_num"),
        "invocation_id": state.get("invocation_id"),
        "episode_id": state.get("episode_id"),
        "pass_index": state.get("pass_index"),
        "pass_label": state.get("pass_label"),
        "target": target_slug,
        "target_slug": target_slug,
        "metadata_source": state.get("metadata_source"),
        "chosen_action": action.get("chosen_action"),
        "action_reason": action.get("action_reason"),
        "terminal_audit_label": terminal_label,
        "terminal_audit_reason": terminal_reason,
        "terminal_policy_action": terminal_policy_action,
        "residual_class": latest_terminal.get("residual_class"),
        "residual_reason": latest_terminal.get("residual_reason"),
        "inventory_before": current_inventory,
        "inventory_after": current_inventory,
        "queue_label": market.get("queue_label"),
        "best_bid": market.get("best_bid"),
        "best_ask": market.get("best_ask"),
        "midpoint": market.get("midpoint"),
        "price_drift_cents": market.get("price_drift_cents"),
        "market_trade_count": market.get("recent_market_trade_count"),
        "sparse_flow": market.get("sparse_flow"),
        "hold_duration_sec": None,
        "raw_order.status": market.get("last_order_status"),
        "size_matched": None,
        "inventory_state_digest": state.get("inventory_state_digest"),
        "market_state_digest": state.get("market_state_digest"),
        "pre_order_state": state.get("pre_order_state"),
        "post_order_state": {
            "inventory_after": {"balance_shares": current_inventory},
            "market_state_after": state.get("market_state_digest"),
        },
        "user_event_digest": state.get("user_event_digest"),
        "market_log": state.get("market_log"),
        "user_log": state.get("user_log"),
        "midpoint_source": state.get("midpoint_source"),
        "error": None,
    }
    terminal_row = _normalize_persisted_row(terminal_row)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(terminal_row) + "\n")

    _stop_snapshot_streams(state)

    print("\n" + "=" * 64)
    print("  TERMINAL INACTIVE POLICY")
    print("=" * 64)
    print(f"  terminal_audit_label: {terminal_label}")
    print(f"  target_slug         : {target_slug}")
    print(f"  current_inventory   : {current_inventory}")
    print(f"  blocked_action      : {action.get('chosen_action')}")
    print(f"  terminal_reason     : {terminal_reason}")
    print(f"  runs_jsonl          : {RUNS_JSONL}")
    return True


def _maybe_apply_legacy_ungoverned_inventory_quarantine(
    state: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    current_inventory = _to_float(state["inventory_state_digest"].get("balance_shares")) or 0.0
    if current_inventory <= 0.0:
        return False

    target_slug = str(state.get("target_slug") or "")
    history = _target_history_digest(target_slug)
    latest_row = history.get("latest_row") or {}
    if not latest_row:
        return False

    latest_has_canonical_context = bool(
        latest_row.get("invocation_id") and latest_row.get("pass_label")
    )
    if latest_has_canonical_context:
        return False

    if bool(history.get("has_terminal_audit")):
        return False

    if bool(history.get("has_residual_class")):
        return False

    fallback_min_size = _to_float(state["inventory_state_digest"].get("fallback_min_size")) or 0.0
    if current_inventory >= fallback_min_size:
        residual_class = "FULL_STRANDED_POSITION"
        residual_reason = (
            f"inventory_after={current_inventory:.4f} >= "
            f"fallback_min_size={fallback_min_size:.4f} legacy quarantine"
        )
    elif current_inventory <= 1.0:
        residual_class = "DUST_LEFTOVER"
        residual_reason = (
            f"inventory_after={current_inventory:.4f} <= dust_threshold=1.0000 "
            "legacy quarantine"
        )
    else:
        residual_class = "PARTIAL_LEFTOVER"
        residual_reason = (
            f"inventory_after={current_inventory:.4f} < "
            f"fallback_min_size={fallback_min_size:.4f} legacy quarantine"
        )

    market = state["market_state_digest"]
    terminal_label = "LEGACY_UNGOVERNED_INVENTORY"
    terminal_reason = (
        f"latest_row_missing_canonical_context=True "
        f"latest_invocation_id={latest_row.get('invocation_id')} "
        f"latest_pass_label={latest_row.get('pass_label')} "
        f"blocked_action={action.get('chosen_action')}"
    )
    terminal_row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle_num": state.get("cycle_num"),
        "invocation_id": state.get("invocation_id"),
        "episode_id": state.get("episode_id"),
        "pass_index": state.get("pass_index"),
        "pass_label": state.get("pass_label"),
        "target": target_slug,
        "target_slug": target_slug,
        "metadata_source": state.get("metadata_source"),
        "chosen_action": action.get("chosen_action"),
        "action_reason": action.get("action_reason"),
        "terminal_audit_label": terminal_label,
        "terminal_audit_reason": terminal_reason,
        "terminal_policy_action": "QUARANTINE_NO_ACTION",
        "residual_class": residual_class,
        "residual_reason": residual_reason,
        "inventory_before": current_inventory,
        "inventory_after": current_inventory,
        "queue_label": market.get("queue_label"),
        "best_bid": market.get("best_bid"),
        "best_ask": market.get("best_ask"),
        "midpoint": market.get("midpoint"),
        "price_drift_cents": market.get("price_drift_cents"),
        "market_trade_count": market.get("recent_market_trade_count"),
        "sparse_flow": market.get("sparse_flow"),
        "hold_duration_sec": None,
        "raw_order.status": market.get("last_order_status"),
        "size_matched": None,
        "inventory_state_digest": state.get("inventory_state_digest"),
        "market_state_digest": state.get("market_state_digest"),
        "pre_order_state": state.get("pre_order_state"),
        "post_order_state": {
            "inventory_after": {"balance_shares": current_inventory},
            "market_state_after": state.get("market_state_digest"),
        },
        "user_event_digest": state.get("user_event_digest"),
        "market_log": state.get("market_log"),
        "user_log": state.get("user_log"),
        "midpoint_source": state.get("midpoint_source"),
        "error": None,
    }
    terminal_row = _normalize_persisted_row(terminal_row)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(terminal_row) + "\n")

    _stop_snapshot_streams(state)

    print("\n" + "=" * 64)
    print("  TERMINAL QUARANTINE POLICY")
    print("=" * 64)
    print(f"  terminal_audit_label: {terminal_label}")
    print(f"  target_slug         : {target_slug}")
    print(f"  current_inventory   : {current_inventory}")
    print(f"  blocked_action      : {action.get('chosen_action')}")
    print(f"  terminal_reason     : {terminal_reason}")
    print(f"  runs_jsonl          : {RUNS_JSONL}")
    return True


def _maybe_apply_parked_stranded_position_gate(
    state: dict[str, Any],
    action: dict[str, Any],
) -> bool:
    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if action.get("chosen_action") not in {"PLACE_SELL", "BOOTSTRAP_BUY"}:
        return False

    current_inventory = _to_float(state["inventory_state_digest"].get("balance_shares")) or 0.0
    if current_inventory <= 0.0:
        return False

    target_slug = str(state.get("target_slug") or "")
    latest_terminal = _latest_terminal_row_for_target(target_slug)
    if not latest_terminal:
        return False

    prior_terminal_label = str(latest_terminal.get("terminal_audit_label") or "")
    if prior_terminal_label != "STALE_STRANDED_LIVE_SELL":
        return False

    fallback_min_size = _to_float(state["inventory_state_digest"].get("fallback_min_size")) or 0.0
    if current_inventory >= fallback_min_size:
        residual_class = "FULL_STRANDED_POSITION"
        residual_reason = (
            f"inventory_after={current_inventory:.4f} >= "
            f"fallback_min_size={fallback_min_size:.4f} parked after prior stale cancel"
        )
    elif current_inventory <= 1.0:
        residual_class = "DUST_LEFTOVER"
        residual_reason = (
            f"inventory_after={current_inventory:.4f} <= dust_threshold=1.0000 "
            "parked after prior stale cancel"
        )
    else:
        residual_class = "PARTIAL_LEFTOVER"
        residual_reason = (
            f"inventory_after={current_inventory:.4f} < "
            f"fallback_min_size={fallback_min_size:.4f} parked after prior stale cancel"
        )

    market = state["market_state_digest"]
    terminal_label = "PARKED_STRANDED_POSITION"
    terminal_reason = (
        f"prior_terminal_label={prior_terminal_label} "
        f"prior_terminal_invocation_id={latest_terminal.get('invocation_id')} "
        f"current_inventory={current_inventory:.4f} "
        f"blocked_action={action.get('chosen_action')}"
    )
    terminal_row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle_num": state.get("cycle_num"),
        "invocation_id": state.get("invocation_id"),
        "episode_id": state.get("episode_id"),
        "pass_index": state.get("pass_index"),
        "pass_label": state.get("pass_label"),
        "target": target_slug,
        "target_slug": target_slug,
        "metadata_source": state.get("metadata_source"),
        "chosen_action": action.get("chosen_action"),
        "action_reason": action.get("action_reason"),
        "terminal_audit_label": terminal_label,
        "terminal_audit_reason": terminal_reason,
        "terminal_policy_action": "PARK_NO_ACTION",
        "residual_class": residual_class,
        "residual_reason": residual_reason,
        "inventory_before": current_inventory,
        "inventory_after": current_inventory,
        "queue_label": market.get("queue_label"),
        "best_bid": market.get("best_bid"),
        "best_ask": market.get("best_ask"),
        "midpoint": market.get("midpoint"),
        "price_drift_cents": market.get("price_drift_cents"),
        "market_trade_count": market.get("recent_market_trade_count"),
        "sparse_flow": market.get("sparse_flow"),
        "hold_duration_sec": None,
        "raw_order.status": market.get("last_order_status"),
        "size_matched": None,
        "inventory_state_digest": state.get("inventory_state_digest"),
        "market_state_digest": state.get("market_state_digest"),
        "pre_order_state": state.get("pre_order_state"),
        "post_order_state": {
            "inventory_after": {"balance_shares": current_inventory},
            "market_state_after": state.get("market_state_digest"),
        },
        "user_event_digest": state.get("user_event_digest"),
        "market_log": state.get("market_log"),
        "user_log": state.get("user_log"),
        "midpoint_source": state.get("midpoint_source"),
        "error": None,
    }
    terminal_row = _normalize_persisted_row(terminal_row)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(terminal_row) + "\n")

    _stop_snapshot_streams(state)

    print("\n" + "=" * 64)
    print("  TERMINAL PARKED POLICY")
    print("=" * 64)
    print(f"  terminal_audit_label: {terminal_label}")
    print(f"  target_slug         : {target_slug}")
    print(f"  current_inventory   : {current_inventory}")
    print(f"  blocked_action      : {action.get('chosen_action')}")
    print(f"  terminal_reason     : {terminal_reason}")
    print(f"  runs_jsonl          : {RUNS_JSONL}")
    return True


def _maybe_apply_stale_stranded_live_sell_policy(
    args: argparse.Namespace,
    client: Any,
    summary: dict[str, Any],
) -> bool:
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        _cancel_order,
    )

    def _to_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if not getattr(args, "live", False):
        return False

    if summary.get("residual_class") != "STRANDED_LIVE_SELL":
        return False

    raw_status_upper = str(summary.get("raw_order.status") or "").upper()
    if raw_status_upper != "LIVE":
        return False

    post_market_state = ((summary.get("post_order_state") or {}).get("market_state_after")) or {}
    order_age_sec = _to_float(
        post_market_state.get("order_age_sec")
        or (summary.get("market_state_digest") or {}).get("order_age_sec")
    )
    if order_age_sec is None or order_age_sec < 600.0:
        return False

    market_trade_count = int(summary.get("market_trade_count") or 0)
    if market_trade_count != 0:
        return False

    if not bool(summary.get("sparse_flow")):
        return False

    best_bid = _to_float(summary.get("best_bid"))
    if best_bid is None or best_bid > 0.01:
        return False

    best_ask = _to_float(summary.get("best_ask"))
    if best_ask is None or best_ask < 0.99:
        return False

    post_order_state = summary.get("post_order_state") or {}
    open_sell = post_order_state.get("open_sell") or {}
    order_id = str(
        open_sell.get("order_id")
        or (summary.get("market_state_digest") or {}).get("open_sell_order_id")
        or ""
    )
    if not order_id:
        return False

    cancel_ok = _cancel_order(client, order_id)
    terminal_label = "STALE_STRANDED_LIVE_SELL"
    terminal_reason = (
        f"cancel_only stale live sell: order_age_sec={order_age_sec:.2f} "
        f"best_bid={best_bid:.4f} best_ask={best_ask:.4f} "
        f"market_trade_count={market_trade_count} sparse_flow=True "
        f"cancel_ok={cancel_ok}"
    )
    terminal_row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle_num": summary.get("cycle_num"),
        "invocation_id": summary.get("invocation_id"),
        "episode_id": summary.get("episode_id"),
        "pass_index": summary.get("pass_index"),
        "pass_label": summary.get("pass_label"),
        "target": summary.get("target"),
        "target_slug": summary.get("target_slug"),
        "metadata_source": summary.get("metadata_source"),
        "terminal_audit_label": terminal_label,
        "terminal_audit_reason": terminal_reason,
        "terminal_policy_action": "CANCEL_OPEN_SELL_ONLY",
        "terminal_cancel_order_id": order_id,
        "terminal_cancel_ok": cancel_ok,
        "residual_class": summary.get("residual_class"),
        "residual_reason": summary.get("residual_reason"),
        "raw_order.status": summary.get("raw_order.status"),
        "inventory_before": summary.get("inventory_before"),
        "inventory_after": summary.get("inventory_after"),
        "market_trade_count": summary.get("market_trade_count"),
        "sparse_flow": summary.get("sparse_flow"),
        "best_bid": summary.get("best_bid"),
        "best_ask": summary.get("best_ask"),
        "hold_duration_sec": summary.get("hold_duration_sec"),
        "order_age_sec": order_age_sec,
    }
    terminal_row = _normalize_persisted_row(terminal_row)
    with RUNS_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(terminal_row) + "\n")

    print("\n" + "=" * 64)
    print("  TERMINAL STALE POLICY")
    print("=" * 64)
    print(f"  terminal_audit_label: {terminal_label}")
    print(f"  target_slug         : {summary.get('target_slug')}")
    print(f"  cancel_order_id     : {order_id}")
    print(f"  cancel_ok           : {cancel_ok}")
    print(f"  terminal_reason     : {terminal_reason}")
    print(f"  runs_jsonl          : {RUNS_JSONL}")
    return True


def main() -> None:
    from research_lines.reward_aware_maker_probe.modules.scoring_activation import (
        CLOB_HOST,
        build_clob_client,
        load_activation_credentials,
    )

    parser = argparse.ArgumentParser(description="Thin single-market unbalanced-leg pilot")
    parser.add_argument("--target", default="netanyahu",
                        help="Target alias or slug (default: netanyahu)")
    parser.add_argument("--live", action="store_true",
                        help="Execute real exchange actions (default: dry-run)")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Number of pilot cycles to run (default: 1)")
    parser.add_argument("--cycle-sleep-seconds", type=float, default=15.0,
                        help="Sleep between cycles when cycles > 1 (default: 15)")
    parser.add_argument("--observe-seconds", type=float, default=5.0,
                        help="Pre-decision WS observation window (default: 5)")
    parser.add_argument("--hold-observe-seconds", type=float, default=5.0,
                        help="Additional observe window when action=HOLD (default: 5)")
    parser.add_argument("--post-action-wait-seconds", type=float, default=2.0,
                        help="Wait after live action before post-state snapshot (default: 2)")
    parser.add_argument("--ask-price", type=float, default=None,
                        help="Explicit SELL price override for PLACE_SELL")
    args = parser.parse_args()

    creds = load_activation_credentials()
    if creds is None:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in environment.")
        sys.exit(1)

    client = build_clob_client(creds, CLOB_HOST)
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    for cycle_num in range(1, max(1, args.cycles) + 1):
        episode_id = f"{invocation_id}_c{cycle_num}"
        args.invocation_id = invocation_id
        args.episode_id = episode_id
        args.pass_index = 1
        args.pass_label = "FIRST_PASS"
        state = snapshot_state(args, client, creds, cycle_num)
        action = decide_action(state)
        if _maybe_apply_inactive_inventory_release_classifier(state, action):
            break
        if _maybe_apply_legacy_ungoverned_inventory_quarantine(state, action):
            break
        if _maybe_apply_parked_stranded_position_gate(state, action):
            break
        result = execute_action(action, state)
        summary = print_summary_and_log(state, action, result)

        if _maybe_apply_stale_stranded_live_sell_policy(args, client, summary):
            break

        if action["chosen_action"] == "BOOTSTRAP_BUY" and not result.get("error"):
            args.pass_index = 2
            args.pass_label = "FOLLOW_THROUGH_PASS"
            follow_state = snapshot_state(args, client, creds, cycle_num)
            follow_action = decide_action(follow_state)
            if _maybe_apply_inactive_inventory_release_classifier(follow_state, follow_action):
                break
            if _maybe_apply_legacy_ungoverned_inventory_quarantine(follow_state, follow_action):
                break
            if _maybe_apply_parked_stranded_position_gate(follow_state, follow_action):
                break
            if follow_action["chosen_action"] != "BOOTSTRAP_BUY":
                follow_result = execute_action(follow_action, follow_state)
                follow_summary = print_summary_and_log(follow_state, follow_action, follow_result)
                if _maybe_apply_stale_stranded_live_sell_policy(args, client, follow_summary):
                    break
            else:
                _stop_snapshot_streams(follow_state)
                print(
                    "  follow_through_skip : fresh snapshot still requires "
                    f"{follow_action['chosen_action']} ({follow_action['action_reason']})"
                )

        if cycle_num < max(1, args.cycles):
            time.sleep(max(args.cycle_sleep_seconds, 0.0))


if __name__ == "__main__":
    main()
