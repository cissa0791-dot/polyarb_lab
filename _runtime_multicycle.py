"""
Multi-cycle continuous runtime validation for negRisk maker.

Runs N consecutive cycles on the same ResearchRunner instance (persistent
in-memory ledger). No config or code changes — uses current settings.yaml.

Prints per-cycle stats and aggregated close_reason / PnL attribution.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.runner import ResearchRunner

SETTINGS  = ROOT / "config" / "settings.yaml"
N_CYCLES  = 8
EXPERIMENT = {
    "experiment_label": "neg_risk_multicycle_20260322",
    "campaign_target_strategy_families": ["neg_risk_rebalancing"],
}


def main():
    print(f"Initialising ResearchRunner (N={N_CYCLES} cycles)...\n", flush=True)
    runner = ResearchRunner(settings_path=str(SETTINGS))

    per_cycle = []

    for i in range(1, N_CYCLES + 1):
        print(f"--- Cycle {i}/{N_CYCLES} ---", flush=True)
        summary = runner.run_once(experiment_context=EXPERIMENT)
        meta = summary.metadata
        elapsed = (summary.ended_ts - summary.started_ts).total_seconds()
        per_cycle.append({
            "cycle":          i,
            "raw":            meta.get("raw_candidates_by_family", {}).get("neg_risk_rebalancing", 0),
            "qual":           meta.get("qualified_candidates_by_family", {}).get("neg_risk_rebalancing", 0),
            "risk_acc":       summary.risk_accepted,
            "risk_rej":       summary.risk_rejected,
            "orders":         summary.paper_orders_created,
            "fills":          summary.fills,
            "open_pos":       summary.open_positions,
            "closed_pos":     summary.closed_positions,
            "realized_pnl":   summary.realized_pnl,
            "unrealized_pnl": summary.unrealized_pnl,
            "errors":         summary.system_errors,
            "elapsed":        elapsed,
        })
        print(
            f"  qual={per_cycle[-1]['qual']}  acc={per_cycle[-1]['risk_acc']}  "
            f"orders={per_cycle[-1]['orders']}  open={per_cycle[-1]['open_pos']}  "
            f"closed={per_cycle[-1]['closed_pos']}  rpnl={summary.realized_pnl:+.6f}  "
            f"err={per_cycle[-1]['errors']}  t={elapsed:.0f}s",
            flush=True,
        )

    # ── Per-cycle table ──────────────────────────────────────────────────────
    print("\n=== Per-cycle table ===")
    hdr = ("C", "raw", "qual", "acc", "rej", "ord", "fill", "open", "clos",
           "rpnl", "upnl", "err", "sec")
    print("%2s  %3s  %4s  %3s  %3s  %3s  %4s  %4s  %4s  %10s  %10s  %3s  %5s" % hdr)
    for r in per_cycle:
        print("%2d  %3d  %4d  %3d  %3d  %3d  %4d  %4d  %4d  %+10.6f  %+10.6f  %3d  %5.0f" % (
            r["cycle"], r["raw"], r["qual"], r["risk_acc"], r["risk_rej"],
            r["orders"], r["fills"], r["open_pos"], r["closed_pos"],
            r["realized_pnl"], r["unrealized_pnl"], r["errors"], r["elapsed"],
        ))

    # ── SQLite attribution query ─────────────────────────────────────────────
    import sqlite3, json
    db_path = ROOT / "data" / "processed" / "paper.db"
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # Identify the run_ids from this session (last N_CYCLES run summaries)
    cur.execute(
        "SELECT payload_json FROM run_summaries ORDER BY id DESC LIMIT ?",
        (N_CYCLES,),
    )
    run_payloads = [json.loads(r[0]) for r in cur.fetchall()]
    run_ids = [p["run_id"] for p in run_payloads]
    earliest_started = min(p["started_ts"] for p in run_payloads)
    # Normalize ISO format (2026-03-22T09:28:22Z) → SQLite space format (2026-03-22 09:28:22)
    earliest_started = earliest_started.replace("T", " ").rstrip("Z")

    # All exit_signal_generated events in this session
    cur.execute(
        """
        SELECT pe.reason_code, COUNT(*) as cnt,
               SUM(COALESCE(json_extract(ts.payload_json,'$.realized_pnl_usd'),0)) as pnl
        FROM position_events pe
        JOIN trade_summaries ts ON ts.position_id = pe.position_id
        WHERE pe.event_type = 'exit_signal_generated'
          AND pe.ts >= ?
        GROUP BY pe.reason_code
        ORDER BY pe.reason_code
        """,
        (earliest_started,),
    )
    reason_rows = cur.fetchall()

    total_closes = sum(r[1] for r in reason_rows)
    total_rpnl   = sum(r[2] for r in reason_rows)

    print("\n=== Close-reason attribution (all cycles) ===")
    print("%-22s  %5s  %6s  %+12s" % ("reason", "count", "frac%", "realized_pnl"))
    for reason, cnt, pnl in reason_rows:
        frac = 100.0 * cnt / total_closes if total_closes else 0
        print("%-22s  %5d  %5.1f%%  %+12.6f" % (reason, cnt, frac, pnl))
    print("%-22s  %5d  %5.1f%%  %+12.6f" % ("TOTAL", total_closes, 100.0, total_rpnl))

    # PnL by basket (candidate_id → market event group)
    cur.execute(
        """
        SELECT ts.candidate_id,
               COUNT(*) as legs,
               SUM(COALESCE(json_extract(ts.payload_json,'$.realized_pnl_usd'),0)) as pnl,
               pe.reason_code
        FROM trade_summaries ts
        JOIN position_events pe ON pe.position_id = ts.position_id
        WHERE pe.event_type = 'exit_signal_generated'
          AND pe.ts >= ?
        GROUP BY ts.candidate_id, pe.reason_code
        ORDER BY pnl DESC
        LIMIT 20
        """,
        (earliest_started,),
    )
    basket_rows = cur.fetchall()

    print("\n=== PnL by basket (top 20 by pnl) ===")
    print("%-38s  %4s  %-22s  %+12s" % ("candidate_id", "legs", "reason", "pnl"))
    for cid, legs, pnl, reason in basket_rows:
        if abs(pnl) > 1e-9:
            print("%-38s  %4d  %-22s  %+12.6f" % (cid[:38], legs, reason, pnl))

    # EDGE_DECAY details: which market slugs triggered it
    cur.execute(
        """
        SELECT ts.market_slug, ts.realized_pnl_usd,
               json_extract(pm.payload_json,'$.avg_entry_price') as entry,
               json_extract(pm.payload_json,'$.mark_price')      as mark
        FROM trade_summaries ts
        JOIN position_events pe ON pe.position_id = ts.position_id
        JOIN position_marks  pm ON pm.position_id = ts.position_id
        WHERE pe.event_type = 'exit_signal_generated'
          AND pe.reason_code = 'EDGE_DECAY'
          AND pe.ts >= ?
        ORDER BY ts.realized_pnl_usd DESC
        """,
        (earliest_started,),
    )
    decay_rows = cur.fetchall()

    if decay_rows:
        print("\n=== EDGE_DECAY exits ===")
        print("%-60s  %7s  %7s  %+10s" % ("market_slug", "entry", "mark", "pnl"))
        for slug, pnl, entry, mark in decay_rows:
            delta = (mark or 0) - (entry or 0)
            print("%-60s  %7.4f  %7.4f  %+10.6f  (delta=%+.4f)" % (
                slug[:60], entry or 0, mark or 0, pnl or 0, delta))
    else:
        print("\n=== EDGE_DECAY exits: none ===")

    # ── Exit gate / basket path summary ─────────────────────────────────────
    cur.execute(
        """
        SELECT COUNT(*)
        FROM position_events
        WHERE event_type = 'exit_suppressed'
          AND ts >= ?
        """,
        (earliest_started,),
    )
    suppressed_exit_count = cur.fetchone()[0]

    cur.execute(
        """
        SELECT pe.candidate_id,
               pe.event_type,
               pe.reason_code,
               pe.payload_json
        FROM position_events pe
        WHERE pe.event_type IN ('position_closed', 'position_expired')
          AND pe.ts >= ?
        ORDER BY pe.ts
        """,
        (earliest_started,),
    )
    basket_events = cur.fetchall()

    baskets = {}
    for candidate_id, event_type, reason_code, payload_json in basket_events:
        payload = json.loads(payload_json)
        metadata = payload.get("metadata", {})
        basket_audit = metadata.get("basket_audit", {})
        exit_path = basket_audit.get("exit_path_classification", "unknown")
        basket = baskets.setdefault(candidate_id, {
            "candidate_id": candidate_id,
            "event_types": set(),
            "reason_codes": set(),
            "paths": set(),
            "ages_sec": [],
            "legs": 0,
            "snapshot": None,
        })
        basket["event_types"].add(event_type)
        basket["reason_codes"].add(reason_code or metadata.get("close_reason") or "unknown")
        basket["paths"].add(exit_path)
        holding_duration_sec = payload.get("holding_duration_sec")
        if holding_duration_sec is not None:
            basket["ages_sec"].append(float(holding_duration_sec))
        basket["legs"] += 1
        if basket["snapshot"] is None and basket_audit:
            basket["snapshot"] = basket_audit

    basket_path_counts = defaultdict(int)
    edge_decay_basket_ages = []
    mha_only_basket_ages = []
    mha_only_adverse_evidence = []
    mha_only_baskets = []

    for basket in baskets.values():
        paths = basket["paths"]
        if len(paths) == 1:
            path = next(iter(paths))
        else:
            path = "mixed"
        basket_path_counts[path] += 1
        avg_age = sum(basket["ages_sec"]) / len(basket["ages_sec"]) if basket["ages_sec"] else 0.0
        if "MHA_only" in paths:
            mha_only_basket_ages.append(avg_age)
            snapshot = basket["snapshot"] or {}
            adverse_evidence = (
                (snapshot.get("time_since_first_adverse_state") is not None and snapshot.get("time_since_first_adverse_state", -1.0) >= 0)
                or abs(snapshot.get("basket_peak_to_current_drawdown", 0.0) or 0.0) > 1e-9
                or (snapshot.get("basket_unrealized_pnl", 0.0) or 0.0) < 0
            )
            mha_only_adverse_evidence.append(adverse_evidence)
            mha_only_baskets.append({
                "candidate_id": basket["candidate_id"],
                "legs": basket["legs"],
                "avg_age_sec": avg_age,
                "snapshot": snapshot,
                "adverse_evidence": adverse_evidence,
            })
        elif any(reason == "EDGE_DECAY" for reason in basket["reason_codes"]):
            edge_decay_basket_ages.append(avg_age)

    candidate_but_not_suppressed_count = basket_path_counts.get("minor_leg_candidate", 0)

    print("\n=== Exit gate summary ===")
    print(f"suppressed_exit_count: {suppressed_exit_count}")
    print(f"candidate_but_not_suppressed_count: {candidate_but_not_suppressed_count}")

    print("\n=== Path A vs Path B confirmation counts ===")
    print("%-34s  %5s" % ("path", "count"))
    print("%-34s  %5d" % ("dominant_leg_confirmation", basket_path_counts.get("dominant_leg_candidate", 0)))
    print("%-34s  %5d" % ("aggregate_deterioration_override", basket_path_counts.get("aggregate_basket_deterioration", 0)))

    print("\n=== MHA-only pre-expiry basket snapshot ===")
    if mha_only_baskets:
        print(
            "%-36s  %4s  %+12s  %+12s  %10s  %10s  %10s  %9s"
            % (
                "candidate_id",
                "legs",
                "basket_upnl",
                "drawdown",
                "dom_loss",
                "trigger",
                "adverse_s",
                "evidence",
            )
        )
        for basket in mha_only_baskets:
            snapshot = basket["snapshot"]
            trigger_leg_loss_share = snapshot.get("trigger_leg_loss_share")
            trigger_display = "n/a" if trigger_leg_loss_share is None else f"{trigger_leg_loss_share:.3f}"
            print(
                "%-36s  %4d  %+12.6f  %+12.6f  %10.3f  %10s  %10.1f  %9s"
                % (
                    basket["candidate_id"][:36],
                    basket["legs"],
                    snapshot.get("basket_unrealized_pnl", 0.0) or 0.0,
                    snapshot.get("basket_peak_to_current_drawdown", 0.0) or 0.0,
                    snapshot.get("dominant_loss_leg_share", 0.0) or 0.0,
                    trigger_display,
                    snapshot.get("time_since_first_adverse_state", -1.0) or -1.0,
                    "yes" if basket["adverse_evidence"] else "no",
                )
            )
    else:
        print("none")

    avg_edge_decay_age = (
        sum(edge_decay_basket_ages) / len(edge_decay_basket_ages)
        if edge_decay_basket_ages else 0.0
    )
    avg_mha_only_age = (
        sum(mha_only_basket_ages) / len(mha_only_basket_ages)
        if mha_only_basket_ages else 0.0
    )
    adverse_before_expiry_count = sum(1 for flag in mha_only_adverse_evidence if flag)

    print("\n=== Age-profile summary ===")
    print(f"average_close_age_edge_decay_baskets_sec: {avg_edge_decay_age:.1f}")
    print(f"average_close_age_mha_only_baskets_sec: {avg_mha_only_age:.1f}")
    print(
        "mha_only_baskets_with_adverse_state_evidence_before_expiry: "
        f"{adverse_before_expiry_count}/{len(mha_only_adverse_evidence)}"
    )

    # ── Healthy but idle MHA-only investigation ────────────────────────────
    mha_candidate_ids = [basket["candidate_id"] for basket in mha_only_baskets]
    idle_path_counts = defaultdict(int)
    formation_counts = defaultdict(int)
    idle_rows = []

    if mha_candidate_ids:
        placeholders = ",".join("?" for _ in mha_candidate_ids)
        cur.execute(
            f"""
            SELECT candidate_id, position_id, ts, mark_price, unrealized_pnl_usd, payload_json
            FROM position_marks
            WHERE candidate_id IN ({placeholders})
            ORDER BY candidate_id, position_id, ts
            """,
            tuple(mha_candidate_ids),
        )
        mark_rows = cur.fetchall()

        marks_by_candidate = defaultdict(lambda: defaultdict(list))
        for candidate_id, position_id, ts, mark_price, unrealized_pnl_usd, payload_json in mark_rows:
            payload = json.loads(payload_json)
            marks_by_candidate[candidate_id][position_id].append({
                "ts": ts,
                "mark_price": float(mark_price or 0.0),
                "unrealized_pnl_usd": float(unrealized_pnl_usd or 0.0),
                "payload": payload,
            })

        for basket in mha_only_baskets:
            candidate_id = basket["candidate_id"]
            legs = marks_by_candidate.get(candidate_id, {})
            repricing_events = 0
            materially_moved = False
            any_candidate_signal = False
            meaningful_state_change = False
            last_meaningful_ts = None
            near_flat_whole_window = True
            candidate_signal_legs = 0
            basket_entry_ts = None
            basket_close_ts = None

            for position_id, leg_marks in legs.items():
                if not leg_marks:
                    continue
                first = leg_marks[0]
                base_price = first["mark_price"]
                if basket_entry_ts is None or first["ts"] < basket_entry_ts:
                    basket_entry_ts = first["ts"]
                for mark in leg_marks[1:]:
                    price_delta = abs(mark["mark_price"] - base_price)
                    pnl_abs = abs(mark["unrealized_pnl_usd"])
                    if abs(mark["mark_price"] - leg_marks[max(0, leg_marks.index(mark)-1)]["mark_price"]) > 1e-9:
                        repricing_events += 1
                    if price_delta >= 0.001:
                        materially_moved = True
                        last_meaningful_ts = mark["ts"]
                    if pnl_abs >= 0.01:
                        any_candidate_signal = True
                        candidate_signal_legs += 1
                        meaningful_state_change = True
                        last_meaningful_ts = mark["ts"]
                        near_flat_whole_window = False
                    elif pnl_abs > 1e-9:
                        meaningful_state_change = True
                        last_meaningful_ts = mark["ts"]
                    if pnl_abs > 0.001:
                        near_flat_whole_window = False
                close_reason = first["payload"].get("metadata", {}).get("close_reason")
                if close_reason and basket_close_ts is None:
                    basket_close_ts = first["payload"].get("ts")

            if basket_close_ts is None:
                # fallback to snapshot average age added onto earliest entry timestamp not available in absolute form
                basket_close_ts = "unknown"

            avg_time_since_entry = basket["avg_age_sec"]

            if repricing_events == 0 and not materially_moved and not meaningful_state_change:
                idle_class = "no_state_evolution"
                formation_counts["zero_edge_decay_candidate_events_before_expiry"] += 1
                formation_counts["no_meaningful_state_evolution_after_entry"] += 1
            elif not any_candidate_signal and near_flat_whole_window:
                idle_class = "near_flat_idle_hold"
                formation_counts["zero_edge_decay_candidate_events_before_expiry"] += 1
            elif any_candidate_signal and not materially_moved:
                idle_class = "weak_signal_never_formed"
                formation_counts["weak_candidate_signals_never_crossed_confirmation"] += 1
            elif any_candidate_signal:
                idle_class = "weak_signal_never_formed"
                formation_counts["weak_candidate_signals_never_crossed_confirmation"] += 1
            elif near_flat_whole_window:
                idle_class = "near_flat_idle_hold"
                formation_counts["no_meaningful_state_evolution_after_entry"] += 1
            else:
                idle_class = "unclear"

            idle_path_counts[idle_class] += 1
            idle_rows.append({
                "candidate_id": candidate_id,
                "legs": basket["legs"],
                "avg_time_since_entry": avg_time_since_entry,
                "repricing_events": repricing_events,
                "materially_moved": materially_moved,
                "any_candidate_signal": any_candidate_signal,
                "candidate_signal_legs": candidate_signal_legs,
                "last_meaningful_ts": last_meaningful_ts or "none",
                "near_flat_whole_window": near_flat_whole_window,
                "idle_class": idle_class,
            })

    avg_time_since_entry_mha = (
        sum(row["avg_time_since_entry"] for row in idle_rows) / len(idle_rows)
        if idle_rows else 0.0
    )
    avg_repricing_events = (
        sum(row["repricing_events"] for row in idle_rows) / len(idle_rows)
        if idle_rows else 0.0
    )
    materially_moved_count = sum(1 for row in idle_rows if row["materially_moved"])
    candidate_signal_count = sum(1 for row in idle_rows if row["any_candidate_signal"])

    print("\n=== Idle basket state summary ===")
    print(f"average_time_since_entry_mha_only_baskets_sec: {avg_time_since_entry_mha:.1f}")
    print(f"average_repricing_events_after_entry_mha_only_baskets: {avg_repricing_events:.1f}")
    print(
        "mha_only_baskets_with_material_price_move_after_entry: "
        f"{materially_moved_count}/{len(idle_rows)}"
    )
    print(
        "mha_only_baskets_with_any_leg_approaching_edge_decay_candidate_state: "
        f"{candidate_signal_count}/{len(idle_rows)}"
    )

    print("\n=== EDGE_DECAY formation summary ===")
    print(
        "zero_edge_decay_candidate_events_before_expiry: "
        f"{formation_counts.get('zero_edge_decay_candidate_events_before_expiry', 0)}"
    )
    print(
        "weak_candidate_signals_never_crossed_confirmation: "
        f"{formation_counts.get('weak_candidate_signals_never_crossed_confirmation', 0)}"
    )
    print(
        "no_meaningful_state_evolution_after_entry: "
        f"{formation_counts.get('no_meaningful_state_evolution_after_entry', 0)}"
    )

    print("\n=== Post-entry activity summary ===")
    if idle_rows:
        print(
            "%-36s  %4s  %10s  %10s  %24s  %9s  %-24s"
            % (
                "candidate_id",
                "legs",
                "age_sec",
                "reprices",
                "last_meaningful_change_ts",
                "near_flat",
                "idle_class",
            )
        )
        for row in idle_rows:
            print(
                "%-36s  %4d  %10.1f  %10d  %24s  %9s  %-24s"
                % (
                    row["candidate_id"][:36],
                    row["legs"],
                    row["avg_time_since_entry"],
                    row["repricing_events"],
                    str(row["last_meaningful_ts"])[:24],
                    "yes" if row["near_flat_whole_window"] else "no",
                    row["idle_class"],
                )
            )
    else:
        print("none")

    print("\n=== Idle-path classification ===")
    print("%-24s  %5s" % ("idle_class", "count"))
    for key in ("no_state_evolution", "weak_signal_never_formed", "near_flat_idle_hold", "unclear"):
        print("%-24s  %5d" % (key, idle_path_counts.get(key, 0)))

    db.close()
    print(f"\nDone. {N_CYCLES} cycles completed.")


if __name__ == "__main__":
    main()
