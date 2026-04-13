"""
Track B Offline Signal Evaluation
===================================
Scope: offline only, no Track A mutation, no live trading, read-only data.
Goal:  Test whether early Track B signals predict later market behavior
       using accumulated ranked outputs, ranker_snapshots, and sidecar DB.

Run:
    py -3 scripts/trackb_signal_eval.py
Output:
    data/reports/trackb_signal_eval_report.json  (machine-readable)
    stdout: compact text report
"""

import json
import os
import glob
import sqlite3
import math
from collections import defaultdict
from datetime import datetime, timezone


# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_DIR   = os.path.join(BASE, "data", "research")
REPORTS_DIR    = os.path.join(BASE, "data", "reports")
SIDECAR_DB     = os.path.join(BASE, "data", "processed", "ab_sidecar.db")
SNAPSHOTS_FILE = os.path.join(RESEARCH_DIR, "ranker_snapshots.json")
REPORT_OUT     = os.path.join(REPORTS_DIR, "trackb_signal_eval_report.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def spearman_rho(xs, ys):
    """Spearman rank correlation between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return None, None
    def rank(seq):
        sorted_idx = sorted(range(n), key=lambda i: seq[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and seq[sorted_idx[j+1]] == seq[sorted_idx[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                r[sorted_idx[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i] - ry[i])**2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n**2 - 1))
    return round(rho, 4), n


def pearson_r(xs, ys):
    """Pearson r between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom = math.sqrt(sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys))
    if denom == 0:
        return None, None
    return round(num / denom, 4), n


def halfpoint_split(series, key="ts"):
    """Split sorted series at midpoint; return (early_list, late_list)."""
    series = sorted(series, key=lambda x: x[key])
    mid = len(series) // 2
    return series[:mid], series[mid:]


# ── load ranked outputs ───────────────────────────────────────────────────────

def load_ranked_files():
    """Return list of (ts_str, [opp_dict, ...]) sorted by ts."""
    pattern = os.path.join(RESEARCH_DIR, "ranked_2*.json")
    files = sorted(glob.glob(pattern))
    result = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        opps = d.get("opportunities", [])
        if opps:
            result.append((d["generated_at"], opps))
    return result


def build_slug_timeseries(ranked_files):
    """Dict slug -> [obs_dict, ...] across all files."""
    slug_obs = defaultdict(list)
    for ts, opps in ranked_files:
        for o in opps:
            slug_obs[o["slug"]].append({
                "ts"               : ts,
                "composite_score"  : o["composite_score"],
                "belief_vol_proxy" : o["belief_vol_proxy"],
                "fragility_score"  : o["fragility_score"],
                "persistence_rounds": o["persistence_rounds"],
                "depth_imbalance"  : o["depth_imbalance"],
                "uncertainty"      : o["uncertainty"],
                "volume_usd"       : o["volume_usd"],
                "yes_depth_shares" : o["yes_depth_shares"],
                "no_depth_shares"  : o["no_depth_shares"],
                "is_jump"          : o.get("is_jump", False),
                "is_executable"    : o.get("is_executable", False),
                "edge_cents"       : o["edge_cents"],
                "spread_cents"     : o["spread_cents"],
            })
    return slug_obs


# ── load ranker snapshots ─────────────────────────────────────────────────────

def load_snapshots():
    """Dict slug -> [{p_yes, edge, ts}, ...] sorted by ts."""
    with open(SNAPSHOTS_FILE) as f:
        d = json.load(f)
    return {slug: sorted(v, key=lambda x: x["ts"]) for slug, v in d.items()}


# ── load sidecar DB ───────────────────────────────────────────────────────────

def load_plateau_table():
    """Returns list of dicts from both98_plateau."""
    conn = sqlite3.connect(SIDECAR_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM both98_plateau")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── per-slug feature computation ──────────────────────────────────────────────

def compute_slug_features(slug, obs_series, snap_series, plateau_row):
    """Compute early/late feature pairs for one slug."""
    obs = sorted(obs_series, key=lambda x: x["ts"])
    early_obs, late_obs = halfpoint_split(obs, key="ts")

    def avg(lst, key):
        vals = [x[key] for x in lst if x[key] is not None]
        return sum(vals) / len(vals) if vals else None

    # composite_score: early vs late
    early_score = avg(early_obs, "composite_score")
    late_score  = avg(late_obs,  "composite_score")
    score_delta = (late_score - early_score) if (early_score and late_score) else None

    # belief_vol_proxy: early vs late
    early_bvp = avg(early_obs, "belief_vol_proxy")
    late_bvp  = avg(late_obs,  "belief_vol_proxy")

    # p_yes from snapshots
    pyes_vals = [e["p_yes"] for e in snap_series] if snap_series else []
    p_yes_drift    = (max(pyes_vals) - min(pyes_vals)) if pyes_vals else 0.0
    p_yes_std      = (math.sqrt(sum((p - sum(pyes_vals)/len(pyes_vals))**2
                     for p in pyes_vals) / len(pyes_vals))
                     if len(pyes_vals) > 1 else 0.0)

    # early half vs late half of snapshots
    if snap_series:
        mid = len(snap_series) // 2
        early_pyes = [e["p_yes"] for e in snap_series[:mid]]
        late_pyes  = [e["p_yes"] for e in snap_series[mid:]]
        early_p_avg = sum(early_pyes) / len(early_pyes) if early_pyes else None
        late_p_avg  = sum(late_pyes)  / len(late_pyes)  if late_pyes  else None
        p_yes_net_drift = (late_p_avg - early_p_avg) if (early_p_avg is not None and late_p_avg is not None) else 0.0
    else:
        p_yes_net_drift = 0.0

    # early p_yes drift → later p_yes drift (predictive test)
    if snap_series and len(snap_series) >= 4:
        mid = len(snap_series) // 2
        early_seg = snap_series[:mid]
        late_seg  = snap_series[mid:]
        early_pyes_drift = max(e["p_yes"] for e in early_seg) - min(e["p_yes"] for e in early_seg)
        late_pyes_drift  = max(e["p_yes"] for e in late_seg)  - min(e["p_yes"] for e in late_seg)
    else:
        early_pyes_drift = 0.0
        late_pyes_drift  = 0.0

    # volume, depth, fragility
    volume_usd = obs[0]["volume_usd"]  # relatively stable
    early_depth_imb = avg(early_obs, "depth_imbalance")
    late_depth_imb  = avg(late_obs,  "depth_imbalance")
    early_frag  = avg(early_obs, "fragility_score")
    late_frag   = avg(late_obs,  "fragility_score")

    # plateau info
    plateau_flag   = plateau_row["plateau_flag"] if plateau_row else None
    total_rounds   = plateau_row["total_rounds_seen"] if plateau_row else None
    exec_rounds    = plateau_row["executable_rounds"] if plateau_row else 0

    return {
        "slug"             : slug,
        "n_obs"            : len(obs),
        "early_score"      : round(early_score, 4) if early_score is not None else None,
        "late_score"       : round(late_score,  4) if late_score  is not None else None,
        "score_delta"      : round(score_delta, 4)  if score_delta is not None else None,
        "early_bvp"        : round(early_bvp, 5)   if early_bvp  is not None else None,
        "late_bvp"         : round(late_bvp,  5)   if late_bvp   is not None else None,
        "p_yes_drift"      : round(p_yes_drift, 4),
        "p_yes_std"        : round(p_yes_std,   4),
        "p_yes_net_drift"  : round(p_yes_net_drift, 4),
        "early_pyes_drift" : round(early_pyes_drift, 4),
        "late_pyes_drift"  : round(late_pyes_drift,  4),
        "volume_usd"       : volume_usd,
        "early_depth_imb"  : round(early_depth_imb, 4) if early_depth_imb is not None else None,
        "late_depth_imb"   : round(late_depth_imb,  4) if late_depth_imb  is not None else None,
        "early_frag"       : round(early_frag, 4)  if early_frag  is not None else None,
        "late_frag"        : round(late_frag,  4)  if late_frag   is not None else None,
        "plateau_flag"     : plateau_flag,
        "total_rounds_seen": total_rounds,
        "executable_rounds": exec_rounds,
    }


# ── rank-stability test ───────────────────────────────────────────────────────

def rank_stability(slug_obs_dict, n_windows=4):
    """
    Split session into n_windows equal time buckets.
    For each consecutive pair, compute Spearman rho on composite_score ranks.
    Returns list of (window_i, window_i+1, rho, n_slugs).
    """
    # collect all timestamps
    all_ts = set()
    for obs in slug_obs_dict.values():
        for o in obs:
            all_ts.add(o["ts"])
    all_ts = sorted(all_ts)
    if len(all_ts) < n_windows * 2:
        return []

    bucket_size = len(all_ts) // n_windows
    buckets = [set(all_ts[i*bucket_size:(i+1)*bucket_size]) for i in range(n_windows)]

    # per-bucket avg score per slug
    def bucket_avg(slug, obs, bucket_ts):
        vals = [o["composite_score"] for o in obs if o["ts"] in bucket_ts]
        return sum(vals) / len(vals) if vals else None

    slugs = list(slug_obs_dict.keys())
    window_avgs = []
    for b in buckets:
        avgs = {}
        for slug in slugs:
            v = bucket_avg(slug, slug_obs_dict[slug], b)
            if v is not None:
                avgs[slug] = v
        window_avgs.append(avgs)

    results = []
    for i in range(n_windows - 1):
        common = [s for s in slugs if s in window_avgs[i] and s in window_avgs[i+1]]
        if len(common) < 3:
            continue
        xs = [window_avgs[i][s]   for s in common]
        ys = [window_avgs[i+1][s] for s in common]
        rho, n = spearman_rho(xs, ys)
        results.append({"window_a": i, "window_b": i+1, "rho": rho, "n_slugs": n})
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Track B Offline Signal Evaluation")
    print(f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print("=" * 64)

    # Load data
    ranked_files   = load_ranked_files()
    slug_obs_dict  = build_slug_timeseries(ranked_files)
    snapshots      = load_snapshots()
    plateau_rows   = load_plateau_table()
    plateau_by_slug = {r["slug"]: r for r in plateau_rows}

    print(f"\nData loaded:")
    print(f"  Ranked files with data : {len(ranked_files)}")
    print(f"  Total observations     : {sum(len(v) for v in slug_obs_dict.values())}")
    print(f"  Slugs tracked          : {len(slug_obs_dict)}")
    print(f"  Snapshot slugs         : {len(snapshots)}")
    print(f"  both98_plateau rows    : {len(plateau_rows)}")

    # Compute per-slug features
    features = []
    for slug, obs in slug_obs_dict.items():
        snap = snapshots.get(slug, [])
        plateau = plateau_by_slug.get(slug)
        feat = compute_slug_features(slug, obs, snap, plateau)
        features.append(feat)
    features.sort(key=lambda x: x["p_yes_drift"], reverse=True)

    # ── Test 1: Early p_yes_drift → Late p_yes_drift ──────────────────────────
    t1_xs = [f["early_pyes_drift"] for f in features if f["early_pyes_drift"] is not None and f["late_pyes_drift"] is not None]
    t1_ys = [f["late_pyes_drift"]  for f in features if f["early_pyes_drift"] is not None and f["late_pyes_drift"] is not None]
    t1_rho, t1_n = spearman_rho(t1_xs, t1_ys)
    t1_r,   _    = pearson_r(t1_xs, t1_ys)

    # ── Test 2: early_bvp → score_delta ──────────────────────────────────────
    t2_xs = [f["early_bvp"]   for f in features if f["early_bvp"] is not None and f["score_delta"] is not None]
    t2_ys = [f["score_delta"] for f in features if f["early_bvp"] is not None and f["score_delta"] is not None]
    t2_rho, t2_n = spearman_rho(t2_xs, t2_ys)
    t2_r,   _    = pearson_r(t2_xs, t2_ys)

    # ── Test 3: log10(volume_usd+1) → p_yes_drift ────────────────────────────
    t3_xs = [math.log10(f["volume_usd"] + 1) for f in features if f["volume_usd"] is not None]
    t3_ys = [f["p_yes_drift"]                for f in features if f["volume_usd"] is not None]
    t3_rho, t3_n = spearman_rho(t3_xs, t3_ys)
    t3_r,   _    = pearson_r(t3_xs, t3_ys)

    # ── Test 4: p_yes_drift → late_bvp ───────────────────────────────────────
    t4_xs = [f["p_yes_drift"] for f in features if f["late_bvp"] is not None]
    t4_ys = [f["late_bvp"]   for f in features if f["late_bvp"] is not None]
    t4_rho, t4_n = spearman_rho(t4_xs, t4_ys)
    t4_r,   _    = pearson_r(t4_xs, t4_ys)

    # ── Test 5: early_score → late_score (score rank stability) ──────────────
    t5_xs = [f["early_score"] for f in features if f["early_score"] is not None and f["late_score"] is not None]
    t5_ys = [f["late_score"]  for f in features if f["early_score"] is not None and f["late_score"] is not None]
    t5_rho, t5_n = spearman_rho(t5_xs, t5_ys)
    t5_r,   _    = pearson_r(t5_xs, t5_ys)

    # ── Test 6: rank stability over 4 windows ─────────────────────────────────
    rank_stab = rank_stability(slug_obs_dict, n_windows=4)

    # ── executable count from plateau ─────────────────────────────────────────
    total_exec_rounds    = sum(r["executable_rounds"] for r in plateau_rows)
    total_rounds_tracked = sum(r["total_rounds_seen"] for r in plateau_rows)
    pct_both98_global    = (sum(r["both_98_rounds"] for r in plateau_rows) /
                            total_rounds_tracked * 100) if total_rounds_tracked else 0

    # ── Print per-slug feature table ──────────────────────────────────────────
    print("\n\n── Per-Slug Feature Table ──────────────────────────────────────────")
    hdr = f"{'Slug':<45} {'n':>4} {'p_yes_drift':>11} {'early_bvp':>9} {'late_bvp':>8} {'score_delta':>11} {'vol_usd':>10}"
    print(hdr)
    print("-" * len(hdr))
    for f in features:
        row = (
            f"{f['slug'][:44]:<44} "
            f"{f['n_obs']:>4d} "
            f"{f['p_yes_drift']:>11.4f} "
            f"{(f['early_bvp'] or 0):>9.5f} "
            f"{(f['late_bvp']  or 0):>8.5f} "
            f"{(f['score_delta'] or 0):>11.4f} "
            f"{f['volume_usd']:>10.0f}"
        )
        print(row)

    # ── Print correlation results ─────────────────────────────────────────────
    print("\n\n── Signal → Target Correlation Results ─────────────────────────────")
    print(f"{'Test':<52} {'Spearman ρ':>10} {'Pearson r':>9} {'n':>4}  Interpretation")
    print("-" * 100)

    def interp(rho):
        if rho is None: return "insufficient data"
        a = abs(rho)
        if a < 0.1:   return "no relationship"
        if a < 0.3:   return "weak"
        if a < 0.5:   return "moderate"
        if a < 0.7:   return "substantial"
        return "strong"

    rows = [
        ("early_pyes_drift → late_pyes_drift",   t1_rho, t1_r, t1_n),
        ("early_bvp → score_delta",              t2_rho, t2_r, t2_n),
        ("log10(volume) → p_yes_drift",          t3_rho, t3_r, t3_n),
        ("p_yes_drift → late_bvp",               t4_rho, t4_r, t4_n),
        ("early_score → late_score (rank stab)", t5_rho, t5_r, t5_n),
    ]
    for name, rho, r_, n in rows:
        rho_s = f"{rho:.4f}" if rho is not None else "  N/A  "
        r_s   = f"{r_:.4f}"  if r_  is not None else "  N/A  "
        n_s   = f"{n}" if n else "-"
        print(f"  {name:<50} {rho_s:>10} {r_s:>9} {n_s:>4}  {interp(rho)}")

    print("\n── Composite Score Rank Stability (4 time windows) ─────────────────")
    if rank_stab:
        for rs in rank_stab:
            rho_s = f"{rs['rho']:.4f}" if rs["rho"] is not None else "N/A"
            print(f"  Window {rs['window_a']}→{rs['window_b']}: Spearman ρ = {rho_s}  (n={rs['n_slugs']})  {interp(rs['rho'])}")
    else:
        print("  Insufficient windows.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n── Summary Table ───────────────────────────────────────────────────")
    print(f"{'Signal/Feature':<30} {'Target Behavior':<30} {'Relationship':<20} {'Spearman ρ':>10}")
    print("-" * 95)
    summary_rows = [
        ("early_pyes_drift",      "late_pyes_drift",      interp(t1_rho), t1_rho),
        ("early_bvp",             "score_delta",           interp(t2_rho), t2_rho),
        ("log10(volume_usd)",     "p_yes_drift",           interp(t3_rho), t3_rho),
        ("p_yes_drift",           "late_bvp",              interp(t4_rho), t4_rho),
        ("early_composite_score", "late_composite_score",  interp(t5_rho), t5_rho),
        ("composite_score rank",  "rank stability (W0→W1)",interp(rank_stab[0]["rho"] if rank_stab else None),
         rank_stab[0]["rho"] if rank_stab else None),
    ]
    for sig, tgt, rel, rho in summary_rows:
        rho_s = f"{rho:.4f}" if rho is not None else "  N/A"
        print(f"  {sig:<28} {tgt:<30} {rel:<20} {rho_s:>10}")

    # ── Primary finding ───────────────────────────────────────────────────────
    print("\n\n── Primary Finding ─────────────────────────────────────────────────")
    print(f"  Total observations:        {sum(len(v) for v in slug_obs_dict.values())}")
    print(f"  Executable events:         {total_exec_rounds}  (out of {total_rounds_tracked} slug-rounds)")
    print(f"  Both-98 saturation:        {pct_both98_global:.1f}%")
    print(f"  Slugs with p_yes_drift>0.1:{sum(1 for f in features if f['p_yes_drift'] > 0.1)}/{len(features)}")
    print()
    print("  EVALUATION VERDICT:")
    if total_exec_rounds == 0:
        print("  PRIMARY TARGET (executable status) has zero variance across all")
        print("  observations. No signal can predict an outcome with no variance.")
        print("  The observable internal signals show:")
        if t5_rho is not None and abs(t5_rho) >= 0.5:
            print(f"  - Composite score ranks ARE stable (ρ={t5_rho}): ranker is self-consistent.")
        if t1_rho is not None and abs(t1_rho) >= 0.3:
            print(f"  - Early p_yes drift predicts late p_yes drift (ρ={t1_rho}): p_yes momentum persists.")
        if t3_rho is not None and abs(t3_rho) >= 0.3:
            print(f"  - Volume predicts p_yes drift (ρ={t3_rho}): more liquid markets move more.")
        print("  CONCLUSION: Cannot confirm or deny predictive lift.")
        print("  No executable events → no ground truth → no promotion evidence.")
        print("  Session is informative only as a HOPELESS-regime baseline.")

    # ── Write report ──────────────────────────────────────────────────────────
    report = {
        "eval_run_ts"        : datetime.now(timezone.utc).isoformat(),
        "data_source"        : {
            "ranked_files_with_data" : len(ranked_files),
            "total_observations"     : sum(len(v) for v in slug_obs_dict.values()),
            "slugs_tracked"          : len(slug_obs_dict),
            "snapshot_slugs"         : len(snapshots),
            "plateau_rows"           : len(plateau_rows),
        },
        "primary_finding"    : {
            "executable_events"       : total_exec_rounds,
            "total_slug_rounds"       : total_rounds_tracked,
            "pct_both98"              : round(pct_both98_global, 2),
            "zero_variance_in_target" : total_exec_rounds == 0,
            "verdict"                 : "no_predictive_lift_measurable",
        },
        "correlations"       : {
            "early_pyes_drift_vs_late_pyes_drift" : {"rho": t1_rho, "r": t1_r, "n": t1_n},
            "early_bvp_vs_score_delta"            : {"rho": t2_rho, "r": t2_r, "n": t2_n},
            "log_volume_vs_pyes_drift"            : {"rho": t3_rho, "r": t3_r, "n": t3_n},
            "pyes_drift_vs_late_bvp"              : {"rho": t4_rho, "r": t4_r, "n": t4_n},
            "early_score_vs_late_score"           : {"rho": t5_rho, "r": t5_r, "n": t5_n},
        },
        "rank_stability"     : rank_stab,
        "slug_features"      : features,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report written: {REPORT_OUT}")
    print("=" * 64)


if __name__ == "__main__":
    main()
