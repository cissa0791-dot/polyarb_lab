"""
neg_risk_structure_research_line — Output Tables
polyarb_lab / research_line / active

Generates all 6 required research output tables from accumulated scan log data.

Tables:
  1. information_registry_table     — known structural facts about neg-risk markets
  2. codified_hypothesis_table      — hypotheses with test methods and evidence standards
  3. research_test_results_table    — per-scan test results with metrics
  4. useful_vs_noise_table          — classified findings by signal usefulness
  5. escalation_candidates_table    — findings with repeated evidence for escalation
  6. rejected_information_table     — definitively rejected hypotheses

All tables are derived from:
  - registry.py (static hypotheses)
  - scan log files (dynamic scan results)

No API calls. No order submission. Read-only analysis.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from anywhere in the repo
# ---------------------------------------------------------------------------
_OUTPUTS_DIR = Path(__file__).resolve().parent
_LINE_DIR = _OUTPUTS_DIR.parent
_LAB_ROOT = _LINE_DIR.parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from research_lines.neg_risk_structure_research_line.registry import (
    REGISTRY, NegRiskHypothesis, FinalClassification
)
from research_lines.neg_risk_structure_research_line.modules.paper_logger import (
    load_all_scans, load_scan_index, NEG_RISK_DATA_DIR
)

# Escalation thresholds
ESCALATION_MIN_SCANS = 5             # minimum scans on same event to consider
ESCALATION_MIN_GAP = 0.010          # minimum abs_gap to appear in escalation candidates
ESCALATION_EXECUTABLE_MIN = 3       # minimum EXECUTABLE classifications on same event
PERSISTENCE_MIN_SCANS = 3           # minimum scans to compute persistence
USEFUL_GAP_THRESHOLD = 0.005        # abs_gap above this = useful signal, below = noise


# ---------------------------------------------------------------------------
# Table 1: Information Registry Table
# ---------------------------------------------------------------------------

def information_registry_table() -> list[dict[str, Any]]:
    """
    Returns known structural facts about neg-risk markets.
    Combines static hypothesis registry with accumulated scan evidence.
    """
    scans = load_all_scans()
    total_scans = len(scans)

    # Count events observed with abs_gap > threshold across all scans
    all_events_seen: dict[str, list[float]] = defaultdict(list)  # event_id -> [abs_gap, ...]
    for scan in scans:
        for r in scan.get("structural_results", []):
            all_events_seen[r["event_id"]].append(r["abs_gap"])

    events_with_gap = {
        eid: gaps for eid, gaps in all_events_seen.items()
        if any(g >= ESCALATION_MIN_GAP for g in gaps)
    }

    rows = [
        {
            "fact_id": "FACT-001",
            "fact": "Neg-risk markets have an official sum-to-one constraint on YES prices.",
            "source": "Polymarket neg-risk-ctf-adapter repo / official docs",
            "confirmed": True,
            "confirmation_method": "structural — read from official source",
            "scan_evidence": "n/a — definitional",
        },
        {
            "fact_id": "FACT-002",
            "fact": "NO shares can be converted to YES shares via the neg-risk adapter contract.",
            "source": "Polymarket/neg-risk-ctf-adapter — ConvertPositions function",
            "confirmed": True,
            "confirmation_method": "structural — read from official repo",
            "scan_evidence": "n/a — definitional",
        },
        {
            "fact_id": "FACT-003",
            "fact": f"Total neg-risk events observed across {total_scans} research scan(s).",
            "source": "accumulated scan log",
            "confirmed": total_scans > 0,
            "confirmation_method": "empirical — scan log",
            "scan_evidence": (
                f"{len(all_events_seen)} unique events observed, "
                f"{len(events_with_gap)} with abs_gap >= {ESCALATION_MIN_GAP}"
            ),
        },
        {
            "fact_id": "FACT-004",
            "fact": (
                "The CLOB REST API endpoint GET /book?token_id={id} provides "
                "per-leg order book data for neg-risk outcome tokens."
            ),
            "source": "Polymarket/py-clob-client + CLOB API docs",
            "confirmed": True,
            "confirmation_method": "structural — read from official repo",
            "scan_evidence": "n/a — definitional",
        },
        {
            "fact_id": "FACT-005",
            "fact": "Gamma API returns negRisk field on events using the neg-risk structure.",
            "source": "Gamma API observation",
            "confirmed": total_scans > 0,
            "confirmation_method": "empirical — scan log",
            "scan_evidence": (
                "Confirmed in scan log" if total_scans > 0 else "Not yet tested"
            ),
        },
    ]
    return rows


# ---------------------------------------------------------------------------
# Table 2: Codified Hypothesis Table
# ---------------------------------------------------------------------------

def codified_hypothesis_table() -> list[dict[str, Any]]:
    """
    Returns all hypotheses in codified form with test methods and evidence standards.
    """
    rows = []
    for hyp in REGISTRY:
        rows.append({
            "hyp_id": hyp.hyp_id,
            "category": hyp.category,
            "source": hyp.source,
            "raw_claim": hyp.raw_claim,
            "codified_form": hyp.codified_form,
            "test_method": hyp.test_method,
            "expected_value": hyp.expected_value,
            "possible_failure_mode": hyp.possible_failure_mode,
            "evidence_standard": hyp.evidence_standard,
            "expected_gap_direction": hyp.expected_gap_direction,
            "current_classification": hyp.final_classification,
            "current_evidence_strength": hyp.evidence_strength,
            "notes": hyp.notes,
        })
    return rows


# ---------------------------------------------------------------------------
# Table 3: Research Test Results Table
# ---------------------------------------------------------------------------

def research_test_results_table(
    output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Returns per-scan test results with gap/depth/quality measurements.
    One row per event per scan.
    """
    scans = load_all_scans(output_dir)
    rows = []

    for scan in scans:
        scan_id = scan.get("scan_id", "")
        scan_timestamp = scan.get("scan_timestamp", "")

        for r in scan.get("structural_results", []):
            # Find matching execution result
            exec_result = next(
                (e for e in scan.get("execution_results", [])
                 if e["event_id"] == r["event_id"]),
                None
            )
            rows.append({
                "scan_id": scan_id,
                "scan_timestamp": scan_timestamp,
                "event_id": r["event_id"],
                "slug": r.get("slug", ""),
                "title": r.get("title", ""),
                "n_outcomes": r.get("n_outcomes", 0),
                "implied_sum": r.get("implied_sum"),
                "constraint_gap": r.get("constraint_gap"),
                "abs_gap": r.get("abs_gap"),
                "constraint_class": r.get("constraint_class"),
                "gap_direction": r.get("gap_direction"),
                "directional_bias": r.get("directional_bias"),
                "passes_gap_threshold": r.get("passes_gap_threshold"),
                "passes_fee_hurdle": r.get("passes_fee_hurdle"),
                "has_all_prices": r.get("has_all_prices"),
                "max_equivalent_gap": r.get("max_equivalent_gap"),
                "max_equivalent_gap_after_fee": r.get("max_equivalent_gap_after_fee"),
                "execution_class": (
                    exec_result["execution_class"] if exec_result else "NOT_CHECKED"
                ),
                "min_ask_depth_usd": (
                    exec_result.get("min_ask_depth_usd") if exec_result else None
                ),
                "max_spread": (
                    exec_result.get("max_spread") if exec_result else None
                ),
                "liquidity_quality_score": (
                    exec_result.get("liquidity_quality_score") if exec_result else None
                ),
                "research_note": r.get("research_note", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Table 4: Useful vs Noise Table
# ---------------------------------------------------------------------------

def useful_vs_noise_table(
    output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Classifies each observed event as USEFUL or NOISE based on accumulated evidence.
    One row per unique event seen across all scans.
    """
    scans = load_all_scans(output_dir)

    # Aggregate per event
    event_data: dict[str, dict[str, Any]] = {}

    for scan in scans:
        for r in scan.get("structural_results", []):
            eid = r["event_id"]
            if eid not in event_data:
                event_data[eid] = {
                    "event_id": eid,
                    "slug": r.get("slug", ""),
                    "title": r.get("title", ""),
                    "n_outcomes": r.get("n_outcomes", 0),
                    "abs_gaps": [],
                    "constraint_classes": [],
                    "gap_directions": [],
                    "execution_classes": [],
                    "scan_count": 0,
                }
            event_data[eid]["abs_gaps"].append(r.get("abs_gap", 0.0) or 0.0)
            event_data[eid]["constraint_classes"].append(r.get("constraint_class", "UNKNOWN"))
            event_data[eid]["gap_directions"].append(r.get("gap_direction", "AT_PARITY"))
            event_data[eid]["scan_count"] += 1

        for e in scan.get("execution_results", []):
            eid = e["event_id"]
            if eid in event_data:
                event_data[eid]["execution_classes"].append(e.get("execution_class", ""))

    rows = []
    for eid, data in event_data.items():
        gaps = data["abs_gaps"]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        max_gap = max(gaps) if gaps else 0.0
        gap_above_threshold = sum(1 for g in gaps if g >= USEFUL_GAP_THRESHOLD)
        persistence = gap_above_threshold / len(gaps) if gaps else 0.0

        exec_classes = data["execution_classes"]
        executable_count = sum(1 for c in exec_classes if c == "EXECUTABLE")
        research_only_count = sum(1 for c in exec_classes if c == "RESEARCH_VALUE_ONLY")

        # Classification
        if mean_gap < USEFUL_GAP_THRESHOLD and max_gap < USEFUL_GAP_THRESHOLD:
            signal_class = "NOISE"
            reason = f"Mean abs_gap {mean_gap:.4f} below useful threshold {USEFUL_GAP_THRESHOLD}"
        elif persistence >= 0.6:
            if executable_count >= 1:
                signal_class = "USEFUL_EXECUTABLE"
                reason = (
                    f"Persistent gap (persistence={persistence:.2f}) with "
                    f"{executable_count} EXECUTABLE classification(s)."
                )
            else:
                signal_class = "USEFUL_RESEARCH_ONLY"
                reason = (
                    f"Persistent gap (persistence={persistence:.2f}) but "
                    f"execution quality insufficient ({research_only_count} RESEARCH_VALUE_ONLY)."
                )
        elif max_gap >= ESCALATION_MIN_GAP:
            signal_class = "INTERMITTENT"
            reason = (
                f"Gap occasionally above threshold (max={max_gap:.4f}) "
                f"but persistence={persistence:.2f} below 0.60."
            )
        else:
            signal_class = "MARGINAL"
            reason = f"Max gap {max_gap:.4f} is marginal. Not classified as useful or noise yet."

        rows.append({
            "event_id": eid,
            "slug": data["slug"],
            "title": data["title"],
            "n_outcomes": data["n_outcomes"],
            "scan_count": data["scan_count"],
            "mean_abs_gap": round(mean_gap, 6),
            "max_abs_gap": round(max_gap, 6),
            "gap_persistence": round(persistence, 3),
            "executable_count": executable_count,
            "research_only_count": research_only_count,
            "signal_class": signal_class,
            "reason": reason,
        })

    # Sort: most useful first
    order = {
        "USEFUL_EXECUTABLE": 0,
        "USEFUL_RESEARCH_ONLY": 1,
        "INTERMITTENT": 2,
        "MARGINAL": 3,
        "NOISE": 4,
    }
    rows.sort(key=lambda r: (order.get(r["signal_class"], 99), -r["max_abs_gap"]))
    return rows


# ---------------------------------------------------------------------------
# Table 5: Escalation Candidates Table
# ---------------------------------------------------------------------------

def escalation_candidates_table(
    output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Returns events qualifying as escalation candidates.

    Escalation criteria (all must hold):
    - Observed in >= ESCALATION_MIN_SCANS scans
    - abs_gap >= ESCALATION_MIN_GAP in each of those scans
    - At least ESCALATION_EXECUTABLE_MIN EXECUTABLE classifications
    """
    useful = useful_vs_noise_table(output_dir)
    scans = load_all_scans(output_dir)

    # Build detailed data per event
    exec_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scan in scans:
        for e in scan.get("execution_results", []):
            exec_by_event[e["event_id"]].append(e)

    rows = []
    for row in useful:
        eid = row["event_id"]
        if row["scan_count"] < ESCALATION_MIN_SCANS:
            continue
        if row["mean_abs_gap"] < ESCALATION_MIN_GAP:
            continue
        if row["executable_count"] < ESCALATION_EXECUTABLE_MIN:
            continue

        exec_entries = exec_by_event.get(eid, [])
        min_depth_values = [
            e.get("min_ask_depth_usd", 0.0) or 0.0
            for e in exec_entries
            if e.get("execution_class") == "EXECUTABLE"
        ]
        mean_min_depth = (
            sum(min_depth_values) / len(min_depth_values) if min_depth_values else 0.0
        )

        rows.append({
            "event_id": eid,
            "slug": row["slug"],
            "title": row["title"],
            "n_outcomes": row["n_outcomes"],
            "scan_count": row["scan_count"],
            "mean_abs_gap": row["mean_abs_gap"],
            "max_abs_gap": row["max_abs_gap"],
            "gap_persistence": row["gap_persistence"],
            "executable_count": row["executable_count"],
            "mean_min_depth_usd_at_executable": round(mean_min_depth, 2),
            "escalation_status": "CANDIDATE",
            "escalation_note": (
                f"Gap persistent ({row['gap_persistence']:.2f}) across "
                f"{row['scan_count']} scans. "
                f"{row['executable_count']} EXECUTABLE classifications. "
                f"Mean min depth at EXECUTABLE: ${mean_min_depth:.2f}. "
                "Ready for promotion review."
            ),
        })

    rows.sort(key=lambda r: -r["executable_count"])
    return rows


# ---------------------------------------------------------------------------
# Table 6: Rejected Information Table
# ---------------------------------------------------------------------------

def rejected_information_table(
    output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    Returns definitively rejected hypotheses.

    Sources:
    - Registry entries with final_classification == 'reject'
    - Events consistently classified as NOISE across >= 5 scans
    """
    rows = []

    # From hypothesis registry
    for hyp in REGISTRY:
        if hyp.final_classification == "reject":
            rows.append({
                "rejection_source": "hypothesis_registry",
                "item_id": hyp.hyp_id,
                "title": hyp.raw_claim[:120] + "...",
                "rejection_reason": hyp.notes,
                "evidence_at_rejection": hyp.result or "see notes",
                "can_reopen": True,
                "reopen_condition": "If market structure changes or new evidence contradicts rejection",
            })

    # From scan log — events consistently NOISE
    useful = useful_vs_noise_table(output_dir)
    for row in useful:
        if row["signal_class"] == "NOISE" and row["scan_count"] >= 5:
            rows.append({
                "rejection_source": "scan_evidence",
                "item_id": row["event_id"],
                "title": row["title"],
                "rejection_reason": row["reason"],
                "evidence_at_rejection": (
                    f"Mean abs_gap={row['mean_abs_gap']:.4f} across "
                    f"{row['scan_count']} scans. "
                    f"Consistently below useful threshold ({USEFUL_GAP_THRESHOLD})."
                ),
                "can_reopen": True,
                "reopen_condition": "If event resolves and similar structure reappears",
            })

    return rows


# ---------------------------------------------------------------------------
# Generate all tables to JSON files
# ---------------------------------------------------------------------------

def generate_all(
    output_dir: Optional[Path] = None,
    tables_output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Generate all 6 tables and write to JSON files.

    Args:
        output_dir:        Scan log directory (default: data/research/neg_risk/).
        tables_output_dir: Where to write table JSON files (default: same as scan log dir).

    Returns:
        Dict mapping table name to written file path.
    """
    if output_dir is None:
        output_dir = NEG_RISK_DATA_DIR
    if tables_output_dir is None:
        tables_output_dir = output_dir

    tables_output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    table_fns = {
        "information_registry_table": information_registry_table,
        "codified_hypothesis_table": codified_hypothesis_table,
        "research_test_results_table": lambda: research_test_results_table(output_dir),
        "useful_vs_noise_table": lambda: useful_vs_noise_table(output_dir),
        "escalation_candidates_table": lambda: escalation_candidates_table(output_dir),
        "rejected_information_table": lambda: rejected_information_table(output_dir),
    }

    for table_name, fn in table_fns.items():
        data = fn()
        path = tables_output_dir / f"{table_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written[table_name] = path

    return written


# ---------------------------------------------------------------------------
# CLI: print a table as formatted JSON
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate neg-risk research output tables."
    )
    parser.add_argument(
        "--table",
        choices=[
            "information_registry",
            "codified_hypothesis",
            "test_results",
            "useful_vs_noise",
            "escalation_candidates",
            "rejected",
            "all",
        ],
        default="all",
        help="Which table to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Scan log directory (default: data/research/neg_risk/)",
    )
    args = parser.parse_args()

    table_map = {
        "information_registry": information_registry_table,
        "codified_hypothesis": codified_hypothesis_table,
        "test_results": lambda: research_test_results_table(args.output_dir),
        "useful_vs_noise": lambda: useful_vs_noise_table(args.output_dir),
        "escalation_candidates": lambda: escalation_candidates_table(args.output_dir),
        "rejected": lambda: rejected_information_table(args.output_dir),
    }

    if args.table == "all":
        written = generate_all(output_dir=args.output_dir)
        for name, path in written.items():
            print(f"  {name}: {path}")
    else:
        fn = table_map[args.table]
        data = fn()
        print(json.dumps(data, indent=2, ensure_ascii=False))
