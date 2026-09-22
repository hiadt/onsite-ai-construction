"""Evaluate within-context candidate ranking on the frozen held-out predictions.

The frozen data do not contain a true planner invocation / candidate-batch key.
Map + confirmed vehicle structure is therefore reported as a proxy context only;
it must not be described as a real deployment batch evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "final" / "tables" / "engineering_heldout_scores.csv"
OUT = ROOT / "reports" / "final" / "tables" / "candidate_batch_proxy_metrics.csv"
SUMMARY = ROOT / "reports" / "final" / "candidate_batch_proxy_summary.json"
SCORES = {
    "model_score": "学习模型",
    "frozen_geometry_rule_score": "冻结几何规则",
    "minimum_clearance_score": "最小净空基线",
}


def tie_aware_top_fraction(group: pd.DataFrame, score: str, fraction: float) -> float:
    """Expected failures captured when tied scores at the cutoff share places."""
    group = group.copy()
    group["failure"] = 1 - group["true_label"].astype(int)  # frozen label: 1=pass, 0=failure
    k = max(1, math.ceil(len(group) * fraction))
    ordered = group.sort_values(score, ascending=False, kind="mergesort")
    cutoff = float(ordered.iloc[k - 1][score])
    above = ordered[ordered[score] > cutoff]
    tied = ordered[ordered[score] == cutoff]
    remaining = max(0, k - len(above))
    tie_fraction = remaining / len(tied) if len(tied) else 0.0
    return float(above["failure"].sum() + tied["failure"].sum() * tie_fraction)


def main() -> None:
    frame = pd.read_csv(SOURCE)
    # Confirmed structure only. Unknown structures are held for review and do
    # not form a valid same-vehicle comparison context.
    frame = frame[frame["vehicle_structure"].isin(["five_axis", "six_axis"])].copy()
    groups = [g for _, g in frame.groupby(["map_id", "vehicle_structure"], sort=True) if len(g) >= 2]
    frame["failure"] = 1 - frame["true_label"].astype(int)  # label convention is 1=pass, 0=failure
    total_failures = int(frame["failure"].sum())
    rows: list[dict[str, object]] = []
    for name, label in SCORES.items():
        expected_captured = sum(tie_aware_top_fraction(group, name, 0.20) for group in groups)
        random_expected = sum(
            math.ceil(len(group) * 0.20) * float((1 - group["true_label"].astype(int)).mean())
            for group in groups
        )
        rows.append({
            "method": label,
            "proxy_groups": len(groups),
            "included_candidates": int(sum(map(len, groups))),
            "failures_in_included_groups": int(sum(int((1 - g["true_label"].astype(int)).sum()) for g in groups)),
            "top_fraction": 0.20,
            "expected_failures_captured_tie_aware": round(expected_captured, 4),
            "capture_rate": round(expected_captured / total_failures, 4) if total_failures else None,
            "random_expected_failures_captured_same_group_budgets": round(random_expected, 4),
            "random_expected_capture_rate": round(random_expected / total_failures, 4) if total_failures else None,
        })
    group_rows = []
    for (map_id, structure), group in frame.groupby(["map_id", "vehicle_structure"], sort=True):
        group_rows.append({
            "map_id": map_id,
            "vehicle_structure": structure,
            "candidate_count": len(group),
            "failures": int((1 - group["true_label"].astype(int)).sum()),
            "eligible_for_top_fraction_proxy": len(group) >= 2,
            **{f"{score}_unique_scores": int(group[score].nunique()) for score in SCORES},
        })
    pd.DataFrame(group_rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    summary = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_rows_confirmed_vehicle_structure": len(frame),
        "label_convention": "true_label=1 means pass; true_label=0 means failure; failure is 1 - true_label",
        "total_failures_confirmed_vehicle_structure": total_failures,
        "proxy_context": "map_id × confirmed vehicle_structure; not a true planner invocation batch",
        "proxy_groups_with_at_least_two_candidates": len(groups),
        "proxy_candidates_in_eligible_groups": int(sum(map(len, groups))),
        "proxy_failures_in_eligible_groups": int(sum(int((1 - g["true_label"].astype(int)).sum()) for g in groups)),
        "selection_budget": "ceil(20% of each eligible proxy group), ties split fractionally",
        "results": rows,
        "limitations": [
            "No planner invocation or candidate-batch identifier exists in the frozen held-out table.",
            "Map × vehicle structure is only a proxy context and may still combine unrelated runs.",
            "Only groups with at least two candidates are eligible; small sample and few failures make estimates unstable.",
            "Unknown vehicle structures are excluded from ranking comparison and should enter evidence review.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
