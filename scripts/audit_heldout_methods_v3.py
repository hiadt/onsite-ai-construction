"""Post-hoc descriptive comparison of frozen held-out route scores (no retraining)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/final/tables/engineering_heldout_scores.csv"
OUT = ROOT / "reports/final/tables/heldout_methods_v3.csv"
SUMMARY = ROOT / "reports/final/heldout_methods_v3_summary.json"


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positive = y == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-score, kind="stable")
    ordered_score, ordered_y = score[order], y[order]
    end = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), len(y) - 1]
    cumulative = np.cumsum(ordered_y)
    previous_tp = 0
    ap = 0.0
    for index in end:
        true_positive = int(cumulative[index])
        ap += (true_positive - previous_tp) / positives * true_positive / (index + 1)
        previous_tp = true_positive
    return float(ap)


def main() -> None:
    frame = pd.read_csv(SOURCE)
    frame["failure"] = (frame["true_label"].astype(int) == 0).astype(int)
    frame["naive_uncalibrated_50_50_hybrid"] = (
        frame["model_score"] + frame["frozen_geometry_rule_score"]
    ) / 2
    methods = {
        "learning_model": "model_score",
        "frozen_geometry_rule": "frozen_geometry_rule_score",
        "naive_uncalibrated_50_50_hybrid": "naive_uncalibrated_50_50_hybrid",
        "minimum_clearance": "minimum_clearance_score",
    }
    rows = []
    for method, score_column in methods.items():
        ranked = frame.sort_values([score_column, "sample_id"], ascending=[False, True], kind="mergesort")
        y = frame["failure"].to_numpy()
        score = frame[score_column].to_numpy(float)
        failures = int(y.sum())
        for k in (10, 20, 34):
            selected = ranked.head(k)
            captured = int(selected["failure"].sum())
            rows.append({
                "method": method,
                "n": len(frame), "failures": failures, "budget": k,
                "captured_failures": captured,
                "failure_capture_rate": captured / failures,
                "precision_at_k": captured / min(k, len(frame)),
                "random_expected_captured": min(k, len(frame)) * failures / len(frame),
                "roc_auc_failure": roc_auc(y, score),
                "pr_auc_failure": average_precision(y, score),
                "tie_break": "sample_id ascending",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    summary = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(frame), "held_out_maps": sorted(frame["map_id"].astype(str).unique().tolist()),
        "failures": int(frame["failure"].sum()),
        "label_convention": "true_label=1 pass, 0 failure",
        "hybrid_definition": "0.5 × model failure-risk score + 0.5 × frozen development-reference engineering stress score",
        "hybrid_status": "uncalibrated retrospective comparator only; not used by product or for threshold/model selection",
        "independence": "The held-out maps and scores have already been inspected in prior work. This is a post-hoc re-tabulation, not a new independent validation.",
        "limitations": [
            "The blended rule percentile is not calibrated as a failure probability.",
            "Global held-out Top-K is not same-planner-batch ranking.",
            "No conclusion of model superiority or field safety follows from this table.",
        ],
        "table": str(OUT.relative_to(ROOT)).replace("\\", "/"),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
