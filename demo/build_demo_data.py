"""Create deterministic demo inputs from committed Gate 2 frozen features."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "demo" / "data"
CONTRACT_SOURCE = ROOT / "data" / "derived_features" / "feature_contract_v2.json"
FEATURE_SOURCE = ROOT / "data" / "derived_features" / "SHANYU000" / "pre_execution_features.csv"

SAMPLE_IDS = [
    "PG-SHANYU000-250b207fe1290aa9",
    "PG-SHANYU000-5618f7d508ec6ef2",
    "PG-SHANYU000-5dfc4fa9d2ac2c08",
    "PG-SHANYU000-87c700de4d844f0d",
    "PG-SHANYU000-e5a441095d278e57",
    "PG-SHANYU000-d070ecb1b1784582",
    "PG-SHANYU000-1266a7e4ee68b5c8",
    "PG-SHANYU000-9b865d34114e5806",
    "PG-SHANYU000-ed82f3f5dcfaac0b",
    "PG-SHANYU000-a2c28197723f0ec6",
]


def main() -> None:
    DEMO_DATA.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_SOURCE.read_text(encoding="utf-8"))
    if contract["feature_version"] != "pathguard_preexec_v2.0.0":
        raise RuntimeError("Unexpected feature version")
    if len(contract["training_feature_columns"]) != 55:
        raise RuntimeError("Unexpected feature count")

    frame = pd.read_csv(FEATURE_SOURCE, dtype={"true_label": "Int64"})
    selected = frame.set_index("sample_id").loc[SAMPLE_IDS].reset_index()
    selected["demo_evidence_status"] = "本机恢复数据派生结果"
    selected.to_csv(DEMO_DATA / "sample_input.csv", index=False, encoding="utf-8", lineterminator="\n")
    shutil.copyfile(CONTRACT_SOURCE, DEMO_DATA / "feature_contract_v2.json")

    summary = {
        "project": "PathGuard",
        "value_proposition": "让每一条候选工程车辆路线，都有可解释的风险等级和下一步动作。",
        "data_snapshot": {
            "supervised_final": 340,
            "features_extracted": 437,
            "unlabeled_metric_pool": 97,
            "npz_v3_sha256_exact_matches": 451,
            "final_five_axis": 99,
            "final_six_axis": 79,
            "final_unknown_structure": 162,
            "source": "前端任务指定的最新冻结数据口径",
        },
        "feature_version": contract["feature_version"],
        "feature_count": len(contract["training_feature_columns"]),
        "schema_sha256": contract["schema_sha256"],
        "model": {
            "status": "missing",
            "expected_path": "demo/models/pathguard_gate3_model.joblib",
            "model_type": None,
        },
        "validation": {
            "independent_map_failure_pr_auc": None,
            "independent_map_roc_auc": None,
            "top_k_failure_capture": None,
            "status": "待Gate 3验证产物补充",
        },
        "sample_input": {
            "row_count": len(selected),
            "source": "data/derived_features/SHANYU000/pre_execution_features.csv",
            "coverage": ["five_axis", "six_axis", "pass_label", "fail_label"],
            "unavailable_without_gate3_outputs": ["verified_high_risk", "verified_low_risk"],
            "unavailable_in_committed_frozen_features": ["unknown_structure"],
        },
        "disclaimer": "演示样本来自离线冻结数据，不代表已经完成实车安全验证。",
    }
    (DEMO_DATA / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
