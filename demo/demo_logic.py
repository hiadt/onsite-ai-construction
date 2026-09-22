"""Pure data preparation helpers used by the Streamlit demo and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from feature_explain import evidence_status, next_action, risk_level, top_reasons, validation_priority
    from inference import FeatureValidationError, ModelLoadError, load_gate3_model, model_display_name, predict_failure_risk, validate_feature_frame
    from rule_baseline import compute_geometry_rule_risk
except ModuleNotFoundError:  # package import when Streamlit loads demo/app.py from repo root
    from demo.feature_explain import evidence_status, next_action, risk_level, top_reasons, validation_priority
    from demo.inference import FeatureValidationError, ModelLoadError, load_gate3_model, model_display_name, predict_failure_risk, validate_feature_frame
    from demo.rule_baseline import compute_geometry_rule_risk


METADATA_DEFAULTS = {
    "sample_id": "",
    "route_file_sha256": "",
    "map_id": "unknown",
    "route_id": "unknown",
    "label_status": "unlabeled",
    "true_label": "",
    "vehicle_structure": "unknown",
    "vehicle_structure_evidence": "unknown",
    "vehicle_axis_count": "",
    "source_machine": "unknown",
    "match_status": "unmatched_candidate",
    "vehicle_configuration_fingerprint_summary": "",
    "feature_version": "",
    "feature_source": "offline_derived_features",
    "feature_missing_reason": "",
}


def normalize_metadata(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    normalized = frame.copy()
    for identity_column, expected in (
        ("feature_version", contract["feature_version"]),
        ("schema_sha256", contract["schema_sha256"]),
    ):
        if identity_column in normalized.columns:
            provided = normalized[identity_column].dropna().astype(str).str.strip()
            provided = provided[provided.ne("")]
            if not provided.empty and not provided.eq(expected).all():
                raise FeatureValidationError(f"输入中的{identity_column}与冻结特征契约不一致。")
    for column in contract.get("metadata_columns", []):
        if column not in normalized.columns:
            normalized[column] = METADATA_DEFAULTS.get(column, "")
    normalized["feature_version"] = contract["feature_version"]
    if "sample_id" not in normalized or normalized["sample_id"].astype(str).str.strip().eq("").any():
        normalized["sample_id"] = [f"UPLOAD-{index + 1:04d}" for index in range(len(normalized))]
    normalized["vehicle_structure"] = (
        normalized["vehicle_structure"].astype(str).where(
            normalized["vehicle_structure"].astype(str).isin(["five_axis", "six_axis", "unknown"]),
            "unknown",
        )
    )
    return normalized


def evaluate_candidates(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    model_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared = normalize_metadata(frame, contract)
    features = validate_feature_frame(prepared, contract)
    rule_risk, components = compute_geometry_rule_risk(features)
    result = prepared.copy()
    result["rule_risk"] = rule_risk

    state: dict[str, Any] = {
        "model_available": False,
        "model_name": "待提供",
        "model_error": "",
    }
    try:
        model, metadata = load_gate3_model(model_path, contract)
        model_risk = predict_failure_risk(model, features)
        result["model_risk"] = model_risk
        # The learned model is the queue score. The rule is an independent
        # engineering cross-check; it is deliberately not blended into a
        # pseudo-probability whose weight would need separate calibration.
        result["combined_risk"] = result["model_risk"]
        if not np.isfinite(result[["model_risk", "rule_risk"]].to_numpy()).all():
            raise ValueError("风险输出包含NaN或无穷值。")
        result["risk_level"] = result["model_risk"].map(risk_level)
        result["decision_status"] = result.apply(
            lambda row: "模型/规则冲突，人工复核"
            if abs(float(row["model_risk"]) - float(row["rule_risk"])) >= 0.25
            else "模型与工程规则方向一致",
            axis=1,
        )
        unknown = result["vehicle_structure"].eq("unknown")
        result.loc[unknown, "risk_level"] = "证据不足"
        result.loc[unknown, "decision_status"] = "证据不足，拒绝确定结论"
        state.update(
            model_available=True,
            model_name=model_display_name(model, metadata),
            model_metadata=metadata,
        )
    except (FileNotFoundError, ModelLoadError, TypeError, ValueError) as error:
        state["model_error"] = str(error)
        result["model_risk"] = pd.NA
        result["combined_risk"] = pd.NA
        result["risk_level"] = "模型待补"
        result["decision_status"] = "仅规则预览，等待模型"
        unknown = result["vehicle_structure"].eq("unknown")
        result.loc[unknown, "risk_level"] = "证据不足"
        result.loc[unknown, "decision_status"] = "证据不足，拒绝确定结论"

    result["risk_reasons"] = [
        "；".join(top_reasons(components.loc[index])) or "当前批次内未发现突出的规则应力项"
        for index in result.index
    ]
    result["evidence_status"] = result.apply(evidence_status, axis=1)
    score_column = "model_risk" if state["model_available"] else "rule_risk"
    result["next_action"] = [
        next_action(float(result.loc[index, score_column]), result.loc[index], state["model_available"])
        for index in result.index
    ]
    if state["model_available"]:
        conflict = result["decision_status"].eq("模型/规则冲突，人工复核") & result["vehicle_structure"].ne("unknown")
        result.loc[conflict, "next_action"] = "人工复核（模型与规则冲突）"
    result["validation_priority"] = [
        validation_priority(
            float(result.loc[index, score_column]),
            str(result.loc[index, "vehicle_structure"]),
            state["model_available"],
        )
        for index in result.index
    ]
    if state['model_available']:
        result.loc[result['decision_status'].eq('模型/规则冲突，人工复核'), 'validation_priority'] = 'P0 人工复核'
    return result, components, state


def apply_filters(
    frame: pd.DataFrame,
    structure: str,
    risk: str,
    map_query: str,
    search: str,
) -> pd.DataFrame:
    filtered = frame.copy()
    if structure != "all":
        filtered = filtered[filtered["vehicle_structure"] == structure]
    if risk != "全部" and "risk_level" in filtered:
        filtered = filtered[filtered["risk_level"] == risk]
    if map_query != "全部":
        filtered = filtered[filtered["map_id"].astype(str) == map_query]
    if search.strip():
        needle = search.strip().lower()
        haystack = (
            filtered["sample_id"].astype(str)
            + " "
            + filtered["route_id"].astype(str)
            + " "
            + filtered["route_file_sha256"].astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(needle, regex=False)]
    return filtered


def select_top_k(frame: pd.DataFrame, selection: str, model_available: bool) -> pd.DataFrame:
    score_column = "model_risk" if model_available else "rule_risk"
    ordered = frame.sort_values(score_column, ascending=False)
    if selection == "Top 10%":
        count = max(1, int(len(ordered) * 0.1 + 0.9999)) if len(ordered) else 0
    else:
        count = min(int(selection.split()[-1]), len(ordered))
    return ordered.head(count)
