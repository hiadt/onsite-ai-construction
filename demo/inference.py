"""Strict model loading and feature validation for the PathGuard demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class FeatureValidationError(ValueError):
    """Raised when uploaded data does not match the frozen feature contract."""


def load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(f"特征契约不存在：{contract_path.name}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {"feature_version", "training_feature_columns", "schema_sha256"}
    missing = sorted(required - set(contract))
    if missing:
        raise FeatureValidationError(f"特征契约缺少字段：{', '.join(missing)}")
    if len(contract["training_feature_columns"]) != 55:
        raise FeatureValidationError("特征契约不是预期的55维标准。")
    return contract


def validate_feature_frame(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        raise FeatureValidationError("输入CSV没有数据行。")
    feature_columns = list(contract["training_feature_columns"])
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise FeatureValidationError("缺少训练特征：" + ", ".join(missing))

    converted = pd.DataFrame(index=frame.index)
    invalid_columns: list[str] = []
    for column in feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            invalid_columns.append(column)
        converted[column] = numeric
    if invalid_columns:
        raise FeatureValidationError("字段无法转换为有限数值：" + ", ".join(invalid_columns))

    values = converted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [
            feature_columns[index]
            for index in np.unique(np.where(~np.isfinite(values))[1]).tolist()
        ]
        raise FeatureValidationError("字段包含NaN或无穷值：" + ", ".join(bad))
    return converted


def load_gate3_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            "Gate 3模型文件不存在。请由项目负责人提供 "
            "demo/models/pathguard_gate3_model.joblib；系统不会使用伪造模型替代。"
        )
    artifact = joblib.load(model_path)
    metadata: dict[str, Any] = {}
    model = artifact
    if isinstance(artifact, dict):
        model = artifact.get("model")
        if model is None:
            model = artifact.get("estimator")
        metadata = {key: value for key, value in artifact.items() if key not in {"model", "estimator"}}
    if model is None or not hasattr(model, "predict_proba"):
        raise TypeError("Gate 3模型必须提供 predict_proba 接口。")
    return model, metadata


def predict_failure_risk(model: Any, features: pd.DataFrame) -> np.ndarray:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != list(features.columns):
        raise FeatureValidationError("模型训练列顺序与 feature_contract_v2.json 不一致。")

    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError("模型输出形状不符合 predict_proba 约定。")
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("模型缺少 classes_，无法确认失败类别，拒绝猜测概率列。")
    class_values = list(np.asarray(classes).tolist())
    if 0 not in class_values:
        raise ValueError("模型 classes_ 不包含失败标签0，无法计算失败风险。")
    failure_risk = probabilities[:, class_values.index(0)]
    if not np.isfinite(failure_risk).all() or ((failure_risk < 0) | (failure_risk > 1)).any():
        raise ValueError("模型输出包含无效概率。")
    return failure_risk


def model_display_name(model: Any, metadata: dict[str, Any]) -> str:
    return str(metadata.get("model_type") or type(model).__name__)
