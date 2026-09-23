"""Risk labels, evidence wording and feature-grounded explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd


RISK_REASON_TEXT = {
    "曲率与变化": "局部曲率或曲率变化相对开发集参考分布偏高",
    "速度曲率耦合": "计划速度与曲率耦合应力相对开发集参考分布偏高",
    "转向应力": "转向前馈幅值或单位距离变化相对开发集参考分布偏高",
    "横摆应力": "计划横摆率或横摆率变化相对开发集参考分布偏高",
    "坡度变化": "路线坡度变化相对开发集参考分布偏高",
    "静态边界余量": "静态左右边界余量相对开发集参考分布偏小",
}


def risk_level(score: float) -> str:
    if score >= 0.70:
        return "高风险"
    if score >= 0.45:
        return "中风险"
    return "低风险"


def top_reasons(component_row: pd.Series, limit: int = 3) -> list[str]:
    ordered = component_row.sort_values(ascending=False).head(limit)
    return [RISK_REASON_TEXT[name] for name, value in ordered.items() if float(value) >= 0.45]


def evidence_status(row: pd.Series) -> str:
    structure = str(row.get("vehicle_structure", "unknown"))
    label_status = str(row.get("label_status", ""))
    feature_source = str(row.get("feature_source", ""))
    if structure == "unknown":
        return "结构未定，待补车型资料"
    label = str(row.get("true_label", "")).strip()
    has_label = label in {"0", "0.0", "1", "1.0"}
    if has_label and label_status in {"labeled", "final"}:
        return "final真实标签"
    if label_status in {"unlabeled", "metric_pool"}:
        return "无标签指标池"
    if feature_source:
        return "离线派生结果"
    return "离线派生结果（标签未随演示输入提供）"


def next_action(score: float, row: pd.Series, model_available: bool) -> str:
    if str(row.get("vehicle_structure", "unknown")) == "unknown":
        return "补齐车型配置后重新评估"
    if not model_available:
        return "学习模型不可用，人工确定验证顺序"
    if score >= 0.70:
        return "优先安排闭环仿真验证"
    if score >= 0.45:
        return "按计划安排仿真验证"
    return "常规计划验证"


def validation_priority(score: float, structure: str, model_available: bool) -> str:
    if structure == "unknown":
        return "资料待补"
    if not model_available:
        return "规则预览"
    if score >= 0.70:
        return "P0 优先验证"
    if score >= 0.45:
        return "P1 计划验证"
    return "P2 常规验证"


def label_text(value: Any) -> str:
    text = str(value).strip()
    if text == "1" or text == "1.0":
        return "通过标签"
    if text == "0" or text == "0.0":
        return "失败标签"
    return "无标签"
