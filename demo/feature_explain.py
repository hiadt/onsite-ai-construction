"""Risk labels, evidence wording and feature-grounded explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd


RISK_REASON_TEXT = {
    "曲率与变化": "局部曲率或曲率变化在当前候选批次中偏高",
    "速度曲率耦合": "计划速度与曲率耦合应力在当前候选批次中偏高",
    "转向应力": "转向前馈幅值或单位距离变化在当前候选批次中偏高",
    "横摆应力": "计划横摆率或横摆率变化在当前候选批次中偏高",
    "坡度变化": "路线坡度变化在当前候选批次中偏高",
    "静态边界余量": "静态左右边界余量在当前候选批次中偏小",
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
        return "结构未定"
    if label_status in {"labeled", "final"} and "npz" in feature_source:
        return "本机恢复数据派生结果（含冻结标签）"
    if label_status in {"labeled", "final"}:
        return "final真实标签"
    return "无标签指标池"


def next_action(score: float, row: pd.Series, model_available: bool) -> str:
    if str(row.get("vehicle_structure", "unknown")) == "unknown":
        return "客户共创确认"
    if not model_available:
        return "人工复核"
    if score >= 0.80 and not str(row.get("true_label", "")).strip():
        return "优先补测"
    if score >= 0.70:
        return "优先仿真复核"
    if score >= 0.45:
        return "人工复核"
    if not str(row.get("true_label", "")).strip():
        return "客户共创确认"
    return "暂不优先处理"


def label_text(value: Any) -> str:
    text = str(value).strip()
    if text == "1" or text == "1.0":
        return "通过标签"
    if text == "0" or text == "0.0":
        return "失败标签"
    return "无标签"
