"""Customer-facing labels derived from source metadata without changing identity."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


STRUCTURE_NAMES = {"five_axis": "五轴", "six_axis": "六轴", "unknown": "车型待确认"}
PLAIN_REASONS = {
    "局部曲率或曲率变化相对开发集参考分布偏高": "部分路段弯道较急或弯度变化较快",
    "计划速度与曲率耦合应力相对开发集参考分布偏高": "计划速度与转弯程度的组合值得检查",
    "转向前馈幅值或单位距离变化相对开发集参考分布偏高": "转向幅度或变化偏大",
    "计划横摆率或横摆率变化相对开发集参考分布偏高": "车辆转动变化偏大",
    "路线坡度变化相对开发集参考分布偏高": "部分路段坡度变化偏大",
    "静态左右边界余量相对开发集参考分布偏小": "输入数据提示道路边界余量偏小",
}


def plain_rule_reasons(raw: object) -> str:
    parts = [part.strip() for part in str(raw or "").split("；") if part.strip()]
    return "；".join(PLAIN_REASONS.get(part, part) for part in parts[:2]) or "目前没有突出的工程规则提示"


def scene_name(raw: object) -> str:
    value = str(raw or "").strip()
    match = re.fullmatch(r"map_(\d+)", value)
    if match:
        return f"场景 {int(match.group(1)):02d}"
    return value or "场景未命名"


def route_name(raw: object, number: int, *, historical: bool) -> str:
    if historical:
        return f"历史路线 {number:03d}"
    stem = Path(str(raw or "")).stem.strip()
    if not stem or re.fullmatch(r"map_\d+", stem):
        return f"候选路线 {number:02d}"
    return f"候选路线 {number:02d} · {stem[:42]}"


def add_display_labels(frame: pd.DataFrame, *, historical: bool, frozen_ids: list[str] | None = None) -> pd.DataFrame:
    result = frame.copy()
    ordered_ids = sorted(set(frozen_ids or []) | set(result["sample_id"].astype(str))) if historical else result["sample_id"].astype(str).tolist()
    positions = {sample_id: index + 1 for index, sample_id in enumerate(ordered_ids)}
    result["display_route"] = [
        route_name(row["route_id"], positions[str(row["sample_id"])], historical=historical)
        for _, row in result.iterrows()
    ]
    result["display_scene"] = result["map_id"].map(scene_name)
    return result


def attach_historical_labels(examples: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """Only restore example labels when both sample ID and source file hash match."""
    if "true_label" in examples:
        return examples.copy()
    identity = ["sample_id", "route_file_sha256"]
    reference = frozen[identity + ["true_label"]].drop_duplicates(identity)
    return examples.merge(reference, on=identity, how="left", validate="many_to_one")
