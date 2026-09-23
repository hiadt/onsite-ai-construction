"""Descriptive Top-K audit for an actual task's comparable verified candidates."""
from __future__ import annotations

import pandas as pd


def evaluate_verified_task(evaluated: pd.DataFrame, feedback: pd.DataFrame,
                           task: dict) -> dict:
    if feedback.empty or "task_id" not in feedback:
        return {"status": "待积累", "reason": "本任务暂无验证结果。"}
    eligible = feedback[
        feedback["task_id"].eq(task["task_id"])
        & feedback["environment_version"].eq(task["environment_version"])
        & feedback["condition_consistency"].eq("与本次评估一致")
        & feedback["review_status"].eq("已核对")
        & feedback["validation_result"].isin(["通过", "失败"])
    ].copy()
    if eligible.empty:
        return {"status": "待积累", "reason": "尚无同场景版本、条件一致且已核对的验证结论。"}
    labels = []
    excluded_conflicts = 0
    for route_hash, group in eligible.groupby("route_sha256"):
        if group["validation_result"].nunique() != 1:
            excluded_conflicts += 1
            continue
        labels.append({"route_file_sha256": route_hash,
                       "failure": int(group.iloc[0]["validation_result"] == "失败"),
                       "severe_failure": int(group.iloc[0]["validation_result"] == "失败" and
                                             group["severity"].eq("严重").any())})
    if not labels:
        return {"status": "待积累", "reason": "同条件验证结论存在争议，尚无可比较标签。",
                "excluded_conflicts": excluded_conflicts}
    labeled = evaluated.merge(pd.DataFrame(labels), on="route_file_sha256", how="inner")
    labeled = labeled.drop_duplicates("route_file_sha256")
    n = len(labeled); failures = int(labeled["failure"].sum())
    if n < 5 or failures < 2 or failures == n:
        return {"status": "样本不足", "reason": "至少需要同任务5条已核对候选，且同时包含通过与至少2条失败。",
                "verified_candidates": n, "failures": failures,
                "excluded_conflicts": excluded_conflicts}
    budgets = sorted(set((1, min(3, n), min(10, n))))
    rows = []
    for k in budgets:
        for method, column in (("学习模型", "model_risk"), ("工程规则", "rule_risk")):
            if column not in labeled or pd.to_numeric(labeled[column], errors="coerce").isna().any():
                continue
            top = labeled.sort_values(column, ascending=False).head(k)
            rows.append({"验证名额": k, "排序方式": method,
                         "失败捕获": int(top["failure"].sum()),
                         "严重失败捕获": int(top["severe_failure"].sum()),
                         "严重失败漏掉": int(labeled["severe_failure"].sum()-top["severe_failure"].sum()),
                         "随机期望失败捕获": round(k*failures/n, 2)})
    return {"status": "描述性结果", "verified_candidates": n, "failures": failures,
            "excluded_conflicts": excluded_conflicts, "rows": rows,
            "note": "同任务描述性Top-K对照；样本较少时不代表跨地图泛化或显著优势。"}
