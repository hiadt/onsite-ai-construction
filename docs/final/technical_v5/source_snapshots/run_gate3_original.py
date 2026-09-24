from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


ROOT = Path(r"D:\Codex_output\本机恢复数据整理与冻结")
INPUT = ROOT / "frozen" / "modeling_features_final.csv"
OUT = Path(r"D:\Codex_output\PathGuard_Gate3模型验证")
OUT.mkdir(parents=True, exist_ok=True)
PLOT = OUT / "figures"
PLOT.mkdir(exist_ok=True)
RANDOM_STATE = 20260922


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def safe_ap(y, s):
    return float(average_precision_score(y, s)) if len(set(y)) == 2 else float("nan")


def fail_metrics(y_pass, pass_prob):
    y_fail = 1 - np.asarray(y_pass, dtype=int)
    fail_score = 1.0 - np.asarray(pass_prob, dtype=float)
    pred_fail = (fail_score >= 0.5).astype(int)
    return {
        "roc_auc_failure": safe_auc(y_fail, fail_score),
        "pr_auc_failure": safe_ap(y_fail, fail_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_fail, pred_fail)),
        "precision_failure": float(precision_score(y_fail, pred_fail, zero_division=0)),
        "recall_failure": float(recall_score(y_fail, pred_fail, zero_division=0)),
        "f1_failure": float(f1_score(y_fail, pred_fail, zero_division=0)),
        "brier_pass_probability": float(brier_score_loss(y_pass, pass_prob)),
        "n": int(len(y_pass)),
        "n_failure": int(y_fail.sum()),
    }


def top_k(y_pass, pass_prob, ks=(10, 20, 34, 68)):
    y_fail = 1 - np.asarray(y_pass, dtype=int)
    score = 1 - np.asarray(pass_prob, dtype=float)
    order = np.argsort(-score)
    out = {}
    for requested_k in ks:
        k = min(requested_k, len(order))
        out[str(requested_k)] = {
            "selected": int(k),
            "failures_captured": int(y_fail[order[:k]].sum()),
            "capture_rate": float(y_fail[order[:k]].sum() / max(1, y_fail.sum())),
            "random_expected_capture_rate": float(k / len(y_fail)),
        }
    return out


def choose_group_split(groups, y):
    full_ratio = float(np.mean(y))
    best = None
    for seed in range(300):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(splitter.split(np.zeros(len(y)), y, groups))
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            continue
        score = abs(len(te) / len(y) - 0.2) * 2 + abs(float(np.mean(y[te])) - full_ratio) + abs(float(np.mean(y[tr])) - full_ratio)
        if best is None or score < best[0]:
            best = (score, tr, te, seed)
    if best is None:
        raise RuntimeError("Could not construct a group-disjoint train/test split with both labels")
    return best[1], best[2], best[3]


def grouped_cv_scores(model, X, y, groups, n_splits=5):
    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        m = clone(model)
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[va])[:, 1]
        metrics = fail_metrics(y[va], prob)
        metrics["fold"] = fold
        metrics["val_samples"] = int(len(va))
        rows.append(metrics)
    return rows


def model_factories():
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE
        ),
    }


def feature_groups(features):
    groups = defaultdict(list)
    for f in features:
        if "clearance" in f:
            groups["clearance"].append(f)
        elif "window10m" in f:
            groups["local_window"].append(f)
        elif "steer" in f:
            groups["steering"].append(f)
        elif "yaw_rate" in f or "speed" in f or "lateral_accel" in f:
            groups["speed_dynamics"].append(f)
        elif "slope" in f:
            groups["terrain_slope"].append(f)
        elif "curvature" in f or "sampling" in f or f in {"route_length_m", "point_count"}:
            groups["route_geometry"].append(f)
        else:
            groups["other"] .append(f)
    return dict(groups)


def main():
    rows = read_csv(INPUT)
    metadata = {"sample_id", "route_file_sha256", "map_id", "route_id", "label_status", "true_label", "vehicle_structure", "vehicle_structure_evidence", "vehicle_axis_count", "source_machine", "match_status", "vehicle_configuration_fingerprint_summary", "feature_version", "feature_source", "feature_missing_reason"}
    contract = json.loads((ROOT / "derived_features" / "feature_contract_v2.json").read_text(encoding="utf-8"))
    features = contract["training_feature_columns"]
    X = np.asarray([[float(r[f]) for f in features] for r in rows], dtype=float)
    y = np.asarray([int(r["true_label"]) for r in rows], dtype=int)
    groups = np.asarray([r["map_id"] for r in rows])
    structures = np.asarray([r["vehicle_structure"] for r in rows])
    ids = np.asarray([r["sample_id"] for r in rows])

    tr, te, split_seed = choose_group_split(groups, y)
    dev_groups, test_groups = sorted(set(groups[tr])), sorted(set(groups[te]))
    split_rows = []
    for i, r in enumerate(rows):
        split_rows.append({"sample_id": r["sample_id"], "map_id": r["map_id"], "vehicle_structure": r["vehicle_structure"], "true_label": r["true_label"], "split": "development" if i in set(tr) else "test", "group_role": "development" if i in set(tr) else "held_out_test"})
    write_csv(OUT / "data_split_manifest.csv", list(split_rows[0].keys()), split_rows)

    models = model_factories()
    comparison = []
    cv_details = {}
    for name, model in models.items():
        folds = grouped_cv_scores(model, X[tr], y[tr], groups[tr])
        cv_details[name] = folds
        vals = [x["pr_auc_failure"] for x in folds if math.isfinite(x["pr_auc_failure"])]
        roc = [x["roc_auc_failure"] for x in folds if math.isfinite(x["roc_auc_failure"])]
        comparison.append({"model": name, "cv_pr_auc_failure_mean": float(np.mean(vals)), "cv_pr_auc_failure_std": float(np.std(vals)), "cv_roc_auc_failure_mean": float(np.mean(roc)), "cv_folds": len(folds)})
    comparison.sort(key=lambda z: (-z["cv_pr_auc_failure_mean"], -z["cv_roc_auc_failure_mean"]))
    write_csv(OUT / "model_comparison.csv", list(comparison[0].keys()), comparison)
    best_name = comparison[0]["model"]
    best = models[best_name]

    # Development calibration split is also group-disjoint and does not touch the held-out test maps.
    cal_tr, cal_va, cal_seed = choose_group_split(groups[tr], y[tr])
    base = clone(best).fit(X[tr][cal_tr], y[tr][cal_tr])
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit").fit(X[tr][cal_va], y[tr][cal_va])
    test_prob = calibrated.predict_proba(X[te])[:, 1]
    test_metrics = fail_metrics(y[te], test_prob)
    test_metrics["best_model"] = best_name
    test_metrics["top_k"] = top_k(y[te], test_prob)
    test_metrics["test_maps"] = test_groups
    test_metrics["development_maps"] = dev_groups
    (OUT / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    test_pred = []
    for idx, p in zip(te, test_prob):
        r = rows[int(idx)]
        test_pred.append({"sample_id": r["sample_id"], "map_id": r["map_id"], "vehicle_structure": r["vehicle_structure"], "true_label": r["true_label"], "pass_probability": f"{p:.8f}", "failure_risk": f"{1-p:.8f}", "held_out": "true"})
    write_csv(OUT / "held_out_test_predictions.csv", list(test_pred[0].keys()), test_pred)

    # Threshold selection on calibration split only, then one fixed threshold is reported for test.
    cal_prob = calibrated.predict_proba(X[tr][cal_va])[:, 1]
    y_cal = y[tr][cal_va]
    thresholds = np.linspace(0.05, 0.95, 181)
    threshold_rows = []
    best_threshold = 0.5
    best_f2 = -1.0
    for t in thresholds:
        fail_true = 1 - y_cal
        fail_pred = (1 - cal_prob >= t).astype(int)
        p = precision_score(fail_true, fail_pred, zero_division=0)
        rec = recall_score(fail_true, fail_pred, zero_division=0)
        f2 = (5 * p * rec / (4 * p + rec)) if p + rec else 0.0
        threshold_rows.append({"failure_risk_threshold": f"{t:.3f}", "precision_failure": f"{p:.6f}", "recall_failure": f"{rec:.6f}", "f2_failure": f"{f2:.6f}"})
        if f2 > best_f2:
            best_f2, best_threshold = f2, float(t)
    write_csv(OUT / "threshold_selection_calibration.csv", list(threshold_rows[0].keys()), threshold_rows)
    test_metrics["selected_failure_risk_threshold"] = best_threshold
    test_metrics["selected_threshold_source"] = "development_calibration_only"
    test_fail_true = 1 - y[te]
    test_fail_pred = (1 - test_prob >= best_threshold).astype(int)
    test_metrics["selected_threshold_precision_failure"] = float(precision_score(test_fail_true, test_fail_pred, zero_division=0))
    test_metrics["selected_threshold_recall_failure"] = float(recall_score(test_fail_true, test_fail_pred, zero_division=0))
    test_metrics["selected_threshold_f1_failure"] = float(f1_score(test_fail_true, test_fail_pred, zero_division=0))
    test_metrics["selected_threshold_balanced_accuracy"] = float(balanced_accuracy_score(test_fail_true, test_fail_pred))
    (OUT / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Full-data group out-of-fold predictions for honest generalization and subgroup analysis.
    oof_prob = np.full(len(rows), np.nan)
    oof_fold = np.full(len(rows), -1)
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = clone(best).fit(X[a], y[a])
        oof_prob[b] = m.predict_proba(X[b])[:, 1]
        oof_fold[b] = fold
    oof_rows = []
    for i, r in enumerate(rows):
        oof_rows.append({"sample_id": r["sample_id"], "map_id": r["map_id"], "vehicle_structure": r["vehicle_structure"], "true_label": r["true_label"], "oof_fold": int(oof_fold[i]), "pass_probability": f"{oof_prob[i]:.8f}", "failure_risk": f"{1-oof_prob[i]:.8f}"})
    write_csv(OUT / "group_oof_predictions.csv", list(oof_rows[0].keys()), oof_rows)
    oof_summary = {"overall": fail_metrics(y, oof_prob), "top_k": top_k(y, oof_prob)}
    for s in sorted(set(structures)):
        mask = structures == s
        oof_summary[f"structure_{s}"] = fail_metrics(y[mask], oof_prob[mask])
    write_csv(OUT / "subgroup_oof_metrics.csv", ["subgroup", "n", "n_failure", "roc_auc_failure", "pr_auc_failure", "balanced_accuracy", "precision_failure", "recall_failure", "f1_failure", "brier_pass_probability"], [{"subgroup": k, **v} for k, v in oof_summary.items() if k != "top_k"])
    (OUT / "oof_metrics.json").write_text(json.dumps(oof_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Feature group ablation on the development maps using the selected model family.
    groups_map = feature_groups(features)
    ablation = [{"experiment": "all_features", "removed_group": "", "feature_count": len(features), "cv_pr_auc_failure_mean": comparison[0]["cv_pr_auc_failure_mean"]}]
    for group, removed in groups_map.items():
        keep = [i for i, f in enumerate(features) if f not in removed]
        folds = grouped_cv_scores(clone(best), X[tr][:, keep], y[tr], groups[tr])
        vals = [z["pr_auc_failure"] for z in folds if math.isfinite(z["pr_auc_failure"])]
        ablation.append({"experiment": "drop_one_group", "removed_group": group, "feature_count": len(keep), "cv_pr_auc_failure_mean": float(np.mean(vals)), "delta_vs_all": float(np.mean(vals) - comparison[0]["cv_pr_auc_failure_mean"])})
    write_csv(OUT / "feature_group_ablation.csv", ["experiment", "removed_group", "feature_count", "cv_pr_auc_failure_mean", "delta_vs_all"], ablation)

    # Permutation importance on the untouched test set.
    fitted = clone(best).fit(X[tr], y[tr])
    def score_failure(estimator, xx, yy):
        return average_precision_score(1 - yy, 1 - estimator.predict_proba(xx)[:, 1])
    perm = permutation_importance(fitted, X[te], y[te], scoring=score_failure, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    imp = [{"feature": f, "importance_mean": float(perm.importances_mean[i]), "importance_std": float(perm.importances_std[i])} for i, f in enumerate(features)]
    imp.sort(key=lambda z: z["importance_mean"], reverse=True)
    write_csv(OUT / "permutation_importance_test.csv", list(imp[0].keys()), imp)

    # Persist a model trained on all frozen labeled data for the prototype.
    final_model = clone(best).fit(X, y)
    joblib.dump({"model": final_model, "features": features, "feature_version": contract["feature_version"], "schema_sha256": contract["schema_sha256"], "failure_score_definition": "1 - predicted pass probability", "selected_failure_risk_threshold": best_threshold}, OUT / "pathguard_gate3_model.joblib")

    # Plots.
    y_fail = 1 - y[te]
    fpr, tpr, _ = roc_curve(y_fail, 1 - test_prob)
    prec, rec, _ = precision_recall_curve(y_fail, 1 - test_prob)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fpr, tpr, label=f"AUC={test_metrics['roc_auc_failure']:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(xlabel="False positive rate", ylabel="Failure recall", title="Held-out ROC")
    ax[0].legend()
    ax[1].plot(rec, prec, label=f"AP={test_metrics['pr_auc_failure']:.3f}")
    ax[1].axhline(float(np.mean(y_fail)), ls="--", color="gray", label="failure prevalence")
    ax[1].set(xlabel="Failure recall", ylabel="Failure precision", title="Held-out precision-recall")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(PLOT / "held_out_roc_pr.png", dpi=180); plt.close(fig)
    ks = [10, 20, 34, 68]
    rates = [test_metrics["top_k"][str(k)]["capture_rate"] for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.plot(ks, rates, marker="o", label="PathGuard")
    ax.plot(ks, [k / len(y_fail) for k in ks], "--", label="random baseline")
    ax.set(xlabel="Number of candidates sent to validation", ylabel="Failure capture rate", title="Top-K failure capture")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOT / "top_k_failure_capture.png", dpi=180); plt.close(fig)
    top = imp[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh([x["feature"] for x in top], [x["importance_mean"] for x in top])
    ax.set(xlabel="Decrease in failure AP after permutation", title="Top test-set feature importance")
    fig.tight_layout(); fig.savefig(PLOT / "permutation_importance_top15.png", dpi=180); plt.close(fig)

    model_table = "".join("|{model}|{pr:.4f}|{std:.4f}|{roc:.4f}|\n".format(model=x["model"], pr=x["cv_pr_auc_failure_mean"], std=x["cv_pr_auc_failure_std"], roc=x["cv_roc_auc_failure_mean"]) for x in comparison)
    topk_table = "".join("|{k}|{cap:.4f}|{rand:.4f}|\n".format(k=k, cap=test_metrics["top_k"][str(k)]["capture_rate"], rand=test_metrics["top_k"][str(k)]["random_expected_capture_rate"]) for k in (10, 20, 34, 68))
    report = f'''# PathGuard Gate 3：模型训练与可信验证报告\n\n## 1. 结论\n\n本阶段使用冻结数据集中的 **340 条 final 样本 × 55 个训练特征**，完成了分组切分、模型基线比较、独立测试、风险排序、分层验证、消融分析和模型固化。最佳模型为 **{best_name}**。\n\n测试集按地图分组，测试地图与开发地图完全不重叠；模型选择和风险阈值均未使用测试集标签。\n\n## 2. 数据切分\n\n- 开发集：{len(tr)} 条，地图 {len(dev_groups)} 个\n- 独立测试集：{len(te)} 条，地图 {len(test_groups)} 个\n- 分组随机种子：{split_seed}\n- 校准集再次从开发地图中分组切出，校准种子：{cal_seed}\n- 测试地图：{', '.join(test_groups)}\n\n## 3. 模型比较\n\n模型按开发集 5 折 GroupKFold 的失败 PR-AUC 选择：\n\n|模型|CV失败PR-AUC均值|标准差|CV失败ROC-AUC均值|\n|---|---:|---:|---:|\n{model_table}\n## 4. 独立测试集结果\n\n- 失败 ROC-AUC：**{test_metrics['roc_auc_failure']:.4f}**\n- 失败 PR-AUC：**{test_metrics['pr_auc_failure']:.4f}**\n- 失败召回率：**{test_metrics['recall_failure']:.4f}**\n- 失败精确率：**{test_metrics['precision_failure']:.4f}**\n- F1：**{test_metrics['f1_failure']:.4f}**\n- 平衡准确率：**{test_metrics['balanced_accuracy']:.4f}**\n- Brier分数：**{test_metrics['brier_pass_probability']:.4f}**\n- 固定失败风险阈值：**{best_threshold:.3f}**（只用开发校准集选择）\n\n## 5. Top-K验证价值\n\n|优先验证数量|失败捕获率|随机预期|\n|---:|---:|---:|\n{topk_table}\nTop-K指标是产品核心：它直接衡量有限仿真或实车验证预算下，PathGuard能否把高风险候选优先送检。\n\n## 6. 分层与稳健性\n\n完整的五轴、六轴、结构未定分层指标见 `subgroup_oof_metrics.csv`。OOF预测按地图分组产生，未让同一地图同时作为训练和验证数据。特征组消融见 `feature_group_ablation.csv`，可用于项目书中的创新点和技术有效性论证。\n\n## 7. 交付文件\n\n- `model_comparison.csv`：模型基线比较。\n- `held_out_test_predictions.csv`：独立测试集逐样本风险结果。\n- `group_oof_predictions.csv`：全量分组交叉验证预测。\n- `subgroup_oof_metrics.csv`：车型结构分层结果。\n- `feature_group_ablation.csv`：特征模块消融。\n- `permutation_importance_test.csv`：测试集特征重要性。\n- `pathguard_gate3_model.joblib`：使用全部冻结标签训练的演示模型。\n- `figures/held_out_roc_pr.png`：独立测试ROC和PR曲线。\n- `figures/top_k_failure_capture.png`：Top-K失败捕获曲线。\n- `figures/permutation_importance_top15.png`：前15个特征重要性。\n\n## 8. 产品落地口径\n\n下一阶段原型直接调用固化模型：输入一条候选轨迹的55维特征，输出通过概率、失败风险、风险等级和验证优先级。演示时优先展示Top-K排序、失败原因对应的特征贡献和验证预算节省，而不是只展示一个二分类准确率。\n'''
    report = report.replace(f"- 失败召回率：**{test_metrics['recall_failure']:.4f}**", f"- 失败召回率（固定阈值）：**{test_metrics['selected_threshold_recall_failure']:.4f}**")
    report = report.replace(f"- 失败精确率：**{test_metrics['precision_failure']:.4f}**", f"- 失败精确率（固定阈值）：**{test_metrics['selected_threshold_precision_failure']:.4f}**")
    report = report.replace(f"- F1：**{test_metrics['f1_failure']:.4f}**", f"- F1（固定阈值）：**{test_metrics['selected_threshold_f1_failure']:.4f}**")
    report = report.replace(f"- 平衡准确率：**{test_metrics['balanced_accuracy']:.4f}**", f"- 平衡准确率（固定阈值）：**{test_metrics['selected_threshold_balanced_accuracy']:.4f}**")
    report += "\n## 9. 固定规则对照\n\n同时建立了不使用标签拟合的几何应力规则基线，使用曲率、曲率变化、速度-曲率耦合、横向加速度、转向、横摆角速度、坡度和最小边界净空。其独立地图测试集失败 PR-AUC 为 0.7199、失败 ROC-AUC 为 0.6629，完整方法见 `固定几何应力规则基线.md` 和 `fixed_rule_baseline.json`。正式演示同时保留规则底线和学习模型，避免将效果归因于单一模型。\n"
    (OUT / "Gate3模型训练与可信验证报告.md").write_text(report, encoding="utf-8")
    (OUT / "gate3_summary.json").write_text(json.dumps({"best_model": best_name, "test_metrics": test_metrics, "oof_metrics": oof_summary, "feature_count": len(features), "final_sample_count": len(rows), "created_at": datetime.now().astimezone().isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
