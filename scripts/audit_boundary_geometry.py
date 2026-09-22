"""Audit frozen NPZ boundary coverage and run a guarded spatial-feature experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


REQUIRED_BOUNDARY = {"x_m", "y_m", "yaw_rad", "left_boundary_xyz_m", "right_boundary_xyz_m"}
SPATIAL_COLUMNS = [
    "body_clearance_min_m", "body_clearance_p05_m", "body_clearance_p50_m",
    "body_clearance_mean_m", "body_clearance_negative_ratio", "body_clearance_lt_0p5_ratio",
    "longest_body_clearance_lt_0p5_m", "corridor_width_min_m",
]


def longest_span(mask: np.ndarray, station: np.ndarray) -> float:
    if len(mask) < 2:
        return 0.0
    segment = np.maximum(np.diff(station), 0.0) * (mask[:-1] | mask[1:])
    best = current = 0.0
    for value, active in zip(segment, mask[:-1] | mask[1:]):
        current = current + float(value) if active else 0.0
        best = max(best, current)
    return best


def finite_array(archive, name: str) -> np.ndarray | None:
    if name not in archive.files:
        return None
    value = np.asarray(archive[name], dtype=float)
    return value if value.size and np.isfinite(value).all() else None


def inspect_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        center_x = finite_array(archive, "x_m")
        center_y = finite_array(archive, "y_m")
        station = finite_array(archive, "s_m")
        body = finite_array(archive, "body_clearance_m")
        left = finite_array(archive, "left_clearance_m")
        right = finite_array(archive, "right_clearance_m")
        full = REQUIRED_BOUNDARY <= keys
        semantics_ok = False
        if full and center_x is not None and center_y is not None and left is not None and right is not None:
            left_xyz = finite_array(archive, "left_boundary_xyz_m")
            right_xyz = finite_array(archive, "right_boundary_xyz_m")
            if left_xyz is not None and right_xyz is not None and left_xyz.shape[0] == len(center_x) and right_xyz.shape[0] == len(center_x):
                center = np.column_stack([center_x, center_y])
                left_error = np.max(np.abs(np.linalg.norm(left_xyz[:, :2] - center, axis=1) - left))
                right_error = np.max(np.abs(np.linalg.norm(right_xyz[:, :2] - center, axis=1) - right))
                semantics_ok = bool(max(left_error, right_error) <= 1e-5)
        feature = {name: np.nan for name in SPATIAL_COLUMNS}
        if body is not None:
            feature.update({
                "body_clearance_min_m": float(np.min(body)),
                "body_clearance_p05_m": float(np.quantile(body, 0.05)),
                "body_clearance_p50_m": float(np.quantile(body, 0.50)),
                "body_clearance_mean_m": float(np.mean(body)),
                "body_clearance_negative_ratio": float(np.mean(body < 0)),
                "body_clearance_lt_0p5_ratio": float(np.mean(body < 0.5)),
                "longest_body_clearance_lt_0p5_m": longest_span(body < 0.5, station) if station is not None and len(station) == len(body) else np.nan,
            })
        if left is not None and right is not None and len(left) == len(right):
            feature["corridor_width_min_m"] = float(np.min(left + right))
        return {
            "readable": True,
            "has_xy_yaw": {"x_m", "y_m", "yaw_rad"} <= keys,
            "has_left_right_clearance": {"left_clearance_m", "right_clearance_m"} <= keys,
            "has_body_clearance": "body_clearance_m" in keys,
            "has_full_boundary": full,
            "has_boundary_confidence": "boundary_confidence" in keys,
            "has_vehicle_profile_id": "vehicle_profile_id" in keys,
            "has_vehicle_profile_fingerprint": "vehicle_profile_fingerprint" in keys,
            "boundary_distance_semantics_verified": semantics_ok,
            **feature,
        }


def score(y: np.ndarray, probability: np.ndarray) -> dict:
    failure = 1 - y.astype(int)
    order = np.argsort(-probability)
    result = {
        "n": int(len(y)), "failures": int(failure.sum()),
        "pr_auc_failure": float(average_precision_score(failure, probability)),
        "roc_auc_failure": float(roc_auc_score(failure, probability)),
    }
    for k in (10, 20):
        chosen = failure[order[:min(k, len(order))]]
        result[f"top_{k}_captured"] = int(chosen.sum())
        result[f"top_{k}_capture_rate"] = float(chosen.sum() / max(failure.sum(), 1))
        result[f"top_{k}_precision"] = float(chosen.mean())
    return result


def grouped_experiment(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[dict, pd.DataFrame]:
    eligible = frame[frame["has_full_boundary"] & frame["boundary_distance_semantics_verified"]].copy()
    eligible = eligible.dropna(subset=feature_columns + SPATIAL_COLUMNS + ["true_label", "map_id"])
    y = eligible["true_label"].astype(int).to_numpy()
    groups = eligible["map_id"].astype(str).to_numpy()
    predictions = {"baseline_55": np.full(len(eligible), np.nan), "augmented_spatial": np.full(len(eligible), np.nan)}
    logo = LeaveOneGroupOut()
    params = dict(learning_rate=0.04, max_iter=250, max_leaf_nodes=15, min_samples_leaf=10,
                  l2_regularization=1.0, random_state=20260923, early_stopping=False)
    for train, test in logo.split(eligible, y, groups):
        for name, columns in (("baseline_55", feature_columns), ("augmented_spatial", feature_columns + SPATIAL_COLUMNS)):
            model = HistGradientBoostingClassifier(**params)
            model.fit(eligible.iloc[train][columns], y[train])
            predictions[name][test] = 1.0 - model.predict_proba(eligible.iloc[test][columns])[:, 1]
    metrics = {name: score(y, value) for name, value in predictions.items()}
    delta = {
        key: metrics["augmented_spatial"][key] - metrics["baseline_55"][key]
        for key in ("pr_auc_failure", "roc_auc_failure", "top_10_capture_rate", "top_20_capture_rate")
    }
    structures = eligible["vehicle_structure"].value_counts().to_dict()
    production_eligible = len(eligible) >= 100 and len(structures) >= 2 and delta["pr_auc_failure"] > 0.02 and delta["top_10_capture_rate"] >= 0
    summary = {
        "scope": "exploratory_map_grouped_oof",
        "eligible_samples": int(len(eligible)), "map_count": int(eligible["map_id"].nunique()),
        "vehicle_structures": {str(k): int(v) for k, v in structures.items()},
        "grouping": "LeaveOneGroupOut by map_id",
        "baseline": metrics["baseline_55"], "augmented": metrics["augmented_spatial"], "delta": delta,
        "production_eligible": production_eligible,
        "exit_conditions": {
            "minimum_samples": 100, "minimum_vehicle_structures": 2,
            "minimum_pr_auc_gain": 0.02, "top10_capture_must_not_decrease": True,
        },
        "decision": "adopt_for_production" if production_eligible else "retain_as_exploratory_evidence_only",
    }
    output = eligible[["sample_id", "map_id", "vehicle_structure", "true_label"]].copy()
    for name, value in predictions.items(): output[name + "_failure_risk"] = value
    return summary, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    features = pd.read_csv(args.features)
    manifest = manifest[manifest["sample_id"].astype(str).isin(set(features["sample_id"].astype(str)))].copy()
    feature_lookup = features.set_index("sample_id")
    rows = []
    for _, item in manifest.iterrows():
        record = {"sample_id": str(item["sample_id"]), "vehicle_structure": str(item["vehicle_structure"])}
        try:
            record.update(inspect_npz(Path(item["route_path_local"])))
        except Exception as error:
            record.update(readable=False, error_type=type(error).__name__)
        if record["sample_id"] in feature_lookup.index:
            row = feature_lookup.loc[record["sample_id"]]
            for column in ("map_id", "true_label"):
                record[column] = row[column]
        rows.append(record)
    audit = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_columns = [column for column in audit.columns if column not in {"route_path_local"}]
    audit[safe_columns].to_csv(args.output_dir / "boundary_coverage_samples.csv", index=False, encoding="utf-8-sig")
    flags = ["readable", "has_xy_yaw", "has_left_right_clearance", "has_body_clearance", "has_full_boundary",
             "has_boundary_confidence", "has_vehicle_profile_id", "has_vehicle_profile_fingerprint",
             "boundary_distance_semantics_verified"]
    by_structure = audit.groupby("vehicle_structure")[flags].sum().astype(int)
    by_structure.insert(0, "samples", audit.groupby("vehicle_structure").size())
    by_structure.to_csv(args.output_dir / "boundary_coverage_by_structure.csv", encoding="utf-8-sig")
    model_bundle = joblib.load(args.model)
    experiment, predictions = grouped_experiment(audit.merge(features, on=["sample_id", "map_id", "true_label", "vehicle_structure"]), model_bundle["features"])
    predictions.to_csv(args.output_dir / "spatial_feature_experiment_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "frozen_samples": int(len(audit)), "readable_npz": int(audit["readable"].sum()),
        "full_boundary_samples": int(audit["has_full_boundary"].sum()),
        "verified_boundary_semantics_samples": int(audit["boundary_distance_semantics_verified"].sum()),
        "body_clearance_samples": int(audit["has_body_clearance"].sum()),
        "boundary_confidence_samples": int(audit["has_boundary_confidence"].sum()),
        "vehicle_profile_id_samples": int(audit["has_vehicle_profile_id"].sum()),
        "vehicle_profile_fingerprint_samples": int(audit["has_vehicle_profile_fingerprint"].sum()),
        "coverage_by_structure": by_structure.reset_index().to_dict(orient="records"),
        "semantics": "left/right boundary coordinates reproduce centerline-to-boundary clearance within 1e-5 m",
        "experiment": experiment,
    }
    (args.output_dir / "boundary_coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
