#!/usr/bin/env python3
"""Extract label-isolated pre-execution features for SHANYU000 Gate 2.

Only route geometry, planned speed/control arrays, static boundary margins and
vehicle-structure metadata are passed to the feature extractor. Outcome labels
are joined after feature calculation and are never feature inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MEMBER = "SHANYU000"
V3_COMMIT = "d423f0e"
V3_OBJECT = f"{V3_COMMIT}:data/frozen/raw_samples/{MEMBER}/manifest.csv"
LOCAL_CANDIDATES = REPO_ROOT / "analysis" / "data_recovery" / MEMBER / "local_candidate_samples.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "derived_features" / MEMBER
CONTRACT_PATH = REPO_ROOT / "data" / "derived_features" / "feature_contract_v2.json"
REPORT_PATH = REPO_ROOT / "reports" / "data_recovery" / MEMBER / "gate2_feature_extraction.md"
FEATURE_VERSION = "pathguard_preexec_v2.0.0"
FEATURE_SOURCE = "npz_pre_execution_arrays_only"
WINDOW_M = 10.0
SOURCE_MACHINE_ALIAS = f"{MEMBER}_local"

FORBIDDEN_ARRAY_PATTERNS = (
    "hard_certificate",
    "failure_reason",
    "dynamic_pcd_clearance",
    "maximum_slip",
    "tire_utilization",
    "lateral_error",
    "heading_error",
    "completed",
)

METADATA_COLUMNS = [
    "sample_id", "route_file_sha256", "map_id", "route_id", "label_status",
    "true_label", "vehicle_structure", "vehicle_structure_evidence",
    "vehicle_axis_count", "source_machine", "match_status",
    "vehicle_configuration_fingerprint_summary", "feature_version",
    "feature_source", "feature_missing_reason",
]

TRAINING_FEATURE_COLUMNS = [
    "route_length_m", "point_count", "sampling_spacing_mean_m",
    "sampling_spacing_median_m", "sampling_spacing_p95_m",
    "curvature_mean_1pm", "curvature_abs_mean_1pm", "curvature_abs_max_1pm",
    "curvature_abs_p50_1pm", "curvature_abs_p90_1pm", "curvature_abs_p95_1pm",
    "curvature_abs_p99_1pm", "curvature_change_abs_mean_1pm2",
    "curvature_change_abs_max_1pm2", "curvature_change_abs_p95_1pm2",
    "slope_mean", "slope_abs_mean", "slope_abs_max", "slope_abs_p95",
    "slope_change_abs_mean_1pm", "slope_change_abs_max_1pm",
    "slope_change_abs_p95_1pm", "speed_mean_mps", "speed_max_mps",
    "speed_p50_mps", "speed_p90_mps", "speed_p95_mps",
    "speed_abs_curvature_product_mean_mpspm",
    "speed_abs_curvature_product_p95_mpspm",
    "lateral_accel_proxy_mean_mps2", "lateral_accel_proxy_max_mps2",
    "lateral_accel_proxy_p95_mps2", "speed_abs_curvature_correlation",
    "steer_abs_mean_rad", "steer_abs_max_rad", "steer_abs_p50_rad",
    "steer_abs_p90_rad", "steer_abs_p95_rad", "steer_change_abs_mean_radpm",
    "steer_change_abs_max_radpm", "steer_change_abs_p95_radpm",
    "yaw_rate_abs_mean_radps", "yaw_rate_abs_max_radps", "yaw_rate_abs_p95_radps",
    "yaw_rate_change_abs_mean_radps_per_m", "yaw_rate_change_abs_max_radps_per_m",
    "yaw_rate_change_abs_p95_radps_per_m", "static_left_clearance_min_m",
    "static_right_clearance_min_m", "static_boundary_clearance_min_m",
    "window10m_max_mean_abs_curvature_1pm",
    "window10m_max_mean_abs_steer_change_radpm",
    "window10m_min_mean_left_clearance_m",
    "window10m_min_mean_right_clearance_m",
    "steering_source_dim",
]

FEATURE_UNITS_AND_DEFINITIONS = {
    "route_length_m": ("m", "Difference between the final and initial route station values."),
    "point_count": ("count", "Number of ordered route samples."),
    "sampling_spacing_mean_m": ("m", "Mean positive difference between consecutive route stations."),
    "sampling_spacing_median_m": ("m", "Median positive difference between consecutive route stations."),
    "sampling_spacing_p95_m": ("m", "95th percentile of positive consecutive route-station differences."),
    "curvature_mean_1pm": ("1/m", "Arithmetic mean of signed planned route curvature."),
    "curvature_abs_mean_1pm": ("1/m", "Arithmetic mean of absolute planned route curvature."),
    "curvature_abs_max_1pm": ("1/m", "Maximum absolute planned route curvature."),
    "curvature_abs_p50_1pm": ("1/m", "50th percentile of absolute planned route curvature."),
    "curvature_abs_p90_1pm": ("1/m", "90th percentile of absolute planned route curvature."),
    "curvature_abs_p95_1pm": ("1/m", "95th percentile of absolute planned route curvature."),
    "curvature_abs_p99_1pm": ("1/m", "99th percentile of absolute planned route curvature."),
    "curvature_change_abs_mean_1pm2": ("1/m^2", "Mean absolute curvature derivative with respect to route station."),
    "curvature_change_abs_max_1pm2": ("1/m^2", "Maximum absolute curvature derivative with respect to route station."),
    "curvature_change_abs_p95_1pm2": ("1/m^2", "95th percentile of absolute curvature derivative with respect to route station."),
    "slope_mean": ("m/m", "Mean route elevation derivative with respect to route station."),
    "slope_abs_mean": ("m/m", "Mean absolute route elevation derivative with respect to route station."),
    "slope_abs_max": ("m/m", "Maximum absolute route elevation derivative with respect to route station."),
    "slope_abs_p95": ("m/m", "95th percentile of absolute route elevation derivative."),
    "slope_change_abs_mean_1pm": ("1/m", "Mean absolute slope derivative with respect to route station."),
    "slope_change_abs_max_1pm": ("1/m", "Maximum absolute slope derivative with respect to route station."),
    "slope_change_abs_p95_1pm": ("1/m", "95th percentile of absolute slope derivative with respect to route station."),
    "speed_mean_mps": ("m/s", "Arithmetic mean of planned route speed."),
    "speed_max_mps": ("m/s", "Maximum planned route speed."),
    "speed_p50_mps": ("m/s", "50th percentile of planned route speed."),
    "speed_p90_mps": ("m/s", "90th percentile of planned route speed."),
    "speed_p95_mps": ("m/s", "95th percentile of planned route speed."),
    "speed_abs_curvature_product_mean_mpspm": ("1/s", "Mean of planned speed multiplied by absolute planned curvature."),
    "speed_abs_curvature_product_p95_mpspm": ("1/s", "95th percentile of planned speed multiplied by absolute planned curvature."),
    "lateral_accel_proxy_mean_mps2": ("m/s^2", "Mean of planned speed squared multiplied by absolute planned curvature."),
    "lateral_accel_proxy_max_mps2": ("m/s^2", "Maximum planned speed squared multiplied by absolute planned curvature."),
    "lateral_accel_proxy_p95_mps2": ("m/s^2", "95th percentile of planned speed squared multiplied by absolute planned curvature."),
    "speed_abs_curvature_correlation": ("dimensionless", "Pearson correlation between planned speed and absolute planned curvature, or zero for a constant input."),
    "steer_abs_mean_rad": ("rad", "Mean pointwise steering magnitude from the selected pre-execution steering array."),
    "steer_abs_max_rad": ("rad", "Maximum pointwise steering magnitude from the selected pre-execution steering array."),
    "steer_abs_p50_rad": ("rad", "50th percentile of pointwise steering magnitude."),
    "steer_abs_p90_rad": ("rad", "90th percentile of pointwise steering magnitude."),
    "steer_abs_p95_rad": ("rad", "95th percentile of pointwise steering magnitude."),
    "steer_change_abs_mean_radpm": ("rad/m", "Mean absolute steering-magnitude derivative with respect to route station."),
    "steer_change_abs_max_radpm": ("rad/m", "Maximum absolute steering-magnitude derivative with respect to route station."),
    "steer_change_abs_p95_radpm": ("rad/m", "95th percentile of absolute steering-magnitude derivative with respect to route station."),
    "yaw_rate_abs_mean_radps": ("rad/s", "Mean absolute planned yaw rate, using the stored value or the documented pre-execution derivation."),
    "yaw_rate_abs_max_radps": ("rad/s", "Maximum absolute planned yaw rate."),
    "yaw_rate_abs_p95_radps": ("rad/s", "95th percentile of absolute planned yaw rate."),
    "yaw_rate_change_abs_mean_radps_per_m": ("rad/(s*m)", "Mean absolute planned yaw-rate derivative with respect to route station."),
    "yaw_rate_change_abs_max_radps_per_m": ("rad/(s*m)", "Maximum absolute planned yaw-rate derivative with respect to route station."),
    "yaw_rate_change_abs_p95_radps_per_m": ("rad/(s*m)", "95th percentile of absolute planned yaw-rate derivative with respect to route station."),
    "static_left_clearance_min_m": ("m", "Minimum planned static left-boundary clearance."),
    "static_right_clearance_min_m": ("m", "Minimum planned static right-boundary clearance."),
    "static_boundary_clearance_min_m": ("m", "Minimum of the planned static left- and right-boundary clearances."),
    "window10m_max_mean_abs_curvature_1pm": ("1/m", "Maximum trailing-10-m window mean of absolute planned curvature."),
    "window10m_max_mean_abs_steer_change_radpm": ("rad/m", "Maximum trailing-10-m window mean of absolute steering change per route metre."),
    "window10m_min_mean_left_clearance_m": ("m", "Minimum trailing-10-m window mean of planned static left-boundary clearance."),
    "window10m_min_mean_right_clearance_m": ("m", "Minimum trailing-10-m window mean of planned static right-boundary clearance."),
    "steering_source_dim": ("count", "Last-dimension size of the NPZ steering array actually used for feature calculation; this is not vehicle axis count."),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_v3() -> list[dict[str, str]]:
    result = subprocess.run(
        ["git", "show", V3_OBJECT], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    )
    return list(csv.DictReader(io.StringIO(result.stdout.lstrip("\ufeff"))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_error(exc: Exception) -> str:
    text = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "<local_path>", f"{type(exc).__name__}: {exc}")
    return text[:240]


def finite_1d(value: Any, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(f"{name} length {array.size} does not match route length {length}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is empty or non-finite")
    return array


def derivative(values: np.ndarray, s_m: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.zeros_like(values)
    edge_order = 2 if values.size >= 3 else 1
    return np.gradient(values, s_m, edge_order=edge_order)


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def rolling_means(values: np.ndarray, s_m: np.ndarray, window_m: float) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    starts = np.searchsorted(s_m, s_m - window_m, side="left")
    counts = np.arange(1, values.size + 1) - starts
    return (prefix[1:] - prefix[starts]) / counts


def vector_magnitude(array: np.ndarray, length: int) -> tuple[np.ndarray, int]:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 1:
        if value.size != length:
            raise ValueError("steering array length mismatch")
        return np.abs(value), 1
    if value.ndim == 2 and value.shape[0] == length:
        return np.max(np.abs(value), axis=1), int(value.shape[1])
    raise ValueError(f"unsupported steering shape {value.shape}")


def fingerprint_summary(arrays: dict[str, np.ndarray], vehicle_structure: str, steer_dim: int) -> str:
    if "vehicle_profile_fingerprint" in arrays:
        raw = np.asarray(arrays["vehicle_profile_fingerprint"]).reshape(-1)
        if raw.size:
            value = str(raw[0]).strip()
            if value:
                return value[:16]
    material = f"derived:{vehicle_structure}:steering_source_dim={steer_dim}"
    return "derived-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def normalize_vehicle_structure(raw: str) -> tuple[str, str, int | str]:
    if raw == "five_axis_explicit":
        return "five_axis", "explicit", 5
    if raw == "six_axis_explicit":
        return "six_axis", "explicit", 6
    if raw == "six_axis_structural":
        return "six_axis", "structural_inference", 6
    return "unknown", "unknown", ""


def build_feature_contract() -> tuple[dict[str, Any], str]:
    if list(FEATURE_UNITS_AND_DEFINITIONS) != TRAINING_FEATURE_COLUMNS:
        raise RuntimeError("Feature definition order does not match training feature order")
    ordered_columns = METADATA_COLUMNS + TRAINING_FEATURE_COLUMNS
    schema_material = FEATURE_VERSION + "\n" + "\n".join(ordered_columns)
    schema_sha256 = hashlib.sha256(schema_material.encode("utf-8")).hexdigest()
    contract = {
        "feature_version": FEATURE_VERSION,
        "metadata_columns": METADATA_COLUMNS,
        "training_feature_columns": TRAINING_FEATURE_COLUMNS,
        "training_features": [
            {"name": name, "unit": FEATURE_UNITS_AND_DEFINITIONS[name][0], "definition": FEATURE_UNITS_AND_DEFINITIONS[name][1]}
            for name in TRAINING_FEATURE_COLUMNS
        ],
        "excluded_from_training": METADATA_COLUMNS,
        "prohibited_source_fields": [
            "hard_certificate_passed", "hard_failure_reasons", "dynamic_pcd_clearance_m",
            "maximum_slip_ratio", "maximum_tire_utilization", "report_path",
            "build_family", "source_folder_name",
        ],
        "local_window_m": 10,
        "label_definition": "1=pass, 0=fail",
        "schema_sha256_algorithm": "SHA-256 over UTF-8 feature_version followed by metadata_columns and training_feature_columns, each separated by LF",
        "schema_sha256": schema_sha256,
    }
    return contract, schema_sha256


def extract_features(arrays: dict[str, np.ndarray], vehicle_structure: str) -> tuple[dict[str, Any], list[str], list[str]]:
    """Calculate features without accepting any outcome/label argument."""
    used_arrays: list[str] = []
    missing: list[str] = []
    for name in arrays:
        lowered = name.lower()
        if any(pattern in lowered for pattern in FORBIDDEN_ARRAY_PATTERNS):
            raise ValueError(f"forbidden result-like array presented to extractor: {name}")

    required = ["s_m", "z_m", "kappa_1pm", "v_profile_mps", "left_clearance_m", "right_clearance_m"]
    absent = [name for name in required if name not in arrays]
    if absent:
        raise ValueError("missing required pre-execution arrays: " + ";".join(absent))

    s_m = finite_1d(arrays["s_m"], "s_m")
    n = s_m.size
    if n < 2 or not np.all(np.diff(s_m) > 0):
        raise ValueError("s_m must be strictly increasing with at least two points")
    used_arrays.append("s_m")
    ds = np.diff(s_m)
    route_length = float(s_m[-1] - s_m[0])

    z_m = finite_1d(arrays["z_m"], "z_m", n); used_arrays.append("z_m")
    kappa = finite_1d(arrays["kappa_1pm"], "kappa_1pm", n); used_arrays.append("kappa_1pm")
    speed = finite_1d(arrays["v_profile_mps"], "v_profile_mps", n); used_arrays.append("v_profile_mps")
    left = finite_1d(arrays["left_clearance_m"], "left_clearance_m", n); used_arrays.append("left_clearance_m")
    right = finite_1d(arrays["right_clearance_m"], "right_clearance_m", n); used_arrays.append("right_clearance_m")

    if "dkappa_ds_1pm2" in arrays:
        dkappa = finite_1d(arrays["dkappa_ds_1pm2"], "dkappa_ds_1pm2", n)
        used_arrays.append("dkappa_ds_1pm2")
    else:
        dkappa = derivative(kappa, s_m)
        missing.append("dkappa_ds_1pm2:derived_from_kappa_and_s")

    slope = derivative(z_m, s_m)
    slope_change = derivative(slope, s_m)
    abs_kappa = np.abs(kappa)
    abs_dkappa = np.abs(dkappa)

    if "steer_ff_rad" in arrays:
        steer, steer_dim = vector_magnitude(arrays["steer_ff_rad"], n)
        used_arrays.append("steer_ff_rad")
    elif "beta_ref_rad" in arrays:
        steer, steer_dim = vector_magnitude(arrays["beta_ref_rad"], n)
        used_arrays.append("beta_ref_rad")
        missing.append("steer_ff_rad:used_beta_ref_rad")
    else:
        raise ValueError("missing steering control array")
    steer_rate = np.abs(derivative(steer, s_m))

    if "yaw_rate_ref_radps" in arrays:
        yaw_rate = finite_1d(arrays["yaw_rate_ref_radps"], "yaw_rate_ref_radps", n)
        used_arrays.append("yaw_rate_ref_radps")
    elif "yaw_per_m_ref_1pm" in arrays:
        yaw_per_m = finite_1d(arrays["yaw_per_m_ref_1pm"], "yaw_per_m_ref_1pm", n)
        yaw_rate = yaw_per_m * speed
        used_arrays.append("yaw_per_m_ref_1pm")
        missing.append("yaw_rate_ref_radps:derived_from_yaw_per_m_and_speed")
    else:
        yaw_rate = kappa * speed
        missing.append("yaw_rate_ref_radps:derived_from_curvature_and_speed")
    yaw_rate_change = np.abs(derivative(yaw_rate, s_m))

    product = speed * abs_kappa
    lateral_proxy = speed * speed * abs_kappa
    if np.std(speed) > 0 and np.std(abs_kappa) > 0:
        correlation = float(np.corrcoef(speed, abs_kappa)[0, 1])
    else:
        correlation = 0.0
        missing.append("speed_abs_curvature_correlation:constant_input_set_to_zero")

    features: dict[str, Any] = {
        "route_length_m": route_length,
        "point_count": int(n),
        "sampling_spacing_mean_m": float(np.mean(ds)),
        "sampling_spacing_median_m": float(np.median(ds)),
        "sampling_spacing_p95_m": quantile(ds, 0.95),
        "curvature_mean_1pm": float(np.mean(kappa)),
        "curvature_abs_mean_1pm": float(np.mean(abs_kappa)),
        "curvature_abs_max_1pm": float(np.max(abs_kappa)),
        "curvature_abs_p50_1pm": quantile(abs_kappa, 0.50),
        "curvature_abs_p90_1pm": quantile(abs_kappa, 0.90),
        "curvature_abs_p95_1pm": quantile(abs_kappa, 0.95),
        "curvature_abs_p99_1pm": quantile(abs_kappa, 0.99),
        "curvature_change_abs_mean_1pm2": float(np.mean(abs_dkappa)),
        "curvature_change_abs_max_1pm2": float(np.max(abs_dkappa)),
        "curvature_change_abs_p95_1pm2": quantile(abs_dkappa, 0.95),
        "slope_mean": float(np.mean(slope)),
        "slope_abs_mean": float(np.mean(np.abs(slope))),
        "slope_abs_max": float(np.max(np.abs(slope))),
        "slope_abs_p95": quantile(np.abs(slope), 0.95),
        "slope_change_abs_mean_1pm": float(np.mean(np.abs(slope_change))),
        "slope_change_abs_max_1pm": float(np.max(np.abs(slope_change))),
        "slope_change_abs_p95_1pm": quantile(np.abs(slope_change), 0.95),
        "speed_mean_mps": float(np.mean(speed)),
        "speed_max_mps": float(np.max(speed)),
        "speed_p50_mps": quantile(speed, 0.50),
        "speed_p90_mps": quantile(speed, 0.90),
        "speed_p95_mps": quantile(speed, 0.95),
        "speed_abs_curvature_product_mean_mpspm": float(np.mean(product)),
        "speed_abs_curvature_product_p95_mpspm": quantile(product, 0.95),
        "lateral_accel_proxy_mean_mps2": float(np.mean(lateral_proxy)),
        "lateral_accel_proxy_max_mps2": float(np.max(lateral_proxy)),
        "lateral_accel_proxy_p95_mps2": quantile(lateral_proxy, 0.95),
        "speed_abs_curvature_correlation": correlation,
        "steer_abs_mean_rad": float(np.mean(steer)),
        "steer_abs_max_rad": float(np.max(steer)),
        "steer_abs_p50_rad": quantile(steer, 0.50),
        "steer_abs_p90_rad": quantile(steer, 0.90),
        "steer_abs_p95_rad": quantile(steer, 0.95),
        "steer_change_abs_mean_radpm": float(np.mean(steer_rate)),
        "steer_change_abs_max_radpm": float(np.max(steer_rate)),
        "steer_change_abs_p95_radpm": quantile(steer_rate, 0.95),
        "yaw_rate_abs_mean_radps": float(np.mean(np.abs(yaw_rate))),
        "yaw_rate_abs_max_radps": float(np.max(np.abs(yaw_rate))),
        "yaw_rate_abs_p95_radps": quantile(np.abs(yaw_rate), 0.95),
        "yaw_rate_change_abs_mean_radps_per_m": float(np.mean(yaw_rate_change)),
        "yaw_rate_change_abs_max_radps_per_m": float(np.max(yaw_rate_change)),
        "yaw_rate_change_abs_p95_radps_per_m": quantile(yaw_rate_change, 0.95),
        "static_left_clearance_min_m": float(np.min(left)),
        "static_right_clearance_min_m": float(np.min(right)),
        "static_boundary_clearance_min_m": float(min(np.min(left), np.min(right))),
        "window10m_max_mean_abs_curvature_1pm": float(np.max(rolling_means(abs_kappa, s_m, WINDOW_M))),
        "window10m_max_mean_abs_steer_change_radpm": float(np.max(rolling_means(steer_rate, s_m, WINDOW_M))),
        "window10m_min_mean_left_clearance_m": float(np.min(rolling_means(left, s_m, WINDOW_M))),
        "window10m_min_mean_right_clearance_m": float(np.min(rolling_means(right, s_m, WINDOW_M))),
        "steering_source_dim": steer_dim,
        "vehicle_configuration_fingerprint_summary": fingerprint_summary(arrays, vehicle_structure, steer_dim),
    }
    numeric = [value for name, value in features.items() if name != "vehicle_configuration_fingerprint_summary"]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("computed feature contains NaN or infinity")
    return features, sorted(used_arrays), missing


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def map_family(map_id: str) -> str:
    lowered = map_id.lower()
    for family in ("ramp", "cliff", "cross", "turnaround", "narrow", "overpass"):
        if lowered.startswith(family):
            return "narrow/overpass" if family in {"narrow", "overpass"} else family
    return "other"


def main() -> None:
    v3_rows = read_v3()
    candidate_rows = read_csv(LOCAL_CANDIDATES)
    contract, schema_sha256 = build_feature_contract()
    v3_by_hash = {row["route_file_sha256"].lower(): row for row in v3_rows}
    v3_by_id = {row["sample_id"]: row for row in v3_rows}

    candidates_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidates_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        route_hash = row.get("route_file_sha256", "").strip().lower()
        sample_id = row.get("sample_id", "").strip()
        if route_hash:
            candidates_by_hash[route_hash].append(row)
        if sample_id:
            candidates_by_id[sample_id].append(row)

    feature_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    damaged_rows: list[dict[str, Any]] = []
    matched_candidate_hashes: set[str] = set()

    for v3 in v3_rows:
        expected_hash = v3["route_file_sha256"].lower()
        vehicle_structure, structure_evidence, axis_count = normalize_vehicle_structure(v3["vehicle_structure"])
        candidates = candidates_by_hash.get(expected_hash, [])
        match_method = "route_file_sha256"
        if not candidates:
            candidates = candidates_by_id.get(v3["sample_id"], [])
            match_method = "sample_id"
        candidates = sorted(candidates, key=lambda row: row.get("route_path_local", "").casefold())
        existing = [row for row in candidates if Path(row.get("route_path_local", "")).is_file()]
        if not existing:
            damaged_rows.append({
                "sample_id": v3["sample_id"], "route_file_sha256": expected_hash,
                "map_id": v3["map"], "vehicle_structure": vehicle_structure,
                "vehicle_structure_evidence": structure_evidence, "vehicle_axis_count": axis_count,
                "source_machine": SOURCE_MACHINE_ALIAS, "feature_version": FEATURE_VERSION,
                "match_status": "v3_matched", "error_type": "local_file_not_found",
                "error_message": "No available local NPZ matched the V3 record.",
            })
            continue

        extracted = False
        errors: list[tuple[str, str]] = []
        for candidate in existing:
            path = Path(candidate["route_path_local"])
            try:
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    raise ValueError("SHA-256 does not match the V3 route_file_sha256")
                with np.load(path, allow_pickle=False) as archive:
                    allowed_names = {
                        "s_m", "z_m", "kappa_1pm", "dkappa_ds_1pm2", "v_profile_mps",
                        "left_clearance_m", "right_clearance_m", "steer_ff_rad",
                        "beta_ref_rad", "yaw_rate_ref_radps", "yaw_per_m_ref_1pm",
                        "vehicle_profile_fingerprint",
                    }
                    arrays = {name: archive[name] for name in archive.files if name in allowed_names}
                features, used_arrays, missing = extract_features(arrays, v3["vehicle_structure"])
            except Exception as exc:
                errors.append((type(exc).__name__, clean_error(exc)))
                continue

            target = 1 if v3["hard_certificate_passed"] == "1" else 0
            fingerprint = features.pop("vehicle_configuration_fingerprint_summary")
            feature_rows.append({
                "sample_id": v3["sample_id"], "route_file_sha256": expected_hash,
                "map_id": v3["map"], "route_id": v3["map"],
                "label_status": v3["label_status"], "true_label": target,
                "vehicle_structure": vehicle_structure,
                "vehicle_structure_evidence": structure_evidence,
                "vehicle_axis_count": axis_count, "source_machine": SOURCE_MACHINE_ALIAS,
                "match_status": "v3_exact_match",
                "vehicle_configuration_fingerprint_summary": fingerprint,
                "feature_version": FEATURE_VERSION, "feature_source": FEATURE_SOURCE,
                "feature_missing_reason": ";".join(missing), **features,
            })
            quality_rows.append({
                "sample_id": v3["sample_id"], "route_file_sha256": expected_hash,
                "map_id": v3["map"], "vehicle_structure": vehicle_structure,
                "vehicle_structure_evidence": structure_evidence,
                "vehicle_axis_count": axis_count, "feature_version": FEATURE_VERSION,
                "schema_sha256": schema_sha256,
                "match_method": match_method, "sha256_verified": "true",
                "npz_readable": "true", "feature_row_included": "true",
                "numeric_feature_count": len(TRAINING_FEATURE_COLUMNS),
                "missing_numeric_feature_count": 0,
                "fallback_or_derived_count": len(missing),
                "feature_missing_reason": ";".join(missing),
                "source_arrays_used": json.dumps(used_arrays, separators=(",", ":")),
                "finite_numeric_features": "true", "source_machine": SOURCE_MACHINE_ALIAS,
            })
            matched_candidate_hashes.add(expected_hash)
            extracted = True
            break
        if not extracted:
            error_type, message = errors[0] if errors else ("unknown_error", "Feature extraction failed.")
            damaged_rows.append({
                "sample_id": v3["sample_id"], "route_file_sha256": expected_hash,
                "map_id": v3["map"], "vehicle_structure": vehicle_structure,
                "vehicle_structure_evidence": structure_evidence, "vehicle_axis_count": axis_count,
                "source_machine": SOURCE_MACHINE_ALIAS, "feature_version": FEATURE_VERSION,
                "match_status": "v3_matched", "error_type": error_type,
                "error_message": message,
            })

    unmatched_rows: list[dict[str, Any]] = []
    seen_unmatched: set[str] = set()
    for route_hash in sorted(candidates_by_hash):
        if route_hash in v3_by_hash or route_hash in matched_candidate_hashes or route_hash in seen_unmatched:
            continue
        candidates = sorted(candidates_by_hash[route_hash], key=lambda row: row.get("route_path_local", "").casefold())
        existing = [row for row in candidates if Path(row.get("route_path_local", "")).is_file()]
        if not existing:
            candidate = candidates[0]
            unmatched_structure, unmatched_evidence, unmatched_axis_count = normalize_vehicle_structure(
                candidate.get("vehicle_structure", "")
            )
            damaged_rows.append({
                "sample_id": "UNMATCHED-SHANYU000-" + route_hash[:16],
                "route_file_sha256": route_hash, "map_id": candidate.get("map", ""),
                "vehicle_structure": unmatched_structure,
                "vehicle_structure_evidence": unmatched_evidence,
                "vehicle_axis_count": unmatched_axis_count,
                "source_machine": SOURCE_MACHINE_ALIAS, "feature_version": FEATURE_VERSION,
                "match_status": "unmatched_candidate", "error_type": "local_file_not_found",
                "error_message": "No currently available local NPZ remained for this candidate hash.",
            })
            continue
        candidate = existing[0]
        readable = False
        read_errors: list[str] = []
        for possible in existing:
            try:
                with np.load(Path(possible["route_path_local"]), allow_pickle=False) as archive:
                    _ = tuple(archive.files)
                candidate = possible
                readable = True
                break
            except Exception as exc:
                read_errors.append(clean_error(exc))
        if not readable:
            unmatched_structure, unmatched_evidence, unmatched_axis_count = normalize_vehicle_structure(
                candidate.get("vehicle_structure", "")
            )
            damaged_rows.append({
                "sample_id": "UNMATCHED-SHANYU000-" + route_hash[:16],
                "route_file_sha256": route_hash, "map_id": candidate.get("map", ""),
                "vehicle_structure": unmatched_structure,
                "vehicle_structure_evidence": unmatched_evidence,
                "vehicle_axis_count": unmatched_axis_count,
                "source_machine": SOURCE_MACHINE_ALIAS, "feature_version": FEATURE_VERSION,
                "match_status": "unmatched_candidate", "error_type": "npz_read_error",
                "error_message": read_errors[0] if read_errors else "NPZ could not be opened.",
            })
            continue
        seen_unmatched.add(route_hash)
        unmatched_structure, unmatched_evidence, unmatched_axis_count = normalize_vehicle_structure(
            candidate.get("vehicle_structure", "")
        )
        unmatched_rows.append({
            "sample_id": "UNMATCHED-SHANYU000-" + route_hash[:16],
            "route_file_sha256": route_hash,
            "map_id": candidate.get("map", ""),
            "vehicle_structure": unmatched_structure,
            "vehicle_structure_evidence": unmatched_evidence,
            "vehicle_axis_count": unmatched_axis_count,
            "label_status": "unmatched_candidate",
            "true_label": "",
            "route_id": candidate.get("map", ""),
            "match_status": "unmatched_candidate",
            "candidate_file_available": "true",
            "npz_readable": "true",
            "source_machine": SOURCE_MACHINE_ALIAS,
            "feature_version": FEATURE_VERSION,
            "exclusion_reason": "route_file_sha256_and_sample_id_not_found_in_v3",
        })

    feature_rows.sort(key=lambda row: row["sample_id"])
    quality_rows.sort(key=lambda row: row["sample_id"])
    damaged_rows.sort(key=lambda row: (row["match_status"], row["sample_id"]))

    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    write_csv(OUTPUT_DIR / "pre_execution_features.csv", METADATA_COLUMNS + TRAINING_FEATURE_COLUMNS, feature_rows)
    write_csv(OUTPUT_DIR / "feature_quality.csv", [
        "sample_id", "route_file_sha256", "map_id", "vehicle_structure",
        "vehicle_structure_evidence", "vehicle_axis_count", "feature_version", "schema_sha256", "match_method",
        "sha256_verified", "npz_readable", "feature_row_included", "numeric_feature_count",
        "missing_numeric_feature_count", "fallback_or_derived_count",
        "feature_missing_reason", "source_arrays_used",
        "finite_numeric_features", "source_machine",
    ], quality_rows)
    write_csv(OUTPUT_DIR / "unmatched_candidates.csv", [
        "sample_id", "route_file_sha256", "map_id", "vehicle_structure",
        "vehicle_structure_evidence", "vehicle_axis_count", "label_status",
        "true_label", "route_id", "match_status", "candidate_file_available", "npz_readable",
        "source_machine", "feature_version", "exclusion_reason",
    ], unmatched_rows)
    write_csv(OUTPUT_DIR / "damaged_files.csv", [
        "sample_id", "route_file_sha256", "map_id", "vehicle_structure",
        "vehicle_structure_evidence", "vehicle_axis_count", "source_machine",
        "feature_version", "match_status", "error_type", "error_message",
    ], damaged_rows)

    axes = Counter("five_axis" if row["vehicle_structure"].startswith("five_axis") else "six_axis" for row in feature_rows)
    labels = Counter(row["true_label"] for row in feature_rows)
    families = sorted({map_family(row["map_id"]) for row in feature_rows})
    missing_numeric_rows = sum(int(row["missing_numeric_feature_count"]) > 0 for row in quality_rows)
    fallback_rows = sum(bool(row["feature_missing_reason"]) for row in feature_rows)
    fallback_total = sum(int(row["fallback_or_derived_count"]) for row in quality_rows)
    v3_local_risk = sum(
        any(token in row["hard_failure_reasons"] for token in (
            "closed_loop_pcd_clearance", "closed_loop_txt_clearance", "closed_loop_heightmap_footprint"
        )) for row in v3_rows if row["sample_id"] in {item["sample_id"] for item in feature_rows}
    )
    five_local_risk = sum(
        row["vehicle_structure"].startswith("five_axis") and
        any(token in row["hard_failure_reasons"] for token in (
            "closed_loop_pcd_clearance", "closed_loop_txt_clearance", "closed_loop_heightmap_footprint"
        )) for row in v3_rows if row["sample_id"] in {item["sample_id"] for item in feature_rows}
    )
    six_local_risk = v3_local_risk - five_local_risk

    report = f"""# Gate 2 执行前特征提取报告（{MEMBER}）

## 1. 对齐范围与标签隔离

本次以 `{V3_OBJECT}` 的 {len(v3_rows)} 条记录作为当前 V3 映射。匹配顺序为 `route_file_sha256` 优先、`sample_id` 次优。标签仅在数值特征完成计算后按 `sample_id` 回填为整数 `true_label`，原有通过/失败结果未修改。

本次统一采用 `{FEATURE_VERSION}`。唯一契约为 `data/derived_features/feature_contract_v2.json`，训练列数量为 {len(TRAINING_FEATURE_COLUMNS)}，`schema_sha256` 为 `{schema_sha256}`。主表列顺序严格等于契约中的 `metadata_columns` 后接 `training_feature_columns`。

特征计算函数只接收路线、计划速度、控制向量、静态左右边界余量和车辆结构信息。它不接收硬证书结果、失败原因、报告路径、构建批次、运行后动态净空、最大滑移率、最大轮胎利用率或其他结果字段。未训练模型。

## 2. 结果统计

| 项目 | 数量 |
|---|---:|
| V3 记录 | {len(v3_rows)} |
| 有效特征记录 | {len(feature_rows)} |
| 未命中 V3 的唯一候选哈希 | {len(unmatched_rows)} |
| 损坏或无法提取 | {len(damaged_rows)} |
| 数值特征缺失记录 | {missing_numeric_rows} |
| 含回退或派生说明的特征记录 | {fallback_rows} |
| 回退或派生说明总项数 | {fallback_total} |
| 五轴记录 | {axes['five_axis']} |
| 六轴记录 | {axes['six_axis']} |
| 通过标签 | {labels[1]} |
| 失败标签 | {labels[0]} |
| 由既有证书原因识别的局部风险覆盖 | {v3_local_risk} |

覆盖的地图类别为：{', '.join(families)}。每条记录保留 `map_id` 和 `route_id`；后续切分应按 `route_id` 或地图组留出，禁止把同一路线组随机拆入训练集和测试集。

## 3. 特征定义

- 路线：长度、点数、采样间距均值/中位数/95 分位数。
- 曲率：有符号均值、绝对值均值/最大值/50、90、95、99 分位数及单位距离变化率。
- 坡度：由 `z_m` 对 `s_m` 求导，并统计坡度及单位距离坡度变化。
- 速度：均值、最大值及 50、90、95 分位数。
- 速度曲率耦合：`v·|kappa|`、`v²·|kappa|` 及速度与绝对曲率相关系数；`v²·|kappa|` 仅为执行前几何-速度耦合代理量，不是运行后测量指标。
- 转向：优先读取 `steer_ff_rad`；缺失时使用执行前 `beta_ref_rad`，并在质量表记录回退。多维转向取同一点各维绝对值最大值，再统计幅值与单位距离变化率。
- 横摆率：优先读取 `yaw_rate_ref_radps`；缺失时由 `yaw_per_m_ref_1pm·v_profile_mps` 推导，再统计幅值和单位距离变化。
- 局部窗口：固定 10 m 后向空间窗口，统计窗口均值的最大绝对曲率、最大转向变化和最小左右静态边界余量。
- 车辆结构：`vehicle_structure` 仅使用 `five_axis`、`six_axis`、`unknown`，证据来源单列记录。`vehicle_axis_count` 和配置指纹摘要属于元数据。`steering_source_dim` 仅描述实际参与计算的转向数组末维，不代表车辆轴数。

所有数值特征均使用字段名中的单位；生成过程拒绝 NaN 和无穷值。`feature_missing_reason` 记录允许的执行前回退或派生方式，不使用结果字段补值。

## 4. 覆盖限制

当前 V3 映射只有 {labels[1]} 条通过样本，其中五轴通过 2 条、六轴通过 2 条。五轴失败 17 条、局部风险 {five_local_risk} 条；六轴失败 3 条、局部风险 {six_local_risk} 条，且六轴记录总量仅 {axes['six_axis']} 条。因此无法同时达到“五轴和六轴的通过、失败、局部风险各至少 5 条”。本次保留全部可用 V3 样本并明确缺口，不从未命中候选中补标签，也不修改原标签。

仓库仍未发现明确命名为“数据冻结 V3”的文件。本次沿用 Gate 2 已采用的提交 `{V3_COMMIT}` 映射，仍待数据负责人确认。

## 5. 输出与数据边界

- `data/derived_features/{MEMBER}/pre_execution_features.csv`：仅 V3 命中的主建模候选表。
- `data/derived_features/{MEMBER}/feature_quality.csv`：哈希、可读性、有限值和回退说明。
- `data/derived_features/{MEMBER}/unmatched_candidates.csv`：未命中 V3 的候选，不进入主表。
- `data/derived_features/{MEMBER}/damaged_files.csv`：损坏或无法提取记录。
- `data/derived_features/feature_contract_v2.json`：全项目唯一 v2 特征契约。
- `reports/data_recovery/{MEMBER}/gate2_feature_extraction.md`：本报告。

未提交原始 NPZ、本机绝对路径、报告路径、地图文件名或构建批次名。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")

    print(json.dumps({
        "v3_records": len(v3_rows), "features": len(feature_rows),
        "unmatched": len(unmatched_rows), "damaged": len(damaged_rows),
        "five_axis": axes["five_axis"], "six_axis": axes["six_axis"],
        "pass": labels[1], "fail": labels[0], "training_features": len(TRAINING_FEATURE_COLUMNS),
        "schema_sha256": schema_sha256,
        "missing_numeric_feature_rows": missing_numeric_rows, "fallback_rows": fallback_rows,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
