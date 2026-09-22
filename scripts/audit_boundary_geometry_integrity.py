"""Re-audit raw boundary correspondence and continuity without copying NPZs.

The manifest path is read locally but is never written to any output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final" / "tables"


def audit_one(path: Path, expected_sha: str, sample_id: str) -> dict[str, object]:
    result: dict[str, object] = {"sample_id": sample_id, "readable": False}
    if not path.is_file():
        return {**result, "reason": "manifest_path_missing"}
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest.lower() != str(expected_sha).lower():
        return {**result, "reason": "sha256_mismatch"}
    result["readable"] = True
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"x_m", "y_m", "s_m", "left_clearance_m", "right_clearance_m",
                        "left_boundary_xyz_m", "right_boundary_xyz_m"}
            if not required.issubset(archive.files):
                return {**result, "reason": "missing_required_boundary_arrays"}
            x = np.asarray(archive["x_m"], dtype=float).reshape(-1)
            y = np.asarray(archive["y_m"], dtype=float).reshape(-1)
            s = np.asarray(archive["s_m"], dtype=float).reshape(-1)
            lc = np.asarray(archive["left_clearance_m"], dtype=float).reshape(-1)
            rc = np.asarray(archive["right_clearance_m"], dtype=float).reshape(-1)
            left = np.asarray(archive["left_boundary_xyz_m"], dtype=float)
            right = np.asarray(archive["right_boundary_xyz_m"], dtype=float)
        n = len(s)
        if not (n >= 3 and len(x) == len(y) == len(lc) == len(rc) == n
                and left.ndim == right.ndim == 2 and left.shape[0] == right.shape[0] == n
                and left.shape[1] >= 2 and right.shape[1] >= 2):
            return {**result, "reason": "array_shape_mismatch"}
        center = np.column_stack([x, y])
        left_xy, right_xy = left[:, :2], right[:, :2]
        if not all(np.isfinite(a).all() for a in (center, s, lc, rc, left_xy, right_xy)):
            return {**result, "reason": "non_finite_geometry"}
        distance_error = max(
            float(np.max(np.abs(np.linalg.norm(left_xy - center, axis=1) - lc))),
            float(np.max(np.abs(np.linalg.norm(right_xy - center, axis=1) - rc))),
        )
        ds = np.diff(s)
        center_step = np.linalg.norm(np.diff(center, axis=0), axis=1)
        if np.any(ds <= 0) or np.any(center_step <= 1e-8):
            return {**result, "reason": "non_monotone_station_or_duplicate_centerline"}
        tangent = np.gradient(center, s, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        if np.any(norm <= 1e-8):
            return {**result, "reason": "undefined_route_tangent"}
        tangent /= norm[:, None]
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        left_signed = np.sum((left_xy - center) * normal, axis=1)
        right_signed = np.sum((right_xy - center) * normal, axis=1)
        left_fraction = float(np.mean(left_signed > 0))
        right_fraction = float(np.mean(right_signed < 0))
        ordered_fraction = float(np.mean(left_signed > right_signed))
        ratio_left = np.linalg.norm(np.diff(left_xy, axis=0), axis=1) / center_step
        ratio_right = np.linalg.norm(np.diff(right_xy, axis=0), axis=1) / center_step
        p999 = float(max(np.quantile(ratio_left, 0.999), np.quantile(ratio_right, 0.999)))
        max_ratio = float(max(np.max(ratio_left), np.max(ratio_right)))
        checks = {
            "distance_semantics_pass": distance_error <= 1e-5,
            "left_side_orientation_pass": left_fraction >= 0.99,
            "right_side_orientation_pass": right_fraction >= 0.99,
            "left_right_order_pass": ordered_fraction >= 0.99,
            # A large boundary step relative to the corresponding centerline
            # step suggests a jump between unrelated segments; report this as
            # a conservative screen, not a polygon-topology proof.
            "continuity_screen_pass": p999 <= 3.0 and max_ratio <= 5.0,
        }
        return {
            **result, "point_count": n, "reason": "", "max_distance_error_m": distance_error,
            "left_side_fraction": left_fraction, "right_side_fraction": right_fraction,
            "ordered_cross_section_fraction": ordered_fraction,
            "boundary_to_center_step_ratio_p99_9": p999,
            "boundary_to_center_step_ratio_max": max_ratio,
            "boundary_to_center_step_ratio_max": max_ratio,
            **checks, "integrity_screen_pass": all(checks.values()),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {**result, "reason": f"read_error:{type(error).__name__}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    coverage = pd.read_csv(ROOT / "analysis/boundary_geometry/boundary_coverage_samples.csv")
    verified_ids = coverage.loc[
        coverage["vehicle_structure"].eq("five_axis")
        & coverage["boundary_distance_semantics_verified"].astype(str).str.lower().isin(["true", "1"]),
        "sample_id",
    ]
    candidates = manifest[manifest["sample_id"].isin(verified_ids)]
    rows = [audit_one(Path(row.route_path_local), row.route_file_sha256, row.sample_id)
            for row in candidates.itertuples(index=False)]
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    details_path = OUT / "boundary_integrity_reaudit.csv"
    summary_path = ROOT / "reports/final/boundary_integrity_reaudit_summary.json"
    frame.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary = {
        "candidate_rows": len(frame),
        "readable_and_hash_matched": int(frame["readable"].fillna(False).sum()),
        "strict_distance_semantics_pass": int(frame.get("distance_semantics_pass", pd.Series(dtype=bool)).fillna(False).sum()),
        "orientation_and_continuity_screen_pass": int(frame.get("integrity_screen_pass", pd.Series(dtype=bool)).fillna(False).sum()),
        "method": "raw full-resolution arrays; 1e-5 m distance reproduction, centerline normal side/order, station monotonicity, and boundary-to-center step ratios P99.9 <= 3 and max <= 5",
        "limitations": [
            "The continuity screen is a conservative point-sequence check, not proof of a simple polygon or a self-intersection-free corridor.",
            "Vehicle sweep remains a sampled local cross-section estimate, not continuous collision detection.",
            "Manifest local paths are not copied into results.",
        ],
        "details_file": str(details_path.relative_to(ROOT)).replace("\\", "/"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
