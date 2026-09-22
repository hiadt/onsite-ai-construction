"""Build compact, auditable route geometry for the Streamlit demo.

The output contains downsampled centerline and clearance boundaries derived
directly from the matched NPZ files. It does not invent obstacle coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def downsample_indices(length: int, limit: int = 240) -> np.ndarray:
    count = min(length, limit)
    return np.unique(np.linspace(0, length - 1, count, dtype=int))


def pairs(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(x, y)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    sample_ids = set(pd.read_csv(args.samples)["sample_id"].astype(str))
    selected = manifest[manifest["sample_id"].astype(str).isin(sample_ids)]
    output: dict[str, object] = {
        "source": "matched_npz_pre_execution_arrays",
        "coordinate_note": "centerline and clearance boundaries derived from x_m/y_m/yaw_rad/left_clearance_m/right_clearance_m",
        "routes": {},
    }

    for row in selected.itertuples(index=False):
        path = Path(row.route_path_local)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            required = {"x_m", "y_m", "yaw_rad", "left_clearance_m", "right_clearance_m"}
            if not required.issubset(data.files):
                continue
            x = np.asarray(data["x_m"], dtype=float)
            y = np.asarray(data["y_m"], dtype=float)
            yaw = np.asarray(data["yaw_rad"], dtype=float)
            left = np.asarray(data["left_clearance_m"], dtype=float)
            right = np.asarray(data["right_clearance_m"], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw) & np.isfinite(left) & np.isfinite(right)
            x, y, yaw, left, right = x[valid], y[valid], yaw[valid], left[valid], right[valid]
            if len(x) < 2:
                continue
            idx = downsample_indices(len(x))
            x, y, yaw, left, right = x[idx], y[idx], yaw[idx], left[idx], right[idx]
            left_x = x - np.sin(yaw) * left
            left_y = y + np.cos(yaw) * left
            right_x = x + np.sin(yaw) * right
            right_y = y - np.cos(yaw) * right
            output["routes"][str(row.sample_id)] = {
                "source_sha256": str(row.route_file_sha256),
                "vehicle_structure": str(row.vehicle_structure),
                "centerline": pairs(x, y),
                "left_boundary": pairs(left_x, left_y),
                "right_boundary": pairs(right_x, right_y),
                "point_count_original": int(valid.sum()),
                "point_count_display": int(len(x)),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(output['routes'])} routes to {args.output}")


if __name__ == "__main__":
    main()
