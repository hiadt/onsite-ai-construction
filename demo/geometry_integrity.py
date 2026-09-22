"""Conservative checks for boundary coordinates aligned to a route centerline."""
from __future__ import annotations

import numpy as np


def screen_boundary_integrity(center, station, left, right, left_clearance, right_clearance):
    center = np.asarray(center, dtype=float)
    station = np.asarray(station, dtype=float).reshape(-1)
    left = np.asarray(left, dtype=float)[:, :2]
    right = np.asarray(right, dtype=float)[:, :2]
    left_clearance = np.asarray(left_clearance, dtype=float).reshape(-1)
    right_clearance = np.asarray(right_clearance, dtype=float).reshape(-1)
    n = len(station)
    if not (center.shape == left.shape == right.shape == (n, 2)
            and len(left_clearance) == len(right_clearance) == n and n >= 3):
        return {"distance_semantics_pass": False, "integrity_screen_pass": False, "reason": "array_shape_mismatch"}
    if not all(np.isfinite(a).all() for a in (center, station, left, right, left_clearance, right_clearance)):
        return {"distance_semantics_pass": False, "integrity_screen_pass": False, "reason": "non_finite_geometry"}
    distance_error = max(
        float(np.max(np.abs(np.linalg.norm(left - center, axis=1) - left_clearance))),
        float(np.max(np.abs(np.linalg.norm(right - center, axis=1) - right_clearance))),
    )
    ds = np.diff(station)
    center_step = np.linalg.norm(np.diff(center, axis=0), axis=1)
    if np.any(ds <= 0) or np.any(center_step <= 1e-8):
        return {"distance_semantics_pass": distance_error <= 1e-5,
                "integrity_screen_pass": False, "reason": "non_monotone_station_or_duplicate_centerline",
                "max_distance_error_m": distance_error}
    tangent = np.gradient(center, station, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm <= 1e-8):
        return {"distance_semantics_pass": distance_error <= 1e-5,
                "integrity_screen_pass": False, "reason": "undefined_route_tangent",
                "max_distance_error_m": distance_error}
    tangent /= norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    left_signed = np.sum((left - center) * normal, axis=1)
    right_signed = np.sum((right - center) * normal, axis=1)
    left_fraction = float(np.mean(left_signed > 0))
    right_fraction = float(np.mean(right_signed < 0))
    ordered_fraction = float(np.mean(left_signed > right_signed))
    ratio_left = np.linalg.norm(np.diff(left, axis=0), axis=1) / center_step
    ratio_right = np.linalg.norm(np.diff(right, axis=0), axis=1) / center_step
    p999 = float(max(np.quantile(ratio_left, 0.999), np.quantile(ratio_right, 0.999)))
    max_ratio = float(max(np.max(ratio_left), np.max(ratio_right)))
    checks = {
        "distance_semantics_pass": distance_error <= 1e-5,
        "left_side_orientation_pass": left_fraction >= 0.99,
        "right_side_orientation_pass": right_fraction >= 0.99,
        "left_right_order_pass": ordered_fraction >= 0.99,
        "continuity_screen_pass": p999 <= 3.0 and max_ratio <= 5.0,
    }
    return {
        **checks, "integrity_screen_pass": all(checks.values()), "reason": "",
        "max_distance_error_m": distance_error, "left_side_fraction": left_fraction,
        "right_side_fraction": right_fraction, "ordered_cross_section_fraction": ordered_fraction,
        "boundary_to_center_step_ratio_p99_9": p999,
        "boundary_to_center_step_ratio_max": max_ratio,
    }
