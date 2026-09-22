"""Transparent batch-relative geometry stress baseline for PathGuard."""

from __future__ import annotations

import numpy as np
import pandas as pd


RULE_GROUPS = {
    "曲率与变化": [
        ("curvature_abs_p95_1pm", "high"),
        ("curvature_change_abs_p95_1pm2", "high"),
        ("window10m_max_mean_abs_curvature_1pm", "high"),
    ],
    "速度曲率耦合": [
        ("speed_abs_curvature_product_p95_mpspm", "high"),
        ("lateral_accel_proxy_p95_mps2", "high"),
    ],
    "转向应力": [
        ("steer_abs_p95_rad", "high"),
        ("steer_change_abs_p95_radpm", "high"),
        ("window10m_max_mean_abs_steer_change_radpm", "high"),
    ],
    "横摆应力": [
        ("yaw_rate_abs_p95_radps", "high"),
        ("yaw_rate_change_abs_p95_radps_per_m", "high"),
    ],
    "坡度变化": [("slope_change_abs_p95_1pm", "high")],
    "静态边界余量": [
        ("static_left_clearance_min_m", "low"),
        ("static_right_clearance_min_m", "low"),
        ("static_boundary_clearance_min_m", "low"),
        ("window10m_min_mean_left_clearance_m", "low"),
        ("window10m_min_mean_right_clearance_m", "low"),
    ],
}


def required_rule_features() -> list[str]:
    return list(dict.fromkeys(name for entries in RULE_GROUPS.values() for name, _ in entries))


def _percentile_stress(series: pd.Series, direction: str) -> pd.Series:
    if len(series) < 2 or series.nunique(dropna=False) < 2:
        return pd.Series(0.5, index=series.index, dtype=float)
    ascending = direction == "high"
    return series.rank(method="average", pct=True, ascending=ascending).astype(float)


def compute_geometry_rule_risk(features: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    missing = [column for column in required_rule_features() if column not in features.columns]
    if missing:
        raise ValueError("几何规则缺少特征：" + ", ".join(missing))

    components = pd.DataFrame(index=features.index)
    for group_name, entries in RULE_GROUPS.items():
        stresses = [
            _percentile_stress(features[column].astype(float), direction)
            for column, direction in entries
        ]
        components[group_name] = pd.concat(stresses, axis=1).mean(axis=1)
    risk = components.mean(axis=1).clip(0.0, 1.0)
    if not np.isfinite(risk.to_numpy()).all():
        raise ValueError("几何规则输出包含NaN或无穷值。")
    return risk, components
