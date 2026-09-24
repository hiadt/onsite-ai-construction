"""Read-only checks that V5 equations agree with the actual inference code.

Synthetic fixtures here validate numerical definitions, not model performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from route_input import extractor, compute_rigid_body_sweep, compute_boundary_rule
from rule_baseline import _percentile_stress, RULE_GROUPS
from demo_logic import select_validation_queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = {}
    s = np.array([0., .3, 1.2, 3., 8., 10.4, 17.])
    f = s ** 2
    hm, hp = np.diff(s)[:-1], np.diff(s)[1:]
    expression = (-hp*f[:-2]/(hm*(hm+hp)) + (hp-hm)*f[1:-1]/(hm*hp)
                  + hm*f[2:]/(hp*(hm+hp)))
    np.testing.assert_allclose(expression, extractor.derivative(f, s)[1:-1], atol=1e-12)
    checks["nonuniform_gradient"] = "passed"
    expected = [f[(s >= value-10) & (s <= value)].mean() for value in s]
    np.testing.assert_allclose(expected, extractor.rolling_means(f, s, 10))
    checks["trailing_10m_sample_mean"] = "passed"
    np.testing.assert_allclose(_percentile_stress(pd.Series([2., 4.]), "high", [1, 2, 2, 4]), [.5, .875])
    np.testing.assert_allclose(_percentile_stress(pd.Series([2., 4.]), "low", [1, 2, 2, 4]), [.5, .125])
    assert [len(group) for group in RULE_GROUPS.values()] == [3, 2, 3, 2, 1, 5]
    checks["midrank_and_group_sizes"] = "passed"
    geometry = {"centerline": [[0., 0.], [10., 0.]], "yaw_rad": [0., np.pi/2]}
    x, y = compute_rigid_body_sweep(geometry, 10, 4, 5)
    np.testing.assert_allclose(x[0], [5, 5, -5, -5])
    np.testing.assert_allclose(y[1], [5, 5, -5, -5])
    checks["rigid_transform"] = "passed"
    station = np.arange(0., 101.)
    g = {"centerline": np.column_stack([station, station*0]).tolist(),
         "station_m": station.tolist(), "yaw_rad": (station*0).tolist(),
         "left_boundary": np.column_stack([station, station*0+3]).tolist(),
         "right_boundary": np.column_stack([station, station*0-3]).tolist(),
         "boundary_integrity_screen_passed": True, "boundary_semantics_verified": True}
    result = compute_boundary_rule(g, 10, 4, 5)
    assert result["available"]
    np.testing.assert_allclose(result["minimum_margin_m"], 1.)
    checks["straight_corridor_margin"] = "passed"
    frame = pd.DataFrame({"model_risk": np.linspace(.9, .1, 12), "rule_risk": .5,
                          "mandatory_review": [False]*10 + [True]*2})
    queue = select_validation_queue(frame, "Top 3", True)
    assert len(queue) == 5 and int(queue.mandatory_review.sum()) == 2
    checks["ordinary_budget_plus_mandatory"] = "passed"
    artifact = joblib.load(ROOT / "demo/models/pathguard_gate3_model.joblib")
    model = artifact["model"]
    assert type(model).__name__ == "HistGradientBoostingClassifier"
    assert model.n_iter_ == 250 and model.n_features_in_ == 55
    assert model.get_params()["min_samples_leaf"] == 20
    checks["deployed_model_identity"] = "passed"
    files = ["demo/inference.py", "demo/rule_baseline.py", "demo/demo_logic.py", "demo/route_input.py",
             "demo/geometry_integrity.py", "demo/feedback_store.py", "demo/tracking_evidence.py",
             "scripts/data_audit/extract_gate2_pre_execution_features_shanyu000.py",
             "demo/models/pathguard_gate3_model.joblib", "demo/data/feature_contract_v2.json",
             "reports/final/tables/heldout_methods_v3.csv"]
    audit = {"scope": "synthetic definition checks, not additional training or evaluation samples",
             "checks": checks, "model_class": type(model).__name__, "parameters": model.get_params(),
             "n_iter": int(model.n_iter_), "early_stopping_active": bool(model.do_early_stopping_),
             "artifact_metadata": {k:v for k,v in artifact.items() if k not in {"model", "features"}},
             "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
