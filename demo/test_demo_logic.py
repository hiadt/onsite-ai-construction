"""Focused regression tests for the PathGuard demo contract and queue logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from demo_logic import apply_filters, evaluate_candidates, select_top_k
from inference import FeatureValidationError, load_contract, validate_feature_frame
from rule_baseline import compute_geometry_rule_risk


BASE_DIR = Path(__file__).resolve().parent


class DemoLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract(BASE_DIR / "data" / "feature_contract_v2.json")
        cls.sample = pd.read_csv(BASE_DIR / "data" / "sample_input.csv")

    def test_contract_and_sample_are_exact_v2(self) -> None:
        self.assertEqual(self.contract["feature_version"], "pathguard_preexec_v2.0.0")
        self.assertEqual(len(self.contract["training_feature_columns"]), 55)
        self.assertEqual(
            self.contract["schema_sha256"],
            "b365d5b00779cc9f1e2858e695f0063112d94771a2cbc611c734d1110983d334",
        )
        validated = validate_feature_frame(self.sample, self.contract)
        self.assertEqual(validated.shape, (10, 55))
        self.assertTrue(np.isfinite(validated.to_numpy()).all())

    def test_bad_feature_inputs_are_rejected(self) -> None:
        missing = self.sample.drop(columns=[self.contract["training_feature_columns"][0]])
        with self.assertRaises(FeatureValidationError):
            validate_feature_frame(missing, self.contract)
        nonnumeric = self.sample.copy()
        nonnumeric[self.contract["training_feature_columns"][1]] = nonnumeric[
            self.contract["training_feature_columns"][1]
        ].astype(object)
        nonnumeric.loc[0, self.contract["training_feature_columns"][1]] = "bad"
        with self.assertRaises(FeatureValidationError):
            validate_feature_frame(nonnumeric, self.contract)

    def test_rule_scores_are_finite(self) -> None:
        validated = validate_feature_frame(self.sample, self.contract)
        risk, components = compute_geometry_rule_risk(validated)
        self.assertTrue(np.isfinite(risk.to_numpy()).all())
        self.assertTrue(((risk >= 0) & (risk <= 1)).all())
        self.assertEqual(components.shape, (10, 6))

    def test_missing_model_never_creates_model_or_combined_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _, state = evaluate_candidates(
                self.sample, self.contract, Path(directory) / "missing.joblib"
            )
        self.assertFalse(state["model_available"])
        self.assertTrue(result["model_risk"].isna().all())
        self.assertTrue(result["combined_risk"].isna().all())
        self.assertTrue(result["risk_level"].eq("模型待补").all())

    def test_structure_filter_and_top_k(self) -> None:
        result, _, state = evaluate_candidates(
            self.sample, self.contract, BASE_DIR / "models" / "does-not-exist.joblib"
        )
        six_axis = apply_filters(result, "six_axis", "全部", "全部", "")
        self.assertEqual(len(six_axis), 5)
        self.assertTrue(six_axis["vehicle_structure"].eq("six_axis").all())
        top = select_top_k(result, "Top 10%", state["model_available"])
        self.assertEqual(len(top), 1)

    def test_summary_contains_no_invented_validation_metrics(self) -> None:
        summary = json.loads((BASE_DIR / "data" / "demo_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(all(value is None for key, value in summary["validation"].items() if key != "status"))


if __name__ == "__main__":
    unittest.main()
