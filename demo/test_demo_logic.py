"""Focused regression tests for the PathGuard demo contract and queue logic."""

from __future__ import annotations

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
        cls.sample = pd.read_csv(BASE_DIR / "data" / "sample_input.csv", encoding="utf-8-sig")

    def test_contract_and_sample_are_exact_v2(self) -> None:
        self.assertEqual(self.contract["feature_version"], "pathguard_preexec_v2.0.0")
        self.assertEqual(len(self.contract["training_feature_columns"]), 55)
        self.assertEqual(
            self.contract["schema_sha256"],
            "b365d5b00779cc9f1e2858e695f0063112d94771a2cbc611c734d1110983d334",
        )
        validated = validate_feature_frame(self.sample, self.contract)
        self.assertEqual(validated.shape, (12, 55))
        self.assertTrue(np.isfinite(validated.to_numpy()).all())

    def test_bad_feature_inputs_are_rejected(self) -> None:
        missing = self.sample.drop(columns=[self.contract["training_feature_columns"][0]])
        with self.assertRaisesRegex(FeatureValidationError, "route_length_m"):
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
        self.assertEqual(components.shape, (12, 6))

    def test_real_gate3_model_is_loaded_and_scores_are_finite(self) -> None:
        result, _, state = evaluate_candidates(
            self.sample, self.contract, BASE_DIR / "models" / "pathguard_gate3_model.joblib"
        )
        self.assertTrue(state["model_available"])
        self.assertEqual(state["model_name"], "HistGradientBoostingClassifier")
        self.assertTrue(
            np.isfinite(result[["model_risk", "rule_risk", "combined_risk"]].to_numpy()).all()
        )
        self.assertTrue(result["combined_risk"].between(0, 1).all())

    def test_missing_model_never_creates_model_or_combined_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _, state = evaluate_candidates(
                self.sample, self.contract, Path(directory) / "missing.joblib"
            )
        self.assertFalse(state["model_available"])
        self.assertIn("模型文件不存在", state["model_error"])
        self.assertTrue(result["model_risk"].isna().all())
        self.assertTrue(result["combined_risk"].isna().all())
        self.assertTrue(
            result.apply(
                lambda row: row["risk_level"] == ("证据不足" if row["vehicle_structure"] == "unknown" else "模型待补"),
                axis=1,
            ).all()
        )

    def test_structure_filter_and_top_k(self) -> None:
        result, _, state = evaluate_candidates(
            self.sample, self.contract, BASE_DIR / "models" / "pathguard_gate3_model.joblib"
        )
        six_axis = apply_filters(result, "six_axis", "全部", "全部", "")
        self.assertEqual(len(six_axis), 4)
        self.assertTrue(six_axis["vehicle_structure"].eq("six_axis").all())
        unknown = apply_filters(result, "unknown", "全部", "全部", "")
        self.assertEqual(len(unknown), 4)
        self.assertTrue(unknown["next_action"].eq("人工复核").all())
        top = select_top_k(result, "Top 34", state["model_available"])
        self.assertEqual(len(top), 12)
        self.assertEqual(len(select_top_k(result, "Top 10", True)), 10)
        self.assertEqual(len(select_top_k(result, "Top 20", True)), 12)

    def test_input_identity_mismatch_is_rejected(self) -> None:
        mismatched = self.sample.copy()
        mismatched["feature_version"] = "wrong"
        with self.assertRaises(FeatureValidationError):
            evaluate_candidates(
                mismatched, self.contract, BASE_DIR / "models" / "pathguard_gate3_model.joblib"
            )


if __name__ == "__main__":
    unittest.main()
