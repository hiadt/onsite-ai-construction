"""Task package and contradictory validation records remain traceable."""
import os
import tempfile
import unittest
import zipfile
from io import BytesIO
from unittest.mock import patch

from feedback_store import (classify_disagreement, load_feedback, make_condition_key,
                            save_feedback, set_training_review)
from task_store import create_task, export_task, load_routes, save_attachment
from batch_evaluation import evaluate_verified_task
import pandas as pd


class TaskWorkflowTests(unittest.TestCase):
    def test_task_preserves_route_and_evidence_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LOCALAPPDATA": directory}):
            task = create_task(title="东侧道路", scene="地图A", environment_version="测绘版1",
                               intake_status="待评估", vehicle_config={"vehicle_structure": "five_axis"},
                               routes=[("candidate.npz", b"route bytes")],
                               feature_version="v2", schema_sha256="schema", model_sha256="model")
            self.assertEqual(load_routes(task), [("candidate.npz", b"route bytes")])
            reference = save_attachment(task, "result.pdf", b"evidence bytes")
            self.assertIn("#sha256=", reference)
            with zipfile.ZipFile(BytesIO(export_task(task, b"id,result\n"))) as archive:
                names = archive.namelist()
                self.assertIn("task.json", names)
                self.assertIn("validation_feedback.csv", names)
                self.assertTrue(any(name.startswith("routes/") for name in names))
                self.assertTrue(any(name.startswith("evidence/") for name in names))

    def test_new_conflict_revokes_training_candidate(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LOCALAPPDATA": directory}):
            common = dict(task_id="task-1", sample_id="route-1", route_sha256="hash-1",
                          environment_version="map-v1", route_version="route-v1",
                          vehicle_config={"length_m": 12, "width_m": 3},
                          condition_consistency="与本次评估一致", review_status="已核对",
                          model_risk=0.2, validation_method="闭环仿真")
            first = save_feedback({**common, "validation_result": "通过"})
            set_training_review(first, "可用于训练候选")
            save_feedback({**common, "validation_result": "失败", "validation_method": "实车/现场测试"})
            records = load_feedback()
            self.assertTrue(records["training_review"].eq("待审核").all())
            self.assertTrue(records["evidence_level"].eq("已核对但证据待补").all())
            key = make_condition_key("hash-1", "map-v1", "route-v1", common["vehicle_config"])
            self.assertIn("冲突", classify_disagreement(records, 0.2, key))
            with self.assertRaisesRegex(ValueError, "矛盾"):
                set_training_review(first, "可用于训练候选")

    def test_batch_audit_uses_only_comparable_verified_routes(self):
        scored = pd.DataFrame({"route_file_sha256": [f"h{i}" for i in range(6)],
                               "model_risk": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
                               "rule_risk": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]})
        feedback = pd.DataFrame([{
            "task_id": "t1", "route_sha256": f"h{i}", "environment_version": "e1",
            "condition_consistency": "与本次评估一致", "review_status": "已核对",
            "validation_result": "失败" if i < 2 else "通过",
            "severity": "严重" if i == 0 else "一般",
        } for i in range(6)])
        result = evaluate_verified_task(scored, feedback, {"task_id": "t1", "environment_version": "e1"})
        self.assertEqual(result["status"], "描述性结果")
        model_top3 = next(row for row in result["rows"] if row["验证名额"] == 3 and row["排序方式"] == "学习模型")
        self.assertEqual(model_top3["失败捕获"], 2)
        self.assertEqual(model_top3["严重失败漏掉"], 0)


if __name__ == "__main__":
    unittest.main()
