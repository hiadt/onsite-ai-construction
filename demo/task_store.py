"""Local, reproducible assessment tasks and evidence packages."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def task_root() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PathGuard" / "tasks"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]", "_", Path(name).name, flags=re.UNICODE)[:100] or "route.npz"


def create_task(*, title: str, scene: str, environment_version: str,
                intake_status: str, vehicle_config: dict, routes: list[tuple[str, bytes]],
                feature_version: str = "", schema_sha256: str = "", model_sha256: str = "") -> dict:
    if not title.strip() or not scene.strip() or not environment_version.strip():
        raise ValueError("请填写任务名称、场景和环境版本。")
    if not routes:
        raise ValueError("请至少上传一条候选路线 NPZ。")
    if intake_status not in {"待评估", "已有验证记录"}:
        raise ValueError("任务输入状态无效。")
    task_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    folder = task_root() / task_id
    folder.mkdir()
    candidates = []
    seen = set()
    for original_name, payload in routes:
        if not payload or len(payload) > 40 * 1024 * 1024:
            raise ValueError(f"{original_name} 为空或超过40MB。")
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        filename = f"{len(candidates)+1:03d}_{_safe_name(original_name)}"
        (folder / filename).write_bytes(payload)
        candidates.append({"display_name": Path(original_name).stem, "filename": filename,
                           "sha256": digest, "route_version": "原始上传"})
    manifest = {
        "task_id": task_id, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title.strip(), "scene": scene.strip(), "environment_version": environment_version.strip(),
        "intake_status": intake_status, "training_review": "待审核",
        "vehicle_config": vehicle_config, "candidates": candidates,
        "feature_version": feature_version, "schema_sha256": schema_sha256,
        "model_sha256": model_sha256,
        "note": "每个文件是一条输入候选记录；是否为完整任务路线由数据提供方确认。",
    }
    (folder / "task.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def list_tasks() -> list[dict]:
    tasks = []
    for path in task_root().glob("*/task.json"):
        try:
            tasks.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(tasks, key=lambda item: item["created_at_utc"], reverse=True)


def load_routes(task: dict) -> list[tuple[str, bytes]]:
    folder = task_root() / task["task_id"]
    result = []
    for candidate in task["candidates"]:
        path = folder / candidate["filename"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != candidate["sha256"]:
            raise ValueError(f"路线文件哈希不匹配：{candidate['filename']}")
        result.append((candidate["display_name"] + ".npz", payload))
    return result


def save_attachment(task: dict, name: str, payload: bytes) -> str:
    if not payload or len(payload) > 40 * 1024 * 1024:
        raise ValueError("验证附件为空或超过40MB。")
    digest = hashlib.sha256(payload).hexdigest()
    folder = task_root() / task["task_id"] / "evidence"
    folder.mkdir(exist_ok=True)
    filename = digest[:16] + "_" + _safe_name(name)
    (folder / filename).write_bytes(payload)
    return "evidence/" + filename + "#sha256=" + digest


def save_assessment(task: dict, evaluated: pd.DataFrame, model_name: str) -> None:
    """Save a versioned prediction snapshot without replacing an earlier assessment."""
    folder = task_root() / task["task_id"] / "assessments"
    folder.mkdir(exist_ok=True)
    columns = [column for column in ("sample_id", "route_id", "route_file_sha256",
               "vehicle_structure", "model_risk", "rule_risk", "risk_level",
               "validation_priority", "decision_status") if column in evaluated]
    csv = evaluated[columns].to_csv(index=False).encode("utf-8-sig")
    assessment_id = hashlib.sha256((task.get("model_sha256", "") + task.get("schema_sha256", "")).encode() + csv).hexdigest()[:16]
    path = folder / (assessment_id + ".csv")
    if not path.exists():
        path.write_bytes(csv)
        (folder / (assessment_id + ".json")).write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": model_name, "model_sha256": task.get("model_sha256", ""),
            "schema_sha256": task.get("schema_sha256", ""), "assessment_csv_sha256": hashlib.sha256(csv).hexdigest(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def export_task(task: dict, feedback_csv: bytes) -> bytes:
    folder = task_root() / task["task_id"]
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(folder / "task.json", "task.json")
        for candidate in task["candidates"]:
            path = folder / candidate["filename"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["sha256"]:
                raise ValueError("路线文件发生变化，拒绝导出不一致的证据包。")
            archive.write(path, "routes/" + candidate["filename"])
        evidence_folder = folder / "evidence"
        if evidence_folder.exists():
            for path in evidence_folder.iterdir():
                if path.is_file():
                    archive.write(path, "evidence/" + path.name)
        assessment_folder = folder / "assessments"
        if assessment_folder.exists():
            for path in assessment_folder.iterdir():
                if path.is_file():
                    archive.write(path, "assessments/" + path.name)
        archive.writestr("validation_feedback.csv", feedback_csv)
    return result.getvalue()
