#!/usr/bin/env python3
"""Read-only inventory and candidate recovery for local PathGuard artifacts.

The repository currently has no frozen per-sample baseline manifest.  Therefore
comparison_status is deliberately set to ``pending_baseline_merge`` and the
script only makes local duplicate/conflict determinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # recorded in the report; JSON inventory still works
    np = None


USER = "SHANYU000"
IMPORTANT_JSON = {
    "summary.json",
    "safety_summary.json",
    "validation_report.json",
    "selection_report.json",
    "manifest.json",
    "checkpoint.json",
    "calibration_manifest.json",
    "profile.json",
    "vehicle_profile.json",
}
PROFILE_HINTS = ("profile", "calibration", "vehicle", "manifest")
RELEVANT_FIELDS = {
    "map",
    "map_name",
    "map_id",
    "scene",
    "scenario",
    "hard_certificate_passed",
    "hard_failure_reasons",
    "vehicle_profile_id",
    "vehicle_profile_fingerprint",
    "axle_count",
    "wheel_count",
    "steer_ff_rad",
    "dynamic_pcd_clearance",
    "dynamic_pcd_clearance_m",
    "closed_loop_pcd_clearance",
    "maximum_slip_ratio",
    "maximum_tire_utilization",
    "maximum_abs_lateral_error",
    "maximum_abs_lateral_error_m",
    "maximum_abs_heading_error",
    "maximum_abs_heading_error_deg",
    "completed",
    "policy_id",
    "hard_thresholds",
    "policy",
    "thresholds",
}
SKIP_DIRS = {
    ".git",
    ".idea",
    ".vs",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "trt_cache",
}

INVENTORY_FIELDS = [
    "source_machine", "scan_root", "path_local", "file_name", "kind",
    "size_bytes", "modified_time", "sha256", "parse_status", "error",
    "transfer_recommended",
]
CANDIDATE_FIELDS = [
    "source_machine", "map", "report_path_local", "route_path_local",
    "build_family", "report_file_sha256", "route_file_sha256",
    "route_input_hash", "route_input_hash_method", "hard_certificate_passed",
    "hard_failure_reasons", "completed", "policy_id", "hard_thresholds",
    "dynamic_pcd_clearance_m", "maximum_slip_ratio",
    "maximum_tire_utilization", "maximum_abs_lateral_error_m",
    "maximum_abs_heading_error_deg", "vehicle_profile_id",
    "vehicle_profile_fingerprint", "axle_count", "wheel_count",
    "steer_ff_rad_second_dim", "vehicle_structure", "identity_evidence",
    "route_length_m", "point_count", "report_modified_time",
    "route_modified_time", "comparison_status", "local_relation",
    "transfer_recommended", "parse_notes",
]
VEHICLE_FIELDS = [
    "source_machine", "evidence_path_local", "evidence_file_sha256", "map",
    "vehicle_profile_id", "vehicle_profile_fingerprint", "axle_count",
    "wheel_count", "steer_ff_rad_second_dim", "vehicle_structure",
    "identity_evidence", "calibration_or_profile_kind", "modified_time",
    "parse_notes",
]


def clean_scalar(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def json_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def flatten_json(value: Any, out: dict[str, Any], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in RELEVANT_FIELDS and key_text not in out:
                out[key_text] = child
            flatten_json(child, out, next_prefix)
    elif isinstance(value, list):
        for child in value[:1000]:
            flatten_json(child, out, prefix)


def find_npz_strings(value: Any, parent_key: str = "") -> list[str]:
    found: list[tuple[int, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_paths = find_npz_strings(child, str(key))
            for child_path in child_paths:
                key_text = str(key).lower()
                priority = 0 if any(hint in key_text for hint in ("route", "candidate", "trajectory", "input")) else 1
                if any(hint in key_text for hint in ("replay", "telemetry", "diagnostic")):
                    priority = 2
                found.append((priority, child_path))
    elif isinstance(value, list):
        for child in value[:1000]:
            for child_path in find_npz_strings(child, parent_key):
                found.append((1, child_path))
    elif isinstance(value, str) and value.lower().endswith(".npz"):
        return [value]
    ordered: list[str] = []
    seen: set[str] = set()
    for _, path in sorted(found, key=lambda item: item[0]):
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def first_value(flat: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = flat.get(key)
        if value is not None:
            return clean_scalar(value)
    return ""


def infer_map(path: Path, flat: dict[str, Any]) -> str:
    def normalize(value: Any) -> str:
        text = str(value).strip()
        match = re.search(
            r"(turnaround|overpass|mounarea|bridge|cliff|cross|ramp|narrow)[_-]?([1-6])(?!\d)",
            text,
            re.I,
        )
        if match:
            return f"{match.group(1).lower()}_{match.group(2)}"
        if re.search(r"roadc", text, re.I):
            return "RoadC"
        return text

    for key in ("map", "map_name", "map_id", "scene", "scenario"):
        if flat.get(key):
            return normalize(flat[key])
    pattern = re.compile(
        r"((?:turnaround|overpass|mounarea|bridge|cliff|cross|ramp|narrow)[_-]?[1-6](?!\d)|roadc)",
        re.I,
    )
    match = pattern.search(str(path))
    return normalize(match.group(1)) if match else ""


def infer_build_family(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
        return relative.parts[0] if len(relative.parts) > 1 else root.name
    except ValueError:
        return root.name


def structure_from_evidence(
    profile_id: Any, wheel_count: Any, axle_count: Any, steer_dim: Any
) -> tuple[str, str]:
    profile_text = str(profile_id or "").strip()
    try:
        wheels = int(wheel_count) if str(wheel_count) not in ("", "None") else None
    except (TypeError, ValueError):
        wheels = None
    try:
        axles = int(axle_count) if str(axle_count) not in ("", "None") else None
    except (TypeError, ValueError):
        axles = None
    explicit: str | None = None
    if wheels == 12 or axles == 6 or re.search(r"six|6.?axis|6.?axle", profile_text, re.I):
        explicit = "six_axis_explicit"
    elif wheels == 10 or axles == 5 or re.search(r"five|5.?axis|5.?axle", profile_text, re.I):
        explicit = "five_axis_explicit"
    structural = None
    try:
        dim = int(steer_dim)
        structural = {12: "six_axis_structural", 10: "five_axis_structural"}.get(dim)
    except (TypeError, ValueError):
        structural = None
    if explicit and structural and explicit.split("_axis")[0] != structural.split("_axis")[0]:
        return "unknown_or_conflict", "conflicting_evidence"
    if profile_text:
        return explicit or "unknown_or_conflict", "explicit_profile_id"
    if wheels is not None:
        return explicit or "unknown_or_conflict", "explicit_wheel_count"
    if axles is not None:
        return explicit or "unknown_or_conflict", "explicit_axle_count"
    if structural:
        return structural, "steer_vector_dimension"
    return "unknown_or_conflict", "path_only"


def safe_array_scalar(data: Any, key: str) -> Any:
    if key not in data.files:
        return ""
    arr = data[key]
    if getattr(arr, "size", 0) == 1:
        return clean_scalar(arr.reshape(-1)[0])
    return ""


def inspect_npz(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "point_count": "", "route_length_m": "", "steer_ff_rad_second_dim": "",
        "vehicle_profile_id": "", "vehicle_profile_fingerprint": "",
        "axle_count": "", "wheel_count": "", "route_input_hash": "",
        "route_input_hash_method": "", "artifact_role": "unknown_npz",
        "parse_notes": "",
    }
    if np is None:
        result["parse_notes"] = "numpy_unavailable"
        return result
    try:
        with np.load(path, allow_pickle=False) as data:
            files = list(data.files)
            file_set = set(files)
            coordinate_keys = {"x_m", "y_m"}.issubset(file_set) or any(
                key in file_set for key in ("position", "positions", "xyz", "route_xyz", "pos", "path")
            )
            route_signals = {
                "s_m", "steer_ff_rad", "kappa_1pm", "dkappa_ds_1pm2",
                "v_profile_mps", "body_clearance_m", "left_clearance_m", "right_clearance_m",
            }
            route_path_hint = bool(re.search(r"(?:^|[\\/])routes?(?:[\\/]|$)", str(path), re.I))
            telemetry_signals = {"lateral_error_m", "heading_error_rad", "tire_utilization", "max_slip_ratio"}
            if coordinate_keys and (file_set & route_signals or route_path_hint):
                result["artifact_role"] = "candidate_route"
            elif file_set & telemetry_signals or re.search(r"replay|telemetry|certificate", str(path), re.I):
                result["artifact_role"] = "telemetry_or_derived_npz"
            for key in ("vehicle_profile_id", "vehicle_profile_fingerprint", "axle_count", "wheel_count"):
                result[key] = safe_array_scalar(data, key)
            if "steer_ff_rad" in files:
                steer = data["steer_ff_rad"]
                if steer.ndim >= 2:
                    result["steer_ff_rad_second_dim"] = int(steer.shape[1])
            position = None
            if {"x_m", "y_m"}.issubset(file_set):
                columns = [np.asarray(data["x_m"], dtype=np.float64)]
                columns.append(np.asarray(data["y_m"], dtype=np.float64))
                if "z_m" in file_set and data["z_m"].shape == data["x_m"].shape:
                    columns.append(np.asarray(data["z_m"], dtype=np.float64))
                if all(column.ndim == 1 for column in columns):
                    position = np.column_stack(columns)
            for key in ("position", "positions", "xyz", "route_xyz", "pos", "path"):
                if position is None and key in files:
                    arr = data[key]
                    if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] >= 2:
                        position = np.asarray(arr[:, :3], dtype=np.float64)
                        break
            if position is not None:
                result["point_count"] = int(position.shape[0])
                if position.shape[0] > 1:
                    result["route_length_m"] = float(
                        np.linalg.norm(np.diff(position, axis=0), axis=1).sum()
                    )
            elif files:
                shapes = [getattr(data[key], "shape", ()) for key in files]
                first_dims = [shape[0] for shape in shapes if shape]
                if first_dims:
                    result["point_count"] = int(max(first_dims))
            # No repository implementation of the historical route-input hash is
            # present.  Do not invent a supposedly compatible hash here.
            result["parse_notes"] = "route_input_hash_pending_repository_rule"
    except Exception as exc:
        result["parse_notes"] = f"npz_parse_error:{type(exc).__name__}:{exc}"
    return result


def resolve_route(
    report_path: Path, raw_paths: list[str], npz_by_name: dict[str, list[Path]],
    candidate_paths: set[Path],
) -> tuple[Path | None, str]:
    notes: list[str] = []
    for raw in raw_paths:
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file() and candidate in candidate_paths:
            return candidate, "explicit_path"
        local = report_path.parent / candidate.name
        if local.exists() and local.is_file() and local in candidate_paths:
            return local, "explicit_basename_near_report"
        matches = npz_by_name.get(candidate.name.lower(), [])
        if len(matches) == 1:
            return matches[0], "explicit_basename_unique"
        if len(matches) > 1:
            raw_parts = [part.casefold() for part in candidate.parts]
            scored: list[tuple[int, Path]] = []
            for match in matches:
                match_parts = [part.casefold() for part in match.parts]
                score = 0
                for left, right in zip(reversed(raw_parts), reversed(match_parts)):
                    if left != right:
                        break
                    score += 1
                scored.append((score, match))
            best_score = max(score for score, _ in scored)
            best = [match for score, match in scored if score == best_score]
            if best_score >= 3 and len(best) == 1:
                return best[0], f"explicit_suffix_match:{best_score}"
            notes.append(f"ambiguous_route_basename:{candidate.name}:{len(matches)}:best_suffix={best_score}")
    nearby = [path for path in report_path.parent.glob("*.npz") if path in candidate_paths]
    if len(nearby) == 1:
        return nearby[0], "single_npz_near_report"
    if len(nearby) > 1:
        notes.append(f"multiple_npz_near_report:{len(nearby)}")
    return None, ";".join(notes) or "route_not_resolved"


def classify_file(path: Path) -> str | None:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in IMPORTANT_JSON:
        if name in {"profile.json", "vehicle_profile.json", "calibration_manifest.json"}:
            return "vehicle_or_calibration_json"
        return "report_or_manifest_json"
    if suffix == ".npz":
        return "route_or_data_npz"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json" and any(hint in name for hint in PROFILE_HINTS):
        return "vehicle_or_calibration_json"
    return None


def walk_roots(roots: list[Path]) -> tuple[list[tuple[Path, Path]], list[dict[str, str]]]:
    files: list[tuple[Path, Path]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            errors.append({"root": str(root), "path": str(root), "error": "root_missing"})
            continue
        def record_walk_error(exc: OSError, scan_root: Path = root) -> None:
            errors.append({
                "root": str(scan_root),
                "path": str(getattr(exc, "filename", "") or scan_root),
                "error": f"{type(exc).__name__}:{exc}",
            })

        for current, dirs, names in os.walk(root, topdown=True, onerror=record_walk_error):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            current_path = Path(current)
            for name in names:
                path = current_path / name
                if classify_file(path) is None:
                    continue
                key = os.path.normcase(str(path.resolve()))
                if key not in seen:
                    seen.add(key)
                    files.append((root, path))
    return files, errors


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_text(row.get(key, "")) for key in fields})


def load_hash_cache(path: Path) -> dict[tuple[str, str, str], str]:
    cache: dict[tuple[str, str, str], str] = {}
    if not path.exists():
        return cache
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                digest = row.get("sha256", "")
                if digest:
                    cache[(row.get("path_local", ""), row.get("size_bytes", ""), row.get("modified_time", ""))] = digest
    except Exception:
        return {}
    return cache


def parse_json(path: Path) -> tuple[Any, dict[str, Any], str]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        flat: dict[str, Any] = {}
        flatten_json(payload, flat)
        return payload, flat, "ok"
    except Exception as exc:
        return None, {}, f"json_parse_error:{type(exc).__name__}:{exc}"


def bool_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def local_relations(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = row.get("route_input_hash") or row.get("route_file_sha256")
        if identity:
            groups[str(identity)].append(row)
        else:
            row["local_relation"] = "unresolved_route"
    for group in groups.values():
        labels = {bool_label(row.get("hard_certificate_passed")) for row in group}
        labels.discard(None)
        if len(labels) > 1:
            for row in group:
                row["local_relation"] = "conflict"
        else:
            ordered = sorted(
                group,
                key=lambda row: (
                    bool_label(row.get("hard_certificate_passed")) is None,
                    not bool(row.get("report_path_local")),
                    str(row.get("report_path_local", "")),
                ),
            )
            representative = ordered[0]
            representative["local_relation"] = (
                "new_route_with_label"
                if bool_label(representative.get("hard_certificate_passed")) is not None
                else "new_route_without_label"
            )
            representative_report = representative.get("report_file_sha256")
            for row in ordered[1:]:
                report_hash = row.get("report_file_sha256")
                row["local_relation"] = (
                    "existing_route_new_report"
                    if report_hash and report_hash != representative_report
                    else "existing_exact"
                )


def markdown_report(
    roots: list[Path], inventory: list[dict[str, Any]], candidates: list[dict[str, Any]],
    vehicles: list[dict[str, Any]], scan_errors: list[dict[str, str]], output: Path,
) -> None:
    kinds = Counter(row["kind"] for row in inventory)
    summary_names = {
        "summary.json", "safety_summary.json", "validation_report.json", "selection_report.json"
    }
    summary_count = sum(str(row.get("file_name", "")).lower() in summary_names for row in inventory)
    certificate_json_count = sum(
        row.get("kind") == "report_or_manifest_json"
        and "certificate" in str(row.get("path_local", "")).lower()
        for row in inventory
    )
    inventory_failures = [row for row in inventory if row.get("parse_status") == "failed"]
    relations = Counter(row.get("local_relation", "") for row in candidates)
    representative_rows = [
        row for row in candidates
        if row.get("local_relation") in {"new_route_with_label", "new_route_without_label"}
    ]
    structures = Counter(row.get("vehicle_structure", "unknown_or_conflict") for row in representative_rows)
    valid = [
        row for row in representative_rows
        if row.get("route_path_local") and bool_label(row.get("hard_certificate_passed")) is not None
    ]
    maps = sorted({str(row["map"]) for row in candidates if row.get("map")})
    build_families = sorted({
        str(row["build_family"]) for row in representative_rows if row.get("build_family")
    })
    candidate_route_files = {
        str(row["route_path_local"]) for row in candidates if row.get("route_path_local")
    }
    failures: set[str] = set()
    for row in candidates:
        raw = row.get("hard_failure_reasons")
        if not raw:
            continue
        if isinstance(raw, list):
            failures.update(map(str, raw))
        else:
            failures.add(str(raw))
    recommended = [row for row in inventory if row.get("transfer_recommended")]
    recommended_bytes = sum(int(row.get("size_bytes") or 0) for row in recommended)
    report_lines = [
        "# SHANYU000 本机测试数据扩展报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "本次仅进行只读搜索、识别、哈希、内部去重和清单整理；未修改原始文件，未进行模型实验。",
        "",
        "## 扫描范围",
        "",
        *[f"- `{root}`" for root in roots],
        "",
        "固定磁盘识别结果：`C:` 与 `D:`。Windows、软件安装目录及无关个人资料未扫描。",
        "",
        "## 统计摘要",
        "",
        f"- 索引文件总数：{len(inventory)}",
        f"- summary/validation/selection 报告：{summary_count}",
        f"- certificate 路径下的报告或清单 JSON：{certificate_json_count}",
        f"- summary/report/manifest JSON：{kinds['report_or_manifest_json']}",
        f"- 路线或数据 NPZ：{kinds['route_or_data_npz']}",
        f"- 识别为候选路线的 NPZ 文件：{len(candidate_route_files)}",
        f"- 车型/标定 JSON：{kinds['vehicle_or_calibration_json']}",
        f"- JSONL：{kinds['jsonl']}；CSV：{kinds['csv']}",
        f"- 可形成有效样本（已关联路线且有布尔标签）：{len(valid)}",
        f"- 本机唯一路线且有标签：{relations['new_route_with_label']}",
        f"- 本机唯一路线但无标签：{relations['new_route_without_label']}",
        f"- 同路线新报告：{relations['existing_route_new_report']}",
        f"- 完全重复：{relations['existing_exact']}",
        f"- 冲突：{relations['conflict']}",
        f"- 未能关联路线：{relations['unresolved_route']}",
        "",
        "以上“本机唯一”不是相对项目主数据集的最终新增结论。仓库目前没有完整基线清单，所有样本均标为 `pending_baseline_merge`，待项目负责人合并去重。",
        "",
        "## 车型结构证据",
        "",
        f"- 六轴显式：{structures['six_axis_explicit']}",
        f"- 六轴结构推断：{structures['six_axis_structural']}",
        f"- 五轴显式：{structures['five_axis_explicit']}",
        f"- 五轴结构推断：{structures['five_axis_structural']}",
        f"- 未知或冲突：{structures['unknown_or_conflict']}",
        f"- 车辆/标定证据记录：{len(vehicles)}",
        "",
        "## 地图、构建批次和失败类型",
        "",
        f"- 本机清单识别地图：{', '.join(maps) if maps else '未从可解析字段或路径可靠识别'}",
        f"- 本机构建批次/族（{len(build_families)}）：{', '.join(build_families) if build_families else '未识别'}",
        f"- 失败类型：{'; '.join(sorted(failures)) if failures else '未从可解析报告识别'}",
        "- 新增地图和新增失败类型：待与主数据集合并去重。",
        "- 新车辆配置或标定包：已写入 `vehicle_configuration_evidence.csv`，是否属于项目新增待与主数据集合并确认。",
        "",
        "## 建议后续集中复制",
        "",
        f"- 标记 `transfer_recommended=true`：{len(recommended)} 个文件，总大小 {recommended_bytes} 字节。",
        "- 第一阶段不上传任何原始 NPZ、完整工程、模型、压缩备份或敏感配置；只提交清单、哈希、脚本和本报告。",
        "",
        "## 限制与异常",
        "",
        "- 仓库未提供完整基线逐样本清单，无法在本机阶段给出相对246条主分析路线的最终新增数量。",
        "- 仓库未提供历史 `route_input_hash` 的可执行算法，因此未伪造兼容哈希；本机去重优先使用路线文件 SHA-256。",
        "- 旧报告中的绝对路线可能已失效；脚本仅在路径存在、同目录唯一NPZ或全局文件名唯一时建立关联。",
        "- `steer_ff_rad` 结构推断只作为车型结构证据，不填充具体商业车型。",
        f"- NumPy 状态：{'可用 ' + np.__version__ if np is not None else '不可用，NPZ内部字段未解析'}。",
        f"- 扫描根或权限异常：{len(scan_errors)}；详见 `local_file_inventory.csv` 中解析状态及本次运行日志。",
        f"- 文件读取或解析失败：{len(inventory_failures)}；失败路径和错误已保留在 `local_file_inventory.csv`。",
        "- 技术性能数字未在本报告中新增；仓库既有246条路线、25张地图等口径以冻结报告为准。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="Read-only scan root; repeatable")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    roots = [Path(item).resolve() for item in args.root]
    if not roots:
        parser.error("at least one --root is required")
    repo = args.repo.resolve()
    analysis_dir = repo / "analysis" / "data_recovery" / USER
    report_path = repo / "reports" / "data_recovery" / USER / "本机测试数据扩展报告.md"
    source_machine = socket.gethostname()

    hash_cache = load_hash_cache(analysis_dir / "local_file_inventory.csv")

    discovered, scan_errors = walk_roots(roots)
    inventory: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    parsed_json: dict[str, tuple[Any, dict[str, Any], str]] = {}
    for root, path in discovered:
        kind = classify_file(path) or "other"
        row = {
            "source_machine": source_machine,
            "scan_root": str(root),
            "path_local": str(path),
            "file_name": path.name,
            "kind": kind,
            "size_bytes": "",
            "modified_time": "",
            "sha256": "",
            "parse_status": "not_parsed",
            "error": "",
            "transfer_recommended": False,
        }
        try:
            row["size_bytes"] = path.stat().st_size
            row["modified_time"] = iso_mtime(path)
            cache_key = (str(path), str(row["size_bytes"]), str(row["modified_time"]))
            row["sha256"] = hash_cache.get(cache_key) or sha256_file(path)
            file_hashes[str(path)] = row["sha256"]
            if path.suffix.lower() == ".json":
                parsed_json[str(path)] = parse_json(path)
                row["parse_status"] = parsed_json[str(path)][2]
            elif path.suffix.lower() == ".npz":
                row["parse_status"] = "indexed_npz"
            else:
                row["parse_status"] = "indexed_not_deep_parsed"
        except Exception as exc:
            row["parse_status"] = "failed"
            row["error"] = f"{type(exc).__name__}:{exc}"
        inventory.append(row)

    npz_cache: dict[str, dict[str, Any]] = {}
    npz_by_name: dict[str, list[Path]] = defaultdict(list)
    candidate_paths: set[Path] = set()
    for _, path in discovered:
        if path.suffix.lower() != ".npz":
            continue
        info = inspect_npz(path)
        npz_cache[str(path)] = info
        if info.get("artifact_role") == "candidate_route":
            npz_by_name[path.name.lower()].append(path)
            candidate_paths.add(path)

    candidates: list[dict[str, Any]] = []
    vehicles: list[dict[str, Any]] = []
    for root, report_path_local in discovered:
        if report_path_local.suffix.lower() != ".json":
            continue
        payload, flat, status = parsed_json.get(str(report_path_local), (None, {}, "not_parsed"))
        kind = classify_file(report_path_local)
        if status != "ok":
            continue
        is_report = "hard_certificate_passed" in flat or any(
            key in flat for key in ("hard_failure_reasons", "dynamic_pcd_clearance_m", "maximum_slip_ratio")
        )
        raw_npz_paths = find_npz_strings(payload)
        route_path, resolution_note = resolve_route(
            report_path_local, raw_npz_paths, npz_by_name, candidate_paths
        )
        route_info: dict[str, Any] = {}
        if route_path is not None:
            route_info = npz_cache.setdefault(str(route_path), inspect_npz(route_path))
        profile_id = first_value(flat, "vehicle_profile_id") or route_info.get("vehicle_profile_id", "")
        fingerprint = first_value(flat, "vehicle_profile_fingerprint") or route_info.get("vehicle_profile_fingerprint", "")
        axle_count = first_value(flat, "axle_count") or route_info.get("axle_count", "")
        wheel_count = first_value(flat, "wheel_count") or route_info.get("wheel_count", "")
        steer_dim = route_info.get("steer_ff_rad_second_dim", "")
        structure, evidence = structure_from_evidence(profile_id, wheel_count, axle_count, steer_dim)
        map_name = infer_map(report_path_local, flat)

        if kind == "vehicle_or_calibration_json" or profile_id or fingerprint or axle_count or wheel_count:
            vehicles.append({
                "source_machine": source_machine,
                "evidence_path_local": str(report_path_local),
                "evidence_file_sha256": file_hashes.get(str(report_path_local), ""),
                "map": map_name,
                "vehicle_profile_id": profile_id,
                "vehicle_profile_fingerprint": fingerprint,
                "axle_count": axle_count,
                "wheel_count": wheel_count,
                "steer_ff_rad_second_dim": steer_dim,
                "vehicle_structure": structure,
                "identity_evidence": evidence,
                "calibration_or_profile_kind": kind,
                "modified_time": iso_mtime(report_path_local),
                "parse_notes": resolution_note,
            })
        if not is_report:
            continue
        label = bool_label(first_value(flat, "hard_certificate_passed"))
        candidate = {
            "source_machine": source_machine,
            "map": map_name,
            "report_path_local": str(report_path_local),
            "route_path_local": str(route_path) if route_path else "",
            "build_family": infer_build_family(report_path_local, root),
            "report_file_sha256": file_hashes.get(str(report_path_local), ""),
            "route_file_sha256": file_hashes.get(str(route_path), "") if route_path else "",
            "route_input_hash": route_info.get("route_input_hash", ""),
            "route_input_hash_method": route_info.get("route_input_hash_method", ""),
            "hard_certificate_passed": label if label is not None else "",
            "hard_failure_reasons": first_value(flat, "hard_failure_reasons"),
            "completed": first_value(flat, "completed"),
            "policy_id": first_value(flat, "policy_id"),
            "hard_thresholds": first_value(flat, "hard_thresholds", "policy", "thresholds"),
            "dynamic_pcd_clearance_m": first_value(flat, "dynamic_pcd_clearance_m", "dynamic_pcd_clearance", "closed_loop_pcd_clearance"),
            "maximum_slip_ratio": first_value(flat, "maximum_slip_ratio"),
            "maximum_tire_utilization": first_value(flat, "maximum_tire_utilization"),
            "maximum_abs_lateral_error_m": first_value(flat, "maximum_abs_lateral_error_m", "maximum_abs_lateral_error"),
            "maximum_abs_heading_error_deg": first_value(flat, "maximum_abs_heading_error_deg", "maximum_abs_heading_error"),
            "vehicle_profile_id": profile_id,
            "vehicle_profile_fingerprint": fingerprint,
            "axle_count": axle_count,
            "wheel_count": wheel_count,
            "steer_ff_rad_second_dim": steer_dim,
            "vehicle_structure": structure,
            "identity_evidence": evidence,
            "route_length_m": route_info.get("route_length_m", ""),
            "point_count": route_info.get("point_count", ""),
            "report_modified_time": iso_mtime(report_path_local),
            "route_modified_time": iso_mtime(route_path) if route_path else "",
            "comparison_status": "pending_baseline_merge",
            "local_relation": "",
            "transfer_recommended": False,
            "parse_notes": ";".join(filter(None, [resolution_note, route_info.get("parse_notes", "")])),
        }
        candidates.append(candidate)

    # Preserve unlinked NPZ files as candidate routes without labels.
    linked_routes = {row["route_path_local"] for row in candidates if row.get("route_path_local")}
    for root, route_path in discovered:
        if route_path.suffix.lower() != ".npz" or str(route_path) in linked_routes:
            continue
        info = npz_cache.setdefault(str(route_path), inspect_npz(route_path))
        if info.get("artifact_role") != "candidate_route":
            continue
        structure, evidence = structure_from_evidence(
            info.get("vehicle_profile_id"), info.get("wheel_count"),
            info.get("axle_count"), info.get("steer_ff_rad_second_dim"),
        )
        candidates.append({
            "source_machine": source_machine,
            "map": infer_map(route_path, {}),
            "report_path_local": "",
            "route_path_local": str(route_path),
            "build_family": infer_build_family(route_path, root),
            "report_file_sha256": "",
            "route_file_sha256": file_hashes.get(str(route_path), ""),
            "route_input_hash": info.get("route_input_hash", ""),
            "route_input_hash_method": info.get("route_input_hash_method", ""),
            "hard_certificate_passed": "",
            "hard_failure_reasons": "",
            "completed": "",
            "policy_id": "",
            "hard_thresholds": "",
            "dynamic_pcd_clearance_m": "",
            "maximum_slip_ratio": "",
            "maximum_tire_utilization": "",
            "maximum_abs_lateral_error_m": "",
            "maximum_abs_heading_error_deg": "",
            "vehicle_profile_id": info.get("vehicle_profile_id", ""),
            "vehicle_profile_fingerprint": info.get("vehicle_profile_fingerprint", ""),
            "axle_count": info.get("axle_count", ""),
            "wheel_count": info.get("wheel_count", ""),
            "steer_ff_rad_second_dim": info.get("steer_ff_rad_second_dim", ""),
            "vehicle_structure": structure,
            "identity_evidence": evidence,
            "route_length_m": info.get("route_length_m", ""),
            "point_count": info.get("point_count", ""),
            "report_modified_time": "",
            "route_modified_time": iso_mtime(route_path),
            "comparison_status": "pending_baseline_merge",
            "local_relation": "",
            "transfer_recommended": False,
            "parse_notes": info.get("parse_notes", ""),
        })

    local_relations(candidates)
    new_candidates = [row for row in candidates if row.get("local_relation") in {"new_route_with_label", "new_route_without_label"}]
    duplicate_conflict = [row for row in candidates if row.get("local_relation") in {"existing_exact", "existing_route_new_report", "conflict"}]

    recommended_paths: set[str] = set()
    for row in candidates:
        recommend = row.get("local_relation") in {"new_route_with_label", "new_route_without_label", "conflict"}
        row["transfer_recommended"] = recommend
        if recommend:
            for key in ("route_path_local", "report_path_local"):
                if row.get(key):
                    recommended_paths.add(str(row[key]))
    for row in inventory:
        row["transfer_recommended"] = row.get("path_local") in recommended_paths

    write_csv(analysis_dir / "local_file_inventory.csv", INVENTORY_FIELDS, inventory)
    write_csv(analysis_dir / "local_candidate_samples.csv", CANDIDATE_FIELDS, candidates)
    write_csv(analysis_dir / "new_candidate_samples.csv", CANDIDATE_FIELDS, new_candidates)
    write_csv(analysis_dir / "duplicate_or_conflict_samples.csv", CANDIDATE_FIELDS, duplicate_conflict)
    write_csv(analysis_dir / "vehicle_configuration_evidence.csv", VEHICLE_FIELDS, vehicles)
    markdown_report(roots, inventory, candidates, vehicles, scan_errors, report_path)
    print(json.dumps({
        "inventory": len(inventory), "candidates": len(candidates),
        "new_local": len(new_candidates), "duplicate_or_conflict": len(duplicate_conflict),
        "vehicle_evidence": len(vehicles), "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
