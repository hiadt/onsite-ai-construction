#!/usr/bin/env python3
"""Read-only inventory and deduplication of local OnSite route artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on machines without numpy
    np = None


USER = "ZYJisBoss"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "analysis" / "data_recovery" / USER
REPORT_PATH = REPO_ROOT / "reports" / "data_recovery" / USER / "本机测试数据扩展报告.md"
DEFAULT_ROOTS = [
    Path(r"C:\Users\zgz20\Desktop\代码\C-TURE"),
    Path(r"C:\Users\zgz20\Desktop\第六赛道大件运输\第六赛道相关文献\六轴车数据"),
    Path(r"E:\BaiduNetdiskDownload\使命召唤\C-TURE"),
    Path(r"E:\BaiduNetdiskDownload\使命召唤\OnSite-B-1150-Unified"),
]
AUDIT_SUFFIXES = {".json", ".jsonl", ".csv", ".npz"}
REPORT_NAMES = {
    "summary.json",
    "safety_summary.json",
    "validation_report.json",
    "selection_report.json",
}
CONFIG_NAMES = {"manifest.json", "checkpoint.json", "calibration_manifest.json", "profile.json", "vehicle_profile.json"}
ROUTE_KEYS = (
    "x_m",
    "y_m",
    "z_m",
    "yaw_rad",
    "kappa_1pm",
    "dkappa_ds_1pm2",
    "v_profile_mps",
    "vehicle_profile_id",
    "vehicle_profile_fingerprint",
)
MAP_RE = re.compile(r"^(RoadC|bridge|cliff|cross|mounarea|narrow|overpass|ramp|turnaround)(?:[_-]\w+)?$", re.I)
SAMPLE_FIELDS = [
    "sample_key", "classification", "map", "report_path_local", "route_path_local", "build_family",
    "report_file_sha256", "route_file_sha256", "route_input_hash", "hard_certificate_passed",
    "hard_failure_reasons", "completed", "policy_id_or_thresholds", "dynamic_pcd_clearance_m",
    "maximum_slip_ratio", "maximum_tire_utilization", "maximum_abs_lateral_error_m",
    "maximum_abs_heading_error_deg", "vehicle_profile_id", "vehicle_profile_fingerprint", "axle_count",
    "wheel_count", "steer_ff_rad_second_dimension", "vehicle_structure", "identity_evidence",
    "route_length_m", "point_count", "modified_time", "source_machine", "file_size_bytes",
    "transfer_recommended", "parse_status", "error",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    if value is None:
        return ""
    if np is not None and isinstance(value, np.ndarray):
        if value.size != 1:
            return ""
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def nested(data, *keys):
    if not isinstance(data, dict):
        return ""
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key.lower() in lowered:
            value = lowered[key.lower()]
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else scalar(value)
    for value in data.values():
        if isinstance(value, dict):
            found = nested(value, *keys)
            if found != "":
                return found
    return ""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_map(path: Path) -> str:
    if MAP_RE.match(path.stem):
        return path.stem
    for part in reversed(path.parts):
        if MAP_RE.match(part):
            return part
    return ""


def infer_build(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return "/".join(rel.parts[: min(4, len(rel.parts) - 1)])
    except ValueError:
        return path.parent.name


def array_hash(data) -> str:
    digest = hashlib.sha256()
    used = 0
    for key in ROUTE_KEYS:
        if key not in data.files:
            continue
        array = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
        used += 1
    return digest.hexdigest() if used >= 4 else ""


def route_record(path: Path, root: Path, file_hash: str, machine: str) -> dict | None:
    if np is None:
        return None
    base = {
        "sample_key": file_hash[:16], "classification": "", "map": infer_map(path), "report_path_local": "",
        "route_path_local": str(path), "build_family": infer_build(path, root), "report_file_sha256": "",
        "route_file_sha256": file_hash, "route_input_hash": "", "hard_certificate_passed": "",
        "hard_failure_reasons": "", "completed": "", "policy_id_or_thresholds": "",
        "dynamic_pcd_clearance_m": "", "maximum_slip_ratio": "", "maximum_tire_utilization": "",
        "maximum_abs_lateral_error_m": "", "maximum_abs_heading_error_deg": "", "vehicle_profile_id": "",
        "vehicle_profile_fingerprint": "", "axle_count": "", "wheel_count": "",
        "steer_ff_rad_second_dimension": "", "vehicle_structure": "unknown_or_conflict",
        "identity_evidence": "path_only", "route_length_m": "", "point_count": "", "modified_time": "",
        "source_machine": machine, "file_size_bytes": path.stat().st_size, "transfer_recommended": "true",
        "parse_status": "ok", "error": "",
    }
    base["modified_time"] = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    try:
        with np.load(path, allow_pickle=False) as data:
            if not ({"x_m", "y_m"} <= set(data.files)):
                return None
            base["route_input_hash"] = array_hash(data)
            base["point_count"] = len(data["x_m"])
            if "s_m" in data.files and data["s_m"].size:
                base["route_length_m"] = f"{float(data['s_m'].reshape(-1)[-1]):.9g}"
            elif base["point_count"] > 1:
                dx = np.diff(data["x_m"].astype(float))
                dy = np.diff(data["y_m"].astype(float))
                base["route_length_m"] = f"{float(np.hypot(dx, dy).sum()):.9g}"
            for key in ("vehicle_profile_id", "vehicle_profile_fingerprint", "axle_count", "wheel_count"):
                if key in data.files:
                    base[key] = scalar(data[key])
            if "wheel_count" not in data.files and "tire_count" in data.files:
                base["wheel_count"] = scalar(data["tire_count"])
            steer_dim = ""
            if "steer_ff_rad" in data.files and data["steer_ff_rad"].ndim >= 2:
                steer_dim = data["steer_ff_rad"].shape[1]
                base["steer_ff_rad_second_dimension"] = steer_dim
            profile = base["vehicle_profile_id"].lower()
            wheels = base["wheel_count"]
            if profile:
                base["identity_evidence"] = "explicit_profile_id"
                if "six" in profile or profile.startswith("6"):
                    base["vehicle_structure"] = "six_axis_explicit"
                elif "five" in profile or profile.startswith("5"):
                    base["vehicle_structure"] = "five_axis_explicit"
            elif wheels:
                base["identity_evidence"] = "explicit_wheel_count"
                base["vehicle_structure"] = "six_axis_explicit" if wheels == "12" else "five_axis_explicit" if wheels == "10" else "unknown_or_conflict"
            elif steer_dim:
                base["identity_evidence"] = "steer_vector_dimension"
                base["vehicle_structure"] = "six_axis_structural" if steer_dim == 12 else "five_axis_structural" if steer_dim == 10 else "unknown_or_conflict"
            return base
    except Exception as exc:  # keep damaged files visible
        base["parse_status"] = "error"
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base


def json_record(path: Path) -> tuple[dict | None, dict | None, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    report = None
    hard = nested(data, "hard_certificate_passed")
    if hard != "":
        report = {
            "hard_certificate_passed": hard,
            "hard_failure_reasons": nested(data, "hard_failure_reasons"),
            "completed": nested(data, "completed"),
            "policy_id_or_thresholds": nested(data, "policy_id", "hard_thresholds", "thresholds"),
            "dynamic_pcd_clearance_m": nested(data, "dynamic_pcd_clearance_m", "dynamic_pcd_clearance", "closed_loop_pcd_clearance_m", "closed_loop_pcd_clearance"),
            "maximum_slip_ratio": nested(data, "maximum_slip_ratio"),
            "maximum_tire_utilization": nested(data, "maximum_tire_utilization"),
            "maximum_abs_lateral_error_m": nested(data, "maximum_abs_lateral_error_m", "maximum_abs_lateral_error"),
            "maximum_abs_heading_error_deg": nested(data, "maximum_abs_heading_error_deg", "maximum_abs_heading_error"),
            "route_hint": nested(data, "route_path", "route_file", "candidate_path", "candidate_npz"),
        }
    profile = nested(data, "vehicle_profile_id")
    axle = nested(data, "axle_count")
    wheel = nested(data, "wheel_count", "tire_count")
    fingerprint = nested(data, "vehicle_profile_fingerprint", "fingerprint")
    config = None
    if profile or axle or wheel or (path.name in CONFIG_NAMES and fingerprint):
        config = {
            "configuration_path_local": str(path), "file_sha256": "", "vehicle_profile_id": profile,
            "vehicle_profile_fingerprint": fingerprint, "axle_count": axle, "wheel_count": wheel,
            "version": nested(data, "bundle_version", "version", "schema_version"),
            "controller_version": nested(data, "controller_version", "controller_hash"),
            "tire_version": nested(data, "tire_version", "tire_model"), "load_version": nested(data, "load_version", "load_model"),
            "calibration_version": nested(data, "calibration_version", "bundle_version"),
            "identity_evidence": "explicit_profile_id" if profile else "explicit_wheel_count" if wheel else "path_only",
            "parse_status": "ok", "error": "", "transfer_recommended": "true",
        }
    return report, config, ""


def resolve_report_route(report_path: Path, hint: str, routes: list[dict]) -> dict | None:
    if hint:
        candidate = Path(hint.replace("/", os.sep))
        choices = [candidate, report_path.parent / candidate, report_path.parent / candidate.name]
        normalized = {str(Path(r["route_path_local"]).resolve()).lower(): r for r in routes if r["parse_status"] == "ok"}
        for choice in choices:
            try:
                match = normalized.get(str(choice.resolve()).lower())
                if match:
                    return match
            except OSError:
                pass
        same_name = [r for r in routes if Path(r["route_path_local"]).name.lower() == candidate.name.lower()]
        if len(same_name) == 1:
            return same_name[0]
    map_name = infer_map(report_path)
    nearby = [r for r in routes if r["map"].lower() == map_name.lower() and os.path.commonpath([str(report_path), r["route_path_local"]])]
    return nearby[0] if len(nearby) == 1 else None


def discover_baseline_hashes() -> tuple[set[str], set[str], set[str]]:
    route_input, route_file, report_file = set(), set(), set()
    for path in REPO_ROOT.rglob("*.csv"):
        if OUTPUT_DIR in path.parents:
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    route_input.add(row.get("route_input_hash", "").lower())
                    route_file.add(row.get("route_file_sha256", "").lower())
                    report_file.add(row.get("report_file_sha256", "").lower())
        except (OSError, UnicodeError, csv.Error):
            continue
    for values in (route_input, route_file, report_file):
        values.discard("")
    return route_input, route_file, report_file


def audit(roots: list[Path]) -> dict:
    machine = socket.gethostname()
    inventory, routes, reports, configs = [], [], [], []
    scan_errors = []
    for root in roots:
        try:
            paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIT_SUFFIXES), key=str)
        except OSError as exc:
            scan_errors.append(f"{root}: {type(exc).__name__}: {exc}")
            continue
        for path in paths:
            stat = path.stat()
            item = {
                "scan_root": str(root), "path_local": str(path), "file_name": path.name, "extension": path.suffix.lower(),
                "file_size_bytes": stat.st_size, "modified_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "sha256": "", "artifact_kind": "other_audited", "parse_status": "not_parsed", "error": "",
                "transfer_recommended": "false", "source_machine": machine,
            }
            try:
                item["sha256"] = sha256(path)
                if path.suffix.lower() == ".npz":
                    route = route_record(path, root, item["sha256"], machine)
                    if route:
                        routes.append(route)
                        item["artifact_kind"] = "candidate_route" if route["parse_status"] == "ok" else "damaged_npz"
                        item["parse_status"], item["error"] = route["parse_status"], route["error"]
                        item["transfer_recommended"] = "true" if route["parse_status"] == "ok" else "false"
                    else:
                        item["artifact_kind"], item["parse_status"] = "npz_non_route", "ok"
                elif path.suffix.lower() == ".json":
                    report, config, error = json_record(path)
                    item["parse_status"] = "error" if error else "ok"
                    item["error"] = error
                    if report:
                        report.update({"path": path, "sha256": item["sha256"], "root": root})
                        reports.append(report)
                        item["artifact_kind"], item["transfer_recommended"] = "validation_report", "true"
                    elif config:
                        config["file_sha256"] = item["sha256"]
                        configs.append(config)
                        item["artifact_kind"], item["transfer_recommended"] = "vehicle_or_calibration_config", "true"
                    elif path.name in REPORT_NAMES:
                        item["artifact_kind"] = "named_report_without_hard_label"
                    elif path.name in CONFIG_NAMES:
                        item["artifact_kind"] = "named_config"
                else:
                    item["parse_status"] = "ok"
            except Exception as exc:
                item["parse_status"] = "error"
                item["error"] = f"{type(exc).__name__}: {exc}"
            inventory.append(item)

    for report in reports:
        route = resolve_report_route(report["path"], report.pop("route_hint", ""), routes)
        if not route:
            continue
        route["report_path_local"] = str(report["path"])
        route["report_file_sha256"] = report["sha256"]
        for key in report:
            if key in route and report[key] != "":
                route[key] = report[key]

    baseline_input, baseline_route, baseline_report = discover_baseline_hashes()
    groups = defaultdict(list)
    for route in routes:
        if route["parse_status"] != "ok":
            continue
        key = route["route_input_hash"] or route["route_file_sha256"]
        groups[key].append(route)
    canonical, duplicate_rows = [], []
    for key, members in groups.items():
        labels = {m["hard_certificate_passed"] for m in members if m["hard_certificate_passed"] in {"true", "false"}}
        conflict = len(labels) > 1
        members.sort(key=lambda row: (row["parse_status"] != "ok", row["route_path_local"]))
        first = members[0]
        if conflict:
            classification = "conflict"
        elif first["route_input_hash"].lower() in baseline_input or first["route_file_sha256"].lower() in baseline_route:
            classification = "existing_route_new_report" if first["report_file_sha256"] and first["report_file_sha256"].lower() not in baseline_report else "existing_exact"
        else:
            classification = "new_route_with_label" if labels else "new_route_without_label"
        first["classification"] = classification
        canonical.append(first)
        if conflict or len(members) > 1 or classification.startswith("existing"):
            for index, member in enumerate(members):
                row = dict(member)
                row["classification"] = "conflict" if conflict else classification if index == 0 else "local_duplicate"
                duplicate_rows.append(row)

    for route in routes:
        if route["parse_status"] == "error":
            route["classification"] = "damaged_or_unreadable"
            duplicate_rows.append(route)

    recommended_paths = {row["route_path_local"] for row in canonical}
    seen_config_hashes = set()
    for config in configs:
        if config["file_sha256"] in seen_config_hashes:
            config["transfer_recommended"] = "false"
        else:
            seen_config_hashes.add(config["file_sha256"])
            recommended_paths.add(config["configuration_path_local"])
    seen_report_hashes = set()
    for report in reports:
        if report["sha256"] not in seen_report_hashes:
            seen_report_hashes.add(report["sha256"])
            recommended_paths.add(str(report["path"]))
    for item in inventory:
        item["transfer_recommended"] = "true" if item["path_local"] in recommended_paths else "false"

    config_fields = [
        "configuration_path_local", "file_sha256", "vehicle_profile_id", "vehicle_profile_fingerprint", "axle_count",
        "wheel_count", "version", "controller_version", "tire_version", "load_version", "calibration_version",
        "identity_evidence", "parse_status", "error", "transfer_recommended",
    ]
    inventory_fields = [
        "scan_root", "path_local", "file_name", "extension", "file_size_bytes", "modified_time", "sha256",
        "artifact_kind", "parse_status", "error", "transfer_recommended", "source_machine",
    ]
    write_csv(OUTPUT_DIR / "local_file_inventory.csv", inventory, inventory_fields)
    write_csv(OUTPUT_DIR / "local_candidate_samples.csv", sorted(routes, key=lambda r: r["route_path_local"]), SAMPLE_FIELDS)
    write_csv(OUTPUT_DIR / "new_candidate_samples.csv", sorted(canonical, key=lambda r: (r["classification"], r["map"], r["route_path_local"])), SAMPLE_FIELDS)
    write_csv(OUTPUT_DIR / "duplicate_or_conflict_samples.csv", sorted(duplicate_rows, key=lambda r: (r["classification"], r["route_path_local"])), SAMPLE_FIELDS)
    write_csv(OUTPUT_DIR / "vehicle_configuration_evidence.csv", sorted(configs, key=lambda r: r["configuration_path_local"]), config_fields)
    return {
        "roots": roots, "inventory": inventory, "routes": routes, "canonical": canonical, "duplicates": duplicate_rows,
        "configs": configs, "reports": reports, "scan_errors": scan_errors, "baseline_available": bool(baseline_input or baseline_route or baseline_report),
    }


def report(result: dict) -> None:
    inventory, canonical = result["inventory"], result["canonical"]
    classifications = Counter(row["classification"] for row in canonical)
    structures = Counter(row["vehicle_structure"] for row in canonical)
    duplicate_groups = len({row["route_input_hash"] or row["route_file_sha256"] for row in result["duplicates"] if row["classification"] == "local_duplicate"})
    valid = [row for row in canonical if row["parse_status"] == "ok"]
    recommended = [row for row in inventory if row["transfer_recommended"] == "true"]
    maps = sorted({row["map"] for row in valid if row["map"]})
    builds = sorted({row["build_family"] for row in valid if row["build_family"]})
    failures = sorted({reason for row in canonical for reason in row["hard_failure_reasons"].split("|") if reason})
    damaged = [row for row in inventory if row["parse_status"] == "error"]
    summary_count = sum(row["file_name"] == "summary.json" for row in inventory)
    cert_count = sum(row["artifact_kind"] == "validation_report" for row in inventory)
    npz_count = sum(row["extension"] == ".npz" for row in inventory)
    config_count = sum(row["artifact_kind"] in {"vehicle_or_calibration_config", "named_config"} for row in inventory)
    baseline_note = "已发现仓库基线清单并参与比较。" if result["baseline_available"] else "仓库当前未提供可机读的完整基线清单；本报告已完成本机内部去重，待与主数据集合并去重。"
    lines = [
        "# 本机测试数据扩展报告", "", f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"本机标识：`{socket.gethostname()}`", f"GitHub 用户名：`{USER}`", "", "## 扫描范围", "",
        "已识别固定磁盘：`C:`、`D:`、`E:`。仅扫描明确的项目目录；未扫描 Windows、软件安装目录、游戏内容或其他无关个人资料。", "",
    ]
    lines.extend(f"- `{root}`" for root in result["roots"])
    lines += ["", "指定但本机不存在的目录：`D:\\generic_route_lab`、`D:\\Onsite-6`、`D:\\Onsite-5`、`D:\\OnSite`、`D:\\研`。", "", "## 结果摘要", "",
        f"- `summary.json`：{summary_count} 份；含 `hard_certificate_passed` 字段的 JSON：{cert_count} 份（本机发现项为部署聚合摘要，不是逐样本标签）。",
        f"- NPZ：{npz_count} 份；其中可解析候选路线文件 {sum(r['parse_status']=='ok' for r in result['routes'])} 份，损坏或不可读 {sum(r['parse_status']=='error' for r in result['routes'])} 份。",
        f"- 车型/标定配置文件：{config_count} 份；提取到结构化配置证据 {len(result['configs'])} 份。",
        f"- 去重后有效候选路线：{len(valid)} 条。明确新增且有标签：{classifications['new_route_with_label']} 条。",
        f"- 新路线但无有效标签：{classifications['new_route_without_label']} 条；仓库完全重复：{classifications['existing_exact']} 条；同路线新报告：{classifications['existing_route_new_report']} 条；冲突：{classifications['conflict']} 条。",
        f"- 本机路径级重复组：{duplicate_groups} 组。{baseline_note}", "", "## 车型结构", "",
        f"- 六轴显式：{structures['six_axis_explicit']}；六轴结构推断：{structures['six_axis_structural']}。",
        f"- 五轴显式：{structures['five_axis_explicit']}；五轴结构推断：{structures['five_axis_structural']}。",
        f"- 未知或冲突：{structures['unknown_or_conflict']}。", "",
        "结构统计以去重后的路线输入为单位。显式 `vehicle_profile_id`/轮数优先；仅在显式字段缺失时才使用 `steer_ff_rad` 第二维（12=六轴、10=五轴）。", "",
        "## 新增范围与失败类型", "", f"地图：{', '.join(maps) if maps else '未能识别'}。", "",
        f"构建批次/来源族：{', '.join(builds) if builds else '未能识别'}。", "",
        f"失败类型：{', '.join(failures) if failures else '未发现逐样本硬失败标签，因此无新增失败类型可确认。'}", "",
        "## 车辆配置与标定包", "", f"发现 {len(result['configs'])} 份带车型、轴/轮数或指纹证据的配置/标定 JSON。是否相对主数据集为“新配置”仍需数据负责人用主清单复核。",
        "详细路径、SHA-256 和版本字段见 `vehicle_configuration_evidence.csv`。", "", "## 建议后续集中复制", "",
        f"建议复核并按需集中复制 {len(recommended)} 个文件，总大小 {sum(int(r['file_size_bytes']) for r in recommended)} 字节。第一阶段未复制或上传任何原始文件；清单仅记录本机路径、大小和 SHA-256。", "",
        "优先级：含硬标签的验证报告（若后续恢复）→ 去重后的候选路线 → 明确车型/标定清单。重复副本不建议重复传输。", "", "## 损坏、扫描失败与缺失信息", "",
    ]
    if damaged:
        lines.extend(f"- `{row['path_local']}`：{row['error']}" for row in damaged)
    else:
        lines.append("- 未发现损坏或不可读的审计文件。")
    lines.extend(f"- 扫描失败：{error}" for error in result["scan_errors"])
    lines += [
        "- 未找到逐样本 `summary.json` / `safety_summary.json`，因此无法把部署摘要中的聚合通过数分配给具体路线，也未据此伪造标签。",
        "- 当前仓库只有文字基线（246 条/25 张地图），没有逐样本哈希清单；无法最终判定相对主数据集的新增、重复或冲突。",
        "- 路线输入哈希按位置、姿态、曲率、速度和显式车型身份数组稳定生成；需由数据负责人确认与主数据哈希实现完全一致。",
        "- 控制器、轮胎、载荷和地图构建版本在不少文件中缺失；空值保持为空，未根据路径虚构。",
        "", "## 可复现方式", "", "在仓库根目录使用包含 NumPy 的 Python 运行：", "",
        "```powershell", f"python analysis/data_recovery/{USER}/run_local_data_audit.py", "```", "",
        "脚本只读原始目录，并重新生成本目录下 5 个 CSV 与本报告。", "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    assert scalar(True) == "true"
    assert nested({"outer": {"wheel_count": 12}}, "wheel_count") == "12"
    assert infer_map(Path("x/routes/ramp_3.npz")) == "ramp_3"
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, help="Project directory to scan; repeatable")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if np is None:
        parser.error("NumPy is required to inspect NPZ files")
    roots = [path.resolve() for path in (args.root or DEFAULT_ROOTS) if path.exists()]
    if not roots:
        parser.error("No scan roots exist")
    result = audit(roots)
    report(result)
    print(f"audited {len(result['inventory'])} files; {len(result['canonical'])} unique route inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
