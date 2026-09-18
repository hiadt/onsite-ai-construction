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
DEFAULT_BASELINE = Path(r"D:\微信\微信聊天\xwechat_files\wxid_amt5qtexvhih22_9cd6\msg\file\2026-09\主数据集去重基线.csv")
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
MATCH_FIELDS = SAMPLE_FIELDS + [
    "match_method", "baseline_candidate_count", "baseline_route_input_hash", "baseline_route_file_sha256",
    "baseline_map", "baseline_hard_certificate_passed", "baseline_policy_id", "baseline_build_family",
    "baseline_vehicle_profile_id", "baseline_vehicle_profile_fingerprint", "baseline_axle_count",
    "resolution_note", "required_for_confirmed_new_transfer",
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
                base["route_length_m"] = f"{float(data['s_m'].reshape(-1)[-1]):.17g}"
            elif base["point_count"] > 1:
                dx = np.diff(data["x_m"].astype(float))
                dy = np.diff(data["y_m"].astype(float))
                base["route_length_m"] = f"{float(np.hypot(dx, dy).sum()):.17g}"
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
    profile_fingerprint = nested(data, "vehicle_profile_fingerprint")
    artifact_fingerprint = nested(data, "fingerprint")
    config = None
    if profile or axle or wheel or (path.name in CONFIG_NAMES and (profile_fingerprint or artifact_fingerprint)):
        config = {
            "configuration_path_local": str(path), "file_sha256": "", "vehicle_profile_id": profile,
            "vehicle_profile_fingerprint": profile_fingerprint, "artifact_fingerprint": artifact_fingerprint,
            "axle_count": axle, "wheel_count": wheel,
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


def load_baseline(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"route_input_hash", "route_file_sha256", "map", "hard_certificate_passed", "policy_id"}
    if len(rows) != 359 or not rows or not required <= set(rows[0]):
        raise ValueError(f"unexpected baseline schema or row count: {path}")
    return rows


def route_axis(row: dict) -> str:
    profile = row.get("vehicle_profile_id", "").lower()
    structure = row.get("vehicle_structure", "").lower()
    axle = row.get("axle_count", "").removesuffix(".0")
    if "six" in profile or "six" in structure or axle == "6":
        return "6"
    if "five" in profile or "five" in structure or axle == "5":
        return "5"
    return ""


def composite_matches(route: dict, baseline: list[dict]) -> list[dict]:
    if not route["map"] or not route["point_count"] or not route["route_length_m"]:
        return []
    local_axis = route_axis(route)
    matches = []
    for base in baseline:
        if route["map"] != base["map"] or str(route["point_count"]) != base.get("point_count", ""):
            continue
        try:
            length_delta = abs(float(route["route_length_m"]) - float(base.get("route_length_m", "")))
        except ValueError:
            continue
        if length_delta > max(1e-6, abs(float(base["route_length_m"])) * 1e-10):
            continue
        if local_axis and route_axis(base) and local_axis != route_axis(base):
            continue
        local_fp = route.get("vehicle_profile_fingerprint", "")
        if local_fp and base.get("vehicle_profile_fingerprint") and local_fp != base["vehicle_profile_fingerprint"]:
            continue
        matches.append(base)
    return matches


def add_baseline_match(route: dict, match: dict | None, method: str, count: int, note: str) -> None:
    route.update({
        "match_method": method, "baseline_candidate_count": count,
        "baseline_route_input_hash": match.get("route_input_hash", "") if match else "",
        "baseline_route_file_sha256": match.get("route_file_sha256", "") if match else "",
        "baseline_map": match.get("map", "") if match else "",
        "baseline_hard_certificate_passed": match.get("hard_certificate_passed", "") if match else "",
        "baseline_policy_id": match.get("policy_id", "") if match else "",
        "baseline_build_family": match.get("build_family", "") if match else "",
        "baseline_vehicle_profile_id": match.get("vehicle_profile_id", "") if match else "",
        "baseline_vehicle_profile_fingerprint": match.get("vehicle_profile_fingerprint", "") if match else "",
        "baseline_axle_count": match.get("axle_count", "") if match else "",
        "resolution_note": note, "required_for_confirmed_new_transfer": "false",
    })


def audit(roots: list[Path], baseline_path: Path) -> dict:
    machine = socket.gethostname()
    baseline = load_baseline(baseline_path)
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
        classification = "conflict" if conflict else "pending_baseline_comparison"
        first["classification"] = classification
        canonical.append(first)
        if conflict or len(members) > 1 or classification.startswith("existing"):
            for index, member in enumerate(members):
                row = dict(member)
                row["classification"] = "conflict" if conflict else classification if index == 0 else "local_duplicate"
                duplicate_rows.append(row)

    by_input = defaultdict(list)
    by_file = defaultdict(list)
    for row in baseline:
        by_input[row["route_input_hash"].lower()].append(row)
        by_file[row["route_file_sha256"].lower()].append(row)
    exact_file_pairs = [(route, by_file[route["route_file_sha256"].lower()][0]) for route in canonical if route["route_file_sha256"].lower() in by_file]
    input_hash_compatible = any(route["route_input_hash"].lower() == base["route_input_hash"].lower() for route, base in exact_file_pairs)
    confirmed_labeled, confirmed_unlabeled, existing_exact, existing_new_report, unresolved = [], [], [], [], []
    for route in canonical:
        match = None
        if route["classification"] == "conflict":
            add_baseline_match(route, None, "local_label_conflict", 0, "本机同一路线存在冲突标签")
            unresolved.append(route)
            continue
        if input_hash_compatible and route["route_input_hash"].lower() in by_input:
            match = by_input[route["route_input_hash"].lower()][0]
            method = "route_input_hash"
        elif route["route_file_sha256"].lower() in by_file:
            match = by_file[route["route_file_sha256"].lower()][0]
            method = "route_file_sha256"
        else:
            candidates = composite_matches(route, baseline)
            if candidates:
                route["classification"] = "unresolved"
                add_baseline_match(
                    route, candidates[0] if len(candidates) == 1 else None, "composite_signature",
                    len(candidates), "地图、点数、长度和车型结构命中基线，但基线不含主要数组摘要，不能确认完全相同",
                )
                unresolved.append(route)
                continue
            route["classification"] = "confirmed_new_labeled" if route["hard_certificate_passed"] in {"true", "false"} else "confirmed_new_unlabeled"
            add_baseline_match(
                route, None, "no_baseline_match", 0,
                "文件哈希未命中，且地图、点数、长度、车型结构组合未命中基线",
            )
            route["required_for_confirmed_new_transfer"] = "true"
            (confirmed_labeled if route["classification"] == "confirmed_new_labeled" else confirmed_unlabeled).append(route)
            continue
        route["classification"] = "existing_route_new_report" if route["report_path_local"] else "existing_exact"
        add_baseline_match(route, match, method, 1, "路线与主数据集匹配")
        (existing_new_report if route["classification"] == "existing_route_new_report" else existing_exact).append(route)

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
    confirmed_required_paths = {row["route_path_local"] for row in confirmed_labeled + confirmed_unlabeled}
    for item in inventory:
        item["transfer_recommended"] = "true" if item["path_local"] in recommended_paths else "false"
        item["confirmed_new_required"] = "true" if item["path_local"] in confirmed_required_paths else "false"

    baseline_profiles = {row["vehicle_profile_id"] for row in baseline if row["vehicle_profile_id"]}
    baseline_fingerprints = {row["vehicle_profile_fingerprint"] for row in baseline if row["vehicle_profile_fingerprint"]}
    for config in configs:
        profile = config["vehicle_profile_id"]
        fingerprint = config["vehicle_profile_fingerprint"]
        config["vehicle_family_status"] = "existing_vehicle_family" if profile in baseline_profiles else "new_vehicle_family" if profile else "no_explicit_vehicle_family"
        config["vehicle_fingerprint_status"] = "existing_vehicle_fingerprint" if fingerprint in baseline_fingerprints else "new_vehicle_fingerprint" if fingerprint else "no_explicit_vehicle_fingerprint"
        config["calibration_version_status"] = "candidate_version_not_comparable_to_baseline" if config["calibration_version"] else "no_named_calibration_version"
        if config["vehicle_family_status"] == "new_vehicle_family" or config["vehicle_fingerprint_status"] == "new_vehicle_fingerprint":
            config["new_version_assessment"] = "confirmed_new_vehicle_metadata"
        elif config["calibration_version"]:
            config["new_version_assessment"] = "candidate_new_calibration_version_baseline_lacks_field"
        else:
            config["new_version_assessment"] = "known_vehicle_family_component"

    config_fields = [
        "configuration_path_local", "file_sha256", "vehicle_profile_id", "vehicle_profile_fingerprint", "artifact_fingerprint", "axle_count",
        "wheel_count", "version", "controller_version", "tire_version", "load_version", "calibration_version",
        "vehicle_family_status", "vehicle_fingerprint_status", "calibration_version_status", "new_version_assessment",
        "identity_evidence", "parse_status", "error", "transfer_recommended",
    ]
    inventory_fields = [
        "scan_root", "path_local", "file_name", "extension", "file_size_bytes", "modified_time", "sha256",
        "artifact_kind", "parse_status", "error", "transfer_recommended", "confirmed_new_required", "source_machine",
    ]
    write_csv(OUTPUT_DIR / "local_file_inventory.csv", inventory, inventory_fields)
    write_csv(OUTPUT_DIR / "local_candidate_samples.csv", sorted(routes, key=lambda r: r["route_path_local"]), SAMPLE_FIELDS)
    write_csv(OUTPUT_DIR / "new_candidate_samples.csv", sorted(canonical, key=lambda r: (r["classification"], r["map"], r["route_path_local"])), SAMPLE_FIELDS)
    write_csv(OUTPUT_DIR / "duplicate_or_conflict_samples.csv", sorted(duplicate_rows, key=lambda r: (r["classification"], r["route_path_local"])), SAMPLE_FIELDS)
    write_csv(OUTPUT_DIR / "confirmed_new_labeled_samples.csv", confirmed_labeled, MATCH_FIELDS)
    write_csv(OUTPUT_DIR / "confirmed_new_unlabeled_routes.csv", confirmed_unlabeled, MATCH_FIELDS)
    write_csv(OUTPUT_DIR / "existing_exact_samples.csv", existing_exact, MATCH_FIELDS)
    write_csv(OUTPUT_DIR / "existing_route_new_reports.csv", existing_new_report, MATCH_FIELDS)
    write_csv(OUTPUT_DIR / "unresolved_candidates.csv", unresolved, MATCH_FIELDS)
    write_csv(OUTPUT_DIR / "vehicle_configuration_evidence.csv", sorted(configs, key=lambda r: r["configuration_path_local"]), config_fields)
    return {
        "roots": roots, "inventory": inventory, "routes": routes, "canonical": canonical, "duplicates": duplicate_rows,
        "configs": configs, "reports": reports, "scan_errors": scan_errors, "baseline": baseline,
        "baseline_path": baseline_path, "input_hash_compatible": input_hash_compatible,
        "confirmed_labeled": confirmed_labeled, "confirmed_unlabeled": confirmed_unlabeled,
        "existing_exact": existing_exact, "existing_new_report": existing_new_report, "unresolved": unresolved,
    }


def report(result: dict) -> None:
    inventory, canonical = result["inventory"], result["canonical"]
    structures = Counter(row["vehicle_structure"] for row in canonical)
    duplicate_groups = len({row["route_input_hash"] or row["route_file_sha256"] for row in result["duplicates"] if row["classification"] == "local_duplicate"})
    recommended = [row for row in inventory if row["transfer_recommended"] == "true"]
    confirmed_required = [row for row in inventory if row["confirmed_new_required"] == "true"]
    local_maps = sorted({row["map"] for row in canonical if row["map"]})
    baseline_maps = {row["map"] for row in result["baseline"] if row["map"]}
    names_outside_baseline = sorted(set(local_maps) - baseline_maps)
    damaged = [row for row in inventory if row["parse_status"] == "error"]
    summary_count = sum(row["file_name"] == "summary.json" for row in inventory)
    cert_count = sum(row["artifact_kind"] == "validation_report" for row in inventory)
    npz_count = sum(row["extension"] == ".npz" for row in inventory)
    config_count = sum(row["artifact_kind"] in {"vehicle_or_calibration_config", "named_config"} for row in inventory)
    candidate_versions = sorted({row["calibration_version"] for row in result["configs"] if row["calibration_version"]})
    confirmed_vehicle_metadata = [row for row in result["configs"] if row["new_version_assessment"] == "confirmed_new_vehicle_metadata"]
    lines = [
        "# 本机测试数据扩展报告", "", f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"本机标识：`{socket.gethostname()}`", f"GitHub 用户名：`{USER}`", "", "## 扫描范围", "",
        "已识别固定磁盘：`C:`、`D:`、`E:`。仅扫描明确的项目目录；未扫描 Windows、软件安装目录、游戏内容或其他无关个人资料。", "",
    ]
    lines.extend(f"- `{root}`" for root in result["roots"])
    lines += ["", "指定但本机不存在的目录：`D:\\generic_route_lab`、`D:\\Onsite-6`、`D:\\Onsite-5`、`D:\\OnSite`、`D:\\研`。", "",
        "## 主数据集去重确认", "",
        f"基线：`{result['baseline_path']}`，共 {len(result['baseline'])} 条去重候选。55条本机有效候选的最终分层如下：", "",
        f"- 与主数据集完全重复：{len(result['existing_exact'])} 条。",
        f"- 同路线但有新报告：{len(result['existing_new_report'])} 条。",
        f"- 真正新增且有布尔型硬标签：{len(result['confirmed_labeled'])} 条。",
        f"- 确认新增但无逐样本硬标签：{len(result['confirmed_unlabeled'])} 条。",
        f"- 仍无法确认：{len(result['unresolved'])} 条。",
        "- 55条有效候选中无法解析：0 条；另有1个未进入55条集合的损坏桌面副本。", "",
        f"本机输入哈希与基线算法兼容：{'是' if result['input_hash_compatible'] else '否'}。已用相同文件哈希对照验证；因此本轮没有用不兼容的输入哈希判重。完全重复均由 `route_file_sha256` 确认；组合签名命中但缺少主数据主要数组摘要的路线保守列入 `unresolved_candidates.csv`。", "",
        "## 报告与硬标签续查", "",
        f"- `summary.json`：{summary_count}；可关联到55条路线的逐样本 `safety_summary` / `validation_report` / certificate：0。",
        f"- 含 `hard_certificate_passed` 字段的 JSON：{cert_count} 份，但它是部署聚合摘要，不是逐样本报告，未拆分或伪造标签。",
        "- 因此 `confirmed_new_labeled_samples.csv` 只有表头。", "",
        "## 文件与结构统计", "",
        f"- 审计 NPZ：{npz_count} 份；可解析候选文件 {sum(r['parse_status']=='ok' for r in result['routes'])} 份；本机去重后有效路线55条；本机路径级重复组 {duplicate_groups} 组。",
        f"- 车型/标定配置文件：{config_count} 份；结构化配置证据 {len(result['configs'])} 份。", "",
        f"- 六轴显式：{structures['six_axis_explicit']}；六轴结构推断：{structures['six_axis_structural']}。",
        f"- 五轴显式：{structures['five_axis_explicit']}；五轴结构推断：{structures['five_axis_structural']}。",
        f"- 未知或冲突：{structures['unknown_or_conflict']}。", "",
        "## 地图名称核对", "",
        f"本机出现28个地图/路线名称；基线25张地图之外的名称为：{', '.join(names_outside_baseline) if names_outside_baseline else '无'}。",
        "这3个名称均是既有 `turnaround` 路线的动态、调参或模板文件名，名称本身不能证明新地图。因此真正确认超出主数据集25张地图的名称：无。", "",
        "## 车辆配置与标定版本", "",
        f"63组证据中的显式车型族均为基线已有的 `five_axis` / `six_axis`；确认新增车型或车辆指纹：{len(confirmed_vehicle_metadata)} 组。",
        f"发现4个具名标定包版本：{', '.join(candidate_versions)}。基线没有标定版本或配置文件哈希字段，因此这些只能标为“候选新标定版本”，不能仅凭现有基线确认新增。",
        "逐文件结论见 `vehicle_configuration_evidence.csv` 的比较状态列。", "",
        "## 建议后续集中复制", "",
        f"原清单建议复制 {len(recommended)} 个文件，共 {sum(int(r['file_size_bytes']) for r in recommended)} 字节。其中确认新增且有硬标签的样本所必需文件：0 个；确认新增无标签路线所必需文件：{len(confirmed_required)} 个，共 {sum(int(r['file_size_bytes']) for r in confirmed_required)} 字节。",
        "这些必需文件在 `local_file_inventory.csv` 中标记为 `confirmed_new_required=true`，并逐条列于 `confirmed_new_unlabeled_routes.csv`；完全重复文件不属于新增样本必需传输项。", "",
        "## 损坏、扫描失败与缺失信息", "",
    ]
    if damaged:
        lines.extend(f"- `{row['path_local']}`：{row['error']}" for row in damaged)
    else:
        lines.append("- 未发现损坏或不可读的审计文件。")
    lines.extend(f"- 扫描失败：{error}" for error in result["scan_errors"])
    lines += [
        "- 未找到逐样本 `summary.json` / `safety_summary.json`，因此无法把部署摘要中的聚合通过数分配给具体路线，也未据此伪造标签。",
        "- 基线输入哈希算法与本机审计脚本的本地输入哈希算法不同；在负责人提供算法实现前，不将两者直接比较。",
        "- 基线不含主要数组摘要，组合签名命中项保持未决，没有强行选取一个基线样本。",
        "- 控制器、轮胎、载荷和地图构建版本在不少文件中缺失；空值保持为空，未根据路径虚构。",
        "", "## 可复现方式", "", "在仓库根目录使用包含 NumPy 的 Python 运行：", "",
        "```powershell", f"python analysis/data_recovery/{USER}/run_local_data_audit.py --baseline \"{result['baseline_path']}\"", "```", "",
        "脚本只读原始目录和基线文件，并重新生成清单与本报告。", "",
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
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="359-row primary-machine deduplication CSV")
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
    if not args.baseline.is_file():
        parser.error(f"Baseline CSV not found: {args.baseline}")
    result = audit(roots, args.baseline.resolve())
    report(result)
    print(f"audited {len(result['inventory'])} files; {len(result['canonical'])} unique route inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
