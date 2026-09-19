#!/usr/bin/env python3
"""Freeze a minimal, evidence-complete five/six-axis raw NPZ sample set.

Selection is deterministic and read-only with respect to the source artifacts.
Raw reports are not copied because they may contain local absolute paths; their
verified SHA-256 values are retained in the manifest instead.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MEMBER = "SHANYU000"
AUDIT_DIR = ROOT / "analysis" / "data_recovery" / MEMBER
BASELINE = ROOT / "data" / "manifests" / "baseline_v1_359.csv"
FREEZE_DIR = ROOT / "data" / "frozen" / "raw_samples" / MEMBER
ROUTE_DIR = FREEZE_DIR / "routes"
REPORT_DIR = ROOT / "reports" / "data_freeze" / MEMBER

VALID_STRUCTURES = {
    "five_axis_explicit": "five_axis",
    "five_axis_structural": "five_axis",
    "six_axis_explicit": "six_axis",
    "six_axis_structural": "six_axis",
}

MANIFEST_FIELDS = [
    "sample_id",
    "stored_route_path",
    "route_file_sha256",
    "route_size_bytes",
    "vehicle_structure",
    "vehicle_family",
    "identity_evidence",
    "vehicle_profile_id",
    "vehicle_profile_fingerprint",
    "vehicle_evidence_match_count",
    "map",
    "label_status",
    "hard_certificate_passed",
    "hard_failure_reasons",
    "comparison_status",
    "comparison_method",
    "build_family",
    "report_file_sha256",
    "report_evidence_status",
    "point_count",
    "route_length_m",
    "policy_id",
    "selection_coverage",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_reasons(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if isinstance(value, list):
        return sorted({str(item) for item in value if str(item)})
    return [str(value)]


def is_bool_label(value: str) -> bool:
    return value.strip().lower() in {"true", "false", "0", "1"}


def label_value(value: str) -> str:
    return "1" if value.strip().lower() in {"true", "1"} else "0"


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-._") or "unknown"


def validate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError(f"empty NPZ: {path}")
        for key in archive.files:
            _ = archive[key].shape


def comparison_status(route_hash: str, labeled: bool, baseline_hashes: set[str]) -> str:
    if route_hash in baseline_hashes:
        return "baseline_existing"
    return "confirmed_new_labeled" if labeled else "confirmed_new_unlabeled"


def candidate_tokens(row: dict[str, Any]) -> set[str]:
    tokens = {
        f"map:{row['map']}",
        f"structure:{row['vehicle_structure']}",
        f"status:{row['comparison_status']}",
    }
    if row["hard_certificate_passed"] == "1":
        tokens.add("outcome:passed")
    else:
        reasons = row["failure_reasons"] or ["failed_without_reason"]
        tokens.update(f"failure:{reason}" for reason in reasons)
    return tokens


def load_candidates() -> tuple[list[dict[str, Any]], list[dict[str, str]], set[str]]:
    # All three requested sources are read. local_candidate_samples supplies the
    # full context; new_candidate_samples supplies route-level representatives;
    # vehicle_configuration_evidence is used to count corroborating identities.
    local_rows = read_csv(AUDIT_DIR / "local_candidate_samples.csv")
    new_rows = read_csv(AUDIT_DIR / "new_candidate_samples.csv")
    vehicle_rows = read_csv(AUDIT_DIR / "vehicle_configuration_evidence.csv")
    if not local_rows:
        raise RuntimeError("local_candidate_samples.csv is empty")

    baseline_hashes = {
        row.get("route_file_sha256", "")
        for row in read_csv(BASELINE)
        if row.get("route_file_sha256")
    }
    evidence_counts: Counter[tuple[str, str]] = Counter()
    for row in vehicle_rows:
        evidence_counts[(
            row.get("vehicle_profile_id", ""),
            row.get("vehicle_profile_fingerprint", ""),
        )] += 1

    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in new_rows:
        structure = row.get("vehicle_structure", "")
        if structure not in VALID_STRUCTURES:
            continue
        route_hash = row.get("route_file_sha256", "").lower()
        report_hash = row.get("report_file_sha256", "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", route_hash):
            continue
        if route_hash in seen_hashes:
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
            continue
        if not row.get("map") or not is_bool_label(row.get("hard_certificate_passed", "")):
            continue
        if row.get("local_relation") != "new_route_with_label":
            continue
        route_path = Path(row.get("route_path_local", ""))
        report_path = Path(row.get("report_path_local", ""))
        if not route_path.is_file() or not report_path.is_file():
            continue
        reasons = parse_reasons(row.get("hard_failure_reasons", ""))
        status = comparison_status(route_hash, True, baseline_hashes)
        profile_key = (
            row.get("vehicle_profile_id", ""),
            row.get("vehicle_profile_fingerprint", ""),
        )
        item: dict[str, Any] = dict(row)
        item.update({
            "route_path": route_path,
            "report_path": report_path,
            "route_file_sha256": route_hash,
            "report_file_sha256": report_hash,
            "vehicle_family": VALID_STRUCTURES[structure],
            "hard_certificate_passed": label_value(row["hard_certificate_passed"]),
            "failure_reasons": reasons,
            "comparison_status": status,
            "vehicle_evidence_match_count": evidence_counts[profile_key] if any(profile_key) else 0,
            "route_size_bytes": route_path.stat().st_size,
        })
        item["tokens"] = candidate_tokens(item)
        candidates.append(item)
        seen_hashes.add(route_hash)
    return candidates, vehicle_rows, baseline_hashes


def select_representatives(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    known_maps = {row["map"] for row in candidates}
    structures = {row["vehicle_structure"] for row in candidates}
    statuses = {row["comparison_status"] for row in candidates}
    failures = {
        reason
        for row in candidates
        for reason in row["failure_reasons"]
    }
    required = (
        {f"map:{name}" for name in known_maps}
        | {f"structure:{name}" for name in structures}
        | {f"status:{name}" for name in statuses}
        | {f"failure:{name}" for name in failures}
        | {"outcome:passed"}
    )
    uncovered = set(required)
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while uncovered:
        useful = [row for row in remaining if row["tokens"] & uncovered]
        if not useful:
            break
        chosen = max(
            useful,
            key=lambda row: (
                len(row["tokens"] & uncovered),
                row.get("completed", "").lower() == "true",
                row["comparison_status"] == "baseline_existing",
                row["vehicle_evidence_match_count"],
                -row["route_size_bytes"],
                row["route_file_sha256"],
            ),
        )
        selected.append(chosen)
        uncovered -= chosen["tokens"]
        remaining.remove(chosen)
    selected.sort(key=lambda row: (row["map"].lower(), row["vehicle_structure"], row["route_file_sha256"]))
    return selected, uncovered


def freeze(selected: list[dict[str, Any]], uncovered: set[str]) -> list[dict[str, Any]]:
    if uncovered:
        raise RuntimeError(f"coverage requirements not met: {sorted(uncovered)}")
    ROUTE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    sums: list[str] = []
    selected_names: set[str] = set()
    for row in selected:
        route_path: Path = row["route_path"]
        report_path: Path = row["report_path"]
        validate_npz(route_path)
        actual_route_hash = sha256_file(route_path)
        actual_report_hash = sha256_file(report_path)
        if actual_route_hash != row["route_file_sha256"]:
            raise RuntimeError(f"route hash changed: {route_path}")
        if actual_report_hash != row["report_file_sha256"]:
            raise RuntimeError(f"report hash changed: {report_path}")

        sample_id = f"PG-SHANYU000-{actual_route_hash[:16]}"
        file_name = f"{sample_id}__{sanitize(row['map'])}__{row['vehicle_family']}.npz"
        selected_names.add(file_name)
        destination = ROUTE_DIR / file_name
        if destination.exists() and sha256_file(destination) != actual_route_hash:
            raise RuntimeError(f"refusing to overwrite mismatched destination: {destination}")
        if not destination.exists():
            shutil.copy2(route_path, destination)
        stored_relative = destination.relative_to(ROOT).as_posix()
        sums.append(f"{actual_route_hash}  {stored_relative}")
        manifest_rows.append({
            "sample_id": sample_id,
            "stored_route_path": stored_relative,
            "route_file_sha256": actual_route_hash,
            "route_size_bytes": destination.stat().st_size,
            "vehicle_structure": row["vehicle_structure"],
            "vehicle_family": row["vehicle_family"],
            "identity_evidence": row.get("identity_evidence", ""),
            "vehicle_profile_id": row.get("vehicle_profile_id", ""),
            "vehicle_profile_fingerprint": row.get("vehicle_profile_fingerprint", ""),
            "vehicle_evidence_match_count": row["vehicle_evidence_match_count"],
            "map": row["map"],
            "label_status": "labeled",
            "hard_certificate_passed": row["hard_certificate_passed"],
            "hard_failure_reasons": json.dumps(row["failure_reasons"], ensure_ascii=False),
            "comparison_status": row["comparison_status"],
            "comparison_method": "exact_route_file_sha256_vs_baseline_v1_359",
            "build_family": row.get("build_family", ""),
            "report_file_sha256": actual_report_hash,
            "report_evidence_status": "verified_hash_only_not_uploaded",
            "point_count": row.get("point_count", ""),
            "route_length_m": row.get("route_length_m", ""),
            "policy_id": row.get("policy_id", ""),
            "selection_coverage": ";".join(sorted(row["tokens"])),
        })

    stale = [path.name for path in ROUTE_DIR.glob("*.npz") if path.name not in selected_names]
    if stale:
        raise RuntimeError(f"stale NPZ files require manual review; none were deleted: {stale}")
    write_csv(FREEZE_DIR / "manifest.csv", manifest_rows)
    (FREEZE_DIR / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return manifest_rows


def write_docs(rows: list[dict[str, Any]], candidate_count: int) -> None:
    maps = sorted({str(row["map"]) for row in rows}, key=str.lower)
    structures = Counter(str(row["vehicle_structure"]) for row in rows)
    statuses = Counter(str(row["comparison_status"]) for row in rows)
    labels = Counter(str(row["hard_certificate_passed"]) for row in rows)
    failures = sorted({
        reason
        for row in rows
        for reason in json.loads(str(row["hard_failure_reasons"]))
    })
    total_bytes = sum(int(row["route_size_bytes"]) for row in rows)
    readme = [
        "# SHANYU000 五轴六轴代表原始样本",
        "",
        "本目录仅包含经过哈希校验、可由 NumPy 正常读取且具有逐样本硬标签和报告哈希的代表路线 NPZ。",
        "",
        "- 原始报告未复制：历史报告可能包含本机绝对路径，清单仅保留经重新计算验证的 `report_file_sha256`。",
        "- `comparison_status` 仅按 `route_file_sha256` 与 `baseline_v1_359.csv` 精确比较，不等价于历史 `route_input_hash` 去重。",
        "- `hard_certificate_passed=1` 表示历史报告通过；不构成实车验证、安全认证或闭环仿真的替代。",
        "- `manifest.csv` 是文件、标签、车型结构、地图和报告证据的唯一对应清单。",
        "- `SHA256SUMS.txt` 用于校验仓库中的冻结 NPZ。",
    ]
    FREEZE_DIR.joinpath("README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    report = [
        "# SHANYU000 五轴六轴代表原始样本冻结报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "## 一、筛选口径",
        "",
        "从本人已合并的 `local_candidate_samples.csv`、`new_candidate_samples.csv` 和 `vehicle_configuration_evidence.csv` 读取记录。仅保留五轴/六轴结构明确、路线可读取、路线 SHA-256 与清单一致、逐样本硬标签明确、报告存在且报告 SHA-256 一致、并且不是本机重复路线的记录。",
        "",
        "使用贪心覆盖选择最小代表集，覆盖所有具备完整逐样本报告的地图、已观测车型结构、通过结果、全部失败原因以及基线/新增比较状态。",
        "",
        "## 二、冻结结果",
        "",
        f"- 合格候选：{candidate_count} 条。",
        f"- 冻结代表样本：{len(rows)} 条，总大小 {total_bytes} 字节。",
        f"- 覆盖地图（{len(maps)}）：{', '.join(maps)}。",
        f"- 车型结构：{', '.join(f'{key}={value}' for key, value in sorted(structures.items()))}。",
        f"- 比较状态：{', '.join(f'{key}={value}' for key, value in sorted(statuses.items()))}。",
        f"- 标签：通过={labels['1']}，失败={labels['0']}。",
        f"- 失败原因（{len(failures)}）：{', '.join(failures)}。",
        "",
        "## 三、未覆盖项与边界",
        "",
        "- `bridge_4` 在本人审计清单中只有无标签路线，且没有对应报告或报告哈希，因此未进入本次原始样本冻结集。",
        "- 未发现满足完整报告条件的 `five_axis_structural` 样本；五轴由 `five_axis_explicit` 表示。",
        "- 历史 `route_input_hash` 算法未在仓库提供。本次 `comparison_status` 只反映与 `baseline_v1_359.csv` 的文件 SHA-256 精确比较，逻辑输入是否重复仍待验证。",
        "- 原始报告可能包含本机绝对路径，未上传报告正文；每条样本均保存重新验证的报告 SHA-256。",
        "- 本数据仅用于离线研究和实验准备，不代表实车验证结果或安全认证。",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("代表原始样本冻结报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    candidates, _, _ = load_candidates()
    selected, uncovered = select_representatives(candidates)
    rows = freeze(selected, uncovered)
    write_docs(rows, len(candidates))
    print(json.dumps({
        "eligible_candidates": len(candidates),
        "selected_samples": len(rows),
        "selected_bytes": sum(int(row["route_size_bytes"]) for row in rows),
        "maps": sorted({row["map"] for row in rows}),
        "structures": dict(Counter(row["vehicle_structure"] for row in rows)),
        "comparison_statuses": dict(Counter(row["comparison_status"] for row in rows)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
