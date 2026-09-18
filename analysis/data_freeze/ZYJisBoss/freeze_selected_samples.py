#!/usr/bin/env python3
"""Freeze one verified NPZ per vehicle-structure/map group from the 32 confirmed routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "analysis" / "data_recovery" / "ZYJisBoss"
ROUTES_CSV = SOURCE_DIR / "confirmed_new_unlabeled_routes.csv"
INVENTORY_CSV = SOURCE_DIR / "local_file_inventory.csv"
OUTPUT_DIR = REPO_ROOT / "analysis" / "data_freeze" / "ZYJisBoss"
RAW_DIR = OUTPUT_DIR / "raw_samples"
MANIFEST_PATH = OUTPUT_DIR / "frozen_raw_samples_manifest.csv"
README_PATH = OUTPUT_DIR / "README.md"
MANIFEST_FIELDS = [
    "frozen_vehicle_structure", "map", "source_path_local", "frozen_path_repo", "route_file_sha256",
    "route_input_hash_local", "vehicle_profile_id", "vehicle_profile_fingerprint", "identity_evidence",
    "steer_ff_rad_second_dimension", "route_length_m", "point_count", "file_size_bytes",
    "npz_array_count", "selection_reason",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_structure(row: dict) -> str:
    structure = row["vehicle_structure"]
    if structure.startswith("five_axis"):
        return "five_axis"
    if structure.startswith("six_axis"):
        return "six_axis"
    return "unknown_or_conflict"


def quality(row: dict) -> tuple[int, str]:
    complete = sum(bool(row.get(field)) for field in ("route_length_m", "point_count", "route_file_sha256"))
    return complete, row["route_file_sha256"]


def verify_npz(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        if not {"x_m", "y_m"} <= set(data.files):
            raise ValueError("missing x_m/y_m route arrays")
        for name in data.files:
            _ = data[name].shape
        return len(data.files)


def freeze() -> list[dict]:
    routes = read_csv(ROUTES_CSV)
    inventory = {row["path_local"]: row for row in read_csv(INVENTORY_CSV)}
    if len(routes) != 32:
        raise ValueError(f"expected 32 confirmed routes, found {len(routes)}")

    groups = defaultdict(list)
    for row in routes:
        groups[(frozen_structure(row), row["map"])].append(row)
    selected = [max(rows, key=quality) for rows in groups.values()]

    manifest = []
    for row in sorted(selected, key=lambda item: (frozen_structure(item), item["map"], item["route_path_local"])):
        source = Path(row["route_path_local"])
        item = inventory.get(str(source))
        if not item or item["artifact_kind"] != "candidate_route" or item["parse_status"] != "ok":
            raise ValueError(f"inventory does not confirm a valid candidate route: {source}")
        actual_hash = sha256(source)
        if actual_hash != row["route_file_sha256"] or actual_hash != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {source}")
        array_count = verify_npz(source)
        structure = frozen_structure(row)
        destination = RAW_DIR / structure / row["map"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != actual_hash:
            raise ValueError(f"copied file hash mismatch: {destination}")
        manifest.append({
            "frozen_vehicle_structure": structure,
            "map": row["map"],
            "source_path_local": str(source),
            "frozen_path_repo": destination.relative_to(REPO_ROOT).as_posix(),
            "route_file_sha256": actual_hash,
            "route_input_hash_local": row["route_input_hash"],
            "vehicle_profile_id": row["vehicle_profile_id"],
            "vehicle_profile_fingerprint": row["vehicle_profile_fingerprint"],
            "identity_evidence": row["identity_evidence"],
            "steer_ff_rad_second_dimension": row["steer_ff_rad_second_dimension"],
            "route_length_m": row["route_length_m"],
            "point_count": row["point_count"],
            "file_size_bytes": row["file_size_bytes"],
            "npz_array_count": array_count,
            "selection_reason": "highest metadata completeness in vehicle-structure/map group",
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    counts = Counter(row["frozen_vehicle_structure"] for row in manifest)
    README_PATH.write_text(
        "\n".join([
            "# 新增无标签路线代表样冻结", "",
            "输入仅为 `confirmed_new_unlabeled_routes.csv` 与 `local_file_inventory.csv`，未重新扫描磁盘。", "",
            f"- 输入候选：{len(routes)} 条",
            f"- 车型结构 × 地图分组：{len(groups)} 组",
            f"- 冻结代表样：{len(manifest)} 条",
            f"- five_axis：{counts['five_axis']} 条",
            f"- six_axis：{counts['six_axis']} 条",
            f"- unknown_or_conflict：{counts['unknown_or_conflict']} 条",
            "- 损坏或哈希不一致：0 条", "",
            "每组优先选择路线长度、点数和文件哈希完整的记录。复制前后均校验 SHA-256，并实际读取 NPZ 全部数组；此前候选集合中的重复副本未上传。", "",
            "详细原始路径、仓库路径、车型证据和哈希见 `frozen_raw_samples_manifest.csv`。", "",
        ]),
        encoding="utf-8",
    )
    return manifest


def self_test() -> None:
    assert frozen_structure({"vehicle_structure": "five_axis_explicit"}) == "five_axis"
    assert frozen_structure({"vehicle_structure": "six_axis_structural"}) == "six_axis"
    assert frozen_structure({"vehicle_structure": "unknown_or_conflict"}) == "unknown_or_conflict"
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    manifest = freeze()
    print(f"froze {len(manifest)} verified representative NPZ files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
