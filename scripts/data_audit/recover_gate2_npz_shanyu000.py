#!/usr/bin/env python3
"""Build the SHANYU000 Gate 2 NPZ recovery manifest from local audit evidence.

The script verifies local files but never writes local absolute paths to repository
outputs and never copies raw NPZ files into the repository.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MEMBER = "SHANYU000"
FREEZE_COMMIT = "d423f0e"
FREEZE_OBJECT = f"{FREEZE_COMMIT}:data/frozen/raw_samples/{MEMBER}/manifest.csv"
LOCAL_CANDIDATES = (
    REPO_ROOT / "analysis" / "data_recovery" / MEMBER / "local_candidate_samples.csv"
)
OUTPUT_DIR = REPO_ROOT / "data" / "recovered_npz" / MEMBER
REPORT_DIR = REPO_ROOT / "reports" / "data_recovery" / MEMBER

SELECTION_ROLES = {
    "PG-SHANYU000-250b207fe1290aa9": "five_axis_pass",
    "PG-SHANYU000-07775005aa964d84": "five_axis_fail",
    "PG-SHANYU000-5dfc4fa9d2ac2c08": "five_axis_local_risk",
    "PG-SHANYU000-d070ecb1b1784582": "six_axis_pass",
    "PG-SHANYU000-9b865d34114e5806": "six_axis_fail",
    "PG-SHANYU000-ed82f3f5dcfaac0b": "six_axis_local_risk",
}

MANIFEST_FIELDS = [
    "sample_id",
    "route_file_sha256",
    "vehicle_structure",
    "map_id",
    "file_sha256",
    "array_names",
    "array_shapes",
    "source_machine",
    "label_status",
    "hard_certificate_passed",
    "hard_failure_reasons",
    "coverage_role",
    "file_size_bytes",
    "npz_readable",
    "report_file_sha256",
    "controlled_storage_reference",
    "controlled_storage_status",
    "freeze_source",
]

DAMAGED_FIELDS = [
    "sample_id",
    "route_file_sha256",
    "vehicle_structure",
    "map_id",
    "source_machine",
    "error_type",
    "error_message",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_freeze_manifest() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "show", FREEZE_OBJECT],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return list(csv.DictReader(io.StringIO(completed.stdout.lstrip("\ufeff"))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_npz(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    names: list[str] = []
    shapes: dict[str, list[int]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            value = np.asarray(archive[name])
            names.append(name)
            shapes[name] = list(value.shape)
    return names, shapes


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    freeze_rows = load_freeze_manifest()
    candidate_rows = load_csv(LOCAL_CANDIDATES)
    if len(freeze_rows) != 24:
        raise RuntimeError(f"Expected 24 freeze records, found {len(freeze_rows)}")

    candidates_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        route_hash = row.get("route_file_sha256", "").strip().lower()
        route_path = row.get("route_path_local", "").strip()
        if route_hash and route_path:
            candidates_by_hash[route_hash].append(row)

    valid: dict[str, dict[str, object]] = {}
    damaged: list[dict[str, object]] = []
    locally_found = 0

    for freeze in freeze_rows:
        expected_hash = freeze["route_file_sha256"].lower()
        candidates = sorted(
            candidates_by_hash.get(expected_hash, []),
            key=lambda item: item["route_path_local"].casefold(),
        )
        existing = [row for row in candidates if Path(row["route_path_local"]).is_file()]
        if existing:
            locally_found += 1

        errors: list[str] = []
        for candidate in existing:
            path = Path(candidate["route_path_local"])
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                errors.append("sha256_mismatch")
                continue
            try:
                array_names, array_shapes = inspect_npz(path)
            except Exception:
                errors.append("npz_read_error")
                continue
            valid[freeze["sample_id"]] = {
                "freeze": freeze,
                "candidate": candidate,
                "path": path,
                "actual_hash": actual_hash,
                "array_names": array_names,
                "array_shapes": array_shapes,
            }
            break

        if freeze["sample_id"] not in valid:
            error_type = errors[0] if errors else "local_file_not_found"
            damaged.append(
                {
                    "sample_id": freeze["sample_id"],
                    "route_file_sha256": expected_hash,
                    "vehicle_structure": freeze["vehicle_structure"],
                    "map_id": freeze["map"],
                    "source_machine": candidates[0]["source_machine"] if candidates else MEMBER,
                    "error_type": error_type,
                    "error_message": {
                        "sha256_mismatch": "Local candidate hash does not match the freeze record.",
                        "npz_read_error": "Local candidate could not be read with NumPy allow_pickle=False.",
                        "local_file_not_found": "No currently available local file matched this freeze record.",
                    }[error_type],
                }
            )

    missing_selected = sorted(set(SELECTION_ROLES) - set(valid))
    if missing_selected:
        raise RuntimeError(f"Selected samples failed validation: {missing_selected}")

    output_rows: list[dict[str, object]] = []
    for sample_id, coverage_role in SELECTION_ROLES.items():
        item = valid[sample_id]
        freeze = item["freeze"]
        candidate = item["candidate"]
        output_rows.append(
            {
                "sample_id": sample_id,
                "route_file_sha256": freeze["route_file_sha256"],
                "vehicle_structure": freeze["vehicle_structure"],
                "map_id": freeze["map"],
                "file_sha256": item["actual_hash"],
                "array_names": json.dumps(item["array_names"], ensure_ascii=True, separators=(",", ":")),
                "array_shapes": json.dumps(item["array_shapes"], ensure_ascii=True, separators=(",", ":")),
                "source_machine": candidate["source_machine"],
                "label_status": freeze["label_status"],
                "hard_certificate_passed": freeze["hard_certificate_passed"],
                "hard_failure_reasons": freeze["hard_failure_reasons"],
                "coverage_role": coverage_role,
                "file_size_bytes": item["path"].stat().st_size,
                "npz_readable": "true",
                "report_file_sha256": freeze["report_file_sha256"],
                "controlled_storage_reference": f"restricted_raw/{MEMBER}/{sample_id}.npz",
                "controlled_storage_status": "verified_local_pending_controlled_transfer",
                "freeze_source": FREEZE_OBJECT,
            }
        )

    manifest_path = OUTPUT_DIR / "npz_manifest.csv"
    damaged_path = OUTPUT_DIR / "damaged_files.csv"
    write_csv(manifest_path, MANIFEST_FIELDS, output_rows)
    write_csv(damaged_path, DAMAGED_FIELDS, damaged)

    structures = Counter(row["vehicle_structure"] for row in output_rows)
    roles = Counter(str(row["coverage_role"]).rsplit("_", 1)[-1] for row in output_rows)
    pass_count = sum(row["hard_certificate_passed"] == "1" for row in output_rows)
    fail_count = sum(row["hard_certificate_passed"] == "0" for row in output_rows)
    map_count = len({row["map_id"] for row in output_rows})

    report = f"""# Gate 2 NPZ 原始样本回查报告（{MEMBER}）

## 1. 任务边界

本次仅对本机已有 PathGuard 原始 NPZ 进行哈希匹配、可读性检查、数组名称与形状提取，以及代表样本清单冻结。未训练模型、未修改标签，未把运行后的指标写入输入特征，也未向仓库复制原始 NPZ、工程源码、地图源码、控制器源码、模型文件或压缩备份。

仓库中未发现文件名或正文明确标注为“数据冻结 V3”的清单。本次将既有正式冻结提交 `{FREEZE_COMMIT}` 中的 24 条 `manifest.csv` 作为 Gate 2 的 V3 输入映射；该版本映射需要由数据负责人确认。所有标签、硬证书结果与报告哈希均原样继承该清单。

## 2. 扫描与验证结果

| 项目 | 数量 |
|---|---:|
| 冻结记录扫描 | {len(freeze_rows)} |
| 本机找到对应文件 | {locally_found} |
| SHA-256 一致且 NPZ 可读取 | {len(valid)} |
| 损坏或缺失记录 | {len(damaged)} |
| 新增代表样本清单记录 | {len(output_rows)} |
| 覆盖地图 | {map_count} |

NPZ 使用 `numpy.load(..., allow_pickle=False)` 打开，并逐一读取全部数组以确认结构可访问；`file_sha256` 与冻结清单中的 `route_file_sha256` 逐条比较。数组名称和形状已写入 `npz_manifest.csv`。

## 3. 代表样本覆盖

| 车型结构 | 通过 | 失败 | 局部风险 | 合计 |
|---|---:|---:|---:|---:|
| five_axis_explicit | 1 | 1 | 1 | {structures['five_axis_explicit']} |
| six_axis_structural | 1 | 1 | 1 | {structures['six_axis_structural']} |
| 合计 | {roles['pass']} | {roles['fail']} | {roles['risk']} | {len(output_rows)} |

代表样本共包含 {pass_count} 条硬证书通过记录和 {fail_count} 条硬证书失败记录。局部风险覆盖角色依据冻结清单已有的 `closed_loop_pcd_clearance`、`closed_loop_txt_clearance` 或 `closed_loop_heightmap_footprint` 失败原因确定，仅用于挑选代表样本，不构成新标签。

## 4. 受控存储说明

仓库清单中的 `controlled_storage_reference` 是供后续受控传输使用的逻辑标识，不是本机绝对路径，也不表示文件已上传到受控存储。6 个已验证的原始 NPZ 继续保留在源机器上，状态统一记录为 `verified_local_pending_controlled_transfer`。后续如需主线提取特征，应由数据负责人按清单哈希将文件传入获批的受控数据区，并在传输后再次核验 SHA-256。

主仓库本次只提交脱敏清单、哈希、数组名称与形状、损坏记录及本报告，不提交原始 NPZ。

## 5. 输出文件

- `data/recovered_npz/{MEMBER}/npz_manifest.csv`
- `data/recovered_npz/{MEMBER}/damaged_files.csv`
- `reports/data_recovery/{MEMBER}/gate2_npz_recovery.md`

## 6. 待确认事项

- 待数据负责人确认提交 `{FREEZE_COMMIT}` 的 24 条冻结清单是否即项目所称“数据冻结 V3”。
- 待确定获批的受控原始数据存储位置；在此之前不执行 NPZ 传输。
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "gate2_npz_recovery.md").write_text(report, encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "scanned": len(freeze_rows),
                "locally_found": locally_found,
                "valid": len(valid),
                "damaged": len(damaged),
                "selected": len(output_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
