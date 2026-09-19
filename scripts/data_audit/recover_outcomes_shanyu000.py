#!/usr/bin/env python3
"""Recover outcome evidence for SHANYU000's 3,061 unlabeled routes.

The script is deliberately conservative: planning feasibility is treated as a
metric, not as a final safety label, and aggregate counts are never propagated
to individual routes.  It reads the prior local audit inventory and performs a
bounded discovery for explicitly named result artifacts below those same scan
roots; it does not scan unrelated disks or directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


USER = "SHANYU000"
ALLOWED_STATUSES = {
    "recovered_passed",
    "recovered_failed",
    "has_metrics_no_final_label",
    "aggregate_only",
    "no_result_found",
    "ambiguous_result",
}
RESULT_BASENAMES = {
    "summary",
    "safety_summary",
    "validation_report",
    "selection_report",
    "certificate",
    "certificate_full",
    "result",
    "metrics",
    "run_summary",
}
RESULT_EXTENSIONS = {".json", ".csv", ".jsonl", ".log"}
SKIP_DIRS = {
    ".git", ".idea", ".vs", ".pytest_cache", ".ruff_cache",
    "__pycache__", "node_modules", ".venv", "venv", "trt_cache",
}
FINAL_LABEL_KEYS = (
    "hard_certificate_passed",
    "certificate_passed",
    "validation_passed",
    "final_passed",
    "final_result",
    "outcome_label",
    "strict_path_pass",
    "accepted",
)
METRIC_HINTS = (
    "metric", "failure", "reason", "completed", "completion", "feasible",
    "clearance", "slip", "utilization", "lateral_error", "heading_error",
    "speed", "length", "samples", "point_count", "classification",
)
AGGREGATE_HINTS = (
    "passed_count", "failed_count", "pass_count", "fail_count", "total_count",
    "total_routes", "aggregate", "pass_rate", "failure_rate",
)
HASH_KEYS = {
    "route_file_sha256", "route_sha256", "route_hash", "input_sha256",
    "artifact_sha256", "sha256",
}
ROUTE_KEYS = {
    "route", "route_path", "route_file", "route_path_local", "input_route",
    "trajectory", "trajectory_path", "candidate", "candidate_path", "output",
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    scan_root: Path
    sha256: str
    size_bytes: int
    modified_time: str


@dataclass(frozen=True)
class Evidence:
    target_hash: str
    source: SourceFile
    method: str
    confidence: str
    label: bool | None
    label_key: str
    has_metrics: bool
    note: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar_text(row.get(field, "")) for field in fields})


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "passed", "pass", "accepted", "yes", "1"}:
            return True
        if text in {"false", "failed", "fail", "rejected", "no", "0"}:
            return False
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value.strip().strip('"')))


def structure_group(value: str) -> str:
    text = value.casefold()
    if "five" in text or re.search(r"(^|\D)5([_-]?(axis|axle))", text):
        return "five_axis"
    if "six" in text or re.search(r"(^|\D)6([_-]?(axis|axle))", text):
        return "six_axis"
    return ""


def source_build_family(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
        return relative.parts[0] if len(relative.parts) > 1 else root.name
    except ValueError:
        return root.name


def lower_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().casefold(): value for key, value in record.items()}


def direct_record_signal(record: dict[str, Any]) -> bool:
    keys = {str(key).casefold() for key in record}
    return bool(
        keys.intersection(FINAL_LABEL_KEYS)
        or keys.intersection(HASH_KEYS)
        or keys.intersection(ROUTE_KEYS)
        or any(any(hint in key for hint in METRIC_HINTS) for key in keys)
        or any(any(hint in key for hint in AGGREGATE_HINTS) for key in keys)
    )


def collect_json_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if direct_record_signal(value):
            records.append(value)
        for child in value.values():
            if isinstance(child, (dict, list)):
                records.extend(collect_json_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(collect_json_records(child))
    return records


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return collect_json_records(json.load(handle))


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            records.extend(collect_json_records(value))
    return records


def read_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not any(token in line.casefold() for token in (".npz", "sha256", "passed", "failed")):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                route_match = re.search(r"([A-Za-z]:\\[^\r\n\"']+?\.npz)", line, re.I)
                label_match = re.search(
                    r"(hard_certificate_passed|certificate_passed|validation_passed|final_passed)\s*[:=]\s*(true|false)",
                    line,
                    re.I,
                )
                if route_match and label_match:
                    records.append({
                        "route": route_match.group(1),
                        label_match.group(1): label_match.group(2),
                        "_line": line_number,
                    })
                continue
            records.extend(collect_json_records(value))
    return records


def read_source_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".json":
        return read_json(path)
    if path.suffix.casefold() == ".csv":
        return read_csv_records(path)
    if path.suffix.casefold() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.casefold() == ".log":
        return read_log(path)
    return []


def discover_sources(
    inventory: list[dict[str, str]], roots: list[Path]
) -> tuple[list[SourceFile], list[str]]:
    inventory_by_path = {normalized_path(row["path_local"]): row for row in inventory}
    candidates: dict[str, tuple[Path, Path]] = {}
    for row in inventory:
        path = Path(row["path_local"])
        if path.stem.casefold() in RESULT_BASENAMES and path.suffix.casefold() in RESULT_EXTENSIONS:
            candidates[normalized_path(str(path))] = (path, Path(row["scan_root"]))

    discovery_errors: list[str] = []
    for root in roots:
        if not root.exists():
            discovery_errors.append(f"root_missing:{root}")
            continue
        for current, dirs, files in os.walk(root, topdown=True):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            current_path = Path(current)
            for name in files:
                path = current_path / name
                if path.stem.casefold() not in RESULT_BASENAMES:
                    continue
                if path.suffix.casefold() not in RESULT_EXTENSIONS:
                    continue
                candidates.setdefault(normalized_path(str(path)), (path, root))

    sources: list[SourceFile] = []
    for key in sorted(candidates):
        path, root = candidates[key]
        try:
            stat = path.stat()
            cached = inventory_by_path.get(key, {})
            digest = cached.get("sha256") or sha256_file(path)
            modified = cached.get("modified_time") or datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat()
            sources.append(SourceFile(path, root, digest, stat.st_size, modified))
        except OSError as exc:
            discovery_errors.append(f"source_unreadable:{path}:{type(exc).__name__}:{exc}")
    return sources, discovery_errors


def record_route_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key, value in record.items():
        if not isinstance(value, str):
            continue
        key_text = str(key).casefold()
        if value.casefold().endswith(".npz") and (
            key_text in ROUTE_KEYS or any(hint in key_text for hint in ("route", "trajectory", "candidate", "input", "output"))
        ):
            refs.append(value)
    return list(dict.fromkeys(refs))


def record_hashes(record: dict[str, Any]) -> list[str]:
    hashes: list[str] = []
    for key, value in record.items():
        key_text = str(key).casefold()
        text = str(value).strip().casefold()
        if (key_text in HASH_KEYS or "sha256" in key_text) and re.fullmatch(r"[0-9a-f]{64}", text):
            hashes.append(text)
    return list(dict.fromkeys(hashes))


def final_label(record: dict[str, Any]) -> tuple[bool | None, str, bool]:
    lowered = lower_record(record)
    found: list[tuple[bool, str]] = []
    for key in FINAL_LABEL_KEYS:
        if key not in lowered:
            continue
        value = parse_bool(lowered[key])
        if value is not None:
            found.append((value, key))
    if not found:
        return None, "", False
    values = {value for value, _ in found}
    if len(values) > 1:
        return None, "+".join(key for _, key in found), True
    return found[0][0], found[0][1], False


def record_has_metrics(record: dict[str, Any]) -> bool:
    return any(
        any(hint in str(key).casefold() for hint in METRIC_HINTS)
        for key in record
    )


def record_is_aggregate(record: dict[str, Any]) -> bool:
    keys = [str(key).casefold() for key in record]
    return any(any(hint in key for hint in AGGREGATE_HINTS) for key in keys)


def first_value(record: dict[str, Any], *keys: str) -> str:
    lowered = lower_record(record)
    for key in keys:
        value = lowered.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def candidate_composite_matches(
    record: dict[str, Any], source: SourceFile, targets: list[dict[str, str]]
) -> list[dict[str, str]]:
    map_name = first_value(record, "map", "map_name", "map_id", "scene", "scenario")
    point_count = numeric(first_value(record, "point_count", "samples", "sample_count", "num_points"))
    length_m = numeric(first_value(record, "route_length_m", "length_m", "path_length_m"))
    structure = structure_group(first_value(record, "vehicle_structure", "axis", "axle_structure", "vehicle_axis"))
    family = source_build_family(source.path, source.scan_root)
    if not map_name or point_count is None or length_m is None or not structure or not family:
        return []
    matches: list[dict[str, str]] = []
    for target in targets:
        target_points = numeric(target.get("point_count", ""))
        target_length = numeric(target.get("route_length_m", ""))
        if target_points is None or target_length is None:
            continue
        if target.get("map", "").casefold() != map_name.casefold():
            continue
        if target.get("build_family", "").casefold() != family.casefold():
            continue
        if structure_group(target.get("vehicle_structure", "")) != structure:
            continue
        if int(round(target_points)) != int(round(point_count)):
            continue
        if abs(target_length - length_m) > max(0.05, abs(target_length) * 1e-5):
            continue
        matches.append(target)
    return matches


def sample_id(route_hash: str) -> str:
    return f"SHANYU000-{route_hash[:16]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    audit_dir = repo / "analysis" / "data_recovery" / USER
    inventory_path = audit_dir / "local_file_inventory.csv"
    input_path = audit_dir / "new_candidate_samples.csv"

    all_candidates = read_csv(input_path)
    targets = [
        row for row in all_candidates
        if parse_bool(row.get("hard_certificate_passed", "")) is None
        and row.get("local_relation") == "new_route_without_label"
    ]
    if len(targets) != 3061:
        raise RuntimeError(f"expected 3061 unlabeled unique routes, found {len(targets)}")
    route_hashes = {row["route_file_sha256"] for row in targets}
    if len(route_hashes) != len(targets) or "" in route_hashes:
        raise RuntimeError("target route hashes are missing or not unique")

    inventory = read_csv(inventory_path)
    roots = sorted({Path(row["scan_root"]) for row in inventory}, key=lambda item: str(item).casefold())
    sources, discovery_errors = discover_sources(inventory, roots)
    inventory_hash_by_path = {
        normalized_path(row["path_local"]): row.get("sha256", "") for row in inventory
        if row.get("sha256")
    }
    target_by_hash = {row["route_file_sha256"].casefold(): row for row in targets}
    target_by_path = {normalized_path(row["route_path_local"]): row for row in targets}
    target_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in targets:
        target_by_name[PureWindowsPath(row["route_path_local"]).name.casefold()].append(row)

    evidence_by_target: dict[str, list[Evidence]] = defaultdict(list)
    aggregate_sources: list[SourceFile] = []
    parse_errors: list[str] = []
    parsed_records = 0
    final_label_records = 0
    matched_final_label_records = 0
    internally_conflicting_records = 0
    final_label_source_files: set[str] = set()
    matched_final_label_source_files: set[str] = set()

    for source in sources:
        try:
            records = read_source_records(source.path)
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            parse_errors.append(f"source_parse_error:{source.path}:{type(exc).__name__}:{exc}")
            continue
        if not records:
            continue
        for record_index, record in enumerate(records, 1):
            parsed_records += 1
            label, label_key, internal_label_conflict = final_label(record)
            if label is not None or internal_label_conflict:
                final_label_records += 1
                final_label_source_files.add(normalized_path(str(source.path)))
            if internal_label_conflict:
                internally_conflicting_records += 1
            has_metrics = record_has_metrics(record)
            matched: dict[str, tuple[str, str]] = {}

            for digest in record_hashes(record):
                if digest in target_by_hash:
                    matched[digest] = ("route_file_sha256", "high")

            refs = record_route_refs(record)
            for ref in refs:
                ref_key = normalized_path(ref)
                if ref_key in target_by_path:
                    digest = target_by_path[ref_key]["route_file_sha256"].casefold()
                    matched.setdefault(digest, ("explicit_route_path", "high"))
                    continue
                ref_digest = inventory_hash_by_path.get(ref_key, "").casefold()
                if ref_digest in target_by_hash:
                    matched.setdefault(ref_digest, ("referenced_route_sha256", "high"))
                    continue
                name_matches = target_by_name.get(PureWindowsPath(ref).name.casefold(), [])
                if len(name_matches) == 1:
                    digest = name_matches[0]["route_file_sha256"].casefold()
                    matched.setdefault(digest, ("unique_route_filename", "medium"))

            if not matched:
                composite = candidate_composite_matches(record, source, targets)
                if len(composite) == 1:
                    digest = composite[0]["route_file_sha256"].casefold()
                    matched[digest] = ("unique_map_points_length_structure_build", "medium")

            if not matched and record_is_aggregate(record) and not refs and not record_hashes(record):
                aggregate_sources.append(source)

            if matched and (label is not None or internal_label_conflict):
                matched_final_label_records += 1
                matched_final_label_source_files.add(normalized_path(str(source.path)))

            for digest, (method, confidence) in matched.items():
                note_parts = [f"record={record_index}"]
                if internal_label_conflict:
                    note_parts.append(f"conflicting_final_fields={label_key}")
                elif label_key:
                    note_parts.append(f"final_field={label_key}")
                if has_metrics:
                    note_parts.append("metrics_present")
                evidence_by_target[digest].append(Evidence(
                    target_hash=digest,
                    source=source,
                    method=method,
                    confidence=confidence,
                    label=label,
                    label_key=label_key,
                    has_metrics=has_metrics,
                    note=";".join(note_parts),
                ))

    # Aggregate-only evidence is attached only when a result file shares the
    # route directory tree and exactly one target has the same map/build family.
    aggregate_by_target: dict[str, list[SourceFile]] = defaultdict(list)
    for source in dict.fromkeys(aggregate_sources):
        family = source_build_family(source.path, source.scan_root)
        source_parts = [part.casefold() for part in PureWindowsPath(source.path).parts]
        possible: list[dict[str, str]] = []
        for target in targets:
            if target.get("build_family", "").casefold() != family.casefold():
                continue
            route_parts = [part.casefold() for part in PureWindowsPath(target["route_path_local"]).parts]
            common = 0
            for left, right in zip(source_parts, route_parts):
                if left != right:
                    break
                common += 1
            if common >= 4:
                possible.append(target)
        if len(possible) == 1:
            aggregate_by_target[possible[0]["route_file_sha256"].casefold()].append(source)

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    output_rows: list[dict[str, Any]] = []
    chosen_sources: dict[str, list[SourceFile]] = defaultdict(list)
    for target in targets:
        digest = target["route_file_sha256"].casefold()
        evidence = evidence_by_target.get(digest, [])
        labels = {item.label for item in evidence if item.label is not None}
        conflict_fields = any("conflicting_final_fields=" in item.note for item in evidence)
        if conflict_fields or len(labels) > 1:
            status = "ambiguous_result"
            outcome_label: bool | None = None
        elif labels == {True}:
            status = "recovered_passed"
            outcome_label = True
        elif labels == {False}:
            status = "recovered_failed"
            outcome_label = False
        elif any(item.has_metrics for item in evidence):
            status = "has_metrics_no_final_label"
            outcome_label = None
        elif aggregate_by_target.get(digest):
            status = "aggregate_only"
            outcome_label = None
        else:
            status = "no_result_found"
            outcome_label = None

        ordered = sorted(
            evidence,
            key=lambda item: (
                0 if item.label is not None else 1,
                confidence_rank[item.confidence],
                str(item.source.path).casefold(),
            ),
        )
        if ordered:
            primary = ordered[0]
            source_path = str(primary.source.path)
            source_hash = primary.source.sha256
            match_method = primary.method
            confidence = primary.confidence
            chosen_sources[digest] = list(dict.fromkeys(item.source for item in ordered[:3]))
            notes = f"evidence_records={len(ordered)};{primary.note}"
            if len(labels) > 1:
                notes += ";conflicting_labels=true"
        elif aggregate_by_target.get(digest):
            primary_source = aggregate_by_target[digest][0]
            source_path = str(primary_source.path)
            source_hash = primary_source.sha256
            match_method = "shared_run_directory_aggregate"
            confidence = "low"
            chosen_sources[digest] = [primary_source]
            notes = f"aggregate_sources={len(aggregate_by_target[digest])};not_assigned_as_sample_label"
        else:
            source_path = source_hash = match_method = confidence = ""
            notes = "no_matching_result_artifact_under_audited_roots"

        row = dict(target)
        row.update({
            "sample_id": sample_id(digest),
            "outcome_recovery_status": status,
            "outcome_label": outcome_label,
            "outcome_source_path": source_path,
            "outcome_source_sha256": source_hash,
            "outcome_match_method": match_method,
            "outcome_confidence": confidence,
            "outcome_notes": notes,
        })
        output_rows.append(row)

    extra_fields = [
        "sample_id", "outcome_recovery_status", "outcome_label",
        "outcome_source_path", "outcome_source_sha256", "outcome_match_method",
        "outcome_confidence", "outcome_notes",
    ]
    output_fields = list(all_candidates[0]) + extra_fields
    output_path = repo / "analysis" / "data_recovery" / "outcome_recovery" / f"{USER}_outcome_recovery.csv"
    write_csv(output_path, output_fields, output_rows)

    inventory_by_local_path = {normalized_path(row["path_local"]): row for row in inventory}
    transfer_fields = [
        "sample_id", "artifact_role", "path_local", "file_sha256", "size_bytes",
        "route_file_sha256", "vehicle_structure", "map", "outcome_recovery_status",
        "outcome_label", "transfer_reason",
    ]
    transfer_rows: list[dict[str, Any]] = []
    status_by_hash = {row["route_file_sha256"].casefold(): row for row in output_rows}
    for digest, source_list in chosen_sources.items():
        outcome = status_by_hash[digest]
        if outcome["outcome_recovery_status"] == "no_result_found":
            continue
        route_path = outcome["route_path_local"]
        route_inventory = inventory_by_local_path.get(normalized_path(route_path), {})
        transfer_rows.append({
            "sample_id": outcome["sample_id"],
            "artifact_role": "route_npz",
            "path_local": route_path,
            "file_sha256": digest,
            "size_bytes": route_inventory.get("size_bytes", ""),
            "route_file_sha256": digest,
            "vehicle_structure": outcome["vehicle_structure"],
            "map": outcome["map"],
            "outcome_recovery_status": outcome["outcome_recovery_status"],
            "outcome_label": outcome["outcome_label"],
            "transfer_reason": "matched_route_for_outcome_evidence",
        })
        for source in source_list:
            transfer_rows.append({
                "sample_id": outcome["sample_id"],
                "artifact_role": "outcome_source",
                "path_local": str(source.path),
                "file_sha256": source.sha256,
                "size_bytes": source.size_bytes,
                "route_file_sha256": digest,
                "vehicle_structure": outcome["vehicle_structure"],
                "map": outcome["map"],
                "outcome_recovery_status": outcome["outcome_recovery_status"],
                "outcome_label": outcome["outcome_label"],
                "transfer_reason": "primary_or_supporting_outcome_evidence",
            })
    transfer_path = repo / "analysis" / "data_recovery" / "outcome_recovery" / f"{USER}_files_to_transfer.csv"
    write_csv(transfer_path, transfer_fields, transfer_rows)

    status_counts = Counter(row["outcome_recovery_status"] for row in output_rows)
    recovered_by_structure: Counter[tuple[str, str]] = Counter()
    for row in output_rows:
        if row["outcome_recovery_status"] not in {"recovered_passed", "recovered_failed"}:
            continue
        recovered_by_structure[(structure_group(row["vehicle_structure"]) or "unknown", row["outcome_recovery_status"])] += 1
    transfer_role_counts = Counter(row["artifact_role"] for row in transfer_rows)
    transfer_bytes = sum(int(row["size_bytes"] or 0) for row in transfer_rows)
    unique_transfer_files: dict[str, dict[str, Any]] = {}
    for row in transfer_rows:
        unique_transfer_files.setdefault(normalized_path(row["path_local"]), row)
    unique_transfer_role_counts = Counter(row["artifact_role"] for row in unique_transfer_files.values())
    unique_transfer_bytes = sum(int(row["size_bytes"] or 0) for row in unique_transfer_files.values())

    report_path = repo / "reports" / "data_recovery" / "outcome_recovery" / f"{USER}_3061条结果回查报告.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"# {USER} 3061 条结果回查报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "本次仅回查既有审计清单中的 3061 条本机唯一路线且无有效硬证书标签的记录。搜索范围限定为既有审计根目录，并仅补充发现指定名称的结果文件；未执行全盘重扫、模型训练或闭环仿真。",
        "",
        "## 状态统计",
        "",
        f"- recovered_passed：{status_counts['recovered_passed']}",
        f"- recovered_failed：{status_counts['recovered_failed']}",
        f"- has_metrics_no_final_label：{status_counts['has_metrics_no_final_label']}",
        f"- aggregate_only：{status_counts['aggregate_only']}",
        f"- no_result_found：{status_counts['no_result_found']}",
        f"- ambiguous_result：{status_counts['ambiguous_result']}",
        f"- 合计：{sum(status_counts.values())}",
        "",
        "## 恢复标签的车型结构分布",
        "",
        f"- 五轴 recovered_passed：{recovered_by_structure[('five_axis', 'recovered_passed')]}；recovered_failed：{recovered_by_structure[('five_axis', 'recovered_failed')]}。",
        f"- 六轴 recovered_passed：{recovered_by_structure[('six_axis', 'recovered_passed')]}；recovered_failed：{recovered_by_structure[('six_axis', 'recovered_failed')]}。",
        f"- 未知或冲突结构 recovered_passed：{recovered_by_structure[('unknown', 'recovered_passed')]}；recovered_failed：{recovered_by_structure[('unknown', 'recovered_failed')]}。",
        "",
        "## 匹配与判定规则",
        "",
        "- 匹配优先级为路线文件 SHA-256、精确路线路径或全局唯一文件名、地图+点数+长度+车型结构+构建族的唯一组合。运行目录和修改时间只作辅助信息，不单独产生标签。",
        "- 只有明确的最终判定字段才写入 `outcome_label`；`feasible`、规划成功、完成状态和性能指标不替代硬证书结论。",
        "- 批次通过数量等聚合信息不分摊到单条路线；存在互相冲突的最终判定时标记为 `ambiguous_result`。",
        "- 车型结构沿用原审计中的显式配置或转向向量维度证据，不从最终结果反推车型。",
        "",
        "## 建议传输文件",
        "",
        f"- 路线 NPZ：{transfer_role_counts['route_npz']} 条清单记录。",
        f"- 结果证据文件：{transfer_role_counts['outcome_source']} 条清单记录。",
        f"- 清单记录总字节数（同一物理文件按样本引用计数）：{transfer_bytes}。",
        f"- 去重后物理文件：路线 NPZ {unique_transfer_role_counts['route_npz']} 个，结果证据 {unique_transfer_role_counts['outcome_source']} 个，共 {unique_transfer_bytes} 字节。",
        "- 本次提交仅包含回查清单、脚本和报告，未复制或上传原始 NPZ/结果文件；后续应按 `files_to_transfer.csv` 复核、去重后集中传输。",
        "",
        "## 执行记录与限制",
        "",
        f"- 检查结果源文件：{len(sources)}；解析记录：{parsed_records}。",
        f"- 含明确最终判定的结果源文件：{len(final_label_source_files)}；其中与 3061 条目标路线可靠匹配：{len(matched_final_label_source_files)}；源记录内部判定冲突：{internally_conflicting_records}。",
        f"- 定向发现异常：{len(discovery_errors)}；解析异常：{len(parse_errors)}。",
        "- 本报告不把规划可行性或批次汇总解释为安全认证结论。缺少明确最终字段的记录保留为待验证。",
        "- 绝对本机路径只用于团队内数据定位；传输前仍需检查隐私、凭证、缓存、模型和无关大文件。",
    ]
    if discovery_errors or parse_errors:
        report_lines.extend(["", "### 异常摘要", ""])
        report_lines.extend(f"- `{item}`" for item in (discovery_errors + parse_errors)[:20])
        if len(discovery_errors) + len(parse_errors) > 20:
            report_lines.append(f"- 其余 {len(discovery_errors) + len(parse_errors) - 20} 条异常未在报告展开，待验证。")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if set(status_counts) - ALLOWED_STATUSES:
        raise RuntimeError(f"unexpected status: {set(status_counts) - ALLOWED_STATUSES}")
    print(json.dumps({
        "targets": len(targets),
        "sources": len(sources),
        "parsed_records": parsed_records,
        "final_label_records": final_label_records,
        "matched_final_label_records": matched_final_label_records,
        "final_label_source_files": len(final_label_source_files),
        "matched_final_label_source_files": len(matched_final_label_source_files),
        "statuses": dict(sorted(status_counts.items())),
        "transfer_rows": len(transfer_rows),
        "unique_transfer_files": len(unique_transfer_files),
        "discovery_errors": len(discovery_errors),
        "parse_errors": len(parse_errors),
        "output": str(output_path),
        "transfer": str(transfer_path),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
