#!/usr/bin/env python3
"""Normalize merged local audit CSVs into a reproducible multi-configuration index.

This script only reads audit metadata already committed to the repository. It never
opens or copies raw NPZ/model/credential files.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "analysis" / "data_recovery"
OUT = ROOT / "data" / "manifests"
DERIVED = ROOT / "data" / "derived"

STATUS_MAP = {
    "existing_exact": "baseline_existing",
    "existing_route_new_report": "same_route_new_report",
    "confirmed_new_labeled": "confirmed_new_labeled",
    "confirmed_new_unlabeled": "confirmed_new_unlabeled",
    "unresolved": "unresolved",
    "conflict": "conflict",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def structure(row: dict[str, str]) -> str:
    raw = (row.get("vehicle_structure") or "").lower()
    if "six" in raw or row.get("steer_ff_rad_second_dimension") == "12" or row.get("steer_ff_rad_second_dim") == "12":
        return "six_axis"
    if "five" in raw or row.get("steer_ff_rad_second_dimension") == "10" or row.get("steer_ff_rad_second_dim") == "10":
        return "five_axis"
    return "unknown_or_conflict"


def main() -> None:
    rows: list[dict[str, str]] = []
    for member in ("ZYJisBoss", "SHANYU000"):
        for name in ("local_candidate_samples.csv", "new_candidate_samples.csv", "duplicate_or_conflict_samples.csv"):
            rows.extend(read_rows(AUDIT_ROOT / member / name))
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("route_file_sha256") or row.get("sample_key") or row.get("route_path_local")
        if not key:
            continue
        row["vehicle_structure"] = structure(row)
        old = unique.get(key)
        if old is None:
            unique[key] = row
        elif old.get("comparison_status") in ("unresolved", "") and row.get("comparison_status"):
            unique[key] = row
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in unique.values() for k in row})
    target = OUT / "pathguard_multiconfig_v1_manifest.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique.values())
    groups = {
        "labeled_samples.csv": lambda r: bool(r.get("hard_certificate_passed")) and r.get("comparison_status") != "duplicate",
        "unlabeled_routes.csv": lambda r: r.get("comparison_status") == "confirmed_new_unlabeled",
        "same_route_new_reports.csv": lambda r: r.get("comparison_status") == "same_route_new_report",
        "duplicates.csv": lambda r: r.get("comparison_status") == "duplicate",
        "conflicts.csv": lambda r: r.get("comparison_status") == "conflict",
        "unresolved.csv": lambda r: r.get("comparison_status") in ("unresolved", "pending_baseline_merge"),
        "vehicle_structure_evidence.csv": lambda r: r.get("vehicle_structure") in ("five_axis", "six_axis"),
    }
    for name, predicate in groups.items():
        selected = [r for r in unique.values() if predicate(r)]
        with (DERIVED / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    print(f"normalized={len(unique)} manifest={target}")


if __name__ == "__main__":
    main()
