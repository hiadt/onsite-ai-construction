#!/usr/bin/env python3
"""Build the unified PathGuard multi-configuration manifest."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "analysis" / "data_recovery"
BASELINE = ROOT / "data" / "manifests" / "baseline_v1_359.csv"
OUT = ROOT / "data" / "manifests"
DERIVED = ROOT / "data" / "derived"

def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def structure(row: dict[str, str]) -> str:
    raw = (row.get("vehicle_structure") or row.get("vehicle_profile_id") or "").lower()
    dim = row.get("steer_ff_rad_second_dimension") or row.get("steer_ff_rad_second_dim") or ""
    if "six" in raw or "6axle" in raw or dim == "12":
        return "six_axis"
    if "five" in raw or "5axle" in raw or dim == "10":
        return "five_axis"
    return "unknown_or_conflict"

def normalize_row(row: dict[str, str], status: str, source: str) -> dict[str, str]:
    out = dict(row)
    out["comparison_status"] = status
    out["source_member"] = source
    out["vehicle_structure"] = structure(out)
    if out.get("hard_certificate_passed") in ("0", "1"):
        out["label_status"] = "labeled"
    elif status == "baseline_existing":
        out["label_status"] = "aggregate_only" if not out.get("hard_certificate_passed") else "labeled"
    else:
        out["label_status"] = "unlabeled"
    out["sample_id"] = out.get("sample_key") or out.get("route_file_sha256") or out.get("route_input_hash") or out.get("route_path_local", "")
    return out

def main() -> None:
    rows: list[dict[str, str]] = []
    for row in read_rows(BASELINE):
        rows.append(normalize_row(row, "baseline_existing", "baseline_v1_359"))
    for member in ("ZYJisBoss", "SHANYU000"):
        for name in ("local_candidate_samples.csv", "new_candidate_samples.csv", "duplicate_or_conflict_samples.csv"):
            for row in read_rows(AUDIT_ROOT / member / name):
                status = row.get("comparison_status") or row.get("classification") or "unresolved"
                status = {"existing_exact":"baseline_existing", "existing_route_new_report":"same_route_new_report"}.get(status, status)
                rows.append(normalize_row(row, status, member))
    unique: dict[str, dict[str, str]] = {}
    priority = {"conflict":5, "confirmed_new_labeled":4, "confirmed_new_unlabeled":3, "same_route_new_report":2, "baseline_existing":1, "duplicate":0}
    for row in rows:
        key = row["sample_id"]
        if not key:
            continue
        old = unique.get(key)
        if old is None or priority.get(row["comparison_status"], 0) > priority.get(old["comparison_status"], 0):
            unique[key] = row
    all_rows = list(unique.values())
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in all_rows for k in row})
    with (OUT / "pathguard_multiconfig_v1_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    groups = {
        "labeled_samples.csv": lambda r: r["label_status"] == "labeled",
        "unlabeled_routes.csv": lambda r: r["comparison_status"] == "confirmed_new_unlabeled",
        "same_route_new_reports.csv": lambda r: r["comparison_status"] == "same_route_new_report",
        "duplicates.csv": lambda r: r["comparison_status"] == "duplicate",
        "conflicts.csv": lambda r: r["comparison_status"] == "conflict",
        "unresolved.csv": lambda r: r["comparison_status"] == "unresolved",
        "vehicle_structure_evidence.csv": lambda r: r["vehicle_structure"] in ("five_axis", "six_axis"),
    }
    for name, predicate in groups.items():
        selected = [r for r in all_rows if predicate(r)]
        with (DERIVED / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    print(f"normalized={len(all_rows)}")

if __name__ == "__main__":
    main()
