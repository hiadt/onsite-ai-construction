"""Build path-only display evidence for frozen records from a local NPZ manifest.

The output intentionally contains no local source paths or raw NPZ payloads. Each
included route must match the frozen SHA-256 and all 55 execution-time features.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from route_input import parse_route  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "demo" / "data" / "route_geometry_all.json.gz",
    )
    args = parser.parse_args()

    frozen = pd.read_csv(ROOT / "demo" / "data" / "modeling_features_final.csv")
    manifest = pd.read_csv(args.manifest).set_index("sample_id")
    contract = json.loads((ROOT / "demo" / "data" / "feature_contract_v2.json").read_text(encoding="utf-8"))
    columns = contract["training_feature_columns"]
    defaults = {
        "five_axis": (12.0, 3.2, 5.0),
        "six_axis": (15.0, 3.4, 6.0),
        "unknown": (12.0, 3.2, 5.0),
    }
    routes: dict[str, dict] = {}
    for number, sample in enumerate(frozen.itertuples(index=False), start=1):
        entry = manifest.loc[sample.sample_id]
        if isinstance(entry, pd.DataFrame):
            entry = entry.loc[entry["route_file_sha256"].eq(sample.route_file_sha256)].iloc[0]
        path = Path(entry.route_path_local)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != sample.route_file_sha256 or digest != entry.route_file_sha256:
            raise ValueError(f"SHA-256 mismatch for {sample.sample_id}")
        length, width, rear_reference = defaults[sample.vehicle_structure]
        config = {
            "vehicle_structure": sample.vehicle_structure,
            "map_id": sample.map_id,
            "length_m": length,
            "width_m": width,
            "reference_from_rear_m": rear_reference,
            "dimension_source": "illustrative_demo_values",
        }
        features, _, geometry = parse_route(payload, path.name, config)
        if geometry is None:
            raise ValueError(f"No usable coordinates for {sample.sample_id}")
        if str(features.iloc[0]["route_file_sha256"]) != digest:
            raise ValueError(f"Extracted hash mismatch for {sample.sample_id}")
        expected = np.asarray([getattr(sample, column) for column in columns], dtype=float)
        actual = features[columns].to_numpy(dtype=float)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7,
                                   err_msg=f"Feature mismatch for {sample.sample_id}")
        # Only geometry supported by the same raw route is serialized. Paths and
        # credentials never enter the output.
        routes[sample.sample_id] = geometry
        if number % 25 == 0 or number == len(frozen):
            print(f"verified {number}/{len(frozen)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as target:
        json.dump({"routes": routes}, target, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {len(routes)} verified routes to {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
