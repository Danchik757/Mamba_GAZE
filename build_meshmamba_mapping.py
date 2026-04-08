#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mamba_gaze.name_mapping import build_dataset_mapping
from mamba_gaze.pipeline import DatasetPaths


def main() -> None:
    defaults = DatasetPaths.local_defaults()

    parser = argparse.ArgumentParser(description="Build name mapping across MeshMamba CSV/OBJ/JSON/GT directories.")
    parser.add_argument("--gaze-csv-dir", default=str(defaults.gaze_csv_dir))
    parser.add_argument("--mesh-dir", default=str(defaults.mesh_dir))
    parser.add_argument("--json-dir", default=str(defaults.json_dir))
    parser.add_argument("--gt-dir", default=str(defaults.gt_dir) if defaults.gt_dir is not None else "")
    parser.add_argument("--output-json", default=str(defaults.mapping_json))
    args = parser.parse_args()

    payload = build_dataset_mapping(
        gaze_csv_dir=Path(args.gaze_csv_dir),
        mesh_dir=Path(args.mesh_dir),
        json_dir=Path(args.json_dir),
        gt_dir=None if not args.gt_dir else Path(args.gt_dir),
        output_path=Path(args.output_json),
    )
    print(f"Mapping saved to: {args.output_json}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
