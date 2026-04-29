#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_gaze.name_mapping import build_dataset_mapping
from mamba_gaze.pipeline import DatasetPaths


def _is_resolved(match: dict | None) -> bool:
    if match is None:
        return False
    return match.get("match_type") in {"exact", "canonical"} and match.get("path") is not None


def main() -> None:
    defaults = DatasetPaths.local_defaults()
    parser = argparse.ArgumentParser(description="List resolved MeshMamba model names available for runs.")
    parser.add_argument("--gaze-csv-dir", default=str(defaults.gaze_csv_dir))
    parser.add_argument("--mesh-dir", default=str(defaults.mesh_dir))
    parser.add_argument("--json-dir", default=str(defaults.json_dir))
    parser.add_argument("--gt-dir", default=str(defaults.gt_dir) if defaults.gt_dir is not None else "")
    parser.add_argument("--mapping-json", default=None)
    parser.add_argument("--output-txt", default=None)
    parser.add_argument("--require-gt", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.mapping_json and Path(args.mapping_json).is_file():
        payload = json.loads(Path(args.mapping_json).read_text())
    else:
        payload = build_dataset_mapping(
            gaze_csv_dir=Path(args.gaze_csv_dir),
            mesh_dir=Path(args.mesh_dir),
            json_dir=Path(args.json_dir),
            gt_dir=None if not args.gt_dir else Path(args.gt_dir),
            output_path=None if args.mapping_json is None else Path(args.mapping_json),
        )

    model_names: list[str] = []
    for model_name, entry in sorted(payload["entries"].items()):
        has_mesh = _is_resolved(entry["mesh"])
        has_json = _is_resolved(entry["json"])
        has_gt = True if not args.require_gt else _is_resolved(entry["gt"])
        if has_mesh and has_json and has_gt:
            model_names.append(model_name)

    print(f"resolved_models={len(model_names)}")
    for model_name in model_names:
        print(model_name)

    if args.output_txt:
        output_path = Path(args.output_txt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(f"{name}\n" for name in model_names), encoding="utf-8")
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
