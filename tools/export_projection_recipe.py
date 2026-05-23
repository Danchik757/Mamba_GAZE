#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_gaze.camera_utils import build_projection_matrix_from_fov_degrees
from mamba_gaze.io_utils import load_json, load_obj, write_json
from mamba_gaze.name_mapping import build_dataset_mapping, resolve_model_from_mapping
from mamba_gaze.pipeline import DatasetPaths


def apply_projection_transform_np(
    vertices: np.ndarray,
    *,
    scale_xyz: np.ndarray,
    rotation_z_rad: float,
    translation_xyz: np.ndarray,
    recenter_to_bbox_center: bool = False,
    extra_rotate_x_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    scale_xyz = np.asarray(scale_xyz, dtype=np.float64).reshape(1, 3)
    translation_xyz = np.asarray(translation_xyz, dtype=np.float64).reshape(1, 3)

    bbox_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    transformed = vertices.copy()
    if recenter_to_bbox_center:
        transformed = transformed - bbox_center.reshape(1, 3)

    transformed = transformed * scale_xyz

    a = float(rotation_z_rad)
    ca = math.cos(a)
    sa = math.sin(a)
    x = transformed[:, 0]
    y = transformed[:, 1]
    z = transformed[:, 2]

    x2 = ca * x - sa * y
    y2 = sa * x + ca * y
    z2 = z

    rx = math.radians(float(extra_rotate_x_deg))
    if abs(rx) > 1e-12:
        crx = math.cos(rx)
        srx = math.sin(rx)
        y3 = crx * y2 - srx * z2
        z3 = srx * y2 + crx * z2
    else:
        y3 = y2
        z3 = z2

    out = np.stack([x2, y3, z3], axis=1)
    out = out + translation_xyz
    return out, bbox_center


def build_parser() -> argparse.ArgumentParser:
    defaults = DatasetPaths.local_defaults()
    parser = argparse.ArgumentParser(
        description="Export the exact mesh and camera transform recipe used during gaze projection."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--gaze-csv-dir", default=str(defaults.gaze_csv_dir))
    parser.add_argument("--mesh-dir", default=str(defaults.mesh_dir))
    parser.add_argument("--json-dir", default=str(defaults.json_dir))
    parser.add_argument("--gt-dir", default=str(defaults.gt_dir) if defaults.gt_dir is not None else "")
    parser.add_argument("--mapping-json", default=str(defaults.mapping_json))
    parser.add_argument("--recenter-to-bbox-center", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--extra-rotate-x-deg", type=float, default=0.0)
    parser.add_argument("--override-fov-deg", type=float, default=None)
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index used for preview transform export.")
    parser.add_argument("--output-json", default=None, help="Optional explicit output JSON path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset_paths = DatasetPaths(
        gaze_csv_dir=Path(args.gaze_csv_dir),
        mesh_dir=Path(args.mesh_dir),
        json_dir=Path(args.json_dir),
        gt_dir=None if not args.gt_dir else Path(args.gt_dir),
        output_dir=REPO_ROOT / "projection_recipes",
        mapping_json=Path(args.mapping_json),
    )
    mapping = build_dataset_mapping(
        gaze_csv_dir=dataset_paths.gaze_csv_dir,
        mesh_dir=dataset_paths.mesh_dir,
        json_dir=dataset_paths.json_dir,
        gt_dir=dataset_paths.gt_dir,
        output_path=dataset_paths.mapping_json,
    )
    resolved = resolve_model_from_mapping(args.model, mapping)

    vertices_np, faces_np = load_obj(resolved.mesh_path)
    metadata = load_json(resolved.json_path)

    frame_index = int(np.clip(args.frame_index, 0, len(metadata["frames"]) - 1))
    frame_meta = metadata["frames"][frame_index]
    scale_xyz = np.asarray(metadata["model_static"]["scale"], dtype=np.float64)
    translation_xyz = np.asarray(metadata["model_static"]["location"], dtype=np.float64)

    transformed_vertices, bbox_center = apply_projection_transform_np(
        vertices_np,
        scale_xyz=scale_xyz,
        rotation_z_rad=float(frame_meta["rotation_z_radians"]),
        translation_xyz=translation_xyz,
        recenter_to_bbox_center=bool(args.recenter_to_bbox_center),
        extra_rotate_x_deg=float(args.extra_rotate_x_deg),
    )

    fov_original = float(metadata["camera_static"]["fov_degrees"])
    fov_effective = float(args.override_fov_deg) if args.override_fov_deg is not None else fov_original
    if args.override_fov_deg is None:
        projection_matrix = np.asarray(metadata["camera_static"]["projection_matrix"], dtype=np.float64)
    else:
        projection_matrix = build_projection_matrix_from_fov_degrees(
            fov_degrees=fov_effective,
            aspect_ratio=float(metadata["video_info"]["aspect_ratio"]),
            clip_start=float(metadata["camera_static"]["clip_start"]),
            clip_end=float(metadata["camera_static"]["clip_end"]),
        ).astype(np.float64)

    recipe: dict[str, Any] = {
        "model": resolved.requested_model,
        "resolved_paths": resolved.to_jsonable(),
        "frame_index": frame_index,
        "frame_timestamp": float(frame_meta["timestamp"]),
        "frame_rotation_z_radians": float(frame_meta["rotation_z_radians"]),
        "frame_rotation_z_degrees": float(frame_meta["rotation_z_degrees"]),
        "transform_order": (
            "recenter_to_bbox_center -> scale -> frame_rotation_z -> extra_rotation_x -> translation"
            if args.recenter_to_bbox_center
            else "scale -> frame_rotation_z -> extra_rotation_x -> translation"
        ),
        "obj_runtime_transform": {
            "raw_obj_bbox_center": bbox_center.astype(float).tolist(),
            "recenter_to_bbox_center": bool(args.recenter_to_bbox_center),
            "scale_xyz_from_json": scale_xyz.astype(float).tolist(),
            "extra_rotate_x_deg": float(args.extra_rotate_x_deg),
            "translation_xyz_from_json": translation_xyz.astype(float).tolist(),
        },
        "camera_runtime": {
            "location_from_json": [float(x) for x in metadata["camera_static"]["location"]],
            "view_matrix_from_json": metadata["camera_static"]["view_matrix"],
            "projection_matrix_used": projection_matrix.tolist(),
            "fov_degrees_original_json": fov_original,
            "fov_degrees_effective": fov_effective,
            "clip_start": float(metadata["camera_static"]["clip_start"]),
            "clip_end": float(metadata["camera_static"]["clip_end"]),
            "aspect_ratio": float(metadata["video_info"]["aspect_ratio"]),
        },
        "raw_obj_geometry": {
            "num_vertices": int(vertices_np.shape[0]),
            "num_faces": int(faces_np.shape[0]),
            "bbox_min": vertices_np.min(axis=0).astype(float).tolist(),
            "bbox_max": vertices_np.max(axis=0).astype(float).tolist(),
        },
        "frame_geometry_after_runtime_transform": {
            "bbox_min": transformed_vertices.min(axis=0).astype(float).tolist(),
            "bbox_max": transformed_vertices.max(axis=0).astype(float).tolist(),
        },
        "notes": [
            "This recipe describes runtime transforms applied during gaze projection.",
            "recenter_to_bbox_center emulates Blender origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS').",
            "override_fov_deg changes camera projection only; it does not modify OBJ geometry.",
            "extra_rotate_x_deg is a runtime correction added in MAMBA_GAZE and is not stored in the source JSON.",
        ],
    }

    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_root = dataset_paths.output_dir
        output_root.mkdir(parents=True, exist_ok=True)
        output_json = output_root / f"{resolved.csv_model_name}_projection_recipe.json"

    write_json(output_json, recipe)
    print(f"Saved projection recipe: {output_json}")


if __name__ == "__main__":
    main()
