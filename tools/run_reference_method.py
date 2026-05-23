#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_gaze.pipeline import DatasetPaths, RuntimeConfig
from mamba_gaze.reference_methods import MeshMambaReferenceProjector, ReferenceMethodConfig


def build_parser() -> argparse.ArgumentParser:
    defaults = DatasetPaths.local_defaults()

    parser = argparse.ArgumentParser(
        description=(
            "Run a MeshMamba-adapted reference reprojection baseline based on "
            "Mesh-Saliency-Projection/reprojection_methods."
        )
    )
    parser.add_argument("--model", required=True, help="Model name, e.g. Aquarium_Deep_Sea_Diver_v1_L1")
    parser.add_argument(
        "--method",
        required=True,
        choices=["screen_space_gaussian", "cone_projection_on_mesh"],
        help="Reference-style baseline to run.",
    )
    parser.add_argument("--gaze-csv-dir", default=str(defaults.gaze_csv_dir))
    parser.add_argument("--mesh-dir", default=str(defaults.mesh_dir))
    parser.add_argument("--json-dir", default=str(defaults.json_dir))
    parser.add_argument("--gt-dir", default=str(defaults.gt_dir) if defaults.gt_dir is not None else "")
    parser.add_argument("--output-dir", default=str(defaults.output_dir))
    parser.add_argument("--mapping-json", default=str(defaults.mapping_json))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-alignment", choices=["nearest", "floor"], default="nearest")
    parser.add_argument("--point-weight-mode", choices=["unit", "delta_t"], default="unit")
    parser.add_argument("--sigma-px", type=float, required=True, help="Screen-space Gaussian sigma in pixels.")
    parser.add_argument(
        "--radius-sigma-mult",
        type=float,
        default=3.0,
        help="Only used for cone_projection_on_mesh. Query radius in sigma units.",
    )
    parser.add_argument("--extra-rotate-x-deg", type=float, default=0.0)
    parser.add_argument(
        "--recenter-to-bbox-center",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--override-fov-deg", type=float, default=None)
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--max-points-per-participant", type=int, default=None)
    parser.add_argument("--participant-ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--save-participant-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--precompute-all-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--proxy-fixation-percentiles",
        nargs="+",
        type=float,
        default=[90.0, 95.0, 99.0],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_paths = DatasetPaths(
        gaze_csv_dir=Path(args.gaze_csv_dir),
        mesh_dir=Path(args.mesh_dir),
        json_dir=Path(args.json_dir),
        gt_dir=None if not args.gt_dir else Path(args.gt_dir),
        output_dir=Path(args.output_dir),
        mapping_json=Path(args.mapping_json),
    )
    runtime_config = RuntimeConfig(
        device=args.device,
        frame_alignment=args.frame_alignment,
        point_weight_mode=args.point_weight_mode,
        extra_rotate_x_deg=args.extra_rotate_x_deg,
        recenter_to_bbox_center=args.recenter_to_bbox_center,
        override_fov_deg=args.override_fov_deg,
        participant_ids=None if args.participant_ids is None else tuple(args.participant_ids),
        max_participants=args.max_participants,
        max_points_per_participant=args.max_points_per_participant,
        save_participant_maps=args.save_participant_maps,
        precompute_all_frames=args.precompute_all_frames,
        proxy_fixation_percentiles=tuple(args.proxy_fixation_percentiles),
    )
    method_config = ReferenceMethodConfig(
        method=args.method,
        sigma_px=args.sigma_px,
        radius_sigma_mult=args.radius_sigma_mult,
    )
    projector = MeshMambaReferenceProjector(
        dataset_paths=dataset_paths,
        runtime_config=runtime_config,
        method_config=method_config,
    )
    summary = projector.run_model(args.model)
    print(f"Done. Outputs: {summary['output_dir']}")


if __name__ == "__main__":
    main()
