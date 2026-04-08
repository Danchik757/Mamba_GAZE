#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mamba_gaze.pipeline import DatasetPaths, MeshMambaFaceProjector, RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    defaults = DatasetPaths.local_defaults()

    parser = argparse.ArgumentParser(
        description="Project MeshMamba non-texture gaze trajectories onto face-level saliency maps."
    )
    parser.add_argument("--model", required=True, help="Model name, e.g. Aquarium_Deep_Sea_Diver_v1_L1")
    parser.add_argument("--gaze-csv-dir", default=str(defaults.gaze_csv_dir))
    parser.add_argument("--mesh-dir", default=str(defaults.mesh_dir))
    parser.add_argument("--json-dir", default=str(defaults.json_dir))
    parser.add_argument("--gt-dir", default=str(defaults.gt_dir) if defaults.gt_dir is not None else "")
    parser.add_argument("--output-dir", default=str(defaults.output_dir))
    parser.add_argument("--mapping-json", default=str(defaults.mapping_json))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--frame-alignment", choices=["nearest", "floor"], default="nearest")
    parser.add_argument("--point-weight-mode", choices=["unit", "delta_t"], default="unit")
    parser.add_argument("--ray-batch-size", type=int, default=64)
    parser.add_argument("--smoothing-mode", choices=["none", "diffusion", "geodesic_kde"], default="diffusion")
    parser.add_argument("--smoothing-steps", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.6)
    parser.add_argument(
        "--geodesic-kde-sigma-scale",
        type=float,
        default=3.0,
        help="sigma = scale * mean face-adjacency edge length",
    )
    parser.add_argument(
        "--geodesic-kde-radius-scale",
        type=float,
        default=3.0,
        help="truncate geodesic Gaussian KDE at radius_scale * sigma",
    )
    parser.add_argument("--max-participants", type=int, default=None)
    parser.add_argument("--max-points-per-participant", type=int, default=None)
    parser.add_argument(
        "--participant-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional participation_id filter. If provided, only these participants are processed.",
    )
    parser.add_argument(
        "--save-participant-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw/smoothed/normalized participant maps.",
    )
    parser.add_argument(
        "--precompute-all-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache transformed vertices for all frames before projection.",
    )
    parser.add_argument(
        "--proxy-fixation-percentiles",
        nargs="+",
        type=float,
        default=[90.0, 95.0, 99.0],
        help="Top-percentile GT masks used for proxy NSS/AUC metrics.",
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
        ray_batch_size=args.ray_batch_size,
        smoothing_mode=args.smoothing_mode,
        smoothing_steps=args.smoothing_steps,
        smoothing_alpha=args.smoothing_alpha,
        geodesic_kde_sigma_scale=args.geodesic_kde_sigma_scale,
        geodesic_kde_radius_scale=args.geodesic_kde_radius_scale,
        participant_ids=None if args.participant_ids is None else tuple(args.participant_ids),
        max_participants=args.max_participants,
        max_points_per_participant=args.max_points_per_participant,
        save_participant_maps=args.save_participant_maps,
        precompute_all_frames=args.precompute_all_frames,
        proxy_fixation_percentiles=tuple(args.proxy_fixation_percentiles),
    )

    projector = MeshMambaFaceProjector(dataset_paths=dataset_paths, runtime_config=runtime_config)
    summary = projector.run_model(args.model)
    print(f"Done. Outputs: {summary['output_dir']}")


if __name__ == "__main__":
    main()
