#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_gaze.io_utils import ensure_dir, write_json
from mamba_gaze.pipeline import DatasetPaths, MeshMambaFaceProjector, RuntimeConfig
from mamba_gaze.reference_methods import MeshMambaReferenceProjector, ReferenceMethodConfig


def build_parser() -> argparse.ArgumentParser:
    defaults = DatasetPaths.local_defaults()

    parser = argparse.ArgumentParser(
        description=(
            "Run the current exact ray-casting pipeline and Mesh-Saliency-Projection-style "
            "reference baselines on the same MeshMamba model and compare outputs."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["our_pipeline", "screen_space_gaussian", "cone_projection_on_mesh"],
        choices=["our_pipeline", "screen_space_gaussian", "cone_projection_on_mesh"],
    )
    parser.add_argument("--gaze-csv-dir", default=str(defaults.gaze_csv_dir))
    parser.add_argument("--mesh-dir", default=str(defaults.mesh_dir))
    parser.add_argument("--json-dir", default=str(defaults.json_dir))
    parser.add_argument("--gt-dir", default=str(defaults.gt_dir) if defaults.gt_dir is not None else "")
    parser.add_argument("--output-root", default=str(Path.cwd() / "method_comparisons"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-alignment", choices=["nearest", "floor"], default="nearest")
    parser.add_argument("--point-weight-mode", choices=["unit", "delta_t"], default="unit")
    parser.add_argument("--ray-batch-size", type=int, default=64)
    parser.add_argument("--smoothing-mode", choices=["none", "diffusion", "geodesic_kde"], default="diffusion")
    parser.add_argument("--smoothing-steps", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.6)
    parser.add_argument("--geodesic-kde-sigma-scale", type=float, default=3.0)
    parser.add_argument("--geodesic-kde-radius-scale", type=float, default=3.0)
    parser.add_argument("--reference-sigmas-px", nargs="+", type=float, default=[25.0])
    parser.add_argument("--cone-radius-sigma-mult", type=float, default=3.0)
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


def _build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        device=args.device,
        frame_alignment=args.frame_alignment,
        point_weight_mode=args.point_weight_mode,
        ray_batch_size=args.ray_batch_size,
        smoothing_mode=args.smoothing_mode,
        smoothing_steps=args.smoothing_steps,
        smoothing_alpha=args.smoothing_alpha,
        geodesic_kde_sigma_scale=args.geodesic_kde_sigma_scale,
        geodesic_kde_radius_scale=args.geodesic_kde_radius_scale,
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


def _summary_row(summary: dict, method_label: str, sigma_px: float | None = None) -> dict:
    metrics_path = Path(summary["output_dir"]) / "metrics_vs_gt.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    aggregate = metrics_payload.get("aggregate_sum", {})
    runtime = summary.get("runtime_config", {})
    return {
        "method": method_label,
        "sigma_px": sigma_px,
        "frame_alignment": runtime.get("frame_alignment"),
        "point_weight_mode": runtime.get("point_weight_mode"),
        "smoothing_mode": runtime.get("smoothing_mode"),
        "smoothing_steps": runtime.get("smoothing_steps"),
        "smoothing_alpha": runtime.get("smoothing_alpha"),
        "geodesic_kde_sigma_scale": runtime.get("geodesic_kde_sigma_scale"),
        "geodesic_kde_radius_scale": runtime.get("geodesic_kde_radius_scale"),
        "hit_rate": summary.get("global_hit_rate"),
        "assignment_rate": summary.get("global_assignment_rate", summary.get("global_hit_rate")),
        "CC": aggregate.get("CC"),
        "SIM": aggregate.get("SIM"),
        "KLD": aggregate.get("KLD"),
        "MSE": aggregate.get("MSE"),
        "Spearman": aggregate.get("Spearman"),
        "Cosine": aggregate.get("Cosine"),
        "MeshMamba_CC": aggregate.get("MeshMamba_CC"),
        "MeshMamba_SIM": aggregate.get("MeshMamba_SIM"),
        "MeshMamba_KLD": aggregate.get("MeshMamba_KLD"),
        "MeshMamba_MSE_raw": aggregate.get("MeshMamba_MSE_raw"),
        "output_dir": summary.get("output_dir"),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_paths_base = DatasetPaths(
        gaze_csv_dir=Path(args.gaze_csv_dir),
        mesh_dir=Path(args.mesh_dir),
        json_dir=Path(args.json_dir),
        gt_dir=None if not args.gt_dir else Path(args.gt_dir),
        output_dir=Path(args.output_root),
        mapping_json=Path(args.output_root) / "mapping.json",
    )
    runtime_config = _build_runtime_config(args)
    output_root = ensure_dir(Path(args.output_root) / args.model)
    rows: list[dict] = []

    if "our_pipeline" in args.methods:
        our_output_root = ensure_dir(output_root / "our_pipeline")
        projector = MeshMambaFaceProjector(
            dataset_paths=DatasetPaths(
                gaze_csv_dir=dataset_paths_base.gaze_csv_dir,
                mesh_dir=dataset_paths_base.mesh_dir,
                json_dir=dataset_paths_base.json_dir,
                gt_dir=dataset_paths_base.gt_dir,
                output_dir=our_output_root,
                mapping_json=our_output_root / "mapping.json",
            ),
            runtime_config=runtime_config,
        )
        summary = projector.run_model(args.model)
        rows.append(_summary_row(summary, method_label="our_pipeline"))

    for sigma_px in args.reference_sigmas_px:
        for method in args.methods:
            if method == "our_pipeline":
                continue
            method_tag = f"{method}__sigma_px-{str(sigma_px).replace('.', 'p')}"
            method_output_root = ensure_dir(output_root / method_tag)
            ref_projector = MeshMambaReferenceProjector(
                dataset_paths=DatasetPaths(
                    gaze_csv_dir=dataset_paths_base.gaze_csv_dir,
                    mesh_dir=dataset_paths_base.mesh_dir,
                    json_dir=dataset_paths_base.json_dir,
                    gt_dir=dataset_paths_base.gt_dir,
                    output_dir=method_output_root,
                    mapping_json=method_output_root / "mapping.json",
                ),
                runtime_config=runtime_config,
                method_config=ReferenceMethodConfig(
                    method=method,
                    sigma_px=float(sigma_px),
                    radius_sigma_mult=args.cone_radius_sigma_mult,
                ),
            )
            summary = ref_projector.run_model(args.model)
            rows.append(_summary_row(summary, method_label=method, sigma_px=float(sigma_px)))

    comparison_df = pd.DataFrame(rows)
    comparison_csv = output_root / "comparison_summary.csv"
    comparison_json = output_root / "comparison_summary.json"
    comparison_df.to_csv(comparison_csv, index=False)
    write_json(
        comparison_json,
        {
            "model": args.model,
            "rows": comparison_df.to_dict(orient="records"),
        },
    )
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved comparison JSON: {comparison_json}")
    if not comparison_df.empty:
        print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
