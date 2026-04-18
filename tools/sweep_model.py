#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _fmt_float(value: float) -> str:
    return str(value).replace(".", "p")


def build_run_name(
    frame_alignment: str,
    point_weight_mode: str,
    smoothing_mode: str,
    smoothing_steps: int | None = None,
    smoothing_alpha: float | None = None,
    geodesic_kde_sigma_scale: float | None = None,
    geodesic_kde_radius_scale: float | None = None,
    extra_rotate_x_deg: float = 0.0,
    recenter_to_bbox_center: bool = False,
    override_fov_deg: float | None = None,
) -> str:
    parts = [
        f"fa-{frame_alignment}",
        f"pw-{point_weight_mode}",
        f"sm-{smoothing_mode}",
    ]
    if recenter_to_bbox_center:
        parts.append("bbox-center")
    if abs(float(extra_rotate_x_deg)) > 1e-12:
        parts.append(f"rx-{_fmt_float(float(extra_rotate_x_deg))}")
    if override_fov_deg is not None:
        parts.append(f"fov-{_fmt_float(float(override_fov_deg))}")
    if smoothing_mode == "diffusion":
        parts.append(f"steps-{int(smoothing_steps or 0)}")
        parts.append(f"alpha-{_fmt_float(float(smoothing_alpha or 0.0))}")
    elif smoothing_mode == "geodesic_kde":
        parts.append(f"sigma-{_fmt_float(float(geodesic_kde_sigma_scale or 0.0))}")
        parts.append(f"radius-{_fmt_float(float(geodesic_kde_radius_scale or 0.0))}")
    return "__".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parameter sweep for one model.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gaze-csv-dir", type=Path, default=None)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--json-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--frame-alignments", nargs="+", default=["nearest", "floor"])
    parser.add_argument("--point-weight-modes", nargs="+", default=["unit", "delta_t"])
    parser.add_argument("--smoothing-modes", nargs="+", default=["diffusion"])
    parser.add_argument("--smoothing-steps", nargs="+", type=int, default=[0, 8, 16, 32])
    parser.add_argument("--smoothing-alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--geodesic-kde-sigma-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--geodesic-kde-radius-scales", nargs="+", type=float, default=[3.0])
    parser.add_argument("--extra-rotate-x-deg", type=float, default=0.0)
    parser.add_argument("--recenter-to-bbox-center", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--override-fov-deg", type=float, default=None)
    parser.add_argument("--participant-ids", nargs="+", type=int, default=None)
    parser.add_argument("--ray-batch-size", type=int, default=128)
    parser.add_argument("--precompute-all-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = (args.output_root or repo_root / "sweeps").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combinations: list[dict[str, object]] = []
    for frame_alignment, point_weight_mode, smoothing_mode in itertools.product(
        args.frame_alignments,
        args.point_weight_modes,
        args.smoothing_modes,
    ):
        if smoothing_mode == "diffusion":
            for smoothing_steps, smoothing_alpha in itertools.product(args.smoothing_steps, args.smoothing_alphas):
                combinations.append(
                    {
                        "frame_alignment": frame_alignment,
                        "point_weight_mode": point_weight_mode,
                        "smoothing_mode": smoothing_mode,
                        "smoothing_steps": smoothing_steps,
                        "smoothing_alpha": smoothing_alpha,
                        "geodesic_kde_sigma_scale": None,
                    }
                )
        elif smoothing_mode == "geodesic_kde":
            for sigma_scale, radius_scale in itertools.product(
                args.geodesic_kde_sigma_scales,
                args.geodesic_kde_radius_scales,
            ):
                combinations.append(
                    {
                        "frame_alignment": frame_alignment,
                        "point_weight_mode": point_weight_mode,
                        "smoothing_mode": smoothing_mode,
                        "smoothing_steps": None,
                        "smoothing_alpha": None,
                        "geodesic_kde_sigma_scale": sigma_scale,
                        "geodesic_kde_radius_scale": radius_scale,
                    }
                )
        elif smoothing_mode == "none":
            combinations.append(
                {
                    "frame_alignment": frame_alignment,
                    "point_weight_mode": point_weight_mode,
                    "smoothing_mode": smoothing_mode,
                    "smoothing_steps": None,
                    "smoothing_alpha": None,
                    "geodesic_kde_sigma_scale": None,
                    "geodesic_kde_radius_scale": None,
                }
            )
        else:
            raise ValueError(f"Unsupported smoothing mode: {smoothing_mode}")

    rows = []
    for combo in combinations:
        frame_alignment = str(combo["frame_alignment"])
        point_weight_mode = str(combo["point_weight_mode"])
        smoothing_mode = str(combo["smoothing_mode"])
        smoothing_steps = combo["smoothing_steps"]
        smoothing_alpha = combo["smoothing_alpha"]
        sigma_scale = combo["geodesic_kde_sigma_scale"]
        radius_scale = combo.get("geodesic_kde_radius_scale")

        run_name = build_run_name(
            frame_alignment=frame_alignment,
            point_weight_mode=point_weight_mode,
            smoothing_mode=smoothing_mode,
            smoothing_steps=None if smoothing_steps is None else int(smoothing_steps),
            smoothing_alpha=None if smoothing_alpha is None else float(smoothing_alpha),
            geodesic_kde_sigma_scale=None if sigma_scale is None else float(sigma_scale),
            geodesic_kde_radius_scale=None if radius_scale is None else float(radius_scale),
            extra_rotate_x_deg=float(args.extra_rotate_x_deg),
            recenter_to_bbox_center=bool(args.recenter_to_bbox_center),
            override_fov_deg=None if args.override_fov_deg is None else float(args.override_fov_deg),
        )
        run_root = output_root / run_name
        model_output_dir = run_root / args.model
        metrics_path = model_output_dir / "metrics_vs_gt.json"
        summary_path = model_output_dir / "run_summary.json"

        if args.resume and metrics_path.exists() and summary_path.exists():
            status = "skipped_existing"
        else:
            run_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(repo_root / "run_meshmamba_gaze.py"),
                "--model",
                args.model,
                "--device",
                args.device,
                "--extra-rotate-x-deg",
                str(float(args.extra_rotate_x_deg)),
                "--smoothing-mode",
                smoothing_mode,
                "--frame-alignment",
                frame_alignment,
                "--point-weight-mode",
                point_weight_mode,
                "--ray-batch-size",
                str(args.ray_batch_size),
                "--output-dir",
                str(run_root),
                "--mapping-json",
                str(output_root / "meshmamba_non_texture_name_mapping.json"),
            ]
            if args.override_fov_deg is not None:
                cmd += ["--override-fov-deg", str(float(args.override_fov_deg))]

            if smoothing_mode == "diffusion":
                cmd += ["--smoothing-steps", str(int(smoothing_steps))]
                cmd += ["--smoothing-alpha", str(float(smoothing_alpha))]
            elif smoothing_mode == "geodesic_kde":
                cmd += ["--geodesic-kde-sigma-scale", str(float(sigma_scale))]
                cmd += ["--geodesic-kde-radius-scale", str(float(radius_scale))]

            if args.precompute_all_frames:
                cmd.append("--precompute-all-frames")
            else:
                cmd.append("--no-precompute-all-frames")
            if args.recenter_to_bbox_center:
                cmd.append("--recenter-to-bbox-center")
            else:
                cmd.append("--no-recenter-to-bbox-center")

            if args.gaze_csv_dir is not None:
                cmd += ["--gaze-csv-dir", str(args.gaze_csv_dir)]
            if args.mesh_dir is not None:
                cmd += ["--mesh-dir", str(args.mesh_dir)]
            if args.json_dir is not None:
                cmd += ["--json-dir", str(args.json_dir)]
            if args.gt_dir is not None:
                cmd += ["--gt-dir", str(args.gt_dir)]
            if args.participant_ids is not None:
                cmd += ["--participant-ids", *[str(value) for value in args.participant_ids]]

            print(f"\n=== {run_name} ===")
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                rows.append(
                    {
                        "run_name": run_name,
                        "status": f"failed_{completed.returncode}",
                        "frame_alignment": frame_alignment,
                        "point_weight_mode": point_weight_mode,
                        "smoothing_mode": smoothing_mode,
                        "smoothing_steps": smoothing_steps,
                        "smoothing_alpha": smoothing_alpha,
                        "geodesic_kde_sigma_scale": sigma_scale,
                        "geodesic_kde_radius_scale": radius_scale,
                        "participant_ids": None if args.participant_ids is None else ",".join(map(str, args.participant_ids)),
                        "extra_rotate_x_deg": float(args.extra_rotate_x_deg),
                        "recenter_to_bbox_center": bool(args.recenter_to_bbox_center),
                        "override_fov_deg": None if args.override_fov_deg is None else float(args.override_fov_deg),
                    }
                )
                continue
            status = "completed"

        metrics = json.loads(metrics_path.read_text())
        summary = json.loads(summary_path.read_text())
        agg = metrics["aggregate_sum"]
        rows.append(
            {
                "run_name": run_name,
                "status": status,
                "frame_alignment": frame_alignment,
                "point_weight_mode": point_weight_mode,
                "smoothing_mode": smoothing_mode,
                "smoothing_steps": smoothing_steps,
                "smoothing_alpha": smoothing_alpha,
                "geodesic_kde_sigma_scale": sigma_scale,
                "geodesic_kde_radius_scale": radius_scale if smoothing_mode == "geodesic_kde" else None,
                "participant_ids": None if args.participant_ids is None else ",".join(map(str, args.participant_ids)),
                "extra_rotate_x_deg": float(args.extra_rotate_x_deg),
                "recenter_to_bbox_center": bool(args.recenter_to_bbox_center),
                "override_fov_deg": None if args.override_fov_deg is None else float(args.override_fov_deg),
                "hit_rate": summary["global_hit_rate"],
                "points_used_total": summary["points_used_total"],
                "hits_total": summary["hits_total"],
                "CC": agg["CC"],
                "SIM": agg["SIM"],
                "KLD": agg["KLD"],
                "MSE": agg["MSE"],
                "Spearman": agg["Spearman"],
                "Cosine": agg["Cosine"],
                "output_dir": summary["output_dir"],
            }
        )

    df = pd.DataFrame(rows)
    leaderboard_path = output_root / f"{args.model}_leaderboard.csv"
    df.to_csv(leaderboard_path, index=False)
    print(f"\nleaderboard saved: {leaderboard_path}")

    if not df.empty:
        ranking = (
            df[df["status"].isin(["completed", "skipped_existing"])]
            .sort_values(["CC", "SIM", "KLD", "hit_rate"], ascending=[False, False, True, False])
        )
        print("\nTop 10 runs:")
        print(
            ranking[
                [
                    "run_name",
                    "CC",
                    "SIM",
                    "KLD",
                    "MSE",
                    "hit_rate",
                    "frame_alignment",
                    "point_weight_mode",
                    "smoothing_mode",
                    "smoothing_steps",
                    "smoothing_alpha",
                    "geodesic_kde_sigma_scale",
                    "geodesic_kde_radius_scale",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
