#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "hit_rate",
    "CC",
    "SIM",
    "KLD",
    "MSE",
    "Spearman",
    "Cosine",
    "MeshMamba_CC",
    "MeshMamba_SIM",
    "MeshMamba_KLD",
    "MeshMamba_MSE_raw",
]

CONFIG_COLUMNS = [
    "frame_alignment",
    "point_weight_mode",
    "smoothing_mode",
    "smoothing_steps",
    "smoothing_alpha",
    "geodesic_kde_sigma_scale",
    "geodesic_kde_radius_scale",
    "extra_rotate_x_deg",
    "recenter_to_bbox_center",
    "override_fov_deg",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate leaderboard CSVs across multiple model sweeps.")
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--status-allow", nargs="+", default=["completed", "skipped_existing"])
    args = parser.parse_args()

    sweeps_root = args.sweeps_root.resolve()
    output_dir = (args.output_dir or sweeps_root / "_aggregate").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_paths = sorted(sweeps_root.glob("*/*_leaderboard.csv"))
    if not leaderboard_paths:
        raise FileNotFoundError(f"No leaderboard CSV files found under {sweeps_root}")

    frames: list[pd.DataFrame] = []
    for path in leaderboard_paths:
        model_name = path.stem.removesuffix("_leaderboard")
        df = pd.read_csv(path)
        df["model_name"] = model_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["status"].isin(args.status_allow)].copy()
    combined_path = output_dir / "combined_runs.csv"
    combined.to_csv(combined_path, index=False)

    available_metric_columns = [column for column in METRIC_COLUMNS if column in combined.columns]
    grouped = combined.groupby(CONFIG_COLUMNS, dropna=False)
    summary = grouped[available_metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["models_count"] = grouped["model_name"].nunique()
    summary["runs_count"] = grouped.size()
    summary = summary.reset_index()

    aggregate_path = output_dir / "aggregate_by_config.csv"
    summary.to_csv(aggregate_path, index=False)

    ranking_columns = [
        column
        for column in [
            "MeshMamba_KLD_mean",
            "MeshMamba_SIM_mean",
            "KLD_mean",
            "SIM_mean",
            "CC_mean",
            "hit_rate_mean",
        ]
        if column in summary.columns
    ]
    if ranking_columns:
        sorted_summary = summary.sort_values(
            by=ranking_columns,
            ascending=[True, False, True, False, False, False][: len(ranking_columns)],
        )
    else:
        sorted_summary = summary

    top_path = output_dir / "top_configs.csv"
    sorted_summary.head(50).to_csv(top_path, index=False)

    print(f"combined_runs: {combined_path}")
    print(f"aggregate_by_config: {aggregate_path}")
    print(f"top_configs: {top_path}")
    print(f"models_covered: {combined['model_name'].nunique()}")
    print(f"rows_combined: {len(combined)}")


if __name__ == "__main__":
    main()
