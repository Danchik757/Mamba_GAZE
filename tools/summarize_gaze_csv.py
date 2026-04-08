#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize participant gaze sample counts for one model CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--top", type=int, default=10, help="How many highest-count participants to print.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    rows = []
    for _, row in df.iterrows():
        gaze = ast.literal_eval(row["data_gazes"])
        times = gaze["t"]
        count = len(times)
        duration = float(times[-1]) if times else 0.0
        rows.append(
            {
                "participation_id": int(row["participation_id"]),
                "samples": count,
                "duration_seconds_last_timestamp": duration,
            }
        )

    stats_df = pd.DataFrame(rows)
    sample_values = stats_df["samples"].to_numpy(dtype=float)

    print(f"csv_path: {args.csv_path}")
    print(f"participants: {len(stats_df)}")
    print(f"samples_total: {int(sample_values.sum())}")
    print(f"samples_mean: {sample_values.mean():.3f}")
    print(f"samples_median: {np.median(sample_values):.3f}")
    print(f"samples_min: {int(sample_values.min())}")
    print(f"samples_max: {int(sample_values.max())}")
    print(
        "samples_quantiles:",
        {
            "q10": float(np.quantile(sample_values, 0.10)),
            "q25": float(np.quantile(sample_values, 0.25)),
            "q50": float(np.quantile(sample_values, 0.50)),
            "q75": float(np.quantile(sample_values, 0.75)),
            "q90": float(np.quantile(sample_values, 0.90)),
        },
    )
    print()
    print(f"top_{args.top}_participants_by_samples:")
    print(
        stats_df.sort_values("samples", ascending=False)
        .head(args.top)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
