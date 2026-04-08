#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXCLUDE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "outputs_smoke",
    "run_outputs",
    "sweeps",
    ".DS_Store",
    "MAMBA_GAZE.bundle",
}


def copy_repo(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for path in src_root.iterdir():
        if path.name in EXCLUDE_NAMES:
            continue
        target = dst_root / path.name
        if path.is_dir():
            shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        else:
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained workspace for one model.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True, help="Directory that contains <model>/<model>.obj")
    parser.add_argument("--gt-path", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    target_dir = args.target_dir.resolve()
    gaze_root = target_dir.parent
    model = args.model

    copy_repo(repo_root, target_dir)

    csv_target = target_dir / "data" / "csv_for_models" / "MeshMamba_non_texture"
    json_target = target_dir / "data" / "jsons_for_models" / "Mamba_non_textured"
    mesh_target = target_dir / "data" / "datasets" / "MeshMamba" / "MeshMambaSaliency" / "MeshFile" / "non_texture" / model
    gt_target = target_dir / "data" / "datasets" / "MeshMamba" / "MeshMambaSaliency" / "SaliencyMap" / "non_texture"

    csv_target.mkdir(parents=True, exist_ok=True)
    json_target.mkdir(parents=True, exist_ok=True)
    mesh_target.mkdir(parents=True, exist_ok=True)
    gt_target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.csv_path, csv_target / args.csv_path.name)
    shutil.copy2(args.json_path, json_target / args.json_path.name)
    shutil.copy2(args.gt_path, gt_target / args.gt_path.name)

    source_mesh_dir = args.mesh_dir / model
    if not source_mesh_dir.is_dir():
        raise FileNotFoundError(f"Model mesh directory not found: {source_mesh_dir}")

    for path in source_mesh_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, mesh_target / path.name)

    config_path = target_dir / "configs" / "test1_local.env"
    config_path.write_text(
        "\n".join(
            [
                f'REPO_ROOT="{target_dir}"',
                f'ENV_ROOT="{gaze_root / "environments"}"',
                'ENV_NAME="meshmamba_gaze"',
                'ENV_PATH="${ENV_ROOT}/${ENV_NAME}"',
                'RENDER_ROOT=""',
                'VIDEOS_DIR=""',
                'JSON_DIR="${REPO_ROOT}/data/jsons_for_models/Mamba_non_textured"',
                'MESH_DIR="${REPO_ROOT}/data/datasets/MeshMamba/MeshMambaSaliency/MeshFile/non_texture"',
                'GT_DIR="${REPO_ROOT}/data/datasets/MeshMamba/MeshMambaSaliency/SaliencyMap/non_texture"',
                'GAZE_CSV_DIR="${REPO_ROOT}/data/csv_for_models/MeshMamba_non_texture"',
                'OUTPUT_DIR="${REPO_ROOT}/run_outputs"',
                'TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"self-contained workspace created: {target_dir}")
    print(f"config: {config_path}")
    print("next steps:")
    print(f"  cd {target_dir}")
    print("  CONFIG_PATH=configs/test1_local.env bash scripts/create_conda_env.sh")
    print(
        f"  CONFIG_PATH=configs/test1_local.env bash scripts/run_model_server.sh {model} "
        "--device cpu --precompute-all-frames"
    )


if __name__ == "__main__":
    main()
