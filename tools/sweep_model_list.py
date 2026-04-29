#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _read_model_names(model_list: Path) -> list[str]:
    models = []
    for line in model_list.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        models.append(stripped)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the same sweep configuration for multiple models.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-list", type=Path, required=True, help="Text file with one model name per line.")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent directory for all model sweeps.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args, sweep_args = parser.parse_known_args()
    if sweep_args and sweep_args[0] == "--":
        sweep_args = sweep_args[1:]

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    models = _read_model_names(args.model_list.resolve())
    if not models:
        raise ValueError(f"No model names found in {args.model_list}")

    overall_failures: list[tuple[str, int]] = []
    for index, model_name in enumerate(models, start=1):
        model_output_root = output_root / model_name
        model_output_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "sweep_model.py"),
            "--repo-root",
            str(repo_root),
            "--model",
            model_name,
            "--output-root",
            str(model_output_root),
        ]
        if args.resume:
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")
        if sweep_args:
            cmd.extend(sweep_args)

        print(f"\n===== [{index}/{len(models)}] {model_name} =====")
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            overall_failures.append((model_name, completed.returncode))

    print("\n===== Batch summary =====")
    print(f"models_total={len(models)}")
    print(f"models_failed={len(overall_failures)}")
    if overall_failures:
        for model_name, return_code in overall_failures:
            print(f"failed: {model_name} return_code={return_code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
