#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe one MeshMamba non_mvp_data JSON file.")
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text())
    print(f"json_path: {args.json_path}")
    print("top_level_keys:", list(payload.keys()))
    print()

    print("model_name:", payload.get("model_name"))
    print("file_version:", payload.get("file_version"))
    print("generated_at:", payload.get("generated_at"))
    print()

    video_info = payload.get("video_info", {})
    camera_static = payload.get("camera_static", {})
    model_static = payload.get("model_static", {})
    animation = payload.get("animation", {})
    frames = payload.get("frames", [])

    print("video_info:", video_info)
    print()
    print("camera_static keys:", list(camera_static.keys()))
    print("camera_location:", camera_static.get("location"))
    print("camera_fov_degrees:", camera_static.get("fov_degrees"))
    print("camera_clip:", camera_static.get("clip_start"), camera_static.get("clip_end"))
    print()
    print("model_static:", model_static)
    print()
    print("animation:", animation)
    print()
    print("frames_count:", len(frames))
    if frames:
        print("first_frame:", frames[0])
        print("last_frame:", frames[-1])


if __name__ == "__main__":
    main()
