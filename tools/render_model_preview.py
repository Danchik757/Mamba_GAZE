#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_gaze.io_utils import ensure_dir, load_json, load_obj


def rotate_vertices_z(vertices: np.ndarray, angle_radians: float) -> np.ndarray:
    cos_angle = float(np.cos(angle_radians))
    sin_angle = float(np.sin(angle_radians))
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    rotated_x = cos_angle * x - sin_angle * y
    rotated_y = sin_angle * x + cos_angle * y
    return np.stack([rotated_x, rotated_y, z], axis=1)


def rotate_vertices_x(vertices: np.ndarray, angle_radians: float) -> np.ndarray:
    cos_angle = float(np.cos(angle_radians))
    sin_angle = float(np.sin(angle_radians))
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    rotated_y = cos_angle * y - sin_angle * z
    rotated_z = sin_angle * y + cos_angle * z
    return np.stack([x, rotated_y, rotated_z], axis=1)


def write_ppm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    ensure_dir(path.parent)
    height, width, _ = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def edge_function(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    px: np.ndarray,
    py: np.ndarray,
) -> np.ndarray:
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def render_preview(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    camera_origin: np.ndarray,
    width: int,
    height: int,
    background_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    vertex_count = vertices_world.shape[0]
    world_h = np.concatenate([vertices_world, np.ones((vertex_count, 1), dtype=np.float64)], axis=1)
    clip_h = (projection_matrix @ (view_matrix @ world_h.T)).T

    clip_w = clip_h[:, 3]
    valid_vertices = np.abs(clip_w) > 1e-8
    ndc = np.full((vertex_count, 3), np.nan, dtype=np.float64)
    ndc[valid_vertices] = clip_h[valid_vertices, :3] / clip_w[valid_vertices, None]

    screen_x = (ndc[:, 0] + 1.0) * 0.5 * (width - 1)
    screen_y = (1.0 - ndc[:, 1]) * 0.5 * (height - 1)
    screen_xy = np.stack([screen_x, screen_y], axis=1)
    screen_z = ndc[:, 2]

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = background_rgb.reshape(1, 1, 3)
    z_buffer = np.full((height, width), np.inf, dtype=np.float64)

    face_vertices_world = vertices_world[faces]
    edge1 = face_vertices_world[:, 1, :] - face_vertices_world[:, 0, :]
    edge2 = face_vertices_world[:, 2, :] - face_vertices_world[:, 0, :]
    face_normals = np.cross(edge1, edge2)
    face_norm_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.clip(face_norm_lengths, 1e-12, None)
    face_centroids = face_vertices_world.mean(axis=1)
    view_directions = camera_origin.reshape(1, 3) - face_centroids
    view_directions /= np.clip(np.linalg.norm(view_directions, axis=1, keepdims=True), 1e-12, None)
    lighting = np.abs(np.sum(face_normals * view_directions, axis=1))
    intensity = 0.18 + 0.82 * lighting
    face_colors = np.clip((intensity[:, None] * np.array([[225.0, 235.0, 245.0]])).round(), 0, 255).astype(np.uint8)

    drawn_faces = 0
    pixel_updates = 0
    min_draw_x = width
    min_draw_y = height
    max_draw_x = -1
    max_draw_y = -1

    for face_index, face in enumerate(faces):
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        if not (valid_vertices[i0] and valid_vertices[i1] and valid_vertices[i2]):
            continue

        xy = screen_xy[[i0, i1, i2]]
        z = screen_z[[i0, i1, i2]]
        if np.any(~np.isfinite(xy)) or np.any(~np.isfinite(z)):
            continue
        if np.any(z < -1.0) and np.any(z > 1.0):
            continue

        min_x = max(0, int(np.floor(np.min(xy[:, 0]))))
        max_x = min(width - 1, int(np.ceil(np.max(xy[:, 0]))))
        min_y = max(0, int(np.floor(np.min(xy[:, 1]))))
        max_y = min(height - 1, int(np.ceil(np.max(xy[:, 1]))))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = float(xy[0, 0]), float(xy[0, 1])
        x1, y1 = float(xy[1, 0]), float(xy[1, 1])
        x2, y2 = float(xy[2, 0]), float(xy[2, 1])
        area = edge_function(x0, y0, x1, y1, np.array(x2), np.array(y2))
        area_value = float(area)
        if abs(area_value) <= 1e-8:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        px, py = np.meshgrid(xs, ys)

        w0 = edge_function(x1, y1, x2, y2, px, py)
        w1 = edge_function(x2, y2, x0, y0, px, py)
        w2 = edge_function(x0, y0, x1, y1, px, py)
        if area_value > 0.0:
            inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        else:
            inside = (w0 <= 0.0) & (w1 <= 0.0) & (w2 <= 0.0)
        if not np.any(inside):
            continue

        bary0 = w0 / area_value
        bary1 = w1 / area_value
        bary2 = w2 / area_value
        depth = bary0 * z[0] + bary1 * z[1] + bary2 * z[2]
        z_slice = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update_mask = inside & (depth < z_slice)
        if not np.any(update_mask):
            continue

        z_slice[update_mask] = depth[update_mask]
        image_slice = image[min_y : max_y + 1, min_x : max_x + 1]
        image_slice[update_mask] = face_colors[face_index]

        drawn_faces += 1
        pixel_updates += int(update_mask.sum())
        min_draw_x = min(min_draw_x, min_x)
        min_draw_y = min(min_draw_y, min_y)
        max_draw_x = max(max_draw_x, max_x)
        max_draw_y = max(max_draw_y, max_y)

    stats = {
        "drawn_faces": float(drawn_faces),
        "pixel_updates": float(pixel_updates),
        "coverage_ratio": float(pixel_updates / float(width * height)) if width > 0 and height > 0 else 0.0,
        "bbox_min_x": float(min_draw_x if max_draw_x >= 0 else -1),
        "bbox_min_y": float(min_draw_y if max_draw_y >= 0 else -1),
        "bbox_max_x": float(max_draw_x),
        "bbox_max_y": float(max_draw_y),
    }
    return image, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a quick first-frame preview from MeshMamba JSON camera + OBJ.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--resolution-scale", type=float, default=0.5)
    parser.add_argument("--extra-rotate-x-deg", type=float, default=0.0)
    parser.add_argument("--recenter-to-bbox-center", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    mesh_path = args.mesh_dir / args.model / f"{args.model}.obj"
    json_path = args.json_dir / f"MeshMamba_non_texture_{args.model}.json"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"OBJ not found: {mesh_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    vertices, faces = load_obj(mesh_path)
    bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    if args.recenter_to_bbox_center:
        vertices = vertices - bbox_center.reshape(1, 3)
    metadata = load_json(json_path)
    frame_index = int(np.clip(args.frame_index, 0, len(metadata["frames"]) - 1))
    angle = float(metadata["frames"][frame_index]["rotation_z_radians"])
    extra_rotate_x_deg = float(args.extra_rotate_x_deg)
    extra_rotate_x_rad = np.deg2rad(extra_rotate_x_deg)

    scale = np.asarray(metadata["model_static"]["scale"], dtype=np.float64).reshape(1, 3)
    translation = np.asarray(metadata["model_static"]["location"], dtype=np.float64).reshape(1, 3)
    transformed_vertices = rotate_vertices_z(vertices * scale, angle)
    if abs(extra_rotate_x_rad) > 1e-12:
        transformed_vertices = rotate_vertices_x(transformed_vertices, extra_rotate_x_rad)
    vertices_world = transformed_vertices + translation

    view_matrix = np.asarray(metadata["camera_static"]["view_matrix"], dtype=np.float64)
    projection_matrix = np.asarray(metadata["camera_static"]["projection_matrix"], dtype=np.float64)
    inv_view = np.linalg.inv(view_matrix)
    camera_origin_h = inv_view @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    camera_origin = camera_origin_h[:3] / camera_origin_h[3]

    base_width = int(metadata["video_info"]["resolution_width"])
    base_height = int(metadata["video_info"]["resolution_height"])
    scale_factor = max(0.05, float(args.resolution_scale))
    width = max(64, int(round(base_width * scale_factor)))
    height = max(64, int(round(base_height * scale_factor)))

    image, stats = render_preview(
        vertices_world=vertices_world,
        faces=faces,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        camera_origin=camera_origin,
        width=width,
        height=height,
        background_rgb=np.asarray([18, 22, 28], dtype=np.uint8),
    )
    write_ppm(args.output_image, image)

    print(f"Preview saved: {args.output_image}")
    print(f"Model: {args.model}")
    print(f"Frame index: {frame_index}")
    print(f"Angle radians: {angle}")
    print(f"Extra rotate X degrees: {extra_rotate_x_deg}")
    print(f"Image size: {width}x{height}")
    print(f"Camera origin: {camera_origin.tolist()}")
    print(f"Model translation: {translation.reshape(-1).tolist()}")
    print(f"Model scale: {scale.reshape(-1).tolist()}")
    print(f"Recenter to bbox center: {bool(args.recenter_to_bbox_center)}")
    print(f"Raw OBJ bbox center: {bbox_center.astype(float).tolist()}")
    print(f"Coverage ratio: {stats['coverage_ratio']:.6f}")
    print(
        "Projected bbox: "
        f"({int(stats['bbox_min_x'])}, {int(stats['bbox_min_y'])}) - "
        f"({int(stats['bbox_max_x'])}, {int(stats['bbox_max_y'])})"
    )


if __name__ == "__main__":
    main()
