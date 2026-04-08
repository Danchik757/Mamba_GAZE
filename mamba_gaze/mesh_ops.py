from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class FaceAdjacency:
    src: torch.Tensor
    dst: torch.Tensor
    degree: torch.Tensor
    edge_lengths: torch.Tensor
    mean_edge_length: float
    neighbors: tuple[tuple[int, ...], ...]
    neighbor_lengths: tuple[tuple[float, ...], ...]


def build_face_adjacency(vertices: np.ndarray, faces: np.ndarray, device: torch.device) -> FaceAdjacency:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edges = (
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((c, a))),
        )
        for edge in edges:
            edge_to_faces[edge].append(face_index)

    pairs: set[tuple[int, int]] = set()
    for face_indices in edge_to_faces.values():
        if len(face_indices) < 2:
            continue
        for idx in range(len(face_indices)):
            for jdx in range(idx + 1, len(face_indices)):
                first = face_indices[idx]
                second = face_indices[jdx]
                if first == second:
                    continue
                pairs.add((min(first, second), max(first, second)))

    if not pairs:
        num_faces = int(faces.shape[0])
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        degree = torch.zeros(num_faces, dtype=torch.float32, device=device)
        neighbors = tuple(() for _ in range(num_faces))
        neighbor_lengths = tuple(() for _ in range(num_faces))
        return FaceAdjacency(
            src=empty_long,
            dst=empty_long,
            degree=degree,
            edge_lengths=empty_float,
            mean_edge_length=0.0,
            neighbors=neighbors,
            neighbor_lengths=neighbor_lengths,
        )

    centroids = vertices[faces].mean(axis=1).astype(np.float64)
    sorted_pairs = sorted(pairs)
    pair_lengths: list[float] = []
    neighbors: list[list[int]] = [[] for _ in range(int(faces.shape[0]))]
    neighbor_lengths: list[list[float]] = [[] for _ in range(int(faces.shape[0]))]

    for first, second in sorted_pairs:
        distance = float(np.linalg.norm(centroids[first] - centroids[second]))
        pair_lengths.append(distance)
        neighbors[first].append(second)
        neighbors[second].append(first)
        neighbor_lengths[first].append(distance)
        neighbor_lengths[second].append(distance)

    src = torch.tensor([pair[0] for pair in sorted_pairs], dtype=torch.long, device=device)
    dst = torch.tensor([pair[1] for pair in sorted_pairs], dtype=torch.long, device=device)
    edge_lengths = torch.tensor(pair_lengths, dtype=torch.float32, device=device)
    degree = torch.zeros(int(faces.shape[0]), dtype=torch.float32, device=device)
    ones = torch.ones(src.shape[0], dtype=torch.float32, device=device)
    degree.index_add_(0, src, ones)
    degree.index_add_(0, dst, ones)
    mean_edge_length = float(np.mean(pair_lengths)) if pair_lengths else 0.0
    return FaceAdjacency(
        src=src,
        dst=dst,
        degree=degree,
        edge_lengths=edge_lengths,
        mean_edge_length=mean_edge_length,
        neighbors=tuple(tuple(item) for item in neighbors),
        neighbor_lengths=tuple(tuple(item) for item in neighbor_lengths),
    )


def frame_indices_from_timestamps(
    timestamps: np.ndarray,
    frame_timestamps: np.ndarray,
    mode: Literal["nearest", "floor"] = "nearest",
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float32)
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float32)

    if mode == "floor":
        indices = np.searchsorted(frame_timestamps, timestamps, side="right") - 1
        return np.clip(indices, 0, len(frame_timestamps) - 1)

    right_indices = np.searchsorted(frame_timestamps, timestamps, side="left")
    right_indices = np.clip(right_indices, 0, len(frame_timestamps) - 1)
    left_indices = np.clip(right_indices - 1, 0, len(frame_timestamps) - 1)

    left_distance = np.abs(frame_timestamps[left_indices] - timestamps)
    right_distance = np.abs(frame_timestamps[right_indices] - timestamps)
    choose_right = right_distance < left_distance
    return np.where(choose_right, right_indices, left_indices)


def compute_point_weights(
    timestamps: np.ndarray,
    video_duration_seconds: float,
    mode: Literal["unit", "delta_t"] = "unit",
    max_delta_t_seconds: float = 0.1,
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float32)
    if timestamps.size == 0:
        return np.asarray([], dtype=np.float32)

    if mode == "unit":
        return np.ones_like(timestamps, dtype=np.float32)

    if timestamps.size == 1:
        return np.ones_like(timestamps, dtype=np.float32)

    deltas = np.diff(timestamps)
    median_delta = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 1.0 / 30.0
    last_delta = min(max_delta_t_seconds, max(1e-3, video_duration_seconds - float(timestamps[-1])))
    deltas = np.concatenate([deltas, [last_delta if last_delta > 0 else median_delta]])
    deltas = np.clip(deltas, 1e-3, max_delta_t_seconds)
    return deltas.astype(np.float32)


def build_camera_rays(
    points_xy: torch.Tensor,
    inv_projection: torch.Tensor,
    inv_view: torch.Tensor,
    camera_origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = points_xy.device
    dtype = points_xy.dtype
    count = points_xy.shape[0]
    ones = torch.ones((count, 1), dtype=dtype, device=device)

    ndc_x = points_xy[:, 0:1] * 2.0 - 1.0
    ndc_y = 1.0 - points_xy[:, 1:2] * 2.0
    ndc_far = torch.cat([ndc_x, ndc_y, ones, ones], dim=1)

    camera_far = (inv_projection @ ndc_far.T).T
    camera_far = camera_far / camera_far[:, 3:4]

    world_far = (inv_view @ camera_far.T).T
    world_far = world_far[:, :3] / world_far[:, 3:4]

    origins = camera_origin.unsqueeze(0).expand(count, -1)
    directions = world_far - origins
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return origins, directions


def intersect_rays_with_triangles(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    triangle_vertices: torch.Tensor,
    ray_batch_size: int = 64,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = triangle_vertices.device
    dtype = triangle_vertices.dtype

    num_rays = int(ray_origins.shape[0])
    hit_faces = torch.full((num_rays,), -1, dtype=torch.long, device=device)
    hit_distance = torch.full((num_rays,), torch.inf, dtype=dtype, device=device)

    if num_rays == 0:
        return hit_faces, hit_distance

    v0 = triangle_vertices[:, 0, :]
    v1 = triangle_vertices[:, 1, :]
    v2 = triangle_vertices[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0

    for start in range(0, num_rays, max(1, ray_batch_size)):
        end = min(start + max(1, ray_batch_size), num_rays)
        origin_batch = ray_origins[start:end].unsqueeze(1)
        direction_batch = ray_directions[start:end].unsqueeze(1)

        h = torch.cross(direction_batch, edge2.unsqueeze(0), dim=-1)
        a = (edge1.unsqueeze(0) * h).sum(dim=-1)
        parallel_mask = torch.abs(a) <= epsilon

        inv_a = torch.zeros_like(a)
        inv_a[~parallel_mask] = 1.0 / a[~parallel_mask]

        s = origin_batch - v0.unsqueeze(0)
        u = inv_a * (s * h).sum(dim=-1)
        q = torch.cross(s, edge1.unsqueeze(0), dim=-1)
        v = inv_a * (direction_batch * q).sum(dim=-1)
        t = inv_a * (edge2.unsqueeze(0) * q).sum(dim=-1)

        valid = (
            (~parallel_mask)
            & (u >= 0.0)
            & (v >= 0.0)
            & ((u + v) <= 1.0)
            & (t > epsilon)
        )

        masked_t = torch.where(valid, t, torch.full_like(t, torch.inf))
        local_distance, local_face = masked_t.min(dim=1)
        local_hit_mask = torch.isfinite(local_distance)

        face_segment = hit_faces[start:end]
        distance_segment = hit_distance[start:end]
        face_segment[local_hit_mask] = local_face[local_hit_mask]
        distance_segment[local_hit_mask] = local_distance[local_hit_mask]

    return hit_faces, hit_distance


def diffuse_face_values(
    face_values: torch.Tensor,
    adjacency: FaceAdjacency,
    steps: int,
    alpha: float,
) -> torch.Tensor:
    if steps <= 0 or adjacency.src.numel() == 0:
        return face_values.clone()

    current = face_values.clone()
    for _ in range(steps):
        neighbor_sum = torch.zeros_like(current)
        neighbor_sum.index_add_(0, adjacency.src, current[adjacency.dst])
        neighbor_sum.index_add_(0, adjacency.dst, current[adjacency.src])
        neighbor_avg = neighbor_sum / adjacency.degree.clamp_min(1.0)
        current = (1.0 - alpha) * current + alpha * neighbor_avg
    return current


def _truncated_geodesic_distances(
    source_face: int,
    adjacency: FaceAdjacency,
    max_distance: float,
) -> dict[int, float]:
    queue: list[tuple[float, int]] = [(0.0, int(source_face))]
    best = {int(source_face): 0.0}

    while queue:
        current_distance, current_face = heapq.heappop(queue)
        if current_distance > max_distance:
            break
        if current_distance > best.get(current_face, math.inf):
            continue

        neighbor_ids = adjacency.neighbors[current_face]
        neighbor_costs = adjacency.neighbor_lengths[current_face]
        for neighbor_face, edge_length in zip(neighbor_ids, neighbor_costs):
            next_distance = current_distance + edge_length
            if next_distance > max_distance:
                continue
            if next_distance >= best.get(neighbor_face, math.inf):
                continue
            best[neighbor_face] = next_distance
            heapq.heappush(queue, (next_distance, neighbor_face))

    return best


def geodesic_gaussian_face_kde(
    face_values: torch.Tensor,
    adjacency: FaceAdjacency,
    sigma: float,
    radius_scale: float = 3.0,
) -> torch.Tensor:
    if sigma <= 0.0 or adjacency.mean_edge_length <= 0.0:
        return face_values.clone()

    face_values_np = face_values.detach().cpu().numpy().astype(np.float64, copy=False)
    source_faces = np.flatnonzero(face_values_np > 0.0)
    if source_faces.size == 0:
        return face_values.clone()

    max_distance = float(max(0.0, radius_scale) * sigma)
    if max_distance <= 0.0:
        max_distance = sigma

    sigma_sq_inv = 1.0 / max(sigma * sigma, 1e-12)
    smoothed = np.zeros_like(face_values_np, dtype=np.float64)

    for source_face in source_faces.tolist():
        source_weight = float(face_values_np[source_face])
        if source_weight <= 0.0:
            continue

        visited = _truncated_geodesic_distances(source_face=source_face, adjacency=adjacency, max_distance=max_distance)
        if not visited:
            smoothed[source_face] += source_weight
            continue

        visited_faces = np.fromiter(visited.keys(), dtype=np.int64)
        visited_distances = np.fromiter(visited.values(), dtype=np.float64)
        kernel = np.exp(-0.5 * visited_distances * visited_distances * sigma_sq_inv)
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 0.0:
            smoothed[source_face] += source_weight
            continue

        smoothed[visited_faces] += source_weight * (kernel / kernel_sum)

    return torch.tensor(smoothed, dtype=face_values.dtype, device=face_values.device)


def normalize_sum_tensor(values: torch.Tensor) -> torch.Tensor:
    total = values.sum()
    if float(total) <= 0.0:
        return values.clone()
    return values / total


def normalize_sum_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return values.copy()
    return values / total


def normalize_minmax_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)
