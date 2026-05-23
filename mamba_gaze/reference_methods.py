from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .camera_utils import build_projection_matrix_from_fov_degrees
from .io_utils import (
    ensure_dir,
    load_json,
    load_obj,
    load_vector_csv,
    parse_gaze_row,
    read_gaze_dataframe,
    write_json,
    write_vector_csv,
)
from .mesh_ops import (
    compute_point_weights,
    frame_indices_from_timestamps,
    normalize_minmax_np,
    normalize_sum_np,
    normalize_sum_tensor,
)
from .metrics import compute_metrics
from .name_mapping import build_dataset_mapping, resolve_model_from_mapping
from .pipeline import DatasetPaths, FrameVertexCache, RuntimeConfig


@dataclass
class ReferenceMethodConfig:
    method: str = "screen_space_gaussian"
    sigma_px: float = 25.0
    radius_sigma_mult: float = 3.0


class FrameFaceProjectionCache:
    def __init__(
        self,
        vertex_cache: FrameVertexCache,
        faces_t: torch.Tensor,
        view_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        camera_origin: torch.Tensor,
        screen_width: int,
        screen_height: int,
    ) -> None:
        self.vertex_cache = vertex_cache
        self.faces_t = faces_t
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix
        self.camera_origin = camera_origin
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_index = int(frame_index)
        if frame_index not in self._cache:
            self._cache[frame_index] = self._project_frame(frame_index)
        return self._cache[frame_index]

    def _project_frame(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_vertices = self.vertex_cache.get(frame_index)
        face_vertices = frame_vertices[self.faces_t]
        centroids = face_vertices.mean(dim=1)

        edge1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
        edge2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
        normals = torch.cross(edge1, edge2, dim=1)

        to_camera = self.camera_origin.view(1, 3) - centroids
        front_facing = (normals * to_camera).sum(dim=1) > 0.0

        points_h = torch.cat(
            [centroids, torch.ones((centroids.shape[0], 1), dtype=centroids.dtype, device=centroids.device)],
            dim=1,
        )
        camera_points = (self.view_matrix @ points_h.T).T
        clip_points = (self.projection_matrix @ camera_points.T).T
        w = clip_points[:, 3]
        safe_w = torch.where(torch.abs(w) > 1e-8, w, torch.ones_like(w))
        ndc = clip_points[:, :3] / safe_w.unsqueeze(1)

        screen_x = (ndc[:, 0] + 1.0) * 0.5
        screen_y = (1.0 - ndc[:, 1]) * 0.5
        screen_xy = torch.stack(
            [
                screen_x * float(max(1, self.screen_width - 1)),
                screen_y * float(max(1, self.screen_height - 1)),
            ],
            dim=1,
        )

        visible = (
            front_facing
            & torch.isfinite(screen_xy).all(dim=1)
            & torch.isfinite(ndc).all(dim=1)
            & (camera_points[:, 2] < -1e-6)
            & (ndc[:, 2] >= -1.0)
            & (ndc[:, 2] <= 1.0)
            & (screen_x >= 0.0)
            & (screen_x <= 1.0)
            & (screen_y >= 0.0)
            & (screen_y <= 1.0)
        )
        visible_idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
        visible_xy = screen_xy[visible_idx]
        return visible_idx, visible_xy


class MeshMambaReferenceProjector:
    def __init__(
        self,
        dataset_paths: DatasetPaths,
        runtime_config: RuntimeConfig,
        method_config: ReferenceMethodConfig,
    ) -> None:
        self.paths = dataset_paths
        self.config = runtime_config
        self.method_config = method_config

    def run_model(self, model_name: str) -> dict[str, Any]:
        started_at = time.time()
        ensure_dir(self.paths.output_dir)
        mapping = build_dataset_mapping(
            gaze_csv_dir=self.paths.gaze_csv_dir,
            mesh_dir=self.paths.mesh_dir,
            json_dir=self.paths.json_dir,
            gt_dir=self.paths.gt_dir,
            output_path=self.paths.mapping_json,
        )
        resolved = resolve_model_from_mapping(model_name, mapping)

        print(f"Method: {self.method_config.method}")
        print(f"Model:  {resolved.requested_model}")
        print(f"  CSV:  {resolved.csv_path}")
        print(f"  OBJ:  {resolved.mesh_path}")
        print(f"  JSON: {resolved.json_path}")
        print(f"  GT:   {resolved.gt_path if resolved.gt_path is not None else 'not resolved'}")

        output_dir = ensure_dir(self.paths.output_dir / resolved.csv_model_name)
        write_json(output_dir / "resolved_paths.json", resolved.to_jsonable())

        vertices_np, faces_np = load_obj(resolved.mesh_path)
        bbox_center_np = (vertices_np.min(axis=0) + vertices_np.max(axis=0)) * 0.5
        if self.config.recenter_to_bbox_center:
            vertices_np = vertices_np - bbox_center_np.reshape(1, 3)

        ground_truth = None if resolved.gt_path is None else load_vector_csv(resolved.gt_path)
        if ground_truth is not None and ground_truth.shape[0] != faces_np.shape[0]:
            raise ValueError(
                f"Ground truth length {ground_truth.shape[0]} does not match face count {faces_np.shape[0]}"
            )

        metadata = load_json(resolved.json_path)
        gaze_df = read_gaze_dataframe(resolved.csv_path)
        if self.config.participant_ids:
            participant_ids = {int(value) for value in self.config.participant_ids}
            gaze_df = gaze_df[gaze_df["participation_id"].astype(int).isin(participant_ids)].copy()
            if gaze_df.empty:
                raise ValueError(
                    f"No rows found for participant_ids={sorted(participant_ids)} in {resolved.csv_path}"
                )
        if self.config.max_participants is not None:
            gaze_df = gaze_df.head(self.config.max_participants).copy()

        device = self._resolve_device(self.config.device)
        print(f"Device: {device}")

        frame_timestamps = np.asarray([frame["timestamp"] for frame in metadata["frames"]], dtype=np.float32)
        frame_angles = torch.tensor(
            [frame["rotation_z_radians"] for frame in metadata["frames"]],
            dtype=torch.float32,
            device=device,
        )
        vertices_t = torch.tensor(vertices_np, dtype=torch.float32, device=device)
        faces_t = torch.tensor(faces_np, dtype=torch.long, device=device)

        model_scale = metadata["model_static"]["scale"]
        scale_t = torch.tensor(model_scale, dtype=torch.float32, device=device)
        translation_t = torch.tensor(metadata["model_static"]["location"], dtype=torch.float32, device=device)
        vertex_cache = FrameVertexCache(
            base_vertices=vertices_t,
            frame_angles_radians=frame_angles,
            scale=scale_t,
            translation=translation_t,
            precompute_all=self.config.precompute_all_frames,
            extra_rotate_x_radians=math.radians(float(self.config.extra_rotate_x_deg)),
        )

        view_matrix = torch.tensor(metadata["camera_static"]["view_matrix"], dtype=torch.float32, device=device)
        effective_fov_deg = (
            float(self.config.override_fov_deg)
            if self.config.override_fov_deg is not None
            else float(metadata["camera_static"]["fov_degrees"])
        )
        if self.config.override_fov_deg is None:
            projection_matrix = torch.tensor(
                metadata["camera_static"]["projection_matrix"],
                dtype=torch.float32,
                device=device,
            )
        else:
            projection_matrix_np = build_projection_matrix_from_fov_degrees(
                fov_degrees=effective_fov_deg,
                aspect_ratio=float(metadata["video_info"]["aspect_ratio"]),
                clip_start=float(metadata["camera_static"]["clip_start"]),
                clip_end=float(metadata["camera_static"]["clip_end"]),
            )
            projection_matrix = torch.tensor(projection_matrix_np, dtype=torch.float32, device=device)
        inv_view = torch.inverse(view_matrix)
        camera_origin_h = inv_view @ torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        camera_origin = camera_origin_h[:3] / camera_origin_h[3]
        face_projection_cache = FrameFaceProjectionCache(
            vertex_cache=vertex_cache,
            faces_t=faces_t,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix,
            camera_origin=camera_origin,
            screen_width=int(metadata["video_info"]["resolution_width"]),
            screen_height=int(metadata["video_info"]["resolution_height"]),
        )

        participant_summaries: list[dict[str, Any]] = []
        participant_maps_for_aggregation: list[np.ndarray] = []

        with torch.no_grad():
            for participant_order, (_, row) in enumerate(gaze_df.iterrows(), start=1):
                participant_id = int(row["participation_id"])
                print(
                    f"  Participant {participant_order}/{len(gaze_df)} "
                    f"(participation_id={participant_id})"
                )
                result = self._project_single_participant(
                    row=row,
                    participant_order=participant_order,
                    frame_timestamps=frame_timestamps,
                    face_projection_cache=face_projection_cache,
                    video_duration_seconds=float(metadata["video_info"]["duration_seconds"]),
                )
                participant_summaries.append(result["summary"])
                if result["summary"]["used_for_aggregation"]:
                    participant_maps_for_aggregation.append(result["normalized_map"])
                if self.config.save_participant_maps:
                    self._save_participant_outputs(output_dir, result)

        aggregate_map_sum = self._aggregate_participant_maps(participant_maps_for_aggregation, faces_np.shape[0])
        aggregate_map_max = normalize_minmax_np(aggregate_map_sum)
        write_vector_csv(output_dir / "aggregate_face_saliency_sum.csv", aggregate_map_sum)
        write_vector_csv(output_dir / "aggregate_face_saliency_max.csv", aggregate_map_max)

        metrics_payload: dict[str, Any] = {}
        if ground_truth is not None:
            metrics_payload = {
                "aggregate_sum": compute_metrics(
                    aggregate_map_sum,
                    ground_truth,
                    proxy_fixation_percentiles=self.config.proxy_fixation_percentiles,
                ),
                "aggregate_max": compute_metrics(
                    aggregate_map_max,
                    ground_truth,
                    proxy_fixation_percentiles=self.config.proxy_fixation_percentiles,
                ),
                "notes": [
                    "Metrics are computed in the same face-level space as the exact ray-casting pipeline.",
                    "Reference methods here are MeshMamba adaptations of the screen-space and cone-projection ideas from Mesh-Saliency-Projection/reprojection_methods.",
                ],
            }
            write_json(output_dir / "metrics_vs_gt.json", metrics_payload)

        participant_summary_df = pd.DataFrame(participant_summaries)
        participant_summary_df.to_csv(output_dir / "participant_summary.csv", index=False)

        total_points = int(participant_summary_df["points_used"].sum()) if not participant_summary_df.empty else 0
        total_hits = int(participant_summary_df["assigned_count"].sum()) if not participant_summary_df.empty else 0
        run_summary = {
            "method": self.method_config.method,
            "model": resolved.requested_model,
            "output_dir": str(output_dir),
            "device": str(device),
            "num_vertices": int(vertices_np.shape[0]),
            "num_faces": int(faces_np.shape[0]),
            "participants_loaded": int(len(gaze_df)),
            "participants_used_for_aggregation": int(len(participant_maps_for_aggregation)),
            "points_used_total": total_points,
            "hits_total": total_hits,
            "global_hit_rate": float(total_hits / total_points) if total_points > 0 else 0.0,
            "global_assignment_rate": float(total_hits / total_points) if total_points > 0 else 0.0,
            "participant_normalization": "sum=1 per participant, then mean aggregation across participants",
            "resolved_paths": resolved.to_jsonable(),
            "runtime_config": {
                "frame_alignment": self.config.frame_alignment,
                "point_weight_mode": self.config.point_weight_mode,
                "method": self.method_config.method,
                "sigma_px": self.method_config.sigma_px,
                "radius_sigma_mult": self.method_config.radius_sigma_mult,
                "extra_rotate_x_deg": self.config.extra_rotate_x_deg,
                "recenter_to_bbox_center": self.config.recenter_to_bbox_center,
                "override_fov_deg": self.config.override_fov_deg,
                "effective_fov_deg": effective_fov_deg,
                "transform_order": (
                    "recenter_to_bbox_center -> scale -> frame_rotation_z -> extra_rotation_x -> translation"
                    if self.config.recenter_to_bbox_center
                    else "scale -> frame_rotation_z -> extra_rotation_x -> translation"
                ),
                "participant_ids": (
                    None if not self.config.participant_ids else [int(value) for value in self.config.participant_ids]
                ),
                "max_participants": self.config.max_participants,
                "max_points_per_participant": self.config.max_points_per_participant,
                "save_participant_maps": self.config.save_participant_maps,
                "precompute_all_frames": self.config.precompute_all_frames,
            },
            "method_reference": self._method_reference(),
            "method_notes": self._method_notes(),
            "screen_space": {
                "width": int(metadata["video_info"]["resolution_width"]),
                "height": int(metadata["video_info"]["resolution_height"]),
            },
            "surface_graph": {
                "bbox_center_raw_obj": bbox_center_np.astype(float).tolist(),
            },
            "runtime_seconds": round(time.time() - started_at, 3),
        }
        write_json(output_dir / "run_summary.json", run_summary)
        return run_summary

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current environment")
        return torch.device(requested_device)

    def _project_single_participant(
        self,
        row: pd.Series,
        participant_order: int,
        frame_timestamps: np.ndarray,
        face_projection_cache: FrameFaceProjectionCache,
        video_duration_seconds: float,
    ) -> dict[str, Any]:
        gaze_payload, _ = parse_gaze_row(row)
        timestamps = np.asarray(gaze_payload["t"], dtype=np.float32)
        x_coords = np.asarray(gaze_payload["x"], dtype=np.float32)
        y_coords = np.asarray(gaze_payload["y"], dtype=np.float32)

        valid_mask = (
            np.isfinite(timestamps)
            & np.isfinite(x_coords)
            & np.isfinite(y_coords)
            & (x_coords >= 0.0)
            & (x_coords <= 1.0)
            & (y_coords >= 0.0)
            & (y_coords <= 1.0)
        )
        timestamps = timestamps[valid_mask]
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        if self.config.max_points_per_participant is not None:
            limit = max(0, int(self.config.max_points_per_participant))
            timestamps = timestamps[:limit]
            x_coords = x_coords[:limit]
            y_coords = y_coords[:limit]

        point_weights = compute_point_weights(
            timestamps=timestamps,
            video_duration_seconds=video_duration_seconds,
            mode=self.config.point_weight_mode,
        )
        frame_indices = frame_indices_from_timestamps(
            timestamps=timestamps,
            frame_timestamps=frame_timestamps,
            mode=self.config.frame_alignment,
        )

        face_scores = torch.zeros(
            face_projection_cache.faces_t.shape[0],
            dtype=torch.float32,
            device=face_projection_cache.faces_t.device,
        )
        assigned_count = 0

        unique_frames = np.unique(frame_indices)
        sigma_sq = float(self.method_config.sigma_px * self.method_config.sigma_px)
        radius_sq = float(self.method_config.radius_sigma_mult * self.method_config.sigma_px) ** 2

        for frame_index in unique_frames:
            frame_mask = frame_indices == frame_index
            if not np.any(frame_mask):
                continue

            visible_idx, visible_xy = face_projection_cache.get(int(frame_index))
            if visible_idx.numel() == 0:
                continue

            frame_points = np.stack(
                [
                    x_coords[frame_mask] * float(max(1, face_projection_cache.screen_width - 1)),
                    y_coords[frame_mask] * float(max(1, face_projection_cache.screen_height - 1)),
                ],
                axis=1,
            )
            frame_weights = point_weights[frame_mask]
            points_t = torch.tensor(frame_points, dtype=torch.float32, device=visible_xy.device)
            weights_t = torch.tensor(frame_weights, dtype=torch.float32, device=visible_xy.device)

            dxy = points_t.unsqueeze(1) - visible_xy.unsqueeze(0)
            d2 = (dxy * dxy).sum(dim=2)
            gaussian = torch.exp(-0.5 * d2 / max(sigma_sq, 1e-8))

            if self.method_config.method == "screen_space_gaussian":
                effective_weights = gaussian * weights_t.unsqueeze(1)
                assigned_count += int(points_t.shape[0])
            elif self.method_config.method == "cone_projection_on_mesh":
                inside = d2 <= radius_sq
                effective_weights = gaussian * inside.to(gaussian.dtype)
                empty_rows = inside.sum(dim=1) == 0
                if bool(empty_rows.any()):
                    nearest_local = d2[empty_rows].argmin(dim=1)
                    fallback = torch.zeros_like(effective_weights[empty_rows])
                    fallback.scatter_(1, nearest_local.unsqueeze(1), 1.0)
                    effective_weights[empty_rows] = fallback
                effective_weights = effective_weights * weights_t.unsqueeze(1)
                assigned_count += int(points_t.shape[0])
            else:
                raise ValueError(f"Unsupported reference method: {self.method_config.method}")

            face_scores.index_add_(0, visible_idx, effective_weights.sum(dim=0))

        normalized_map_t = normalize_sum_tensor(face_scores)
        raw_scores = face_scores.detach().cpu().numpy().astype(np.float32)
        normalized_map = normalized_map_t.detach().cpu().numpy().astype(np.float32)
        used_for_aggregation = bool(normalized_map.sum() > 0.0)
        summary = {
            "participant_order": int(participant_order),
            "participation_id": int(row["participation_id"]),
            "points_used": int(timestamps.shape[0]),
            "weight_sum": float(point_weights.sum()) if point_weights.size else 0.0,
            "assigned_count": int(assigned_count),
            "hit_count": int(assigned_count),
            "hit_rate": float(assigned_count / timestamps.shape[0]) if timestamps.shape[0] > 0 else 0.0,
            "nonzero_faces_raw": int(np.count_nonzero(raw_scores)),
            "nonzero_faces_smoothed": int(np.count_nonzero(normalized_map)),
            "used_for_aggregation": used_for_aggregation,
        }
        return {
            "summary": summary,
            "raw_hits": raw_scores,
            "smoothed_map": raw_scores,
            "normalized_map": normalized_map,
        }

    def _aggregate_participant_maps(self, participant_maps: list[np.ndarray], face_count: int) -> np.ndarray:
        if not participant_maps:
            return np.zeros(face_count, dtype=np.float32)
        stacked = np.stack(participant_maps, axis=0).astype(np.float64)
        aggregated = stacked.mean(axis=0)
        return normalize_sum_np(aggregated).astype(np.float32)

    def _save_participant_outputs(self, output_dir: Path, participant_result: dict[str, Any]) -> None:
        summary = participant_result["summary"]
        prefix = f"p{summary['participant_order']:03d}_id{summary['participation_id']}"
        write_vector_csv(output_dir / "participants" / f"{prefix}_raw_hits.csv", participant_result["raw_hits"])
        write_vector_csv(output_dir / "participants" / f"{prefix}_smoothed.csv", participant_result["smoothed_map"])
        write_vector_csv(output_dir / "participants" / f"{prefix}_norm_sum.csv", participant_result["normalized_map"])

    def _method_reference(self) -> str:
        if self.method_config.method == "screen_space_gaussian":
            return (
                "/Users/admin/Documents/LAB/SALIENCY_code/#meshes_2.0/GITHUB/"
                "Mesh-Saliency-Projection/reprojection_methods/screen_space_gaussian/"
                "eval_holdout_screenspace.py"
            )
        if self.method_config.method == "cone_projection_on_mesh":
            return (
                "/Users/admin/Documents/LAB/SALIENCY_code/#meshes_2.0/GITHUB/"
                "Mesh-Saliency-Projection/reprojection_methods/cone_projection_on_mesh/"
                "eval_visual_attention_style_saliency3d_clear.py"
            )
        raise ValueError(f"Unsupported reference method: {self.method_config.method}")

    def _method_notes(self) -> list[str]:
        if self.method_config.method == "screen_space_gaussian":
            return [
                "Adaptation of the screen_space_gaussian idea to MeshMamba.",
                "For each frame, visible face centroids are projected to screen and receive Gaussian weights directly in screen space.",
                "Unlike the original screen-space baseline, the final output is a face-level mesh map so that it can be compared to MeshMamba GT.",
                "For this method, global_hit_rate should be interpreted as assignment rate to visible faces, not as exact ray-triangle hit rate.",
            ]
        if self.method_config.method == "cone_projection_on_mesh":
            return [
                "Adaptation of the cone_projection_on_mesh idea to MeshMamba.",
                "For each frame, projected visible face centroids inside radius_sigma_mult * sigma_px receive Gaussian weights in screen space.",
                "Unlike the original Saliency3D_clear script, this adaptation works on faces, uses JSON camera matrices, and does not rely on raw 3D hit points stored in the dataset.",
                "For this method, global_hit_rate should be interpreted as assignment rate to visible faces, not as exact ray-triangle hit rate.",
            ]
        raise ValueError(f"Unsupported reference method: {self.method_config.method}")
