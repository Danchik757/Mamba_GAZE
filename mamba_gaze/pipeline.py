from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .io_utils import (
    ResolvedModelPaths,
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
    FaceAdjacency,
    build_camera_rays,
    build_face_adjacency,
    compute_point_weights,
    diffuse_face_values,
    frame_indices_from_timestamps,
    intersect_rays_with_triangles,
    normalize_minmax_np,
    normalize_sum_np,
    normalize_sum_tensor,
)
from .metrics import compute_metrics
from .name_mapping import build_dataset_mapping, resolve_model_from_mapping


@dataclass
class DatasetPaths:
    gaze_csv_dir: Path
    mesh_dir: Path
    json_dir: Path
    gt_dir: Optional[Path]
    output_dir: Path
    mapping_json: Path

    @classmethod
    def local_defaults(cls, output_dir: Optional[Path] = None) -> "DatasetPaths":
        base = Path("/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA")
        target_output = output_dir or Path.cwd() / "outputs"
        return cls(
            gaze_csv_dir=base / "csv_for_models" / "MeshMamba_non_texture",
            mesh_dir=base / "datasets" / "MeshMamba" / "MeshMambaSaliency" / "MeshFile" / "non_texture",
            json_dir=base / "jsons_for_models" / "Mamba_non_textured",
            gt_dir=base / "datasets" / "MeshMamba" / "MeshMambaSaliency" / "SaliencyMap" / "non_texture",
            output_dir=target_output,
            mapping_json=target_output / "meshmamba_non_texture_name_mapping.json",
        )


@dataclass
class RuntimeConfig:
    device: str = "auto"
    frame_alignment: str = "nearest"
    point_weight_mode: str = "unit"
    ray_batch_size: int = 64
    smoothing_steps: int = 8
    smoothing_alpha: float = 0.6
    max_participants: Optional[int] = None
    max_points_per_participant: Optional[int] = None
    save_participant_maps: bool = True
    precompute_all_frames: bool = False
    proxy_fixation_percentiles: tuple[float, ...] = (90.0, 95.0, 99.0)


class FrameVertexCache:
    def __init__(
        self,
        base_vertices: torch.Tensor,
        frame_angles_radians: torch.Tensor,
        scale: torch.Tensor,
        translation: torch.Tensor,
        precompute_all: bool,
    ) -> None:
        self.base_vertices = base_vertices
        self.frame_angles_radians = frame_angles_radians
        self.scale = scale.view(1, 3)
        self.translation = translation.view(1, 3)
        self.scaled_vertices = self.base_vertices * self.scale
        self._cache: dict[int, torch.Tensor] = {}
        if precompute_all:
            for index in range(int(frame_angles_radians.shape[0])):
                self._cache[index] = self._transform_frame(index)

    def _transform_frame(self, frame_index: int) -> torch.Tensor:
        angle = self.frame_angles_radians[frame_index]
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        x = self.scaled_vertices[:, 0]
        y = self.scaled_vertices[:, 1]
        z = self.scaled_vertices[:, 2]

        rotated_x = cos_angle * x - sin_angle * y
        rotated_y = sin_angle * x + cos_angle * y
        rotated = torch.stack([rotated_x, rotated_y, z], dim=1)
        return rotated + self.translation

    def get(self, frame_index: int) -> torch.Tensor:
        frame_index = int(frame_index)
        if frame_index not in self._cache:
            self._cache[frame_index] = self._transform_frame(frame_index)
        return self._cache[frame_index]


class MeshMambaFaceProjector:
    def __init__(self, dataset_paths: DatasetPaths, runtime_config: RuntimeConfig) -> None:
        self.paths = dataset_paths
        self.config = runtime_config

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

        print(f"Model: {resolved.requested_model}")
        print(f"  CSV:  {resolved.csv_path}")
        print(f"  OBJ:  {resolved.mesh_path}")
        print(f"  JSON: {resolved.json_path}")
        print(f"  GT:   {resolved.gt_path if resolved.gt_path is not None else 'not resolved'}")

        output_dir = ensure_dir(self.paths.output_dir / resolved.csv_model_name)
        write_json(output_dir / "resolved_paths.json", resolved.to_jsonable())

        vertices_np, faces_np = load_obj(resolved.mesh_path)
        ground_truth = None if resolved.gt_path is None else load_vector_csv(resolved.gt_path)
        if ground_truth is not None and ground_truth.shape[0] != faces_np.shape[0]:
            raise ValueError(
                f"Ground truth length {ground_truth.shape[0]} does not match face count {faces_np.shape[0]}"
            )

        metadata = load_json(resolved.json_path)
        gaze_df = read_gaze_dataframe(resolved.csv_path)
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
        adjacency = build_face_adjacency(faces_np, device=device)

        model_scale = metadata["model_static"]["scale"]
        scale_t = torch.tensor(
            [float(model_scale[0]), float(model_scale[1]), float(model_scale[2])],
            dtype=torch.float32,
            device=device,
        )
        translation_t = torch.tensor(metadata["model_static"]["location"], dtype=torch.float32, device=device)
        vertex_cache = FrameVertexCache(
            base_vertices=vertices_t,
            frame_angles_radians=frame_angles,
            scale=scale_t,
            translation=translation_t,
            precompute_all=self.config.precompute_all_frames,
        )

        view_matrix = torch.tensor(metadata["camera_static"]["view_matrix"], dtype=torch.float32, device=device)
        projection_matrix = torch.tensor(
            metadata["camera_static"]["projection_matrix"],
            dtype=torch.float32,
            device=device,
        )
        inv_view = torch.inverse(view_matrix)
        inv_projection = torch.inverse(projection_matrix)
        camera_origin_h = inv_view @ torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        camera_origin = camera_origin_h[:3] / camera_origin_h[3]

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
                    faces_t=faces_t,
                    adjacency=adjacency,
                    vertex_cache=vertex_cache,
                    inv_view=inv_view,
                    inv_projection=inv_projection,
                    camera_origin=camera_origin,
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
                    "CC, SIM, KLD and MSE/SE are primary face-level comparison metrics.",
                    "NSS/AUC here are proxy metrics derived from top-percentile GT masks because raw fixation masks are unavailable.",
                ],
            }
            write_json(output_dir / "metrics_vs_gt.json", metrics_payload)

        participant_summary_df = pd.DataFrame(participant_summaries)
        participant_summary_df.to_csv(output_dir / "participant_summary.csv", index=False)

        total_points = int(participant_summary_df["points_used"].sum()) if not participant_summary_df.empty else 0
        total_hits = int(participant_summary_df["hit_count"].sum()) if not participant_summary_df.empty else 0
        run_summary = {
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
            "participant_normalization": "sum=1 per participant, then mean aggregation across participants",
            "aggregate_outputs": {
                "aggregate_face_saliency_sum.csv": str(output_dir / "aggregate_face_saliency_sum.csv"),
                "aggregate_face_saliency_max.csv": str(output_dir / "aggregate_face_saliency_max.csv"),
            },
            "resolved_paths": resolved.to_jsonable(),
            "runtime_config": {
                "frame_alignment": self.config.frame_alignment,
                "point_weight_mode": self.config.point_weight_mode,
                "ray_batch_size": self.config.ray_batch_size,
                "smoothing_steps": self.config.smoothing_steps,
                "smoothing_alpha": self.config.smoothing_alpha,
                "max_participants": self.config.max_participants,
                "max_points_per_participant": self.config.max_points_per_participant,
                "save_participant_maps": self.config.save_participant_maps,
                "precompute_all_frames": self.config.precompute_all_frames,
                "proxy_fixation_percentiles": list(self.config.proxy_fixation_percentiles),
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
        faces_t: torch.Tensor,
        adjacency: FaceAdjacency,
        vertex_cache: FrameVertexCache,
        inv_view: torch.Tensor,
        inv_projection: torch.Tensor,
        camera_origin: torch.Tensor,
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

        face_hits = torch.zeros(faces_t.shape[0], dtype=torch.float32, device=faces_t.device)
        hit_count = 0

        unique_frames = np.unique(frame_indices)
        for frame_index in unique_frames:
            frame_mask = frame_indices == frame_index
            if not np.any(frame_mask):
                continue

            frame_points = np.stack([x_coords[frame_mask], y_coords[frame_mask]], axis=1)
            frame_weights = point_weights[frame_mask]

            points_t = torch.tensor(frame_points, dtype=torch.float32, device=faces_t.device)
            weights_t = torch.tensor(frame_weights, dtype=torch.float32, device=faces_t.device)
            ray_origins, ray_directions = build_camera_rays(
                points_xy=points_t,
                inv_projection=inv_projection,
                inv_view=inv_view,
                camera_origin=camera_origin,
            )

            frame_vertices = vertex_cache.get(int(frame_index))
            face_vertices = frame_vertices[faces_t]
            hit_faces, _ = intersect_rays_with_triangles(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                triangle_vertices=face_vertices,
                ray_batch_size=self.config.ray_batch_size,
            )

            valid_hits = hit_faces >= 0
            if bool(valid_hits.any()):
                face_hits.index_add_(0, hit_faces[valid_hits], weights_t[valid_hits])
                hit_count += int(valid_hits.sum().item())

        smoothed_hits = diffuse_face_values(
            face_values=face_hits,
            adjacency=adjacency,
            steps=self.config.smoothing_steps,
            alpha=self.config.smoothing_alpha,
        )
        normalized_map_t = normalize_sum_tensor(smoothed_hits)

        raw_hits = face_hits.detach().cpu().numpy().astype(np.float32)
        smoothed_map = smoothed_hits.detach().cpu().numpy().astype(np.float32)
        normalized_map = normalized_map_t.detach().cpu().numpy().astype(np.float32)

        used_for_aggregation = bool(normalized_map.sum() > 0.0)
        summary = {
            "participant_order": int(participant_order),
            "participation_id": int(row["participation_id"]),
            "points_used": int(timestamps.shape[0]),
            "weight_sum": float(point_weights.sum()) if point_weights.size else 0.0,
            "hit_count": int(hit_count),
            "hit_rate": float(hit_count / timestamps.shape[0]) if timestamps.shape[0] > 0 else 0.0,
            "nonzero_faces_raw": int(np.count_nonzero(raw_hits)),
            "nonzero_faces_smoothed": int(np.count_nonzero(smoothed_map)),
            "used_for_aggregation": used_for_aggregation,
        }

        return {
            "summary": summary,
            "raw_hits": raw_hits,
            "smoothed_map": smoothed_map,
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
