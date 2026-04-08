from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ResolvedModelPaths:
    requested_model: str
    csv_model_name: str
    mesh_model_name: str
    json_model_name: str
    gt_model_name: Optional[str]
    csv_path: Path
    mesh_path: Path
    json_path: Path
    gt_path: Optional[Path]

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("csv_path", "mesh_path", "json_path", "gt_path"):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        return payload


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vector_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def write_vector_csv(path: Path, values: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savetxt(path, np.asarray(values, dtype=np.float32).reshape(-1), fmt="%.8f")


def read_gaze_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_literal(value: Any) -> Any:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_gaze_row(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    gaze_payload = parse_literal(row["data_gazes"])
    fps_payload = parse_literal(row["data_fps"])
    return gaze_payload, fps_payload


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue

                face_indices: list[int] = []
                for token in parts:
                    index_token = token.split("/")[0]
                    if not index_token:
                        continue
                    face_indices.append(int(index_token) - 1)

                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) > 3:
                    anchor = face_indices[0]
                    for idx in range(1, len(face_indices) - 1):
                        faces.append([anchor, face_indices[idx], face_indices[idx + 1]])

    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    if not faces:
        raise ValueError(f"No faces found in OBJ: {path}")

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)
