from __future__ import annotations

import math

import numpy as np


def build_projection_matrix_from_fov_degrees(
    fov_degrees: float,
    aspect_ratio: float,
    clip_start: float,
    clip_end: float,
) -> np.ndarray:
    fov_radians = math.radians(float(fov_degrees))
    f = 1.0 / math.tan(fov_radians * 0.5)
    near = float(clip_start)
    far = float(clip_end)
    return np.asarray(
        [
            [f / float(aspect_ratio), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
