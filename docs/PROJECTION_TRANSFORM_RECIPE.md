# Projection Transform Recipe

This note captures the exact runtime transforms that were applied in `MAMBA_GAZE`
when projecting 2D gaze points onto a 3D mesh.

It is intended for porting the projection logic to another dataset or another
reprojection implementation.

## Important distinction

There are two different classes of changes:

1. **OBJ / mesh-space transforms**
   These change how the mesh geometry is positioned before projection.
2. **Camera calibration overrides**
   These do **not** change the OBJ. They change how the camera sees the mesh.

For `Aquarium_Deep_Sea_Diver_v1_L1`, the main fixes were:

- `recenter_to_bbox_center = true`
- `extra_rotate_x_deg = 90`
- `override_fov_deg = 35.8972065`

Only the first two affect the mesh transform chain directly.

## Exact transform order

The current pipeline applies transforms in this order:

If `recenter_to_bbox_center = false`:

```text
scale -> frame_rotation_z -> extra_rotation_x -> translation
```

If `recenter_to_bbox_center = true`:

```text
recenter_to_bbox_center -> scale -> frame_rotation_z -> extra_rotation_x -> translation
```

This matches the runtime summary written by:

- [pipeline.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/mamba_gaze/pipeline.py#L321)

## What each step means

### 1. `recenter_to_bbox_center`

This emulates the Blender call:

- [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L414)

Equivalent idea:

```python
bbox_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
vertices = vertices - bbox_center
```

Why this was needed:

- The Blender video-generation pipeline changes the object origin to the bbox center.
- That origin shift is not stored as a separate field in the JSON.
- If this step is omitted, later rotation happens around the wrong pivot.

### 2. `scale`

Scale comes from JSON:

- [pipeline.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/mamba_gaze/pipeline.py#L199)
- [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L642)

Equivalent:

```python
vertices = vertices * scale_xyz
```

Important:

- We did **not** manually invent a new mesh scale for `Aquarium`.
- The runtime uses the scale stored in JSON.
- The apparent size mismatch seen on preview was later handled via camera FOV override, not by changing the JSON scale.

### 3. `frame_rotation_z`

Per-frame Z rotation comes from:

- `frames[i].rotation_z_radians`
- [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L679)

Equivalent:

```python
x2 = cos(a) * x - sin(a) * y
y2 = sin(a) * x + cos(a) * y
z2 = z
```

### 4. `extra_rotation_x`

This is **our debugging/runtime correction**, not part of the original Blender
video-generation script.

For `Aquarium`, we used:

```text
extra_rotate_x_deg = 90
```

Equivalent:

```python
y3 = cos(rx) * y2 - sin(rx) * z2
z3 = sin(rx) * y2 + cos(rx) * z2
x3 = x2
```

Why it was introduced:

- Visual inspection of the preview showed that the object orientation in the
  projection pipeline did not match the perceived orientation in the source video.

### 5. `translation`

Translation comes from JSON:

- [pipeline.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/mamba_gaze/pipeline.py#L207)
- [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L855)

Equivalent:

```python
vertices = vertices + translation_xyz
```

## Camera-side calibration

### `override_fov_deg`

This is **not an OBJ transform**.

It changes the projection matrix used by the camera while keeping the JSON view
matrix:

- [pipeline.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/mamba_gaze/pipeline.py#L217)

For `Aquarium`, we used:

```text
override_fov_deg = 35.8972065
```

Why it was needed:

- The object looked too small on preview even though JSON scale was already correct.
- This indicated a camera/framing mismatch rather than an OBJ scale mismatch.

## Minimal NumPy reference

```python
import math
import numpy as np


def apply_projection_transform(
    vertices,
    *,
    scale_xyz,
    rotation_z_rad,
    translation_xyz,
    recenter_to_bbox_center=False,
    extra_rotate_x_deg=0.0,
):
    vertices = np.asarray(vertices, dtype=np.float64)
    scale_xyz = np.asarray(scale_xyz, dtype=np.float64).reshape(1, 3)
    translation_xyz = np.asarray(translation_xyz, dtype=np.float64).reshape(1, 3)

    if recenter_to_bbox_center:
        bbox_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        vertices = vertices - bbox_center.reshape(1, 3)

    vertices = vertices * scale_xyz

    a = float(rotation_z_rad)
    ca = math.cos(a)
    sa = math.sin(a)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    x2 = ca * x - sa * y
    y2 = sa * x + ca * y
    z2 = z

    rx = math.radians(float(extra_rotate_x_deg))
    if abs(rx) > 1e-12:
        crx = math.cos(rx)
        srx = math.sin(rx)
        y3 = crx * y2 - srx * z2
        z3 = srx * y2 + crx * z2
    else:
        y3 = y2
        z3 = z2

    out = np.stack([x2, y3, z3], axis=1)
    out = out + translation_xyz
    return out
```

## Practical rule for another dataset

If the object looks wrong on another dataset, check in this order:

1. Is Blender changing the object origin before export or animation?
2. Is the stored JSON scale really the runtime scale?
3. Is there an axis correction missing, similar to `extra_rotate_x_deg = 90`?
4. Is the object shape correct but the apparent screen size wrong?
   If yes, the problem is likely camera FOV / framing, not OBJ scale.

## File references in current code

- Transform chain in runtime: [pipeline.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/mamba_gaze/pipeline.py#L164)
- Frame transform implementation: [pipeline.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/mamba_gaze/pipeline.py#L110)
- Blender origin-centering step: [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L414)
- Blender scale-to-bbox step: [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L420)
- Blender camera placement: [mamba_render_1.py](/Users/admin/Documents/LAB/SALIENCY_code/GAZE_DATA/MAMBA_GAZE/video_creation_scripts/mamba_render_1.py#L466)
