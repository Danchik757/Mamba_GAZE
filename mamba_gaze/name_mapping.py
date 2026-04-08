from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .io_utils import ResolvedModelPaths, write_json


JSON_PREFIXES = (
    "MeshMamba_non_texture_",
    "MeshMamba_rgb_texture_",
)


@dataclass(frozen=True)
class NamedPath:
    name: str
    path: Path


def canonicalize_name(name: str) -> str:
    lowered = name.lower().replace(" copy", "")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def strip_json_prefix(name: str) -> str:
    for prefix in JSON_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _index_items(items: list[NamedPath]) -> tuple[dict[str, list[NamedPath]], dict[str, list[NamedPath]]]:
    exact_index: dict[str, list[NamedPath]] = defaultdict(list)
    canonical_index: dict[str, list[NamedPath]] = defaultdict(list)
    for item in items:
        exact_index[item.name].append(item)
        canonical_index[canonicalize_name(item.name)].append(item)
    return exact_index, canonical_index


def _resolve_match(name: str, exact_index: dict[str, list[NamedPath]], canonical_index: dict[str, list[NamedPath]]) -> dict[str, Any]:
    exact_candidates = exact_index.get(name, [])
    if len(exact_candidates) == 1:
        item = exact_candidates[0]
        return {
            "name": item.name,
            "path": str(item.path),
            "match_type": "exact",
            "candidate_names": [item.name],
        }

    canonical_candidates = canonical_index.get(canonicalize_name(name), [])
    if len(canonical_candidates) == 1:
        item = canonical_candidates[0]
        return {
            "name": item.name,
            "path": str(item.path),
            "match_type": "canonical",
            "candidate_names": [item.name],
        }

    if len(canonical_candidates) > 1:
        return {
            "name": None,
            "path": None,
            "match_type": "ambiguous",
            "candidate_names": sorted(item.name for item in canonical_candidates),
        }

    return {
        "name": None,
        "path": None,
        "match_type": "missing",
        "candidate_names": [],
    }


def _collect_dir_status(path: Optional[Path], pattern: str) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "is_dir": False,
            "matched_count": 0,
            "sample_names": [],
        }

    exists = path.exists()
    is_dir = path.is_dir()
    matched_paths = sorted(path.glob(pattern)) if exists and is_dir else []
    sample_names = [matched.name for matched in matched_paths[:5]]
    return {
        "path": str(path),
        "exists": exists,
        "is_dir": is_dir,
        "matched_count": len(matched_paths),
        "sample_names": sample_names,
    }


def build_dataset_mapping(
    gaze_csv_dir: Path,
    mesh_dir: Path,
    json_dir: Path,
    gt_dir: Optional[Path],
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    diagnostics = {
        "gaze_csv_dir": _collect_dir_status(gaze_csv_dir, "*.csv"),
        "mesh_dir": _collect_dir_status(mesh_dir, "*"),
        "json_dir": _collect_dir_status(json_dir, "*.json"),
        "gt_dir": _collect_dir_status(gt_dir, "*.csv") if gt_dir is not None else None,
    }

    csv_items = [NamedPath(path.stem, path) for path in sorted(gaze_csv_dir.glob("*.csv"))]
    mesh_items = [NamedPath(path.name, path / f"{path.name}.obj") for path in sorted(mesh_dir.iterdir()) if path.is_dir()]
    json_items = [NamedPath(strip_json_prefix(path.stem), path) for path in sorted(json_dir.glob("*.json"))]
    gt_items = [] if gt_dir is None else [NamedPath(path.stem, path) for path in sorted(gt_dir.glob("*.csv"))]

    mesh_exact, mesh_canonical = _index_items(mesh_items)
    json_exact, json_canonical = _index_items(json_items)
    gt_exact, gt_canonical = _index_items(gt_items)

    entries: dict[str, Any] = {}
    for csv_item in csv_items:
        entries[csv_item.name] = {
            "csv": {
                "name": csv_item.name,
                "path": str(csv_item.path),
                "match_type": "primary",
                "candidate_names": [csv_item.name],
            },
            "mesh": _resolve_match(csv_item.name, mesh_exact, mesh_canonical),
            "json": _resolve_match(csv_item.name, json_exact, json_canonical),
            "gt": None if gt_dir is None else _resolve_match(csv_item.name, gt_exact, gt_canonical),
        }

    summary = {
        "csv_models": len(csv_items),
        "mesh_models": len(mesh_items),
        "json_models": len(json_items),
        "gt_models": len(gt_items),
        "resolved_mesh": sum(1 for entry in entries.values() if entry["mesh"]["match_type"] in {"exact", "canonical"}),
        "resolved_json": sum(1 for entry in entries.values() if entry["json"]["match_type"] in {"exact", "canonical"}),
        "resolved_gt": sum(
            1
            for entry in entries.values()
            if entry["gt"] is not None and entry["gt"]["match_type"] in {"exact", "canonical"}
        ),
        "ambiguous_gt": sum(
            1
            for entry in entries.values()
            if entry["gt"] is not None and entry["gt"]["match_type"] == "ambiguous"
        ),
    }

    payload = {
        "source_dirs": {
            "gaze_csv_dir": str(gaze_csv_dir),
            "mesh_dir": str(mesh_dir),
            "json_dir": str(json_dir),
            "gt_dir": None if gt_dir is None else str(gt_dir),
        },
        "diagnostics": diagnostics,
        "summary": summary,
        "entries": entries,
    }

    if output_path is not None:
        write_json(output_path, payload)

    return payload


def resolve_model_from_mapping(model_name: str, mapping: dict[str, Any]) -> ResolvedModelPaths:
    entries = mapping["entries"]

    if model_name in entries:
        entry_name = model_name
    else:
        canonical = canonicalize_name(model_name)
        candidates = [name for name in entries if canonicalize_name(name) == canonical]
        if len(candidates) != 1:
            diagnostics = mapping.get("diagnostics", {})
            summary = mapping.get("summary", {})
            raise ValueError(
                "Model "
                f"'{model_name}' is not uniquely resolvable. "
                f"Canonical matches: {sorted(candidates)}. "
                f"Summary: csv={summary.get('csv_models', 0)}, "
                f"mesh={summary.get('mesh_models', 0)}, "
                f"json={summary.get('json_models', 0)}, "
                f"gt={summary.get('gt_models', 0)}. "
                f"CSV dir diagnostics: {diagnostics.get('gaze_csv_dir')}"
            )
        entry_name = candidates[0]

    entry = entries[entry_name]

    mesh_match = entry["mesh"]
    json_match = entry["json"]
    gt_match = entry["gt"]

    if mesh_match["path"] is None:
        raise ValueError(f"Mesh path is unresolved for model '{entry_name}'")
    if json_match["path"] is None:
        raise ValueError(f"JSON path is unresolved for model '{entry_name}'")

    gt_name = None if gt_match is None else gt_match["name"]
    gt_path = None if gt_match is None or gt_match["path"] is None else Path(gt_match["path"])

    return ResolvedModelPaths(
        requested_model=model_name,
        csv_model_name=entry["csv"]["name"],
        mesh_model_name=mesh_match["name"],
        json_model_name=json_match["name"],
        gt_model_name=gt_name,
        csv_path=Path(entry["csv"]["path"]),
        mesh_path=Path(mesh_match["path"]),
        json_path=Path(json_match["path"]),
        gt_path=gt_path,
    )
