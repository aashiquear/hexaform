"""Run reports.

The SciSpace review's section 27 asks for a machine-readable summary saved
*alongside* every Exodus file, because a stochastic mesher is judged by its
success rate across seeds rather than by one attractive mesh.  This module
writes that summary and provides a helper to flatten many reports into a CSV for
a Monte Carlo campaign.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if hasattr(obj, "as_dict"):
        return _clean(obj.as_dict())
    if hasattr(obj, "to_dict"):
        return _clean(obj.to_dict())
    return str(obj)


def build_report(
    config=None, mesh=None, packing=None, validation=None, timings: Dict | None = None
) -> Dict[str, Any]:
    """Assemble the full report dictionary."""
    report: Dict[str, Any] = {"schema": "hexaform.report/1"}
    if config is not None:
        report["config"] = _clean(config.to_dict())
        report["seed"] = int(getattr(config, "seed", 0))
    if packing is not None:
        report["packing"] = _clean(packing.stats)
        report["inclusions"] = [
            {
                "particle_id": inc.particle_id,
                "shape": type(inc.shape).__name__,
                "center": _clean(np.asarray(inc.shape.center)),
                "bounding_radius": float(inc.shape.bounding_radius()),
                "volume": float(inc.shape.volume()),
                "macrobox_lo": _clean(np.asarray(inc.box_lo)),
                "macrobox_cells": _clean(np.asarray(inc.k)),
            }
            for inc in packing.inclusions
        ]
    if mesh is not None:
        report["mesh"] = {
            "n_nodes": mesh.n_nodes,
            "n_elements": mesh.n_elements,
            "element_type": mesh.element_type,
            "nodes_per_element": mesh.nodes_per_element,
            "blocks": {
                mesh.block_name(b): int((mesh.block_id == b).sum()) for b in mesh.block_ids
            },
            "side_sets": {k: int(v[0].size) for k, v in mesh.side_sets.items()},
            "node_sets": {k: int(v.size) for k, v in mesh.node_sets.items()},
        }
        report["metadata"] = _clean(
            {k: v for k, v in mesh.metadata.items() if k != "template_stats"}
        )
    if validation is not None:
        report["validation"] = _clean(validation.as_dict())
        report["success"] = bool(validation.passed)
    if timings is not None:
        report["timings"] = _clean(timings)
    return report


def write_report(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_report(**kwargs), indent=2))
    return path


# ----------------------------------------------------------------------
# Monte Carlo campaign helpers
# ----------------------------------------------------------------------
CAMPAIGN_COLUMNS = [
    "seed",
    "success",
    "particle_count",
    "particle_volume_fraction",
    "minimum_particle_gap",
    "minimum_wall_gap",
    "element_count",
    "node_count",
    "minimum_scaled_jacobian",
    "mean_scaled_jacobian",
    "maximum_edge_ratio",
    "relative_volume_error",
    "max_node_surface_error",
    "minimum_elements_per_ligament",
    "meshing_time",
]


def flatten_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a full report to the campaign columns."""
    pack = report.get("packing", {}) or {}
    val = report.get("validation", {}) or {}
    qual = val.get("quality", {}) or {}
    geo = val.get("geometry", {}) or {}
    lig = val.get("ligaments", {}) or {}
    mesh = report.get("mesh", {}) or {}
    timings = report.get("timings", {}) or {}
    return {
        "seed": report.get("seed"),
        "success": report.get("success"),
        "particle_count": pack.get("placed"),
        "particle_volume_fraction": pack.get("volume_fraction"),
        "minimum_particle_gap": pack.get("min_surface_gap"),
        "minimum_wall_gap": pack.get("min_wall_gap"),
        "element_count": mesh.get("n_elements"),
        "node_count": mesh.get("n_nodes"),
        "minimum_scaled_jacobian": qual.get("min_scaled_jacobian"),
        "mean_scaled_jacobian": qual.get("mean_scaled_jacobian"),
        "maximum_edge_ratio": qual.get("max_edge_ratio"),
        "relative_volume_error": geo.get("relative_volume_error"),
        "max_node_surface_error": geo.get("max_node_surface_error"),
        "minimum_elements_per_ligament": lig.get("min_ligament_elements"),
        "meshing_time": timings.get("total"),
    }


def write_campaign_csv(path: str | Path, reports: Iterable[Dict[str, Any]]) -> Path:
    """Write one row per realisation, for success-rate analysis across seeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = [flatten_report(r) for r in reports]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CAMPAIGN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_report(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def campaign_summary(reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate success rate and quality statistics over many realisations."""
    rows = [flatten_report(r) for r in reports]
    if not rows:
        return {"n_realisations": 0}

    def col(name: str) -> np.ndarray:
        vals = [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        return np.asarray(vals, dtype=float) if vals else np.zeros(0)

    ok = [bool(r.get("success")) for r in rows]
    sj = col("minimum_scaled_jacobian")
    return {
        "n_realisations": len(rows),
        "success_rate": float(np.mean(ok)) if ok else 0.0,
        "min_scaled_jacobian_worst": float(sj.min()) if sj.size else None,
        "min_scaled_jacobian_mean": float(sj.mean()) if sj.size else None,
        "element_count_mean": float(col("element_count").mean()) if rows else None,
        "volume_fraction_mean": float(col("particle_volume_fraction").mean()),
        "meshing_time_mean": float(col("meshing_time").mean()),
    }
