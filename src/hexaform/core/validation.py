"""Acceptance checks for a generated mesh.

This module implements the validation targets both source reviews specify.  The
guiding principle from the ScholarGPT review is worth restating, because it
determines the order of the gates below:

    validity first, quality second, geometric fidelity third, element count
    fourth -- a beautiful sphere surrounded by a few inverted hexes is unusable.

Checks
------
topology
    all-hex, conforming, no hanging nodes, no unused nodes.
validity
    every element has a positive Jacobian.
quality
    scaled Jacobian and edge-ratio statistics against thresholds.
geometry
    node-wise radial error against the analytic surface (should be at machine
    precision, since interface nodes are placed *on* the surface), and the
    facet-induced volume error, which is the real measure of HEX8 geometric
    fidelity and converges at second order in the surface resolution.
ligaments
    number of matrix elements across the narrowest inclusion gap.
phase_field
    ``h / ell`` statistics against the regularisation length, since for
    ``ell << R`` it is fracture physics, not sphere curvature, that sets the
    element count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from ..packing.rsa import PackedInclusion
from .mesh import HexMesh
from .quality import QualityReport, assess, edge_ratio, element_volumes, scaled_jacobian
from .topology import check_conformity


@dataclass
class ValidationThresholds:
    min_scaled_jacobian: float = 0.2
    """Below this a solid-mechanics solver is likely to struggle; 0.2 is a
    common floor, 0.3 a conservative one."""

    max_edge_ratio: float = 12.0
    max_volume_error: float = 0.05
    """Relative error between meshed and analytic inclusion volume."""

    min_ligament_elements: int = 2
    require_conforming: bool = True


@dataclass
class ValidationReport:
    passed: bool
    quality: QualityReport
    topology: Dict[str, object]
    geometry: Dict[str, float]
    ligaments: Dict[str, float]
    phase_field: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "quality": self.quality.as_dict(),
            "topology": self.topology,
            "geometry": self.geometry,
            "ligaments": self.ligaments,
            "phase_field": self.phase_field,
            "failures": self.failures,
        }

    def __str__(self) -> str:
        head = "VALID" if self.passed else "INVALID"
        lines = [
            f"=== Hexaform validation: {head} ===",
            str(self.quality),
            f"  conforming={self.topology.get('conforming')}  "
            f"hanging nodes={self.topology.get('n_hanging_nodes')}  "
            f"non-manifold faces={self.topology.get('n_nonmanifold_faces')}",
        ]
        # Geometry and ligament checks need the analytic shapes, which are not
        # recoverable from an Exodus file alone.  Omit them rather than
        # reporting nan when validating a mesh read back from disk.
        if "max_node_surface_error" in self.geometry:
            lines.append(
                f"  interface node radial error (max) = "
                f"{self.geometry['max_node_surface_error']:.3e}"
            )
        if "relative_volume_error" in self.geometry:
            lines.append(
                f"  inclusion volume error (rel)      = "
                f"{self.geometry['relative_volume_error']:.5f}"
            )
        lig = self.ligaments.get("min_ligament_elements")
        if lig is not None and np.isfinite(lig):
            lines.append(f"  min ligament elements             = {lig:.2f}")
        if not self.geometry and not self.ligaments:
            lines.append(
                "  (geometry and ligament checks skipped: analytic shapes not supplied)"
            )
        if self.phase_field:
            lines.append(
                f"  h/ell: interface={self.phase_field.get('h_over_ell_interface', float('nan')):.3f}  "
                f"bulk={self.phase_field.get('h_over_ell_bulk', float('nan')):.3f}"
            )
        for f in self.failures:
            lines.append(f"  FAIL: {f}")
        return "\n".join(lines)


def validate(
    mesh: HexMesh,
    inclusions: Sequence[PackedInclusion] = (),
    cell_size: float | None = None,
    inclusion_block_ids: Sequence[int] | None = None,
    thresholds: ValidationThresholds | None = None,
    phase_field_length: float | None = None,
    check_topology: bool = True,
) -> ValidationReport:
    """Run the full acceptance battery."""
    th = thresholds or ValidationThresholds()
    failures: List[str] = []

    q = assess(mesh, th.min_scaled_jacobian)
    topo = check_conformity(mesh) if check_topology else {"conforming": None}

    # ---- validity and quality gates ---------------------------------
    if q.n_inverted:
        failures.append(f"{q.n_inverted} inverted elements (Jacobian <= 0)")
    if q.min_scaled_jacobian < th.min_scaled_jacobian:
        failures.append(
            f"min scaled Jacobian {q.min_scaled_jacobian:.4f} < {th.min_scaled_jacobian}"
        )
    if q.max_edge_ratio > th.max_edge_ratio:
        failures.append(f"max edge ratio {q.max_edge_ratio:.2f} > {th.max_edge_ratio}")
    if th.require_conforming and check_topology and not topo.get("conforming"):
        failures.append("mesh is not conforming (hanging or non-manifold entities)")

    # ---- geometric fidelity -----------------------------------------
    geometry = _geometry_report(mesh, inclusions, inclusion_block_ids)
    if geometry.get("relative_volume_error", 0.0) > th.max_volume_error:
        failures.append(
            f"inclusion volume error {geometry['relative_volume_error']:.4f} "
            f"> {th.max_volume_error}"
        )

    # ---- ligaments ---------------------------------------------------
    ligaments = _ligament_report(inclusions, cell_size)
    min_lig = ligaments.get("min_ligament_elements", float("inf"))
    if np.isfinite(min_lig) and min_lig < th.min_ligament_elements:
        failures.append(
            f"only {min_lig:.2f} elements across the narrowest ligament "
            f"(< {th.min_ligament_elements})"
        )

    # ---- phase field --------------------------------------------------
    pf = _phase_field_report(mesh, inclusions, cell_size, phase_field_length)

    return ValidationReport(
        passed=not failures,
        quality=q,
        topology=topo,
        geometry=geometry,
        ligaments=ligaments,
        phase_field=pf,
        failures=failures,
    )


def _geometry_report(
    mesh: HexMesh,
    inclusions: Sequence[PackedInclusion],
    inclusion_block_ids: Sequence[int] | None,
) -> Dict[str, float]:
    if not len(inclusions):
        return {}

    if inclusion_block_ids is None:
        inclusion_block_ids = [b for b in mesh.block_ids if int(b) >= 100]
    mask = np.isin(mesh.block_id, np.asarray(list(inclusion_block_ids), dtype=np.int32))
    vol = element_volumes(mesh.points, mesh.cells)
    meshed_volume = float(vol[mask].sum())
    exact_volume = float(sum(inc.shape.volume() for inc in inclusions))

    # Node-wise fidelity: how far is each interface node from the analytic
    # surface?  Should be at round-off, because nodes are *placed* there.
    max_node_err = 0.0
    mean_node_err = 0.0
    n_checked = 0
    for inc in inclusions:
        shape = inc.shape
        r_out = shape.bounding_radius()
        d = np.linalg.norm(mesh.points - shape.center, axis=1)
        near = np.flatnonzero(d < 1.35 * r_out)
        if near.size == 0:
            continue
        scale = shape.radial_scale(mesh.points[near])
        on_surface = near[np.abs(scale - 1.0) < 1e-6]
        if on_surface.size == 0:
            continue
        err = np.abs(
            np.linalg.norm(mesh.points[on_surface] - shape.center, axis=1)
            - shape.support_radius(
                (mesh.points[on_surface] - shape.center)
                / np.linalg.norm(mesh.points[on_surface] - shape.center, axis=1)[:, None]
            )
        )
        max_node_err = max(max_node_err, float(err.max()))
        mean_node_err += float(err.sum())
        n_checked += int(err.size)

    return {
        "meshed_inclusion_volume": meshed_volume,
        "analytic_inclusion_volume": exact_volume,
        "relative_volume_error": abs(meshed_volume - exact_volume) / max(exact_volume, 1e-300),
        "max_node_surface_error": max_node_err,
        "mean_node_surface_error": mean_node_err / max(n_checked, 1),
        "n_surface_nodes_checked": float(n_checked),
        "total_mesh_volume": float(vol.sum()),
    }


def _ligament_report(
    inclusions: Sequence[PackedInclusion], cell_size: float | None
) -> Dict[str, float]:
    if len(inclusions) < 1 or cell_size is None:
        return {}
    centres = np.array([inc.shape.center for inc in inclusions])
    radii = np.array([inc.shape.bounding_radius() for inc in inclusions])

    if len(inclusions) > 1:
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        gap = d - radii[:, None] - radii[None, :]
        np.fill_diagonal(gap, np.inf)
        min_gap = float(gap.min())
    else:
        min_gap = float("inf")

    # Guaranteed matrix elements across a ligament: the macro-boxes are disjoint
    # with a whole-cell margin, so the count is bounded below by the box gap.
    box_gaps = []
    for i, a in enumerate(inclusions):
        for b in inclusions[i + 1 :]:
            sep = np.maximum(
                np.maximum(a.box_lo - (b.box_lo + b.k), b.box_lo - (a.box_lo + a.k)), 0
            )
            box_gaps.append(int(sep.max()))
    return {
        "min_surface_gap": min_gap,
        "min_surface_gap_cells": min_gap / cell_size,
        "min_macrobox_gap_cells": float(min(box_gaps)) if box_gaps else float("inf"),
        "min_ligament_elements": float(min(box_gaps)) if box_gaps else float("inf"),
    }


def _phase_field_report(
    mesh: HexMesh,
    inclusions: Sequence[PackedInclusion],
    cell_size: float | None,
    ell: float | None,
) -> Dict[str, float]:
    if ell is None or ell <= 0:
        return {}
    er = edge_ratio(mesh.points, mesh.cells)  # touch to keep metric coherent
    del er
    vol = element_volumes(mesh.points, mesh.cells)
    h_eff = np.cbrt(np.maximum(vol, 1e-300))
    out = {
        "phase_field_length": float(ell),
        "h_over_ell_max": float(h_eff.max() / ell),
        "h_over_ell_mean": float(h_eff.mean() / ell),
        "h_over_ell_bulk": float((cell_size or h_eff.max()) / ell),
        "fraction_elements_resolving_ell": float(np.mean(h_eff <= 0.5 * ell)),
    }
    if inclusions:
        surf = np.mean(
            [
                float(np.mean(inc.shape.support_radius(np.eye(3)))) * 2.0 / float(np.mean(inc.k))
                for inc in inclusions
            ]
        )
        out["h_over_ell_interface"] = float(surf / ell)
    return out


def recommended_mesh_density(
    length: float,
    radius_mean: float,
    phase_field_length: float | None = None,
    elements_per_radius: float = 4.0,
    elements_per_ell: float = 2.0,
) -> int:
    """Suggest a background division count from geometry and physics.

    Applies the size field both reviews converge on,

    .. math:: h \\le \\min(C_R R,\\; C_\\ell \\ell,\\; h_\\text{bulk}),

    and returns the number of background cells per domain edge.  When
    ``ell << R`` the phase-field term dominates -- which is exactly the point of
    removing voxel staircasing, so that physics rather than geometry sets ``h``.
    """
    h = radius_mean / max(elements_per_radius, 1e-9)
    if phase_field_length:
        h = min(h, phase_field_length / max(elements_per_ell, 1e-9))
    return max(int(np.ceil(length / h)), 4)
