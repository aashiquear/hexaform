"""Assembly helpers for the adaptive octree background (Backend B).

The :func:`assemble_with_octree` function is the public entry point that
takes a list of :class:`~hexaform.shapes.base.Shape` (or
:class:`StarShape`) bodies and produces a :class:`HexMesh` with one
element block per material plus the standard exterior side sets.

For star-shaped bodies the per-inclusion macro-block is the right tool
(see :mod:`hexaform.assembly.builder`); the octree is the general
fallback for **non-star-shaped** bodies and for tight ligaments.  The
two paths compose: the macro-block still gives the inclusion block its
boundary layer, and the octree fills the matrix around it.

This module also provides :func:`project_surface_nodes`, the post-step
that snaps every interface node to the exact analytic body.  It is the
operation the ScholarGPT review (S7) calls "analytic sphere projection",
generalised to any SDF-defined body.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..background.octree import OctreeGrid, sphere_material_of, sphere_size_field
from ..core.elements import normalize_element_type
from ..core.mesh import HexMesh
from ..core import topology as topology_mod
from ..shapes.base import Shape, StarShape, normalize

MATRIX_BLOCK_ID = 1
INCLUSION_BLOCK_BASE = 100


@dataclass
class OctreeAssemblyOptions:
    """Tunable parameters for the octree assembly."""

    bulk_h: float = 0.05
    """Cell size far from any material interface."""

    interface_h: float = 0.02
    """Cell size inside the interface refinement band."""

    interface_band: float = 0.05
    """Width of the interface refinement band around every material boundary."""

    max_level: int = 6
    """Hard cap on the octree depth."""

    project_surface_nodes: bool = True
    """If True, every node whose ``|sdf| < surface_tol`` is snapped onto the
    analytic body after the assembly."""

    surface_tol: float = 1e-7
    """Numerical tolerance for the surface-node test."""


# ----------------------------------------------------------------------
def assemble_with_octree(
    shapes: Sequence[Shape],
    domain_length: float = 1.0,
    domain_origin: np.ndarray | None = None,
    options: OctreeAssemblyOptions | None = None,
) -> HexMesh:
    """Build a mesh using the adaptive octree background.

    The octree refines inside a band of ``interface_band`` around every
    material interface (defined by the implicit ``sdf`` of each body).
    The returned mesh contains the matrix as block 1 and one block per
    inclusion (100, 101, ...).

    This is the lower-level entry point; the cubed-sphere macroblock is
    still recommended for star-shaped bodies because it produces
    higher-quality interface elements.
    """
    if not shapes:
        return _empty_octree_mesh(options)
    opts = options or OctreeAssemblyOptions()

    origin = np.zeros(3) if domain_origin is None else np.asarray(domain_origin, dtype=np.float64).reshape(3)

    # Build the size field and material classifier from the shapes.
    centres, radii, surface_projectors = _shape_descriptors(shapes)
    size_f = sphere_size_field(centres, radii, opts.bulk_h, opts.interface_h, opts.interface_band)
    mat_f = sphere_material_of(centres, radii)

    grid = OctreeGrid(
        origin=origin,
        length=domain_length,
        h_min=opts.interface_h,
        max_level=opts.max_level,
        material_of=mat_f,
        size_field=size_f,
    )
    points, cells, _idx = grid.build()

    # Classify each cell into matrix or inclusion
    cell_centres = points[cells].mean(axis=1)
    cell_mats = mat_f(cell_centres)
    block_id = np.where(cell_mats > 0, INCLUSION_BLOCK_BASE + cell_mats - 1, MATRIX_BLOCK_ID).astype(np.int32)

    # Build the mesh
    block_names = {MATRIX_BLOCK_ID: "matrix"}
    for i, _ in enumerate(shapes, start=1):
        bid = INCLUSION_BLOCK_BASE + i - 1
        block_names[bid] = f"inclusion_{i:04d}"

    mesh = HexMesh(
        points=points,
        cells=cells,
        block_id=block_id,
        element_type="HEX8",
        block_names=block_names,
        metadata={
            "background": "octree",
            "experimental": True,
            "conforming_transition_templates": False,
            "warning": (
                "Adaptive leaves may contain hanging nodes; analytic projection can "
                "invert interface cells. Run full validation before using this prototype."
            ),
            "h_min": opts.interface_h,
            "h_bulk": opts.bulk_h,
            "interface_band": opts.interface_band,
            "max_level": opts.max_level,
        },
    )

    if opts.project_surface_nodes:
        mesh = project_surface_nodes(mesh, shapes, tol=opts.surface_tol, projectors=surface_projectors)
    _attach_boundary_sets(mesh, origin, domain_length)
    return mesh


# ----------------------------------------------------------------------
def _shape_descriptors(
    shapes: Sequence[Shape],
) -> Tuple[np.ndarray, np.ndarray, List[Optional[Callable[[np.ndarray], np.ndarray]]]]:
    """Return (centres, radii, projector_callables) for a sequence of shapes.

    For star-shaped bodies we use the centre and the bounding radius; the
    projection falls back to ``Shape.project`` (which by default only
    exists on :class:`StarShape`).

    For non-star-shaped bodies the centre and bounding radius still
    describe the macro geometry, but the projection is taken from the
    shape itself (e.g. :meth:`Torus.project`).
    """
    centres = np.array([np.asarray(s.center) for s in shapes], dtype=np.float64)
    radii = np.array([s.bounding_radius() for s in shapes], dtype=np.float64)
    projectors: List[Optional[Callable[[np.ndarray], np.ndarray]]] = []
    for s in shapes:
        if hasattr(s, "project"):
            projectors.append(s.project)
        else:
            projectors.append(None)
    return centres, radii, projectors


# ----------------------------------------------------------------------
def project_surface_nodes(
    mesh: HexMesh,
    shapes: Sequence[Shape],
    tol: float = 1e-7,
    projectors: Sequence[Optional[Callable[[np.ndarray], np.ndarray]]] | None = None,
) -> HexMesh:
    """Snap every interface node onto the analytic surface.

    A node is "on the surface" if its signed distance to any shape is below
    ``tol``.  The node is then moved to the closest point on that shape.

    This is the operation the ScholarGPT review identifies as the "analytic
    sphere projection" step (section 6 of the Cubit/Sculpt workflow).  Here
    it works for any body that exposes ``project`` -- the cubed-sphere
    pipeline already does this for star shapes; the octree extends it to
    non-star shapes like :class:`~hexaform.shapes.torus.Torus`.
    """
    pts = mesh.points.copy()
    if projectors is None:
        projectors = [
            s.project if hasattr(s, "project") else None for s in shapes
        ]
    n_projected = 0
    for shape, projector in zip(shapes, projectors):
        if projector is None:
            continue
        # Use sdf (cheap) to detect candidates, then project them all at once.
        # The selection band is ``max(tol, local_cell_size)``: a node is a
        # candidate if it lies within one cell of the surface.  We estimate
        # the local cell size from the octree's metadata.
        sdf = np.asarray(shape.sdf(pts), dtype=np.float64)
        cell_size = float(mesh.metadata.get("h_min", tol) or tol)
        near_band = max(tol * 10.0, cell_size)
        near = np.flatnonzero(np.abs(sdf) < near_band)
        if near.size == 0:
            continue
        # Refine: take only those where projection moves the point
        original = pts[near].copy()
        projected = projector(original)
        delta = np.linalg.norm(projected - original, axis=1)
        moved = np.flatnonzero(delta > 1e-12)
        if moved.size == 0:
            continue
        # Only accept nodes whose distance after projection is below the
        # original sdf band (i.e., the projection actually moved the node
        # onto the surface).
        sdf_after = np.asarray(shape.sdf(projected[moved]), dtype=np.float64)
        accept = np.flatnonzero(np.abs(sdf_after) < near_band)
        if accept.size == 0:
            continue
        pts[near[moved[accept]]] = projected[moved[accept]]
        n_projected += int(accept.size)
    new_mesh = HexMesh(
        points=pts,
        cells=mesh.cells,
        block_id=mesh.block_id,
        element_type=mesh.element_type,
        block_names=dict(mesh.block_names),
        side_sets={k: (e.copy(), s.copy()) for k, (e, s) in mesh.side_sets.items()},
        node_sets={k: v.copy() for k, v in mesh.node_sets.items()},
        element_fields={k: v.copy() for k, v in mesh.element_fields.items()},
        metadata=dict(mesh.metadata),
    )
    if n_projected:
        new_mesh.metadata["n_surface_nodes_projected"] = n_projected
    return new_mesh


# ----------------------------------------------------------------------
def _attach_boundary_sets(mesh: HexMesh, origin: np.ndarray, length: float) -> None:
    """Attach the six exterior side sets to the octree mesh."""
    pts = mesh.points
    tol = max(1e-9 * length, 1e-12)
    for axis in range(3):
        for side, value in ((0, origin[axis]), (1, origin[axis] + length)):
            elems, sides = topology_mod.faces_on_plane(mesh, axis, value, tol)
            name = ("x-", "x+", "y-", "y+", "z-", "z+")[2 * axis + side]
            mesh.add_side_set(name, elems, sides)
            mesh.add_node_set(name, topology_mod.nodes_on_plane(mesh, axis, value, tol))


def _empty_octree_mesh(options: OctreeAssemblyOptions | None) -> HexMesh:
    return HexMesh(
        points=np.zeros((0, 3)),
        cells=np.zeros((0, 8), dtype=np.int64),
        block_id=np.zeros((0,), dtype=np.int32),
        element_type="HEX8",
        metadata={"background": "octree"},
    )


__all__ = [
    "OctreeAssemblyOptions",
    "assemble_with_octree",
    "project_surface_nodes",
]
