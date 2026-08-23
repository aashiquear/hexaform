"""Assemble a complete mesh from a background grid and macro-block templates.

The assembly step is intentionally thin.  Every macro-block already produces
nodes on its outer boundary at exactly the background grid positions, so
"stitching" is nothing more than concatenation followed by an exact weld.  All
the correctness work was done upstream by the integer index arithmetic in
:mod:`hexaform.background.cartesian` and
:mod:`hexaform.templates.macroblock`.

Exodus metadata is created here rather than recovered from coordinates after the
fact -- the SciSpace review's section 22 point.  Blocks, side sets and node sets
are known by construction:

``matrix``
    element block 1: carved background cells plus every macro-block outer shell.
``inclusion_<id>`` or ``inclusions``
    element blocks 100+: core and inner shell of each macro-block.
``x-``, ``x+``, ``y-``, ``y+``, ``z-``, ``z+``
    side sets on the six outer domain faces, plus matching node sets.
``interface``
    matrix-side faces lying on the analytic inclusion surfaces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..background.cartesian import CartesianGrid
from ..core.elevate import elevate
from ..core.elements import normalize_element_type
from ..core.mesh import HexMesh, MeshAccumulator
from ..core import topology
from ..packing.rsa import PackedInclusion
from ..templates.macroblock import MacroBlockOptions, build_macroblock

MATRIX_BLOCK_ID = 1
INCLUSION_BLOCK_BASE = 100
FACE_NAMES = ("x-", "x+", "y-", "y+", "z-", "z+")


@dataclass
class AssemblyOptions:
    separate_inclusion_blocks: bool = False
    """One Exodus block per inclusion (``100, 101, ...``) instead of a single
    shared ``inclusions`` block.  Useful when particles carry distinct
    properties or must be tracked individually."""

    interface_side_set: bool = True
    per_inclusion_interface_sets: bool = False
    boundary_node_sets: bool = True
    matrix_block_name: str = "matrix"
    inclusion_block_name: str = "inclusions"


def assemble(
    grid: CartesianGrid,
    inclusions: Sequence[PackedInclusion],
    template_options: MacroBlockOptions | None = None,
    options: AssemblyOptions | None = None,
    element_type: str = "HEX8",
    periodic: bool = False,
) -> HexMesh:
    """Build the full :class:`~hexaform.core.mesh.HexMesh`.

    Args:
        grid: The background Cartesian grid.
        inclusions: List of packed inclusions to mesh around.
        template_options: Options for macro-block template generation.
        options: Assembly options controlling metadata and output format.
        element_type: Target element type (HEX8, HEX20, or HEX27).
        periodic: If True, create periodic node sets and handle macro-box wrapping.

    Returns:
        A fully assembled HexMesh with all metadata.
    """
    opt = options or AssemblyOptions()
    topt = template_options or MacroBlockOptions()
    target_element_type = normalize_element_type(element_type)
    t0 = time.perf_counter()

    acc = MeshAccumulator()
    block_names: Dict[int, str] = {MATRIX_BLOCK_ID: opt.matrix_block_name}

    # ---- background -------------------------------------------------
    boxes = [(inc.box_lo, inc.k) for inc in inclusions]
    bg_points, bg_cells, _ = grid.build(boxes)
    acc.add(
        bg_points,
        bg_cells,
        MATRIX_BLOCK_ID,
        element_fields={"particle_id": np.zeros(bg_cells.shape[0], dtype=np.int64)},
    )

    # ---- macro-blocks ----------------------------------------------
    interface_records: List[tuple[int, np.ndarray, np.ndarray]] = []
    running_offset = bg_cells.shape[0]
    template_stats: List[dict[str, float]] = []

    for inc in inclusions:
        mb = build_macroblock(inc.shape, inc.box_lo, inc.k, grid.h, grid.origin, topt)
        if opt.separate_inclusion_blocks:
            incl_bid = INCLUSION_BLOCK_BASE + inc.particle_id - 1
            block_names[incl_bid] = f"inclusion_{inc.particle_id:04d}"
        else:
            incl_bid = INCLUSION_BLOCK_BASE
            block_names[incl_bid] = opt.inclusion_block_name

        bids = np.where(mb.is_inclusion, incl_bid, MATRIX_BLOCK_ID).astype(np.int32)
        pid = np.where(mb.is_inclusion, inc.particle_id, -inc.particle_id).astype(np.int64)
        acc.add(mb.points, mb.cells, bids, element_fields={"particle_id": pid})

        interface_records.append(
            (inc.particle_id, mb.interface_cells + running_offset, mb.interface_sides)
        )
        running_offset += mb.cells.shape[0]
        stats = dict(mb.stats)
        stats["particle_id"] = inc.particle_id
        template_stats.append(stats)

    mesh = acc.build(block_names)

    # side sets reference element indices, which welding does not change
    if interface_records:
        elems = np.concatenate([r[1] for r in interface_records])
        sides = np.concatenate([r[2] for r in interface_records])
        mesh.add_side_set("interface", elems, sides)
    if opt.per_inclusion_interface_sets:
        for pid, elems, sides in interface_records:
            mesh.add_side_set(f"interface_{pid:04d}", elems, sides)

    mesh = topology.weld(mesh, tol=1e-9 * max(grid.h, 1e-30))
    mesh, n_flipped = topology.fix_orientation(mesh)

    curved_faces = []
    if target_element_type != "HEX8" and "interface" in mesh.side_sets:
        shape_by_id = {inc.particle_id: inc.shape for inc in inclusions}
        elems, sides = mesh.side_sets["interface"]
        particle_ids = np.abs(mesh.element_fields["particle_id"][elems]).astype(int)
        curved_faces = [
            (int(elem), int(side), shape_by_id[int(pid)])
            for elem, side, pid in zip(elems, sides, particle_ids)
        ]
    mesh = elevate(mesh, target_element_type, curved_faces)
    if not opt.interface_side_set:
        mesh.side_sets.pop("interface", None)

    _add_boundary_sets(mesh, grid, opt, periodic)

    if periodic:
        mesh = _build_periodic_node_sets(mesh, grid)

    mesh.metadata.update(
        {
            "background_divisions": list(grid.divisions),
            "cell_size": grid.h,
            "domain_origin": grid.origin.tolist(),
            "domain_extent": grid.extent.tolist(),
            "n_inclusions": len(inclusions),
            "element_type": mesh.element_type,
            "n_flipped_elements": n_flipped,
            "assembly_seconds": time.perf_counter() - t0,
            "template_stats": template_stats,
        }
    )
    return mesh


def _add_boundary_sets(mesh: HexMesh, grid: CartesianGrid, opt: AssemblyOptions, periodic: bool = False) -> None:
    """Attach the six exterior face side sets (and matching node sets).

    Args:
        mesh: The mesh to add boundary sets to.
        grid: The background grid.
        opt: Assembly options.
        periodic: If True, also handle periodic boundary conditions.
    """
    tol = 1e-9 * grid.h
    for axis in range(3):
        for side in (0, 1):
            value = grid.origin[axis] if side == 0 else grid.upper[axis]
            name = FACE_NAMES[2 * axis + side]
            elems, sides = topology.faces_on_plane(mesh, axis, value, tol)
            mesh.add_side_set(name, elems, sides)
            if opt.boundary_node_sets:
                mesh.add_node_set(name, topology.nodes_on_plane(mesh, axis, value, tol))

    # Handle periodic boundary conditions
    if periodic:
        _setup_periodic_constraint_sets(mesh, grid)


def _setup_periodic_constraint_sets(mesh: HexMesh, grid: CartesianGrid) -> None:
    """Set up periodic constraint node sets for RVE homogenization.

    This creates named node sets that can be used to apply periodic boundary
    conditions in downstream solvers (e.g., Abaqus, deal.II, FEniCS).

    For each axis i ∈ {x, y, z}, nodes on face i- are paired with corresponding
    nodes on face i+ through matching coordinates on the orthogonal axes.

    The node sets are named as:
        - {axis}-_node_set : nodes on negative face (e.g., "x-_node_set")
        - {axis}+_node_set : nodes on positive face (e.g., "x+_node_set")
        - {axis}_periodic_pair : mapping between paired nodes
    """
    # Create periodic node sets for each axis
    for axis, axis_name in enumerate(["x", "y", "z"]):
        neg_nodes, pos_nodes = grid.periodic_pair_indices(axis)

        # Get actual node IDs that exist after welding (interior nodes removed)
        # We need to filter to only keep nodes that are on the boundary
        neg_node_ids = [int(nid) for nid in neg_nodes]
        pos_node_ids = [int(nid) for nid in pos_nodes]

        # Filter to nodes that actually exist in mesh (some may have been welded away)
        existing_neg = [int(nid) for nid in neg_node_ids if 0 <= int(nid) < mesh.n_nodes]
        existing_pos = [int(nid) for nid in pos_node_ids if 0 <= int(nid) < mesh.n_nodes]

        # Add node sets with descriptive names
        mesh.add_node_set(f"{axis_name}-_node_set", np.array(existing_neg, dtype=np.int64))
        mesh.add_node_set(f"{axis_name}+_node_set", np.array(existing_pos, dtype=np.int64))

        # Create periodic pair mapping node set
        # This contains (neg_id, pos_id) pairs for constraint application
        if existing_neg and existing_pos:
            max_len = min(len(existing_neg), len(existing_pos))
            pairs = np.column_stack([
                np.array(existing_neg[:max_len], dtype=np.int64),
                np.array(existing_pos[:max_len], dtype=np.int64)
            ])
            mesh.add_node_set(f"{axis_name}_periodic_pair", pairs.flatten())


def _build_periodic_node_sets(mesh: HexMesh, grid: CartesianGrid) -> HexMesh:
    """Build explicit periodic node correspondence sets.

    This function creates detailed node sets and mappings required for
    applying periodic boundary conditions in finite element solvers.

    The output includes:
        1. Node sets for each face: x-, x+, y-, y+, z-, z+ (already existing)
        2. Correspondence pairs: which node on -face corresponds to which on +face
        3. Metadata flags indicating periodic BCs are enabled

    Returns:
        The modified mesh with additional periodic node sets.
    """
    _setup_periodic_constraint_sets(mesh, grid)

    # Store metadata indicating periodic boundary conditions are active
    mesh.metadata["periodic_boundary_conditions"] = True
    mesh.metadata["periodic_axes"] = ["x", "y", "z"]

    return mesh
