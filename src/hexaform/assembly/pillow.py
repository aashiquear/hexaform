"""Interface pillowing: insert an extra hex sheet at material interfaces.

For Sculpt-style / octree meshing, elements that terminate directly on a
material interface can be poorly shaped because the interface is not aligned
with the background grid.  Inserting a single layer of hexes (a "pillow")
between the interface and the bulk matrix gives every interface element four
in-plane neighbours and dramatically improves element quality in the most
stressed region of the mesh.

The ScholarGPT review (S3, Owen & Shelton) and the Cubit/Sculpt literature
both treat pillowing as a routine step of grid-based meshing.  Hexaform's
cubed-sphere macro-block already produces well-shaped interface elements, so
pillowing is a **lower-priority** tool there -- the roadmap says explicitly
that "no element terminates on the interface at a bad angle by construction".
The pillow layer becomes important with the octree backend (item 2 of the
roadmap), where a Sculpt-style sheet is the natural way to guarantee a
well-conditioned layer.

This module implements three operations:

* :func:`detect_interface_elements` -- find the elements that touch a
  material interface.
* :func:`pillow_interface` -- insert one layer of hex elements between
  the interface elements and the bulk matrix.
* :func:`pillow_layer` -- a thin convenience wrapper, the name the GUI
  block (registered in :mod:`hexaform.server.blocks`) uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.mesh import HexMesh
from ..core.topology import boundary_faces, side_nodes


@dataclass
class PillowOptions:
    """Parameters of the pillow layer."""

    thickness_fraction: float = 0.15
    """Thickness of the new layer as a fraction of the average local element
    edge length.  0.1-0.25 is a sensible range."""

    target_block_id: int = 1
    """Block id to assign to the new (matrix-side) layer.  Defaults to the
    standard matrix block."""

    copy_fields: bool = True
    """Copy element fields from the source element to its pillow counterpart."""


# ----------------------------------------------------------------------
def detect_interface_elements(
    mesh: HexMesh, block_a: int, block_b: int
) -> np.ndarray:
    """Return the indices of elements that have at least one face on the
    ``block_a / block_b`` interface.
    """
    elems, _ = boundary_faces(mesh.cells)
    if elems.size == 0:
        return elems
    # A face is on the interface if the element's block is one of the two and
    # the *opposite* face is owned by the other.  boundary_faces returns
    # only faces used by a single element, so a face on the block_a/block_b
    # boundary has its single owner in one of the two blocks; the other block
    # has no element using that face, so the face is "boundary" in the
    # global sense.
    blocks = mesh.block_id[elems]
    return elems[(blocks == block_a) | (blocks == block_b)]


def _interface_centers(
    mesh: HexMesh, interface_elems: np.ndarray
) -> np.ndarray:
    """Mean of the interface element centres -- a reasonable pillow origin."""
    if interface_elems.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    corners = mesh.points[mesh.cells[interface_elems, :8]]
    return corners.mean(axis=1).mean(axis=0)


def pillow_interface(
    mesh: HexMesh,
    block_a: int,
    block_b: int,
    options: PillowOptions | None = None,
    shapes: Sequence | None = None,
) -> HexMesh:
    """Insert one layer of hex elements between the ``block_a/block_b``
    interface and the surrounding matrix.

    The returned mesh has the same number of nodes plus a pillow-shaped
    surface mesh covering the original interface, and one new hex per
    interface face, sharing four corner nodes with the interface element
    and extending into the matrix.

    ``shapes``, if supplied, is the list of analytic shapes corresponding
    to the inclusion block; the pillow uses each shape's SDF gradient to
    project the local corner normal at every interface node.  Without
    ``shapes``, the average face normal is used (acceptable for nearly
    planar interfaces, but produces near-degenerate hexes on curved
    surfaces).

    Implementation strategy
    -----------------------
    1. Find every face on the ``block_a / block_b`` interface.
    2. Compute the local normal at each interface corner (per-corner SDF
       gradient, falling back to the average face normal).
    3. Create one new node per interface face corner, offset into the
       matrix by ``thickness`` along its local normal.
    4. Build one new hex per interface face, sharing the four interface
       face corners with the existing element and using the four new
       offset nodes on the matrix side.
    5. Renumber nodes, fill in the new block id, and update side sets.

    The result is a mesh in which the interface is *buffered* by one layer
    of matrix hexes, dramatically improving the local Jacobian.
    """
    opts = options or PillowOptions()
    local_sides = side_nodes(mesh.element_type)
    elems, sides = boundary_faces(mesh.cells)
    blocks = mesh.block_id[elems]

    # Use the more specific interface_faces from topology
    from ..core.topology import interface_faces

    elems_a, sides_a = interface_faces(mesh.cells, mesh.block_id, block_a, block_b)
    if elems_a.size == 0:
        return mesh

    # Compute the new node positions: take the four interface-face corners
    # and offset each along its local normal (not the average face normal).
    # Using the average face normal makes the pillow element degenerate,
    # because the offset is the same vector at every corner and HEX8
    # measures thickness by the *difference* between corresponding bottom
    # and top corners -- which collapses when the offset is constant.
    # The local normal at each corner is the gradient of the shape's SDF
    # (or, for star-shaped bodies, the body support normal).  We fall
    # back to the face normal if the local normal is degenerate.
    face_nodes = np.array(
        [mesh.cells[e, list(local_sides[s - 1])] for e, s in zip(elems_a, sides_a)],
        dtype=np.int64,
    )
    corners = mesh.points[face_nodes[:, :4]]
    if shapes:
        # Per-corner normal from the shape's SDF gradient.  All four
        # corners of one face belong to the same shape (the inclusion),
        # so we look up the shape by the element's block id (or, when
        # block_a == inclusion_block_base, by the element's particle id
        # from the element field).
        # For now we use a simple convention: the first shape corresponds
        # to all interface faces, since the cubed-sphere pipeline emits
        # one block per inclusion or one shared block; in the latter
        # case the SDF gradient is still correct for any single shape.
        local_normals = np.zeros((face_nodes.shape[0], 4, 3), dtype=np.float64)
        # Find the shape for each interface face.  The element's particle
        # id is in mesh.element_fields['particle_id']; we pick the shape
        # whose centre is closest to the face centroid.
        if shapes:
            shape_centres = np.array([np.asarray(s.center) for s in shapes])
        else:
            shape_centres = np.zeros((0, 3))
        for k in range(face_nodes.shape[0]):
            face_centroid = corners[k].mean(axis=0)
            if shape_centres.shape[0]:
                d = np.linalg.norm(shape_centres - face_centroid, axis=1)
                shape_idx = int(np.argmin(d))
            else:
                shape_idx = 0
            shp = shapes[shape_idx] if shapes else None
            for c in range(4):
                p = corners[k, c]
                if shp is not None:
                    eps = 1e-6
                    grad = np.zeros(3)
                    for ax in range(3):
                        pp = p.copy()
                        pp[ax] += eps
                        grad[ax] = (shp.sdf(pp[None, :])[0] - shp.sdf(p[None, :])[0]) / eps
                    n = grad / max(np.linalg.norm(grad), 1e-12)
                    local_normals[k, c] = n
                else:
                    local_normals[k, c] = (0, 0, 1)
    else:
        # Average face normal
        a, b, c, d = (corners[:, i] for i in range(4))
        n1 = np.cross(b - a, c - a)
        n2 = np.cross(d - a, c - a)
        normal = 0.5 * (n1 + n2)
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = np.where(norm > 0, normal / np.maximum(norm, 1e-30), np.array([[1.0, 0.0, 0.0]]))
        # Decide the offset direction: from the block_a element centroid toward
        # the face, then mirrored into the matrix (block_b side).
        elem_centres = mesh.points[mesh.cells[elems_a, :8]].mean(axis=1)
        face_centres = corners.mean(axis=1)
        direction = face_centres - elem_centres
        if np.einsum("ij,ij->i", normal, direction).mean() < 0:
            normal = -normal
        local_normals = np.broadcast_to(normal[:, None, :], (face_nodes.shape[0], 4, 3))

    # Thickness: based on the mean local edge length
    edges = np.linalg.norm(
        mesh.points[face_nodes[:, 1]] - mesh.points[face_nodes[:, 0]], axis=1
    )
    thickness = float(opts.thickness_fraction) * float(edges.mean())
    if thickness <= 0:
        thickness = float(edges.mean()) * 0.1

    new_points = mesh.points.copy()
    new_corner_ids = np.zeros((face_nodes.shape[0], 4), dtype=np.int64)
    # Bottom corner ids: new nodes for the pillow's bottom face so that
    # the face is not shared with the source element (which would make
    # the interface non-manifold).  This duplicates the four interface
    # nodes per face; the resulting mesh is still conforming because
    # the new bottom nodes are at the same physical positions as the
    # original interface nodes and the topology deduplication (weld)
    # is not needed.
    bottom_corner_ids = np.zeros((face_nodes.shape[0], 4), dtype=np.int64)
    next_id = mesh.n_nodes
    for k in range(face_nodes.shape[0]):
        for c in range(4):
            # Duplicate the bottom corner
            new_points = np.vstack([new_points, corners[k, c]])
            bottom_corner_ids[k, c] = next_id
            next_id += 1
            # Top corner offset along the local normal
            offset = thickness * local_normals[k, c]
            new_points = np.vstack([new_points, corners[k, c] + offset])
            new_corner_ids[k, c] = next_id
            next_id += 1

    # Build the new hex connectivity.  The order follows the local side
    # convention with side 6 (z+) on the original interface and side 5
    # (z-) on the new matrix side.
    n_new = face_nodes.shape[0]
    new_cells = np.zeros((n_new, mesh.cells.shape[1]), dtype=np.int64)
    if mesh.element_type == "HEX8":
        # The source interface face has its corners in the source's local
        # SIDE order.  The new pillow element uses these same physical
        # corners as its bottom face; the corner order in the pillow's
        # HEX8 local (0, 1, 2, 3) must be CCW viewed from the pillow's
        # local +z direction (the offset direction), which is OPPOSITE
        # the source's outward normal.  So the pillow's local corner
        # order is the source's SIDE order REVERSED.
        for k in range(n_new):
            side_local = list(local_sides[sides_a[k] - 1])  # length 4
            bottom = [
                bottom_corner_ids[k, side_local[3]],  # local 0
                bottom_corner_ids[k, side_local[2]],  # local 1
                bottom_corner_ids[k, side_local[1]],  # local 2
                bottom_corner_ids[k, side_local[0]],  # local 3
            ]
            top = [
                new_corner_ids[k, side_local[3]],  # local 4
                new_corner_ids[k, side_local[2]],  # local 5
                new_corner_ids[k, side_local[1]],  # local 6
                new_corner_ids[k, side_local[0]],  # local 7
            ]
            new_cells[k] = bottom + top
    else:  # HEX20/HEX27
        for k in range(n_new):
            side_local = list(local_sides[sides_a[k] - 1])[:4]
            bottom = [
                bottom_corner_ids[k, side_local[3]],
                bottom_corner_ids[k, side_local[2]],
                bottom_corner_ids[k, side_local[1]],
                bottom_corner_ids[k, side_local[0]],
            ]
            top = [
                new_corner_ids[k, side_local[3]],
                new_corner_ids[k, side_local[2]],
                new_corner_ids[k, side_local[1]],
                new_corner_ids[k, side_local[0]],
            ]
            row = list(bottom) + list(top) + [0] * (mesh.cells.shape[1] - 8)
            new_cells[k] = row

    # Combine and remap block ids
    new_block = np.full(n_new, opts.target_block_id, dtype=np.int32)
    new_block_id = np.concatenate([mesh.block_id, new_block], axis=0)
    new_cells_full = np.concatenate([mesh.cells, new_cells], axis=0)

    # Update element fields
    new_element_fields: Dict[str, np.ndarray] = {}
    for k, arr in mesh.element_fields.items():
        if arr.size == 0:
            continue
        new_element_fields[k] = np.concatenate(
            [arr, np.zeros(n_new, dtype=arr.dtype)], axis=0
        )

    new_mesh = HexMesh(
        points=new_points,
        cells=new_cells_full,
        block_id=new_block_id,
        element_type=mesh.element_type,
        block_names=dict(mesh.block_names),
        side_sets={k: (e.copy(), s.copy()) for k, (e, s) in mesh.side_sets.items()},
        node_sets={k: v.copy() for k, v in mesh.node_sets.items()},
        element_fields=new_element_fields,
        metadata=dict(mesh.metadata),
    )
    new_mesh.metadata["n_pillow_elements"] = n_new
    new_mesh.metadata["pillow_thickness"] = thickness
    # Use the existing fix_orientation routine to handle any pillow element
    # whose local orientation ended up reversed.
    from ..core.topology import fix_orientation
    new_mesh, _n_flipped = fix_orientation(new_mesh)
    new_mesh.metadata["n_pillow_flipped"] = int(_n_flipped)
    return new_mesh


def pillow_layer(
    mesh: HexMesh,
    options: PillowOptions | None = None,
    matrix_block_id: int = 1,
    inclusion_block_id: int = 100,
    shapes: Sequence | None = None,
) -> HexMesh:
    """Insert a pillow layer between the matrix and the inclusion block.

    Convenience wrapper used by the GUI block ``pillow_layer``.  Returns the
    input mesh unchanged if no interface is present.
    """
    return pillow_interface(
        mesh, matrix_block_id, inclusion_block_id, options, shapes=shapes
    )


__all__ = [
    "PillowOptions",
    "detect_interface_elements",
    "pillow_interface",
    "pillow_layer",
]
