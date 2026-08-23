"""Topological operations on hexahedral meshes.

The functions here are deliberately independent of how a mesh was produced.
They provide the guarantees the rest of the pipeline relies on:

* :func:`weld` merges coincident nodes contributed by independent templates,
* :func:`boundary_faces` / :func:`interface_faces` build Exodus side sets,
* :func:`fix_orientation` guarantees positive Jacobians,
* :func:`check_conformity` detects hanging nodes and non-manifold faces.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .elements import HEX8_SIDES, ORIENTATION_FLIP, element_type_from_width, side_nodes
from .mesh import HexMesh


def weld(mesh: HexMesh, tol: float = 1e-9) -> HexMesh:
    """Merge nodes closer than ``tol`` and drop unreferenced nodes.

    Uses lattice quantisation rather than a spatial tree: every coordinate is
    snapped to a grid of size ``tol`` and identical keys are merged.  Because
    Hexaform generates coincident nodes from *identical* arithmetic expressions
    (background grid nodes and macro-box surface nodes are both ``origin + i*h``
    with integer ``i``), the coordinates agree bit-for-bit and quantisation is
    exact rather than heuristic.

    A two-pass scheme with a half-cell offset is used so that pairs of nodes
    that straddle a lattice boundary are still merged.
    """
    pts = mesh.points
    if pts.shape[0] == 0:
        return mesh

    def _keys(offset: float) -> np.ndarray:
        return np.round(pts / tol + offset).astype(np.int64)

    remap = np.arange(pts.shape[0], dtype=np.int64)
    for offset in (0.0, 0.5):
        keys = _keys(offset)
        keys = keys[remap]  # follow current merge state
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        remap = remap[first][inverse]

    # compact the surviving node ids
    used = np.unique(remap[np.unique(mesh.cells)])
    compact = -np.ones(pts.shape[0], dtype=np.int64)
    compact[used] = np.arange(used.size, dtype=np.int64)

    new_points = pts[used]
    new_cells = compact[remap[mesh.cells]]
    if np.any(new_cells < 0):
        raise RuntimeError("welding produced dangling connectivity")

    def _remap_nodes(arr: np.ndarray) -> np.ndarray:
        return np.unique(compact[remap[np.asarray(arr, dtype=np.int64)]])

    return HexMesh(
        points=new_points,
        cells=new_cells,
        block_id=mesh.block_id.copy(),
        element_type=mesh.element_type,
        block_names=dict(mesh.block_names),
        node_sets={k: _remap_nodes(v) for k, v in mesh.node_sets.items()},
        side_sets={k: (e.copy(), s.copy()) for k, (e, s) in mesh.side_sets.items()},
        element_fields={k: v.copy() for k, v in mesh.element_fields.items()},
        node_fields={k: np.asarray(v)[used].copy() for k, v in mesh.node_fields.items()},
        metadata=dict(mesh.metadata),
    )


def element_faces(cells: np.ndarray) -> np.ndarray:
    """Return all local nodes on every element face in Exodus side order."""
    cells = np.asarray(cells)
    sides = side_nodes(element_type_from_width(cells.shape[1]))
    return np.stack([cells[:, list(side)] for side in sides], axis=1)


def _face_table(cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted face keys plus their owning element and local side (1-based)."""
    faces = element_faces(cells)
    faces = faces.reshape(-1, faces.shape[-1])
    owner = np.repeat(np.arange(cells.shape[0], dtype=np.int64), 6)
    side = np.tile(np.arange(1, 7, dtype=np.int32), cells.shape[0])
    keys = np.sort(faces, axis=1)
    return keys, owner, side


def boundary_faces(cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Element indices and 1-based side ids of faces used by exactly one element."""
    keys, owner, side = _face_table(cells)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    mask = counts[inverse] == 1
    return owner[mask], side[mask]


def interface_faces(
    cells: np.ndarray, block_id: np.ndarray, block_a: int, block_b: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Faces shared between two element blocks, oriented outward from ``block_a``.

    This is how the matrix/inclusion interface side set is recovered generically,
    independent of the template that produced it.
    """
    keys, owner, side = _face_table(cells)
    block_of_face = np.asarray(block_id)[owner]
    order = np.lexsort(keys.T[::-1])
    keys_s, owner_s, side_s, blk_s = keys[order], owner[order], side[order], block_of_face[order]

    same_as_next = np.all(keys_s[:-1] == keys_s[1:], axis=1)
    idx = np.flatnonzero(same_as_next)
    pair_blocks = np.stack([blk_s[idx], blk_s[idx + 1]], axis=1)
    hit = ((pair_blocks[:, 0] == block_a) & (pair_blocks[:, 1] == block_b)) | (
        (pair_blocks[:, 0] == block_b) & (pair_blocks[:, 1] == block_a)
    )
    idx = idx[hit]
    take_first = blk_s[idx] == block_a
    elems = np.where(take_first, owner_s[idx], owner_s[idx + 1])
    sides = np.where(take_first, side_s[idx], side_s[idx + 1])
    return elems.astype(np.int64), sides.astype(np.int32)


def faces_on_plane(
    mesh: HexMesh, axis: int, value: float, tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Boundary faces whose four nodes all lie on ``x[axis] == value``."""
    elems, sides = boundary_faces(mesh.cells)
    local_sides = side_nodes(mesh.element_type)
    nodes = np.stack(
        [mesh.cells[e, list(local_sides[s - 1])] for e, s in zip(elems, sides)]
    ) if elems.size else np.zeros((0, len(local_sides[0])), dtype=np.int64)
    if elems.size == 0:
        return elems, sides
    coords = mesh.points[nodes, axis]
    mask = np.all(np.abs(coords - value) <= tol, axis=1)
    return elems[mask], sides[mask]


def nodes_on_plane(mesh: HexMesh, axis: int, value: float, tol: float = 1e-8) -> np.ndarray:
    return np.flatnonzero(np.abs(mesh.points[:, axis] - value) <= tol)


def signed_volumes(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed physical volume of every hexahedron."""
    cells = np.asarray(cells)
    if cells.shape[1] > 8:
        from .quality import element_volumes

        return element_volumes(points, cells)
    p = points[cells[:, :8]]
    centroid = p.mean(axis=1)
    total = np.zeros(cells.shape[0])
    for side in HEX8_SIDES:
        f = p[:, list(side), :]
        fc = f.mean(axis=1)
        for k in range(4):
            a = f[:, k, :] - centroid
            b = f[:, (k + 1) % 4, :] - centroid
            c = fc - centroid
            total += np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return total


def fix_orientation(mesh: HexMesh) -> Tuple[HexMesh, int]:
    """Flip inverted hexahedra so all signed volumes are positive."""
    vol = signed_volumes(mesh.points, mesh.cells)
    bad = vol < 0
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        cells = mesh.cells.copy()
        cells[bad] = cells[bad][:, ORIENTATION_FLIP[mesh.element_type]]
        side_sets = {}
        for name, (elems, sides) in mesh.side_sets.items():
            elems = elems.copy()
            sides = sides.copy()
            affected = bad[elems]
            old_sides = sides.copy()
            sides[affected & (old_sides == 5)] = 6
            sides[affected & (old_sides == 6)] = 5
            side_sets[name] = (elems, sides)
        mesh = HexMesh(
            points=mesh.points,
            cells=cells,
            block_id=mesh.block_id,
            element_type=mesh.element_type,
            block_names=mesh.block_names,
            node_sets=mesh.node_sets,
            side_sets=side_sets,
            element_fields=mesh.element_fields,
            node_fields=mesh.node_fields,
            metadata=mesh.metadata,
        )
    return mesh, n_bad


def check_conformity(mesh: HexMesh) -> Dict[str, object]:
    """Detect duplicated elements, non-manifold faces and hanging nodes.

    A hanging node is found by testing whether any node lies in the interior of
    a boundary face it does not belong to.  Because Hexaform's macro-boxes are
    aligned to the background grid, a correct build reports zero of these; a
    non-zero count means a template resolution did not match its host cell.
    """
    keys, owner, _ = _face_table(mesh.cells)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    face_counts = counts[inverse]

    report: Dict[str, object] = {
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "n_duplicate_elements": int(
            mesh.n_elements - np.unique(np.sort(mesh.cells, axis=1), axis=0).shape[0]
        ),
        "n_degenerate_elements": int(
            np.count_nonzero(
                [len(set(c.tolist())) < mesh.nodes_per_element for c in mesh.cells]
            )
        ),
        "n_nonmanifold_faces": int(np.count_nonzero(face_counts > 2) // 1),
        "n_boundary_faces": int(np.count_nonzero(face_counts == 1)),
        "n_unused_nodes": int(mesh.n_nodes - np.unique(mesh.cells).size),
    }

    report["n_hanging_nodes"] = _count_hanging_nodes(mesh)
    report["conforming"] = (
        report["n_hanging_nodes"] == 0
        and report["n_nonmanifold_faces"] == 0
        and report["n_degenerate_elements"] == 0
        and report["n_duplicate_elements"] == 0
        and report["n_unused_nodes"] == 0
    )
    return report


def _count_hanging_nodes(mesh: HexMesh, tol: float = 1e-9) -> int:
    """Count boundary nodes that lie strictly inside another boundary face.

    Restricting the test to the boundary is sufficient: an interior hanging node
    necessarily creates a boundary-like (once-used) face pair, which shows up in
    the surface extracted below.
    """
    from scipy.spatial import cKDTree  # local import keeps core import-light

    elems, sides = boundary_faces(mesh.cells)
    if elems.size == 0:
        return 0
    local_sides = side_nodes(mesh.element_type)
    faces = np.stack(
        [mesh.cells[e, list(local_sides[s - 1])] for e, s in zip(elems, sides)]
    )
    surf_nodes = np.unique(faces)
    pts = mesh.points
    node_pts = pts[surf_nodes]

    fp = pts[faces[:, :4]]  # corner quads bound the hanging-node search
    centres = fp.mean(axis=1)
    radii = np.linalg.norm(fp - centres[:, None, :], axis=2).max(axis=1)

    tree = cKDTree(node_pts)
    neighbours = tree.query_ball_point(centres, radii + 1e-9)

    face_sets = [set(f.tolist()) for f in faces]
    hanging = 0
    for f, cand in enumerate(neighbours):
        if not cand:
            continue
        cand = np.asarray(cand, dtype=np.int64)
        cand = cand[~np.isin(surf_nodes[cand], list(face_sets[f]))]
        if cand.size == 0:
            continue
        quad = fp[f]
        n = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        norm = np.linalg.norm(n)
        if norm < tol:
            continue
        n = n / norm
        on_plane = cand[np.abs((node_pts[cand] - quad[0]) @ n) <= 1e-8]
        if on_plane.size == 0:
            continue
        hanging += int(np.count_nonzero(_points_in_quad(node_pts[on_plane], quad, n)))
    return hanging


def _points_in_quad(p: np.ndarray, quad: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Boolean mask: is each coplanar point strictly inside the quad?"""
    inside = np.ones(p.shape[0], dtype=bool)
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        edge = b - a
        cross = np.cross(edge, p - a)
        inside &= (cross @ normal) > 1e-10
    return inside
