"""Promote a conforming HEX8 mesh to curved HEX20 or HEX27 elements."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..shapes.base import StarShape
from .elements import (
    HEX27_FACE_CENTER_SIDES,
    HEX8_SIDES,
    HEX_EDGES,
    normalize_element_type,
    side_nodes,
)
from .mesh import HexMesh


def elevate(
    mesh: HexMesh,
    element_type: str,
    curved_faces: Iterable[tuple[int, int, StarShape]] = (),
) -> HexMesh:
    """Elevate a welded HEX8 mesh while preserving global conformity.

    Shared edge and face nodes are allocated from sorted corner-node keys, so
    adjacent elements reference exactly the same higher-order nodes.  Entries in
    ``curved_faces`` are ``(element_index, Exodus_side_id, shape)`` triples;
    newly-created nodes on those faces are projected with ``shape.project()``.
    """
    target = normalize_element_type(element_type)
    if target == mesh.element_type:
        return mesh
    if mesh.element_type != "HEX8":
        raise ValueError(f"can only elevate HEX8 meshes, got {mesh.element_type}")
    if target == "HEX8":
        return mesh

    corners = mesh.cells[:, :8]
    old_n = mesh.n_nodes

    edge_pairs = np.sort(corners[:, np.asarray(HEX_EDGES, dtype=np.int64)], axis=2)
    unique_edges, edge_inverse = np.unique(
        edge_pairs.reshape(-1, 2), axis=0, return_inverse=True
    )
    edge_points = mesh.points[unique_edges].mean(axis=1)
    edge_ids = old_n + edge_inverse.reshape(mesh.n_elements, len(HEX_EDGES))

    point_chunks = [mesh.points, edge_points]
    cell_chunks = [corners, edge_ids]
    interpolation_groups: list[np.ndarray] = [unique_edges]

    if target == "HEX27":
        face_corner_ids = np.stack(
            [corners[:, list(HEX8_SIDES[side - 1])] for side in HEX27_FACE_CENTER_SIDES],
            axis=1,
        )
        face_keys = np.sort(face_corner_ids, axis=2)
        unique_faces, face_inverse = np.unique(
            face_keys.reshape(-1, 4), axis=0, return_inverse=True
        )
        face_points = mesh.points[unique_faces].mean(axis=1)
        face_base = old_n + edge_points.shape[0]
        face_ids = face_base + face_inverse.reshape(mesh.n_elements, 6)

        centre_points = mesh.points[corners].mean(axis=1)
        centre_base = face_base + face_points.shape[0]
        centre_ids = centre_base + np.arange(mesh.n_elements, dtype=np.int64)[:, None]

        # Exodus HEX27 stores the volume centre before the six face centres.
        point_chunks.extend([face_points, centre_points])
        cell_chunks.extend([centre_ids, face_ids])
        interpolation_groups.extend([unique_faces, corners])

    points = np.concatenate(point_chunks, axis=0)
    cells = np.concatenate(cell_chunks, axis=1)

    ownership: dict[int, StarShape] = {}
    for elem, side, shape in curved_faces:
        elem = int(elem)
        side = int(side)
        if elem < 0 or elem >= mesh.n_elements or side < 1 or side > 6:
            raise ValueError(f"invalid curved face ({elem}, {side})")
        local = side_nodes(target)[side - 1][4:]
        for node in cells[elem, list(local)]:
            node = int(node)
            prior = ownership.get(node)
            if prior is not None and prior is not shape:
                raise ValueError("a higher-order node belongs to multiple analytic surfaces")
            ownership[node] = shape

    if ownership:
        by_shape: dict[int, tuple[StarShape, list[int]]] = {}
        for node, shape in ownership.items():
            key = id(shape)
            if key not in by_shape:
                by_shape[key] = (shape, [])
            by_shape[key][1].append(node)
        for shape, nodes in by_shape.values():
            ids = np.asarray(nodes, dtype=np.int64)
            points[ids] = shape.project(points[ids])

    node_fields = _elevate_node_fields(mesh, interpolation_groups)
    metadata = dict(mesh.metadata)
    metadata["element_type"] = target
    metadata["nodes_per_element"] = cells.shape[1]
    metadata["n_higher_order_nodes"] = points.shape[0] - old_n

    return HexMesh(
        points=points,
        cells=cells,
        block_id=mesh.block_id.copy(),
        element_type=target,
        block_names=dict(mesh.block_names),
        node_sets={k: v.copy() for k, v in mesh.node_sets.items()},
        side_sets={k: (e.copy(), s.copy()) for k, (e, s) in mesh.side_sets.items()},
        element_fields={k: v.copy() for k, v in mesh.element_fields.items()},
        node_fields=node_fields,
        metadata=metadata,
    )


def _elevate_node_fields(mesh: HexMesh, groups: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Interpolate existing nodal fields onto new edge/face/centre nodes."""
    out: dict[str, np.ndarray] = {}
    for name, values in mesh.node_fields.items():
        arr = np.asarray(values)
        if arr.shape[0] != mesh.n_nodes:
            raise ValueError(f"node field {name!r} length does not match mesh nodes")
        chunks = [arr]
        for source_nodes in groups:
            chunks.append(arr[source_nodes].mean(axis=1))
        out[name] = np.concatenate(chunks, axis=0)
    return out
