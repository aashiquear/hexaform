"""Solver-agnostic hexahedral mesh container.

The :class:`HexMesh` is the single data structure that every stage of the
Hexaform pipeline produces or consumes.  Backends (template mesher, background
grid, future octree mesher) all emit ``HexMesh`` objects; writers consume them.

Corner-node ordering follows the Exodus II ``HEX8`` convention; HEX20/HEX27
append nodes in canonical Exodus order::

        7-------6
       /|      /|          zeta
      4-------5 |          |  eta
      | |     | |          | /
      | 3-----|-2          |/
      |/      |/           o----- xi
      0-------1

with nodes 0..3 on the ``zeta = -1`` face (counter-clockwise seen from
``zeta = +1``) and nodes 4..7 the corresponding nodes on ``zeta = +1``.
Under this ordering the Jacobian determinant is positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .elements import ELEMENT_TYPES, HEX8_SIDES, element_type_from_width, normalize_element_type


@dataclass
class HexMesh:
    """An all-hexahedral mesh with Exodus-ready metadata.

    Attributes
    ----------
    points:
        ``(n_nodes, 3)`` float64 nodal coordinates.
    cells:
        ``(n_elems, 8|20|27)`` int64 zero-based Exodus connectivity.
    block_id:
        ``(n_elems,)`` int32 element block id for every element.
    element_type:
        Homogeneous ``HEX8``, ``HEX20`` or ``HEX27`` interpolation.
    block_names:
        Mapping ``block_id -> name`` written to the Exodus ``eb_names`` variable.
    node_sets:
        Mapping ``name -> (n,) int64`` array of zero-based node indices.
    side_sets:
        Mapping ``name -> (elem_idx, side_id)`` where ``elem_idx`` is a
        zero-based element index array and ``side_id`` is a 1-based Exodus side
        number in ``1..6``.
    element_fields / node_fields:
        Optional named arrays written as Exodus element/nodal variables at the
        first time step.  Used for e.g. per-element ``particle_id``.
    metadata:
        Free-form provenance dictionary (seed, parameters, timings).  Serialized
        alongside the mesh by :mod:`hexaform.io.report`, not into Exodus.
    """

    points: np.ndarray
    cells: np.ndarray
    block_id: np.ndarray
    block_names: Dict[int, str] = field(default_factory=dict)
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    side_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    element_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    node_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    element_type: str = ""

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.cells = np.ascontiguousarray(self.cells, dtype=np.int64)
        self.block_id = np.ascontiguousarray(self.block_id, dtype=np.int32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (n, 3), got {self.points.shape}")
        if self.cells.ndim != 2:
            raise ValueError(f"cells must be a 2-D array, got {self.cells.shape}")
        inferred = element_type_from_width(self.cells.shape[1])
        self.element_type = normalize_element_type(self.element_type or inferred)
        if ELEMENT_TYPES[self.element_type] != self.cells.shape[1]:
            raise ValueError(
                f"{self.element_type} requires {ELEMENT_TYPES[self.element_type]} nodes per "
                f"element, got {self.cells.shape[1]}"
            )
        if self.block_id.shape[0] != self.cells.shape[0]:
            raise ValueError("block_id length must match number of elements")

    @classmethod
    def empty(cls) -> "HexMesh":
        return cls(
            points=np.zeros((0, 3)),
            cells=np.zeros((0, 8), dtype=np.int64),
            block_id=np.zeros((0,), dtype=np.int32),
            element_type="HEX8",
        )

    # ------------------------------------------------------------------
    # basic queries
    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.cells.shape[0])

    @property
    def nodes_per_element(self) -> int:
        return ELEMENT_TYPES[self.element_type]

    @property
    def block_ids(self) -> np.ndarray:
        return np.unique(self.block_id)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_nodes == 0:
            return np.zeros(3), np.zeros(3)
        return self.points.min(axis=0), self.points.max(axis=0)

    def block_name(self, bid: int) -> str:
        return self.block_names.get(int(bid), f"block_{int(bid)}")

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def add_side_set(self, name: str, elems: np.ndarray, sides: np.ndarray) -> None:
        elems = np.asarray(elems, dtype=np.int64).ravel()
        sides = np.asarray(sides, dtype=np.int32).ravel()
        if elems.shape != sides.shape:
            raise ValueError("side set element and side arrays must have equal length")
        if elems.size and (sides.min() < 1 or sides.max() > 6):
            raise ValueError("Exodus hexahedron side ids must lie in 1..6")
        self.side_sets[name] = (elems, sides)

    def add_node_set(self, name: str, nodes: np.ndarray) -> None:
        self.node_sets[name] = np.unique(np.asarray(nodes, dtype=np.int64).ravel())

    def summary(self) -> str:
        lo, hi = self.bounding_box()
        lines = [
            f"HexMesh: {self.n_nodes} nodes, {self.n_elements} {self.element_type} elements",
            f"  bbox: [{lo[0]:.4g}, {lo[1]:.4g}, {lo[2]:.4g}] .. "
            f"[{hi[0]:.4g}, {hi[1]:.4g}, {hi[2]:.4g}]",
            "  blocks:",
        ]
        for bid in self.block_ids:
            count = int(np.count_nonzero(self.block_id == bid))
            lines.append(f"    {int(bid):>6}  {self.block_name(bid):<24} {count:>9} elems")
        if self.side_sets:
            lines.append("  side sets:")
            for name, (elems, _) in self.side_sets.items():
                lines.append(f"    {name:<28} {elems.size:>9} faces")
        if self.node_sets:
            lines.append("  node sets:")
            for name, nodes in self.node_sets.items():
                lines.append(f"    {name:<28} {nodes.size:>9} nodes")
        return "\n".join(lines)


class MeshAccumulator:
    """Incrementally concatenates independently generated hex blocks.

    Each contributed patch carries its own local node numbering; the
    accumulator offsets connectivity and defers node welding to
    :func:`hexaform.core.topology.weld`.  This keeps every template generator
    completely local and unaware of global numbering.
    """

    def __init__(self) -> None:
        self._points: list[np.ndarray] = []
        self._cells: list[np.ndarray] = []
        self._blocks: list[np.ndarray] = []
        self._fields: Dict[str, list[np.ndarray]] = {}
        self._offset = 0

    def add(
        self,
        points: np.ndarray,
        cells: np.ndarray,
        block_id: int | np.ndarray,
        element_fields: Dict[str, np.ndarray] | None = None,
    ) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 8)
        n_el = cells.shape[0]
        if np.isscalar(block_id):
            blocks = np.full(n_el, int(block_id), dtype=np.int32)
        else:
            blocks = np.asarray(block_id, dtype=np.int32).ravel()
        self._points.append(points)
        self._cells.append(cells + self._offset)
        self._blocks.append(blocks)
        self._offset += points.shape[0]

        fields = element_fields or {}
        # Keep every field dense across all contributions (zero-filled default).
        for key in set(self._fields) | set(fields):
            arr = fields.get(key)
            if arr is None:
                arr = np.zeros(n_el, dtype=np.int64)
            else:
                arr = np.asarray(arr).ravel()
                if arr.size == 1:
                    arr = np.full(n_el, arr[0])
            self._fields.setdefault(key, [])
            # back-fill earlier contributions that never saw this key
            missing = len(self._cells) - 1 - len(self._fields[key])
            for _ in range(missing):
                self._fields[key].append(np.zeros(0, dtype=arr.dtype))
            self._fields[key].append(arr)

    def build(self, block_names: Dict[int, str] | None = None) -> HexMesh:
        if not self._points:
            return HexMesh.empty()
        # pad any field lists that fall short of the contribution count
        n_contrib = len(self._cells)
        for key, chunks in self._fields.items():
            while len(chunks) < n_contrib:
                chunks.append(np.zeros(0, dtype=np.int64))
        counts = [c.shape[0] for c in self._cells]
        fields = {}
        for key, chunks in self._fields.items():
            fixed = [
                c if c.size == n else np.zeros(n, dtype=c.dtype if c.size else np.int64)
                for c, n in zip(chunks, counts)
            ]
            fields[key] = np.concatenate(fixed) if fixed else np.zeros(0)
        return HexMesh(
            points=np.concatenate(self._points, axis=0),
            cells=np.concatenate(self._cells, axis=0),
            block_id=np.concatenate(self._blocks, axis=0),
            block_names=dict(block_names or {}),
            element_fields=fields,
        )
