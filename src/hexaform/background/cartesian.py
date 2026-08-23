"""Uniform Cartesian background grid with integer macro-box carving.

The background carries the matrix material away from inclusions.  It is
deliberately the dumbest possible mesher -- a structured grid -- because all the
geometric intelligence lives in the macro-block templates.  Its only real job is
to hand back a cavity whose surface nodes coincide *exactly* with the outer
layer of a macro-block.

Exactness is achieved by keeping everything in integer index space: a cell is
identified by its lower index triple, a node position is always evaluated as
``origin + index * h``, and carving is a pure integer set operation.  No
tolerance is involved anywhere, so the subsequent weld is exact rather than
approximate.

A future adaptive (octree) background is the natural drop-in replacement here;
see ``docs/roadmap.md``.  The interface it must satisfy is exactly the one
below: produce hexes, and expose the node ids on each carved cavity face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class CartesianGrid:
    """A ``divisions``-cubed uniform grid over ``[origin, origin + extent]``."""

    divisions: Tuple[int, int, int]
    origin: np.ndarray
    h: float

    def __post_init__(self) -> None:
        self.divisions = tuple(int(d) for d in np.broadcast_to(self.divisions, (3,)))
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.h = float(self.h)
        if min(self.divisions) < 1:
            raise ValueError("grid divisions must be >= 1")

    # ------------------------------------------------------------------
    @classmethod
    def cube(cls, length: float, divisions: int, origin: Sequence[float] | None = None) -> "CartesianGrid":
        """Grid over a cube of edge ``length`` with ``divisions`` cells per edge."""
        org = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        return cls((divisions, divisions, divisions), org, length / divisions)

    @property
    def node_shape(self) -> Tuple[int, int, int]:
        return tuple(d + 1 for d in self.divisions)  # type: ignore[return-value]

    @property
    def extent(self) -> np.ndarray:
        return np.array(self.divisions, dtype=float) * self.h

    @property
    def upper(self) -> np.ndarray:
        return self.origin + self.extent

    def node_coordinates(self) -> np.ndarray:
        """``(n_nodes, 3)`` positions in lexicographic ``(i, j, k)`` order."""
        nx, ny, nz = self.node_shape
        i = np.arange(nx, dtype=np.int64)
        j = np.arange(ny, dtype=np.int64)
        k = np.arange(nz, dtype=np.int64)
        idx = np.stack(np.meshgrid(i, j, k, indexing="ij"), axis=-1)
        return (self.origin + idx * self.h).reshape(-1, 3)

    def node_id(self, idx: np.ndarray) -> np.ndarray:
        """Flat node id from an integer index triple (or array of triples)."""
        nx, ny, nz = self.node_shape
        idx = np.asarray(idx, dtype=np.int64)
        return (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]

    def contains_box(self, lo: np.ndarray, k: int) -> bool:
        lo = np.asarray(lo, dtype=np.int64)
        return bool(np.all(lo >= 0) and np.all(lo + k <= np.array(self.divisions)))

    def point_to_index(self, x: np.ndarray) -> np.ndarray:
        """Nearest *node* index to a physical point."""
        return np.rint((np.asarray(x, dtype=float) - self.origin) / self.h).astype(np.int64)

    # ------------------------------------------------------------------
    def build(
        self, carve_boxes: Iterable[Tuple[np.ndarray, np.ndarray | int]] = ()
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(points, cells, cell_index)`` with the given boxes removed.

        ``carve_boxes`` is an iterable of ``(lower_index, k)`` pairs, where
        ``k`` is a scalar (cubic box) or a per-axis ``(kx, ky, kz)`` vector.
        The returned ``points`` array still contains every grid node (including
        the now-unused ones inside cavities); the global weld pass drops them.
        Keeping the full node array preserves the exact index-to-id relation
        that makes the macro-block junction exact.
        """
        nx, ny, nz = self.divisions
        alive = np.ones((nx, ny, nz), dtype=bool)
        for lo, k in carve_boxes:
            lo = np.asarray(lo, dtype=np.int64).reshape(3)
            k = np.broadcast_to(np.asarray(k, dtype=np.int64), (3,))
            sl = tuple(slice(int(lo[a]), int(lo[a] + k[a])) for a in range(3))
            alive[sl] = False

        ci, cj, ck = np.nonzero(alive)
        cell_index = np.stack([ci, cj, ck], axis=1)

        offsets = np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            dtype=np.int64,
        )
        corner_idx = cell_index[:, None, :] + offsets[None, :, :]
        cells = self.node_id(corner_idx)
        return self.node_coordinates(), cells.astype(np.int64), cell_index

    # ------------------------------------------------------------------
    def boundary_node_indices(self, axis: int, side: int) -> np.ndarray:
        """Flat node ids on one outer face of the grid."""
        nx, ny, nz = self.node_shape
        grids = [np.arange(nx), np.arange(ny), np.arange(nz)]
        grids[axis] = np.array([0 if side == 0 else self.divisions[axis]])
        idx = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
        return self.node_id(idx)

    # ------------------------------------------------------------------
    # Periodic boundary conditions support
    def periodic_pair_indices(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Return paired node indices for periodic boundary on given axis.

        Returns:
            A tuple (neg_nodes, pos_nodes) where each index in neg_nodes at position
            i corresponds to the node at pos_nodes[i] across the periodic boundary.

        The pairing is done by matching coordinates on all axes except the current
        one. Both arrays have the same length and are ordered consistently.
        """
        nx, ny, nz = self.node_shape

        # Get indices on both faces
        neg_indices = []
        pos_indices = []

        # Create index grids for all dimensions
        i_vals = np.arange(nx)
        j_vals = np.arange(ny)
        k_vals = np.arange(nz)

        # For each combination of the two non-axis dimensions, pair (0, j, k) with (nx-1, j, k)
        for j in j_vals:
            for k in k_vals:
                # Node on negative face (axis=0)
                neg_idx = np.array([0, j, k])
                neg_indices.append(neg_idx.tolist())

                # Node on positive face (axis=max)
                pos_idx = np.array([nx - 1, j, k])
                pos_indices.append(pos_idx.tolist())

        neg_idx_array = np.array(neg_indices, dtype=np.int64)
        pos_idx_array = np.array(pos_indices, dtype=np.int64)

        # Convert to flat indices
        neg_flat = self.node_id(neg_idx_array)
        pos_flat = self.node_id(pos_idx_array)

        return neg_flat, pos_flat

    def periodic_node_coordinates(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Return coordinates of paired nodes for periodic boundary on given axis.

        Returns:
            A tuple (neg_coords, pos_coords) where each coordinate pair represents
            nodes that should be constrained together in a periodic simulation.
        """
        neg_flat, pos_flat = self.periodic_pair_indices(axis)
        coords = self.node_coordinates()
        return coords[neg_flat], coords[pos_flat]

    def is_periodic_boundary_node(self, node_id: int) -> tuple[bool, int | None]:
        """Check if a node lies on a periodic boundary.

        Args:
            node_id: The flat node ID to check.

        Returns:
            A tuple (is_boundary, boundary_axis) where is_boundary indicates whether
            the node is on any face of the domain, and boundary_axis is None for
            interior nodes or an axis index (0=x, 1=y, 2=z) if on a face.
            Note: Corner/edge nodes may be on multiple faces - only one axis is returned.
        """
        coords = self.node_coordinates()
        node_coord = coords[node_id]

        # Check each axis
        for axis in range(3):
            lower_threshold = self.origin[axis] + 1e-9 * self.h
            upper_threshold = self.upper[axis] - 1e-9 * self.h

            # On negative face (or close to it)
            if np.abs(node_coord[axis] - self.origin[axis]) < 1e-6 * self.h:
                return True, axis

            # On positive face (or close to it)
            if np.abs(node_coord[axis] - self.upper[axis]) < 1e-6 * self.h:
                return True, axis

        return False, None


def free_cells(divisions: Sequence[int], boxes: Iterable[Tuple[np.ndarray, np.ndarray | int]]) -> int:
    """Number of surviving background cells after carving (for cost estimates)."""
    total = int(np.prod(divisions))
    used = sum(int(np.prod(np.broadcast_to(np.asarray(k, dtype=np.int64), (3,)))) for _, k in boxes)
    return total - used


class OccupancyMap:
    """Integer occupancy bookkeeping used by the packer.

    Marking whole macro-boxes rather than testing sphere-sphere distances makes
    the admissibility criterion of both reviews (``|c_i - c_j| > R_i + R_j +
    g_min``) *exact and constructive*: two inclusions are acceptable precisely
    when their macro-boxes are disjoint, which is a trivial integer test and is
    also the condition the mesher actually needs.  Box edges are per-axis
    vectors ``(kx, ky, kz)``; a scalar is broadcast to a cube.
    """

    def __init__(self, divisions: Sequence[int]) -> None:
        self.divisions = tuple(int(d) for d in divisions)
        self._occupied = np.zeros(self.divisions, dtype=bool)
        self.boxes: List[Tuple[np.ndarray, np.ndarray]] = []

    @staticmethod
    def _vec(k: np.ndarray | int) -> np.ndarray:
        return np.broadcast_to(np.asarray(k, dtype=np.int64), (3,))

    def fits(self, lo: np.ndarray, k: np.ndarray | int, margin: int = 0) -> bool:
        lo = np.asarray(lo, dtype=np.int64).reshape(3)
        k = self._vec(k)
        a = lo - margin
        b = lo + k + margin
        if np.any(lo < 0) or np.any(lo + k > np.array(self.divisions)):
            return False
        a = np.maximum(a, 0)
        b = np.minimum(b, np.array(self.divisions))
        return not self._occupied[a[0] : b[0], a[1] : b[1], a[2] : b[2]].any()

    def place(self, lo: np.ndarray, k: np.ndarray | int) -> None:
        lo = np.asarray(lo, dtype=np.int64).reshape(3)
        k = self._vec(k).copy()
        self._occupied[lo[0] : lo[0] + k[0], lo[1] : lo[1] + k[1], lo[2] : lo[2] + k[2]] = True
        self.boxes.append((lo, k))

    @property
    def occupied_fraction(self) -> float:
        return float(self._occupied.mean())
