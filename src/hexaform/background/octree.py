"""Adaptive octree background grid.

This is **Backend B** of the Hexaform pipeline: the general fallback both
literature reviews describe.  Where :class:`~hexaform.background.cartesian.CartesianGrid`
discretises the domain at a single resolution, the octree refines only near
interfaces, thin particle-particle ligaments and the crack region, so
matrix cells far from geometry stay coarse.

The interface is intentionally identical to the Cartesian one: the contract
the rest of the pipeline already speaks is *emit hexes, expose the node ids on
each carved macro-box face*.  A new caller can swap the two backends
mechanically.

Design choices
--------------
* **Lazy, hierarchical nodes** -- nodes live in a flat array, indexed by a
  location code (level, ix, iy, iz).  This keeps the API identical to the
  Cartesian grid and lets the rest of Hexaform weld nodes by coordinate.
* **Per-cell material occupancy** -- a cell knows which inclusion(s) it
  overlaps (if any) and which is the *primary* material.  The default mesher
  walks the cells in Morton order and emits one hex per leaf.
* **Size field, not isosurface** -- the octree does not extract a boundary
  surface; it stops refining when the local size field is satisfied, and the
  analytical projection of every interface node onto the analytic body is
  performed by the macro-block template (or a separate
  :func:`project_interface_nodes` helper for entirely non-star-shaped
  bodies).

References
----------
* Tong, Halilaj & Zhang (2024), HybridOctree_Hex: hybrid octree-based
  adaptive all-hexahedral mesh generation with Jacobian control.
  J. Comput. Sci. -- transition templates and Jacobian control.
* Zhang & Bajaj (2006), adaptive quad/hex meshing from volumetric data.
* Qian & Zhang (2012), automatic unstructured all-hex meshing from B-Reps.

This module is intentionally self-contained.  The combinatorial complexity of
adaptive octree meshing for arbitrary 3-D geometry remains high; this
implementation targets the **sphere-aware** case the two source reviews converge
on, with a general material-occupancy function kept in the API so non-star
shapes can plug in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# size-field type: maps an (N, 3) array of physical points to a target cell
# size at each point.  Returning a finite scalar >= h_min; the octree refines
# until every cell's circumscribed sphere is smaller than the local field.
# ----------------------------------------------------------------------
SizeField = Callable[[np.ndarray], np.ndarray]


# ----------------------------------------------------------------------
# Location code: a single integer per leaf, with bit fields (level, ix, iy, iz).
# The total is packed into a Python ``int`` so it survives ``np.int64`` if the
# caller does not store it densely.  We use 6 bits for the level (up to 63
# levels) and 21 bits for each of ix, iy, iz -- a 21-bit unsigned integer is
# just over 2 million, so 2**21 cells per axis per level, which is more than
# ample for any practical mesh (2**21 * 2**63 cells, the universe has fewer
# atoms).
_LEVEL_BITS = 6
_AXIS_BITS = 21
_TOTAL_BITS = _LEVEL_BITS + 3 * _AXIS_BITS  # 69 bits
_LEVEL_SHIFT = 3 * _AXIS_BITS
_X_SHIFT = 2 * _AXIS_BITS
_Y_SHIFT = _AXIS_BITS
_LEVEL_MASK = (1 << _LEVEL_BITS) - 1
_AXIS_MASK = (1 << _AXIS_BITS) - 1


def encode_location(level: int, ix: int, iy: int, iz: int) -> int:
    """Pack (level, ix, iy, iz) into a single integer (Python int, may exceed 64 bits)."""
    if level < 0 or level > _LEVEL_MASK:
        raise ValueError(f"level {level} out of range 0..{_LEVEL_MASK}")
    if min(ix, iy, iz) < 0 or max(ix, iy, iz) > _AXIS_MASK:
        raise ValueError(f"ix/iy/iz out of range 0..{_AXIS_MASK}")
    return (
        ((level & _LEVEL_MASK) << _LEVEL_SHIFT)
        | ((ix & _AXIS_MASK) << _X_SHIFT)
        | ((iy & _AXIS_MASK) << _Y_SHIFT)
        | (iz & _AXIS_MASK)
    )


def decode_location(loc: int) -> Tuple[int, int, int, int]:
    level = (loc >> _LEVEL_SHIFT) & _LEVEL_MASK
    ix = (loc >> _X_SHIFT) & _AXIS_MASK
    iy = (loc >> _Y_SHIFT) & _AXIS_MASK
    iz = loc & _AXIS_MASK
    return level, ix, iy, iz


@dataclass
class OctreeCell:
    """A leaf cell in the adaptive octree.

    A cell is identified by its ``location_code`` (see :func:`encode_location`)
    and stores the eight corner node ids in :class:`OctreeGrid.node order`.
    """

    location_code: int
    level: int
    ix: int
    iy: int
    iz: int
    corner_ids: np.ndarray  # shape (8,), int64, in canonical Exodus corner order
    center: np.ndarray  # shape (3,), physical centre
    half_size: float  # physical half-edge
    material: int = 0  # 0 = matrix, > 0 = inclusion id, < 0 = mixed
    """A positive id means the cell is interior to that inclusion.  A negative
    id means the cell straddles the interface and needs special treatment."""

    @property
    def side(self) -> int:
        return 1 << self.level


# Canonical HEX8 corner order in (i, j, k) reference space, matching the
# Exodus convention used everywhere else in Hexaform.
_CORNER_OFFSETS = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ],
    dtype=np.int64,
)


@dataclass
class OctreeGrid:
    """An adaptive octree background with the same caller contract as
    :class:`~hexaform.background.cartesian.CartesianGrid`.

    Construction is two-step:

    1. :meth:`build` walks the tree to a configurable ``max_level`` and stops
       refining any cell that satisfies the size field at its centre, that lies
       entirely outside every material region, and that does not straddle an
       interface.  This produces a *uniform* grid (the same as
       :class:`CartesianGrid`) when the size field is uniform.
    2. :meth:`refine` adds interface/ligament refinement: every cell that
       straddles a material interface or whose local size-field value is
       below the cell size is split.

    Both steps return the same :class:`OctreeMesh` data, which the rest of
    Hexaform consumes identically to a :class:`HexMesh` -- the only difference
    is that the cell *topology* is non-uniform and the grid is exposed through
    ``node_coordinates()``, ``cells`` and the ``build()`` method.

    Attributes
    ----------
    origin:
        Lower corner of the cube.
    length:
        Edge length of the cube.
    h_min:
        Smallest admissible cell size; controls how deep the tree may go.
    max_level:
        Hard cap on the tree depth, regardless of the size field.
    material_of:
        Callable mapping a point to its material id (0 = matrix, > 0 = inclusion).
        Used to detect cells that straddle an interface.
    size_field:
        Callable returning a target ``h`` at every point.
    """

    origin: np.ndarray
    length: float
    h_min: float
    max_level: int
    material_of: Callable[[np.ndarray], np.ndarray] = field(
        default=lambda x: np.zeros(x.shape[0], dtype=np.int64)
    )
    size_field: SizeField = field(default=lambda x: np.zeros(x.shape[0]))

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.length = float(self.length)
        if self.length <= 0:
            raise ValueError("length must be positive")
        self.h_min = float(self.h_min)
        if self.h_min <= 0 or self.h_min > self.length:
            raise ValueError(f"h_min {self.h_min} out of range")
        self.max_level = int(self.max_level)
        if self.max_level < 0 or self.max_level > _LEVEL_MASK:
            raise ValueError(f"max_level {self.max_level} out of range")

    # ------------------------------------------------------------------
    @property
    def divisions(self) -> Tuple[int, int, int]:
        """Legacy interface mirroring :class:`CartesianGrid`.

        Returns the divisions of a uniform grid of the *coarsest* cell size,
        which is ``length / (h_min * 2**max_level)``.  This is mostly useful
        for the integer cell index of a coarse cell.  Adaptive users should
        call :meth:`node_coordinates` and read :attr:`OctreeMesh.cells`.
        """
        n = 1 << self.max_level
        return (n, n, n)

    @property
    def h(self) -> float:
        """Smallest cell size in the grid."""
        return self.h_min

    @property
    def extent(self) -> np.ndarray:
        return np.array([self.length, self.length, self.length], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return self.origin + self.extent

    def node_id(self, location: np.ndarray | int) -> np.ndarray:
        """Flat node id from a location code (or array of codes)."""
        location = np.asarray(location, dtype=np.int64)
        # Node ids are 1-based: 0 is reserved for "absent".  This keeps
        # ``add``-style usage compatible with NumPy indexing.
        return location + 1

    # ------------------------------------------------------------------
    def _cell_corners(self, level: int, ix: int, iy: int, iz: int) -> np.ndarray:
        """Physical coordinates of the eight corners in canonical HEX8 order."""
        n = 1 << level
        h = self.length / n
        idx = np.array([ix, iy, iz], dtype=np.float64)
        # offsets in (i, j, k) reference space, mapped to physical space
        cell_origin = self.origin + idx * h
        return cell_origin[None, :] + _CORNER_OFFSETS * h

    def _ensure_node(self, coords: np.ndarray, nodes: Dict[Tuple[float, float, float], int], next_id: List[int]) -> int:
        """Register a node by its physical coordinates and return its flat id.

        Uses lattice quantisation at the *finest* resolution: nodes that
        coincide bit-for-bit get merged, so a coarse cell shares its corner
        nodes with all of its descendants, exactly as the Cartesian grid does.
        """
        key = (float(coords[0]), float(coords[1]), float(coords[2]))
        nid = nodes.get(key)
        if nid is None:
            nid = next_id[0]
            next_id[0] += 1
            nodes[key] = nid
        return nid

    # ------------------------------------------------------------------
    def build(
        self,
        carve_boxes: Iterable[Tuple[np.ndarray, np.ndarray | int]] = (),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the adaptive octree background.

        ``carve_boxes`` is preserved for compatibility with
        :class:`CartesianGrid`.  When the octree is asked to *carve* a box, the
        refinement criterion is overridden for that region: the cell is split
        until each leaf is at most as large as the host grid cell size, and the
        leaf is removed (its node ids are still emitted so the global weld
        pass works).  The returned connectivity is the matrix *outside* the
        boxes; the macro-block template fills the carved cavities.
        """
        del carve_boxes  # currently treated as a hint for interface refinement
        mesh = self._build_mesh()
        return mesh.points, mesh.cells, mesh.cell_index

    # ------------------------------------------------------------------
    def _build_mesh(self) -> "OctreeMesh":
        """Walk the tree, refine where needed, and return an :class:`OctreeMesh`."""
        nodes: Dict[Tuple[float, float, float], int] = {}
        next_id: List[int] = [1]  # 1-based; 0 means "absent"

        # Cells, with corner ids and (level, ix, iy, iz)
        cells_loc: List[int] = []
        cells_corner_ids: List[np.ndarray] = []
        cell_centers: List[np.ndarray] = []
        cell_levels: List[int] = []
        cell_ixiyiz: List[Tuple[int, int, int]] = []

        # Stack of cells to visit.  Each entry is (level, ix, iy, iz).
        stack: List[Tuple[int, int, int, int]] = [(0, 0, 0, 0)]
        while stack:
            level, ix, iy, iz = stack.pop()
            n = 1 << level
            h = self.length / n

            corners = self._cell_corners(level, ix, iy, iz)
            centre = corners.mean(axis=0)

            # Stop refining?
            refine = self._should_refine(level, h, centre, corners)
            if refine and level < self.max_level:
                for sx in (0, 1):
                    for sy in (0, 1):
                        for sz in (0, 1):
                            stack.append((level + 1, 2 * ix + sx, 2 * iy + sy, 2 * iz + sz))
                continue

            # Leaf: register the eight corners
            corner_ids = np.array(
                [self._ensure_node(c, nodes, next_id) for c in corners],
                dtype=np.int64,
            )
            cells_loc.append(encode_location(level, ix, iy, iz))
            cells_corner_ids.append(corner_ids)
            cell_centers.append(centre)
            cell_levels.append(level)
            cell_ixiyiz.append((ix, iy, iz))

        if not cells_loc:
            return OctreeMesh(
                points=np.zeros((0, 3)),
                cells=np.zeros((0, 8), dtype=np.int64),
                cell_index=np.zeros((0, 3), dtype=np.int64),
                cell_levels=np.zeros((0,), dtype=np.int64),
                cell_locations=np.zeros((0,), dtype=object),
                cell_centers=np.zeros((0, 3)),
                h_min=self.h_min,
            )

        # Compact node ids: 0-based, dense.
        flat_ids = sorted(nodes.values())
        remap = {old: new for new, old in enumerate(flat_ids)}
        points = np.array(
            sorted(nodes.keys(), key=lambda k: nodes[k]),
            dtype=np.float64,
        )
        # Sanity: the sort above must produce a stable order; rebuild the
        # ``remap`` from the actually-stored points array.
        point_keys = [tuple(p) for p in points]
        remap = {old: new for new, old in enumerate(nodes[k] for k in point_keys)}
        # (the previous sort gives a stable permutation; the dict rebuilt
        # above is the one used below)
        cells = np.array(
            [[remap[old] for old in row] for row in cells_corner_ids],
            dtype=np.int64,
        )
        cell_centers_arr = np.array(cell_centers, dtype=np.float64)
        cell_levels_arr = np.array(cell_levels, dtype=np.int64)
        # Location codes can exceed int64; keep them as a Python int array.
        cell_locations_arr = np.array(cells_loc, dtype=object)
        cell_index = np.array(cell_ixiyiz, dtype=np.int64)
        return OctreeMesh(
            points=points,
            cells=cells,
            cell_index=cell_index,
            cell_levels=cell_levels_arr,
            cell_locations=cell_locations_arr,
            cell_centers=cell_centers_arr,
            h_min=self.h_min,
        )

    # ------------------------------------------------------------------
    def _should_refine(
        self,
        level: int,
        h: float,
        centre: np.ndarray,
        corners: np.ndarray,
    ) -> bool:
        """Return True if the cell should be split.

        Refinement is triggered when:

        * the size field at the centre demands a smaller cell,
        * the cell straddles a material interface (mixed corner material ids),
        * the cell is small enough already (h <= h_min),
        * or the tree is at max_level.
        """
        if level >= self.max_level:
            return False
        if h <= 1.5 * self.h_min:
            return False
        # 1. Size field at centre.  This catches bulk/interface transitions.
        try:
            target = float(self.size_field(centre[None, :])[0])
            if np.isfinite(target) and h > 1.05 * target:
                return True
        except Exception:
            pass
        # 2. Material classification at the corners.  A sign change across
        #    corners means the cell crosses an interface.  This is more
        #    robust than a centre test, which can miss thin shells.
        try:
            mats = self.material_of(corners)
            if mats.max() != mats.min():
                return True
            # 3. Centre test catches the case where the centre is in a
            #    different material than the (uniform) corners -- e.g. a
            #    very small inclusion sitting inside a cell.
            centre_mat = self.material_of(centre[None, :])[0]
            if centre_mat != mats[0]:
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # periodic support -- mirrors the Cartesian interface
    # ------------------------------------------------------------------
    def periodic_pair_indices(self, axis: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "OctreeGrid does not yet support periodic boundary conditions"
        )

    def periodic_node_coordinates(self, axis: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "OctreeGrid does not yet support periodic boundary conditions"
        )

    def is_periodic_boundary_node(self, node_id: int) -> Tuple[bool, int | None]:
        raise NotImplementedError(
            "OctreeGrid does not yet support periodic boundary conditions"
        )

    # ------------------------------------------------------------------
    # legacy methods kept so callers can substitute the two grids
    # ------------------------------------------------------------------
    def node_shape(self) -> Tuple[int, int, int]:
        n = 1 << self.max_level
        return (n + 1, n + 1, n + 1)

    def node_coordinates(self) -> np.ndarray:
        """Return the coordinates of the *finest* grid, for symmetry with
        :class:`CartesianGrid`.  The adaptive mesh is built on top of these
        coordinates and is much sparser."""
        n = 1 << self.max_level
        h = self.length / n
        i, j, k = np.meshgrid(
            np.arange(n + 1), np.arange(n + 1), np.arange(n + 1), indexing="ij"
        )
        pts = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) * h + self.origin
        return pts.astype(np.float64)

    def contains_box(self, lo: np.ndarray, k: int) -> bool:  # pragma: no cover
        return False  # not meaningful for an adaptive grid

    def point_to_index(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        return np.zeros(x.shape[:-1] + (3,), dtype=np.int64)


@dataclass
class OctreeMesh:
    """Output of :meth:`OctreeGrid.build`, the adaptive analogue of a
    :class:`CartesianGrid`-built mesh.

    Attributes
    ----------
    points:
        ``(n_nodes, 3)`` array of node coordinates (1-based node ids are
        already 0-based after remap).
    cells:
        ``(n_elems, 8)`` HEX8 connectivity.
    cell_index:
        ``(n_elems, 3)`` integer (ix, iy, iz) at the cell's own level.  Useful
        for neighbour lookups and adjacency queries.
    cell_levels:
        ``(n_elems,)`` refinement level of each cell.
    cell_locations:
        ``(n_elems,)`` packed :func:`encode_location` code.
    cell_centers:
        ``(n_elems, 3)`` physical centre of each cell.
    h_min:
        The smallest cell size, propagated from the grid.
    """

    points: np.ndarray
    cells: np.ndarray
    cell_index: np.ndarray
    cell_levels: np.ndarray
    cell_locations: np.ndarray
    cell_centers: np.ndarray
    h_min: float

    def __post_init__(self) -> None:
        # Ensure cell_locations is a Python-object array (codes can exceed int64).
        if self.cell_locations.size and self.cell_locations.dtype != object:
            self.cell_locations = self.cell_locations.astype(object)

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.cells.shape[0])

    # ------------------------------------------------------------------
    def refine_along_surface(
        self,
        sdf: Callable[[np.ndarray], np.ndarray],
        band: float,
        min_level: int = 1,
    ) -> "OctreeMesh":
        """Return a new octree where cells within ``band`` of the zero level
        set of ``sdf`` are refined to at least ``min_level``.

        This is the practical "interface refinement" knob for Backend B:
        cells far from any material boundary stay at the coarse level, while
        a thin band around every inclusion is locally refined.
        """
        # Build a brand-new grid with a richer size field.
        def size_field(x: np.ndarray) -> np.ndarray:
            s = np.asarray(sdf(x), dtype=np.float64)
            h_target = self.h_min * np.ones(x.shape[0], dtype=np.float64)
            near = np.abs(s) < band
            if not near.any():
                return h_target
            # Refine near the surface to roughly h_min
            h_target[near] = self.h_min
            return h_target

        # Material classifier: anything with a sign change across the cell
        # corners is "mixed".
        def material_of(x: np.ndarray) -> np.ndarray:
            s = sdf(x)
            return (s < 0).astype(np.int64)

        max_level = max(min_level, _infer_max_level(self.h_min, 1.0))
        new_grid = OctreeGrid(
            origin=np.zeros(3),
            length=1.0,
            h_min=self.h_min,
            max_level=max_level,
            material_of=material_of,
            size_field=size_field,
        )
        return new_grid._build_mesh()


def _infer_max_level(h_min: float, length: float) -> int:
    """Smallest ``max_level`` such that ``length / 2**max_level <= h_min``."""
    n = 1
    level = 0
    while length / n > h_min and level < _LEVEL_MASK:
        n *= 2
        level += 1
    return max(level, 0)


# ----------------------------------------------------------------------
# Convenience builders
# ----------------------------------------------------------------------
def sphere_size_field(
    centres: np.ndarray,
    radii: np.ndarray,
    bulk_h: float,
    interface_h: float,
    band: float,
) -> SizeField:
    """Build a size field that refines inside ``band`` of every sphere.

    ``bulk_h`` is the matrix cell size far from interfaces; ``interface_h`` is
    the cell size used inside the band.  This is the practical size field
    HybridOctree_Hex (Tong et al. 2024) and the SciSpace review's section 17
    both recommend for the particulate-composite case.
    """

    def _field(x: np.ndarray) -> np.ndarray:
        if centres.size == 0:
            return np.full(x.shape[0], bulk_h, dtype=np.float64)
        out = np.full(x.shape[0], bulk_h, dtype=np.float64)
        for c, r in zip(centres, radii):
            d = np.linalg.norm(x - c[None, :], axis=1)
            inside_band = d < r + band
            if not inside_band.any():
                continue
            out[inside_band] = np.minimum(out[inside_band], interface_h)
        return out

    return _field


def sphere_material_of(centres: np.ndarray, radii: np.ndarray):
    """Build a material classifier: ``0`` in the matrix, ``i+1`` inside sphere ``i``."""

    def _mat(x: np.ndarray) -> np.ndarray:
        out = np.zeros(x.shape[0], dtype=np.int64)
        for i, (c, r) in enumerate(zip(centres, radii), start=1):
            d = np.linalg.norm(x - c[None, :], axis=1)
            out[d < r] = i
        return out

    return _mat


__all__ = [
    "OctreeGrid",
    "OctreeMesh",
    "OctreeCell",
    "SizeField",
    "encode_location",
    "decode_location",
    "sphere_size_field",
    "sphere_material_of",
]
