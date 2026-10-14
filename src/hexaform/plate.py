"""Plates with a conforming circular or elliptical through-hole.

A rectangular plate ``width x height x thickness`` is meshed with a uniform
Cartesian background of in-plane cell size ``h`` and ONE hexahedral element
through the thickness (pseudo-3-D: generalised plane stress when both z faces
are left free).  An optional through-hole (circle or rotated ellipse) is
meshed with a block-structured O-grid ring that maps the hole boundary onto a
grid-aligned transition rectangle, so that

* every node on the hole lies exactly on the analytic curve (no staircase,
  which would seed spurious stress concentrations where damage initiates);
* the ring hands over to the background without any transition element: the
  rectangle's sides carry the same division count as the background cells
  they replace;
* the radial layers are graded independently of the background, so the
  element size at the hole edge can resolve a process zone or a phase-field
  length without refining the whole plate.

Block layout (in-plane)::

    +---------+---------------+---------+   y = height
    | corner  |   top strip   | corner  |
    +---------+---------------+---------+   cy + sy
    |  left   |    O-grid     |  right  |
    |  strip  |  (hole here)  |  strip  |
    +---------+---------------+---------+   cy - sy
    | corner  | bottom strip  | corner  |
    +---------+---------------+---------+   y = 0
  x = 0    cx - sx         cx + sx     x = width

The eight outer blocks are plain structured grids of size ``h``; the
transition half-extents ``sx, sy`` are whole multiples of ``h`` so every block
interface conforms by construction.

Sets written to the mesh: node and side sets ``x-``, ``x+``, ``y-``, ``y+``,
``z-``, ``z+`` (hexaform's boundary names), ``hole`` (node and side set on the
hole surface), the node set ``outer`` (the four in-plane edges) and the node
set ``grips`` (the two x faces together).  Blocks:
``1 = plate`` (background) and ``2 = ring`` (the O-grid).

Without a hole the plate is the plain structured grid with the same sets
(``hole`` absent), so a perforated and an unperforated plate of the same
``h`` share their background exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.elements import HEX8_SIDES
from .core.mesh import HexMesh
from .core.topology import boundary_faces, faces_on_plane, nodes_on_plane


@dataclass
class HoleSpec:
    """A through-hole in the plate's mid-plane.

    Attributes
    ----------
    kind:
        ``"circle"`` or ``"ellipse"``.
    a, b:
        Semi-axes (``b`` defaults to ``a``; a circle uses ``a`` only).
    centre:
        In-plane centre ``(cx, cy)``; ``None`` is the plate centre.
    angle:
        Rotation of the ``a`` axis from +x, in degrees (ellipse only).
    """

    kind: str = "circle"
    a: float = 1.0
    b: Optional[float] = None
    centre: Optional[Tuple[float, float]] = None
    angle: float = 0.0

    def __post_init__(self) -> None:
        self.kind = str(self.kind).lower()
        if self.kind not in ("circle", "ellipse"):
            raise ValueError(f"hole kind must be 'circle' or 'ellipse', got {self.kind!r}")
        if self.a <= 0:
            raise ValueError("hole semi-axis a must be positive")
        if self.kind == "circle":
            self.b = self.a
            self.angle = 0.0
        elif self.b is None:
            self.b = self.a
        if self.b <= 0:
            raise ValueError("hole semi-axis b must be positive")

    # -- geometry ------------------------------------------------------
    def radius(self, theta: np.ndarray) -> np.ndarray:
        """Distance from the centre to the hole boundary along polar angle ``theta``."""
        phi = theta - math.radians(self.angle)
        a, b = float(self.a), float(self.b)
        return a * b / np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2)

    def perimeter(self) -> float:
        a, b = float(self.a), float(self.b)
        hh = ((a - b) / (a + b)) ** 2
        return math.pi * (a + b) * (1.0 + 3.0 * hh / (10.0 + math.sqrt(4.0 - 3.0 * hh)))

    def extent(self) -> Tuple[float, float]:
        """Half-extents of the hole's bounding box along x and y."""
        ang = math.radians(self.angle)
        a, b = float(self.a), float(self.b)
        ex = math.hypot(a * math.cos(ang), b * math.sin(ang))
        ey = math.hypot(a * math.sin(ang), b * math.cos(ang))
        return ex, ey

    def on_boundary(self, x: np.ndarray, y: np.ndarray, cx: float, cy: float, tol: float) -> np.ndarray:
        theta = np.arctan2(y - cy, x - cx)
        r = np.hypot(x - cx, y - cy)
        return np.abs(r - self.radius(theta)) <= tol


@dataclass
class PlateOptions:
    """Parameters of :func:`plate_mesh`.

    ``width``, ``height`` and ``thickness`` are the plate's edges along x, y
    and z; ``h`` the in-plane background cell size (``width/h`` and
    ``height/h`` must be whole numbers).  ``transition`` is the half-extent of
    the O-grid rectangle in background cells along x and y (``None`` chooses
    the smallest rectangle that leaves at least ``ring_cells`` background cells
    between the hole's bounding box and the rectangle).  ``n_r`` is the number
    of radial layers in the ring and ``grading`` the ratio of the outermost to
    the innermost radial spacing; ``None`` for either sizes the layers so the
    innermost spacing equals the hole-edge arc length (near-square elements at
    the hole) and the outermost equals ``h``.
    """

    width: float = 100.0
    height: float = 50.0
    thickness: float = 1.0
    h: float = 2.5
    hole: Optional[HoleSpec] = None
    transition: Optional[Tuple[int, int]] = None
    ring_cells: int = 2
    n_r: Optional[int] = None
    grading: Optional[float] = None
    element_type: str = "HEX8"
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _divisions(length: float, h: float, name: str) -> int:
    n = length / h
    if abs(n - round(n)) > 1e-9 * max(1.0, n):
        raise ValueError(f"{name} = {length} is not a whole multiple of h = {h}")
    return int(round(n))


def _graded(n: int, ratio: float) -> np.ndarray:
    """``n + 1`` stations on [0, 1] with spacing growing by ``ratio`` end to end."""
    if n < 1:
        raise ValueError("at least one radial layer is required")
    if abs(ratio - 1.0) < 1e-12 or n == 1:
        return np.linspace(0.0, 1.0, n + 1)
    r = ratio ** (1.0 / (n - 1))
    w = r ** np.arange(n)
    s = np.concatenate([[0.0], np.cumsum(w)])
    return s / s[-1]


def _auto_layers(length: float, d_inner: float, d_outer: float) -> Tuple[int, float]:
    """Layers and grading so the first spacing is ``d_inner`` and the last ``d_outer``.

    The spacings form a geometric series that sums to ``length``.  If the
    series cannot reach both ends (very thin rings) the layer count is the
    rounded ``length / d_inner`` with uniform spacing.
    """
    length = float(length)
    if length <= 0:
        raise ValueError("the transition rectangle does not clear the hole")
    d_inner = min(float(d_inner), length)
    d_outer = min(max(float(d_outer), d_inner), length)
    best: Tuple[int, float] | None = None
    for n in range(1, 10_000):
        if n == 1:
            if length <= d_outer * 1.0000001:
                return 1, 1.0
            continue
        ratio = d_outer / d_inner
        r = ratio ** (1.0 / (n - 1))
        total = d_inner * (r ** n - 1.0) / (r - 1.0) if abs(r - 1.0) > 1e-12 else d_inner * n
        best = (n, ratio)
        if total >= length:
            break
    assert best is not None
    n, ratio = best
    # Re-fit the grading so that exactly n layers span the length with the
    # requested inner spacing (bisection on the last/first ratio).
    lo, hi = 1.0, max(ratio, 1.0) * 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        r = mid ** (1.0 / (n - 1))
        total = d_inner * (r ** n - 1.0) / (r - 1.0) if abs(r - 1.0) > 1e-12 else d_inner * n
        if total < length:
            lo = mid
        else:
            hi = mid
    return n, 0.5 * (lo + hi)


class _NodePool:
    """Deduplicating 2-D node pool so that blocks share interface nodes exactly."""

    def __init__(self, tol: float) -> None:
        self.pts: List[Tuple[float, float]] = []
        self._index: Dict[Tuple[int, int], int] = {}
        self.tol = tol

    def add(self, x: float, y: float) -> int:
        key = (int(round(x / self.tol)), int(round(y / self.tol)))
        got = self._index.get(key)
        if got is not None:
            return got
        self._index[key] = len(self.pts)
        self.pts.append((float(x), float(y)))
        return self._index[key]

    def grid(self, pts: np.ndarray) -> np.ndarray:
        ids = np.empty(pts.shape[:-1], dtype=np.int64)
        for idx in np.ndindex(*pts.shape[:-1]):
            ids[idx] = self.add(*pts[idx])
        return ids


def _rect_nodes(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int) -> np.ndarray:
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy], axis=-1)


def _rect_quads(ids: np.ndarray, quads: List[Tuple[int, int, int, int]]) -> None:
    nx, ny = ids.shape[0] - 1, ids.shape[1] - 1
    for i in range(nx):
        for j in range(ny):
            quads.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]))


def _rect_perimeter(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int) -> np.ndarray:
    """Counter-clockwise perimeter stations starting at the +x edge's lower corner."""
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    pts: List[Tuple[float, float]] = []
    pts += [(x1, y) for y in ys[:-1]]           # right edge, bottom -> top
    pts += [(x, y1) for x in xs[::-1][:-1]]     # top edge, right -> left
    pts += [(x0, y) for y in ys[::-1][:-1]]     # left edge, top -> bottom
    pts += [(x, y0) for x in xs[:-1]]           # bottom edge, left -> right
    return np.asarray(pts, dtype=float)


def _faces_of_nodes(mesh: HexMesh, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Boundary faces whose corner nodes all belong to ``nodes``."""
    elems, sides = boundary_faces(mesh.cells)
    if elems.size == 0:
        return elems, sides
    member = np.zeros(mesh.n_nodes, dtype=bool)
    member[nodes] = True
    keep = np.array(
        [bool(np.all(member[mesh.cells[e, list(HEX8_SIDES[s - 1])]])) for e, s in zip(elems, sides)]
    )
    return elems[keep], sides[keep]


# ----------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------
def plate_mesh(opts: PlateOptions) -> HexMesh:
    """Build the plate (see the module docstring).  Returns a HEX8 :class:`HexMesh`."""
    if str(opts.element_type).upper() != "HEX8":
        raise ValueError("plate_mesh builds HEX8 meshes; elevate afterwards for HEX20/HEX27")
    W, H, T, h = float(opts.width), float(opts.height), float(opts.thickness), float(opts.h)
    if min(W, H, T, h) <= 0:
        raise ValueError("width, height, thickness and h must be positive")
    nx, ny = _divisions(W, h, "width"), _divisions(H, h, "height")
    ox, oy, oz = (float(v) for v in opts.origin)
    tol = 1e-9 * h
    pool = _NodePool(tol)
    quads: List[Tuple[int, int, int, int]] = []
    block: List[int] = []
    meta: Dict[str, object] = {
        "generator": "plate_mesh",
        "width": W, "height": H, "thickness": T, "h": h,
        "divisions": [nx, ny, 1],
        "hole": None,
    }

    hole = opts.hole
    if hole is None:
        ids = pool.grid(_rect_nodes(0.0, W, 0.0, H, nx, ny))
        _rect_quads(ids, quads)
        block += [1] * (nx * ny)
        ring_first = None
    else:
        cx, cy = hole.centre if hole.centre is not None else (0.5 * W, 0.5 * H)
        ex, ey = hole.extent()
        if opts.transition is None:
            kx = int(math.ceil(ex / h - 1e-9)) + int(opts.ring_cells)
            ky = int(math.ceil(ey / h - 1e-9)) + int(opts.ring_cells)
        else:
            kx, ky = (int(v) for v in opts.transition)
        sx, sy = kx * h, ky * h
        for name, c, s, L in (("x", cx, sx, W), ("y", cy, sy, H)):
            if (c - s) / h < -1e-9 or (c + s) / h > L / h + 1e-9:
                raise ValueError(f"the transition rectangle leaves the plate along {name}")
            i0 = (c - s) / h
            if abs(i0 - round(i0)) > 1e-9:
                raise ValueError(
                    f"hole centre {name} = {c} minus the transition half-extent {s} is not on the "
                    f"background grid (h = {h}); move the centre or change the transition"
                )
        if ex >= sx or ey >= sy:
            raise ValueError("the hole does not fit inside the transition rectangle")
        x0, x1, y0, y1 = cx - sx, cx + sx, cy - sy, cy + sy
        i0, i1 = int(round(x0 / h)), int(round(x1 / h))
        j0, j1 = int(round(y0 / h)), int(round(y1 / h))

        # O-grid ring: perimeter stations of the rectangle, the same polar
        # angles on the hole, straight radial lines in between.
        rect = _rect_perimeter(x0, x1, y0, y1, 2 * kx, 2 * ky)
        theta = np.arctan2(rect[:, 1] - cy, rect[:, 0] - cx)
        r_hole = hole.radius(theta)
        ring_in = np.stack([cx + r_hole * np.cos(theta), cy + r_hole * np.sin(theta)], axis=1)
        gap = np.linalg.norm(rect - ring_in, axis=1)
        n_theta = rect.shape[0]
        arc = hole.perimeter() / n_theta
        if opts.n_r is None or opts.grading is None:
            n_auto, g_auto = _auto_layers(float(gap.min()), arc, h)
            n_r = int(opts.n_r) if opts.n_r is not None else n_auto
            grading = float(opts.grading) if opts.grading is not None else g_auto
        else:
            n_r, grading = int(opts.n_r), float(opts.grading)
        t = _graded(n_r, grading)[:, None, None]
        og = (1.0 - t) * ring_in[None, :, :] + t * rect[None, :, :]
        og_id = pool.grid(og)
        for i in range(n_r):
            for j in range(n_theta):
                jn = (j + 1) % n_theta
                # radial-then-angular winding: counter-clockwise, like the blocks
                quads.append((og_id[i, j], og_id[i + 1, j], og_id[i + 1, jn], og_id[i, jn]))
        block += [2] * (n_r * n_theta)
        ring_first = float(np.linalg.norm(og[1] - og[0], axis=1).min())

        # eight background blocks around the rectangle
        for (bx0, bx1, by0, by1, bnx, bny) in (
            (0.0, x0, y0, y1, i0, j1 - j0),            # left strip
            (x1, W, y0, y1, nx - i1, j1 - j0),         # right strip
            (x0, x1, 0.0, y0, i1 - i0, j0),            # bottom strip
            (x0, x1, y1, H, i1 - i0, ny - j1),         # top strip
            (0.0, x0, 0.0, y0, i0, j0),                # corners
            (x1, W, 0.0, y0, nx - i1, j0),
            (0.0, x0, y1, H, i0, ny - j1),
            (x1, W, y1, H, nx - i1, ny - j1),
        ):
            if bnx <= 0 or bny <= 0:
                continue
            ids = pool.grid(_rect_nodes(bx0, bx1, by0, by1, bnx, bny))
            n_before = len(quads)
            _rect_quads(ids, quads)
            block += [1] * (len(quads) - n_before)
        meta["hole"] = {
            "kind": hole.kind, "a": float(hole.a), "b": float(hole.b), "centre": [cx, cy],
            "angle": float(hole.angle), "transition_cells": [kx, ky], "n_r": n_r,
            "grading": grading, "n_theta": n_theta, "arc_at_hole": arc,
            "first_layer": ring_first,
        }

    pts2 = np.asarray(pool.pts, dtype=float)
    n2 = pts2.shape[0]
    points = np.zeros((2 * n2, 3))
    points[:n2, :2] = pts2
    points[n2:, :2] = pts2
    points[n2:, 2] = T
    points += np.array([ox, oy, oz])
    cells = np.asarray([[a, b, c, d, a + n2, b + n2, c + n2, d + n2] for (a, b, c, d) in quads], dtype=np.int64)
    mesh = HexMesh(points=points, cells=cells, block_id=np.asarray(block, dtype=np.int32),
                   block_names={1: "plate", 2: "ring"} if hole is not None else {1: "plate"},
                   element_type="HEX8", metadata=meta)

    # boundary sets in hexaform's naming
    bounds = ((0, ox), (0, ox + W), (1, oy), (1, oy + H), (2, oz), (2, oz + T))
    names = ("x-", "x+", "y-", "y+", "z-", "z+")
    for name, (axis, value) in zip(names, bounds):
        elems, sides = faces_on_plane(mesh, axis, value, tol)
        mesh.add_side_set(name, elems, sides)
        mesh.add_node_set(name, nodes_on_plane(mesh, axis, value, tol))
    outer = np.concatenate([mesh.node_sets[n] for n in ("x-", "x+", "y-", "y+")])
    mesh.add_node_set("outer", outer)
    # The two loaded ends together (a damage Dirichlet set that leaves the
    # free edges and the hole unconstrained).
    mesh.add_node_set("grips", np.concatenate([mesh.node_sets["x-"], mesh.node_sets["x+"]]))
    if hole is not None:
        cx, cy = meta["hole"]["centre"]  # type: ignore[index]
        on = hole.on_boundary(mesh.points[:, 0] - ox, mesh.points[:, 1] - oy, cx, cy, 1e-6 * h)
        hole_nodes = np.flatnonzero(on)
        mesh.add_node_set("hole", hole_nodes)
        elems, sides = _faces_of_nodes(mesh, hole_nodes)
        mesh.add_side_set("hole", elems, sides)
    return mesh


def plate_summary(mesh: HexMesh) -> Dict[str, object]:
    """Key numbers of a plate mesh for reports: counts, hole-edge sizes, quality."""
    from .core.quality import assess

    q = assess(mesh)
    out: Dict[str, object] = {
        "n_nodes": mesh.n_nodes, "n_elements": mesh.n_elements,
        "n_ring_elements": int(np.sum(mesh.block_id == 2)),
        "min_scaled_jacobian": float(q.min_scaled_jacobian),
        "max_edge_ratio": float(q.max_edge_ratio),
        "n_inverted": int(q.n_inverted),
    }
    out.update({k: v for k, v in mesh.metadata.items() if k != "generator"})
    return out
