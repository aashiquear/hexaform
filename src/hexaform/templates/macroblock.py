"""The Hexaform macro-block: a star-shaped body meshed inside a grid cell box.

This is the algorithmic centre of the package and the answer to the question
both source reviews pose -- how to obtain a conforming, non-voxelated,
all-hexahedral mesh of a domain containing many curved internal boundaries
without solving the general all-hex problem.

The construction
----------------
Each inclusion is given a **macro-box**: an axis-aligned cube of
``k x k x k`` background grid cells, aligned to the grid and containing the body
with margin.  Inside the macro-box the mesh is fully structured::

    core block         (kx*ky*kz hexes)      inclusion material
      |  transfinite fill of a spherified cube
    inner shell        (2*(kx*ky + ky*kz + kz*kx) * n_in hexes)  inclusion material
      |  radial sweep from core surface to the exact analytic surface
    ===== analytic surface: every node satisfies |x - c| = r(d) exactly =====
    outer shell        (2*(kx*ky + ky*kz + kz*kx) * n_out hexes)  matrix material
      |  radial sweep from the surface to the flat macro-box faces
    macro-box boundary (kx x ky x kz cells, uniform nodes per face)

**The key identity is per-axis ``n = k``.**  Each cubed-sphere patch edge is
sampled at exactly the uniform ``k`` node positions of the background grid on
the corresponding box face edge, using integer index arithmetic, so the
macro-block and the carved background grid share those nodes bit-for-bit.
Welding is exact, there are no hanging nodes, and no global remeshing is
required when a particle moves.

``k`` may be a scalar (a cube, with one cubed-sphere resolution) or a per-axis
``(kx, ky, kz)`` vector: an anisotropic macro-box whose patch resolutions match
the box dimensions.  The packer derives it from the body's bounding box so that
elongated inclusions stop wasting elements on the short axes.

Why this is *not* voxelisation
------------------------------
A voxel mesh takes its geometry from the grid.  Here the grid supplies only the
*topology* of the outer boundary of a local block; the interface nodes are
placed by evaluating the analytic support radius.  Refining the background grid
refines the matrix, not the sphere -- the sphere is already exact at every node.
This is the distinction the SciSpace review makes in its section 18.

Shape genericity
----------------
The template emits the proven linear corner topology.  After global welding,
:mod:`hexaform.core.elevate` adds shared HEX20/HEX27 nodes and projects the
higher-order interface nodes onto the same analytic surface.

Nothing below mentions "sphere".  The template consumes a
:class:`~hexaform.shapes.base.StarShape` and calls ``support_radius`` along the
cubed-sphere directions.  Ellipsoids, superquadrics and spherical-harmonic
grains flow through the identical code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..shapes.base import StarShape
from .cubed_sphere import FACE_ORDER, CubedSphere
from .transfinite import graded_weights, tfi_volume

# For each cubed-sphere patch: which lattice face of the core block it is, and
# how the patch (i, j) indices map onto that face's two lattice indices.
# Entries are (axis, side, transpose) where the patch supplies the core face
# array either directly or transposed.
_PATCH_TO_LATTICE: Dict[str, Tuple[int, int, bool]] = {
    "x+": (0, 1, False),  # patch i -> y, j -> z
    "x-": (0, 0, True),   # patch i -> z, j -> y
    "y+": (1, 1, True),   # patch i -> z, j -> x
    "y-": (1, 0, False),  # patch i -> x, j -> z
    "z+": (2, 1, False),  # patch i -> x, j -> y
    "z-": (2, 0, True),   # patch i -> y, j -> x
}


@dataclass
class MacroBlockOptions:
    """Tunable parameters of the macro-block template."""

    core_fraction: float = 0.55
    """Corner extent of the core block as a fraction of the local support
    radius.  Must stay below ``1``; ``0.5-0.6`` gives the best element quality."""

    core_roundness: float = 0.35
    """Blend of the core boundary between a cube (0) and a scaled copy of the
    body (1).  A mild value keeps the core well conditioned while letting the
    inner shell absorb most of the surface curvature."""

    n_inner: int = 2
    """Radial layers between the core surface and the body surface."""

    n_outer: int = 2
    """Radial layers between the body surface and the macro-box faces."""

    inner_grading: float = 1.0
    """Ratio of largest to smallest inner-shell layer; >1 clusters at the
    interface (where plastic strain and phase-field gradients concentrate)."""

    outer_grading: float = 1.0
    """Same for the matrix-side shell."""

    patch_spacing: str = "blend"
    """``uniform``, ``equiangular`` or ``blend`` (see
    :func:`~hexaform.templates.cubed_sphere.patch_spacing`)."""

    patch_blend: float = 0.6
    """Blend factor when ``patch_spacing == "blend"``.  Values near 1 favour a
    uniform sphere surface; values near 0 favour a low-shear transition shell."""


@dataclass
class MacroBlock:
    """A generated macro-block: local nodes, hexes and node-set bookkeeping."""

    points: np.ndarray
    cells: np.ndarray
    is_inclusion: np.ndarray
    surface_nodes: np.ndarray
    """Local ids of nodes lying exactly on the analytic surface."""
    box_face_nodes: Dict[str, np.ndarray]
    """Per box face, the local ids on that face of the macro-box.
    (``ky+1, kz+1``) on ``x+``/``x-``, (``kz+1, kx+1``) on ``y+``/``y-``,
    (``kx+1, ky+1``) on ``z+``/``z-``."""
    interface_cells: np.ndarray
    """Local ids of the matrix-side elements touching the analytic surface."""
    interface_sides: np.ndarray
    """Exodus side ids (1..6) of those elements' faces on the surface."""
    stats: Dict[str, float] = field(default_factory=dict)


def build_macroblock(
    shape: StarShape,
    box_lo_idx: np.ndarray,
    k: int | np.ndarray,
    h: float,
    grid_origin: np.ndarray,
    options: MacroBlockOptions | None = None,
) -> MacroBlock:
    """Mesh one star-shaped inclusion inside a grid-aligned macro-box.

    Parameters
    ----------
    shape:
        The inclusion.  Must be strictly inside the macro-box.
    box_lo_idx:
        Integer index of the macro-box's lower corner in the background grid.
    k:
        Cells per macro-box edge: a scalar (cube) or a per-axis ``(kx, ky, kz)``
        vector.  Also sets the cubed-sphere patch resolution per axis.
    h:
        Background cell size.
    grid_origin:
        Physical coordinates of background grid index ``(0, 0, 0)``.
    """
    opt = options or MacroBlockOptions()
    k = np.broadcast_to(np.asarray(k, dtype=np.int64), (3,)).copy()
    box_lo_idx = np.asarray(box_lo_idx, dtype=np.int64).reshape(3)
    grid_origin = np.asarray(grid_origin, dtype=np.float64).reshape(3)
    if np.any(k < 2):
        raise ValueError("macro-box must span at least 2 background cells per axis")

    cs = CubedSphere(tuple(int(a) for a in k), spacing=opt.patch_spacing, blend=opt.patch_blend)
    n_surf = cs.n_nodes
    centre = np.asarray(shape.center, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. the three key surfaces, evaluated on the shared cubed-sphere nodes
    # ------------------------------------------------------------------
    dirs = cs.directions                       # (n_surf, 3)
    radii = shape.support_radius(dirs)         # (n_surf,) exact analytic radius
    surf_pts = centre + radii[:, None] * dirs  # the analytic surface

    # core boundary: a spherified cube scaled anisotropically by the local radius
    q = cs.cube_points
    q_scaled = q / np.sqrt(3.0)
    core_dir = (1.0 - opt.core_roundness) * q_scaled + opt.core_roundness * dirs
    core_pts = centre + opt.core_fraction * radii[:, None] * core_dir

    # macro-box faces: uniform nodes computed by *integer* index arithmetic so
    # they coincide bit-for-bit with the carved background grid.
    box_pts = _box_face_points(cs, box_lo_idx, k, h, grid_origin)

    # ------------------------------------------------------------------
    # 2. node allocation
    # ------------------------------------------------------------------
    core_lattice = _core_lattice(cs, core_pts, k)          # (kx+1, ky+1, kz+1, 3)
    n_core = int(np.prod(k + 1))
    w_in = graded_weights(opt.n_inner, opt.inner_grading, "end")
    w_out = graded_weights(opt.n_outer, opt.outer_grading, "start")

    inner_layers = [
        (1.0 - t) * core_pts + t * surf_pts for t in w_in[1:]
    ]  # layers 1..n_inner; layer n_inner is the analytic surface
    outer_layers = [
        (1.0 - t) * surf_pts + t * box_pts for t in w_out[1:]
    ]  # layers 1..n_outer; layer n_outer is the macro-box surface

    points = [core_lattice.reshape(-1, 3)] + inner_layers + outer_layers
    points = np.concatenate(points, axis=0)

    def inner_base(t: int) -> int:
        return n_core + (t - 1) * n_surf

    def outer_base(t: int) -> int:
        return n_core + opt.n_inner * n_surf + (t - 1) * n_surf

    surface_base = inner_base(opt.n_inner)
    surface_nodes = np.arange(surface_base, surface_base + n_surf, dtype=np.int64)

    # surface node -> core lattice node id (layer 0 of the inner shell)
    surf_to_core = _surface_to_core_map(cs, k)

    def layer_ids(kind: str, t: int) -> np.ndarray:
        """Global local-node ids for one radial layer, indexed by surface node."""
        if kind == "inner":
            if t == 0:
                return surf_to_core
            return np.arange(inner_base(t), inner_base(t) + n_surf, dtype=np.int64)
        if t == 0:
            return surface_nodes
        return np.arange(outer_base(t), outer_base(t) + n_surf, dtype=np.int64)

    # ------------------------------------------------------------------
    # 3. connectivity
    # ------------------------------------------------------------------
    cells = [_core_cells(k)]
    is_incl = [np.ones(int(np.prod(k)), dtype=bool)]

    for t in range(opt.n_inner):
        cells.append(_shell_cells(cs, layer_ids("inner", t), layer_ids("inner", t + 1)))
        is_incl.append(np.ones(cs.n_quads, dtype=bool))

    first_outer = sum(c.shape[0] for c in cells)
    for t in range(opt.n_outer):
        cells.append(_shell_cells(cs, layer_ids("outer", t), layer_ids("outer", t + 1)))
        is_incl.append(np.zeros(cs.n_quads, dtype=bool))

    cells = np.concatenate(cells, axis=0)
    is_incl = np.concatenate(is_incl, axis=0)

    # The first outer-shell layer sits directly on the analytic surface.  Its
    # elements were built with the surface quad as local face 5 (zeta-).
    interface_cells = np.arange(first_outer, first_outer + cs.n_quads, dtype=np.int64)
    interface_sides = np.full(interface_cells.size, 5, dtype=np.int32)

    box_face_nodes = {
        name: layer_ids("outer", opt.n_outer)[cs.index[name]] for name in FACE_ORDER
    }

    stats = _macroblock_stats(shape, points, surface_nodes, radii, k, h)
    return MacroBlock(
        points=points,
        cells=cells,
        is_inclusion=is_incl,
        surface_nodes=surface_nodes,
        box_face_nodes=box_face_nodes,
        interface_cells=interface_cells,
        interface_sides=interface_sides,
        stats=stats,
    )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _box_face_points(
    cs: CubedSphere, box_lo_idx: np.ndarray, k: np.ndarray, h: float, grid_origin: np.ndarray
) -> np.ndarray:
    """Macro-box surface positions at every cubed-sphere node.

    Computed as ``grid_origin + integer_index * h`` so the result is bitwise
    identical to the corresponding background-grid node.
    """
    pts = np.zeros((cs.n_nodes, 3))
    filled = np.zeros(cs.n_nodes, dtype=bool)
    ar = [np.arange(int(a) + 1, dtype=np.int64) for a in k]
    from .cubed_sphere import FACES

    for name in FACE_ORDER:
        normal, idir, jdir = FACES[name]
        axis = int(np.argmax(np.abs(normal)))
        side = int(k[axis]) if normal[axis] > 0 else 0
        ai = int(np.argmax(np.abs(idir)))
        aj = int(np.argmax(np.abs(jdir)))
        # index along i grows with +idir if idir is positive, else it reverses;
        # all FACES entries use positive unit vectors, so this is a plain map.
        idx = np.zeros((len(ar[ai]), len(ar[aj]), 3), dtype=np.int64)
        idx[..., axis] = side
        idx[..., ai] = ar[ai][:, None]
        idx[..., aj] = ar[aj][None, :]
        coords = grid_origin + (box_lo_idx + idx) * h
        ids = cs.index[name]
        pts[ids] = coords
        filled[ids] = True
    if not filled.all():
        raise RuntimeError("macro-box face sampling missed cubed-sphere nodes")
    return pts


def _core_lattice(cs: CubedSphere, core_pts: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Transfinite fill of the core block from its six cubed-sphere patches."""
    patches = {name: core_pts[cs.index[name]] for name in FACE_ORDER}
    # Reorient each patch into lattice (axis-major) ordering.
    x_hi = patches["x+"]                 # (y, z)
    x_lo = np.transpose(patches["x-"], (1, 0, 2))  # patch (z, y) -> (y, z)
    y_hi = np.transpose(patches["y+"], (1, 0, 2))  # patch (z, x) -> (x, z)
    y_lo = patches["y-"]                 # (x, z)
    z_hi = patches["z+"]                 # (x, y)
    z_lo = np.transpose(patches["z-"], (1, 0, 2))  # patch (y, x) -> (x, y)
    return tfi_volume(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)


def _surface_to_core_map(cs: CubedSphere, k: np.ndarray) -> np.ndarray:
    """Map each cubed-sphere node id to its core-lattice flat node id."""
    lattice = np.arange(int(np.prod(k + 1)), dtype=np.int64).reshape(*(k + 1))
    out = -np.ones(cs.n_nodes, dtype=np.int64)
    for name in FACE_ORDER:
        axis, side, transpose = _PATCH_TO_LATTICE[name]
        plane = np.take(lattice, int(k[axis]) if side else 0, axis=axis)  # (a, b) in remaining axes
        if transpose:
            plane = plane.T
        out[cs.index[name]] = plane
    if np.any(out < 0):
        raise RuntimeError("surface-to-core mapping is incomplete")
    return out


def _core_cells(k: np.ndarray) -> np.ndarray:
    """Structured HEX8 connectivity of a ``kx*ky*kz`` lattice block."""
    shape = tuple(int(a) + 1 for a in k)
    idx = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    a = idx[:-1, :-1, :-1].ravel()
    b = idx[1:, :-1, :-1].ravel()
    c = idx[1:, 1:, :-1].ravel()
    d = idx[:-1, 1:, :-1].ravel()
    e = idx[:-1, :-1, 1:].ravel()
    f = idx[1:, :-1, 1:].ravel()
    g = idx[1:, 1:, 1:].ravel()
    hh = idx[:-1, 1:, 1:].ravel()
    return np.stack([a, b, c, d, e, f, g, hh], axis=1)


def _shell_cells(cs: CubedSphere, inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Hexes sweeping one radial layer, wound for a positive Jacobian.

    The inner quad becomes local face 5 (``zeta-``) and the outer quad face 6.
    Because each patch's ``(i, j)`` directions satisfy ``i_dir x j_dir = outward``,
    listing the inner quad as ``(i,j), (i+1,j), (i+1,j+1), (i,j+1)`` and the
    outer quad in the same order gives ``det J > 0`` directly.
    """
    out = []
    for name in FACE_ORDER:
        ids = cs.index[name]
        a = ids[:-1, :-1].ravel()
        b = ids[1:, :-1].ravel()
        c = ids[1:, 1:].ravel()
        d = ids[:-1, 1:].ravel()
        quad = np.stack([a, b, c, d], axis=1)
        out.append(np.concatenate([inner[quad], outer[quad]], axis=1))
    return np.concatenate(out, axis=0)


def _macroblock_stats(
    shape: StarShape,
    points: np.ndarray,
    surface_nodes: np.ndarray,
    radii: np.ndarray,
    k: np.ndarray,
    h: float,
) -> Dict[str, float]:
    sp = points[surface_nodes]
    rho = np.linalg.norm(sp - shape.center, axis=1)
    err = np.abs(rho - radii)
    km = float(np.mean(k))
    return {
        "k": km,
        "k_x": float(k[0]),
        "k_y": float(k[1]),
        "k_z": float(k[2]),
        "surface_nodes": float(surface_nodes.size),
        "max_node_radius_error": float(err.max()),
        "surface_edge_length": float(np.mean(radii) * 2.0 / km),
        "box_edge_length": float(km * h),
    }
