"""Flat dog-bone tensile specimens with a thickness profile along the gage.

A flat tensile bar of total length ``length`` along x is meshed as one
structured block whose in-plane width follows the dog-bone planform and whose
thickness follows a prescribed profile ``t(x)``:

* grips of width ``grip_width`` at both ends, a gage of width ``gage_width``
  and length ``gage_length`` in the middle, joined by circular shoulder
  fillets of radius ``fillet_radius`` tangent to the gage edge (the arc meets
  the grip edge at the fillet's end, as on the specimen drawings);
* the thickness is constant ``grip_thickness`` in the grips, follows the
  measured ``stations`` (positions along the gage from its first station and
  the thickness there, natural cubic spline) inside the gage, and blends
  linearly across each fillet from the gage-end value to the grip value;
* ``n_z`` hexahedra through the thickness (one by default: pseudo-3-D).

The grid is a single mapped block: ``nx`` columns along x (cell size ``h``
in the gage, ``h`` growing geometrically to ``h_grip`` across the fillets
and ``h_grip`` in the grips), ``ny = gage_width / h`` rows across the width
at every column (so the grips carry wider cells), ``n_z`` layers scaled by
the local thickness.  No transition elements; every interface conforms.

Sets: node and side sets ``x-``, ``x+`` (the grip ends), ``z-``, ``z+``
(the two faces, planar only for a constant thickness and therefore written
as node sets of the bottom/top layer), node sets ``edge-``/``edge+`` (the
free in-plane edges), ``grips`` (both ends), ``gage-``/``gage+`` (the
columns at the gage's first and last station, for a gauge-length strain)
and ``station_k`` for every thickness station.  Blocks ``1 = gage``,
``2 = shoulder``, ``3 = grip``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core.mesh import HexMesh
from .core.topology import faces_on_plane, nodes_on_plane


@dataclass
class DogboneOptions:
    """Parameters of :func:`dogbone_mesh` (lengths in one unit, e.g. mm)."""

    length: float = 25.4
    gage_length: float = 8.0
    gage_width: float = 1.85
    grip_width: float = 4.8
    fillet_radius: float = 3.175
    grip_thickness: float = 0.35
    stations: Sequence[tuple[float, float]] | None = None   # (x from the gage start, thickness)
    h: float = 0.2
    h_grip: float | None = None       # grip cell size along x (default 2 h)
    n_z: int = 1
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    element_type: str = "HEX8"

    def __post_init__(self) -> None:
        if min(self.length, self.gage_length, self.gage_width, self.grip_width, self.fillet_radius, self.grip_thickness, self.h) <= 0:
            raise ValueError("every dog-bone dimension must be positive")
        if self.gage_width >= self.grip_width:
            raise ValueError("gage_width must be smaller than grip_width")
        if self.n_z < 1:
            raise ValueError("n_z must be at least 1")
        if self.h_grip is None:
            self.h_grip = 2.0 * self.h
        step = 0.5 * (self.grip_width - self.gage_width)
        if self.fillet_radius < step:
            raise ValueError("fillet_radius must be at least half the width step")
        self.fillet_extent = math.sqrt(self.fillet_radius ** 2 - (self.fillet_radius - step) ** 2)
        if self.gage_length + 2 * self.fillet_extent >= self.length:
            raise ValueError("the gage and the fillets do not fit inside the length")
        if self.stations:
            xs = [float(x) for x, _ in self.stations]
            if any(x < -1e-9 or x > self.gage_length + 1e-9 for x in xs) or xs != sorted(xs):
                raise ValueError("stations must be sorted and lie within [0, gage_length]")


# ----------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------
def _natural_cubic_spline(xs: np.ndarray, ys: np.ndarray):
    """Natural cubic spline through (xs, ys); returns f(x) clamped to the data range."""
    n = len(xs)
    if n == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), ys[0])
    if n == 2:
        return lambda x: np.interp(np.asarray(x, dtype=float), xs, ys)
    h = np.diff(xs)
    a = np.zeros((n, n)); r = np.zeros(n)
    a[0, 0] = a[-1, -1] = 1.0
    for i in range(1, n - 1):
        a[i, i - 1], a[i, i], a[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        r[i] = 3 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    c = np.linalg.solve(a, r)
    b = (np.diff(ys) / h) - h * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    def f(x):
        x = np.clip(np.asarray(x, dtype=float), xs[0], xs[-1])
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, n - 2)
        dx = x - xs[i]
        return ys[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return f


def half_width(opts: DogboneOptions, x: np.ndarray) -> np.ndarray:
    """Half-width of the planform at axial position ``x`` (from the bar's left end)."""
    x = np.asarray(x, dtype=float)
    x0 = 0.5 * (opts.length - opts.gage_length)          # gage start
    x1 = x0 + opts.gage_length                            # gage end
    R, w2, W2, d = opts.fillet_radius, 0.5 * opts.gage_width, 0.5 * opts.grip_width, opts.fillet_extent
    out = np.full_like(x, W2)
    gage = (x >= x0) & (x <= x1)
    out[gage] = w2
    left = (x >= x0 - d) & (x < x0)
    out[left] = w2 + R - np.sqrt(np.maximum(R ** 2 - (x[left] - x0) ** 2, 0.0))
    right = (x > x1) & (x <= x1 + d)
    out[right] = w2 + R - np.sqrt(np.maximum(R ** 2 - (x[right] - x1) ** 2, 0.0))
    return out


def thickness(opts: DogboneOptions, x: np.ndarray) -> np.ndarray:
    """Thickness at axial position ``x``: spline through the stations in the gage, linear
    blend to the grip thickness across each fillet, constant in the grips."""
    x = np.asarray(x, dtype=float)
    if not opts.stations:
        return np.full_like(x, opts.grip_thickness)
    x0 = 0.5 * (opts.length - opts.gage_length); x1 = x0 + opts.gage_length; d = opts.fillet_extent
    sx = np.asarray([x0 + float(s) for s, _ in opts.stations]); st = np.asarray([float(t) for _, t in opts.stations])
    f = _natural_cubic_spline(sx, st)
    out = np.full_like(x, opts.grip_thickness)
    gage = (x >= x0) & (x <= x1)
    out[gage] = f(x[gage])
    t_left, t_right = float(f(x0)), float(f(x1))
    left = (x >= x0 - d) & (x < x0)
    out[left] = opts.grip_thickness + (t_left - opts.grip_thickness) * (x[left] - (x0 - d)) / d
    right = (x > x1) & (x <= x1 + d)
    out[right] = t_right + (opts.grip_thickness - t_right) * (x[right] - x1) / d
    return out


def _x_stations(opts: DogboneOptions) -> tuple[np.ndarray, np.ndarray]:
    """Column positions along x and a per-cell region code (0 gage, 1 shoulder, 2 grip)."""
    x0 = 0.5 * (opts.length - opts.gage_length); x1 = x0 + opts.gage_length; d = opts.fillet_extent
    h, hg = opts.h, float(opts.h_grip)
    n_g = max(1, round(opts.gage_length / h))
    gage = np.linspace(x0, x1, n_g + 1)
    # fillet: geometric growth from h to hg over d
    n_f = max(1, round(2 * d / (h + hg)))
    r = (hg / h) ** (1.0 / max(n_f - 1, 1)) if n_f > 1 else 1.0
    w = h * r ** np.arange(n_f); s = np.concatenate([[0.0], np.cumsum(w)]); s = s / s[-1] * d
    left_f = x0 - d + s                      # ascending, ends at x0
    right_f = x1 + s                         # starts at x1
    grip_len = x0 - d
    n_grip = max(1, round(grip_len / hg))
    left_g = np.linspace(0.0, x0 - d, n_grip + 1)
    right_g = np.linspace(x1 + d, opts.length, n_grip + 1)
    xs = np.concatenate([left_g[:-1], left_f[:-1], gage[:-1], right_f[:-1], right_g])
    region = np.concatenate([np.full(n_grip, 2), np.full(n_f, 1), np.full(n_g, 0), np.full(n_f, 1), np.full(n_grip, 2)])
    return xs, region


def dogbone_mesh(opts: DogboneOptions) -> HexMesh:
    """Build the dog-bone (see the module docstring).  Returns a HEX8 :class:`HexMesh`."""
    if str(opts.element_type).upper() != "HEX8":
        raise ValueError("dogbone_mesh builds HEX8 meshes; elevate afterwards for HEX20/HEX27")
    xs, region = _x_stations(opts)
    ny = max(1, round(opts.gage_width / opts.h))
    nz = int(opts.n_z)
    nx = len(xs) - 1
    yw = half_width(opts, xs); tz = thickness(opts, xs)
    eta = np.linspace(-1.0, 1.0, ny + 1); zeta = np.linspace(0.0, 1.0, nz + 1)
    ox, oy, oz = (float(v) for v in opts.origin)
    pts = np.empty(((nx + 1), (ny + 1), (nz + 1), 3))
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                pts[i, j, k] = (ox + xs[i], oy + yw[i] * eta[j], oz + tz[i] * zeta[k])
    ids = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
    cells: list[list[int]] = []; block: list[int] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cells.append([ids[i, j, k], ids[i + 1, j, k], ids[i + 1, j + 1, k], ids[i, j + 1, k],
                              ids[i, j, k + 1], ids[i + 1, j, k + 1], ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]])
                block.append(int(region[i]) + 1)
    meta: dict[str, object] = {
        "generator": "dogbone_mesh", "length": opts.length, "gage_length": opts.gage_length,
        "gage_width": opts.gage_width, "grip_width": opts.grip_width, "fillet_radius": opts.fillet_radius,
        "fillet_extent": opts.fillet_extent, "grip_thickness": opts.grip_thickness,
        "stations": [list(map(float, s)) for s in (opts.stations or [])], "h": opts.h, "h_grip": opts.h_grip,
        "divisions": [nx, ny, nz], "gage_cells": int((region == 0).sum()),
        "thickness_range": [float(tz.min()), float(tz.max())],
    }
    mesh = HexMesh(points=pts.reshape(-1, 3), cells=np.asarray(cells, dtype=np.int64),
                   block_id=np.asarray(block, dtype=np.int32),
                   block_names={1: "gage", 2: "shoulder", 3: "grip"}, element_type="HEX8", metadata=meta)
    tol = 1e-9 * opts.h
    for name, value in (("x-", ox), ("x+", ox + opts.length)):
        elems, sides = faces_on_plane(mesh, 0, value, tol)
        mesh.add_side_set(name, elems, sides)
        mesh.add_node_set(name, nodes_on_plane(mesh, 0, value, tol))
    mesh.add_node_set("grips", np.concatenate([mesh.node_sets["x-"], mesh.node_sets["x+"]]))
    mesh.add_node_set("z-", ids[:, :, 0].ravel()); mesh.add_node_set("z+", ids[:, :, nz].ravel())
    mesh.add_node_set("edge-", ids[:, 0, :].ravel()); mesh.add_node_set("edge+", ids[:, ny, :].ravel())
    x0 = 0.5 * (opts.length - opts.gage_length); x1 = x0 + opts.gage_length
    for name, value in (("gage-", x0), ("gage+", x1)):
        mesh.add_node_set(name, nodes_on_plane(mesh, 0, ox + value, tol))
    for k, (s, _) in enumerate(opts.stations or []):
        i = int(np.argmin(np.abs(xs - (x0 + float(s)))))
        mesh.add_node_set(f"station_{k + 1}", ids[i].ravel())
    return mesh


def dogbone_summary(mesh: HexMesh) -> dict[str, object]:
    from .core.topology import signed_volumes

    vol = signed_volumes(mesh.points, mesh.cells)
    meta = dict(mesh.metadata)
    meta.update({"n_nodes": int(mesh.n_nodes), "n_elements": int(mesh.n_elements), "n_inverted": int((vol <= 0).sum()),
                 "blocks": {int(b): int((mesh.block_id == b).sum()) for b in np.unique(mesh.block_id)}})
    return meta
