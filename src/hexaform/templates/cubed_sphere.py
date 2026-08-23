"""Six-patch cubed-sphere quadrangulation.

Both source reviews single this out as the correct surface parameterisation for
an inclusion: unlike a latitude/longitude grid it has **no polar singularity**,
so every quad on the surface is well shaped and every hexahedron swept from it
is well shaped too.

Construction
------------
Start from the surface of the cube :math:`[-1,1]^3`, split into six
:math:`(n+1)\\times(n+1)` structured patches, and normalise each point::

    d = q / |q|,   q on the cube surface.

Two 1-D spacings are supported for the patch parameter :math:`u_i = -1 + 2i/n`:

``uniform``
    ``s = u`` (gnomonic).  Element sizes on the sphere grow towards patch
    centres by up to ``sqrt(3)``, but the patch parameterisation matches a
    uniform Cartesian face exactly.
``equiangular``
    ``s = tan(pi u / 4)``.  Near-uniform arc length on the sphere; the standard
    choice in atmospheric dynamical cores.

``blend`` interpolates between them, which matters for Hexaform because the
outer end of every macro-block shell must match a *uniform* background grid
face.  A partial blend trades a little sphere-surface uniformity for a lot less
shear in the transition shell; see ``docs/algorithm.md`.

Anisotropic resolution
----------------------
``n`` may be a scalar (the isotropic cubed sphere) or a per-axis triple
``(nx, ny, nz)``.  The ``x+`` patch then carries ``ny x nz`` quads, the ``y+``
patch ``nz x nx``, and the ``z+`` patch ``nx x ny``; the same for the negative
faces.  A shared edge always uses the 1-D spacing of the axis it runs along,
produced by identical arithmetic on both adjacent patches, so the exact welding
survives anisotropy.  This is what lets a ``kx x ky x kz`` macro-box match the
background grid face-by-face; see ``packing/rsa.py`` for the per-axis sizing.

Patch orientation
-----------------
Each face is defined by an outward normal and two in-plane index directions
chosen so that ``i_dir x j_dir = outward``.  Quads generated as
``(i,j), (i+1,j), (i+1,j+1), (i,j+1)`` therefore wind counter-clockwise seen
from outside, and hexahedra swept radially outward from them have positive
Jacobian without any post-hoc flipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])

# name -> (outward normal, i direction, j direction); i x j == normal
FACES: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "x+": (_EX, _EY, _EZ),
    "x-": (-_EX, _EZ, _EY),
    "y+": (_EY, _EZ, _EX),
    "y-": (-_EY, _EX, _EZ),
    "z+": (_EZ, _EX, _EY),
    "z-": (-_EZ, _EY, _EX),
}
FACE_ORDER: List[str] = ["x+", "x-", "y+", "y-", "z+", "z-"]


def patch_spacing(n: int, mode: str = "equiangular", blend: float = 1.0) -> np.ndarray:
    """1-D patch coordinate array of length ``n + 1`` spanning ``[-1, 1]``."""
    u = np.linspace(-1.0, 1.0, n + 1)
    if mode == "uniform":
        return u
    if mode not in ("equiangular", "blend"):
        raise ValueError(f"unknown patch spacing mode {mode!r}")
    s_eq = np.tan(0.25 * np.pi * u)
    if mode == "equiangular":
        blend = 1.0
    w = float(np.clip(blend, 0.0, 1.0))
    return (1.0 - w) * u + w * s_eq


@dataclass
class CubedSphere:
    """Unit-sphere cubed quadrangulation, isotropic or anisotropic.

    Attributes
    ----------
    n:
        Cells along each patch edge: a scalar, or per-axis ``(nx, ny, nz)``.
        With ``a = nx+1, b = ny+1, c = nz+1`` the total surface nodes are
        ``2(ab + bc + ca) - 4(a + b + c) + 8`` and the total quads
        ``2(nx*ny + ny*nz + nz*nx)``.
    n_axis:
        The per-axis resolution as an ``(nx, ny, nz)`` int array.
    cube_points:
        ``(n_nodes, 3)`` welded points on the cube surface (before normalising).
    directions:
        ``(n_nodes, 3)`` unit vectors -- the sampling directions handed to
        :meth:`~hexaform.shapes.base.StarShape.support_radius`.
    index:
        ``face name -> (n+1, n+1)`` int array of global node ids.
    quads:
        ``(6 n^2, 4)`` outward-wound quadrilaterals.
    """

    n: int | Tuple[int, int, int]
    """Cells along each patch edge: one value, or one per axis ``(nx, ny, nz)``."""

    spacing: str = "equiangular"
    blend: float = 1.0

    def __post_init__(self) -> None:
        na = np.broadcast_to(np.asarray(self.n, dtype=np.int64), (3,))
        if np.any(na < 1):
            raise ValueError("cubed-sphere resolution n must be >= 1 per axis")
        self.n_axis = na
        # one 1-D spacing per lattice axis; a patch edge running along axis a
        # is sampled with s_a, whatever the two adjacent patches are
        self._s = tuple(patch_spacing(int(a), self.spacing, self.blend) for a in na)

        raw: Dict[str, np.ndarray] = {}
        face_shape: Dict[str, Tuple[int, int]] = {}
        for name in FACE_ORDER:
            normal, idir, jdir = FACES[name]
            ai = int(np.argmax(np.abs(idir)))
            aj = int(np.argmax(np.abs(jdir)))
            si = self._s[ai][:, None, None] * idir[None, None, :]
            sj = self._s[aj][None, :, None] * jdir[None, None, :]
            raw[name] = normal[None, None, :] + si + sj
            face_shape[name] = raw[name].shape[:2]

        # Weld shared patch edges.  Coordinates on shared edges are produced by
        # identical arithmetic on both patches, so exact hashing is valid.
        flat = np.concatenate([raw[f].reshape(-1, 3) for f in FACE_ORDER], axis=0)
        keys = np.round(flat, 12)
        uniq, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        self.cube_points = flat[first]
        norms = np.linalg.norm(self.cube_points, axis=1, keepdims=True)
        self.directions = self.cube_points / norms

        self.index = {}
        start = 0
        for name in FACE_ORDER:
            mi, mj = face_shape[name]
            self.index[name] = inverse[start : start + mi * mj].reshape(mi, mj)
            start += mi * mj

        quads = []
        for name in FACE_ORDER:
            idx = self.index[name]
            a = idx[:-1, :-1].ravel()
            b = idx[1:, :-1].ravel()
            c = idx[1:, 1:].ravel()
            d = idx[:-1, 1:].ravel()
            quads.append(np.stack([a, b, c, d], axis=1))
        self.quads = np.concatenate(quads, axis=0).astype(np.int64)

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.cube_points.shape[0])

    @property
    def n_quads(self) -> int:
        return int(self.quads.shape[0])

    @property
    def spacing_1d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The per-axis 1-D patch coordinate arrays used to build the patches."""
        return self._s

    def face_directions(self, name: str) -> np.ndarray:
        """``(n+1, n+1, 3)`` unit directions for one patch."""
        return self.directions[self.index[name]]

    def face_cube_points(self, name: str) -> np.ndarray:
        """``(n+1, n+1, 3)`` cube-surface points for one patch."""
        return self.cube_points[self.index[name]]

    def check(self) -> Dict[str, object]:
        """Self-consistency checks used by the test suite."""
        a, b, c = tuple(int(v) + 1 for v in self.n_axis)
        # faces - shared edges + vertices of a rectangular cubed quadrangulation
        expected_nodes = 2 * (a * b + b * c + c * a) - 4 * (a + b + c) + 8
        expected_quads = 2 * ((a - 1) * (b - 1) + (b - 1) * (c - 1) + (c - 1) * (a - 1))
        areas = np.linalg.norm(
            np.cross(
                self.directions[self.quads[:, 1]] - self.directions[self.quads[:, 0]],
                self.directions[self.quads[:, 3]] - self.directions[self.quads[:, 0]],
            ),
            axis=1,
        )
        centroids = self.directions[self.quads].mean(axis=1)
        normals = np.cross(
            self.directions[self.quads[:, 1]] - self.directions[self.quads[:, 0]],
            self.directions[self.quads[:, 3]] - self.directions[self.quads[:, 0]],
        )
        outward = np.einsum("ij,ij->i", normals, centroids)
        return {
            "n_nodes": self.n_nodes,
            "expected_nodes": expected_nodes,
            "nodes_ok": self.n_nodes == expected_nodes,
            "n_quads": self.n_quads,
            "expected_quads": expected_quads,
            "quads_ok": self.n_quads == expected_quads,
            "min_quad_area": float(areas.min()),
            "all_outward": bool(np.all(outward > 0)),
            "max_radius_error": float(
                np.abs(np.linalg.norm(self.directions, axis=1) - 1.0).max()
            ),
        }
