"""Geometry abstraction for inclusions.

The single most important design decision in Hexaform is this interface.  The
macro-block template mesher never asks "what radius is this sphere?".  It asks
"how far is the surface from the centre along this direction?".  Any body that
can answer that question -- a sphere, an ellipsoid, a superquadric, a
spherical-harmonic grain, a rounded polyhedron -- can be meshed by exactly the
same code path.

That is what turns "a cube containing spheres" into "a cube containing
arbitrary star-shaped natural forms" without rewriting the mesher.

Definitions
-----------
A body is **star-shaped** with respect to an interior point ``c`` if every ray
from ``c`` crosses the surface exactly once.  Formally there is a single-valued
support radius

.. math::  r : S^2 \\to \\mathbb{R}^+, \\qquad
           \\partial\\Omega = \\{ c + r(\\hat{d})\\,\\hat{d} \\}.

Every shape in :mod:`hexaform.shapes` implements ``support_radius``.  Bodies
that are *not* star-shaped (a torus, a bone, a branching pore network) cannot
use the template backend and are the target of the future octree backend
described in ``docs/roadmap.md``.
"""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np


def normalize(d: np.ndarray) -> np.ndarray:
    """Unit vectors, with a well-defined answer at the origin.

    A zero vector is mapped to ``+x`` rather than to another zero vector.  This
    matters because the centre node of an inclusion's core block lies exactly at
    the shape centre, and every ``support_radius`` implementation would divide by
    zero if handed a zero direction.
    """
    d = np.atleast_1d(np.asarray(d, dtype=np.float64))
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    out = np.divide(d, n, out=np.zeros_like(d), where=n > 0.0)
    degenerate = np.broadcast_to(n <= 0.0, out.shape)
    if degenerate.any():
        fallback = np.zeros(out.shape[-1])
        fallback[0] = 1.0
        out = np.where(degenerate, fallback, out)
    return out


class Shape(abc.ABC):
    """Base class for an implicitly defined solid body."""

    name: str = "shape"

    @property
    @abc.abstractmethod
    def center(self) -> np.ndarray:
        """Interior reference point (centroid for the shipped shapes)."""

    @abc.abstractmethod
    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Signed distance (negative inside).  May be an approximation."""

    @abc.abstractmethod
    def bounding_radius(self) -> float:
        """Radius of a sphere about :attr:`center` containing the body."""

    @abc.abstractmethod
    def volume(self) -> float:
        """Exact or accurately estimated volume, used for validation."""

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        r = self.bounding_radius()
        return self.center - r, self.center + r

    def contains(self, x: np.ndarray) -> np.ndarray:
        return self.sdf(x) < 0.0


class StarShape(Shape):
    """A body meshable by the cubed-sphere macro-block template."""

    @abc.abstractmethod
    def support_radius(self, directions: np.ndarray) -> np.ndarray:
        """Distance from :attr:`center` to the surface along unit ``directions``.

        Parameters
        ----------
        directions:
            ``(..., 3)`` array of unit vectors.

        Returns
        -------
        ``(...,)`` array of positive radii.
        """

    # -- derived operations shared by every star shape ---------------------
    def with_center(self, center: np.ndarray) -> "StarShape":
        """A copy of this body translated so its :attr:`center` is ``center``.

        The surface shape is unchanged (``support_radius`` does not depend on
        the centre), so the packer can size a macro-box from a body created at
        a placeholder centre and move it once its box is placed.  The default
        implementation re-instantiates dataclass-based shapes whose centre is
        stored in a ``centre`` or ``center`` field; other classes override it.
        """
        import dataclasses

        if dataclasses.is_dataclass(self) and not isinstance(self, type):
            names = {f.name for f in dataclasses.fields(self)}
            for name in ("centre", "center"):
                if name in names:
                    return dataclasses.replace(
                        self, **{name: np.asarray(center, dtype=np.float64).reshape(3)}
                    )
        raise NotImplementedError(
            f"{type(self).__name__} does not store its centre in a 'centre' "
            "or 'center' dataclass field; override with_center to support "
            "grid-aware packing"
        )

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of the body, from a cubed-sphere sampling.

        The sample is dense enough that the error is far below any macro-box
        margin (``box_ratio >= 1.05`` in the packer), and the box is used only
        for per-axis macro-box sizing -- never for validity.  For rotated
        bodies this is an approximation to the true AABB; for axis-aligned
        bodies it is exact up to floating-point error.
        """
        from ..templates.cubed_sphere import CubedSphere  # local import: avoid cycle

        pts = self.surface_points(CubedSphere(48).directions)
        return pts.min(axis=0), pts.max(axis=0)

    def surface_points(self, directions: np.ndarray) -> np.ndarray:
        """Points on the exact analytic surface along ``directions``."""
        d = normalize(directions)
        return self.center + self.support_radius(d)[..., None] * d

    def project(self, x: np.ndarray) -> np.ndarray:
        """Radially project arbitrary points onto the exact surface.

        This is the operation that removes voxel staircasing: mesh topology may
        be produced by any means, but every interface node is snapped to the
        analytic body before export.
        """
        x = np.asarray(x, dtype=np.float64)
        return self.surface_points(x - self.center)

    def radial_scale(self, x: np.ndarray) -> np.ndarray:
        """``|x - c| / r(dir)``: 1 on the surface, <1 inside, >1 outside."""
        v = np.asarray(x, dtype=np.float64) - self.center
        rho = np.linalg.norm(v, axis=-1)
        return rho / np.maximum(self.support_radius(normalize(v)), 1e-300)

    def sdf(self, x: np.ndarray) -> np.ndarray:  # noqa: D102 - inherited
        v = np.asarray(x, dtype=np.float64) - self.center
        rho = np.linalg.norm(v, axis=-1)
        return rho - self.support_radius(normalize(v))

    def surface_area(self, n: int = 64) -> float:
        """Monte-Carlo-free surface area estimate from a cubed-sphere sampling."""
        from ..templates.cubed_sphere import CubedSphere  # local import: avoid cycle

        cs = CubedSphere(n)
        pts = self.surface_points(cs.directions)
        quads = cs.quads
        a = pts[quads[:, 0]]
        b = pts[quads[:, 1]]
        c = pts[quads[:, 2]]
        d = pts[quads[:, 3]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        area += 0.5 * np.linalg.norm(np.cross(c - a, d - a), axis=1)
        return float(area.sum())

    def volume(self) -> float:  # noqa: D102 - generic fallback
        from ..templates.cubed_sphere import CubedSphere

        cs = CubedSphere(96)
        pts = self.surface_points(cs.directions) - self.center
        q = cs.quads
        a, b, c, d = pts[q[:, 0]], pts[q[:, 1]], pts[q[:, 2]], pts[q[:, 3]]
        vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        vol += np.einsum("ij,ij->i", a, np.cross(c, d)) / 6.0
        return float(abs(vol.sum()))
