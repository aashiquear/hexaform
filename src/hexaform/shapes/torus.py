"""Non-star-shaped bodies.

The simplest body that is *not* star-shaped is a torus.  A torus has a hole
in the middle, so a ray from the centre crosses its surface either zero
times (toward the hole) or twice (on the other side).  This rules out
single-valued ``support_radius`` and is the reason the cubed-sphere macroblock
template is restricted to star shapes.

Tori, branching pore networks and interlocking fibres are the "real frontier
of the arbitrary shape goal" in ``docs/roadmap.md`` §6.  This module provides
the analytic surface definition and a *projection* operator -- the operation
that places an arbitrary point on the nearest surface point -- that the octree
backend uses to snap interface nodes to the exact body.

The API
-------
:class:`Torus` is a :class:`~hexaform.shapes.base.Shape` (not
:class:`StarShape`): it provides ``sdf``, ``bounding_radius`` and
``volume``, but not ``support_radius``.  This signals to the rest of
Hexaform that the cubed-sphere template cannot be used; the octree backend
must be selected instead.

The :func:`project_to_torus` function implements the closed-form projection
of a point onto the analytic torus surface, derived by solving

.. math::  \\arg\\min_{p \\in \\partial\\Omega} \\|x - p\\|_2

in the local coordinate frame of the torus.  This is the operation the
roadmap points to when it says "surface projection against a general implicit
or B-Rep surface."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Shape, normalize


@dataclass
class Torus(Shape):
    """Analytic torus, the prototypical non-star-shaped body.

    A torus with major radius ``R`` and minor radius ``r`` is the surface of
    revolution at distance ``R`` from the z-axis of a circle of radius ``r``
    in the xz-plane.  Its implicit equation is

    .. math::  (\\sqrt{x^2 + y^2} - R)^2 + z^2 = r^2.

    The body is **not** star-shaped when ``r < R`` (the inner hole means
    certain directions from the centre never meet the surface).  The
    pipeline therefore routes it to the octree backend with the
    :func:`project_to_torus` snap-to-surface operation.
    """

    centre: np.ndarray
    major_radius: float  # R, distance from the centre of the tube to the centre of the torus
    minor_radius: float  # r, radius of the tube
    axis: np.ndarray = None  # axis of revolution; defaults to z
    name: str = "torus"

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.major_radius = float(self.major_radius)
        self.minor_radius = float(self.minor_radius)
        if self.major_radius <= 0 or self.minor_radius <= 0:
            raise ValueError("torus radii must be positive")
        if self.axis is None:
            self.axis = np.array([0.0, 0.0, 1.0])
        else:
            self.axis = np.asarray(self.axis, dtype=np.float64).reshape(3)
            self.axis = self.axis / max(np.linalg.norm(self.axis), 1e-12)
        self._rotation = self._basis()

    def _basis(self) -> np.ndarray:
        """Build an orthonormal frame (e1, e2, axis)."""
        axis = self.axis
        # Pick a vector not parallel to axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = ref - (ref @ axis) * axis
        e1 = e1 / max(np.linalg.norm(e1), 1e-12)
        e2 = np.cross(axis, e1)
        return np.stack([e1, e2, axis], axis=1)  # (3, 3): columns are basis vectors

    @property
    def center(self) -> np.ndarray:
        return self.centre

    def sdf(self, x: np.ndarray) -> np.ndarray:
        """Signed distance (negative inside the solid torus)."""
        x = np.asarray(x, dtype=np.float64)
        # Bring to torus frame
        v = x - self.centre
        body = v @ self._rotation  # (..., 3) in (e1, e2, axis) coordinates
        rho = np.sqrt(body[..., 0] ** 2 + body[..., 1] ** 2)
        q = np.sqrt((rho - self.major_radius) ** 2 + body[..., 2] ** 2)
        return q - self.minor_radius

    def bounding_radius(self) -> float:
        """Bounding sphere radius, including the rim of the tube."""
        return self.major_radius + self.minor_radius

    def bounding_box(self) -> tuple:
        """Axis-aligned bounding box (using the canonical frame)."""
        r = self.bounding_radius()
        return self.centre - r, self.centre + r

    def volume(self) -> float:
        """Volume of a solid torus: ``2 pi^2 R r^2``."""
        return 2.0 * np.pi ** 2 * self.major_radius * self.minor_radius ** 2

    def surface_area(self) -> float:
        """Surface area: ``4 pi^2 R r``."""
        return 4.0 * np.pi ** 2 * self.major_radius * self.minor_radius

    # ------------------------------------------------------------------
    # The projection operation -- this is what makes the octree backend
    # able to place interface nodes exactly on the analytic surface.
    # ------------------------------------------------------------------
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project points onto the analytic torus surface.

        Implements the closed-form projection from a point to a torus
        surface, in the local frame (e1, e2, axis).  Algorithm:

        1. Translate and rotate to the local frame (x', y', z').
        2. Compute the polar coordinates of (x', y') in the e1-e2 plane.
        3. The closest point on the *central circle* (rho = R) is
           ``(R cos(phi), R sin(phi), 0)``.
        4. The closest point on the *tube* surface is the radial direction
           from the central circle, at distance ``r`` from it.
        """
        return project_to_torus(x, self.centre, self.major_radius,
                                self.minor_radius, self._rotation)


def project_to_torus(
    x: np.ndarray,
    centre: np.ndarray,
    R: float,
    r: float,
    rotation: np.ndarray,
) -> np.ndarray:
    """Closed-form projection of points onto an analytic torus.

    Parameters
    ----------
    x:
        ``(N, 3)`` array of points in world space.
    centre, R, r, rotation:
        The torus's centre, major/minor radii, and the (3, 3) body-to-world
        rotation matrix whose columns are ``(e1, e2, axis)``.

    Returns
    -------
    ``(N, 3)`` array of projected points, in world space.
    """
    x = np.asarray(x, dtype=np.float64)
    v = x - centre
    body = v @ rotation  # (N, 3) in local frame
    x_local = body[..., 0]
    y_local = body[..., 1]
    z_local = body[..., 2]
    rho = np.sqrt(x_local ** 2 + y_local ** 2)

    # For each point we have a 1-D optimisation in the angle theta along the
    # minor circle.  The phi direction is fixed by the (x, y) input -- the
    # central circle is at (R cos phi, R sin phi, 0) -- and we need the
    # closest point on the circle of radius r in the plane spanned by the
    # outward radial direction (cos phi, sin phi, 0) and the z axis.
    safe_rho = np.where(rho > 1e-12, rho, 1e-12)
    cos_phi = np.where(rho > 1e-12, x_local / safe_rho, 1.0)
    sin_phi = np.where(rho > 1e-12, y_local / safe_rho, 0.0)
    # The closest point on the central circle is at (R cos phi, R sin phi, 0).
    # The vector from there to the input point lives in the (radial, z) plane.
    radial_offset = rho - R  # signed distance from the central circle in the radial direction
    z_offset = z_local
    # The projection onto the tube of radius r in this plane is the (radial,
    # z) point closest to (radial_offset, z_offset) on the circle of radius
    # r.  The trivial projection `r * v / |v|` would be wrong because the
    # circle is centred at the origin, but the closest point on the circle
    # to v is `r * v / |v|` ONLY when |v| is the distance from the origin --
    # which it is here, since the circle IS centred at the origin in this
    # 2-D plane.
    v_norm = np.sqrt(radial_offset ** 2 + z_offset ** 2)
    safe_v = np.where(v_norm > 1e-12, v_norm, 1e-12)
    radial_proj = r * radial_offset / safe_v
    z_proj = r * z_offset / safe_v
    # For the degenerate case where the input is exactly on the central
    # circle (rho = R, z = 0), the radial+axial direction is undefined.
    # Choose the closest point on the tube: this is the inner or outer
    # point of the central circle, which is (R - r, 0, 0) or (R + r, 0, 0)
    # in the local (radial, z) frame.  For a generic query, the input is
    # inside the tube (SDF < 0); snap to the outer rim (R + r) so the
    # interface stays on the surface of the *solid* torus.
    on_central = v_norm <= 1e-12
    if np.any(on_central):
        # If the input is on the axis of revolution, choose the inner rim
        # to keep the closest point definition unambiguous.
        radial_proj = np.where(on_central, r, radial_proj)
        z_proj = np.where(on_central, 0.0, z_proj)

    # Now construct the world point: the central circle point plus the
    # radial+axial offset.
    px_local = R * cos_phi + radial_proj * cos_phi
    py_local = R * sin_phi + radial_proj * sin_phi
    pz_local = z_proj
    # Handle the rho=0 case (the input is on the axis).  Project onto the
    # inner rim of the torus: (R - r, 0, 0) in local frame.
    axis_input = rho <= 1e-12
    if np.any(axis_input):
        # Choose the closest of (R - r) and (R + r) to the input -- but the
        # closest point to the origin is always (R - r) on the inner rim.
        px_local = np.where(axis_input, R - r, px_local)
        py_local = np.where(axis_input, 0.0, py_local)
        pz_local = np.where(axis_input, 0.0, pz_local)

    body_proj = np.stack([px_local, py_local, pz_local], axis=-1)
    return body_proj @ rotation.T + centre


__all__ = ["Torus", "project_to_torus"]
