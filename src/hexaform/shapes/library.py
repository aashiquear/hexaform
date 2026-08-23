"""Concrete star-shaped bodies.

``Sphere`` is the shape required by the first use case.  The others exist to
demonstrate -- and to keep honest -- the claim that the template mesher is
shape-agnostic: they are consumed by exactly the same
:class:`~hexaform.templates.macroblock.MacroBlockTemplate` code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import StarShape, normalize


@dataclass
class Sphere(StarShape):
    """Analytic sphere: ``|x - c| = R``."""

    centre: np.ndarray
    radius: float
    name: str = "sphere"

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.radius = float(self.radius)
        if self.radius <= 0:
            raise ValueError("sphere radius must be positive")

    @property
    def center(self) -> np.ndarray:
        return self.centre

    def support_radius(self, directions: np.ndarray) -> np.ndarray:
        d = np.asarray(directions, dtype=np.float64)
        return np.full(d.shape[:-1], self.radius)

    def bounding_radius(self) -> float:
        return self.radius

    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.radius**3

    def surface_area(self, n: int = 64) -> float:
        return 4.0 * np.pi * self.radius**2

    def sdf(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(x) - self.centre, axis=-1) - self.radius


@dataclass
class Ellipsoid(StarShape):
    """Axis-aligned or rotated ellipsoid with semi-axes ``a, b, c``."""

    centre: np.ndarray
    semi_axes: np.ndarray
    rotation: np.ndarray | None = None  # (3, 3) body -> world
    name: str = "ellipsoid"

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.semi_axes = np.asarray(self.semi_axes, dtype=np.float64).reshape(3)
        if np.any(self.semi_axes <= 0):
            raise ValueError("semi-axes must be positive")
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)

    @property
    def center(self) -> np.ndarray:
        return self.centre

    def _to_body(self, d: np.ndarray) -> np.ndarray:
        return d if self.rotation is None else d @ self.rotation

    def support_radius(self, directions: np.ndarray) -> np.ndarray:
        d = self._to_body(normalize(directions))
        return 1.0 / np.sqrt(np.sum((d / self.semi_axes) ** 2, axis=-1))

    def bounding_radius(self) -> float:
        return float(self.semi_axes.max())

    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * float(np.prod(self.semi_axes))


@dataclass
class Superquadric(StarShape):
    """Superellipsoid ``(|x/a|^r + |y/b|^r)^{t/r} + |z/c|^t = 1``.

    With ``r = t = 2`` this is an ellipsoid; larger exponents give the
    rounded-cuboidal forms typical of crushed aggregate and mineral grains,
    smaller exponents give octahedral/pillow forms.
    """

    centre: np.ndarray
    semi_axes: np.ndarray
    exponent_xy: float = 2.0
    exponent_z: float = 2.0
    rotation: np.ndarray | None = None
    name: str = "superquadric"

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.semi_axes = np.asarray(self.semi_axes, dtype=np.float64).reshape(3)
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)

    @property
    def center(self) -> np.ndarray:
        return self.centre

    def support_radius(self, directions: np.ndarray) -> np.ndarray:
        d = normalize(directions)
        if self.rotation is not None:
            d = d @ self.rotation
        u = np.abs(d / self.semi_axes)
        r, t = self.exponent_xy, self.exponent_z
        # F(s*d) = 1 with F homogeneous of the implicit form below
        f = (u[..., 0] ** r + u[..., 1] ** r) ** (t / r) + u[..., 2] ** t
        return f ** (-1.0 / t)

    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.semi_axes))


@dataclass
class SphericalHarmonicGrain(StarShape):
    """Random natural grain: a sphere modulated by real spherical harmonics.

    .. math:: r(\\theta,\\phi) = R_0\\left(1 + \\sum_{l,m} a_{lm} Y_{lm}\\right)

    This is the standard representation for irregular aggregate, sand and
    powder particles.  Coefficients decay as ``l^{-beta}`` so the surface stays
    smooth and star-shaped; ``roughness`` scales the overall perturbation and is
    clamped so ``r > 0`` everywhere.
    """

    centre: np.ndarray
    radius: float
    coefficients: dict = field(default_factory=dict)  # (l, m) -> a_lm
    name: str = "sh_grain"

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.radius = float(self.radius)

    @classmethod
    def random(
        cls,
        centre: Sequence[float],
        radius: float,
        rng: np.random.Generator,
        l_max: int = 4,
        roughness: float = 0.12,
        decay: float = 2.0,
    ) -> "SphericalHarmonicGrain":
        coeffs: dict = {}
        for l in range(2, l_max + 1):
            for m in range(-l, l + 1):
                coeffs[(l, m)] = rng.normal(0.0, roughness / l**decay)
        return cls(np.asarray(centre, dtype=float), float(radius), coeffs)

    @property
    def center(self) -> np.ndarray:
        return self.centre

    def support_radius(self, directions: np.ndarray) -> np.ndarray:
        d = normalize(directions)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        pert = np.zeros(d.shape[:-1])
        for (l, m), a in self.coefficients.items():
            if a == 0.0:
                continue
            pert = pert + a * _real_sph_harm(l, m, theta, phi)
        # keep the body star-shaped and strictly positive
        pert = np.clip(pert, -0.6, 2.0)
        return self.radius * (1.0 + pert)

    def bounding_radius(self) -> float:
        from ..templates.cubed_sphere import CubedSphere

        return float(self.support_radius(CubedSphere(32).directions).max())


def _real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real spherical harmonic ``Y_lm`` (Schmidt semi-normalised, real form)."""
    from scipy.special import sph_harm_y  # type: ignore

    val = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return np.sqrt(2.0) * (-1.0) ** m * val.real
    if m < 0:
        return np.sqrt(2.0) * (-1.0) ** m * val.imag
    return val.real


__all__ = ["Sphere", "Ellipsoid", "Superquadric", "SphericalHarmonicGrain"]
