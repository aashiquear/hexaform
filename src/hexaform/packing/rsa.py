"""Random inclusion packing on the background grid.

Both reviews make the same recommendation: keep packing and meshing separate.
The packer's only output is a list of shapes plus the macro-box each will
occupy; the mesher never asks how they were generated.

Hexaform adds one twist to standard random sequential addition (RSA).  Instead
of enforcing a geometric gap

.. math::  |c_i - c_j| > R_i + R_j + g_{\\min}

with an arbitrary tolerance, it enforces **integer macro-box disjointness with a
margin of whole background cells**.  This is strictly stronger, it is exact, and
it is precisely the condition the mesher needs to succeed.  A configuration that
passes the packer is guaranteed meshable -- there is no retry loop between
packing and meshing, which is the failure mode the SciSpace review's section 20
is designed around.

The margin is expressed in cells, so ``gap_cells = 2`` literally means "at least
two matrix elements across every ligament".

Anisotropic macro-boxes
-----------------------
A macro-box is a ``kx x ky x kz`` cuboid, sized from the body's axis-aligned
bounding box: each axis gets ``ceil(box_ratio * width / h)`` cells, so an
elongated body keeps the wide margin on its long axis while the short axes
shrink to the smallest admissible box.  Spheres degenerate to the historical
cubic sizing ``ceil(2 * box_ratio * R / h)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import numpy as np

from ..background.cartesian import CartesianGrid, OccupancyMap
from ..shapes.base import StarShape
from ..shapes.library import Sphere


@dataclass
class PackingSpec:
    """User-facing description of the inclusion population."""

    n_particles: int | Tuple[int, int]
    """Exact count, or an inclusive ``(lo, hi)`` range sampled per realisation."""

    radius_mean: float
    radius_std: float = 0.0
    radius_min: float | None = None
    radius_max: float | None = None

    box_ratio: float = 1.35
    """Target ratio of macro-box half-edge to inclusion bounding size, applied
    to every axis of the body's bounding box.  ``1.2-1.6`` works well: smaller
    wastes fewer elements, larger gives the transition shell more room."""

    gap_cells: int = 2
    """Minimum background cells between two macro-boxes -- i.e. the guaranteed
    number of matrix elements across the thinnest ligament."""

    wall_cells: int = 1
    """Minimum background cells between a macro-box and the domain boundary."""

    min_radius_cells: float = 1.5
    """Reject radii below this many background cells; such an inclusion cannot
    be resolved and would force a degenerate macro-box."""

    max_tries_per_particle: int = 4000
    allow_shortfall: bool = True
    """If ``False``, raise when the requested count cannot be placed."""


@dataclass
class PackedInclusion:
    """One placed inclusion and the macro-box assigned to it."""

    shape: StarShape
    box_lo: np.ndarray
    k: np.ndarray
    """Macro-box edge length in background cells, ``(kx, ky, kz)``."""
    particle_id: int

    @property
    def box_ratio(self) -> float:
        return float(np.mean(self.k)) / 2.0  # in cells; multiply by h / R for the physical ratio


@dataclass
class PackingResult:
    inclusions: List[PackedInclusion]
    requested: int
    seed: int
    stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.inclusions)


def sample_radii(
    n: int, spec: PackingSpec, h: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` radii from a truncated normal, clipped to resolvable sizes."""
    lo = spec.radius_min if spec.radius_min is not None else spec.radius_mean - 3 * spec.radius_std
    hi = spec.radius_max if spec.radius_max is not None else spec.radius_mean + 3 * spec.radius_std
    floor = spec.min_radius_cells * h
    lo = max(lo, floor, 1e-12)
    hi = max(hi, lo)
    if spec.radius_std <= 0:
        return np.full(n, float(np.clip(spec.radius_mean, lo, hi)))

    out = np.empty(n)
    filled = 0
    # rejection sampling keeps the distribution a genuine truncated normal
    # rather than a normal with a pile-up at the clip bounds
    for _ in range(200):
        draw = rng.normal(spec.radius_mean, spec.radius_std, size=max(n * 2, 32))
        draw = draw[(draw >= lo) & (draw <= hi)]
        take = min(draw.size, n - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
        if filled == n:
            return out
    out[filled:] = np.clip(spec.radius_mean, lo, hi)
    return out


def box_cells_for_radius(radius: float, h: float, box_ratio: float, k_min: int = 4) -> int:
    """Macro-box edge in background cells for a given bounding radius."""
    k = int(np.ceil(2.0 * box_ratio * radius / h))
    return max(k, k_min)


def box_cells_for_shape(shape: StarShape, h: float, box_ratio: float, k_min: int = 4) -> np.ndarray:
    """Per-axis macro-box edge in background cells, from the body's bounding box.

    Axis ``a`` gets ``ceil(box_ratio * width_a / h)`` cells, where ``width_a``
    is the body's extent along that axis.  A sphere of radius ``R`` therefore
    reduces to :func:`box_cells_for_radius` on every axis.
    """
    lo, hi = shape.bounding_box()
    width = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    k = np.ceil(box_ratio * width / h)
    return np.maximum(k, k_min).astype(np.int64)


def pack(
    grid: CartesianGrid,
    spec: PackingSpec,
    rng: np.random.Generator,
    shape_factory: Callable[[np.ndarray, float, np.random.Generator], StarShape] | None = None,
) -> PackingResult:
    """Place inclusions by grid-aware random sequential addition.

    ``shape_factory(centre, radius, rng)`` builds the body; the default makes a
    :class:`~hexaform.shapes.library.Sphere`, but passing an ellipsoid or grain
    factory produces the same packing behaviour with a different body, with the
    macro-box sized per axis from the body's bounding box.
    """
    factory = shape_factory or (lambda c, r, _rng: Sphere(c, r))
    seed_state = int(rng.bit_generator.state["state"]["state"] % (2**31))

    if isinstance(spec.n_particles, (tuple, list)):
        lo_n, hi_n = int(spec.n_particles[0]), int(spec.n_particles[1])
        requested = int(rng.integers(lo_n, hi_n + 1))
    else:
        requested = int(spec.n_particles)

    h = grid.h
    radii = sample_radii(requested, spec, h, rng)
    # Each body is created once, at a placeholder centre: its axis-aligned
    # extents do not depend on the centre, and calling the factory a second
    # time would resample random parameters (orientation, grain coefficients).
    # The centre is fixed once the macro-box is placed.
    shapes = [factory(np.zeros(3), float(r), rng) for r in radii]
    ks = np.array([box_cells_for_shape(s, h, spec.box_ratio) for s in shapes], dtype=np.int64)
    ks = ks.reshape(requested, 3)

    # Place the largest boxes first: RSA success rate is dominated by the big ones.
    order = np.argsort(-ks.sum(axis=1))
    occupancy = OccupancyMap(grid.divisions)
    div = np.array(grid.divisions, dtype=np.int64)

    inclusions: List[PackedInclusion] = []
    n_rejected = 0
    for pid, idx in enumerate(order, start=1):
        k = ks[idx]
        hi = div - k - spec.wall_cells
        lo_bound = np.full(3, spec.wall_cells, dtype=np.int64)
        if np.any(hi < lo_bound):
            n_rejected += 1
            continue

        placed = False
        for _ in range(spec.max_tries_per_particle):
            lo = np.array(
                [rng.integers(lo_bound[a], hi[a] + 1) for a in range(3)], dtype=np.int64
            )
            if occupancy.fits(lo, k, margin=spec.gap_cells):
                occupancy.place(lo, k)
                centre = grid.origin + (lo + k / 2.0) * h
                inclusions.append(
                    PackedInclusion(shapes[idx].with_center(centre), lo, k, len(inclusions) + 1)
                )
                placed = True
                break
        if not placed:
            n_rejected += 1

    if n_rejected and not spec.allow_shortfall:
        raise RuntimeError(
            f"could only place {len(inclusions)} of {requested} inclusions; "
            "reduce the count/radius or increase mesh density"
        )

    stats = _packing_stats(grid, inclusions, requested, n_rejected)
    return PackingResult(inclusions, requested, seed_state, stats)


def _packing_stats(
    grid: CartesianGrid, inclusions: Sequence[PackedInclusion], requested: int, rejected: int
) -> dict:
    domain_volume = float(np.prod(grid.extent))
    if not inclusions:
        return {
            "requested": requested,
            "placed": 0,
            "rejected": rejected,
            "volume_fraction": 0.0,
            "domain_volume": domain_volume,
        }

    centres = np.array([inc.shape.center for inc in inclusions])
    radii = np.array([inc.shape.bounding_radius() for inc in inclusions])
    vol = sum(inc.shape.volume() for inc in inclusions)

    if len(inclusions) > 1:
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        gap = d - radii[:, None] - radii[None, :]
        np.fill_diagonal(gap, np.inf)
        min_gap = float(gap.min())
    else:
        min_gap = float("inf")

    wall = np.minimum(
        (centres - grid.origin).min(axis=1), (grid.upper - centres).min(axis=1)
    ) - radii

    return {
        "requested": requested,
        "placed": len(inclusions),
        "rejected": rejected,
        "volume_fraction": vol / domain_volume,
        "inclusion_volume": vol,
        "domain_volume": domain_volume,
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std()),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "min_surface_gap": min_gap,
        "min_surface_gap_cells": min_gap / grid.h,
        "min_wall_gap": float(wall.min()),
        "min_wall_gap_cells": float(wall.min()) / grid.h,
    }
