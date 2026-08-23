"""Collective-rearrangement packing for denser inclusion populations.

Random sequential addition (:mod:`.rsa`) saturates around 10-20% inclusion
volume fraction.  Once the domain is locally crowded, rejection sampling cannot
place the remaining inclusions because it is never allowed to *move* the ones it
has already placed -- the empty space it left is fragmented and no single new
box finds a cavity.  The two research reviews both point to the opposite
strategy for high fractions:

- ScholarGPT review, step 1 of the proposed workflow: *"For high fractions, move
  to collective rearrangement/optimization rather than simple rejection
  sampling"* (Richter's Mote3D and Schneider et al. are the cited references,
  [S12, S13] in the bibliography).
- SciSpace review §14: the particulate-composite literature builds high-density
  random RVEs with Monte Carlo plus collision-driven molecular dynamics, reaching
  particle/matrix volume ratios up to ~50% (Elbana et al. 2024), and insists the
  packing module output *only* the shape list so the mesher never cares how it
  was made.

Hexaform implements that recommendation while keeping the invariant the mesher
actually needs: every accepted state keeps the inclusions' **integer macro-boxes
disjoint with a margin of whole background cells** and clear of the walls.  That
is precisely the "guaranteed meshable" condition (:mod:`.rsa`), so a denser
packing is *still* guaranteed meshable -- there remains no pack → mesh → fail →
repack loop.

Algorithm: a discrete Lubachevsky-Stillinger / force-biased rearrangement on the
macro-box lattice.

1. **Seed** with a largest-first RSA pass -- a feasible starting configuration,
   so the collective result can never place fewer inclusions than that seed.
2. **Fluidise** -- each step performs a batch of moves on the *placed* boxes.
   With probability ``kinetic_fraction`` a move is kinetic (a Lubachevsky-
   Stillinger random walk of the box's lattice position); otherwise it is
   force-biased, pushing the box toward whichever of its six faces has the most
   empty space.  A move is accepted only if it stays inside the domain and
   preserves the macro-box gap, so the configuration is feasible at every step.
   The kinetic share lets the system escape the local minima that pure
   force-directed packing falls into.
3. **Seat** the largest still-unplaced box into a cavity the fluidising opened.

Because the loop only ever *adds* inclusions (moves keep every box placed), the
final configuration contains the seed as a subset and typically many more
inclusions.  Given a single ``rng`` the whole run is reproducible.

This module is deliberately separate from :mod:`.rsa` and shares the same
``pack(grid, spec, rng, shape_factory)`` contract, so the two are drop-in
substitutes behind the pipeline dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..background.cartesian import CartesianGrid
from ..shapes.base import StarShape
from ..shapes.library import Sphere
from .rsa import (
    PackedInclusion,
    PackingResult,
    PackingSpec,
    _packing_stats,
    box_cells_for_shape,
    sample_radii,
)


@dataclass
class CollectiveOptions:
    """Knobs for the collective-rearrangement packer.

    ``kinetic_fraction`` most directly affects *which* feasible configuration is
    found (more kinetic = more exploration, less force-directed optimisation).
    The remaining fields bound the search effort and therefore the runtime, not
    the correctness: a longer search can only place more inclusions, never fewer,
    because the seed is a valid lower bound and moves never un-place a box.
    """

    max_steps: int = 2000
    """Upper bound on fluidise + seat iterations before a shortfall is accepted."""

    seed_tries: int = 400
    """Position tries per box during the initial (largest-first) RSA seed."""

    moves_fraction: float = 0.6
    """Fraction of the placed boxes that move per fluidise pass.  The number of
    moves per step is ``max(1, round(moves_fraction * placed))``, so this scales
    with the population.  Higher fluidises faster and costs more."""

    kinetic_fraction: float = 0.5
    """Probability a collective move is a kinetic (Lubachevsky-Stillinger)
    random walk rather than a force-biased push toward empty space.  ``1.0`` is
    pure kinetic search, ``0.0`` pure force-directed; ``0.3-0.7`` works well for
    a blend of exploration and optimisation."""

    jump: int = 3
    """Maximum integer displacement (in background cells) of a single collective
    move.  Small jumps relax finely, large jumps search coarsely."""

    seat_tries: int = 200
    """Position tries when seating a still-unplaced box into an opened cavity."""

    stall_limit: int = 200
    """Stop early after this many consecutive steps that seat no new box.  Bounds
    the search when the requested fraction is unattainable, so a shortfall is
    reported promptly rather than after the full ``max_steps`` budget."""


def _clamp_fraction(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class _Lattice:
    """Integer macro-box occupancy supporting add/remove and exact RSA validity.

    Mirrors :class:`~hexaform.background.cartesian.OccupancyMap` semantics -- a
    box is admissible when it is clear of the walls by ``wall`` cells and its
    ``gap``-inflated region is free of every other box -- but also supports
    *removing* a box so the collective packer can relocate it.  Owner tags let a
    box's own old cells be ignored when validating a move.
    """

    def __init__(self, divisions: Sequence[int], wall: int, gap: int) -> None:
        self.div = np.asarray(divisions, dtype=np.int64).reshape(3)
        self.wall = int(wall)
        self.gap = int(gap)
        self.occ = np.zeros(self.div, dtype=bool)
        self.owner = np.full(self.div, -1, dtype=np.int64)

    @staticmethod
    def _shape(k: np.ndarray | int) -> np.ndarray:
        return np.broadcast_to(np.asarray(k, dtype=np.int64), (3,))

    def free(self, lo: np.ndarray, k: np.ndarray | int, skip_owner: int | None = None) -> bool:
        """True if the ``gap``-inflated region of a box at ``lo`` is clear of every
        other box (optionally ignoring ``skip_owner``'s own cells)."""
        lo = np.asarray(lo, dtype=np.int64).reshape(3)
        k = self._shape(k)
        a = np.maximum(lo - self.gap, 0)
        b = np.minimum(lo + k + self.gap, self.div)
        region = self.occ[a[0] : b[0], a[1] : b[1], a[2] : b[2]]
        if skip_owner is not None:
            region = region & (self.owner[a[0] : b[0], a[1] : b[1], a[2] : b[2]] != skip_owner)
        return not region.any()

    def set(self, lo: np.ndarray, k: np.ndarray | int, owner: int) -> None:
        lo = np.asarray(lo, dtype=np.int64).reshape(3)
        k = self._shape(k)
        sl = (
            slice(int(lo[0]), int(lo[0] + k[0])),
            slice(int(lo[1]), int(lo[1] + k[1])),
            slice(int(lo[2]), int(lo[2] + k[2])),
        )
        self.occ[sl] = True
        self.owner[sl] = owner

    def clear(self, lo: np.ndarray, k: np.ndarray | int) -> None:
        """Remove a box's cells.  Boxes are disjoint, so this region is exactly
        that box's footprint and clearing it cannot affect any other box."""
        lo = np.asarray(lo, dtype=np.int64).reshape(3)
        k = self._shape(k)
        sl = (
            slice(int(lo[0]), int(lo[0] + k[0])),
            slice(int(lo[1]), int(lo[1] + k[1])),
            slice(int(lo[2]), int(lo[2] + k[2])),
        )
        self.occ[sl] = False
        self.owner[sl] = -1


def _admissible_range(div: np.ndarray, k: np.ndarray, wall: int):
    """The inclusive ``(lo_bound, hi_bound)`` index range a box of size ``k`` may
    occupy while staying ``wall`` cells from every wall, or ``None`` if it cannot
    fit at all."""
    lo_bound = np.full(3, wall, dtype=np.int64)
    hi_bound = div - k - wall
    if np.any(hi_bound < lo_bound):
        return None
    return lo_bound, hi_bound


def _try_place(
    lat: _Lattice,
    k: np.ndarray,
    rng: np.random.Generator,
    tries: int,
) -> Optional[np.ndarray]:
    """Try up to ``tries`` uniform positions for a *new* box of size ``k``.

    Returns the accepted lower index (callers record it in the lattice) or
    ``None``.  The box is not yet in the lattice, so no owner needs to be
    skipped.
    """
    bounds = _admissible_range(lat.div, k, lat.wall)
    if bounds is None:
        return None
    lo_bound, hi_bound = bounds
    for _ in range(tries):
        lo = np.array(
            [rng.integers(lo_bound[a], hi_bound[a] + 1) for a in range(3)], dtype=np.int64
        )
        if lat.free(lo, k):
            return lo
    return None


def _kinetic_move(
    lo: np.ndarray,
    k: np.ndarray,
    div: np.ndarray,
    wall: int,
    jump: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A Lubachevsky-Stillinger step: perturb the box's lattice position by a
    small random displacement, clipped to the admissible range."""
    bounds = _admissible_range(div, k, wall) or (
        np.zeros(3, dtype=np.int64),
        div - np.broadcast_to(np.asarray(k, dtype=np.int64), (3,)) - wall,
    )
    lo_bound, hi_bound = bounds
    cand = np.asarray(lo, dtype=np.int64).copy()
    cand += rng.integers(-jump, jump + 1, size=3).astype(np.int64)
    return np.clip(cand, lo_bound, hi_bound)


def _force_move(
    lo: np.ndarray,
    k: np.ndarray,
    placed_pos: np.ndarray,
    placed_ids: np.ndarray,
    ks: np.ndarray,
    div: np.ndarray,
    wall: int,
    jump: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A force-biased step: push the box toward whichever of its six faces has the
    largest free gap to the nearest neighbouring box or wall.

    ``placed_pos``/``placed_ids`` describe the currently placed boxes.  This is a
    *proposal* only -- it need not be exact, because every candidate is
    re-validated against the lattice before it is accepted.
    """
    lo = np.asarray(lo, dtype=np.int64).reshape(3)
    k = np.broadcast_to(np.asarray(k, dtype=np.int64), (3,))
    best_axis, best_side, best_gap = 0, "low", -1
    for a in range(3):
        o_lo = placed_pos[:, a]
        neigh_hi = o_lo + ks[placed_ids, a]  # each neighbour's high face on axis a
        below = neigh_hi[neigh_hi <= lo[a]]
        low_face = int(below.max()) if below.size else int(wall)
        gap_low = int(lo[a]) - low_face
        above = o_lo[o_lo >= lo[a] + k[a]]
        high_face = int(above.min()) if above.size else int(div[a] - wall)
        gap_high = high_face - int(lo[a] + k[a])
        if gap_low > best_gap:
            best_gap, best_axis, best_side = gap_low, a, "low"
        if gap_high > best_gap:
            best_gap, best_axis, best_side = gap_high, a, "high"

    step = int(rng.integers(1, jump + 1))
    cand = lo.copy()
    bounds = _admissible_range(div, k, wall) or (
        np.zeros(3, dtype=np.int64),
        div - k - wall,
    )
    lo_bound, hi_bound = bounds
    if best_side == "high":
        cand[best_axis] = min(int(cand[best_axis] + step), int(hi_bound[best_axis]))
    else:
        cand[best_axis] = max(int(cand[best_axis] - step), int(lo_bound[best_axis]))
    return cand


def _fluidise(
    lat: _Lattice,
    pos: List[Optional[np.ndarray]],
    ks: np.ndarray,
    placed: List[int],
    opt: CollectiveOptions,
    rng: np.random.Generator,
) -> int:
    """Perform one fluidise pass over the placed boxes; return the number of
    accepted (state-changing) moves.

    Moves are a blend of kinetic (Lubachevsky-Stillinger) and force-biased steps.
    A move is applied only if it keeps the box inside the domain and preserves the
    macro-box gap, so feasibility is maintained at every step.
    """
    if not placed:
        return 0
    m = max(1, int(round(opt.moves_fraction * len(placed))))
    m = min(m, len(placed))

    placed_ids = np.asarray(placed, dtype=np.int64)
    placed_pos = np.asarray([pos[j] for j in placed], dtype=np.int64)
    div, wall, jump = lat.div, lat.wall, max(1, int(opt.jump))
    kinetic = _clamp_fraction(opt.kinetic_fraction)
    pos_row = {int(j): r for r, j in enumerate(placed)}
    accepted = 0
    for _ in range(m):
        j = int(rng.integers(len(placed)))
        i = placed[j]
        lo, k = pos[i], ks[i]
        if rng.random() < kinetic:
            cand = _kinetic_move(lo, k, div, wall, jump, rng)
        else:
            cand = _force_move(lo, k, placed_pos, placed_ids, ks, div, wall, jump, rng)
        if np.array_equal(cand, lo):
            continue
        if lat.free(cand, k, skip_owner=i):
            lat.clear(lo, k)
            lat.set(cand, k, owner=i)
            pos[i] = cand
            # keep the cached neighbour view consistent for later moves this pass
            placed_pos[pos_row[i]] = cand
            accepted += 1
    return accepted


def pack(
    grid: CartesianGrid,
    spec: PackingSpec,
    rng: np.random.Generator,
    shape_factory: Callable[[np.ndarray, float, np.random.Generator], StarShape] | None = None,
    options: CollectiveOptions | None = None,
) -> PackingResult:
    """Place inclusions by collective rearrangement (drop-in for :func:`.rsa.pack`).

    Shares the exact contract of :func:`.rsa.pack` -- ``(grid, spec, rng,
    shape_factory)`` -- and returns the same :class:`PackingResult`, so it is a
    drop-in substitute behind the pipeline dispatcher.  See the module
    docstring for the algorithm and :class:`CollectiveOptions` for the search
    knobs.
    """
    factory = shape_factory or (lambda c, r, _rng: Sphere(c, r))
    opt = options or CollectiveOptions()
    seed_state = int(rng.bit_generator.state["state"]["state"] % (2**31))

    # ---- requested count, radii, shapes, box sizes (identical to rsa) -------
    if isinstance(spec.n_particles, (tuple, list)):
        lo_n, hi_n = int(spec.n_particles[0]), int(spec.n_particles[1])
        requested = int(rng.integers(lo_n, hi_n + 1))
    else:
        requested = int(spec.n_particles)

    h = grid.h
    radii = sample_radii(requested, spec, h, rng)
    # Each body is created once, at a placeholder centre: its axis-aligned extent
    # does not depend on the centre, so the centre is fixed only once its box is
    # placed (identical reasoning to rsa.pack).
    shapes = [factory(np.zeros(3), float(r), rng) for r in radii]
    if requested == 0:
        ks = np.zeros((0, 3), dtype=np.int64)
    else:
        ks = np.array(
            [box_cells_for_shape(s, h, spec.box_ratio) for s in shapes], dtype=np.int64
        ).reshape(requested, 3)
    n = requested

    div = np.asarray(grid.divisions, dtype=np.int64)
    # A box that cannot fit the domain at its wall margin is unplaceable by any
    # method; track it separately so it contributes to the shortfall.
    if n:
        fits_domain = np.all(ks + 2 * spec.wall_cells <= div, axis=1)
    else:
        fits_domain = np.zeros(0, dtype=bool)

    lat = _Lattice(div, spec.wall_cells, spec.gap_cells)
    pos: List[Optional[np.ndarray]] = [None] * n
    placed: List[int] = []
    pending: List[int] = []

    order = np.argsort(-ks.sum(axis=1)) if n else np.zeros(0, dtype=np.int64)
    admissible = [int(i) for i in order if n and fits_domain[i]]
    inadmissible = [int(i) for i in order if n and not fits_domain[i]]

    # ---- Phase 1: largest-first RSA seed (feasible lower bound) -------------
    for idx in admissible:
        lo = _try_place(lat, ks[idx], rng, opt.seed_tries)
        if lo is not None:
            lat.set(lo, ks[idx], owner=idx)
            pos[idx] = lo
            placed.append(idx)
        else:
            pending.append(idx)
    pending.extend(inadmissible)

    # ---- Phase 2: collective rearrangement ----------------------------------
    stall = 0
    steps_done = 0
    for _ in range(opt.max_steps):
        if not pending:
            break
        steps_done += 1
        _fluidise(lat, pos, ks, placed, opt, rng)
        # Seat the largest still-unplaced, domain-fitting box (volume priority),
        # but bound the total samples this iteration to ~``seat_tries`` so each
        # step does a fixed amount of work.  We revisit the unseated in later
        # iterations as fluidising continues to open cavities.
        candidates = sorted(
            (i for i in pending if fits_domain[i]),
            key=lambda i: int(ks[i].sum()),
            reverse=True,
        )
        seated = False
        if candidates:
            per = max(1, opt.seat_tries // len(candidates))
            for idx in candidates:
                lo = _try_place(lat, ks[idx], rng, per)
                if lo is not None:
                    lat.set(lo, ks[idx], owner=idx)
                    pos[idx] = lo
                    placed.append(idx)
                    pending.remove(idx)
                    stall = 0
                    seated = True
                    break
        if not seated:
            stall += 1
            if stall >= opt.stall_limit:
                break

    # ---- build the result ----------------------------------------------------
    inclusions: List[PackedInclusion] = []
    for new_id, idx in enumerate(placed, start=1):
        centre = grid.origin + (np.asarray(pos[idx], dtype=float) + ks[idx] / 2.0) * h
        inclusions.append(
            PackedInclusion(shapes[idx].with_center(centre), pos[idx], ks[idx], new_id)
        )

    n_rejected = n - len(inclusions)
    if n_rejected and not spec.allow_shortfall:
        raise RuntimeError(
            f"could only place {len(inclusions)} of {requested} inclusions; "
            "reduce the count/radius, increase the domain, or loosen gap_cells"
        )

    stats = _packing_stats(grid, inclusions, requested, n_rejected)
    stats["packer"] = "collective"
    stats["collective_steps"] = steps_done
    stats["collective_kinetic_fraction"] = opt.kinetic_fraction
    return PackingResult(inclusions, requested, seed_state, stats)


__all__ = ["CollectiveOptions", "pack"]
