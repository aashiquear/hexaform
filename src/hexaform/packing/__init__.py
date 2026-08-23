"""Inclusion packing, kept strictly separate from meshing.

Two interchangeable implementations share the same ``pack(grid, spec, rng,
shape_factory)`` contract and are drop-in substitutes behind the pipeline
dispatcher:

- :mod:`.rsa` -- grid-aware random sequential addition.  Saturates around 10-20%
  inclusion volume fraction but is the fastest and always feasible.
- :mod:`:`.collective` -- collective rearrangement (Lubachevsky-Stillinger /
  force-biased) that places inclusions more densely by letting already-placed
  boxes move.  Same "guaranteed meshable" invariant.
"""

from .rsa import (
    PackedInclusion,
    PackingResult,
    PackingSpec,
    box_cells_for_radius,
    box_cells_for_shape,
    pack,
    sample_radii,
)
from .collective import CollectiveOptions, pack as collective_pack

__all__ = [
    "PackingSpec",
    "PackedInclusion",
    "PackingResult",
    "pack",
    "collective_pack",
    "CollectiveOptions",
    "sample_radii",
    "box_cells_for_radius",
    "box_cells_for_shape",
]
