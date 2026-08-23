"""Background matrix discretisation.

The uniform :class:`~hexaform.background.cartesian.CartesianGrid` and the
adaptive :class:`~hexaform.background.octree.OctreeGrid` backends are drop-in
replacements: both implement the same *emit hexes, expose node ids on each
carved macro-box face* contract.  The Cartesian grid is the default; the
octree is Backend B and the one to use when particles approach closely or
where the matrix must stay coarse far from interfaces.
"""

from .cartesian import CartesianGrid, OccupancyMap, free_cells
from .octree import (
    OctreeCell,
    OctreeGrid,
    OctreeMesh,
    SizeField,
    decode_location,
    encode_location,
    sphere_material_of,
    sphere_size_field,
)

__all__ = [
    "CartesianGrid",
    "OccupancyMap",
    "free_cells",
    "OctreeGrid",
    "OctreeMesh",
    "OctreeCell",
    "SizeField",
    "encode_location",
    "decode_location",
    "sphere_size_field",
    "sphere_material_of",
]
