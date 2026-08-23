"""Hexahedral element conventions shared by topology, quality and I/O.

Connectivity is stored in Exodus order.  HEX20 adds the twelve edge midpoints
(nodes 9--20 in one-based Exodus notation).  HEX27 then stores the volume
centre as node 21, followed by the six face centres in ``z-``, ``z+``, ``x-``,
``x+``, ``y-``, ``y+`` order.  That last ordering is intentionally different
from VTK's triquadratic-hexahedron convention.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

ELEMENT_TYPES: Dict[str, int] = {"HEX8": 8, "HEX20": 20, "HEX27": 27}

# Corner pairs for Exodus nodes 9--20.
HEX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (4, 5), (5, 6), (6, 7), (7, 4),
)

# Exodus local side definitions.  The first four entries are always corners;
# HEX20 appends four edge midpoints and HEX27 appends the face centre.
HEX8_SIDES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
    (0, 3, 2, 1),
    (4, 5, 6, 7),
)
HEX20_SIDES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 5, 4, 8, 13, 16, 12),
    (1, 2, 6, 5, 9, 14, 17, 13),
    (2, 3, 7, 6, 10, 15, 18, 14),
    (3, 0, 4, 7, 11, 12, 19, 15),
    (0, 3, 2, 1, 11, 10, 9, 8),
    (4, 5, 6, 7, 16, 17, 18, 19),
)
HEX27_SIDES: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 5, 4, 8, 13, 16, 12, 25),
    (1, 2, 6, 5, 9, 14, 17, 13, 24),
    (2, 3, 7, 6, 10, 15, 18, 14, 26),
    (3, 0, 4, 7, 11, 12, 19, 15, 23),
    (0, 3, 2, 1, 11, 10, 9, 8, 21),
    (4, 5, 6, 7, 16, 17, 18, 19, 22),
)
SIDE_NODES = {"HEX8": HEX8_SIDES, "HEX20": HEX20_SIDES, "HEX27": HEX27_SIDES}

# Face-centre connectivity slots 21--26 (zero-based slots 20 is the cell
# centre, then 21--26) correspond to these one-based Exodus side IDs.
HEX27_FACE_CENTER_SIDES: Tuple[int, ...] = (5, 6, 4, 2, 1, 3)

_CORNER_COORDS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)
_EDGE_COORDS = np.array([(_CORNER_COORDS[a] + _CORNER_COORDS[b]) / 2 for a, b in HEX_EDGES])
HEX8_REFERENCE_COORDS = _CORNER_COORDS
HEX20_REFERENCE_COORDS = np.concatenate([_CORNER_COORDS, _EDGE_COORDS], axis=0)
HEX27_REFERENCE_COORDS = np.concatenate(
    [
        HEX20_REFERENCE_COORDS,
        np.array(
            [
                [0.0, 0.0, 0.0],   # Exodus node 21: volume centre
                [0.0, 0.0, -1.0],  # node 22: side 5 (z-)
                [0.0, 0.0, 1.0],   # node 23: side 6 (z+)
                [-1.0, 0.0, 0.0],  # node 24: side 4 (x-)
                [1.0, 0.0, 0.0],   # node 25: side 2 (x+)
                [0.0, -1.0, 0.0],  # node 26: side 1 (y-)
                [0.0, 1.0, 0.0],   # node 27: side 3 (y+)
            ]
        ),
    ],
    axis=0,
)
REFERENCE_COORDS = {
    "HEX8": HEX8_REFERENCE_COORDS,
    "HEX20": HEX20_REFERENCE_COORDS,
    "HEX27": HEX27_REFERENCE_COORDS,
}

# Permutation used when reflecting zeta to repair an inverted element.
ORIENTATION_FLIP = {
    "HEX8": np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int64),
    "HEX20": np.array(
        [4, 5, 6, 7, 0, 1, 2, 3, 16, 17, 18, 19, 12, 13, 14, 15, 8, 9, 10, 11],
        dtype=np.int64,
    ),
    "HEX27": np.array(
        [
            4, 5, 6, 7, 0, 1, 2, 3,
            16, 17, 18, 19, 12, 13, 14, 15, 8, 9, 10, 11,
            20, 22, 21, 23, 24, 25, 26,
        ],
        dtype=np.int64,
    ),
}

# Legacy VTK cell ids and Exodus-to-VTK local connectivity permutations.
VTK_CELL_TYPES = {"HEX8": 12, "HEX20": 25, "HEX27": 29}
EXODUS_TO_VTK = {
    "HEX8": np.arange(8, dtype=np.int64),
    "HEX20": np.array(
        [*range(12), 16, 17, 18, 19, 12, 13, 14, 15], dtype=np.int64
    ),
    "HEX27": np.array(
        [
            *range(12), 16, 17, 18, 19, 12, 13, 14, 15,
            23, 24, 25, 26, 21, 22, 20,
        ],
        dtype=np.int64,
    ),
}


def normalize_element_type(value: str) -> str:
    """Return a canonical supported element name."""
    name = str(value).strip().upper()
    if name not in ELEMENT_TYPES:
        expected = ", ".join(ELEMENT_TYPES)
        raise ValueError(f"unsupported element type {value!r}; expected one of {expected}")
    return name


def element_type_from_width(width: int) -> str:
    """Infer the homogeneous element type from connectivity width."""
    for name, count in ELEMENT_TYPES.items():
        if int(width) == count:
            return name
    raise ValueError(f"unsupported hexahedral connectivity width {width}; expected 8, 20 or 27")


def side_nodes(element_type: str) -> Tuple[Tuple[int, ...], ...]:
    return SIDE_NODES[normalize_element_type(element_type)]
