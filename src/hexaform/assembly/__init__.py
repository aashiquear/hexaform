"""Stitching templates into the background grid and creating Exodus metadata."""

from .builder import (
    FACE_NAMES,
    INCLUSION_BLOCK_BASE,
    MATRIX_BLOCK_ID,
    AssemblyOptions,
    assemble,
)
from .notch import (
    NotchType,
    NotchGeometry,
    NotchOptions,
    insert_notch,
    create_edge_notch,
    create_center_notch,
    create_chevron_notch,
)

__all__ = [
    "assemble",
    "AssemblyOptions",
    "MATRIX_BLOCK_ID",
    "INCLUSION_BLOCK_BASE",
    "FACE_NAMES",
    "NotchType",
    "NotchGeometry",
    "NotchOptions",
    "insert_notch",
    "create_edge_notch",
    "create_center_notch",
    "create_chevron_notch",
]
