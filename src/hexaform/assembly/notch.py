"""Notch and pre-crack geometry for phase-field fracture simulations.

Phase-field fracture simulations typically require an initial crack or notch to
seed crack initiation. This module provides two complementary approaches:

1. **Geometric notch**: A thin slit carved from the background mesh by marking
   elements as void/inactive. This physically removes material.

2. **Phase-field variable initialization**: An element field that initializes the
   phase-field damage variable to 1.0 (fully damaged) in a prescribed region,
   while keeping the mesh intact. This is numerically equivalent but preserves
   the mesh topology.

Both approaches support common notch geometries:
- Edge notch: a straight crack from one domain face
- Center notch: a crack in the domain interior
- Chevron notch: V-shaped notch for mode-I fracture tests
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.mesh import HexMesh


class NotchType(Enum):
    """Type of notch geometry."""

    EDGE = "edge"
    """Straight notch starting from a domain face."""

    CENTER = "center"
    """Straight notch centered in the domain."""

    CHEVRON = "chevron"
    """V-shaped notch for mode-I fracture tests."""


@dataclass
class NotchGeometry:
    """Geometric definition of a notch or pre-crack.

    Attributes:
        notch_type: Type of notch geometry.
        origin: Starting point of the notch in domain coordinates.
        direction: Primary direction of notch propagation (unit vector).
        length: Length of the notch along the primary direction.
        width: Width/thickness of the notch (for geometric carving).
        depth: Depth of the notch perpendicular to the propagation plane.
        plane_normal: Normal vector defining the notch plane (for 3D orientation).
        chevron_angle: Angle for chevron notch (in degrees, default 45°).
    """

    notch_type: NotchType = NotchType.EDGE
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    length: float = 0.5
    width: float = 0.0
    depth: float = 1.0
    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    chevron_angle: float = 45.0

    def __post_init__(self):
        """Validate and normalize the notch geometry."""
        # Normalize direction vector
        dir_vec = np.array(self.direction, dtype=np.float64)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-12:
            raise ValueError("Notch direction vector cannot be zero")
        self.direction = tuple(dir_vec / norm)

        # Normalize plane normal
        plane_vec = np.array(self.plane_normal, dtype=np.float64)
        norm = np.linalg.norm(plane_vec)
        if norm < 1e-12:
            raise ValueError("Notch plane normal cannot be zero")
        self.plane_normal = tuple(plane_vec / norm)

        # Validate parameters
        if self.length <= 0:
            raise ValueError("Notch length must be positive")
        if self.width < 0:
            raise ValueError("Notch width cannot be negative")
        if self.depth <= 0:
            raise ValueError("Notch depth must be positive")
        if not (0 < self.chevron_angle < 180):
            raise ValueError("Chevron angle must be between 0 and 180 degrees")


@dataclass
class NotchOptions:
    """Options controlling notch insertion behavior.

    Attributes:
        method: Method for inserting notch ('geometric' or 'field').
        geometric_width: Target width for geometric notches (in length units).
        phase_field_value: Phase-field damage value to assign (default 1.0).
        variable_name: Name for the phase-field element variable.
        mark_elements: Whether to mark notch elements in a separate set.
        element_set_name: Name for the notch element set.
    """

    method: str = "field"
    """Method: 'geometric' removes elements, 'field' initializes damage variable."""

    geometric_width: float = 0.0
    """Physical width for geometric notches. If 0, marks zero-thickness plane."""

    phase_field_value: float = 1.0
    """Phase-field damage variable value (1.0 = fully damaged)."""

    variable_name: str = "phase_field"
    """Name of the phase-field element variable."""

    mark_elements: bool = True
    """Whether to create an element set for notch elements."""

    element_set_name: str = "notch"
    """Name for the notch element set."""


def insert_notch(
    mesh: HexMesh,
    notch_geom: NotchGeometry,
    options: NotchOptions | None = None,
    domain_extent: tuple[float, float, float] | None = None,
    domain_origin: tuple[float, float, float] | None = None,
) -> HexMesh:
    """Insert a notch or pre-crack into the mesh.

    Args:
        mesh: The input mesh (must have element centroids available or computable).
        notch_geom: Geometric definition of the notch.
        options: Notch insertion options.
        domain_extent: Domain extent [Lx, Ly, Lz]. Required for default origin.
        domain_origin: Domain origin [x0, y0, z0]. Default (0, 0, 0).

    Returns:
        Modified mesh with notch inserted (as element field or geometric removal).

    Raises:
        ValueError: If the notch parameters are invalid or no elements match.
    """
    opt = options or NotchOptions()
    origin = domain_origin or (0.0, 0.0, 0.0)
    extent = domain_extent or (1.0, 1.0, 1.0)

    # Compute element centroids for notch detection (don't store permanently)
    if "centroid" in mesh.element_fields:
        centroids = mesh.element_fields["centroid"]
        centroids_computed = False
    else:
        centroids = _compute_element_centroids(mesh)
        centroids_computed = True

    # Find elements in the notch region
    notch_mask = _find_notch_elements(
        centroids,
        notch_geom,
        origin,
        extent,
        opt.geometric_width if opt.method == "geometric" else 0.0,
    )

    # Clean up temporary centroid field if we computed it
    if centroids_computed and "centroid" in mesh.element_fields:
        del mesh.element_fields["centroid"]

    if not np.any(notch_mask):
        raise ValueError(
            f"No elements found in notch region. "
            f"Type: {notch_geom.notch_type.value}, "
            f"origin: {notch_geom.origin}, "
            f"length: {notch_geom.length}"
        )

    if opt.method == "geometric":
        # For geometric method, mark elements as void (particle_id = 0 or special value)
        # We use a negative particle_id to indicate void elements
        if "particle_id" not in mesh.element_fields:
            mesh.element_fields["particle_id"] = np.full(
                mesh.n_elements, -1, dtype=np.int64
            )
        mesh.element_fields["particle_id"][notch_mask] = 0  # 0 indicates void

        # Mark notch elements for later retrieval (store as metadata, not element field)
        if opt.mark_elements:
            notch_elem_ids = np.where(notch_mask)[0]
            mesh.metadata["notch_element_ids"] = notch_elem_ids

    else:
        # Field method: initialize phase-field variable
        phase_field = np.zeros(mesh.n_elements, dtype=np.float64)
        phase_field[notch_mask] = opt.phase_field_value
        mesh.element_fields[opt.variable_name] = phase_field

        # Mark notch elements for later retrieval (store as metadata, not element field)
        if opt.mark_elements:
            notch_elem_ids = np.where(notch_mask)[0]
            mesh.metadata["notch_element_ids"] = notch_elem_ids

    # Update metadata
    mesh.metadata["notch_type"] = notch_geom.notch_type.value
    mesh.metadata["notch_origin"] = list(notch_geom.origin)
    mesh.metadata["notch_direction"] = list(notch_geom.direction)
    mesh.metadata["notch_length"] = notch_geom.length
    mesh.metadata["notch_method"] = opt.method
    mesh.metadata["n_notch_elements"] = int(np.sum(notch_mask))

    return mesh


def _compute_element_centroids(mesh: HexMesh) -> np.ndarray:
    """Compute centroids of all hexahedral elements.

    Args:
        mesh: Input mesh with connectivity.

    Returns:
        Array of shape (n_elements, 3) with centroid coordinates.
    """
    # For HEX8/HEX20/HEX27, compute centroid as average of corner nodes
    # Corner nodes are always the first 8 nodes in the connectivity
    corners = mesh.cells[:, :8]  # (n_elements, 8)
    corner_coords = mesh.points[corners]  # (n_elements, 8, 3)
    centroids = corner_coords.mean(axis=1)  # (n_elements, 3)
    return centroids


def _find_notch_elements(
    centroids: np.ndarray,
    notch_geom: NotchGeometry,
    domain_origin: tuple[float, float, float],
    domain_extent: tuple[float, float, float],
    geometric_width: float = 0.0,
) -> np.ndarray:
    """Find elements whose centroids lie within the notch region.

    Args:
        centroids: Element centroids (n_elements, 3).
        notch_geom: Notch geometry definition.
        domain_origin: Domain origin coordinates.
        domain_extent: Domain extent (Lx, Ly, Lz).
        geometric_width: Width for geometric notch (0 for field method).

    Returns:
        Boolean mask of elements in the notch region.
    """
    origin = np.array(notch_geom.origin, dtype=np.float64)
    direction = np.array(notch_geom.direction, dtype=np.float64)
    plane_normal = np.array(notch_geom.plane_normal, dtype=np.float64)
    length = notch_geom.length
    depth = notch_geom.depth

    # Compute orthogonal direction (in-plane, perpendicular to propagation)
    in_plane_orth = np.cross(plane_normal, direction)
    if np.linalg.norm(in_plane_orth) < 1e-12:
        # direction and plane_normal are parallel, choose arbitrary orthogonal
        in_plane_orth = np.array([1, 0, 0], dtype=np.float64)
        if abs(np.dot(in_plane_orth, direction)) > 0.9:
            in_plane_orth = np.array([0, 1, 0], dtype=np.float64)
    in_plane_orth = in_plane_orth / np.linalg.norm(in_plane_orth)

    # For each centroid, compute coordinates in notch frame
    # relative_to_origin: vector from notch origin to centroid
    relative_to_origin = centroids - np.array(domain_origin) - origin

    # Projection along notch direction
    proj_direction = np.dot(relative_to_origin, direction)

    # Projection along in-plane orthogonal direction
    proj_orth = np.dot(relative_to_origin, in_plane_orth)

    # Projection along plane normal (thickness direction)
    proj_normal = np.dot(relative_to_origin, plane_normal)

    if notch_geom.notch_type == NotchType.EDGE:
        # Edge notch: extends from origin along direction for 'length'
        # Width is controlled by geometric_width or is zero-thickness
        if geometric_width > 0:
            # Finite width notch
            in_notch = (
                (proj_direction >= 0)
                & (proj_direction <= length)
                & (np.abs(proj_orth) <= geometric_width / 2)
                & (np.abs(proj_normal) <= depth / 2)
            )
        else:
            # Zero-thickness plane notch (field method)
            # Mark elements whose centroid projects onto the notch line/plane
            in_notch = (
                (proj_direction >= 0)
                & (proj_direction <= length)
                & (np.abs(proj_normal) <= depth / 2)
            )

    elif notch_geom.notch_type == NotchType.CENTER:
        # Center notch: extends from origin in both directions
        half_length = length / 2
        if geometric_width > 0:
            in_notch = (
                (proj_direction >= -half_length)
                & (proj_direction <= half_length)
                & (np.abs(proj_orth) <= geometric_width / 2)
                & (np.abs(proj_normal) <= depth / 2)
            )
        else:
            in_notch = (
                (proj_direction >= -half_length)
                & (proj_direction <= half_length)
                & (np.abs(proj_normal) <= depth / 2)
            )

    elif notch_geom.notch_type == NotchType.CHEVRON:
        # Chevron notch: V-shaped with two arms
        # Each arm extends at +/- chevron_angle from the primary direction
        angle_rad = np.deg2rad(notch_geom.chevron_angle)

        # For chevron, we check if point is within either arm
        # Arm 1: direction rotated by +angle
        # Arm 2: direction rotated by -angle

        # Simplified: check if point is within the V region
        # Project onto the plane defined by direction and in_plane_orth
        x_proj = proj_direction  # along primary direction
        y_proj = proj_orth  # perpendicular in-plane

        # Chevron opens in the direction perpendicular to propagation
        # The V opens with angle 2*chevron_angle
        tan_angle = np.tan(angle_rad)

        # Point is in chevron if |y| <= |x| * tan(angle) and x is in [0, length]
        # and within depth bounds
        in_chevron = (
            (x_proj >= 0)
            & (x_proj <= length)
            & (np.abs(y_proj) <= np.abs(x_proj) * tan_angle + 1e-12)
            & (np.abs(proj_normal) <= depth / 2)
        )

        in_notch = in_chevron
    else:
        raise ValueError(f"Unknown notch type: {notch_geom.notch_type}")

    return in_notch


def create_edge_notch(
    length: float = 0.5,
    origin: tuple[float, float, float] | None = None,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    depth: float = 1.0,
) -> NotchGeometry:
    """Convenience function to create an edge notch.

    Args:
        length: Notch length.
        origin: Starting point. Defaults to (0, 0.5, 0.5) for a mid-height edge.
        direction: Propagation direction.
        depth: Notch depth (through-thickness for 3D).

    Returns:
        NotchGeometry configured for an edge notch.
    """
    if origin is None:
        origin = (0.0, 0.5, 0.5)  # Default: start from mid-height of left face
    return NotchGeometry(
        notch_type=NotchType.EDGE,
        origin=origin,
        direction=direction,
        length=length,
        depth=depth,
    )


def create_center_notch(
    length: float = 0.5,
    origin: tuple[float, float, float] | None = None,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    depth: float = 1.0,
) -> NotchGeometry:
    """Convenience function to create a center notch.

    Args:
        length: Total notch length.
        origin: Center point. Defaults to domain center (0.5, 0.5, 0.5).
        direction: Propagation direction.
        depth: Notch depth.

    Returns:
        NotchGeometry configured for a center notch.
    """
    if origin is None:
        origin = (0.5, 0.5, 0.5)  # Default: domain center
    return NotchGeometry(
        notch_type=NotchType.CENTER,
        origin=origin,
        direction=direction,
        length=length,
        depth=depth,
    )


def create_chevron_notch(
    length: float = 0.5,
    origin: tuple[float, float, float] | None = None,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    depth: float = 1.0,
    angle: float = 45.0,
) -> NotchGeometry:
    """Convenience function to create a chevron notch.

    Args:
        length: Notch length along primary direction.
        origin: Vertex point. Defaults to (0, 0.5, 0.5).
        direction: Primary propagation direction (bisector of V).
        depth: Notch depth.
        angle: Chevron half-angle in degrees.

    Returns:
        NotchGeometry configured for a chevron notch.
    """
    if origin is None:
        origin = (0.0, 0.5, 0.5)
    return NotchGeometry(
        notch_type=NotchType.CHEVRON,
        origin=origin,
        direction=direction,
        length=length,
        depth=depth,
        chevron_angle=angle,
    )


__all__ = [
    "NotchGeometry",
    "NotchOptions",
    "NotchType",
    "create_center_notch",
    "create_chevron_notch",
    "create_edge_notch",
    "insert_notch",
]
