"""Tests for the adaptive octree background (Backend B)."""

import numpy as np
import pytest

from hexaform.background.octree import (
    OctreeGrid,
    OctreeMesh,
    decode_location,
    encode_location,
    sphere_material_of,
    sphere_size_field,
)


def test_location_roundtrip():
    """Location codes encode and decode (level, ix, iy, iz) losslessly."""
    for level, ix, iy, iz in [
        (0, 0, 0, 0),
        (5, 3, 7, 11),
        (10, 1023, 511, 1),
        (1, 1, 1, 1),
    ]:
        loc = encode_location(level, ix, iy, iz)
        assert decode_location(loc) == (level, ix, iy, iz)


def test_location_rejects_invalid():
    with pytest.raises(ValueError):
        encode_location(-1, 0, 0, 0)
    with pytest.raises(ValueError):
        encode_location(0, -1, 0, 0)
    with pytest.raises(ValueError):
        encode_location(100, 0, 0, 0)


def test_uniform_field_yields_uniform_grid():
    """A constant size field and no materials give a uniform background."""
    grid = OctreeGrid(
        origin=np.zeros(3),
        length=1.0,
        h_min=0.0625,  # 1/16
        max_level=4,
    )
    points, cells, _idx = grid.build()
    # The default field refines down to h_min; the result is a uniform 16^3 grid.
    assert points.shape[0] == 17 ** 3
    assert cells.shape[0] == 16 ** 3
    # Every cell has the same edge length
    edges = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    assert np.allclose(edges, 0.0625)


def test_interface_refinement_produces_smaller_cells():
    """A sphere with a thin refinement band shrinks cells near the surface."""
    centres = np.array([[0.5, 0.5, 0.5]])
    radii = np.array([0.15])
    size_f = sphere_size_field(centres, radii, bulk_h=0.05, interface_h=0.02, band=0.05)
    mat_f = sphere_material_of(centres, radii)
    grid = OctreeGrid(
        origin=np.zeros(3),
        length=1.0,
        h_min=0.02,
        max_level=6,
        material_of=mat_f,
        size_field=size_f,
    )
    points, cells, _idx = grid.build()
    edges = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    sizes = np.unique(edges)
    # Must include the fine size (h_min) somewhere near the surface
    assert sizes.min() <= 0.04  # at most two levels deeper than bulk
    # Bulk cells are coarser
    assert sizes.max() >= 2 * sizes.min()


def test_total_volume_conservation():
    """The union of all leaf cells has the cube's volume to round-off."""
    centres = np.array([[0.5, 0.5, 0.5]])
    radii = np.array([0.1])
    size_f = sphere_size_field(centres, radii, bulk_h=0.05, interface_h=0.025, band=0.05)
    mat_f = sphere_material_of(centres, radii)
    grid = OctreeGrid(
        origin=np.zeros(3),
        length=1.0,
        h_min=0.025,
        max_level=5,
        material_of=mat_f,
        size_field=size_f,
    )
    points, cells, _idx = grid.build()
    edges = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    total_volume = float((edges ** 3).sum())
    assert total_volume == pytest.approx(1.0, rel=1e-9)


def test_straddling_cells_classified_correctly():
    """Cells whose centre lies in a sphere are flagged as inclusion material."""
    centres = np.array([[0.5, 0.5, 0.5]])
    radii = np.array([0.15])
    mat_f = sphere_material_of(centres, radii)
    size_f = sphere_size_field(centres, radii, bulk_h=0.05, interface_h=0.02, band=0.05)
    grid = OctreeGrid(
        origin=np.zeros(3),
        length=1.0,
        h_min=0.02,
        max_level=5,
        material_of=mat_f,
        size_field=size_f,
    )
    points, cells, _idx = grid.build()
    cell_centres = points[cells].mean(axis=1)
    cell_mats = mat_f(cell_centres)
    inside = cell_mats == 1
    # Volume of all-inside cells should be close to the analytic sphere
    edges = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=1)
    inside_volume = float((edges[inside] ** 3).sum())
    sphere_volume = (4.0 / 3.0) * np.pi * radii[0] ** 3
    assert inside_volume == pytest.approx(sphere_volume, rel=0.05)


def test_refine_along_surface_returns_subgrid():
    """refine_along_surface returns a refined OctreeMesh around the surface."""
    centres = np.array([[0.5, 0.5, 0.5]])
    radii = np.array([0.15])
    size_f = sphere_size_field(centres, radii, bulk_h=0.05, interface_h=0.025, band=0.05)
    mat_f = sphere_material_of(centres, radii)
    grid = OctreeGrid(
        origin=np.zeros(3),
        length=1.0,
        h_min=0.05,
        max_level=4,
        material_of=mat_f,
        size_field=size_f,
    )
    base = grid._build_mesh()
    # Refine around the sphere with a small band
    def sdf(x):
        d = np.linalg.norm(x - 0.5, axis=1) - 0.15
        return d

    refined = base.refine_along_surface(sdf, band=0.05, min_level=2)
    assert refined.n_elements >= base.n_elements
    # Some cells are smaller than the base
    base_edges = np.linalg.norm(base.points[base.cells[:, 1]] - base.points[base.cells[:, 0]], axis=1)
    refined_edges = np.linalg.norm(refined.points[refined.cells[:, 1]] - refined.points[refined.cells[:, 0]], axis=1)
    assert refined_edges.min() <= base_edges.min()


def test_legacy_interface_consistent():
    """OctreeGrid exposes the same basic properties as CartesianGrid."""
    grid = OctreeGrid(
        origin=np.array([0.0, 0.0, 0.0]),
        length=1.0,
        h_min=0.05,
        max_level=4,
    )
    assert grid.h == 0.05
    assert np.allclose(grid.extent, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(grid.upper, np.array([1.0, 1.0, 1.0]))
    # divisions should reflect the finest grid
    assert grid.divisions == (16, 16, 16)


def test_build_matches_uniform_grid_for_default_field():
    """Without a size field or materials, build() should return a uniform grid."""
    grid = OctreeGrid(
        origin=np.zeros(3),
        length=1.0,
        h_min=0.05,
        max_level=4,
    )
    points, cells, idx = grid.build()
    # 16^3 = 4096 cells, 17^3 = 4913 nodes
    assert cells.shape[0] == 4096
    assert points.shape[0] == 4913
    assert idx.shape == (4096, 3)
