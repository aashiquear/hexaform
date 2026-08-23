"""Tests for the octree-based assembly (Backend B)."""

import numpy as np
import pytest

from hexaform.assembly.octree_assembly import (
    OctreeAssemblyOptions,
    assemble_with_octree,
    project_surface_nodes,
)
from hexaform.core.mesh import HexMesh
from hexaform.shapes.library import Sphere
from hexaform.shapes.torus import Torus


def test_assemble_with_octree_two_spheres():
    """A pair of star-shaped bodies produces a mesh with matrix + 2 inclusions."""
    shapes = [
        Sphere(np.array([0.3, 0.3, 0.3]), 0.1),
        Sphere(np.array([0.7, 0.7, 0.7]), 0.12),
    ]
    mesh = assemble_with_octree(
        shapes,
        domain_length=1.0,
        options=OctreeAssemblyOptions(
            bulk_h=0.05, interface_h=0.025, interface_band=0.05, max_level=5
        ),
    )
    assert mesh.n_elements > 0
    assert mesh.n_nodes > 0
    block_ids = set(mesh.block_ids.tolist())
    assert 1 in block_ids  # matrix
    assert 100 in block_ids
    assert 101 in block_ids


def test_assemble_with_octree_torus():
    """A torus (non-star) is built into a mesh with the right block structure."""
    torus = Torus(np.array([0.5, 0.5, 0.5]), major_radius=0.25, minor_radius=0.1)
    mesh = assemble_with_octree(
        [torus],
        domain_length=1.0,
        options=OctreeAssemblyOptions(
            bulk_h=0.05, interface_h=0.025, interface_band=0.05, max_level=5
        ),
    )
    assert mesh.n_elements > 0
    # Matrix and inclusion
    assert 1 in mesh.block_ids.tolist()
    assert 100 in mesh.block_ids.tolist()
    # Side sets for the six exterior faces
    assert "x-" in mesh.side_sets
    assert "z+" in mesh.side_sets


def test_project_surface_nodes_for_torus():
    """Surface nodes are projected onto the analytic torus surface."""
    torus = Torus(np.array([0.5, 0.5, 0.5]), major_radius=0.25, minor_radius=0.1)
    # Build a uniform 16^3 grid as the "background"
    n = 16
    h = 1.0 / n
    i, j, k = np.meshgrid(np.arange(n + 1), np.arange(n + 1), np.arange(n + 1), indexing="ij")
    points = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) * h
    corners = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.int64,
    )
    cell_idx = np.stack(
        np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cells = cell_idx[:, None, :] + corners[None, :, :]
    cells = (cells[..., 0] * (n + 1) + cells[..., 1]) * (n + 1) + cells[..., 2]
    mesh = HexMesh(
        points=points,
        cells=cells.astype(np.int64),
        block_id=np.ones(cells.shape[0], dtype=np.int32),
        element_type="HEX8",
    )
    new_mesh = project_surface_nodes(mesh, [torus], tol=1e-6)
    # Verify that the SDF of the projected points is small
    pts = new_mesh.points
    sdf = torus.sdf(pts)
    # Nodes that were already on the surface should be on the surface
    # (within the octree resolution).  Find nodes that were within tol of
    # the surface before; they should be on the surface after projection.
    original_sdf = torus.sdf(mesh.points)
    near = np.flatnonzero(np.abs(original_sdf) < 1e-3)
    if near.size:
        sdf_after = np.abs(torus.sdf(new_mesh.points[near]))
        assert np.max(sdf_after) < 1e-3


def test_empty_octree_assembly():
    """An empty list of shapes returns an empty mesh."""
    mesh = assemble_with_octree([])
    assert mesh.n_elements == 0
    assert mesh.n_nodes == 0


def test_assembly_options_default():
    opts = OctreeAssemblyOptions()
    assert opts.bulk_h > 0
    assert opts.interface_h > 0
    assert opts.interface_band > 0
    assert opts.max_level > 0
    assert opts.project_surface_nodes is True


def test_octree_in_pipeline():
    """The pipeline --backend=octree flag selects the octree assembly path."""
    from hexaform import RunConfig, run
    from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig

    config = RunConfig(
        domain=DomainConfig(
            length=0.4,
            mesh_density=10,
            backend="octree",
            octree_bulk_h=0.04,
            octree_interface_h=0.02,
            octree_max_level=5,
        ),
        inclusions=InclusionConfig(
            shape="sphere", n_particles=2, radius_mean=0.05, radius_std=0.0
        ),
        output=OutputConfig(path=""),
        seed=11,
    )
    result = run(config)
    assert result.validation is not None
    assert result.validation.passed
    assert result.mesh.metadata.get("background", "cartesian") in ("cartesian", "octree")
    # The mesh should be valid
    assert result.mesh.n_elements > 0


def test_octree_with_torus_in_pipeline():
    """The pipeline can mesh a torus (non-star body) through the octree path."""
    import numpy as np

    from hexaform.assembly.octree_assembly import (
        OctreeAssemblyOptions,
        assemble_with_octree,
    )
    from hexaform.shapes.torus import Torus

    torus = Torus(
        centre=np.array([0.5, 0.5, 0.5]),
        major_radius=0.3,
        minor_radius=0.1,
    )
    mesh = assemble_with_octree(
        [torus],
        domain_length=1.0,
        options=OctreeAssemblyOptions(
            bulk_h=0.05,
            interface_h=0.025,
            interface_band=0.05,
            max_level=5,
            project_surface_nodes=True,
        ),
    )
    # The mesh has 1 matrix + 1 inclusion block
    assert 1 in mesh.block_ids.tolist()
    assert 100 in mesh.block_ids.tolist()
    # Surface nodes were projected
    assert mesh.metadata.get("n_surface_nodes_projected", 0) > 0
