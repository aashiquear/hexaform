"""Tests for the constrained Jacobian-barrier mesh optimisation."""

import numpy as np
import pytest

from hexaform.core.mesh import HexMesh
from hexaform.core.smoothing import (
    SmoothingOptions,
    SmoothingResult,
    detect_surface_nodes,
    smooth_surface,
)
from hexaform.shapes.library import Sphere


def _toy_mesh():
    """A 2x2x2 mesh around a single sphere."""
    pts = np.array(
        [
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    block_id = np.array([1], dtype=np.int32)
    return HexMesh(points=pts, cells=cells, block_id=block_id, element_type="HEX8")


def test_smoothing_runs_and_returns_mesh():
    """A trivial smoothing pass returns a mesh with the same topology."""
    mesh = _toy_mesh()
    opts = SmoothingOptions(n_iterations=2)
    sphere = Sphere(np.array([0.25, 0.25, 0.25]), 0.25)
    surface = np.array([1, 2, 5, 6], dtype=np.int64)  # corners
    res = smooth_surface(mesh, surface, opts, surface_projector=sphere.project)
    assert res.mesh.n_nodes == mesh.n_nodes
    assert res.mesh.n_elements == mesh.n_elements
    assert res.initial_min_scaled_jacobian > 0
    assert res.final_min_scaled_jacobian > 0


def test_smoothing_locks_surface_nodes():
    """Surface nodes remain on the analytic sphere after the pass."""
    centre = np.array([0.25, 0.25, 0.25])
    radius = 0.25
    sphere = Sphere(centre, radius)
    # A mesh where the "surface" nodes are slightly off the sphere
    pts = np.array(
        [
            centre + np.array([radius, 0.0, 0.0]) * (1 + 0.01),  # bumped out
            centre + np.array([0.0, radius, 0.0]) * (1 + 0.01),
            centre + np.array([0.0, 0.0, radius]) * (1 + 0.01),
            centre,
            centre + np.array([-radius, 0.0, 0.0]) * (1 + 0.01),
        ],
        dtype=np.float64,
    )
    # Use a tiny mesh; we only need 5+ nodes
    while pts.shape[0] < 8:
        pts = np.vstack([pts, centre])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    mesh = HexMesh(
        points=pts,
        cells=cells,
        block_id=np.array([1], dtype=np.int32),
        element_type="HEX8",
    )
    surface = np.array([0, 1, 2, 4], dtype=np.int64)
    opts = SmoothingOptions(n_iterations=3, lock_surface=True, surface_tol=0.05)
    res = smooth_surface(mesh, surface, opts, surface_projector=sphere.project)
    for k in surface:
        d = np.linalg.norm(res.mesh.points[k] - centre)
        assert d == pytest.approx(radius, rel=1e-6, abs=1e-6)


def test_smoothing_returns_improvement_or_maintains_quality():
    """A smoothing pass must not decrease the minimum scaled Jacobian."""
    mesh = _toy_mesh()
    sphere = Sphere(np.array([0.25, 0.25, 0.25]), 0.25)
    surface = np.array([], dtype=np.int64)
    opts = SmoothingOptions(n_iterations=5)
    res = smooth_surface(mesh, surface, opts, surface_projector=sphere.project)
    assert res.final_min_scaled_jacobian >= res.initial_min_scaled_jacobian - 1e-6


def test_detect_surface_nodes():
    """detect_surface_nodes returns nodes on the analytic surface."""
    sphere = Sphere(np.zeros(3), 1.0)
    # A point exactly on the surface
    pts = np.array(
        [
            [1.0, 0.0, 0.0],   # on surface
            [0.0, 1.0, 0.0],   # on surface
            [0.0, 0.0, 1.0],   # on surface
            [0.5, 0.0, 0.0],   # inside
            [2.0, 0.0, 0.0],   # outside
        ],
        dtype=np.float64,
    )
    mesh = HexMesh(
        points=pts,
        cells=np.array([[0, 1, 2, 3, 4, 0, 0, 0]], dtype=np.int64),
        block_id=np.array([1], dtype=np.int32),
        element_type="HEX8",
    )
    surface = detect_surface_nodes(mesh, [sphere], tol=1e-6)
    assert set(int(s) for s in surface) == {0, 1, 2}


def test_smoothing_options_default():
    opts = SmoothingOptions()
    assert opts.n_iterations > 0
    assert opts.step_size > 0
    assert opts.min_scaled_jacobian > 0
    assert opts.lock_surface is True


def test_smoothing_on_real_mesh_preserves_validity():
    """A smoothing pass on a real Hexaform mesh keeps minSJ > 0."""
    from hexaform.pipeline.run import particles_in_cube

    result = particles_in_cube(
        length=0.4, n_particles=2, radius_mean=0.05,
        mesh_density=10, seed=11, output=None,
    )
    mesh = result.mesh
    surface = detect_surface_nodes(mesh, result.packing.inclusions)
    shapes = [inc.shape for inc in result.packing.inclusions]
    def projector(x):
        out = np.zeros_like(x)
        for s in shapes:
            d = np.linalg.norm(x - s.center, axis=1, keepdims=True)
            scale = s.radial_scale(x)
            d_to_surf = (scale - 1.0)[:, None] * (x - s.center) / np.maximum(d, 1e-12)
            out = s.project(x)
        return out
    opts = SmoothingOptions(n_iterations=3)
    res = smooth_surface(mesh, surface, opts, surface_projector=projector)
    assert res.final_min_scaled_jacobian > 0
    assert res.mesh.n_elements == mesh.n_elements


def test_smoothing_in_pipeline():
    """The pipeline --smoothing flag invokes the smoothing pass."""
    from hexaform import RunConfig, run
    from hexaform.pipeline.config import (
        DomainConfig,
        InclusionConfig,
        OutputConfig,
        SmoothingConfig,
    )

    config = RunConfig(
        domain=DomainConfig(length=0.4, mesh_density=10),
        inclusions=InclusionConfig(
            shape="sphere", n_particles=1, radius_mean=0.05, radius_std=0.0
        ),
        output=OutputConfig(path=""),
        smoothing=SmoothingConfig(enabled=True, n_iterations=3),
        seed=42,
    )
    result = run(config)
    assert result.validation is not None
    assert result.validation.passed
    assert "smoothing" in result.mesh.metadata.get("timings", {})
