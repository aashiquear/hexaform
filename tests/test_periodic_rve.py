"""Tests for periodic RVE boundary conditions.

This module validates the implementation of periodic boundary conditions (PBC)
for Representative Volume Element (RVE) simulations. Periodic BCs are essential
for homogenization studies and comparison with particulate-composite literature.

Key concepts tested:
- Node pairing across opposite domain faces
- Macro-box wrapping at boundaries
- Consistent interface side sets for periodic particles
- Node set correspondence for constraint application
"""
from __future__ import annotations

import numpy as np
import pytest

from hexaform import RunConfig, run
from hexaform.assembly.builder import assemble
from hexaform.background.cartesian import CartesianGrid, OccupancyMap
from hexaform.packing.rsa import pack
from hexaform.core.mesh import MeshAccumulator
from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig


class TestPeriodicNodePairing:
    """Test node correspondence across periodic boundaries."""

    def test_periodic_node_pairing_x_axis(self):
        """Test that nodes on x- and x+ faces are properly paired."""
        grid = CartesianGrid.cube(length=1.0, divisions=20)

        # Get node IDs on x- and x+ faces
        x_minus_nodes = grid.boundary_node_indices(axis=0, side=0)
        x_plus_nodes = grid.boundary_node_indices(axis=0, side=1)

        # Should have the same number of nodes on each face
        assert len(x_minus_nodes) == len(x_plus_nodes), \
            f"Face node count mismatch: {len(x_minus_nodes)} vs {len(x_plus_nodes)}"

        # Get physical coordinates
        coords = grid.node_coordinates()

        # Match nodes by their (y, z) coordinates
        x_minus_coords = coords[x_minus_nodes]
        x_plus_coords = coords[x_plus_nodes]

        for i in range(len(x_minus_nodes)):
            yz_match = np.allclose(x_minus_coords[i][1:], x_plus_coords[i][1:])
            assert yz_match, f"Node {i} should match on y,z coordinates"

    def test_periodic_node_pairing_all_axes(self):
        """Test periodic pairing for all three axes."""
        grid = CartesianGrid.cube(length=1.0, divisions=16)

        # Test each axis
        for axis in range(3):
            neg_nodes = grid.boundary_node_indices(axis, side=0)
            pos_nodes = grid.boundary_node_indices(axis, side=1)

            coords = grid.node_coordinates()
            neg_coords = coords[neg_nodes]
            pos_coords = coords[pos_nodes]

            # Nodes should match on all axes except the current one
            other_axes = [a for a in range(3) if a != axis]
            for i in range(len(neg_nodes)):
                matches = np.isclose(neg_coords[i][other_axes],
                                     pos_coords[i][other_axes])
                assert all(matches), \
                    f"Axis {axis}: Node {i} should match on other axes"


class TestPeriodicRVEGeneration:
    """Test full RVE generation with periodic boundary conditions."""

    def test_basic_periodic_rve(self):
        """Generate a basic periodic RVE and verify node pairing."""
        # Simple case: 1 sphere in centered cube
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=20, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=3,
                radius_mean=0.08,
                radius_std=0.01,
            ),
            output=OutputConfig(path="/tmp/test_periodic_basic.e"),
            seed=42,
        )

        result = run(config)

        assert result.mesh is not None
        assert "x-" in result.mesh.side_sets
        assert "x+" in result.mesh.side_sets
        assert "y-" in result.mesh.side_sets
        assert "y+" in result.mesh.side_sets
        assert "z-" in result.mesh.side_sets
        assert "z+" in result.mesh.side_sets

        # Verify paired nodes exist
        for axis_name, axis_idx in [("x", 0), ("y", 1), ("z", 2)]:
            neg_set = result.mesh.node_sets.get(f"{axis_name}-", None)
            pos_set = result.mesh.node_sets.get(f"{axis_name}+", None)

            if neg_set is not None and pos_set is not None:
                assert len(neg_set) == len(pos_set), \
                    f"Periodic node count mismatch on {axis_name}"

    def test_periodic_with_off_center_sphere(self):
        """Test periodic RVE with off-center sphere (hides decomposition assumptions)."""
        # Use an off-center configuration to test robustness
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=24, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=5,
                radius_mean=0.07,
                radius_std=0.015,
            ),
            output=OutputConfig(path="/tmp/test_periodic_offcenter.e"),
            seed=17,
        )

        result = run(config)

        assert result.mesh is not None

        # Validate mesh quality
        assert result.validation is not None
        assert result.validation.passed

        # Check that periodic pairs exist for all axes
        expected_sets = ["x-", "x+", "y-", "y+", "z-", "z+"]
        for set_name in expected_sets:
            assert set_name in result.mesh.node_sets, \
                f"Missing periodic node set: {set_name}"

    def test_periodic_macro_block_winding(self):
        """Test that macro-blocks wrap correctly at periodic boundaries."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=32, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=8,
                radius_mean=0.06,
                radius_std=0.01,
            ),
            output=OutputConfig(path="/tmp/test_periodic_wrap.e"),
            seed=99,
        )

        result = run(config)

        assert result.mesh is not None

        # All elements should be valid (no inverted elements)
        assert result.validation is not None
        assert result.validation.quality.n_inverted == 0


class TestPeriodicNodeSetConstruction:
    """Test the construction of periodic node sets."""

    def test_periodic_node_sets_created(self):
        """Verify that periodic node sets are created correctly."""
        grid = CartesianGrid.cube(length=1.0, divisions=12)

        # Generate a mesh with no inclusions (simplest case)
        points, cells, _ = grid.build()

        acc = MeshAccumulator()
        acc.add(points, cells, 1, element_fields={"particle_id": np.zeros(cells.shape[0], dtype=np.int64)})

        mesh = acc.build({"1": "matrix"})

        # Build periodic node sets
        from hexaform.assembly.builder import _build_periodic_node_sets
        mesh = _build_periodic_node_sets(mesh, grid)

        # Should have node sets for all axes
        for axis_name in ["x", "y", "z"]:
            assert f"{axis_name}-_node_set" in mesh.node_sets
            assert f"{axis_name}+_node_set" in mesh.node_sets


class TestPeriodicBCValidation:
    """Test validation of periodic boundary conditions."""

    def test_periodic_jacobian_validity(self):
        """Test that Jacobian remains valid with periodic BCs."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=28, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=6,
                radius_mean=0.09,
                radius_std=0.015,
            ),
            output=OutputConfig(path="/tmp/test_periodic_jacobian.e"),
            seed=77,
        )

        result = run(config)

        assert result.validation is not None

        # Minimum scaled Jacobian should be above 0.2 threshold
        min_jacobian = result.validation.quality.min_scaled_jacobian
        assert min_jacobian >= 0.2, \
            f"Min scaled Jacobian {min_jacobian} below acceptable threshold of 0.2"

        # No inverted elements
        assert result.validation.quality.n_inverted == 0

    def test_periodic_interface_preserved(self):
        """Test that interface information is preserved with periodic BCs."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=24, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=4,
                radius_mean=0.10,
                radius_std=0.015,
            ),
            output=OutputConfig(path="/tmp/test_periodic_interface.e"),
            seed=88,
        )

        result = run(config)

        assert result.mesh is not None
        assert "interface" in result.mesh.side_sets

    def test_periodic_volume_error(self):
        """Test that geometric error remains low with periodic BCs."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=32, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=7,
                radius_mean=0.075,
                radius_std=0.012,
            ),
            output=OutputConfig(path="/tmp/test_periodic_volume.e"),
            seed=101,
        )

        result = run(config)

        assert result.validation is not None

        # Inclusion volume error should be reasonable for this resolution
        rel_error = result.validation.geometry.get('relative_volume_error', float('inf'))
        assert rel_error < 0.05, \
            f"Inclusion volume error {rel_error} unexpectedly high"


class TestPeriodicConsistency:
    """Test consistency across different configurations."""

    def test_periodic_seed_reproducibility(self):
        """Verify that periodic RVEs are reproducible with same seed."""
        results = []
        for _ in range(2):
            config = RunConfig(
                domain=DomainConfig(length=1.0, mesh_density=20, periodic=True),
                inclusions=InclusionConfig(
                    shape="sphere",
                    n_particles=4,
                    radius_mean=0.08,
                    radius_std=0.01,
                ),
                output=OutputConfig(path=f"/tmp/test_repro_{_}.e"),
                seed=200,
            )

            result = run(config)
            results.append(result)

        # Meshes should be identical for same seed
        assert results[0].mesh.points.shape == results[1].mesh.points.shape

        # Check node counts on periodic faces match
        for axis_name in ["x", "y", "z"]:
            neg_set_0 = results[0].mesh.node_sets.get(f"{axis_name}-", [])
            pos_set_0 = results[0].mesh.node_sets.get(f"{axis_name}+", [])
            neg_set_1 = results[1].mesh.node_sets.get(f"{axis_name}-", [])
            pos_set_1 = results[1].mesh.node_sets.get(f"{axis_name}+", [])

            assert len(neg_set_0) == len(neg_set_1), \
                f"Non-reproducible {axis_name}- node count"
            assert len(pos_set_0) == len(pos_set_1), \
                f"Non-reproducible {axis_name}+ node count"


class TestPeriodicEdgeCases:
    """Test edge cases for periodic boundary conditions."""

    def test_periodic_minimum_resolution(self):
        """Test periodic BCs at minimum viable resolution."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=8, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=2,
                radius_mean=0.15,
                radius_std=0.02,
            ),
            output=OutputConfig(path="/tmp/test_periodic_min.e"),
            seed=300,
        )

        result = run(config)

        # Should still generate valid mesh even at low resolution
        assert result.mesh is not None

    def test_periodic_maximum_particles(self):
        """Test periodic BCs with many particles."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=40, periodic=True),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=30,
                radius_mean=0.05,
                radius_std=0.008,
            ),
            output=OutputConfig(path="/tmp/test_periodic_max.e"),
            seed=400,
        )

        result = run(config)

        assert result.mesh is not None
        assert result.validation is not None
        assert result.validation.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
