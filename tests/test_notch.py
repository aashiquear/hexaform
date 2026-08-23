"""Tests for notch and pre-crack insertion."""

import numpy as np
import pytest

from hexaform import RunConfig, run
from hexaform.assembly.notch import (
    NotchGeometry,
    NotchOptions,
    NotchType,
    create_center_notch,
    create_chevron_notch,
    create_edge_notch,
    insert_notch,
)
from hexaform.core.mesh import HexMesh
from hexaform.pipeline.config import DomainConfig, InclusionConfig, NotchConfig, OutputConfig


class TestNotchGeometry:
    """Test notch geometry creation and validation."""

    def test_edge_notch_creation(self):
        """Test creating an edge notch with default parameters."""
        notch = create_edge_notch(length=0.5)
        assert notch.notch_type == NotchType.EDGE
        assert notch.length == 0.5
        assert notch.origin == (0.0, 0.5, 0.5)
        assert notch.direction == (1.0, 0.0, 0.0)
        assert notch.depth == 1.0

    def test_center_notch_creation(self):
        """Test creating a center notch."""
        notch = create_center_notch(length=0.4, origin=(0.5, 0.5, 0.5))
        assert notch.notch_type == NotchType.CENTER
        assert notch.length == 0.4
        assert notch.origin == (0.5, 0.5, 0.5)

    def test_chevron_notch_creation(self):
        """Test creating a chevron notch."""
        notch = create_chevron_notch(length=0.5, angle=60.0)
        assert notch.notch_type == NotchType.CHEVRON
        assert notch.chevron_angle == 60.0

    def test_invalid_direction(self):
        """Test that zero direction vector raises error."""
        with pytest.raises(ValueError, match="direction vector cannot be zero"):
            NotchGeometry(direction=(0, 0, 0), length=0.5)

    def test_invalid_length(self):
        """Test that non-positive length raises error."""
        with pytest.raises(ValueError, match="length must be positive"):
            NotchGeometry(length=-0.5)

    def test_invalid_width(self):
        """Test that negative width raises error."""
        with pytest.raises(ValueError, match="width cannot be negative"):
            NotchGeometry(length=0.5, width=-0.1)

    def test_direction_normalization(self):
        """Test that direction vector is normalized."""
        notch = NotchGeometry(direction=(2.0, 0.0, 0.0), length=0.5)
        assert np.linalg.norm(notch.direction) == pytest.approx(1.0)


class TestNotchInsertion:
    """Test notch insertion into meshes."""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple 10x10x10 mesh for testing."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=10),
            inclusions=InclusionConfig(n_particles=0),  # No particles
            output=OutputConfig(path=""),
            validate_mesh=False,
            verbose=False,
        )
        result = run(config)
        return result.mesh

    def test_edge_notch_field_method(self, simple_mesh):
        """Test edge notch insertion with field method."""
        notch_geom = create_edge_notch(length=0.3, origin=(0.0, 0.5, 0.5))
        notch_opts = NotchOptions(method="field", variable_name="phase_field")

        mesh = insert_notch(
            simple_mesh,
            notch_geom,
            notch_opts,
            domain_extent=(1.0, 1.0, 1.0),
            domain_origin=(0.0, 0.0, 0.0),
        )

        # Check that phase_field variable exists
        assert "phase_field" in mesh.element_fields
        pf = mesh.element_fields["phase_field"]

        # Check that some elements are damaged
        n_damaged = int(np.sum(pf > 0.5))
        assert n_damaged > 0

        # Check metadata
        assert mesh.metadata["notch_type"] == "edge"
        assert mesh.metadata["notch_method"] == "field"
        assert mesh.metadata["n_notch_elements"] == n_damaged

    def test_edge_notch_geometric_method(self, simple_mesh):
        """Test edge notch insertion with geometric method."""
        # For a 10x10x10 mesh with h=0.1, element centroids are at 0.05, 0.15, ..., 0.95
        # Start notch at left face and extend 0.3 units (should cover ~3 elements)
        notch_geom = create_edge_notch(length=0.3, origin=(0.0, 0.5, 0.5))
        notch_opts = NotchOptions(method="geometric", geometric_width=0.15)  # Wider to catch elements

        mesh = insert_notch(
            simple_mesh,
            notch_geom,
            notch_opts,
            domain_extent=(1.0, 1.0, 1.0),
            domain_origin=(0.0, 0.0, 0.0),
        )

        # Check that particle_id field exists
        assert "particle_id" in mesh.element_fields
        pid = mesh.element_fields["particle_id"]

        # Check that some elements are marked as void (particle_id = 0)
        n_void = int(np.sum(pid == 0))
        assert n_void > 0

        # Check metadata
        assert mesh.metadata["notch_type"] == "edge"
        assert mesh.metadata["notch_method"] == "geometric"

    def test_center_notch(self, simple_mesh):
        """Test center notch insertion."""
        notch_geom = create_center_notch(length=0.4, origin=(0.5, 0.5, 0.5))
        notch_opts = NotchOptions(method="field")

        mesh = insert_notch(
            simple_mesh,
            notch_geom,
            notch_opts,
            domain_extent=(1.0, 1.0, 1.0),
        )

        assert "phase_field" in mesh.element_fields
        assert mesh.metadata["notch_type"] == "center"

    def test_chevron_notch(self, simple_mesh):
        """Test chevron notch insertion."""
        notch_geom = create_chevron_notch(length=0.4, angle=45.0)
        notch_opts = NotchOptions(method="field")

        mesh = insert_notch(
            simple_mesh,
            notch_geom,
            notch_opts,
            domain_extent=(1.0, 1.0, 1.0),
        )

        assert "phase_field" in mesh.element_fields
        assert mesh.metadata["notch_type"] == "chevron"

    def test_notch_element_set(self, simple_mesh):
        """Test that notch element indices are stored."""
        notch_geom = create_edge_notch(length=0.3)
        notch_opts = NotchOptions(method="field", mark_elements=True)

        mesh = insert_notch(simple_mesh, notch_geom, notch_opts, domain_extent=(1.0, 1.0, 1.0))

        # Check that notch element indices are stored in metadata
        assert "notch_element_ids" in mesh.metadata
        notch_ids = mesh.metadata["notch_element_ids"]
        assert len(notch_ids) > 0

    def test_no_notch_elements_found(self, simple_mesh):
        """Test error when notch is outside domain."""
        # Notch origin far outside domain
        notch_geom = NotchGeometry(
            notch_type=NotchType.EDGE,
            origin=(10.0, 10.0, 10.0),  # Way outside
            length=0.1,
        )
        notch_opts = NotchOptions()

        with pytest.raises(ValueError, match="No elements found"):
            insert_notch(
                simple_mesh,
                notch_geom,
                notch_opts,
                domain_extent=(1.0, 1.0, 1.0),
            )

    def test_notch_preserves_mesh_topology(self, simple_mesh):
        """Test that notch insertion doesn't change mesh topology."""
        original_n_elements = simple_mesh.n_elements
        original_n_nodes = simple_mesh.n_nodes
        original_cells_shape = simple_mesh.cells.shape

        notch_geom = create_edge_notch(length=0.3, origin=(0.0, 0.5, 0.5))
        notch_opts = NotchOptions(method="field")

        mesh = insert_notch(simple_mesh, notch_geom, notch_opts, domain_extent=(1.0, 1.0, 1.0))

        # Topology should be unchanged
        assert mesh.n_elements == original_n_elements
        assert mesh.n_nodes == original_n_nodes
        assert mesh.cells.shape == original_cells_shape


class TestNotchPipeline:
    """Test notch insertion through the full pipeline."""

    def test_pipeline_with_notch(self):
        """Test full pipeline with notch insertion."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=20),
            inclusions=InclusionConfig(
                shape="sphere",
                n_particles=5,
                radius_mean=0.05,
            ),
            output=OutputConfig(path=""),
            seed=42,  # Seed goes in RunConfig, not InclusionConfig
            notch=NotchConfig(
                enabled=True,
                notch_type="edge",
                origin=(0.0, 0.5, 0.5),
                length=0.3,
                method="field",
            ),
            validate_mesh=False,
            verbose=False,
        )

        result = run(config)

        # Check that notch was inserted
        assert result.mesh.metadata.get("notch_type") == "edge"
        assert "phase_field" in result.mesh.element_fields
        assert result.mesh.metadata["n_notch_elements"] > 0

    def test_pipeline_without_notch(self):
        """Test that pipeline works without notch (default)."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=20),
            inclusions=InclusionConfig(n_particles=5, radius_mean=0.05),
            output=OutputConfig(path=""),
            validate_mesh=False,
            verbose=False,
        )

        result = run(config)

        # Check that no notch metadata is present
        assert "notch_type" not in result.mesh.metadata
        assert "phase_field" not in result.mesh.element_fields

    def test_pipeline_geometric_notch(self):
        """Test full pipeline with geometric notch."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=20),
            inclusions=InclusionConfig(n_particles=3, radius_mean=0.05),
            output=OutputConfig(path=""),
            notch=NotchConfig(
                enabled=True,
                notch_type="edge",
                length=0.25,
                method="geometric",
            ),
            validate_mesh=False,
            verbose=False,
        )

        result = run(config)

        assert result.mesh.metadata.get("notch_type") == "edge"
        assert result.mesh.metadata.get("notch_method") == "geometric"
        assert "particle_id" in result.mesh.element_fields


class TestNotchResolution:
    """Test notch resolution and accuracy."""

    def test_notch_element_count_scales_with_resolution(self):
        """Test that notch element count increases with mesh density."""
        counts = []
        for density in [10, 20, 30]:
            config = RunConfig(
                domain=DomainConfig(length=1.0, mesh_density=density),
                inclusions=InclusionConfig(n_particles=0),
                output=OutputConfig(path=""),
                notch=NotchConfig(
                    enabled=True,
                    notch_type="edge",
                    origin=(0.0, 0.5, 0.5),
                    length=0.3,
                    method="field",
                ),
                validate_mesh=False,
                verbose=False,
            )
            result = run(config)
            counts.append(result.mesh.metadata["n_notch_elements"])

        # Element count should increase with density
        assert counts[0] < counts[1] < counts[2]

    def test_notch_length_accuracy(self):
        """Test that notch length is accurately represented."""
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=40),
            inclusions=InclusionConfig(n_particles=0),
            output=OutputConfig(path=""),
            notch=NotchConfig(
                enabled=True,
                notch_type="edge",
                origin=(0.0, 0.5, 0.5),
                length=0.5,
                method="field",
            ),
            validate_mesh=False,
            verbose=False,
        )
        result = run(config)

        # Compute centroids of damaged elements
        mesh = result.mesh
        pf = mesh.element_fields["phase_field"]
        damaged_mask = pf > 0.5

        # Compute centroids from cell connectivity
        corners = mesh.cells[:, :8]  # (n_elements, 8)
        corner_coords = mesh.points[corners]  # (n_elements, 8, 3)
        centroids = corner_coords.mean(axis=1)  # (n_elements, 3)
        damaged_centroids = centroids[damaged_mask]

        # Check that damaged elements span approximately the notch length
        if len(damaged_centroids) > 0:
            x_extent = damaged_centroids[:, 0].max() - damaged_centroids[:, 0].min()
            # Allow some tolerance due to discrete elements
            assert 0.4 <= x_extent <= 0.6, f"Notch x-extent {x_extent} not close to 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
