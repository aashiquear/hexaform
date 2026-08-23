"""Higher-order HEX20/HEX27 generation, geometry and I/O regressions."""

from __future__ import annotations

import numpy as np
import pytest

from hexaform.assembly.builder import assemble
from hexaform.background.cartesian import CartesianGrid
from hexaform.cli import main
from hexaform.core.elevate import elevate
from hexaform.core.elements import REFERENCE_COORDS, side_nodes
from hexaform.core.mesh import HexMesh
from hexaform.core.quality import assess
from hexaform.core.topology import check_conformity
from hexaform.core.validation import validate
from hexaform.io import exodus, vtk
from hexaform.packing.rsa import PackedInclusion, PackingSpec, pack
from hexaform.shapes.library import Ellipsoid, Sphere, Superquadric


def _one_particle_meshes():
    grid = CartesianGrid.cube(1.0, 8)
    packing = pack(
        grid,
        PackingSpec(n_particles=1, radius_mean=0.12),
        np.random.default_rng(1),
    )
    return grid, packing, {
        kind: assemble(grid, packing.inclusions, element_type=kind)
        for kind in ("HEX8", "HEX20", "HEX27")
    }


def test_exodus_reference_node_order_matches_side_topology():
    expected_edges = np.array(
        [
            [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
            [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
        ],
        dtype=float,
    )
    assert np.array_equal(REFERENCE_COORDS["HEX20"][8:], expected_edges)
    assert np.array_equal(
        REFERENCE_COORDS["HEX27"][20:],
        np.array(
            [
                [0, 0, 0], [0, 0, -1], [0, 0, 1],
                [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
            ],
            dtype=float,
        ),
    )

    for kind in ("HEX8", "HEX20", "HEX27"):
        refs = REFERENCE_COORDS[kind]
        for side, local_nodes in enumerate(side_nodes(kind), start=1):
            face = refs[list(local_nodes)]
            axis = (1, 0, 1, 0, 2, 2)[side - 1]
            value = (-1, 1, 1, -1, -1, 1)[side - 1]
            assert np.all(face[:, axis] == value)


def test_global_elevation_deduplicates_shared_edge_and_face_nodes():
    grid = CartesianGrid((2, 1, 1), np.zeros(3), 1.0)
    points, cells, _ = grid.build()
    linear = HexMesh(points, cells, np.ones(2, dtype=np.int32))

    hex20 = elevate(linear, "HEX20")
    hex27 = elevate(linear, "HEX27")

    assert hex20.cells.shape == (2, 20)
    assert hex20.n_nodes == 12 + 20  # corners + globally unique edges
    assert hex27.cells.shape == (2, 27)
    assert hex27.n_nodes == 12 + 20 + 11 + 2  # + unique faces + cell centres
    assert check_conformity(hex20)["conforming"]
    assert check_conformity(hex27)["conforming"]
    assert assess(hex20).min_scaled_jacobian == pytest.approx(1.0)
    assert assess(hex27).min_scaled_jacobian == pytest.approx(1.0)


@pytest.mark.parametrize("kind,width", [("HEX20", 20), ("HEX27", 27)])
def test_curved_interface_nodes_are_projected_and_mesh_is_valid(kind, width):
    grid, packing, meshes = _one_particle_meshes()
    mesh = meshes[kind]
    shape = packing.inclusions[0].shape
    elems, sides = mesh.side_sets["interface"]
    interface_nodes = np.unique(
        np.concatenate(
            [mesh.cells[e, list(side_nodes(kind)[s - 1])] for e, s in zip(elems, sides)]
        )
    )

    assert mesh.cells.shape[1] == width
    assert np.abs(shape.radial_scale(mesh.points[interface_nodes]) - 1.0).max() < 1e-12
    report = validate(mesh, packing.inclusions, cell_size=grid.h)
    assert report.passed, report.failures
    assert report.topology["conforming"]
    assert report.quality.n_inverted == 0


@pytest.mark.parametrize(
    "shape",
    [
        Sphere(np.array([0.5, 0.5, 0.5]), 0.12),
        Ellipsoid(np.array([0.5, 0.5, 0.5]), np.array([0.12, 0.10, 0.08])),
        Superquadric(
            np.array([0.5, 0.5, 0.5]), np.array([0.12, 0.10, 0.08]), 3.0, 3.0
        ),
    ],
    ids=["sphere", "ellipsoid", "superquadric"],
)
def test_hex27_projection_is_star_shape_generic(shape):
    grid = CartesianGrid.cube(1.0, 8)
    inclusion = PackedInclusion(shape, np.array([2, 2, 2]), 4, 1)
    mesh = assemble(grid, [inclusion], element_type="HEX27")
    elems, sides = mesh.side_sets["interface"]
    nodes = np.unique(
        np.concatenate(
            [mesh.cells[e, list(side_nodes("HEX27")[s - 1])] for e, s in zip(elems, sides)]
        )
    )
    assert np.abs(shape.radial_scale(mesh.points[nodes]) - 1.0).max() < 1e-12
    assert assess(mesh).n_inverted == 0


@pytest.mark.parametrize(
    "kind,expected_boundary_nodes",
    [("HEX8", 81), ("HEX20", 225), ("HEX27", 289)],
)
def test_boundary_node_sets_include_higher_order_nodes(kind, expected_boundary_nodes):
    _, _, meshes = _one_particle_meshes()
    mesh = meshes[kind]
    for name in ("x-", "x+", "y-", "y+", "z-", "z+"):
        assert mesh.side_sets[name][0].size == 64
        assert mesh.node_sets[name].size == expected_boundary_nodes


def test_higher_order_geometry_reduces_volume_error_at_fixed_element_count():
    grid, packing, meshes = _one_particle_meshes()
    errors = {
        kind: validate(mesh, packing.inclusions, cell_size=grid.h).geometry[
            "relative_volume_error"
        ]
        for kind, mesh in meshes.items()
    }
    assert {mesh.n_elements for mesh in meshes.values()} == {1112}
    assert errors["HEX20"] < 0.05 * errors["HEX8"]
    assert errors["HEX27"] < 0.25 * errors["HEX20"]


@pytest.mark.parametrize("kind", ["HEX20", "HEX27"])
def test_higher_order_exodus_and_vtk_output(kind, tmp_path):
    _, _, meshes = _one_particle_meshes()
    mesh = meshes[kind]
    exodus_path = exodus.write(mesh, tmp_path / f"{kind}.e", backend="scipy")
    back = exodus.read(exodus_path)
    check = exodus.round_trip_check(mesh, exodus_path)

    assert back.element_type == kind
    assert np.array_equal(back.cells, np.concatenate(
        [mesh.cells[mesh.block_id == bid] for bid in mesh.block_ids], axis=0
    ))
    assert check["element_type_match"]
    assert check["max_coordinate_error"] == 0.0

    vtk_path = vtk.write(mesh, tmp_path / f"{kind}.vtk")
    text = vtk_path.read_text()
    vtk_id = 25 if kind == "HEX20" else 29
    assert f"CELLS {mesh.n_elements} {mesh.n_elements * (mesh.nodes_per_element + 1)}" in text
    assert f"\n{vtk_id}\n" in text


@pytest.mark.parametrize("kind", ["HEX20", "HEX27"])
def test_particles_in_cube_cli_selects_element_type(kind, tmp_path):
    path = tmp_path / f"cli_{kind}.e"
    code = main(
        [
            "particles-in-cube",
            "--n-particles", "1",
            "--radius-mean", "0.12",
            "--mesh-density", "8",
            "--element-type", kind,
            "--seed", "1",
            "--quiet",
            "-o", str(path),
        ]
    )
    assert code == 0
    assert exodus.read(path).element_type == kind
