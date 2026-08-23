"""Whole-mesh guarantees: conforming, all-hex, watertight, correctly labelled."""

import numpy as np
import pytest

from hexaform.assembly.builder import AssemblyOptions, assemble
from hexaform.background.cartesian import CartesianGrid
from hexaform.core import topology
from hexaform.core.quality import assess
from hexaform.core.validation import validate
from hexaform.packing.rsa import PackingSpec, pack


@pytest.fixture(scope="module")
def mesh_and_pack():
    grid = CartesianGrid.cube(1.0, 20)
    res = pack(
        grid,
        PackingSpec(n_particles=6, radius_mean=0.09, radius_std=0.02),
        np.random.default_rng(5),
    )
    return grid, res, assemble(grid, res.inclusions)


def test_mesh_is_conforming(mesh_and_pack):
    _, _, mesh = mesh_and_pack
    report = topology.check_conformity(mesh)
    assert report["n_hanging_nodes"] == 0
    assert report["n_nonmanifold_faces"] == 0
    assert report["n_degenerate_elements"] == 0
    assert report["n_duplicate_elements"] == 0
    assert report["n_unused_nodes"] == 0
    assert report["conforming"]


def test_no_inverted_elements(mesh_and_pack):
    _, _, mesh = mesh_and_pack
    q = assess(mesh)
    assert q.n_inverted == 0
    assert q.min_scaled_jacobian > 0.2


def test_mesh_exactly_fills_the_domain(mesh_and_pack):
    """Total element volume equals the cube: no gaps, no overlaps."""
    grid, _, mesh = mesh_and_pack
    total = topology.signed_volumes(mesh.points, mesh.cells).sum()
    assert total == pytest.approx(float(np.prod(grid.extent)), rel=1e-12)


def test_only_the_cube_exterior_is_boundary(mesh_and_pack):
    """Internal voids would show up as extra boundary faces."""
    grid, _, mesh = mesh_and_pack
    elems, _ = topology.boundary_faces(mesh.cells)
    assert elems.size == 6 * grid.divisions[0] ** 2


def test_blocks_and_sets_exist_and_are_sized_right(mesh_and_pack):
    grid, res, mesh = mesh_and_pack
    assert set(mesh.block_names.values()) == {"matrix", "inclusions"}
    for name in ("x-", "x+", "y-", "y+", "z-", "z+"):
        assert mesh.side_sets[name][0].size == grid.divisions[0] ** 2
        assert mesh.node_sets[name].size == (grid.divisions[0] + 1) ** 2
    expected = sum(int(2 * (k[0] * k[1] + k[1] * k[2] + k[2] * k[0])) for k in (inc.k for inc in res.inclusions))
    assert mesh.side_sets["interface"][0].size == expected


def test_interface_side_set_is_recoverable_from_block_adjacency():
    """The template-supplied interface must agree with a generic block scan."""
    grid = CartesianGrid.cube(1.0, 16)
    res = pack(grid, PackingSpec(n_particles=3, radius_mean=0.10),
               np.random.default_rng(7))
    mesh = assemble(grid, res.inclusions)
    elems, _ = topology.interface_faces(mesh.cells, mesh.block_id, 1, 100)
    assert elems.size == mesh.side_sets["interface"][0].size


def test_interface_faces_sit_on_the_analytic_surfaces(mesh_and_pack):
    from hexaform.core.mesh import HEX8_SIDES

    _, res, mesh = mesh_and_pack
    elems, sides = mesh.side_sets["interface"]
    for elem, side in zip(elems[::37], sides[::37]):
        nodes = mesh.cells[elem][list(HEX8_SIDES[side - 1])]
        pts = mesh.points[nodes]
        assert any(
            np.abs(inc.shape.radial_scale(pts) - 1.0).max() < 1e-10
            for inc in res.inclusions
        ), "an interface face does not lie on any inclusion surface"


def test_particle_id_field_signs_matrix_and_inclusion(mesh_and_pack):
    _, res, mesh = mesh_and_pack
    pid = mesh.element_fields["particle_id"]
    assert pid.shape[0] == mesh.n_elements
    incl = mesh.block_id >= 100
    assert np.all(pid[incl] > 0)
    assert set(np.unique(pid[incl])) == {inc.particle_id for inc in res.inclusions}


def test_separate_blocks_option_creates_one_block_per_inclusion():
    grid = CartesianGrid.cube(1.0, 16)
    res = pack(grid, PackingSpec(n_particles=3, radius_mean=0.10),
               np.random.default_rng(7))
    mesh = assemble(grid, res.inclusions,
                    options=AssemblyOptions(separate_inclusion_blocks=True))
    assert len(mesh.block_ids) == 1 + res.count


def test_empty_packing_still_produces_a_valid_matrix_mesh():
    grid = CartesianGrid.cube(1.0, 8)
    mesh = assemble(grid, [])
    assert mesh.n_elements == 8**3
    assert assess(mesh).min_scaled_jacobian == pytest.approx(1.0)


def test_full_validation_passes(mesh_and_pack):
    grid, res, mesh = mesh_and_pack
    report = validate(mesh, res.inclusions, cell_size=grid.h)
    assert report.passed, report.failures
    assert report.geometry["max_node_surface_error"] < 1e-12
    assert report.geometry["relative_volume_error"] < 0.05
