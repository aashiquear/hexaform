"""Exodus must survive a round trip on both netCDF backends."""

import numpy as np
import pytest

from hexaform.assembly.builder import assemble
from hexaform.background.cartesian import CartesianGrid
from hexaform.core.mesh import HEX8_SIDES
from hexaform.io import exodus, report as report_io, vtk
from hexaform.packing.rsa import PackingSpec, pack

BACKENDS = ["scipy"]
try:  # netCDF4 is an optional extra
    import netCDF4  # noqa: F401

    BACKENDS.append("netcdf4")
except ImportError:  # pragma: no cover
    pass


@pytest.fixture(scope="module")
def mesh():
    grid = CartesianGrid.cube(1.0, 12)
    res = pack(grid, PackingSpec(n_particles=3, radius_mean=0.12),
               np.random.default_rng(4))
    return assemble(grid, res.inclusions)


@pytest.mark.parametrize("backend", BACKENDS)
def test_round_trip_preserves_everything(mesh, tmp_path, backend):
    path = exodus.write(mesh, tmp_path / f"m_{backend}.e", backend=backend)
    check = exodus.round_trip_check(mesh, path)
    assert check["nodes_match"]
    assert check["elements_match"]
    assert check["blocks_match"]
    assert check["max_coordinate_error"] == 0.0
    assert all(check["side_sets_match"].values())
    assert all(check["node_sets_match"].values())


@pytest.mark.parametrize("backend", BACKENDS)
def test_connectivity_survives_the_round_trip(mesh, tmp_path, backend):
    path = exodus.write(mesh, tmp_path / f"c_{backend}.e", backend=backend)
    back = exodus.read(path)
    a = np.unique(np.sort(mesh.cells, axis=1), axis=0)
    b = np.unique(np.sort(back.cells, axis=1), axis=0)
    assert np.array_equal(a, b)


def test_side_set_indices_are_remapped_to_file_element_order(mesh, tmp_path):
    """Exodus groups elements by block; side sets must follow that reordering."""
    path = exodus.write(mesh, tmp_path / "ss.e")
    back = exodus.read(path)
    elems_in, sides_in = mesh.side_sets["interface"]
    elems_out, sides_out = back.side_sets["interface"]
    assert elems_out.size == elems_in.size

    def faces(m, elems, sides):
        return np.unique(
            np.sort(
                np.stack([m.cells[e][list(HEX8_SIDES[s - 1])]
                          for e, s in zip(elems, sides)]),
                axis=1,
            ),
            axis=0,
        )

    assert np.array_equal(faces(mesh, elems_in, sides_in),
                          faces(back, elems_out, sides_out))


def test_written_blocks_are_declared_hex8(mesh, tmp_path):
    path = exodus.write(mesh, tmp_path / "t.e")
    back = exodus.read(path)  # read() raises if any block is not HEX8
    assert back.n_elements == mesh.n_elements


def test_block_names_survive(mesh, tmp_path):
    path = exodus.write(mesh, tmp_path / "n.e")
    back = exodus.read(path)
    assert set(back.block_names.values()) == set(mesh.block_names.values())


def test_vtk_writer_emits_hexahedra(mesh, tmp_path):
    path = vtk.write(mesh, tmp_path / "m.vtk")
    text = path.read_text()
    assert f"POINTS {mesh.n_nodes} double" in text
    assert f"CELLS {mesh.n_elements}" in text
    assert "block_id" in text


def test_campaign_csv_and_summary(tmp_path):
    reports = [
        {"seed": s, "success": s % 4 != 0,
         "packing": {"placed": 5, "volume_fraction": 0.1},
         "mesh": {"n_elements": 1000, "n_nodes": 1200},
         "validation": {"quality": {"min_scaled_jacobian": 0.5,
                                    "mean_scaled_jacobian": 0.9,
                                    "max_edge_ratio": 4.0}},
         "timings": {"total": 1.0}}
        for s in range(8)
    ]
    csv_path = report_io.write_campaign_csv(tmp_path / "c.csv", reports)
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 9
    summary = report_io.campaign_summary(reports)
    assert summary["n_realisations"] == 8
    assert summary["success_rate"] == pytest.approx(0.75)
