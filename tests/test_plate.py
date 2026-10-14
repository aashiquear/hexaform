"""Plates with a conforming through-hole (``hexaform.plate``)."""

import numpy as np
import pytest

from hexaform.cli import main, parse_hole
from hexaform.core.topology import check_conformity, signed_volumes
from hexaform.core.validation import validate
from hexaform.io import exodus
from hexaform.plate import HoleSpec, PlateOptions, _auto_layers, plate_mesh, plate_summary


def _hole_radius_error(mesh, hole, cx, cy):
    p = mesh.points[mesh.node_sets["hole"]]
    theta = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
    return np.max(np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - hole.radius(theta)))


def test_plain_plate_is_the_background_grid():
    m = plate_mesh(PlateOptions(width=100, height=50, thickness=1, h=2.5))
    assert (m.n_elements, m.n_nodes) == (800, 41 * 21 * 2)
    assert np.all(signed_volumes(m.points, m.cells) > 0)
    assert "hole" not in m.node_sets
    assert len(m.node_sets["x-"]) == 21 * 2 and len(m.side_sets["z-"][0]) == 800
    assert len(m.node_sets["grips"]) == 2 * 21 * 2 and len(m.node_sets["outer"]) == 2 * (41 + 21) * 2 - 8
    assert validate(m, cell_size=2.5).passed


@pytest.mark.parametrize("hole", [HoleSpec("circle", 2.5), HoleSpec("ellipse", 3.75, 1.25), HoleSpec("ellipse", 3.75, 1.25, angle=30.0)])
def test_hole_is_conforming_and_valid(hole):
    m = plate_mesh(PlateOptions(h=2.5, hole=hole))
    assert np.all(signed_volumes(m.points, m.cells) > 0)
    assert check_conformity(m)["conforming"]
    assert validate(m, cell_size=2.5).passed
    # every hole node lies on the analytic curve, both z layers
    assert _hole_radius_error(m, hole, 50.0, 25.0) < 1e-9
    n_theta = m.metadata["hole"]["n_theta"]
    assert len(m.node_sets["hole"]) == 2 * n_theta
    assert len(m.side_sets["hole"][0]) == n_theta
    # the ring is block 2 and the hole faces belong to it
    assert set(m.block_id[m.side_sets["hole"][0]]) == {2}
    # background outside the ring is untouched: the plate's element count is
    # the grid minus the rectangle plus the ring
    kx, ky = m.metadata["hole"]["transition_cells"]
    assert int(np.sum(m.block_id == 1)) == 40 * 20 - (2 * kx) * (2 * ky)


def test_ring_layers_grade_from_the_hole_arc_to_h():
    m = plate_mesh(PlateOptions(h=2.5, hole=HoleSpec("circle", 2.5)))
    hm = m.metadata["hole"]
    assert hm["first_layer"] == pytest.approx(hm["arc_at_hole"], rel=1e-6)
    # the outermost layer is about one background cell thick
    ring = m.cells[m.block_id == 2]
    s = plate_summary(m)
    assert s["n_ring_elements"] == ring.shape[0] == hm["n_r"] * hm["n_theta"]
    n, g = _auto_layers(10.0, 0.5, 2.5)
    r = g ** (1.0 / (n - 1))
    assert 0.5 * (r ** n - 1) / (r - 1) == pytest.approx(10.0, rel=1e-6)
    assert 0.5 * g == pytest.approx(2.5, rel=0.5)


def test_explicit_layers_and_thickness_and_origin():
    m = plate_mesh(PlateOptions(h=2.5, thickness=0.5, origin=(1, 2, 3), hole=HoleSpec("circle", 2.5), n_r=3, grading=1.0, transition=(4, 4)))
    assert m.metadata["hole"]["n_r"] == 3 and m.metadata["hole"]["n_theta"] == 32
    lo, hi = m.points.min(0), m.points.max(0)
    assert np.allclose(lo, [1, 2, 3]) and np.allclose(hi, [101, 52, 3.5])
    assert _hole_radius_error(m, HoleSpec("circle", 2.5), 51.0, 27.0) < 1e-9


def test_rejects_misaligned_and_oversized_holes():
    with pytest.raises(ValueError, match="whole multiple"):
        plate_mesh(PlateOptions(width=101, h=2.5))
    with pytest.raises(ValueError, match="background grid"):
        plate_mesh(PlateOptions(h=2.5, hole=HoleSpec("circle", 2.5, centre=(51.0, 25.0))))
    with pytest.raises(ValueError, match="does not fit"):
        plate_mesh(PlateOptions(h=2.5, hole=HoleSpec("circle", 2.5), transition=(1, 1)))
    with pytest.raises(ValueError, match="leaves the plate"):
        plate_mesh(PlateOptions(h=2.5, hole=HoleSpec("circle", 2.5, centre=(5.0, 25.0))))
    with pytest.raises(ValueError):
        HoleSpec("square", 1.0)


def test_cli_plate_writes_exodus_and_summary(tmp_path):
    out = tmp_path / "plate.e"
    rc = main(["plate", "--hole", "ellipse:7.5,2.5,30", "--h", "2.5", "-o", str(out), "--write-vtk", "--summary", str(tmp_path / "s.json")])
    assert rc == 0 and out.exists() and out.with_suffix(".vtk").exists() and (tmp_path / "s.json").exists()
    m = exodus.read(out)
    assert "hole" in m.node_sets and m.n_elements > 800
    assert parse_hole("circle:5").a == 2.5
    h = parse_hole("ellipse:7.5,2.5,30")
    assert (h.a, h.b, h.angle) == (3.75, 1.25, 30.0)
    assert parse_hole(None) is None
