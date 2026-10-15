"""Flat dog-bone tensile bars with a gage thickness profile (``hexaform.dogbone``)."""

import numpy as np
import pytest

from hexaform.cli import main, parse_stations
from hexaform.core.topology import check_conformity, signed_volumes
from hexaform.core.validation import validate
from hexaform.dogbone import (
    DogboneOptions,
    dogbone_mesh,
    dogbone_summary,
    half_width,
    thickness,
)

STATIONS = [(0.0, 0.3399), (2.0, 0.3441), (4.0, 0.3540), (6.0, 0.3625), (8.0, 0.3715)]


def test_planform_matches_the_drawing():
    o = DogboneOptions()
    x0 = 0.5 * (o.length - o.gage_length)
    assert half_width(o, np.array([0.0, x0 - o.fillet_extent - 1e-6])) == pytest.approx([2.4, 2.4])
    assert half_width(o, np.array([x0, x0 + 4.0, x0 + 8.0])) == pytest.approx([0.925] * 3)
    # the fillet is tangent to the gage edge and reaches the grip width at its end
    assert half_width(o, np.array([x0 - 1e-9]))[0] == pytest.approx(0.925, abs=1e-6)
    assert half_width(o, np.array([x0 - o.fillet_extent]))[0] == pytest.approx(2.4, abs=1e-9)


def test_thickness_profile_interpolates_the_stations_and_blends_to_the_grips():
    o = DogboneOptions(stations=STATIONS, grip_thickness=0.35)
    x0 = 0.5 * (o.length - o.gage_length)
    t = thickness(o, np.array([x0 + s for s, _ in STATIONS]))
    assert t == pytest.approx([v for _, v in STATIONS], abs=1e-12)
    assert thickness(o, np.array([0.0, o.length]))[0] == pytest.approx(0.35)
    mid = x0 - 0.5 * o.fillet_extent
    assert thickness(o, np.array([mid]))[0] == pytest.approx(0.5 * (0.35 + 0.3399), abs=1e-12)


def test_mesh_is_valid_conforming_and_carries_the_sets():
    m = dogbone_mesh(DogboneOptions(stations=STATIONS, h=0.2))
    assert np.all(signed_volumes(m.points, m.cells) > 0)
    assert check_conformity(m)["conforming"]
    assert validate(m, cell_size=0.2).passed
    ny = round(1.85 / 0.2)
    for name in ("x-", "x+", "gage-", "gage+"):
        assert len(m.node_sets[name]) == (ny + 1) * 2
    assert len(m.node_sets["grips"]) == 2 * (ny + 1) * 2
    assert len(m.node_sets["station_3"]) == (ny + 1) * 2
    assert set(m.block_names.values()) == {"gage", "shoulder", "grip"}
    s = dogbone_summary(m)
    assert s["gage_cells"] == 40 and s["divisions"][1] == ny
    assert s["thickness_range"] == pytest.approx([0.3399, 0.3715], abs=1e-9)


def test_two_layers_through_the_thickness():
    m = dogbone_mesh(DogboneOptions(stations=STATIONS, h=0.4, n_z=2))
    assert m.metadata["divisions"][2] == 2
    assert validate(m, cell_size=0.4).passed


def test_rejects_a_gage_that_does_not_fit():
    with pytest.raises(ValueError):
        DogboneOptions(length=10.0, gage_length=8.0)
    with pytest.raises(ValueError):
        DogboneOptions(gage_width=5.0)


def test_cli_writes_exodus(tmp_path):
    out = tmp_path / "bar.e"
    rc = main(["dogbone", "--h", "0.4", "--stations", "0:0.34,4:0.35,8:0.36", "-o", str(out), "--summary", str(tmp_path / "s.json")])
    assert rc == 0 and out.exists() and (tmp_path / "s.json").exists()
    assert parse_stations("0:0.34,8:0.36") == [(0.0, 0.34), (8.0, 0.36)]
