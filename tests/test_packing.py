import numpy as np
import pytest

from hexaform.background.cartesian import CartesianGrid, OccupancyMap
from hexaform.packing.rsa import PackingSpec, box_cells_for_radius, pack, sample_radii


def test_macroboxes_never_overlap_and_respect_the_gap():
    grid = CartesianGrid.cube(1.0, 32)
    rng = np.random.default_rng(2)
    spec = PackingSpec(n_particles=15, radius_mean=0.07, radius_std=0.015, gap_cells=2)
    res = pack(grid, spec, rng)
    boxes = [(inc.box_lo, inc.k) for inc in res.inclusions]
    for i, (lo_a, k_a) in enumerate(boxes):
        for lo_b, k_b in boxes[i + 1:]:
            sep = np.maximum(np.maximum(lo_a - (lo_b + k_b), lo_b - (lo_a + k_a)), 0)
            assert sep.max() >= spec.gap_cells, "macro-boxes are closer than the gap"


def test_inclusions_stay_inside_their_macroboxes():
    grid = CartesianGrid.cube(1.0, 32)
    res = pack(
        grid,
        PackingSpec(n_particles=10, radius_mean=0.07, radius_std=0.02),
        np.random.default_rng(9),
    )
    for inc in res.inclusions:
        lo_b, hi_b = inc.shape.bounding_box()
        box_lo = inc.box_lo * grid.h
        box_hi = box_lo + inc.k * grid.h
        assert np.all(lo_b >= box_lo - 1e-12), "body pierces its own macro-box"
        assert np.all(hi_b <= box_hi + 1e-12), "body pierces its own macro-box"


def test_boxes_stay_inside_the_domain():
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=8, radius_mean=0.08, wall_cells=1)
    res = pack(grid, spec, np.random.default_rng(1))
    div = np.array(grid.divisions)
    for inc in res.inclusions:
        assert np.all(inc.box_lo >= spec.wall_cells)
        assert np.all(inc.box_lo + inc.k <= div - spec.wall_cells)


def test_same_seed_gives_an_identical_packing():
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=8, radius_mean=0.08, radius_std=0.02)
    a = pack(grid, spec, np.random.default_rng(42))
    b = pack(grid, spec, np.random.default_rng(42))
    assert a.count == b.count
    for x, y in zip(a.inclusions, b.inclusions):
        assert np.array_equal(x.box_lo, y.box_lo)
        assert x.shape.bounding_radius() == y.shape.bounding_radius()


def test_different_seeds_give_different_packings():
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=8, radius_mean=0.08, radius_std=0.02)
    a = pack(grid, spec, np.random.default_rng(1))
    b = pack(grid, spec, np.random.default_rng(2))
    assert not np.array_equal(
        np.array([i.box_lo for i in a.inclusions]),
        np.array([i.box_lo for i in b.inclusions]),
    )


def test_radius_sampling_respects_truncation():
    spec = PackingSpec(n_particles=1, radius_mean=0.08, radius_std=0.02,
                       radius_min=0.06, radius_max=0.10)
    r = sample_radii(2000, spec, h=0.01, rng=np.random.default_rng(0))
    assert r.min() >= 0.06 - 1e-12 and r.max() <= 0.10 + 1e-12
    assert 0.075 < r.mean() < 0.085


def test_unresolvably_small_radii_are_lifted_to_the_floor():
    spec = PackingSpec(n_particles=1, radius_mean=0.001, radius_std=0.0,
                       min_radius_cells=1.5)
    r = sample_radii(50, spec, h=0.05, rng=np.random.default_rng(0))
    assert r.min() >= 1.5 * 0.05 - 1e-12


def test_random_particle_count_range():
    grid = CartesianGrid.cube(1.0, 32)
    counts = {
        pack(grid, PackingSpec(n_particles=(4, 9), radius_mean=0.06),
             np.random.default_rng(s)).requested
        for s in range(25)
    }
    assert counts and min(counts) >= 4 and max(counts) <= 9
    assert len(counts) > 1, "the range should actually vary between seeds"


def test_shortfall_can_be_made_fatal():
    grid = CartesianGrid.cube(1.0, 16)
    spec = PackingSpec(n_particles=400, radius_mean=0.12,
                       max_tries_per_particle=50, allow_shortfall=False)
    with pytest.raises(RuntimeError):
        pack(grid, spec, np.random.default_rng(0))


def test_box_sizing_is_monotone_in_radius():
    ks = [box_cells_for_radius(r, 0.02, 1.35) for r in np.linspace(0.03, 0.2, 20)]
    assert all(b >= a for a, b in zip(ks, ks[1:]))


def test_occupancy_map_rejects_overlaps():
    om = OccupancyMap((10, 10, 10))
    assert om.fits(np.array([0, 0, 0]), 4)
    om.place(np.array([0, 0, 0]), 4)
    assert not om.fits(np.array([2, 2, 2]), 4)
    assert om.fits(np.array([6, 6, 6]), 4)
    assert not om.fits(np.array([8, 8, 8]), 4)  # runs off the end
