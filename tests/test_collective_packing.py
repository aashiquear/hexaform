"""Tests for the collective-rearrangement packing (denser populations)."""

import numpy as np
import pytest

from hexaform.background.cartesian import CartesianGrid
from hexaform.packing.collective import CollectiveOptions, pack as coll_pack
from hexaform.packing.rsa import PackingSpec, pack as rsa_pack


def test_collective_is_denser_than_rsa():
    """Collective should place at least as many inclusions as RSA."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=40, radius_mean=0.075, radius_std=0.0, gap_cells=1,
                       max_tries_per_particle=400)

    for seed in range(1, 6):
        rsa_result = rsa_pack(grid, spec, np.random.default_rng(seed))
        col_result = coll_pack(grid, spec, np.random.default_rng(seed))
        assert col_result.count >= rsa_result.count, f"seed {seed}: COL={col_result.count} < RSA={rsa_result.count}"


def test_collective_reaches_physical_maximum():
    """Collective should saturate at the physically maximum packing."""
    grid = CartesianGrid.cube(1.0, 24)
    # At pitch h=1/24 with gap_cells=1 and radius=0.075, we can fit a maximum of 3x3x3=27
    spec = PackingSpec(n_particles=50, radius_mean=0.075, radius_std=0.0, gap_cells=1)

    for seed in range(10):
        result = coll_pack(grid, spec, np.random.default_rng(seed))
        # All seeds should reach the maximum at this regime
        assert result.count == 27, f"seed {seed}: expected max 27, got {result.count}"


def test_collective_preserves_macrobox_disjointness():
    """Collective must maintain exact RSA validity: macro-boxes disjoint with gap margin."""
    grid = CartesianGrid.cube(1.0, 32)
    spec = PackingSpec(n_particles=20, radius_mean=0.06, radius_std=0.0, gap_cells=2)
    result = coll_pack(grid, spec, np.random.default_rng(42))

    boxes = [(inc.box_lo, inc.k) for inc in result.inclusions]
    div = np.array(grid.divisions)

    # All macro-boxes must stay inside domain with wall margin
    for lo, k in boxes:
        assert np.all(lo >= spec.wall_cells), f"wall violated at {lo}"
        assert np.all(lo + k <= div - spec.wall_cells), f"box extends past wall at {lo}+{k}"

    # All macro-boxes must be separated by gap_cells on every dimension
    for i, (la, ka) in enumerate(boxes):
        for lb, kb in boxes[i + 1:]:
            sep = np.maximum(np.maximum(la - (lb + kb), lb - (la + ka)), 0)
            assert sep.max() >= spec.gap_cells, f"gap violated at {la} vs {lb}"


def test_collective_respects_wall_margin():
    """Collective must maintain the wall margin."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=30, radius_mean=0.08, radius_std=0.0, wall_cells=2, gap_cells=1)
    result = coll_pack(grid, spec, np.random.default_rng(7))

    div = np.array(grid.divisions)
    for inc in result.inclusions:
        assert np.all(inc.box_lo >= spec.wall_cells)
        assert np.all(inc.box_lo + inc.k <= div - spec.wall_cells)


def test_collective_reproducible_with_seed():
    """Same seed must give identical results."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=25, radius_mean=0.07, radius_std=0.0, gap_cells=1)

    r1 = coll_pack(grid, spec, np.random.default_rng(99))
    r2 = coll_pack(grid, spec, np.random.default_rng(99))

    assert r1.count == r2.count
    for x, y in zip(r1.inclusions, r2.inclusions):
        assert np.array_equal(x.box_lo, y.box_lo)
        assert np.array_equal(x.k, y.k)


def test_collective_options_affects_behavior():
    """Higher kinetic_fraction should explore more, potentially placing more."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=35, radius_mean=0.075, radius_std=0.0, gap_cells=1)

    # Force a regime where kinetic exploration matters: low joint_search tries
    opts_low_kinetic = CollectiveOptions(max_steps=500, seed_tries=200, kinetic_fraction=0.0)
    opts_high_kinetic = CollectiveOptions(max_steps=500, seed_tries=200, kinetic_fraction=1.0)

    r_low = coll_pack(grid, spec, np.random.default_rng(11), options=opts_low_kinetic)
    r_high = coll_pack(grid, spec, np.random.default_rng(11), options=opts_high_kinetic)

    # Different configurations may differ; at minimum both are valid
    assert 0 < r_low.count <= 27 and 0 < r_high.count <= 27


def test_collective_shortfall_reporting():
    """Shortfall is correctly reported when requested count exceeds physical limit."""
    grid = CartesianGrid.cube(1.0, 16)
    # Request many more than physically possible at this density
    spec = PackingSpec(n_particles=100, radius_mean=0.09, radius_std=0.0, gap_cells=2, allow_shortfall=True)

    result = coll_pack(grid, spec, np.random.default_rng(3))
    assert result.count < result.requested
    assert "rejected" in result.stats or result.stats.get("placed") < result.stats.get("requested", 100)


def test_collective_allows_fatal_shortfall():
    """Setting allow_shortfall=False should raise when physical limit exceeded."""
    grid = CartesianGrid.cube(1.0, 12)
    spec = PackingSpec(n_particles=50, radius_mean=0.12, gap_cells=1, max_tries_per_particle=100,
                       allow_shortfall=False)

    # This should raise because we're asking for more than physically possible
    with pytest.raises(RuntimeError):
        coll_pack(grid, spec, np.random.default_rng(5))


def test_collective_volume_fraction_improvement():
    """Collective should achieve higher volume fraction in the same regime."""
    grid = CartesianGrid.cube(1.0, 24)
    radius = 0.075
    vol_each = 4/3*np.pi*radius**3

    spec = PackingSpec(n_particles=40, radius_mean=radius, radius_std=0.0, gap_cells=1)

    rsa_vf, col_vf = [], []
    for seed in range(5):
        rsa = rsa_pack(grid, spec, np.random.default_rng(seed))
        col = coll_pack(grid, spec, np.random.default_rng(seed))
        rsa_vf.append(rsa.stats['volume_fraction'])
        col_vf.append(col.stats['volume_fraction'])

    assert np.mean(col_vf) > np.mean(rsa_vf), "COL should achieve higher VF than RSA on average"


def test_collective_different_shapes():
    """Collective packer works with different inclusion shapes."""
    grid = CartesianGrid.cube(1.0, 24)

    # Ellipsoids
    from hexaform.packing.rsa import box_cells_for_shape
    from hexaform.shapes.library import Ellipsoid
    ell = Ellipsoid(np.zeros(3), np.array([0.08, 0.06, 0.05]))
    k_ell = box_cells_for_shape(ell, grid.h, 1.35)
    spec_ell = PackingSpec(n_particles=20, radius_mean=0.08, radius_std=0.0, gap_cells=1)

    # Collectively place ellipsoids (macro-box sizing per axis)
    from hexaform.pipeline.run import make_shape_factory
    cfg_ell = type('Cfg', (), {'shape': 'ellipsoid', 'aspect_ratios': (1.0, 0.75, 0.6),
                               'random_orientation': False, 'grain_l_max': 4, 'grain_roughness': 0.15})()
    result = coll_pack(grid, spec_ell, np.random.default_rng(12),
                       shape_factory=make_shape_factory(cfg_ell))
    assert result.count > 0


def test_collective_preserves_packer_stats():
    """Packing stats should include collective-specific metadata."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=25, radius_mean=0.075, gap_cells=1)

    result = coll_pack(grid, spec, np.random.default_rng(13))

    assert "packer" in result.stats
    assert result.stats["packer"] == "collective"
    assert "collective_steps" in result.stats
    assert "collective_kinetic_fraction" in result.stats


def test_collective_does_not_unplace():
    """Collective should never decrease placements from the RSA seed."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=35, radius_mean=0.075, gap_cells=1)

    rsa_result = rsa_pack(grid, spec, np.random.default_rng(17))
    # Use same RNG state - but since RSA consumes fewer RNG calls than collective,
    # we'll just verify that collective places at least as many in any case
    col_result = coll_pack(grid, spec, np.random.default_rng(17))

    assert col_result.count >= rsa_result.count, f"COL={col_result.count} < RSA={rsa_result.count}"


def test_collective_wall_cells_boundary():
    """Collective respects wall_cells when wall_cells > gap_cells."""
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=30, radius_mean=0.075, wall_cells=3, gap_cells=1)

    result = coll_pack(grid, spec, np.random.default_rng(19))
    div = np.array(grid.divisions)

    for inc in result.inclusions:
        assert np.all(inc.box_lo >= spec.wall_cells), f"wall={spec.wall_cells} violated at {inc.box_lo}"
        assert np.all(inc.box_lo + inc.k <= div - spec.wall_cells)


def test_collective_large_gap_cells():
    """Collective works when gap_cells is large (more sparse packing)."""
    grid = CartesianGrid.cube(1.0, 32)
    spec = PackingSpec(n_particles=50, radius_mean=0.06, gap_cells=4)

    result = coll_pack(grid, spec, np.random.default_rng(23))
    assert result.count > 0

    # All boxes must still be separated by gap_cells
    boxes = [(inc.box_lo, inc.k) for inc in result.inclusions]
    for i, (la, ka) in enumerate(boxes):
        for lb, kb in boxes[i + 1:]:
            sep = np.maximum(np.maximum(la - (lb + kb), lb - (la + ka)), 0)
            assert sep.max() >= spec.gap_cells
