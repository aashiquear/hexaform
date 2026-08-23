"""Anisotropic macro-boxes: per-axis cubed-sphere resolution and box sizing.

These are the contract tests for roadmap item 1: a ``kx x ky x kz`` macro-box
whose cubed-sphere patch resolution matches the box dimensions, so elongated
bodies stop wasting elements on their short axes.
"""

import numpy as np
import pytest

from hexaform.background.cartesian import CartesianGrid, OccupancyMap
from hexaform.core.quality import scaled_jacobian
from hexaform.core.topology import check_conformity, signed_volumes
from hexaform.assembly.builder import assemble
from hexaform.packing.rsa import (
    PackingSpec,
    box_cells_for_radius,
    box_cells_for_shape,
    pack,
)
from hexaform.shapes.base import StarShape
from hexaform.shapes.library import Ellipsoid, Sphere, SphericalHarmonicGrain, Superquadric
from hexaform.templates.cubed_sphere import CubedSphere
from hexaform.templates.macroblock import MacroBlockOptions, build_macroblock

H = 0.025
ORIGIN = np.zeros(3)


def _n_surf(n1, n2, n3):
    a, b, c = n1 + 1, n2 + 1, n3 + 1
    return 2 * (a * b + b * c + c * a) - 4 * (a + b + c) + 8


def _n_quads(n1, n2, n3):
    return 2 * (n1 * n2 + n2 * n3 + n3 * n1)


def _centred_lo(k):
    return -np.ceil(np.asarray(k) / 2).astype(np.int64)


# ----------------------------------------------------------------------
# cubed sphere with per-axis resolution
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [(1, 2, 3), (3, 5, 4), (4, 4, 8), (7, 3, 9)])
def test_anisotropic_node_and_quad_counts_match_closed_form(n):
    cs = CubedSphere(n)
    check = cs.check()
    assert check["nodes_ok"] and cs.n_nodes == _n_surf(*n)
    assert check["quads_ok"] and cs.n_quads == _n_quads(*n)


@pytest.mark.parametrize("n", [(2, 3, 4), (4, 4, 8), (6, 2, 5)])
@pytest.mark.parametrize("mode", ["uniform", "equiangular", "blend"])
def test_anisotropic_directions_unit_and_quads_outward(n, mode):
    cs = CubedSphere(n, spacing=mode, blend=0.6)
    assert np.allclose(np.linalg.norm(cs.directions, axis=1), 1.0)
    check = cs.check()
    assert check["all_outward"], "a patch is wound the wrong way"
    assert check["min_quad_area"] > 0


def test_anisotropic_every_node_belongs_to_a_patch_grid():
    cs = CubedSphere((3, 5, 7))
    seen = np.unique(np.concatenate([cs.index[f].ravel() for f in cs.index]))
    assert seen.size == cs.n_nodes
    assert seen.min() == 0 and seen.max() == cs.n_nodes - 1


def test_anisotropic_shared_edges_are_bitwise_identical():
    """Each shared patch edge uses the 1-D spacing of the axis it runs along."""
    n = (3, 5, 7)
    cs = CubedSphere(n)
    xp, yp, zp = cs.face_cube_points("x+"), cs.face_cube_points("y+"), cs.face_cube_points("z+")
    # x+/y+ share the edge running along z (nz+1 points)
    assert np.array_equal(np.sort(xp[-1, :, :], axis=0), np.sort(yp[:, -1, :], axis=0))
    # x+/z+ share the edge running along y (ny+1 points)
    assert np.array_equal(np.sort(xp[:, -1, :], axis=0), np.sort(zp[-1, :, :], axis=0))
    # y+/z+ share the edge running along x (nx+1 points)
    assert np.array_equal(np.sort(yp[-1, :, :], axis=0), np.sort(zp[:, -1, :], axis=0))


def test_scalar_and_vector_resolution_agree():
    a = CubedSphere(6)
    b = CubedSphere((6, 6, 6))
    assert a.n_nodes == b.n_nodes and a.n_quads == b.n_quads
    assert np.allclose(np.sort(a.directions, axis=0), np.sort(b.directions, axis=0))


# ----------------------------------------------------------------------
# shape support: bounding box and re-centring
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        Sphere(np.array([0.1, -0.2, 0.3]), 0.05),
        Ellipsoid(np.array([0.1, -0.2, 0.3]), [0.06, 0.04, 0.03]),
        Superquadric(np.array([0.1, -0.2, 0.3]), [0.06, 0.04, 0.03], 3.0, 3.0),
    ],
    ids=["sphere", "ellipsoid", "superquadric"],
)
def test_star_shape_bounding_box_matches_analytic_extent(shape):
    lo, hi = shape.bounding_box()
    c = shape.center
    extent = hi - lo
    # each axis extent is at least 2 * the extent along that axis direction
    axes = [
        shape.support_radius(np.array([[1.0, 0.0, 0.0]]))[0],
        shape.support_radius(np.array([[0.0, 1.0, 0.0]]))[0],
        shape.support_radius(np.array([[0.0, 0.0, 1.0]]))[0],
    ]
    for a in range(3):
        assert extent[a] >= 2 * axes[a] - 1e-12
    # and no axis extent exceeds the body's bounding diameter (up to sampling)
    assert np.all(extent <= 2 * shape.bounding_radius() + 1e-9)
    assert np.all(lo <= c) and np.all(hi >= c)


@pytest.mark.parametrize(
    "shape",
    [
        Sphere(np.array([0.1, -0.2, 0.3]), 0.05),
        Ellipsoid(np.array([0.1, -0.2, 0.3]), [0.06, 0.04, 0.03]),
        Superquadric(np.array([0.1, -0.2, 0.3]), [0.06, 0.04, 0.03], 3.0, 3.0),
        SphericalHarmonicGrain.random(
            np.array([0.1, -0.2, 0.3]), 0.05, np.random.default_rng(5), l_max=3, roughness=0.1
        ),
    ],
    ids=["sphere", "ellipsoid", "superquadric", "grain"],
)
def test_with_center_translates_without_changing_the_surface(shape):
    moved = shape.with_center(np.array([0.7, 0.8, 0.9]))
    assert isinstance(moved, StarShape)
    assert np.allclose(moved.center, [0.7, 0.8, 0.9])
    dirs = np.random.default_rng(1).normal(size=(50, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    assert np.allclose(moved.support_radius(dirs), shape.support_radius(dirs))


# ----------------------------------------------------------------------
# per-axis macro-box sizing
# ----------------------------------------------------------------------
def test_box_cells_for_shape_is_per_axis_and_contains_the_body():
    shape = Ellipsoid(np.zeros(3), [0.10, 0.06, 0.04])
    k = box_cells_for_shape(shape, H, 1.35)
    assert k.shape == (3,) and np.all(k >= 4)
    lo_b, hi_b = shape.bounding_box()
    assert np.all(k * H >= (hi_b - lo_b) + 1e-12), "box too small on an axis"
    # flat body: the short axes must be strictly smaller than the long one
    assert k[0] > k[1] > k[2]


def test_box_cells_for_shape_reduces_to_radius_sizing_for_spheres():
    for r in (0.03, 0.08, 0.15):
        k = box_cells_for_shape(Sphere(np.array([0.5, 0.5, 0.5]), r), H, 1.35)
        expected = box_cells_for_radius(r, H, 1.35)
        assert int(k[0]) == expected and int(k[1]) == expected and int(k[2]) == expected


def test_box_cells_for_shape_is_monotone_in_extent():
    base = [0.10, 0.08, 0.06]
    for a in range(3):
        prev = box_cells_for_shape(Ellipsoid(np.zeros(3), base), H, 1.35)[a]
        for w in np.linspace(base[a], 0.15, 12):
            width = base.copy()
            width[a] = w
            k = box_cells_for_shape(Ellipsoid(np.zeros(3), width), H, 1.35)
            assert k[a] >= prev, f"axis {a} shrank as its extent grew"
            prev = k[a]


# ----------------------------------------------------------------------
# anisotropic macro-block
# ----------------------------------------------------------------------
def _ellipsoid_block(ratio=(1.0, 0.8, 0.65), br=1.35, options=None):
    shape = Ellipsoid(np.zeros(3), [0.1 * r for r in ratio])
    k = box_cells_for_shape(shape, H, br)
    return shape, k, build_macroblock(shape, _centred_lo(k), k, H, ORIGIN, options)


def test_anisotropic_block_no_inverted_elements_and_good_quality():
    shape, k, mb = _ellipsoid_block()
    assert np.all(signed_volumes(mb.points, mb.cells) > 0)
    assert scaled_jacobian(mb.points, mb.cells).min() > 0.2


def test_anisotropic_block_element_and_node_counts_are_as_designed():
    shape, k, mb = _ellipsoid_block(options=MacroBlockOptions(n_inner=2, n_outer=3))
    kx, ky, kz = (int(v) for v in k)
    n_surf = _n_surf(kx, ky, kz)
    nq = _n_quads(kx, ky, kz)
    assert mb.cells.shape[0] == kx * ky * kz + nq * (2 + 3)
    assert mb.points.shape[0] == (kx + 1) * (ky + 1) * (kz + 1) + n_surf * (2 + 3)


def test_anisotropic_macrobox_face_nodes_match_the_background_grid_bitwise():
    shape, k, mb = _ellipsoid_block()
    lo = _centred_lo(k)
    for name, ids in mb.box_face_nodes.items():
        axis = "xyz".index(name[0])
        side = k[axis] if name[1] == "+" else 0
        coords = mb.points[ids]
        expected = ORIGIN[axis] + (lo[axis] + side) * H
        assert np.array_equal(coords[..., axis], np.full(coords.shape[:-1], expected))
        for a in [x for x in range(3) if x != axis]:
            idx = (coords[..., a] - ORIGIN[a]) / H
            assert np.abs(idx - np.rint(idx)).max() < 1e-12


def test_anisotropic_surface_nodes_lie_on_the_analytic_surface():
    shape, k, mb = _ellipsoid_block()
    assert np.abs(shape.radial_scale(mb.points[mb.surface_nodes]) - 1.0).max() < 1e-12


def test_anisotropic_inclusion_volume_is_accurate():
    shape, k, mb = _ellipsoid_block()
    vol = signed_volumes(mb.points, mb.cells)[mb.is_inclusion].sum()
    exact = 4 / 3 * np.pi * np.prod(shape.semi_axes)
    assert abs(vol - exact) / exact < 0.05


def test_anisotropic_box_uses_fewer_elements_than_an_equivalent_cube():
    """The value proposition: the short axes cost nothing for their margin."""
    shape = Ellipsoid(np.zeros(3), [0.10, 0.06, 0.04])
    k = box_cells_for_shape(shape, H, 1.35)
    k_cube = np.full(3, k.max())
    assert not np.array_equal(k, k_cube), "this body should be anisotropic"
    aniso = build_macroblock(shape, _centred_lo(k), k, H, ORIGIN)
    cube = build_macroblock(shape, _centred_lo(k_cube), k_cube, H, ORIGIN)
    assert aniso.cells.shape[0] < cube.cells.shape[0]
    # the savings are real, not a rounding artifact
    assert cube.cells.shape[0] / aniso.cells.shape[0] > 1.15


def test_single_anisotropic_inclusion_assembles_conformingly():
    grid = CartesianGrid.cube(1.0, 24)
    shape = Ellipsoid(np.zeros(3), [0.10, 0.06, 0.04])
    k = box_cells_for_shape(shape, grid.h, 1.35)
    lo = np.array([10, 10, 10], dtype=np.int64) - k // 2
    centre = grid.origin + (lo + k / 2.0) * grid.h
    shape = shape.with_center(centre)
    from hexaform.packing.rsa import PackedInclusion

    res = pack(grid, PackingSpec(n_particles=0, radius_mean=0.01), np.random.default_rng(0))
    res.inclusions.append(PackedInclusion(shape, lo, k, 1))
    mesh = assemble(grid, res.inclusions)
    report = check_conformity(mesh)
    assert report["conforming"], report
    total = signed_volumes(mesh.points, mesh.cells).sum()
    assert total == pytest.approx(float(np.prod(grid.extent)), rel=1e-12)


# ----------------------------------------------------------------------
# packer with per-axis boxes
# ----------------------------------------------------------------------
def _ellipsoid_factory(ratio):
    r = np.asarray(ratio, dtype=float)
    return lambda c, rad, _rng: Ellipsoid(c, rad * r)


def test_pack_ellipsoids_uses_per_axis_boxes_and_still_meshable():
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=8, radius_mean=0.08, radius_std=0.015, gap_cells=2)
    res = pack(grid, spec, np.random.default_rng(3), shape_factory=_ellipsoid_factory((1.0, 0.7, 0.5)))
    assert res.count > 0
    for inc in res.inclusions:
        assert inc.k.shape == (3,)
        # the flattened body must not pay cube prices on every axis
        assert inc.k.min() < inc.k.max()
        lo_b, hi_b = inc.shape.bounding_box()
        box_lo = grid.origin + inc.box_lo * grid.h
        box_hi = box_lo + inc.k * grid.h
        assert np.all(lo_b >= box_lo - 1e-12) and np.all(hi_b <= box_hi + 1e-12)
    # boxes still respect the integer gap and the walls
    div = np.array(grid.divisions)
    boxes = [(inc.box_lo, inc.k) for inc in res.inclusions]
    for i, (lo_a, k_a) in enumerate(boxes):
        assert np.all(lo_a >= spec.wall_cells)
        assert np.all(lo_a + k_a <= div - spec.wall_cells)
        for lo_b, k_b in boxes[i + 1 :]:
            sep = np.maximum(np.maximum(lo_a - (lo_b + k_b), lo_b - (lo_a + k_a)), 0)
            assert sep.max() >= spec.gap_cells


def test_pack_spheres_degenerates_to_the_classic_cube_sizing():
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=6, radius_mean=0.08, radius_std=0.01)
    res = pack(grid, spec, np.random.default_rng(7))
    for inc in res.inclusions:
        assert np.all(np.diff(inc.k) == 0), "a sphere's macro-box must stay a cube"
        assert inc.k[0] == box_cells_for_radius(inc.shape.bounding_radius(), grid.h, spec.box_ratio)


def test_pack_is_reproducible_with_anisotropic_boxes():
    grid = CartesianGrid.cube(1.0, 24)
    spec = PackingSpec(n_particles=6, radius_mean=0.08, radius_std=0.015)
    factory = _ellipsoid_factory((1.0, 0.7, 0.5))
    a = pack(grid, spec, np.random.default_rng(42), shape_factory=factory)
    b = pack(grid, spec, np.random.default_rng(42), shape_factory=factory)
    assert a.count == b.count
    for x, y in zip(a.inclusions, b.inclusions):
        assert np.array_equal(x.box_lo, y.box_lo)
        assert np.array_equal(x.k, y.k)
        assert np.allclose(x.shape.center, y.shape.center)


def test_occupancy_map_accepts_vector_box_edges():
    om = OccupancyMap((10, 10, 10))
    k = np.array([4, 4, 4])
    assert om.fits(np.array([0, 0, 0]), k)
    om.place(np.array([0, 0, 0]), k)
    assert not om.fits(np.array([2, 2, 2]), k)
    # a box that overlaps the first one on one axis does not fit; one that
    # merely touches it does (no margin requested)
    assert not om.fits(np.array([3, 0, 0]), k)
    assert om.fits(np.array([4, 0, 0]), k)
    assert om.fits(np.array([0, 4, 4]), np.array([4, 4, 4]))
    assert om.fits(np.array([0, 8, 8]), np.array([2, 2, 2]))
    assert not om.fits(np.array([9, 9, 9]), np.array([2, 2, 2]))  # runs off the end
    # scalar and vector forms agree
    om2 = OccupancyMap((10, 10, 10))
    assert om2.fits(np.array([6, 6, 6]), 4) == om2.fits(np.array([6, 6, 6]), k)


def test_background_carving_accepts_vector_box_edges():
    grid = CartesianGrid.cube(1.0, 16)
    k = np.array([4, 2, 2])
    pts, cells, idx = grid.build([(np.array([2, 3, 4]), k)])
    total = grid.divisions[0] * grid.divisions[1] * grid.divisions[2]
    assert len(idx) == total - 4 * 2 * 2
