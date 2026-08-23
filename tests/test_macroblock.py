"""The macro-block is the core algorithm; these are its contract tests."""

import numpy as np
import pytest

from hexaform.core.quality import edge_ratio, scaled_jacobian
from hexaform.core.topology import signed_volumes
from hexaform.shapes.library import (
    Ellipsoid,
    Sphere,
    SphericalHarmonicGrain,
    Superquadric,
)
from hexaform.templates.macroblock import MacroBlockOptions, build_macroblock

H = 0.025
ORIGIN = np.zeros(3)


def _block(shape, k=8, options=None):
    lo = np.zeros(3, dtype=int)
    return build_macroblock(shape, lo, k, H, ORIGIN, options)


def _centred_sphere(k=8, ratio=1.4):
    centre = np.array([k, k, k]) / 2 * H
    return Sphere(centre, (k / 2 * H) / ratio), k


@pytest.mark.parametrize("k", [4, 6, 8, 12])
def test_no_inverted_elements(k):
    sphere, k = _centred_sphere(k)
    mb = _block(sphere, k)
    assert np.all(signed_volumes(mb.points, mb.cells) > 0)
    assert scaled_jacobian(mb.points, mb.cells).min() > 0.5


@pytest.mark.parametrize("k", [4, 8, 12])
def test_element_and_node_counts_are_as_designed(k):
    sphere, k = _centred_sphere(k)
    opt = MacroBlockOptions(n_inner=2, n_outer=3)
    mb = _block(sphere, k, opt)
    assert mb.cells.shape[0] == k**3 + 6 * k * k * (opt.n_inner + opt.n_outer)
    assert mb.points.shape[0] == (k + 1) ** 3 + (opt.n_inner + opt.n_outer) * (6 * k**2 + 2)


def test_interface_nodes_lie_on_the_analytic_surface_to_machine_precision():
    sphere, k = _centred_sphere(8)
    mb = _block(sphere, k)
    r = np.linalg.norm(mb.points[mb.surface_nodes] - sphere.center, axis=1)
    assert np.abs(r - sphere.radius).max() < 1e-14


def test_macrobox_face_nodes_match_the_background_grid_bitwise():
    """This exactness is what makes the global weld free of hanging nodes."""
    sphere, k = _centred_sphere(8)
    mb = _block(sphere, k)
    lo = np.zeros(3, dtype=int)
    for name, ids in mb.box_face_nodes.items():
        axis = "xyz".index(name[0])
        side = k if name[1] == "+" else 0
        coords = mb.points[ids]
        expected = ORIGIN[axis] + (lo[axis] + side) * H
        assert np.array_equal(coords[..., axis], np.full(coords.shape[:-1], expected))
        # in-plane coordinates must be exact multiples of h off the grid origin
        others = [a for a in range(3) if a != axis]
        for a in others:
            idx = (coords[..., a] - ORIGIN[a]) / H
            assert np.abs(idx - np.rint(idx)).max() < 1e-12


@pytest.mark.parametrize("k,tol", [(4, 0.08), (8, 0.02), (16, 0.005)])
def test_inclusion_volume_converges_second_order(k, tol):
    sphere, k = _centred_sphere(k)
    mb = _block(sphere, k)
    vol = signed_volumes(mb.points, mb.cells)[mb.is_inclusion].sum()
    exact = 4 / 3 * np.pi * sphere.radius**3
    assert abs(vol - exact) / exact < tol


def test_volume_error_halves_by_four_when_k_doubles():
    errs = []
    for k in (6, 12, 24):
        sphere, k = _centred_sphere(k)
        mb = _block(sphere, k)
        vol = signed_volumes(mb.points, mb.cells)[mb.is_inclusion].sum()
        exact = 4 / 3 * np.pi * sphere.radius**3
        errs.append(abs(vol - exact) / exact)
    for a, b in zip(errs, errs[1:]):
        assert 3.0 < a / b < 5.0, f"expected ~4x error reduction, got {a / b:.2f}"


def test_interface_side_set_faces_lie_on_the_surface():
    sphere, k = _centred_sphere(8)
    mb = _block(sphere, k)
    from hexaform.core.mesh import HEX8_SIDES

    assert mb.interface_cells.size == 6 * k * k
    for elem, side in zip(mb.interface_cells[:64], mb.interface_sides[:64]):
        nodes = mb.cells[elem][list(HEX8_SIDES[side - 1])]
        r = np.linalg.norm(mb.points[nodes] - sphere.center, axis=1)
        assert np.abs(r - sphere.radius).max() < 1e-12


def test_interface_elements_are_matrix_side():
    sphere, k = _centred_sphere(8)
    mb = _block(sphere, k)
    assert not mb.is_inclusion[mb.interface_cells].any()


@pytest.mark.parametrize(
    "shape_factory",
    [
        lambda c, R: Sphere(c, R),
        lambda c, R: Ellipsoid(c, [R, 0.75 * R, 0.6 * R]),
        lambda c, R: Superquadric(c, [R, 0.85 * R, 0.7 * R], 3.0, 3.0),
        lambda c, R: SphericalHarmonicGrain.random(
            c, 0.85 * R, np.random.default_rng(5), l_max=4, roughness=0.15
        ),
    ],
    ids=["sphere", "ellipsoid", "superquadric", "grain"],
)
def test_every_star_shape_uses_the_same_code_path(shape_factory):
    """Shape genericity is the repo's central claim -- keep it honest."""
    k = 8
    centre = np.array([k, k, k]) / 2 * H
    shape = shape_factory(centre, (k / 2 * H) / 1.5)
    mb = _block(shape, k)

    assert np.all(signed_volumes(mb.points, mb.cells) > 0)
    assert scaled_jacobian(mb.points, mb.cells).min() > 0.3

    pts = mb.points[mb.surface_nodes]
    assert np.abs(shape.radial_scale(pts) - 1.0).max() < 1e-12


def test_grading_moves_nodes_towards_the_interface():
    sphere, k = _centred_sphere(8)
    plain = _block(sphere, k, MacroBlockOptions(n_outer=3, outer_grading=1.0))
    graded = _block(sphere, k, MacroBlockOptions(n_outer=3, outer_grading=4.0))
    d_plain = np.linalg.norm(plain.points - sphere.center, axis=1)
    d_graded = np.linalg.norm(graded.points - sphere.center, axis=1)
    shell = (d_plain > sphere.radius * 1.001) & (d_plain < sphere.radius * 1.4)
    assert d_graded[shell].mean() < d_plain[shell].mean()


def test_more_shell_layers_improve_the_edge_ratio():
    sphere, k = _centred_sphere(8)
    coarse = edge_ratio(*_unpack(_block(sphere, k, MacroBlockOptions(n_inner=1, n_outer=1)))).max()
    fine = edge_ratio(*_unpack(_block(sphere, k, MacroBlockOptions(n_inner=3, n_outer=3)))).max()
    assert fine < coarse


def test_rejects_a_degenerate_macrobox():
    sphere, _ = _centred_sphere(8)
    with pytest.raises(ValueError):
        _block(sphere, 1)


def _unpack(mb):
    return mb.points, mb.cells


def test_shape_query_at_the_exact_centre_is_well_defined():
    """The core block's centre node sits exactly on the shape centre."""
    from hexaform.shapes.base import normalize

    assert np.allclose(normalize(np.zeros(3)), [1.0, 0.0, 0.0])
    centre = np.array([0.1, 0.1, 0.1])
    for shape in (
        Sphere(centre, 0.05),
        Ellipsoid(centre, [0.05, 0.04, 0.03]),
        Superquadric(centre, [0.05, 0.04, 0.03], 3.0, 3.0),
    ):
        r = shape.support_radius(normalize(np.zeros(3)))
        assert np.isfinite(r).all() and np.all(r > 0)
        assert np.isfinite(shape.radial_scale(centre[None, :])).all()
