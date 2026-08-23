"""The cubed sphere must weld exactly and wind outward, at every resolution."""

import numpy as np
import pytest

from hexaform.templates.cubed_sphere import CubedSphere, patch_spacing


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 12, 16])
def test_node_count_matches_euler_formula(n):
    cs = CubedSphere(n)
    assert cs.n_nodes == 6 * n**2 + 2, "shared patch edges were not welded"
    assert cs.n_quads == 6 * n**2


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("mode", ["uniform", "equiangular", "blend"])
def test_directions_are_unit_and_quads_outward(n, mode):
    cs = CubedSphere(n, spacing=mode, blend=0.6)
    assert np.allclose(np.linalg.norm(cs.directions, axis=1), 1.0)
    check = cs.check()
    assert check["all_outward"], "a patch is wound the wrong way"
    assert check["min_quad_area"] > 0


def test_every_node_belongs_to_a_patch_grid():
    cs = CubedSphere(6)
    seen = np.unique(np.concatenate([cs.index[f].ravel() for f in cs.index]))
    assert seen.size == cs.n_nodes
    assert seen.min() == 0 and seen.max() == cs.n_nodes - 1


def test_shared_edges_are_bitwise_identical():
    """Patch x+ and patch y+ must agree exactly along the edge they share."""
    cs = CubedSphere(8)
    xp = cs.face_cube_points("x+")   # (y, z)
    yp = cs.face_cube_points("y+")   # (z, x)
    edge_from_xp = xp[-1, :, :]                 # y = +1
    edge_from_yp = yp[:, -1, :]                 # x = +1
    assert np.array_equal(np.sort(edge_from_xp, axis=0), np.sort(edge_from_yp, axis=0))


def test_patch_spacing_monotone_and_spans_interval():
    for mode, blend in [("uniform", 0.0), ("equiangular", 1.0), ("blend", 0.5)]:
        s = patch_spacing(10, mode, blend)
        assert s[0] == pytest.approx(-1.0)
        assert s[-1] == pytest.approx(1.0)
        assert np.all(np.diff(s) > 0)


def test_equiangular_is_more_uniform_on_the_sphere_than_gnomonic():
    """The reason equiangular spacing exists: fewer extreme quad areas."""
    def spread(mode):
        cs = CubedSphere(12, spacing=mode)
        p = cs.directions[cs.quads]
        area = 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0]), axis=1
        )
        return area.max() / area.min()

    assert spread("equiangular") < spread("uniform")
