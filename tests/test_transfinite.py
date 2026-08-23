import numpy as np
import pytest

from hexaform.templates.transfinite import graded_weights, tfi_volume


def _lattice(nu, nv, nw):
    g = np.stack(
        np.meshgrid(
            np.linspace(0, 1, nu + 1),
            np.linspace(0, 2, nv + 1),
            np.linspace(0, 3, nw + 1),
            indexing="ij",
        ),
        axis=-1,
    )
    return g


def test_tfi_reproduces_a_trilinear_box_exactly():
    g = _lattice(4, 5, 6)
    out = tfi_volume(g[0], g[-1], g[:, 0], g[:, -1], g[:, :, 0], g[:, :, -1])
    assert np.abs(out - g).max() < 1e-14


def test_tfi_reproduces_a_curved_boundary_exactly():
    g = _lattice(4, 5, 6)
    g = g + 0.1 * np.sin(3 * g[..., [1, 2, 0]])
    out = tfi_volume(g[0], g[-1], g[:, 0], g[:, -1], g[:, :, 0], g[:, :, -1])
    for face_out, face_in in [
        (out[0], g[0]), (out[-1], g[-1]),
        (out[:, 0], g[:, 0]), (out[:, -1], g[:, -1]),
        (out[:, :, 0], g[:, :, 0]), (out[:, :, -1], g[:, :, -1]),
    ]:
        assert np.abs(face_out - face_in).max() < 1e-14


def test_tfi_rejects_inconsistent_patch_shapes():
    g = _lattice(4, 5, 6)
    with pytest.raises(ValueError):
        tfi_volume(g[0], g[-1][:-1], g[:, 0], g[:, -1], g[:, :, 0], g[:, :, -1])


@pytest.mark.parametrize("cluster", ["none", "start", "end", "both"])
def test_grading_is_monotone_and_normalised(cluster):
    w = graded_weights(8, 3.0, cluster)
    assert w.size == 9
    assert w[0] == pytest.approx(0.0)
    assert w[-1] == pytest.approx(1.0)
    assert np.all(np.diff(w) > 0)


def test_grading_actually_clusters_where_asked():
    d = np.diff(graded_weights(8, 4.0, "start"))
    assert d[0] < d[-1]
    d = np.diff(graded_weights(8, 4.0, "end"))
    assert d[0] > d[-1]
    d = np.diff(graded_weights(8, 4.0, "both"))
    assert d[0] < d[len(d) // 2 - 1] and d[-1] < d[len(d) // 2]
