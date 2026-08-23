"""Transfinite (Coons) interpolation and 1-D grading.

These are the two structured-meshing primitives Hexaform needs:

* :func:`tfi_volume` fills a logical cube from its six boundary patches, used
  for the O-grid core block;
* :func:`graded_weights` produces clustered radial layer positions, used to
  resolve the steep stress and plastic-strain gradients at inclusion interfaces
  without refining the whole domain.
"""

from __future__ import annotations

import numpy as np


def graded_weights(
    n: int, ratio: float = 1.0, cluster: str = "none"
) -> np.ndarray:
    """Return ``n + 1`` monotone weights spanning ``[0, 1]``.

    Parameters
    ----------
    n:
        Number of intervals.
    ratio:
        Ratio of the largest to the smallest interval.  ``1.0`` is uniform.
    cluster:
        ``"none"``, ``"start"`` (small intervals near 0), ``"end"`` (small
        intervals near 1), or ``"both"``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if ratio <= 0:
        raise ValueError("grading ratio must be positive")
    if cluster == "none" or abs(ratio - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n + 1)

    if cluster == "both":
        half = max(n // 2, 1)
        left = 0.5 * graded_weights(half, ratio, "start")
        right = 0.5 + 0.5 * graded_weights(n - half, ratio, "end")
        return np.concatenate([left, right[1:]])

    g = ratio ** (1.0 / max(n - 1, 1))
    steps = g ** np.arange(n)
    if cluster == "end":
        steps = steps[::-1]
    elif cluster != "start":
        raise ValueError(f"unknown cluster mode {cluster!r}")
    w = np.concatenate([[0.0], np.cumsum(steps)])
    return w / w[-1]


def tfi_volume(
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    wu: np.ndarray | None = None,
    wv: np.ndarray | None = None,
    ww: np.ndarray | None = None,
) -> np.ndarray:
    """Fill an ``(nu+1, nv+1, nw+1, 3)`` lattice from six boundary patches.

    Patch index conventions (all are ``(*, *, 3)`` arrays)::

        x_lo, x_hi : (nv+1, nw+1)   faces at u = 0, 1
        y_lo, y_hi : (nu+1, nw+1)   faces at v = 0, 1
        z_lo, z_hi : (nu+1, nv+1)   faces at w = 0, 1

    The six patches must agree on their twelve shared edges; Hexaform guarantees
    this because they are all cut from the same welded cubed-sphere patch set.
    Uses the standard Boolean-sum (face - edge + corner) formula, which
    reproduces the boundary exactly.
    """
    nv, nw = x_lo.shape[0] - 1, x_lo.shape[1] - 1
    nu = y_lo.shape[0] - 1
    for arr, shape, label in (
        (x_hi, x_lo.shape, "x_hi"),
        (y_hi, y_lo.shape, "y_hi"),
        (z_hi, z_lo.shape, "z_hi"),
    ):
        if arr.shape != shape:
            raise ValueError(f"{label} shape {arr.shape} does not match its partner {shape}")
    if y_lo.shape[:2] != (nu + 1, nw + 1) or z_lo.shape[:2] != (nu + 1, nv + 1):
        raise ValueError("boundary patch shapes are mutually inconsistent")

    u = (np.linspace(0.0, 1.0, nu + 1) if wu is None else np.asarray(wu))[:, None, None, None]
    v = (np.linspace(0.0, 1.0, nv + 1) if wv is None else np.asarray(wv))[None, :, None, None]
    w = (np.linspace(0.0, 1.0, nw + 1) if ww is None else np.asarray(ww))[None, None, :, None]

    # --- face contributions -------------------------------------------------
    U = (1 - u) * x_lo[None, :, :, :] + u * x_hi[None, :, :, :]
    V = (1 - v) * y_lo[:, None, :, :] + v * y_hi[:, None, :, :]
    W = (1 - w) * z_lo[:, :, None, :] + w * z_hi[:, :, None, :]

    # --- edge contributions -------------------------------------------------
    # edges running along w: intersections of the u- and v-faces
    e_00 = x_lo[0, :, :]   # u=0, v=0  -> (nw+1, 3)
    e_01 = x_lo[-1, :, :]  # u=0, v=1
    e_10 = x_hi[0, :, :]   # u=1, v=0
    e_11 = x_hi[-1, :, :]  # u=1, v=1
    UV = (
        (1 - u) * (1 - v) * e_00[None, None, :, :]
        + (1 - u) * v * e_01[None, None, :, :]
        + u * (1 - v) * e_10[None, None, :, :]
        + u * v * e_11[None, None, :, :]
    )

    # edges running along v: intersections of the u- and w-faces
    f_00 = x_lo[:, 0, :]
    f_01 = x_lo[:, -1, :]
    f_10 = x_hi[:, 0, :]
    f_11 = x_hi[:, -1, :]
    UW = (
        (1 - u) * (1 - w) * f_00[None, :, None, :]
        + (1 - u) * w * f_01[None, :, None, :]
        + u * (1 - w) * f_10[None, :, None, :]
        + u * w * f_11[None, :, None, :]
    )

    # edges running along u: intersections of the v- and w-faces
    g_00 = y_lo[:, 0, :]
    g_01 = y_lo[:, -1, :]
    g_10 = y_hi[:, 0, :]
    g_11 = y_hi[:, -1, :]
    VW = (
        (1 - v) * (1 - w) * g_00[:, None, None, :]
        + (1 - v) * w * g_01[:, None, None, :]
        + v * (1 - w) * g_10[:, None, None, :]
        + v * w * g_11[:, None, None, :]
    )

    # --- corner contribution ------------------------------------------------
    c = np.empty((2, 2, 2, 3))
    c[0, 0, 0] = x_lo[0, 0]
    c[0, 0, 1] = x_lo[0, -1]
    c[0, 1, 0] = x_lo[-1, 0]
    c[0, 1, 1] = x_lo[-1, -1]
    c[1, 0, 0] = x_hi[0, 0]
    c[1, 0, 1] = x_hi[0, -1]
    c[1, 1, 0] = x_hi[-1, 0]
    c[1, 1, 1] = x_hi[-1, -1]
    UVW = (
        (1 - u) * (1 - v) * (1 - w) * c[0, 0, 0]
        + (1 - u) * (1 - v) * w * c[0, 0, 1]
        + (1 - u) * v * (1 - w) * c[0, 1, 0]
        + (1 - u) * v * w * c[0, 1, 1]
        + u * (1 - v) * (1 - w) * c[1, 0, 0]
        + u * (1 - v) * w * c[1, 0, 1]
        + u * v * (1 - w) * c[1, 1, 0]
        + u * v * w * c[1, 1, 1]
    )

    return U + V + W - UV - UW - VW + UVW
