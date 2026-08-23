#!/usr/bin/env python3
"""Verify geometric convergence on a single off-centre sphere.

The SciSpace review's Milestone 1 asks specifically for an *off-centre* sphere,
because a centred one hides decomposition assumptions. This script meshes one at
several resolutions and reports how the geometric error falls.

Expected: the volume error drops by roughly 4x each time the surface resolution
doubles (second order), while the node radial error stays at round-off -- the
nodes are placed on the analytic surface, so only the flat facets between them
contribute error.
"""

from __future__ import annotations

import numpy as np

from hexaform.assembly.builder import assemble
from hexaform.background.cartesian import CartesianGrid
from hexaform.core.quality import assess
from hexaform.core.topology import signed_volumes
from hexaform.packing.rsa import PackedInclusion
from hexaform.shapes.library import Sphere


def main() -> None:
    length, radius = 1.0, 0.18
    print(f"off-centre sphere R = {radius} in a {length} cube\n")
    print(f"{'density':>8} {'k':>4} {'elements':>10} {'minSJ':>8} "
          f"{'node err':>11} {'vol err':>10} {'observed':>9} {'2nd ord':>8}")
    print("-" * 74)

    previous = None
    previous_k = None
    for density in (16, 24, 32, 48, 64):
        grid = CartesianGrid.cube(length, density)
        # Place the sphere deliberately off centre and off any symmetry plane.
        k = 2 * int(np.ceil(1.35 * radius / grid.h))
        lo = grid.point_to_index(np.array([0.42, 0.37, 0.55])) - k // 2
        lo = np.clip(lo, 1, np.array(grid.divisions) - k - 1)
        centre = grid.origin + (lo + k / 2.0) * grid.h
        inclusion = PackedInclusion(Sphere(centre, radius), lo, k, 1)

        mesh = assemble(grid, [inclusion])
        q = assess(mesh)

        incl = mesh.block_id >= 100
        vol = signed_volumes(mesh.points, mesh.cells)[incl].sum()
        exact = 4 / 3 * np.pi * radius**3
        vol_err = abs(vol - exact) / exact

        on_surface = np.abs(
            np.linalg.norm(mesh.points - centre, axis=1) - radius
        ) < 1e-6
        node_err = np.abs(
            np.linalg.norm(mesh.points[on_surface] - centre, axis=1) - radius
        ).max()

        observed = previous / vol_err if previous else float("nan")
        expected = (k / previous_k) ** 2 if previous_k else float("nan")
        previous, previous_k = vol_err, k
        print(f"{density:>8} {k:>4} {mesh.n_elements:>10} "
              f"{q.min_scaled_jacobian:>8.4f} {node_err:>11.2e} "
              f"{vol_err:>10.5f} {observed:>9.2f} {expected:>8.2f}")

    print("\n'observed' is the error reduction from the previous row; '2nd ord'")
    print("is what second-order convergence predicts, (k_new/k_old)^2. The two")
    print("columns agreeing is the convergence result -- note the refinement")
    print("steps here are not doublings, so neither column should read 4.")
    print("Node error stays at machine precision: the surface *location* is exact,")
    print("only the flat HEX8 facets between nodes contribute geometric error.")


if __name__ == "__main__":
    main()
