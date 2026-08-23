#!/usr/bin/env python3
"""Every star-shaped body flows through the identical template.

This is the repository's central claim, so it is worth demonstrating rather than
asserting: spheres, ellipsoids, superquadrics and spherical-harmonic grains all
use the same macro-block code path, with no shape-specific meshing logic.
"""

from __future__ import annotations

from hexaform import RunConfig, run
from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig

CASES = [
    ("sphere", {}, "spheres"),
    ("ellipsoid", {"aspect_ratios": (1.0, 0.7, 0.55)}, "ellipsoids"),
    ("superquadric",
     {"aspect_ratios": (1.0, 0.9, 0.75), "exponent_xy": 3.5, "exponent_z": 3.5},
     "aggregate"),
    ("sh_grain", {"grain_roughness": 0.18, "grain_l_max": 5}, "grains"),
]


def main() -> None:
    rows = []
    for shape, extra, label in CASES:
        config = RunConfig(
            domain=DomainConfig(length=1.0, mesh_density=32),
            inclusions=InclusionConfig(
                shape=shape, n_particles=8,
                radius_mean=0.085, radius_std=0.015, **extra,
            ),
            output=OutputConfig(path=f"output/examples/{label}.e"),
            seed=17,
            verbose=False,
        )
        result = run(config)
        v = result.validation
        rows.append((
            label, result.mesh.n_elements,
            v.quality.min_scaled_jacobian,
            v.geometry["max_node_surface_error"],
            v.geometry["relative_volume_error"],
            "pass" if v.passed else "FAIL",
        ))
        print(f"  meshed {label}")

    print(f"\n{'shape':<12} {'elements':>10} {'minSJ':>8} {'node err':>11} "
          f"{'vol err':>9} {'status':>7}")
    print("-" * 62)
    for label, n, sj, node, vol, status in rows:
        print(f"{label:<12} {n:>10} {sj:>8.4f} {node:>11.2e} {vol:>9.5f} {status:>7}")

    print("\nNote the node error column: for every shape family the interface")
    print("nodes lie on the analytic surface to machine precision. Nothing in")
    print("the mesher knows which shape it is meshing -- it only calls")
    print("support_radius(directions).")


if __name__ == "__main__":
    main()
