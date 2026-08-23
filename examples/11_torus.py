#!/usr/bin/env python3
"""Experimental non-star-shaped torus projection audit.

The cubed-sphere macro-block requires a star-shaped body, because the
radial ``support_radius`` direction is undefined on a torus.  A torus
has a hole in the middle, so two different ``phi`` directions from the
centre cross the surface at different distances -- there is no single
``support_radius`` function.

The current octree prototype deliberately lacks conforming transition templates
and robust boundary fitting. This example exercises the research path and
confirms that full validation rejects its hanging nodes/inverted cells. It does
not write a solver-ready mesh.

An invalid result is the expected outcome until the roadmap's Backend B gate is
completed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hexaform.assembly.octree_assembly import (
    OctreeAssemblyOptions,
    assemble_with_octree,
)
from hexaform.core.validation import validate
from hexaform.shapes.torus import Torus


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float, default=1.0)
    p.add_argument("--major", type=float, default=0.30)
    p.add_argument("--minor", type=float, default=0.10)
    p.add_argument("--interface-h", type=float, default=0.025)
    p.add_argument("--max-level", type=int, default=6)
    args = p.parse_args()

    centre = np.array([args.length, args.length, args.length]) / 2
    torus = Torus(
        centre=centre,
        major_radius=args.major,
        minor_radius=args.minor,
    )
    print(f"Torus: centre={tuple(centre)}, R={args.major}, r={args.minor}")
    print(f"Volume: {torus.volume():.4f} (analytic)")
    print(f"Surface area: {torus.surface_area():.4f}")

    mesh = assemble_with_octree(
        [torus],
        domain_length=args.length,
        options=OctreeAssemblyOptions(
            bulk_h=args.interface_h * 2,
            interface_h=args.interface_h,
            interface_band=0.10,
            max_level=args.max_level,
        ),
    )
    report = validate(mesh)
    print()
    print(f"Mesh: {mesh.n_elements} HEX8 elements, {mesh.n_nodes} nodes")
    print(f"Scaled Jacobian: min={report.quality.min_scaled_jacobian:.4f}, "
          f"mean={report.quality.mean_scaled_jacobian:.4f}")
    n_projected = mesh.metadata.get("n_surface_nodes_projected", 0)
    print(f"Surface nodes projected onto the analytic torus: {n_projected}")
    print()
    print(mesh.summary())
    print(report)
    if report.passed:
        print("Unexpected: the experimental non-star mesh passed validation.")
        return 2
    print("Expected: the research prototype was rejected by the production validity gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
