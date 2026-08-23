#!/usr/bin/env python3
"""Constrained Jacobian-barrier mesh optimisation (roadmap §3).

After the cubed-sphere macro-block produces a conforming mesh, individual
hex quality can still be improved by relocating *free* (non-interface,
non-boundary) nodes while keeping fixed nodes on the analytic surface.

This example runs a few iterations of the Knupp-style log-barrier
optimisation with surface nodes locked, on a simple sphere-in-cube.
The Hexaform cubed-sphere already produces min scaled Jacobian 0.577
(an intrinsic O-grid corner value), so the smoothing rarely improves
this metric -- but it remains essential when the octree backend is
enabled and elements cut the interface at non-ideal angles.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from hexaform import RunConfig, run
from hexaform.pipeline.config import (
    DomainConfig,
    InclusionConfig,
    OutputConfig,
    SmoothingConfig,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float, default=0.4)
    p.add_argument("--mesh-density", type=int, default=10)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--iters", type=int, default=20,
                   help="number of smoothing iterations")
    p.add_argument("--output", default="output/examples/smoothed.e")
    args = p.parse_args()

    config = RunConfig(
        domain=DomainConfig(length=args.length, mesh_density=args.mesh_density),
        inclusions=InclusionConfig(
            shape="sphere", n_particles=2, radius_mean=0.05, radius_std=0.0,
        ),
        output=OutputConfig(path=args.output),
        smoothing=SmoothingConfig(
            enabled=True,
            n_iterations=args.iters,
            min_scaled_jacobian=0.5774,
        ),
        seed=args.seed,
    )
    result = run(config)
    print()
    print(f"Mesh: {result.mesh.n_elements} HEX8 elements, {result.mesh.n_nodes} nodes")
    if result.validation:
        print(result.validation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
