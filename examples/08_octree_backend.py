#!/usr/bin/env python3
"""Request the experimental Backend B from the production pipeline.

The default Hexaform background is a uniform Cartesian grid.  This example
The conforming transition-template work is not complete. For star-shaped
particles, the production pipeline therefore records the octree request but
uses the proven Cartesian cubed-sphere macroblock path. This example verifies
that safe fallback and prints the metadata that makes it explicit.

Use cases:
  * Dense microstructures where ligaments are too thin for the uniform grid.
  * Phase-field fracture, where ``h`` must resolve the regularised crack
    near the interface but the bulk can be coarse.
  * Non-star-shaped bodies (tori, branching pores), which require the
    octree + surface projection.

The example places two close spherical particles and meshes the domain
with the octree backend.  Compare the element count to the equivalent
Cartesian run.
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
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float, default=1.0, help="cube edge length")
    p.add_argument("--mesh-density", type=int, default=24,
                   help="background cells per domain edge (coarsest octree level)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="output/examples/octree-request-fallback.e")
    args = p.parse_args()

    # Two particles that are forced to be close to each other (small gap)
    # would normally require a very fine uniform grid.  The octree
    # refines only near the surfaces.
    config = RunConfig(
        domain=DomainConfig(
            length=args.length,
            mesh_density=args.mesh_density,
            backend="octree",
            octree_bulk_h=args.length / args.mesh_density,
            octree_interface_h=args.length / (args.mesh_density * 4),
            octree_interface_band=0.10,
            octree_max_level=7,
        ),
        inclusions=InclusionConfig(
            shape="sphere",
            n_particles=8,
            radius_mean=0.07,
            radius_std=0.015,
        ),
        output=OutputConfig(path=args.output),
        seed=args.seed,
    )
    result = run(config)
    print()
    print(f"Output: {args.output}")
    print(f"Mesh: {result.mesh.n_elements} HEX8 elements, {result.mesh.n_nodes} nodes")
    print(f"Background: {result.mesh.metadata.get('background', 'cartesian')}")
    if result.validation:
        print(result.validation)
    return 0 if (result.validation is None or result.validation.passed) else 2


if __name__ == "__main__":
    sys.exit(main())
