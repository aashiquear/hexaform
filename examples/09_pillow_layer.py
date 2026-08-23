#!/usr/bin/env python3
"""Interface pillow layer.

The ``pillow`` option (roadmap §4) inserts an extra hex sheet at every
material interface, so no element terminates directly on a curved
surface.  The macro-block template already produces well-shaped
interface elements, so the pillow is *low priority* in the default
Cartesian pipeline -- the roadmap says so explicitly.  It becomes
important with the octree backend (Backend B), where background cells
can cut the interface at an arbitrary angle.

This example enables the pillow layer on a small RVE and prints the
delta in element count and quality.
"""

from __future__ import annotations

import argparse
import sys

from hexaform import RunConfig, run
from hexaform.core.quality import scaled_jacobian
from hexaform.pipeline.config import (
    DomainConfig,
    InclusionConfig,
    OutputConfig,
    PillowConfig,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float, default=0.5)
    p.add_argument("--mesh-density", type=int, default=12)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--thickness-fraction", type=float, default=0.20,
                   help="pillow thickness as a fraction of the local cell size")
    p.add_argument("--output", default="output/examples/pillow.e")
    args = p.parse_args()

    # Without pillow
    cfg_no = RunConfig(
        domain=DomainConfig(length=args.length, mesh_density=args.mesh_density),
        inclusions=InclusionConfig(
            shape="sphere", n_particles=3, radius_mean=0.05, radius_std=0.01,
        ),
        output=OutputConfig(path=""),
        seed=args.seed,
        validate_mesh=False,
        verbose=False,
    )
    res_no = run(cfg_no)
    elems_no = res_no.mesh.n_elements
    sj_no = float(scaled_jacobian(res_no.mesh.points, res_no.mesh.cells).min())

    # With pillow
    cfg = RunConfig(
        domain=DomainConfig(length=args.length, mesh_density=args.mesh_density),
        inclusions=InclusionConfig(
            shape="sphere", n_particles=3, radius_mean=0.05, radius_std=0.01,
        ),
        output=OutputConfig(path=args.output),
        pillow=PillowConfig(enabled=True, thickness_fraction=args.thickness_fraction),
        seed=args.seed,
    )
    result = run(cfg)
    n_pillow = result.mesh.metadata.get("n_pillow_elements", 0)
    print()
    print(f"Before pillow: {elems_no} elements, "
          f"min scaled Jacobian = {sj_no:.4f}")
    print(f"After  pillow: {result.mesh.n_elements} elements "
          f"({n_pillow} pillow hexes), "
          f"min scaled Jacobian = {result.validation.quality.min_scaled_jacobian:.4f}")
    if result.validation:
        print(result.validation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
