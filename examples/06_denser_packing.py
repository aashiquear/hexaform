#!/usr/bin/env python3
"""Denser packing example with collective rearrangement.

This example demonstrates how to use the collective-rearrangement packer (Lubachevsky-
Stillinger / force-biased) to achieve significantly denser inclusion populations than
standard random sequential addition (RSA).

While RSA saturates around 10-20% volume fraction, collective rearrangement can reach
35-45%+ by allowing already-placed macro-boxes to move and open cavities for new inclusions.
The "guaranteed meshable" invariant is preserved: every final configuration satisfies the
exact same integer macro-box disjointness with cell margin that RSA does.

Run interactively::

    python examples/06_denser_packing.py

or non-interactively::

    python examples/06_denser_packing.py --length 1.0 --n-particles 50 \\
        --radius-mean 0.075 --packer collective --seed 7
"""

from __future__ import annotations

import argparse
import sys
import numpy as np

from hexaform import RunConfig, run
from hexaform.core.validation import recommended_mesh_density
from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig


def ask(prompt: str, default, cast):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  could not read {raw!r}; using {default}")
        return default


def ask_count(prompt: str, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    if "-" in raw:
        lo, hi = raw.split("-", 1)
        return (int(lo), int(hi))
    return int(raw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float)
    p.add_argument("--n-particles")
    p.add_argument("--radius-mean", type=float)
    p.add_argument("--radius-std", type=float)
    p.add_argument("--mesh-density", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--packer", choices=["rsa", "collective"], default="collective")
    p.add_argument("--collective-max-steps", type=int, default=2000)
    p.add_argument("--collective-kinetic", type=float, default=0.5)
    p.add_argument("--phase-field-length", type=float, default=None)
    p.add_argument("-o", "--output", default="output/examples/denser_packing.e")
    args = p.parse_args()

    interactive = args.length is None
    if interactive:
        print("Hexaform - dense packing with collective rearrangement\n")
        length = ask("Cube edge length", 1.0, float)
        n_particles = ask_count("Number of particles (or a range like 8-15)", 50)
        radius_mean = ask("Mean particle radius", 0.075, float)
        suggested = recommended_mesh_density(length, radius_mean)
        mesh_density = ask("Mesh density (background cells per edge)", suggested, int)
        seed = ask("Random seed", 0, int)
        packer = input("Packer [collective]: ").strip() or "collective"
        max_steps = int(input("Max collective steps [2000]: ") or 2000)
        kinetic = float(input("Kinetic fraction [0.5]: ") or 0.5)
        output = input("Output file [output/examples/denser_packing.e]: ").strip() or "output/examples/denser_packing.e"
    else:
        length = args.length
        raw = args.n_particles or "50"
        n_particles = (
            tuple(int(v) for v in raw.split("-")) if "-" in raw else int(raw)
        )
        radius_mean = args.radius_mean if args.radius_mean is not None else 0.075
        mesh_density = args.mesh_density or recommended_mesh_density(length, radius_mean)
        seed = args.seed
        packer = args.packer
        max_steps = args.collective_max_steps
        kinetic = args.collective_kinetic
        output = args.output

    print(f"\nCell size h = {length/mesh_density:.5f}, about {radius_mean/(length/mesh_density):.1f} cells per mean radius")
    if mesh_density < 20:
        print("  warning: low mesh density - consider raising it for better resolution")

    config = RunConfig(
        domain=DomainConfig(length=length, mesh_density=mesh_density),
        inclusions=InclusionConfig(
            shape="sphere",
            n_particles=n_particles,
            radius_mean=radius_mean,
            radius_std=0.015,  # some size variation helps collective exploration
            gap_cells=1,       # minimal gap for denser packing
            packer=packer,
            collective_max_steps=max_steps,
            collective_kinetic=kinetic,
        ),
        output=OutputConfig(path=output),
        seed=seed,
        phase_field_length=args.phase_field_length,
    )

    if config.inclusions.packer == "collective":
        print(f"Using collective rearrangement (max {max_steps} steps, kinetic={kinetic})")

    print()
    result = run(config)

    if result.validation is not None and not result.validation.passed:
        print("\nThe mesh did NOT pass validation. Suggestions:")
        print("  - raise --mesh-density so particles are better resolved")
        print("  - lower --radius-mean or the particle count")
        return 2

    # Compare to RSA for insight
    from hexaform.background.cartesian import CartesianGrid
    from hexaform.packing.rsa import PackingSpec, pack as rsa_pack

    grid = CartesianGrid.cube(length, mesh_density)
    spec = PackingSpec(n_particles=n_particles, radius_mean=radius_mean, radius_std=0.015,
                       gap_cells=1, max_tries_per_particle=400)
    try:
        rsa_result = rsa_pack(grid, spec, np.random.default_rng(seed))
        print(f"\nComparison with RSA (same seed {seed}):")
        print(f"  Collective placed {result.packing.count} particles, volume fraction {result.packing.stats['volume_fraction']:.3f}")
        print(f"  RSA would place     {rsa_result.count} particles, volume fraction {rsa_result.stats['volume_fraction']:.3f}")
        if result.packing.count > rsa_result.count:
            improvement = (result.packing.count - rsa_result.count) / rsa_result.count * 100
            print(f"  → +{int(improvement)}% more inclusions with collective rearrangement")
    except ImportError:
        # Keep the generated mesh usable even when an optional comparison
        # implementation is unavailable.
        pass

    print("\nReady for your solver:")
    print("  block 1   'matrix'      - matrix material")
    print("  block 100 'inclusions'  - particle material")
    print("  side sets x-,x+,y-,y+,z-,z+ - boundary conditions")
    print("  side set  'interface'   - matrix/particle interface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
