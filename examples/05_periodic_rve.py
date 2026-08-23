#!/usr/bin/env python3
"""Generate an RVE mesh with periodic boundary conditions.

This example demonstrates how to create a hexahedral mesh suitable for
homogenization studies, with automatic node pairing across opposite domain faces.

Run interactively::

    python examples/05_periodic_rve.py

or non-interactively::

    python examples/05_periodic_rve.py --length 1.0 --n-particles 12 \\
        --radius-mean 0.075 --mesh-density 40 --seed 7
"""

from __future__ import annotations

import argparse
import sys

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
    p.add_argument("--phase-field-length", type=float, default=None)
    p.add_argument("-o", "--output", default="output/examples/rve_periodic.e")

    args = p.parse_args()

    interactive = args.length is None
    if interactive:
        print("Hexaform - periodic RVE with boundary condition pairing\n")
        length = ask("Cube edge length", 1.0, float)
        n_particles = ask_count("Number of particles (or a range like 8-15)", 12)
        radius_mean = ask("Mean particle radius", 0.075, float)
        suggested = recommended_mesh_density(length, radius_mean)
        mesh_density = ask("Mesh density (background cells per edge)", suggested, int)
        seed = ask("Random seed", 0, int)
        output = input("Output file [output/examples/rve_periodic.e]: ").strip() or "output/examples/rve_periodic.e"
    else:
        length = args.length
        raw = args.n_particles or "12"
        n_particles = (
            tuple(int(v) for v in raw.split("-")) if "-" in raw else int(raw)
        )
        radius_mean = args.radius_mean if args.radius_mean is not None else 0.075
        radius_std = args.radius_std if args.radius_std is not None else 0.015
        mesh_density = args.mesh_density or recommended_mesh_density(length, radius_mean)
        seed = args.seed
        output = args.output

    # A quick feasibility note before spending time on the mesh.
    h = length / mesh_density
    print(f"\ncell size h = {h:.5g}, about {radius_mean / h:.1f} cells per mean radius")
    if radius_mean / h < 2.0:
        print("  warning: fewer than 2 cells per radius - particles will be coarse")

    config = RunConfig(
        domain=DomainConfig(length=length, mesh_density=mesh_density, periodic=True),
        inclusions=InclusionConfig(
            shape="sphere",
            n_particles=n_particles,
            radius_mean=radius_mean,
            radius_std=radius_std,
        ),
        output=OutputConfig(path=output),
        seed=seed,
        phase_field_length=args.phase_field_length,
    )
    print()
    result = run(config)

    if result.validation is not None and not result.validation.passed:
        print("\nThe mesh did NOT pass validation. Suggestions:")
        print("  - raise --mesh-density so particles are better resolved")
        print("  - lower --radius-mean or the particle count")
        return 2

    # Print periodic boundary set information
    if result.mesh is not None:
        print("\nPeriodic boundary sets created:")
        for axis_name in ["x", "y", "z"]:
            neg_set = result.mesh.node_sets.get(f"{axis_name}-_node_set")
            pos_set = result.mesh.node_sets.get(f"{axis_name}+_node_set")
            pair_set = result.mesh.node_sets.get(f"{axis_name}_periodic_pair")
            if neg_set is not None and pos_set is not None:
                print(f"  {axis_name}: {len(neg_set)} node pairs (neg: first={neg_set[0]}, pos: first={pos_set[0]})")
            if pair_set is not None and len(pair_set) >= 2:
                n_pairs = len(pair_set) // 2
                print(f"           → {axis_name}_periodic_pair contains {n_pairs} (neg_id, pos_id) tuples")

    print("\nReady for your solver:")
    print("  block 1   'matrix'      - matrix material")
    print("  block 100 'inclusions'  - particle material")
    print("  side sets x-,x+,y-,y+,z-,z+ - boundary conditions")
    print("  side set  'interface'   - matrix/particle interface")
    print("\nFor periodic BCs, use the paired node sets in your solver:")
    print("  - {axis}-_node_set and {axis}+_node_set: nodes on opposite faces")
    print("  - {axis}_periodic_pair: flattened (neg_id, pos_id) tuples for constraints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
