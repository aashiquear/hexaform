#!/usr/bin/env python3
"""Example: Inserting a notch or pre-crack for phase-field fracture simulations.

Phase-field fracture simulations require an initial crack or notch to seed
crack initiation. This example demonstrates both approaches:

1. **Field method** (default): Initialize a phase-field damage variable to 1.0
   in elements along the notch path. The mesh remains intact.

2. **Geometric method**: Mark elements as void (particle_id = 0), effectively
   removing material from the notch region.

Run interactively::

    python examples/07_notch_insertion.py

or with command-line options::

    python examples/07_notch_insertion.py --notch-type edge --notch-length 0.3 \\
        --method field --output notch_mesh.e
"""

from __future__ import annotations

import argparse
import sys

from hexaform import RunConfig, run
from hexaform.pipeline.config import (
    DomainConfig,
    InclusionConfig,
    NotchConfig,
    OutputConfig,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=float, default=1.0, help="Domain length")
    p.add_argument("--n-particles", type=int, default=8, help="Number of particles")
    p.add_argument("--radius-mean", type=float, default=0.06, help="Mean particle radius")
    p.add_argument("--radius-std", type=float, default=0.01, help="Particle radius std dev")
    p.add_argument("--mesh-density", type=int, default=40, help="Background cells per edge")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Notch parameters
    p.add_argument(
        "--notch-type",
        choices=["edge", "center", "chevron"],
        default="edge",
        help="Type of notch geometry",
    )
    p.add_argument(
        "--notch-length",
        type=float,
        default=0.3,
        help="Notch length along propagation direction",
    )
    p.add_argument(
        "--notch-depth",
        type=float,
        default=1.0,
        help="Notch depth (through-thickness)",
    )
    p.add_argument(
        "--notch-origin",
        type=float,
        nargs=3,
        default=None,
        help="Notch origin (x, y, z). Default depends on type.",
    )
    p.add_argument(
        "--notch-direction",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        help="Notch propagation direction (dx, dy, dz)",
    )
    p.add_argument(
        "--chevron-angle",
        type=float,
        default=45.0,
        help="Chevron notch half-angle (degrees)",
    )
    p.add_argument(
        "--method",
        choices=["field", "geometric"],
        default="field",
        help="Notch insertion method",
    )
    p.add_argument(
        "--phase-field-variable",
        type=str,
        default="phase_field",
        help="Name of phase-field element variable",
    )

    p.add_argument("-o", "--output", default="output/examples/notch_mesh.e", help="Output file")
    p.add_argument("--phase-field-length", type=float, default=0.02,
                   help="Phase-field regularisation length ℓ")

    args = p.parse_args()

    # Compute cell size and report resolution
    h = args.length / args.mesh_density
    print("Hexaform - Notch insertion example")
    print("=" * 50)
    print(f"Domain: {args.length}³ cube, {args.mesh_density}³ cells (h = {h:.5g})")
    print(f"Particles: {args.n_particles} spheres, r_mean = {args.radius_mean}")
    print(f"Notch: {args.notch_type} notch, length = {args.notch_length}")
    print(f"Method: {args.method}")
    if args.method == "field":
        print(f"Phase-field variable: {args.phase_field_variable}")
    print(f"Cell size h = {h:.5g}, phase-field length ℓ = {args.phase_field_length}")
    print(f"h/ℓ = {h / args.phase_field_length:.3f} (target: ≤ 0.5 for accurate phase-field)")
    print()

    # Configure notch
    if args.notch_origin is None:
        # Default origins based on notch type
        if args.notch_type == "edge":
            notch_origin = (0.0, 0.5, 0.5)
        elif args.notch_type == "center":
            notch_origin = (0.5, 0.5, 0.5)
        else:  # chevron
            notch_origin = (0.0, 0.5, 0.5)
    else:
        notch_origin = tuple(args.notch_origin)

    notch_config = NotchConfig(
        enabled=True,
        notch_type=args.notch_type,
        origin=notch_origin,
        direction=tuple(args.notch_direction),
        length=args.notch_length,
        depth=args.notch_depth,
        chevron_angle=args.chevron_angle,
        method=args.method,
        phase_field_value=1.0,
        phase_field_variable=args.phase_field_variable,
    )

    config = RunConfig(
        domain=DomainConfig(length=args.length, mesh_density=args.mesh_density),
        inclusions=InclusionConfig(
            shape="sphere",
            n_particles=args.n_particles,
            radius_mean=args.radius_mean,
            radius_std=args.radius_std,
        ),
        output=OutputConfig(path=args.output),
        seed=args.seed,
        phase_field_length=args.phase_field_length,
        notch=notch_config,
    )

    print("Generating mesh with notch...")
    result = run(config)

    if result.validation is not None and not result.validation.passed:
        print("\nThe mesh did NOT pass validation.")
        return 2

    # Report notch statistics
    n_notch = result.mesh.metadata.get("n_notch_elements", 0)
    total_elems = result.mesh.n_elements
    print("\nNotch insertion complete:")
    print(f"  Notch type: {args.notch_type}")
    print(f"  Notch elements: {n_notch} ({100*n_notch/total_elems:.2f}% of total)")
    print(f"  Method: {args.method}")

    if args.method == "field":
        print(f"  Phase-field variable: '{args.phase_field_variable}'")
        if args.phase_field_variable in result.mesh.element_fields:
            pf = result.mesh.element_fields[args.phase_field_variable]
            n_damaged = int(np.sum(pf > 0.5))
            print(f"  Elements with damage > 0.5: {n_damaged}")

    print("\nOutput files:")
    print(f"  Mesh: {args.output}")
    if result.report_path:
        print(f"  Report: {result.report_path}")

    print("\nReady for phase-field fracture simulation:")
    print(f"  - Load {args.output} in your solver")
    print(f"  - Apply boundary conditions on x-, x+, y-, y+, z-, z+ side sets")
    print(f"  - Initialize phase-field from '{args.phase_field_variable}' element variable")
    if args.method == "geometric":
        print("  - Notch elements have particle_id = 0 (void)")

    return 0


if __name__ == "__main__":
    # Import numpy here to avoid issues if not needed
    import numpy as np
    sys.exit(main())
