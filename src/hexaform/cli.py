"""Hexaform command line.

::

    hexaform particles-in-cube --length 1.0 --n-particles 12 \\
        --radius-mean 0.075 --radius-std 0.015 --mesh-density 40 \\
        --seed 7 -o rve.e

    hexaform run config.yaml
    hexaform campaign --seeds 0-99 --n-particles 10 -o campaign/
    hexaform inspect rve.e

Every subcommand ends up in :func:`hexaform.pipeline.run.run`; the CLI only
translates arguments into a :class:`~hexaform.pipeline.config.RunConfig`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .pipeline.config import (
    DomainConfig,
    InclusionConfig,
    MacroBlockOptions,
    OutputConfig,
    PillowConfig,
    RunConfig,
    SmoothingConfig,
    ValidationThresholds,
)


def _parse_count(text: str):
    """``"12"`` -> 12; ``"8-15"`` -> ``(8, 15)`` for a random particle count."""
    if "-" in text and not text.startswith("-"):
        lo, hi = text.split("-", 1)
        return (int(lo), int(hi))
    return int(text)


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if "-" in chunk and not chunk.startswith("-"):
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            out.append(int(chunk))
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("domain")
    g.add_argument("--length", type=float, default=1.0, help="cube edge length")
    g.add_argument(
        "--mesh-density", type=int, default=32,
        help="background cells per domain edge (sets h = length / mesh_density)",
    )
    g.add_argument(
        "--backend", type=str.lower, default="cartesian",
        choices=["cartesian", "octree"],
        help="background backend: uniform Cartesian grid or adaptive octree (Backend B)",
    )
    g.add_argument("--periodic", action="store_true",
                   help="enable periodic boundary conditions for homogenization/RVE analysis")

    g = p.add_argument_group("inclusions")
    g.add_argument(
        "--n-particles", type=_parse_count, default=10,
        help="exact count (e.g. 12) or an inclusive range (e.g. 8-15) sampled per seed",
    )
    g.add_argument("--radius-mean", type=float, default=0.08)
    g.add_argument("--radius-std", type=float, default=0.0)
    g.add_argument("--radius-min", type=float, default=None)
    g.add_argument("--radius-max", type=float, default=None)
    g.add_argument(
        "--packer", type=str.lower, default="rsa",
        choices=["rsa", "collective"],
        help="random sequential addition (rsa) or collective rearrangement (collective)"
    )
    g.add_argument("--collective-max-steps", type=int, default=2000,
                   help="maximum search steps for collective packer")
    g.add_argument("--collective-kinetic", type=float, default=0.5,
                   help="fraction of kinetic (exploratory) moves in collective packer")
    g.add_argument(
        "--shape", default="sphere",
        choices=["sphere", "ellipsoid", "superquadric", "sh_grain"],
    )
    g.add_argument(
        "--aspect-ratios", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("A", "B", "C"), help="semi-axis ratios for non-spherical bodies",
    )
    g.add_argument(
        "--box-ratio", type=float, default=1.35,
        help="macro-box half-edge as a multiple of the inclusion bounding radius",
    )
    g.add_argument(
        "--gap-cells", type=int, default=2,
        help="guaranteed matrix elements across the narrowest ligament",
    )
    g.add_argument("--wall-cells", type=int, default=1)

    g = p.add_argument_group("template")
    g.add_argument(
        "--element-type", type=str.upper, default="HEX8",
        choices=["HEX8", "HEX20", "HEX27"],
        help="finite-element interpolation order (default: HEX8)",
    )
    g.add_argument("--n-inner", type=int, default=2, help="radial layers inside the body")
    g.add_argument("--n-outer", type=int, default=2, help="radial layers in the matrix shell")
    g.add_argument(
        "--interface-grading", type=float, default=1.0,
        help="cluster radial layers at the interface (ratio of largest to smallest)",
    )

    g = p.add_argument_group("octree (Backend B)")
    g.add_argument(
        "--pillow", action="store_true",
        help="insert a hex pillow sheet at material interfaces",
    )
    g.add_argument(
        "--pillow-thickness", type=float, default=0.15,
        help="pillow layer thickness as a fraction of the local cell size",
    )
    g.add_argument(
        "--smoothing", action="store_true",
        help="run constrained Jacobian-barrier mesh optimisation",
    )
    g.add_argument(
        "--smoothing-iters", type=int, default=50,
        help="maximum number of smoothing iterations",
    )

    g = p.add_argument_group("run")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--phase-field-length", type=float, default=None,
        help="regularisation length ell; reported as h/ell, never changes the mesh",
    )
    g.add_argument("--min-scaled-jacobian", type=float, default=0.2)
    g.add_argument("--separate-blocks", action="store_true",
                   help="one Exodus element block per inclusion")
    g.add_argument("--no-validate", action="store_true")
    g.add_argument("--write-vtk", action="store_true",
                   help="also write a legacy VTK file beside the Exodus file")
    g.add_argument("--quiet", action="store_true")


def _config_from_args(args: argparse.Namespace, output: str) -> RunConfig:
    return RunConfig(
        domain=DomainConfig(length=args.length, mesh_density=args.mesh_density,
                           periodic=args.periodic, backend=args.backend),
        inclusions=InclusionConfig(
            shape=args.shape,
            n_particles=args.n_particles,
            radius_mean=args.radius_mean,
            radius_std=args.radius_std,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            packer=args.packer,
            collective_max_steps=args.collective_max_steps,
            collective_kinetic=args.collective_kinetic,
            box_ratio=args.box_ratio,
            gap_cells=args.gap_cells,
            wall_cells=args.wall_cells,
            aspect_ratios=tuple(args.aspect_ratios),
        ),
        template=MacroBlockOptions(
            n_inner=args.n_inner,
            n_outer=args.n_outer,
            inner_grading=args.interface_grading,
            outer_grading=args.interface_grading,
        ),
        thresholds=ValidationThresholds(min_scaled_jacobian=args.min_scaled_jacobian),
        output=OutputConfig(path=output, write_vtk=args.write_vtk),
        pillow=PillowConfig(
            enabled=args.pillow,
            thickness_fraction=args.pillow_thickness,
        ),
        smoothing=SmoothingConfig(
            enabled=args.smoothing,
            n_iterations=args.smoothing_iters,
        ),
        seed=args.seed,
        element_type=args.element_type,
        phase_field_length=args.phase_field_length,
        validate_mesh=not args.no_validate,
        verbose=not args.quiet,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexaform",
        description="Structured all-hexahedral meshing for natural-shaped inclusions.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser(
        "particles-in-cube",
        help="mesh a cube containing randomly placed inclusions",
        description="The primary use case: an Exodus II all-hex RVE mesh.",
    )
    _add_common(p)
    p.add_argument("-o", "--output", default="output/mesh.e")

    p = sub.add_parser("run", help="execute a JSON/YAML configuration file")
    p.add_argument("config", help="path to a RunConfig JSON or YAML file")
    p.add_argument("-o", "--output", default=None, help="override the output path")

    p = sub.add_parser(
        "campaign",
        help="generate many realisations and summarise the success rate",
        description=(
            "A stochastic mesher is judged by its success rate across seeds, "
            "not by one attractive mesh."
        ),
    )
    _add_common(p)
    p.add_argument("--seeds", default="0-19", help="e.g. 0-99 or 0,1,5-9")
    p.add_argument("-o", "--output-dir", default="output/campaign")

    p = sub.add_parser("inspect", help="re-read an Exodus file and report on it")
    p.add_argument("path")
    p.add_argument("--min-scaled-jacobian", type=float, default=0.2)

    p = sub.add_parser(
        "suggest-density",
        help="recommend --mesh-density from geometry and phase-field length",
    )
    p.add_argument("--length", type=float, default=1.0)
    p.add_argument("--radius-mean", type=float, required=True)
    p.add_argument("--phase-field-length", type=float, default=None)
    p.add_argument("--elements-per-radius", type=float, default=4.0)
    p.add_argument("--elements-per-ell", type=float, default=2.0)

    p = sub.add_parser(
        "plate",
        help="rectangular plate, one hex through the thickness, optional conforming through-hole",
    )
    p.add_argument("--width", type=float, default=100.0, help="edge along x")
    p.add_argument("--height", type=float, default=50.0, help="edge along y")
    p.add_argument("--thickness", type=float, default=1.0, help="edge along z (one element)")
    p.add_argument("--h", type=float, default=2.5, help="in-plane background cell size")
    p.add_argument(
        "--hole", default=None,
        help="'circle:D' (diameter) or 'ellipse:2A,2B[,ANGLE_DEG]' (full axes); omit for a plain plate",
    )
    p.add_argument("--centre", type=float, nargs=2, default=None, metavar=("CX", "CY"),
                   help="hole centre (default: plate centre)")
    p.add_argument("--transition", type=int, nargs=2, default=None, metavar=("KX", "KY"),
                   help="O-grid rectangle half-extents in background cells")
    p.add_argument("--ring-cells", type=int, default=2,
                   help="background cells between the hole's bounding box and the O-grid rectangle")
    p.add_argument("--n-r", type=int, default=None, help="radial layers in the ring (auto)")
    p.add_argument("--grading", type=float, default=None, help="outer/inner radial spacing (auto)")
    p.add_argument("-o", "--output", default="output/plate.e")
    p.add_argument("--write-vtk", action="store_true")
    p.add_argument("--summary", default=None, help="write the plate summary JSON here")

    p = sub.add_parser(
        "dogbone",
        help="flat dog-bone tensile bar with shoulder fillets and a thickness profile along the gage",
    )
    p.add_argument("--length", type=float, default=25.4, help="total length along x")
    p.add_argument("--gage-length", type=float, default=8.0)
    p.add_argument("--gage-width", type=float, default=1.85)
    p.add_argument("--grip-width", type=float, default=4.8)
    p.add_argument("--fillet-radius", type=float, default=3.175)
    p.add_argument("--grip-thickness", type=float, default=0.35, help="thickness of the grips (and of the whole bar without stations)")
    p.add_argument("--stations", default=None,
                   help="thickness stations along the gage as 'X:T,X:T,...' (X from the gage start); spline inside the gage")
    p.add_argument("--h", type=float, default=0.2, help="cell size in the gage")
    p.add_argument("--h-grip", type=float, default=None, help="cell size along x in the grips (default 2 h)")
    p.add_argument("--n-z", type=int, default=1, help="elements through the thickness")
    p.add_argument("-o", "--output", default="output/dogbone.e")
    p.add_argument("--write-vtk", action="store_true")
    p.add_argument("--summary", default=None, help="write the summary JSON here")

    p = sub.add_parser("serve", help="start the local browser GUI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--output-dir", default="output/gui")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"hexaform {__version__}")
        return 0
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "particles-in-cube":
        return _cmd_particles(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "campaign":
        return _cmd_campaign(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "suggest-density":
        return _cmd_suggest(args)
    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "plate":
        return _cmd_plate(args)
    if args.command == "dogbone":
        return _cmd_dogbone(args)
    parser.print_help()
    return 1


def _cmd_particles(args: argparse.Namespace) -> int:
    from .pipeline.run import run

    cfg = _config_from_args(args, args.output)
    cfg.assembly.separate_inclusion_blocks = args.separate_blocks
    result = run(cfg)
    if result.validation is not None and not result.validation.passed:
        return 2
    return 0


def parse_hole(text: str | None, centre=None):
    """``'circle:5'`` -> circle of diameter 5; ``'ellipse:7.5,2.5,30'`` -> full axes and angle."""
    from .plate import HoleSpec

    if not text:
        return None
    kind, _, rest = text.partition(":")
    vals = [float(v) for v in rest.split(",") if v.strip()]
    kind = kind.strip().lower()
    if kind == "circle" and len(vals) == 1:
        return HoleSpec("circle", 0.5 * vals[0], centre=centre)
    if kind == "ellipse" and len(vals) in (2, 3):
        ang = vals[2] if len(vals) == 3 else 0.0
        return HoleSpec("ellipse", 0.5 * vals[0], 0.5 * vals[1], centre=centre, angle=ang)
    raise SystemExit(f"cannot parse hole {text!r}: use circle:D or ellipse:2A,2B[,ANGLE]")


def _cmd_plate(args: argparse.Namespace) -> int:
    import json

    from .core.validation import validate
    from .io import exodus
    from .plate import PlateOptions, plate_mesh, plate_summary

    centre = tuple(args.centre) if args.centre is not None else None
    opts = PlateOptions(
        width=args.width, height=args.height, thickness=args.thickness, h=args.h,
        hole=parse_hole(args.hole, centre), ring_cells=args.ring_cells,
        transition=tuple(args.transition) if args.transition else None,
        n_r=args.n_r, grading=args.grading,
    )
    mesh = plate_mesh(opts)
    report = validate(mesh, cell_size=args.h)
    out = Path(args.output)
    exodus.write(mesh, out, title="Hexaform plate")
    if args.write_vtk:
        from .io import vtk

        vtk.write(mesh, out.with_suffix(".vtk"))
    summary = plate_summary(mesh)
    summary["validation"] = report.as_dict()
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, default=str))
    print(mesh.summary())
    print(report)
    return 0 if report.passed else 2


def parse_stations(text: str | None):
    if not text:
        return None
    out = []
    for item in text.split(","):
        x, t = item.split(":")
        out.append((float(x), float(t)))
    return out


def _cmd_dogbone(args: argparse.Namespace) -> int:
    import json

    from .core.validation import validate
    from .dogbone import DogboneOptions, dogbone_mesh, dogbone_summary
    from .io import exodus

    opts = DogboneOptions(
        length=args.length, gage_length=args.gage_length, gage_width=args.gage_width,
        grip_width=args.grip_width, fillet_radius=args.fillet_radius, grip_thickness=args.grip_thickness,
        stations=parse_stations(args.stations), h=args.h, h_grip=args.h_grip, n_z=args.n_z,
    )
    mesh = dogbone_mesh(opts)
    report = validate(mesh, cell_size=args.h)
    out = Path(args.output)
    exodus.write(mesh, out, title="Hexaform dog-bone")
    if args.write_vtk:
        from .io import vtk

        vtk.write(mesh, out.with_suffix(".vtk"))
    summary = dogbone_summary(mesh)
    summary["validation"] = report.as_dict()
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, default=str))
    print(mesh.summary())
    print(report)
    return 0 if report.passed else 2


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline.run import run

    cfg = RunConfig.load(args.config)
    if args.output:
        cfg.output.path = args.output
    result = run(cfg)
    return 0 if (result.validation is None or result.validation.passed) else 2


def _cmd_campaign(args: argparse.Namespace) -> int:
    from .io import report as report_io
    from .pipeline.run import run

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    reports = []
    for seed in seeds:
        cfg = _config_from_args(args, str(out_dir / f"mesh_{seed:05d}.e"))
        cfg.assembly.separate_inclusion_blocks = args.separate_blocks
        cfg.seed = seed
        cfg.verbose = False
        try:
            result = run(cfg)
            reports.append(
                report_io.build_report(
                    config=cfg,
                    mesh=result.mesh,
                    packing=result.packing,
                    validation=result.validation,
                    timings=result.timings,
                )
            )
            ok = result.validation.passed if result.validation else True
            print(
                f"seed {seed:>5}: {'ok  ' if ok else 'FAIL'} "
                f"{result.mesh.n_elements:>9} elems  "
                f"minSJ={result.validation.quality.min_scaled_jacobian:.4f}"
                if result.validation
                else f"seed {seed:>5}: ok",
                flush=True,
            )
        except Exception as exc:  # a failed realisation is data, not a crash
            print(f"seed {seed:>5}: ERROR {exc}", flush=True)
            reports.append({"seed": seed, "success": False})

    csv_path = report_io.write_campaign_csv(out_dir / "campaign.csv", reports)
    summary = report_io.campaign_summary(reports)
    print("\n--- campaign summary ---")
    for key, value in summary.items():
        print(f"  {key:<28} {value}")
    print(f"  csv                          {csv_path}")
    return 0 if summary.get("success_rate", 0.0) == 1.0 else 2


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .core.validation import ValidationThresholds, validate
    from .io import exodus

    mesh = exodus.read(args.path)
    print(mesh.summary())
    report = validate(
        mesh, thresholds=ValidationThresholds(min_scaled_jacobian=args.min_scaled_jacobian)
    )
    print(report)
    return 0 if report.passed else 2


def _cmd_suggest(args: argparse.Namespace) -> int:
    from .core.validation import recommended_mesh_density

    n = recommended_mesh_density(
        args.length,
        args.radius_mean,
        args.phase_field_length,
        args.elements_per_radius,
        args.elements_per_ell,
    )
    print(f"--mesh-density {n}    (h = {args.length / n:.6g})")
    if args.phase_field_length:
        driver = (
            "phase-field length"
            if args.phase_field_length / args.elements_per_ell
            < args.radius_mean / args.elements_per_radius
            else "inclusion curvature"
        )
        print(f"limiting constraint: {driver}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .server.app import serve
    except ImportError as exc:
        print(f"GUI server needs the optional extras: pip install 'hexaform[server]' ({exc})")
        return 1
    serve(host=args.host, port=args.port, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
