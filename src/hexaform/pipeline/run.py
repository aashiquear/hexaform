"""The Hexaform pipeline.

::

    RunConfig
       |
       +-- background grid          (integer index space)
       +-- packing                  (grid-aware RSA -> shapes + macro-boxes)
       +-- macro-block templates    (cubed-sphere O-grid per inclusion)
       +-- assembly + exact weld    (conforming all-hex mesh)
       +-- validation               (topology, validity, quality, geometry)
       +-- Exodus II                (native writer) + JSON report

Every stage is independently importable and testable; this module only wires
them together.  A different background (octree) or a different template
(non-star-shaped body) drops into the corresponding slot without touching the
others.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..assembly.builder import assemble
from ..assembly.notch import insert_notch, NotchGeometry, NotchOptions, NotchType
from ..assembly.octree_assembly import OctreeAssemblyOptions, assemble_with_octree
from ..assembly.pillow import PillowOptions, pillow_layer
from ..background.cartesian import CartesianGrid
from ..core.elements import normalize_element_type
from ..core.mesh import HexMesh
from ..core.smoothing import SmoothingOptions, detect_surface_nodes, smooth_surface
from ..core.validation import ValidationReport, validate
from ..io import exodus, report as report_io, vtk
from ..packing.rsa import PackedInclusion, PackingResult, pack as rsa_pack
from ..shapes.base import StarShape
from ..shapes.library import Ellipsoid, Sphere, SphericalHarmonicGrain, Superquadric
from ..shapes.torus import Torus
from .config import InclusionConfig, RunConfig


@dataclass
class RunResult:
    mesh: HexMesh
    packing: PackingResult
    validation: Optional[ValidationReport]
    config: RunConfig
    output_path: Optional[Path]
    report_path: Optional[Path]
    vtk_path: Optional[Path]
    timings: dict

    @property
    def inclusions(self) -> List[PackedInclusion]:
        return self.packing.inclusions


# ----------------------------------------------------------------------
# shape factories
# ----------------------------------------------------------------------
def make_shape_factory(
    cfg: InclusionConfig,
) -> Callable[[np.ndarray, float, np.random.Generator], StarShape]:
    """Build the ``(centre, radius, rng) -> StarShape`` callable for a config.

    ``radius`` is always interpreted as the **bounding** radius, so macro-box
    sizing in the packer is identical for every shape family.
    """
    kind = cfg.shape.lower()
    ratios = np.asarray(cfg.aspect_ratios, dtype=float)
    ratios = ratios / max(ratios.max(), 1e-12)

    def _rotation(rng: np.random.Generator) -> np.ndarray | None:
        if not cfg.random_orientation:
            return None
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    if kind == "sphere":
        return lambda c, r, rng: Sphere(c, r)
    if kind == "ellipsoid":
        return lambda c, r, rng: Ellipsoid(c, r * ratios, _rotation(rng))
    if kind == "superquadric":
        return lambda c, r, rng: Superquadric(
            c, r * ratios, cfg.exponent_xy, cfg.exponent_z, _rotation(rng)
        )
    if kind in ("sh_grain", "grain"):
        return lambda c, r, rng: SphericalHarmonicGrain.random(
            c,
            r / (1.0 + cfg.grain_roughness),
            rng,
            l_max=cfg.grain_l_max,
            roughness=cfg.grain_roughness,
        )
    raise ValueError(
        f"unknown inclusion shape {cfg.shape!r}; "
        f"expected one of sphere, ellipsoid, superquadric, sh_grain"
    )


# ----------------------------------------------------------------------
def run(config: RunConfig) -> RunResult:
    """Execute a full configuration and (optionally) write Exodus + report."""
    config.validate()
    config.element_type = normalize_element_type(config.element_type)
    log = _logger(config.verbose)
    timings: dict = {}
    t_all = time.perf_counter()

    # ---- 1. background grid -----------------------------------------
    grid = CartesianGrid.cube(
        config.domain.length, config.domain.mesh_density, config.domain.origin
    )
    log(
        f"domain      : {config.domain.length} cube, "
        f"{config.domain.mesh_density}^3 background cells (h = {grid.h:.6g})"
    )

    # ---- 2. packing --------------------------------------------------
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    spec, opts = config.inclusions.to_packing_spec_with_opts()
    if config.inclusions.packer == "collective":
        from ..packing.collective import pack as coll_pack
        packing = coll_pack(
            grid, spec, rng, shape_factory=make_shape_factory(config.inclusions), options=opts
        )
    else:
        packing = rsa_pack(
            grid, spec, rng, shape_factory=make_shape_factory(config.inclusions)
        )
    timings["packing"] = time.perf_counter() - t0
    s = packing.stats
    log(
        f"packing     : {s['placed']}/{s['requested']} "
        f"{config.inclusions.shape}s, volume fraction {s['volume_fraction']:.4f}"
        + (f", {s['rejected']} rejected" if s.get("rejected") else "")
    )
    if packing.count == 0:
        log("warning     : no inclusions placed - producing a plain matrix mesh")

    # ---- 4. templates + assembly ---------------------------------
    t0 = time.perf_counter()
    if config.domain.backend == "octree":
        # Backend B: adaptive octree + analytic surface projection.
        # Use the cubed-sphere macroblock if every shape is a StarShape;
        # otherwise fall back to the pure-octree path (handles tori etc.).
        shapes_only = [inc.shape for inc in packing.inclusions]
        all_star = all(isinstance(s, StarShape) for s in shapes_only)
        if all_star:
            # Use the cubed-sphere assembly for the matrix/inclusion
            # interface, then optionally add the octree as a refinement.
            mesh = assemble(
                grid,
                packing.inclusions,
                config.template,
                config.assembly,
                element_type=config.element_type,
                periodic=config.domain.periodic,
            )
            mesh.metadata.update(
                {
                    "backend_requested": "octree",
                    "background": "cartesian",
                    "adaptive_refinement_applied": False,
                    "backend_note": (
                        "Star-shaped inclusions currently use the conforming Cartesian "
                        "macroblock path; adaptive octree transition templates remain research work."
                    ),
                }
            )
            log(
                "backend     : octree requested; using conforming Cartesian macroblocks "
                "(adaptive transition templates are not implemented)"
            )
        else:
            # Pure octree path: a non-star shape is present.
            opts = OctreeAssemblyOptions(
                bulk_h=config.domain.octree_bulk_h or grid.h,
                interface_h=config.domain.octree_interface_h or grid.h / 2,
                interface_band=config.domain.octree_interface_band,
                max_level=config.domain.octree_max_level,
            )
            mesh = assemble_with_octree(
                shapes_only,
                domain_length=config.domain.length,
                domain_origin=np.asarray(config.domain.origin),
                options=opts,
            )
            log(
                f"backend     : octree (non-star path, "
                f"bulk={opts.bulk_h:.4g}, interface={opts.interface_h:.4g})"
            )
    else:
        mesh = assemble(
            grid,
            packing.inclusions,
            config.template,
            config.assembly,
            element_type=config.element_type,
            periodic=config.domain.periodic,
        )
        log(f"backend     : cartesian (uniform {config.domain.mesh_density}^3 grid)")
    timings["assembly"] = time.perf_counter() - t0
    log(
        f"mesh        : {mesh.n_elements} {mesh.element_type} elements, "
        f"{mesh.n_nodes} nodes ({timings['assembly']:.2f} s)"
    )

    # ---- 4b. notch insertion -----------------------------------------
    if config.notch.enabled:
        t0 = time.perf_counter()
        # Map notch type string to enum
        notch_type_map = {
            "edge": NotchType.EDGE,
            "center": NotchType.CENTER,
            "chevron": NotchType.CHEVRON,
        }
        notch_type = notch_type_map.get(config.notch.notch_type.lower(), NotchType.EDGE)

        notch_geom = NotchGeometry(
            notch_type=notch_type,
            origin=config.notch.origin,
            direction=config.notch.direction,
            length=config.notch.length,
            depth=config.notch.depth,
            chevron_angle=config.notch.chevron_angle,
        )

        notch_opts = NotchOptions(
            method=config.notch.method,
            phase_field_value=config.notch.phase_field_value,
            variable_name=config.notch.phase_field_variable,
        )

        mesh = insert_notch(
            mesh,
            notch_geom,
            notch_opts,
            domain_extent=(config.domain.length, config.domain.length, config.domain.length),
            domain_origin=config.domain.origin,
        )
        timings["notch"] = time.perf_counter() - t0
        n_notch = mesh.metadata.get("n_notch_elements", 0)
        log(f"notch       : {config.notch.notch_type} notch, {n_notch} elements ({timings['notch']:.2f} s)")

    # ---- 4c. pillow layer -----------------------------------------
    if config.pillow.enabled:
        t0 = time.perf_counter()
        mesh = pillow_layer(
            mesh,
            options=PillowOptions(thickness_fraction=config.pillow.thickness_fraction),
            matrix_block_id=1,
            inclusion_block_id=100,
            shapes=[inc.shape for inc in packing.inclusions],
        )
        timings["pillow"] = time.perf_counter() - t0
        n_pillow = mesh.metadata.get("n_pillow_elements", 0)
        log(f"pillow      : {n_pillow} elements added ({timings['pillow']:.2f} s)")

    # ---- 4d. constrained smoothing ----------------------------------
    if config.smoothing.enabled:
        t0 = time.perf_counter()
        surface = detect_surface_nodes(mesh, packing.inclusions, tol=config.smoothing.surface_tol)
        shapes = [inc.shape for inc in packing.inclusions]

        def _project(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            out = np.empty_like(x)
            # Snap each input point to the closest analytic surface of any shape.
            best_dist = np.full(x.shape[0], np.inf)
            best_pt = x.copy()
            for s in shapes:
                proj = s.project(x)
                d = np.linalg.norm(proj - x, axis=1)
                better = d < best_dist
                if better.any():
                    best_dist[better] = d[better]
                    best_pt[better] = proj[better]
            return best_pt

        sopts = SmoothingOptions(
            n_iterations=config.smoothing.n_iterations,
            step_size=config.smoothing.step_size,
            min_scaled_jacobian=config.smoothing.min_scaled_jacobian,
            lock_surface=config.smoothing.lock_surface,
            surface_tol=config.smoothing.surface_tol,
        )
        sres = smooth_surface(mesh, surface, sopts, surface_projector=_project)
        mesh = sres.mesh
        timings["smoothing"] = time.perf_counter() - t0
        log(
            f"smoothing   : {sres.n_iterations} iters, minSJ "
            f"{sres.initial_min_scaled_jacobian:.4f} -> {sres.final_min_scaled_jacobian:.4f}"
        )

    # ---- 5. validation ------------------------------------------------
    validation = None
    if config.validate_mesh:
        t0 = time.perf_counter()
        validation = validate(
            mesh,
            packing.inclusions,
            cell_size=grid.h,
            thresholds=config.thresholds,
            phase_field_length=config.phase_field_length,
        )
        timings["validation"] = time.perf_counter() - t0
        log(str(validation))

    # ---- 6. output ----------------------------------------------------
    out_path = None
    rep_path = None
    vtk_path = None
    if config.output.path:
        t0 = time.perf_counter()
        out_path = exodus.write(
            mesh,
            config.output.path,
            title=config.output.title,
            backend=config.output.exodus_backend,
        )
        timings["exodus_write"] = time.perf_counter() - t0
        log(f"wrote       : {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
        if config.output.write_vtk:
            vtk_path = vtk.write(
                mesh,
                config.output.vtk_path or Path(out_path).with_suffix(".vtk"),
            )
            timings["vtk_write"] = time.perf_counter() - t0 - timings["exodus_write"]
            log(f"vtk         : {vtk_path} ({vtk_path.stat().st_size / 1e6:.2f} MB)")

    timings["total"] = time.perf_counter() - t_all
    mesh.metadata["timings"] = timings
    mesh.metadata["seed"] = config.seed
    mesh.metadata["packing"] = packing.stats

    if config.output.write_report and out_path is not None:
        rep_path = report_io.write_report(
            config.output.report_path or Path(out_path).with_suffix(".report.json"),
            config=config,
            mesh=mesh,
            packing=packing,
            validation=validation,
            timings=timings,
        )
        log(f"report      : {rep_path}")

    return RunResult(mesh, packing, validation, config, out_path, rep_path, vtk_path, timings)


def particles_in_cube(
    length: float = 1.0,
    n_particles: int | Sequence[int] = 10,
    radius_mean: float = 0.08,
    radius_std: float = 0.0,
    mesh_density: int = 32,
    seed: int = 0,
    output: str | None = "output/mesh.e",
    shape: str = "sphere",
    element_type: str = "HEX8",
    **kwargs,
) -> RunResult:
    """Convenience entry point for the primary use case.

    Generates an all-hexahedral Exodus mesh of a cube containing randomly placed,
    randomly sized inclusions.

    Parameters
    ----------
    length:
        Cube edge length.
    n_particles:
        Exact count, or a ``(lo, hi)`` pair to draw a random count per seed.
    radius_mean, radius_std:
        Truncated-normal radius distribution.
    mesh_density:
        Background cells per domain edge.
    seed:
        Reproducibility seed; the same seed always gives the same mesh.
    output:
        Exodus path, or ``None`` to skip writing.
    shape:
        ``sphere`` (default), ``ellipsoid``, ``superquadric`` or ``sh_grain``.
    element_type:
        ``HEX8`` (default), ``HEX20`` or ``HEX27``.
    backend:
        ``cartesian`` (default) or ``octree`` for the adaptive Backend B.
    pillow:
        ``True`` to insert an interface pillow layer.
    smoothing:
        ``True`` to run constrained Jacobian-barrier mesh optimisation.
    """
    from .config import DomainConfig, InclusionConfig, OutputConfig, PillowConfig, SmoothingConfig

    n = tuple(n_particles) if isinstance(n_particles, (list, tuple)) else int(n_particles)
    cfg = RunConfig(
        domain=DomainConfig(
            length=length,
            mesh_density=mesh_density,
            periodic=kwargs.get("periodic", False),
            backend=kwargs.get("backend", "cartesian"),
        ),
        inclusions=InclusionConfig(
            shape=shape,
            n_particles=n,
            radius_mean=radius_mean,
            radius_std=radius_std,
            **{k: v for k, v in kwargs.items() if k in InclusionConfig.__dataclass_fields__},
        ),
        output=OutputConfig(path=output or ""),
        pillow=PillowConfig(
            enabled=kwargs.get("pillow", False),
            thickness_fraction=kwargs.get("pillow_thickness", 0.15),
        ),
        smoothing=SmoothingConfig(
            enabled=kwargs.get("smoothing", False),
            n_iterations=kwargs.get("smoothing_iters", 50),
        ),
        seed=seed,
        element_type=element_type,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ("phase_field_length", "verbose", "validate_mesh")
        },
    )
    return run(cfg)


def _logger(verbose: bool):
    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    return log
