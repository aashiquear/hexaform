"""Declarative configuration for a Hexaform run.

Everything a run needs lives in one serialisable :class:`RunConfig`.  This is
deliberate: the browser GUI builds a block graph, serialises it to this
same structure, and calls the same :func:`hexaform.pipeline.run.run` used by the
CLI.  There is no second code path.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..assembly.builder import AssemblyOptions
from ..core.elements import normalize_element_type
from ..core.validation import ValidationThresholds
from ..packing.collective import CollectiveOptions
from ..packing.rsa import PackingSpec
from ..templates.macroblock import MacroBlockOptions


@dataclass
class SmoothingConfig:
    """Configuration for constrained Jacobian-barrier mesh optimisation.

    Attributes:
        enabled: If True, run a smoothing pass on the final mesh.
        n_iterations: Maximum number of optimisation iterations.
        step_size: Initial gradient step.
        min_scaled_jacobian: Stop early once min scaled Jacobian exceeds this.
        lock_surface: If True, project every surface node back to the analytic body.
        surface_tol: Numerical tolerance for surface-node detection.
    """

    enabled: bool = False
    n_iterations: int = 50
    step_size: float = 0.5
    min_scaled_jacobian: float = 0.5774
    lock_surface: bool = True
    surface_tol: float = 1e-6


@dataclass
class PillowConfig:
    """Configuration for the interface pillow layer."""

    enabled: bool = False
    thickness_fraction: float = 0.15


@dataclass
class DomainConfig:
    """The matrix domain and its background discretisation."""

    length: float = 1.0
    """Cube edge length.  Non-cubic domains use ``divisions`` per axis."""

    mesh_density: int = 32
    """Background cells along each domain edge.  Cell size ``h = length /
    mesh_density``; this is the single knob that controls matrix resolution."""

    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    periodic: bool = False
    """Enable periodic boundary conditions for homogenization/RVE analysis.
    When True, creates paired node sets across opposite domain faces."""

    backend: str = "cartesian"
    """Background backend: ``cartesian`` (uniform grid) or ``octree`` (adaptive).
    The octree is Backend B in the roadmap; it refines near interfaces, thin
    ligaments and the crack region while keeping the bulk coarse."""

    octree_bulk_h: float | None = None
    """Override for the bulk cell size in the octree backend.  Defaults to
    ``length / mesh_density`` when ``None``."""

    octree_interface_h: float | None = None
    """Override for the interface cell size in the octree backend."""

    octree_interface_band: float = 0.05
    """Width of the interface refinement band, as a fraction of the bounding
    radius of every inclusion."""

    octree_max_level: int = 6
    """Maximum octree depth, regardless of size field."""


@dataclass
class InclusionConfig:
    """What to put inside the domain."""

    shape: str = "sphere"
    """``sphere``, ``ellipsoid``, ``superquadric`` or ``sh_grain``."""

    n_particles: int | Tuple[int, int] = 10
    radius_mean: float = 0.08
    radius_std: float = 0.0
    radius_min: float | None = None
    radius_max: float | None = None

    packer: str = "rsa"
    """``rsa`` (random sequential addition, fast, ~10-20% volume fraction) or
    ``collective`` (Lubachevsky-Stillinger / force-biased collective
    rearrangement, places denser populations).  Both emit the same
    guaranteed-meshable macro-boxes."""

    collective_max_steps: int = 2000
    """Upper bound on the collective-rearrangement search (``packer`` ==
    ``collective``).  More steps can only place more inclusions, never fewer."""

    collective_kinetic: float = 0.5
    """Fraction of collective moves that are kinetic (exploratory) rather than
    force-biased (optimising).  0.3-0.7 works well."""

    box_ratio: float = 1.35
    gap_cells: int = 2
    wall_cells: int = 1
    max_tries_per_particle: int = 4000
    allow_shortfall: bool = True

    # shape-specific parameters
    aspect_ratios: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Semi-axis ratios for ellipsoid/superquadric bodies."""

    random_orientation: bool = True
    exponent_xy: float = 3.0
    exponent_z: float = 3.0
    grain_l_max: int = 4
    grain_roughness: float = 0.15

    def __post_init__(self) -> None:
        if self.packer not in ("rsa", "collective"):
            raise ValueError(
                f"unknown packer {self.packer!r}; expected 'rsa' or 'collective'"
            )

    def to_packing_spec(self) -> PackingSpec:
        return PackingSpec(
            n_particles=self.n_particles,
            radius_mean=self.radius_mean,
            radius_std=self.radius_std,
            radius_min=self.radius_min,
            radius_max=self.radius_max,
            box_ratio=self.box_ratio,
            gap_cells=self.gap_cells,
            wall_cells=self.wall_cells,
            max_tries_per_particle=self.max_tries_per_particle,
            allow_shortfall=self.allow_shortfall,
        )

    def to_packing_spec_with_opts(self) -> tuple[PackingSpec, CollectiveOptions | None]:
        """Return the packing spec and options for the selected packer."""
        spec = self.to_packing_spec()
        opts: CollectiveOptions | None = None
        if self.packer == "collective":
            opts = CollectiveOptions(
                max_steps=self.collective_max_steps,
                seed_tries=min(400, self.max_tries_per_particle),  # bound to RSA's common value
                kinetic_fraction=self.collective_kinetic,
            )
        return spec, opts


@dataclass
class OutputConfig:
    path: str = "output/mesh.e"
    title: str = "Hexaform mesh"
    write_report: bool = True
    report_path: str | None = None
    exodus_backend: str = "auto"
    write_vtk: bool = False
    vtk_path: str | None = None


@dataclass
class NotchConfig:
    """Configuration for notch/pre-crack insertion.

    Attributes:
        enabled: Whether to insert a notch.
        notch_type: Type of notch ('edge', 'center', 'chevron').
        origin: Starting point of the notch (x, y, z).
        direction: Primary propagation direction (dx, dy, dz).
        length: Length of the notch.
        depth: Depth of the notch (perpendicular to propagation plane).
        chevron_angle: Angle for chevron notch (degrees).
        method: Insertion method ('field' or 'geometric').
        phase_field_value: Phase-field damage value to assign.
        phase_field_variable: Name of the phase-field element variable.
    """

    enabled: bool = False
    notch_type: str = "edge"
    origin: Tuple[float, float, float] = (0.0, 0.5, 0.5)
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    length: float = 0.5
    depth: float = 1.0
    chevron_angle: float = 45.0
    method: str = "field"
    phase_field_value: float = 1.0
    phase_field_variable: str = "phase_field"


@dataclass
class RunConfig:
    """A complete, reproducible run."""

    domain: DomainConfig = field(default_factory=DomainConfig)
    inclusions: InclusionConfig = field(default_factory=InclusionConfig)
    template: MacroBlockOptions = field(default_factory=MacroBlockOptions)
    assembly: AssemblyOptions = field(default_factory=AssemblyOptions)
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    output: OutputConfig = field(default_factory=OutputConfig)
    notch: NotchConfig = field(default_factory=NotchConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    pillow: PillowConfig = field(default_factory=PillowConfig)

    seed: int = 0
    element_type: str = "HEX8"
    """Homogeneous Exodus element type: ``HEX8``, ``HEX20`` or ``HEX27``."""

    phase_field_length: float | None = None
    """Regularisation length ``ell``.  Only used for reporting ``h/ell``; it
    never changes the mesh silently."""

    validate_mesh: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        self.element_type = normalize_element_type(self.element_type)

    def validate(self) -> None:
        """Reject configurations that cannot produce a meaningful mesh.

        This is deliberately called at the pipeline boundary so JSON, YAML,
        CLI, and browser requests receive the same validation.
        """
        if not math.isfinite(self.domain.length) or self.domain.length <= 0:
            raise ValueError("domain.length must be greater than zero")
        if self.domain.mesh_density < 4:
            raise ValueError("domain.mesh_density must be at least 4")
        if not isinstance(self.domain.origin, (tuple, list)) or len(self.domain.origin) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in self.domain.origin
        ):
            raise ValueError("domain.origin must contain three finite coordinates")
        if self.domain.backend not in ("cartesian", "octree"):
            raise ValueError("domain.backend must be 'cartesian' or 'octree'")
        if self.domain.periodic and self.domain.backend == "octree":
            raise ValueError("periodic boundary conditions are not supported by the octree prototype")
        for name in ("octree_bulk_h", "octree_interface_h"):
            value = getattr(self.domain, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"domain.{name} must be greater than zero when set")
        if not math.isfinite(self.domain.octree_interface_band) or self.domain.octree_interface_band < 0:
            raise ValueError("domain.octree_interface_band cannot be negative")
        if self.domain.octree_max_level < 1:
            raise ValueError("domain.octree_max_level must be at least one")

        count = self.inclusions.n_particles
        if isinstance(count, bool):
            raise ValueError("inclusions.n_particles must be an integer or pair")
        if isinstance(count, int):
            bounds = (count, count)
        elif isinstance(count, (tuple, list)) and len(count) == 2 and all(
            isinstance(v, int) and not isinstance(v, bool) for v in count
        ):
            bounds = tuple(count)
        else:
            raise ValueError("inclusions.n_particles must be an integer or ordered integer pair")
        if bounds[0] < 0 or bounds[1] < bounds[0]:
            raise ValueError("inclusions.n_particles must be non-negative or an ordered (min, max) pair")
        if self.inclusions.shape not in ("sphere", "ellipsoid", "superquadric", "sh_grain", "grain"):
            raise ValueError("inclusions.shape is not a supported star-shaped body")
        if self.inclusions.packer not in ("rsa", "collective"):
            raise ValueError("inclusions.packer must be 'rsa' or 'collective'")
        if bounds[1] > 0 and self.inclusions.radius_mean <= 0:
            raise ValueError("inclusions.radius_mean must be greater than zero when particles are requested")
        if not math.isfinite(self.inclusions.radius_mean):
            raise ValueError("inclusions.radius_mean must be finite")
        if not math.isfinite(self.inclusions.radius_std) or self.inclusions.radius_std < 0:
            raise ValueError("inclusions.radius_std cannot be negative")
        for name in ("radius_min", "radius_max"):
            value = getattr(self.inclusions, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"inclusions.{name} must be greater than zero when set")
        if (self.inclusions.radius_min is not None and self.inclusions.radius_max is not None
                and self.inclusions.radius_max < self.inclusions.radius_min):
            raise ValueError("inclusions.radius_max cannot be below radius_min")
        if not math.isfinite(self.inclusions.box_ratio) or self.inclusions.box_ratio <= 1:
            raise ValueError("inclusions.box_ratio must be greater than one")
        if self.inclusions.gap_cells < 0 or self.inclusions.wall_cells < 0:
            raise ValueError("inclusion gap_cells and wall_cells cannot be negative")
        if self.inclusions.max_tries_per_particle < 1:
            raise ValueError("inclusions.max_tries_per_particle must be at least one")
        if self.inclusions.collective_max_steps < 0:
            raise ValueError("inclusions.collective_max_steps cannot be negative")
        if not 0 <= self.inclusions.collective_kinetic <= 1:
            raise ValueError("inclusions.collective_kinetic must be between zero and one")
        ratios = self.inclusions.aspect_ratios
        if not isinstance(ratios, (tuple, list)) or len(ratios) != 3 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in ratios
        ):
            raise ValueError("inclusions.aspect_ratios must contain three positive finite values")
        if self.inclusions.exponent_xy <= 0 or self.inclusions.exponent_z <= 0:
            raise ValueError("superquadric exponents must be greater than zero")
        if self.inclusions.grain_l_max < 2 or not 0 <= self.inclusions.grain_roughness < 1:
            raise ValueError("grain_l_max must be at least 2 and grain_roughness in [0, 1)")

        if self.template.n_inner < 1 or self.template.n_outer < 1:
            raise ValueError("template radial layer counts must be at least one")
        if not 0 < self.template.core_fraction < 1:
            raise ValueError("template.core_fraction must lie between zero and one")
        if not 0 <= self.template.core_roundness <= 1:
            raise ValueError("template.core_roundness must lie between zero and one")
        if (not math.isfinite(self.template.inner_grading)
                or not math.isfinite(self.template.outer_grading)
                or self.template.inner_grading <= 0 or self.template.outer_grading <= 0):
            raise ValueError("template grading ratios must be greater than zero")
        if self.template.patch_spacing not in ("uniform", "equiangular", "blend"):
            raise ValueError("template.patch_spacing is not supported")
        if not 0 <= self.template.patch_blend <= 1:
            raise ValueError("template.patch_blend must lie between zero and one")

        if self.output.exodus_backend not in ("auto", "netcdf4", "scipy"):
            raise ValueError("output.exodus_backend must be auto, netcdf4, or scipy")
        if not 0 <= self.thresholds.min_scaled_jacobian <= 1:
            raise ValueError("thresholds.min_scaled_jacobian must lie in [0, 1]")
        if (not math.isfinite(self.thresholds.max_edge_ratio)
                or not math.isfinite(self.thresholds.max_volume_error)
                or self.thresholds.max_edge_ratio < 1 or self.thresholds.max_volume_error < 0):
            raise ValueError("quality thresholds must use edge ratio >= 1 and volume error >= 0")
        if self.phase_field_length is not None and self.phase_field_length <= 0:
            raise ValueError("phase_field_length must be greater than zero when set")
        if self.pillow.enabled and not 0 < self.pillow.thickness_fraction < 1:
            raise ValueError("pillow.thickness_fraction must lie between zero and one")
        if self.smoothing.enabled:
            if (self.smoothing.n_iterations < 1 or not math.isfinite(self.smoothing.step_size)
                    or self.smoothing.step_size <= 0):
                raise ValueError("smoothing iterations and step size must be positive")
            if not 0 <= self.smoothing.min_scaled_jacobian <= 1 or self.smoothing.surface_tol <= 0:
                raise ValueError("smoothing quality target/tolerance are invalid")
        if self.notch.enabled:
            if self.notch.notch_type not in ("edge", "center", "chevron"):
                raise ValueError("notch.notch_type is not supported")
            if self.notch.method not in ("field", "geometric"):
                raise ValueError("notch.method must be 'field' or 'geometric'")
            if self.notch.length <= 0 or self.notch.depth <= 0:
                raise ValueError("notch length and depth must be greater than zero")

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("run configuration must be an object")
        kinds = {f.name: f.type for f in fields(cls)}
        del kinds
        sub = {
            "domain": DomainConfig,
            "inclusions": InclusionConfig,
            "template": MacroBlockOptions,
            "assembly": AssemblyOptions,
            "thresholds": ValidationThresholds,
            "output": OutputConfig,
            "notch": NotchConfig,
            "smoothing": SmoothingConfig,
            "pillow": PillowConfig,
        }
        kwargs: Dict[str, Any] = {}
        allowed_top = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key not in allowed_top:
                continue
            if key in sub:
                if not isinstance(value, dict):
                    raise ValueError(f"configuration section {key!r} must be an object")
                kwargs[key] = _build(sub[key], value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "PyYAML is required for YAML configs; "
                    "install hexaform[yaml] or use JSON"
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)


def _build(kind, values: Dict[str, Any]):
    if not is_dataclass(kind):
        return values
    allowed = {f.name for f in fields(kind)}
    return kind(**{k: v for k, v in values.items() if k in allowed})


def _json_default(obj: Any):
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


__all__ = [
    "RunConfig",
    "DomainConfig",
    "InclusionConfig",
    "OutputConfig",
    "MacroBlockOptions",
    "AssemblyOptions",
    "ValidationThresholds",
    "NotchConfig",
    "SmoothingConfig",
    "PillowConfig",
]
