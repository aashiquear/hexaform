"""Block registry for the plug-and-play mesh-graph GUI.

Status: **browser frontend and execution wired to the existing pipeline.**

The browser GUI builds a typed graph from its controls and compiles it to a mesh
generator. This module is the contract between that interface and the Python
package. Every addition to Hexaform should declare its GUI surface at the same
time it is written so the frontend remains a view over the pipeline.

Design rules
------------
1. **One execution path.** A block graph is compiled to a
   :class:`~hexaform.pipeline.config.RunConfig` and handed to the same
   :func:`hexaform.pipeline.run.run` the CLI uses.  The GUI cannot do anything
   the CLI cannot, and cannot diverge from it.
2. **Blocks describe, they do not compute.** A :class:`BlockSpec` is pure
   metadata: ports, parameters, JSON schema.  The computation lives in the
   ordinary package modules.
3. **Ports are typed.** A graph is validated before execution by matching port
   types, so an invalid wiring is rejected in the browser rather than producing a
   confusing Python traceback.

Port types
----------
``domain``
    A background discretisation (grid or, later, octree).
``shapes``
    A population of :class:`~hexaform.shapes.base.StarShape` bodies with their
    macro-boxes.
``mesh``
    A :class:`~hexaform.core.mesh.HexMesh`.
``report``
    A validation/metrics dictionary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Sequence

PortType = Literal["domain", "shapes", "mesh", "report"]


@dataclass
class Port:
    name: str
    type: PortType
    required: bool = True
    description: str = ""


@dataclass
class Param:
    name: str
    type: Literal["float", "int", "bool", "string", "enum", "vector3", "range"]
    default: Any = None
    label: str = ""
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: Sequence[str] | None = None
    config_path: str | None = None
    """Dotted path into :class:`RunConfig` this parameter writes to, e.g.
    ``domain.mesh_density``.  This is what makes graph compilation mechanical."""


@dataclass
class BlockSpec:
    key: str
    title: str
    category: Literal["domain", "shape", "packing", "template", "assembly", "output", "quality"]
    description: str
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    status: Literal["stable", "preview", "planned"] = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# the registry
# ----------------------------------------------------------------------
REGISTRY: Dict[str, BlockSpec] = {}


def register(spec: BlockSpec) -> BlockSpec:
    REGISTRY[spec.key] = spec
    return spec


register(
    BlockSpec(
        key="cube_domain",
        title="Cube domain",
        category="domain",
        description=(
            "An axis-aligned cube discretised by a uniform Cartesian background "
            "grid. Cell size h = length / mesh density."
        ),
        outputs=[Port("domain", "domain", description="background grid")],
        params=[
            Param("length", "float", 1.0, "Edge length", config_path="domain.length",
                  minimum=0.0),
            Param("mesh_density", "int", 32, "Background cells per edge",
                  config_path="domain.mesh_density", minimum=4, maximum=512),
            Param("periodic", "bool", False, "Periodic boundary pairs",
                  config_path="domain.periodic"),
        ],
    )
)

register(
    BlockSpec(
        key="octree_domain",
        title="Adaptive octree domain",
        category="domain",
        description=(
            "Experimental Backend B research path. Adaptive leaves are generated, "
            "but conforming transition templates and robust boundary fitting are not "
            "complete; star-shaped production runs use Cartesian macroblocks."
        ),
        outputs=[Port("domain", "domain")],
        params=[
            Param("length", "float", 1.0, "Edge length", config_path="domain.length",
                  minimum=0.0),
            Param("mesh_density", "int", 32, "Fallback background cells per edge",
                  config_path="domain.mesh_density", minimum=4, maximum=512),
            Param("bulk_h", "float", 0.05, "Coarse cell size",
                  config_path="domain.octree_bulk_h", minimum=1e-9),
            Param("interface_h", "float", 0.02, "Cell size at the interface",
                  config_path="domain.octree_interface_h", minimum=1e-9),
            Param("interface_band", "float", 0.05, "Width of the interface band",
                  config_path="domain.octree_interface_band", minimum=0.0),
            Param("max_level", "int", 6, "Maximum octree depth",
                  config_path="domain.octree_max_level", minimum=1, maximum=16),
        ],
        status="preview",
    )
)

register(
    BlockSpec(
        key="superquadric_source",
        title="Superquadrics",
        category="shape",
        description="Rounded aggregate bodies with analytic star-shaped surfaces.",
        outputs=[Port("shapes", "shapes")],
        params=[
            Param("n_particles", "range", 10, config_path="inclusions.n_particles"),
            Param("radius_mean", "float", 0.08, config_path="inclusions.radius_mean",
                  minimum=1e-9),
            Param("radius_std", "float", 0.0, config_path="inclusions.radius_std",
                  minimum=0.0),
            Param("aspect_ratios", "vector3", (1.0, 0.9, 0.75),
                  config_path="inclusions.aspect_ratios"),
            Param("exponent_xy", "float", 3.0, config_path="inclusions.exponent_xy",
                  minimum=1.0, maximum=12.0),
            Param("exponent_z", "float", 3.0, config_path="inclusions.exponent_z",
                  minimum=1.0, maximum=12.0),
        ],
    )
)

register(
    BlockSpec(
        key="sphere_source",
        title="Spheres",
        category="shape",
        description="Analytic spheres with a truncated-normal radius distribution.",
        outputs=[Port("shapes", "shapes")],
        params=[
            Param("n_particles", "range", 10, "Particle count",
                  config_path="inclusions.n_particles",
                  description="A fixed number, or a range sampled per seed."),
            Param("radius_mean", "float", 0.08, "Mean radius",
                  config_path="inclusions.radius_mean", minimum=0.0),
            Param("radius_std", "float", 0.0, "Radius std. dev.",
                  config_path="inclusions.radius_std", minimum=0.0),
        ],
    )
)

register(
    BlockSpec(
        key="ellipsoid_source",
        title="Ellipsoids",
        category="shape",
        description="Randomly oriented ellipsoids with fixed semi-axis ratios.",
        outputs=[Port("shapes", "shapes")],
        params=[
            Param("n_particles", "range", 10, config_path="inclusions.n_particles"),
            Param("radius_mean", "float", 0.08, config_path="inclusions.radius_mean"),
            Param("radius_std", "float", 0.0, config_path="inclusions.radius_std"),
            Param("aspect_ratios", "vector3", (1.0, 0.75, 0.6),
                  config_path="inclusions.aspect_ratios"),
            Param("random_orientation", "bool", True,
                  config_path="inclusions.random_orientation"),
        ],
    )
)

register(
    BlockSpec(
        key="grain_source",
        title="Natural grains",
        category="shape",
        description=(
            "Irregular aggregate/sand grains: a sphere modulated by random real "
            "spherical harmonics with l^-beta coefficient decay."
        ),
        outputs=[Port("shapes", "shapes")],
        params=[
            Param("n_particles", "range", 10, config_path="inclusions.n_particles"),
            Param("radius_mean", "float", 0.08, config_path="inclusions.radius_mean"),
            Param("radius_std", "float", 0.0, config_path="inclusions.radius_std"),
            Param("grain_roughness", "float", 0.15, "Roughness",
                  config_path="inclusions.grain_roughness", minimum=0.0, maximum=0.5),
            Param("grain_l_max", "int", 4, "Max harmonic degree",
                  config_path="inclusions.grain_l_max", minimum=2, maximum=12),
        ],
    )
)

register(
    BlockSpec(
        key="rsa_packing",
        title="Random packing (RSA)",
        category="packing",
        description=(
            "Grid-aware random sequential addition. Saturates around 10-20% volume fraction.\n"
            "For denser populations (40-50%), use collective rearrangement via\n"
            "the CLI --packer=collective flag or GUI collective_max_steps parameter.\n"
            "Collective packing uses Lubachevsky-Stillinger / force-biased methods."
        ),
        inputs=[Port("domain", "domain"), Port("shapes", "shapes")],
        outputs=[Port("shapes", "shapes", description="placed inclusions")],
        params=[
            Param("packer", "enum", "rsa", "Packing algorithm",
                  config_path="inclusions.packer", choices=["rsa", "collective"]),
            Param("box_ratio", "float", 1.35, "Macro-box ratio",
                  config_path="inclusions.box_ratio", minimum=1.05, maximum=2.5),
            Param("gap_cells", "int", 2, "Min elements across a ligament",
                  config_path="inclusions.gap_cells", minimum=1),
            Param("wall_cells", "int", 1, "Min cells to the wall",
                  config_path="inclusions.wall_cells", minimum=0),
            Param("seed", "int", 0, "Random seed", config_path="seed"),
        ],
    )
)

register(
    BlockSpec(
        key="macroblock_template",
        title="Cubed-sphere macro-block",
        category="template",
        description=(
            "Per-inclusion O-grid: transfinite core, radial inner shell to the "
            "analytic surface, radial outer shell to the grid-aligned box face."
        ),
        inputs=[Port("domain", "domain"), Port("shapes", "shapes")],
        outputs=[Port("mesh", "mesh")],
        params=[
            Param(
                "element_type", "enum", "HEX8", "Element type",
                choices=("HEX8", "HEX20", "HEX27"), config_path="element_type",
            ),
            Param("n_inner", "int", 2, "Inner radial layers",
                  config_path="template.n_inner", minimum=1, maximum=8),
            Param("n_outer", "int", 2, "Outer radial layers",
                  config_path="template.n_outer", minimum=1, maximum=8),
            Param("inner_grading", "float", 1.0, "Inner grading ratio",
                  config_path="template.inner_grading", minimum=1.0, maximum=8.0),
            Param("outer_grading", "float", 1.0, "Outer grading ratio",
                  config_path="template.outer_grading", minimum=1.0, maximum=8.0),
            Param("core_fraction", "float", 0.55, "Core size fraction",
                  config_path="template.core_fraction", minimum=0.2, maximum=0.75),
            Param("core_roundness", "float", 0.35, "Core roundness",
                  config_path="template.core_roundness", minimum=0.0, maximum=0.7),
        ],
    )
)

register(
    BlockSpec(
        key="pillow_layer",
        title="Interface pillow",
        category="template",
        description=(
            "Insert an extra hex sheet at material interfaces so no poorly "
            "shaped element terminates directly on the interface.  "
            "Especially important for the octree backend where background "
            "cells can cut the interface at an arbitrary angle."
        ),
        inputs=[Port("mesh", "mesh")],
        outputs=[Port("mesh", "mesh")],
        params=[
            Param("thickness_fraction", "float", 0.15,
                  config_path="pillow.thickness_fraction", minimum=0.0, maximum=1.0),
        ],
        status="stable",
    )
)

register(
    BlockSpec(
        key="quality_gate",
        title="Quality gate",
        category="quality",
        description=(
            "Topology, validity, quality, geometric fidelity and ligament "
            "checks. Fails the graph rather than writing an unusable mesh."
        ),
        inputs=[Port("mesh", "mesh")],
        outputs=[Port("mesh", "mesh"), Port("report", "report")],
        params=[
            Param("min_scaled_jacobian", "float", 0.2,
                  config_path="thresholds.min_scaled_jacobian", minimum=0.0, maximum=1.0),
            Param("max_edge_ratio", "float", 12.0,
                  config_path="thresholds.max_edge_ratio", minimum=1.0),
            Param("max_volume_error", "float", 0.05,
                  config_path="thresholds.max_volume_error", minimum=0.0),
        ],
    )
)

register(
    BlockSpec(
        key="quality_smoothing",
        title="Constrained mesh smoothing",
        category="quality",
        description=(
            "Knupp-style Jacobian-barrier optimisation with interface nodes "
            "locked to the analytic surface. Essential once the octree "
            "backend allows close particle approach."
        ),
        inputs=[Port("mesh", "mesh")],
        outputs=[Port("mesh", "mesh")],
        params=[
            Param("n_iterations", "int", 50, "Maximum iterations",
                  config_path="smoothing.n_iterations", minimum=1, maximum=500),
            Param("step_size", "float", 0.5, "Initial gradient step",
                  config_path="smoothing.step_size", minimum=0.0, maximum=5.0),
            Param("min_scaled_jacobian", "float", 0.5774,
                  config_path="smoothing.min_scaled_jacobian",
                  minimum=0.0, maximum=1.0),
        ],
        status="stable",
    )
)

register(
    BlockSpec(
        key="non_star_body",
        title="Non-star-shaped body (torus, ...)",
        category="shape",
        description=(
            "Tori, branching pore networks and interlocking fibres. Requires "
            "the octree backend plus surface projection against the body's "
            "implicit function. The frontier of the 'arbitrary shape' goal."
        ),
        outputs=[Port("shapes", "shapes")],
        params=[
            Param("shape", "enum", "torus", "Body family",
                  choices=("torus",)),
            Param("major_radius", "float", 0.25, "Major radius (torus)"),
            Param("minor_radius", "float", 0.1, "Minor radius (torus)"),
        ],
        status="planned",
    )
)

register(
    BlockSpec(
        key="exodus_output",
        title="Exodus II output",
        category="output",
        description="Native Exodus II writer with blocks, side sets and node sets.",
        inputs=[Port("mesh", "mesh"), Port("report", "report", required=False)],
        outputs=[],
        params=[
            Param("path", "string", "output/mesh.e", config_path="output.path"),
            Param("separate_inclusion_blocks", "bool", False,
                  config_path="assembly.separate_inclusion_blocks"),
            Param("write_report", "bool", True, config_path="output.write_report"),
            Param("write_vtk", "bool", True, config_path="output.write_vtk"),
        ],
    )
)


def catalogue() -> List[Dict[str, Any]]:
    """Full block catalogue, JSON-serialisable, for the GUI palette."""
    return [spec.to_dict() for spec in REGISTRY.values()]


# ----------------------------------------------------------------------
# graph -> config compilation
# ----------------------------------------------------------------------
def validate_graph(graph: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems; empty means the graph is valid."""
    problems: List[str] = []
    raw_nodes = graph.get("nodes", [])
    if not isinstance(raw_nodes, list):
        return ["graph.nodes must be a list"]
    ids = [n.get("id") for n in raw_nodes if isinstance(n, dict)]
    duplicate_ids = sorted({node_id for node_id in ids if node_id and ids.count(node_id) > 1})
    for node_id in duplicate_ids:
        problems.append(f"duplicate node id {node_id!r}")
    nodes = {
        n.get("id"): n for n in raw_nodes
        if isinstance(n, dict) and isinstance(n.get("id"), str) and n.get("id")
    }
    if len(nodes) != len(raw_nodes):
        problems.append("every graph node must be an object with a non-empty string id")
    for node in nodes.values():
        if node.get("block") not in REGISTRY:
            problems.append(f"node {node.get('id')!r}: unknown block {node.get('block')!r}")
            continue
        spec = REGISTRY[node["block"]]
        values = node.get("params", {})
        if not isinstance(values, dict):
            problems.append(f"node {node['id']!r}: params must be an object")
            continue
        known = {p.name for p in spec.params}
        for unknown in sorted(set(values) - known):
            problems.append(f"node {node['id']!r}: unknown parameter {unknown!r}")
        for param in spec.params:
            if param.name in values:
                problems.extend(_validate_param(node["id"], param, values[param.name]))

    raw_edges = graph.get("edges", [])
    if not isinstance(raw_edges, list):
        problems.append("graph.edges must be a list")
        raw_edges = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            problems.append("every graph edge must be an object")
            continue
        src, dst = nodes.get(edge.get("from")), nodes.get(edge.get("to"))
        if src is None or dst is None:
            problems.append(f"edge {edge} refers to a missing node")
            continue
        if src.get("block") not in REGISTRY or dst.get("block") not in REGISTRY:
            continue
        out_types = {p.type for p in REGISTRY[src["block"]].outputs}
        in_types = {p.type for p in REGISTRY[dst["block"]].inputs}
        if not (out_types & in_types):
            problems.append(
                f"cannot wire {src['block']} -> {dst['block']}: "
                f"outputs {sorted(out_types)} do not match inputs {sorted(in_types)}"
            )

    categories = {REGISTRY[n["block"]].category for n in nodes.values() if n.get("block") in REGISTRY}
    for needed in ("domain", "shape", "output"):
        if needed not in categories:
            problems.append(f"graph has no {needed} block")

    planned = [
        n["id"] for n in nodes.values()
        if n.get("block") in REGISTRY and REGISTRY[n["block"]].status == "planned"
    ]
    for node_id in planned:
        problems.append(f"node {node_id!r} uses a block that is not implemented yet")
    return problems


def _validate_param(node_id: str, param: Param, value: Any) -> List[str]:
    """Validate one JSON parameter using the block catalogue schema."""
    problems: List[str] = []
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid_type = {
        "float": numeric,
        "int": isinstance(value, int) and not isinstance(value, bool),
        "bool": isinstance(value, bool),
        "string": isinstance(value, str),
        "enum": isinstance(value, str),
        "vector3": isinstance(value, (list, tuple)) and len(value) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value),
        "range": (
            isinstance(value, int) and not isinstance(value, bool)
        ) or (
            isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        ),
    }[param.type]
    if not valid_type:
        return [f"node {node_id!r}: parameter {param.name!r} must be {param.type}"]
    if param.choices is not None and value not in param.choices:
        problems.append(
            f"node {node_id!r}: parameter {param.name!r} must be one of {list(param.choices)}"
        )
    if numeric and param.minimum is not None and value < param.minimum:
        problems.append(f"node {node_id!r}: parameter {param.name!r} is below {param.minimum}")
    if numeric and param.maximum is not None and value > param.maximum:
        problems.append(f"node {node_id!r}: parameter {param.name!r} is above {param.maximum}")
    return problems


def compile_graph(graph: Dict[str, Any]):
    """Compile a validated block graph into a :class:`RunConfig`.

    Parameter values are written through each :attr:`Param.config_path`, so a new
    block becomes GUI-drivable simply by declaring where its parameters live.
    """
    from ..pipeline.config import RunConfig

    problems = validate_graph(graph)
    if problems:
        raise ValueError("invalid block graph:\n  " + "\n  ".join(problems))

    config = RunConfig()
    for node in graph.get("nodes", []):
        spec = REGISTRY[node["block"]]
        values = node.get("params", {})
        if spec.key.endswith("_source"):
            config.inclusions.shape = {
                "sphere_source": "sphere",
                "ellipsoid_source": "ellipsoid",
                "superquadric_source": "superquadric",
                "grain_source": "sh_grain",
            }.get(spec.key, "sphere")
        if spec.key == "cube_domain":
            config.domain.backend = "cartesian"
        elif spec.key == "octree_domain":
            config.domain.backend = "octree"
        elif spec.key == "pillow_layer":
            config.pillow.enabled = True
        elif spec.key == "quality_smoothing":
            config.smoothing.enabled = True
        for param in spec.params:
            if param.config_path is None:
                continue
            _set_path(config, param.config_path, values.get(param.name, param.default))
    config.validate()
    return config


def _set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


__all__ = [
    "BlockSpec",
    "Port",
    "Param",
    "REGISTRY",
    "register",
    "catalogue",
    "validate_graph",
    "compile_graph",
]
