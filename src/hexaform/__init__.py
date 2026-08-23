"""Hexaform -- structured all-hexahedral meshes for natural-shaped inclusions.

Hexaform generates conforming, non-voxelated HEX8/HEX20/HEX27 meshes of a
matrix domain containing many curved internal boundaries, and writes them as native
Exodus II files ready for elastoplastic phase-field fracture solvers.

Quick start
-----------
>>> from hexaform import particles_in_cube
>>> result = particles_in_cube(
...     length=1.0, n_particles=12, radius_mean=0.075,
...     radius_std=0.015, mesh_density=40, seed=7, output="rve.e")
>>> result.validation.passed
True

See ``docs/algorithm.md`` for how it works and ``docs/research/`` for the
literature the design is drawn from.
"""

from .core.mesh import HexMesh
from .core.quality import assess, scaled_jacobian
from .core.validation import ValidationThresholds, recommended_mesh_density, validate
from .packing.collective import CollectiveOptions
from .pipeline.config import (
    AssemblyOptions,
    DomainConfig,
    InclusionConfig,
    MacroBlockOptions,
    OutputConfig,
    RunConfig,
)
from .pipeline.run import RunResult, particles_in_cube, run
from .shapes.library import Ellipsoid, Sphere, SphericalHarmonicGrain, Superquadric

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "HexMesh",
    "Sphere",
    "Ellipsoid",
    "Superquadric",
    "SphericalHarmonicGrain",
    "RunConfig",
    "DomainConfig",
    "InclusionConfig",
    "OutputConfig",
    "MacroBlockOptions",
    "AssemblyOptions",
    "ValidationThresholds",
    "CollectiveOptions",
    "RunResult",
    "run",
    "particles_in_cube",
    "validate",
    "assess",
    "scaled_jacobian",
    "recommended_mesh_density",
]
