"""Inclusion geometry.

Implement :class:`~hexaform.shapes.base.StarShape` to add a new body that the
cubed-sphere macro-block template can mesh; the template needs nothing else.

Non-star-shaped bodies (:class:`~hexaform.shapes.base.Shape`, e.g. the
:class:`~hexaform.shapes.torus.Torus`) require the octree backend
(``backend=octree``) and surface projection, because no single
``support_radius`` exists.  They are the target of roadmap item 6.
"""

from .base import Shape, StarShape, normalize
from .library import Ellipsoid, Sphere, SphericalHarmonicGrain, Superquadric
from .torus import Torus, project_to_torus

__all__ = [
    "Shape",
    "StarShape",
    "normalize",
    "Sphere",
    "Ellipsoid",
    "Superquadric",
    "SphericalHarmonicGrain",
    "Torus",
    "project_to_torus",
]
