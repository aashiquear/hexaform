"""Block registry and HTTP backend for the browser GUI.

``blocks`` has no dependencies beyond the standard library and is safe to import
anywhere; ``app`` needs the optional ``hexaform[server]`` extras.
"""

from .blocks import REGISTRY, BlockSpec, Param, Port, catalogue, compile_graph, validate_graph

__all__ = [
    "REGISTRY",
    "BlockSpec",
    "Port",
    "Param",
    "catalogue",
    "validate_graph",
    "compile_graph",
]
