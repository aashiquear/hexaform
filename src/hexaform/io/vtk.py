"""Minimal legacy-VTK writer for visual debugging.

Exodus is the production format; this exists so a mesh can be dropped into
ParaView (or any VTK viewer) without an Exodus reader, and so the test suite can
dump a failing mesh for inspection.  It has no dependencies beyond NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.elements import EXODUS_TO_VTK, VTK_CELL_TYPES
from ..core.mesh import HexMesh


def write(mesh: HexMesh, path: str | Path, binary: bool = False) -> Path:
    """Write an ASCII legacy ``.vtk`` unstructured grid."""
    if binary:  # pragma: no cover - ASCII is enough for debugging
        raise NotImplementedError("binary legacy VTK output is not implemented")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# vtk DataFile Version 3.0",
        "hexaform mesh",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {mesh.n_nodes} double",
    ]
    lines.extend(" ".join(f"{v:.17g}" for v in p) for p in mesh.points)

    n_el = mesh.n_elements
    width = mesh.nodes_per_element
    vtk_cells = mesh.cells[:, EXODUS_TO_VTK[mesh.element_type]]
    lines.append(f"CELLS {n_el} {n_el * (width + 1)}")
    lines.extend(
        f"{width} " + " ".join(str(int(i)) for i in c) for c in vtk_cells
    )
    lines.append(f"CELL_TYPES {n_el}")
    lines.extend([str(VTK_CELL_TYPES[mesh.element_type])] * n_el)

    lines.append(f"CELL_DATA {n_el}")
    lines.append("SCALARS block_id int 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(str(int(b)) for b in mesh.block_id)

    for name, values in mesh.element_fields.items():
        values = np.asarray(values)
        if values.size != n_el:
            continue
        lines.append(f"SCALARS {name} double 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{float(v):.17g}" for v in values)

    path.write_text("\n".join(lines) + "\n")
    return path
