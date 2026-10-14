#!/usr/bin/env python3
"""Example: a thin plate with a conforming circular or elliptical through-hole.

A 100 x 50 x 1 plate meshed at h = 2.5 with one HEX8 through the thickness;
the hole (diameter 5) is meshed by an O-grid ring whose nodes lie on the
analytic curve. The same call with ``HoleSpec("ellipse", 3.75, 1.25, angle=30)``
gives a rotated elliptical hole. Run::

    python examples/12_plate_with_hole.py [output_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hexaform import HoleSpec, PlateOptions, plate_mesh, plate_summary, validate
from hexaform.io import exodus, vtk

out = Path(sys.argv[1] if len(sys.argv) > 1 else "output/plate")
out.mkdir(parents=True, exist_ok=True)
for name, hole in (("circle", HoleSpec("circle", 2.5)), ("ellipse", HoleSpec("ellipse", 3.75, 1.25, angle=30.0))):
    mesh = plate_mesh(PlateOptions(width=100.0, height=50.0, thickness=1.0, h=2.5, hole=hole))
    report = validate(mesh, cell_size=2.5)
    exodus.write(mesh, out / f"plate_{name}.e")
    vtk.write(mesh, out / f"plate_{name}.vtk")
    (out / f"plate_{name}.json").write_text(json.dumps(plate_summary(mesh), indent=2, default=str))
    print(name, mesh.summary().splitlines()[0], "valid:", report.passed)
