#!/usr/bin/env python3
"""Run every example with bounded smoke-test settings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "examples-smoke"
CASES = [
    ("01_spheres_in_cube.py", "--length", "1.0", "--mesh-density", "8", "--n-particles", "1", "--radius-mean", "0.05", "--output", str(OUT / "01.e")),
    ("02_single_sphere_convergence.py",),
    ("03_natural_shapes.py",),
    ("04_monte_carlo_campaign.py", "--seeds", "1", "--mesh-density", "8", "--out-dir", str(OUT / "campaign")),
    ("05_periodic_rve.py", "--length", "1.0", "--mesh-density", "8", "--n-particles", "1", "--output", str(OUT / "05.e")),
    ("06_denser_packing.py", "--length", "1.0", "--mesh-density", "8", "--n-particles", "2", "--output", str(OUT / "06.e")),
    ("07_notch_insertion.py", "--mesh-density", "8", "--n-particles", "0", "--output", str(OUT / "07.e")),
    ("08_octree_backend.py", "--mesh-density", "8", "--output", str(OUT / "08.e")),
    ("09_pillow_layer.py", "--mesh-density", "8", "--output", str(OUT / "09.e")),
    ("10_constrained_smoothing.py", "--mesh-density", "8", "--output", str(OUT / "10.e")),
    ("11_torus.py", "--interface-h", "0.125", "--max-level", "3"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        script, *args = case
        command = [sys.executable, str(ROOT / "examples" / script), *args]
        print(f"\n==> {script}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"\nAll {len(CASES)} examples completed; artifacts: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
