# Hexaform

Hexaform generates conforming all-hexahedral meshes for cuboidal domains with
analytic star-shaped inclusions. It writes native Exodus II and legacy VTK,
validates topology and element quality, and includes a local browser studio for
generation and inspection.

The implementation deliberately solves a restricted problem well. Automatic,
general-purpose all-hex meshing remains an open research problem; the design
uses cubed-sphere O-grid macroblocks aligned to a Cartesian background, as
recommended by the research reviews preserved in
[`docs/research/`](docs/research/).

## Quick start

```bash
git clone git@github.com:aashiquear/hexaform.git
cd hexaform
./scripts/setup.sh
source .venv/bin/activate

hexaform particles-in-cube \
  --length 1.0 --n-particles 8 --radius-mean 0.075 \
  --mesh-density 24 --seed 7 --write-vtk
```

Generated files go to `output/` by default:

- `output/mesh.e` — Exodus II mesh;
- `output/mesh.vtk` — optional VTK mesh;
- `output/mesh.report.json` — configuration, packing, quality, and validation.

For installation variants and troubleshooting, see
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Browser Mesh Studio

```bash
hexaform serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The GUI provides mesh
controls, interactive orbit/zoom visualization of exterior and material
interface faces, quality coloring, validation results, and Exodus/VTK/report
downloads. It compiles to the same `RunConfig` and executes the same pipeline as
the CLI.

## Python API

```python
from hexaform import particles_in_cube

result = particles_in_cube(
    length=1.0,
    n_particles=8,
    radius_mean=0.075,
    mesh_density=24,
    seed=7,
    element_type="HEX27",
    output="output/rve.e",
    verbose=False,
)
assert result.validation.passed
print(result.mesh.summary())
```

HEX8 places interface corner nodes on the analytic surface but retains planar
faces. HEX20/HEX27 add globally shared curved higher-order nodes and reduce
geometric error; they remain polynomial approximations rather than exact CAD
geometry.

## Capabilities

- sphere, ellipsoid, superquadric, and spherical-harmonic grain inclusions;
- anisotropic grid-aligned macroboxes and cubed-sphere O-grids;
- RSA and collective rearrangement packing;
- HEX8, HEX20, and HEX27 connectivity;
- periodic boundary node pairs for Cartesian RVEs;
- interface pillowing, constrained smoothing, and notch/phase-field insertion;
- native Exodus blocks, side sets, node sets, and element fields;
- JSON/YAML configurations, campaigns, VTK output, and a browser GUI;
- validation for conformity, hanging/non-manifold entities, Jacobians, edge
  ratios, geometry error, ligaments, and phase-field resolution.

## Important limitations

- Production-quality meshing is limited to analytic star-shaped inclusions that
  can use the cubed-sphere macroblock.
- The adaptive octree/non-star path is a research prototype. It does not yet
  have conforming transition templates or robust boundary fitting and can
  contain hanging nodes and inverted elements. Production pipeline requests for
  star-shaped octree runs explicitly fall back to Cartesian macroblocks.
- Packing guarantees disjoint grid-aligned macroboxes; touching or overlapping
  inclusions are not supported by the production path.
- Solver-side phase-field and mechanics benchmarks are outside this repository;
  `h/ell` is reported, not certified for a constitutive model.

See the evidence-backed [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)
and current [`docs/roadmap.md`](docs/roadmap.md).

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Tutorial](docs/TUTORIAL.md)
- [User guide](docs/USER_GUIDE.md)
- [Algorithm](docs/algorithm.md)
- [Python/API reference](docs/api/reference.md)
- [Browser GUI](docs/GUI.md)
- [Technical audit report](docs/TECHNICAL_REPORT.md)
- [Development roadmap](docs/roadmap.md)
- [Research provenance](docs/research/README.md)
- [Contributing](CONTRIBUTING.md) and [governance](GOVERNANCE.md)

## Development

```bash
make setup
make test
make examples
make report
```

The complete test suite covers the core mesh pipeline, higher-order elements,
I/O round trips, packing, GUI graph compilation/API delivery, and explicit
experimental-octree invalidity gates.

## License and citation

MIT licensed. See [`CITATION.cff`](CITATION.cff) for citation metadata.
