# Hexaform 0.2 technical implementation audit

Generated report metadata: version `{{VERSION}}`, commit `{{COMMIT}}`, generated
at `{{GENERATED_AT}}`.

## Executive assessment

The Cartesian cubed-sphere macroblock path is a coherent, well-tested research
software implementation for analytic star-shaped inclusions. It is conforming
by construction, supports higher-order geometry, writes a meaningful Exodus
data model, and applies validation as a first-class gate.

The repository was not fully complete against its own roadmap when this audit
started. The browser frontend was absent, VTK configuration was ignored, output
artifacts were scattered, install/governance/report workflows were incomplete,
and several documents overstated the adaptive octree and non-star paths. Those
repository/product gaps are addressed in 0.2. The underlying general adaptive
octree transition/boundary-fitting algorithm remains incomplete and is now
identified as an experimental research path.

Release assessment: **production candidate for the restricted Cartesian,
star-shaped problem; research-only for adaptive octree/non-star meshing.**

## Review basis

The audit compared source and tests with:

- `docs/research/sources/scholargpt-review.md`;
- `docs/research/sources/scispace-review.md`;
- `docs/research/synthesis.md`;
- the pre-audit `docs/roadmap.md` claims;
- all Python modules, examples, and tests.

The preserved reviews are AI-assisted literature maps, not primary authority.
The synthesis and bibliography distinguish verified repository behavior from
solver- or literature-side claims.

## Research-to-code completeness matrix

| Research/roadmap requirement | Evidence | Assessment |
| --- | --- | --- |
| Restrict the problem rather than promise a general all-hex mesher | `shapes/`, macroblock API, stated limits | Complete and appropriate. |
| Six-patch cubed sphere instead of polar parameterization | `templates/cubed_sphere.py`, welding/orientation tests | Complete. |
| Radial O-grid around each analytic inclusion | `templates/macroblock.py`, macroblock/assembly tests | Complete for star-shaped bodies. |
| Exact analytic interface node placement with explicit HEX8 limitation | shape projection, geometry tests, higher-order promotion | Complete; polynomial faces remain approximate as expected. |
| Separate packing from meshing | `packing/` emits `PackedInclusion`; `assembly/` consumes it | Complete. |
| Ligament- and wall-aware admissibility | grid-aligned macrobox constraints and validation metrics | Complete within the conservative macrobox model. |
| Quality/validity before appearance or count | `core/validation.py`, quality/topology tests | Complete for implemented paths. |
| Native Exodus blocks/sets/fields and round trip | `io/exodus.py`, Exodus/high-order tests | Complete at repository level; target-solver certification remains external. |
| HEX20/HEX27 curved geometry | `core/elevate.py`, I/O and geometry regression tests | Complete. |
| Periodic RVE pairing | Cartesian grid/assembly tests | Complete for Cartesian; unavailable for octree. |
| Constrained optimization and interface pillow | smoothing/pillow modules and tests | Implemented optional post-processes. |
| Adaptive octree with conforming transitions and Jacobian control | adaptive leaf prototype, explicit invalidity test | Incomplete. No transition templates; projection can invert cells. |
| Non-star-shaped production fallback | torus SDF/projector and experimental assembly | Incomplete. Geometry utilities exist; solver-ready meshing does not. |
| Monte Carlo success-rate workflow | campaign CLI/example/report CSV | Infrastructure complete; a published large campaign is still needed. |
| Mechanics and phase-field benchmarks | density metrics only | Not implemented; outside pure meshing validation and required before solver claims. |
| Browser generation/visualization | packaged FastAPI/HTML/CSS/JS, API and visual checks | Complete for synchronous local use. |

## Important findings and resolutions

### Critical: octree claims exceeded implementation

For star shapes, `backend=octree` called the Cartesian assembler while logging
adaptive refinement. For non-star shapes, adaptive leaves had hanging nodes and
analytic projection could create inverted cells. A representative torus audit
found 152 hanging nodes and 296 inverted elements at a modest resolution.

Resolution:

- star-shape requests now record the safe Cartesian fallback and that adaptive
  refinement was not applied;
- experimental meshes carry explicit metadata;
- the non-star GUI block is planned/blocked;
- tests require the invalid torus prototype to be rejected;
- roadmap and user documentation no longer call this production-complete.

### High: browser frontend absent

Only the graph/backend contract existed. The new Mesh Studio provides controls,
quality-colored interactive preview, validation, and four artifacts. Browser
runs use the same pipeline and remain under `output/gui`.

### High: graph stages and VTK flags were not effective

Pillow/smoothing graph blocks did not enable their stages, octree parameters had
no config paths, the non-star block silently compiled to the default sphere,
and `OutputConfig.write_vtk` was unused.

Resolution: block defaults/paths are compiled and validated, optional stages are
enabled explicitly, planned non-star graphs are rejected, and VTK is written by
the pipeline with a returned artifact path.

### Medium: artifact and documentation sprawl

Large Exodus/report files and ad-hoc validation scripts lived at the root;
multiple notch/status summaries repeated conflicting claims.

Resolution: generated artifacts default to `output/`; ad-hoc outputs were moved
there; detailed notch documentation has one home; README, installation,
tutorial, user guide, GUI guide, roadmap, and this report have distinct roles.

### Medium: server artifact handling

The preview backend used temporary directories and accepted an unchecked token
as a path component.

Resolution: cryptographic run identifiers, a strict token pattern, an artifact
allow-list, fixed paths, and project-local output persistence.

## Validation evidence

The pre-change baseline passed 239 tests. The release suite adds graph schema,
VTK pipeline, browser API/download, preview, fallback-disclosure, and octree
invalidity regressions. The final Python 3.14.4 run passed **257 tests** in
169.11 seconds with **89% statement coverage** (3,864 statements, 442 missed).
All 11 examples then completed through `scripts/run_examples.py`; valid production
examples passed their gates and the torus research prototype was rejected as
expected. The sweep also exposed and fixed an example-only NumPy shadowing bug
that the unit suite did not exercise.

Browser interaction verification generated a 2,328-element/2,805-node HEX8
mesh, displayed exterior and inclusion interfaces, reported minimum scaled
Jacobian 0.5774 and `VALID`, exposed all four downloads, and produced no browser
console warnings/errors.

## Residual risks

1. Solver compatibility is not established by an Exodus round trip alone.
2. The netCDF4 wheel may emit a NumPy binary-compatibility runtime warning on
   some very new Python/NumPy combinations; the SciPy backend is available.
3. GUI runs are synchronous and have no retention/quotas; keep loopback binding.
4. Quality thresholds are engineering defaults, not constitutive-model proofs.
5. Large-scale stochastic success and mechanics convergence studies remain to
   be published.

## Recommended next implementation sequence

1. Freeze 0.2 restricted-path behavior with CI and tagged release artifacts.
2. Add solver-side single-inclusion and phase-field benchmark repositories.
3. Implement 2:1 octree balancing and conforming transitions before further
   non-star surface work.
4. Use actual SDF classification/reconstruction and constrained Jacobian
   optimization; require the same production gates as Cartesian.
5. Add process-isolated GUI jobs and retention controls.

This ordering follows the source reviews' validity-first principle and avoids
presenting an unsolved general meshing problem as a finished feature.
