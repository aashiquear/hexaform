# User guide

## Output organization

All built-in defaults write beneath `output/`:

```text
output/
  mesh.e
  mesh.vtk                 # when requested
  mesh.report.json
  campaign/
  examples/
  gui/<run-id>/
  reports/
```

Custom absolute or relative paths remain supported. Writers create parent
directories automatically.

## CLI commands

`particles-in-cube`
: Generate one random realization.

`run CONFIG`
: Execute a JSON or YAML `RunConfig`.

`campaign`
: Sweep seeds, retain per-run artifacts, and write `campaign.csv`.

`inspect FILE`
: Read Exodus and rerun topology/quality checks.

`suggest-density`
: Estimate a background density from inclusion radius and optional phase-field
  length. This is a heuristic, not a convergence certificate.

`serve`
: Start the local browser GUI.

Use `hexaform COMMAND --help` for the complete option list.

## Shapes

The production template supports `sphere`, `ellipsoid`, `superquadric`, and
`sh_grain`. Each implements an analytic support radius from its center, which
is the geometric contract required by a cubed-sphere O-grid.

```bash
hexaform particles-in-cube --shape ellipsoid \
  --aspect-ratios 1.0 0.7 0.55 --n-particles 6
```

Torus SDF/projection utilities exist for research. The current adaptive octree
prototype is not conforming and is not solver-ready; see the roadmap.

## Packing

RSA is fast and provides reproducible, disjoint grid-aligned macroboxes. It
typically saturates before dense physical sphere packing because macroboxes,
not just analytic particles, must remain disjoint.

```bash
hexaform particles-in-cube --packer collective \
  --n-particles 20 --radius-mean 0.05 --gap-cells 1
```

Collective packing moves candidates to open cavities but preserves the same
meshability invariant. `gap_cells` sets the guaranteed macrobox separation;
`wall_cells` sets the domain margin.

## Element type

| Type | Geometry | Use |
| --- | --- | --- |
| HEX8 | linear, planar faces | default and widest solver compatibility |
| HEX20 | quadratic edges | improved curved interfaces |
| HEX27 | triquadratic edges/faces | best shipped geometric accuracy per element |

All elements in a mesh use one homogeneous type. Blocks, side sets, element
fields, and element counts are preserved during higher-order promotion.

## Periodic RVEs

```bash
hexaform particles-in-cube --periodic -o output/periodic/rve.e
```

Cartesian periodic runs create paired negative/positive face node sets and
`x_periodic_pair`, `y_periodic_pair`, and `z_periodic_pair` arrays. Octree
periodicity is not implemented and is rejected by configuration validation.

## VTK and Exodus

Add `--write-vtk` to CLI generation or set:

```python
OutputConfig(
    path="output/case/mesh.e",
    write_vtk=True,
    vtk_path="output/case/mesh.vtk",  # optional override
)
```

Exodus is the production data model. VTK is an ASCII unstructured-grid export
for visualization and debugging.

## Notches and phase-field data

Notches can initialize a damage field or mark geometric void elements. Edge,
center, and chevron geometries are supported. See
[`features/notches.md`](features/notches.md) for complete options and examples.

`phase_field_length` adds resolution metrics to validation but never silently
changes the mesh.

## Pillowing and smoothing

`--pillow` inserts an interface sheet. `--smoothing` runs the constrained
Jacobian-barrier optimizer with detected analytic surface nodes locked. Both are
optional post-assembly stages and must still pass the full validation gate.

## Campaigns

```bash
hexaform campaign --seeds 0-19 --n-particles 8-12 \
  --mesh-density 20 -o output/campaign
```

Campaign success rate is more meaningful than one favorable realization.
Inspect `campaign.csv` for packing, quality, geometry, and timing metrics.

## Configuration files

Every field is represented in `RunConfig`. Serialize from Python with
`config.to_json(path)` or load JSON/YAML with `RunConfig.load(path)`. Unknown
dataclass fields are ignored for forward compatibility; pipeline validation
rejects invalid physical/domain values.

## Validation semantics

The production acceptance order is:

1. positive Jacobians and valid topology;
2. scaled Jacobian and edge ratio;
3. conformity (no hanging/non-manifold entities);
4. analytic geometry error;
5. ligament resolution;
6. phase-field resolution reporting.

A written file is not automatically a valid solver input. Always require
`result.validation.passed` or a zero CLI exit status.
