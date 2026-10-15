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

## Plates with a through-hole

`hexaform plate` meshes a rectangular plate `width x height x thickness` with
a uniform in-plane background of cell size `h` and ONE hexahedral element
through the thickness (pseudo-3-D). An optional circular or elliptical
through-hole is meshed with a block-structured O-grid ring whose nodes lie
exactly on the analytic curve, so the hole edge is conforming rather than
staircased; the ring hands over to the background on a grid-aligned
transition rectangle without transition elements.

```bash
hexaform plate --width 100 --height 50 --thickness 1 --h 2.5 \
  --hole circle:5 -o output/plate_hole5.e --write-vtk --summary output/plate_hole5.json
hexaform plate --h 2.5 --hole ellipse:7.5,2.5,30   # full axes 7.5 and 2.5, rotated 30 deg
```

Options: `--centre CX CY` (default plate centre), `--transition KX KY`
(half-extents of the O-grid rectangle in background cells; default leaves
`--ring-cells` cells between the hole's bounding box and the rectangle),
`--n-r` radial layers and `--grading` (outer / inner radial spacing). When
both are omitted the layers are graded so that the innermost spacing equals
the hole-edge arc length (near-square elements where damage initiates) and
the outermost equals `h`. `width/h` and `height/h` must be whole numbers and
`CX - KX*h`, `CY - KY*h` must fall on the background grid.

Sets: node and side sets `x-`, `x+`, `y-`, `y+`, `z-`, `z+`, `hole`, the
node set `outer` (the four in-plane edges, hole excluded) and the node set
`grips` (the two x faces, a damage Dirichlet set that leaves the free edges
and the hole unconstrained). Blocks `1 = plate`, `2 = ring`. Python: `plate_mesh(PlateOptions(
..., hole=HoleSpec("circle", 2.5)))` and `plate_summary(mesh)`.

The plain plate (no `--hole`) is the same background grid with the same
sets, so a perforated and an unperforated plate of equal `h` share every
background element.

## Dog-bone tensile bars

`hexaform dogbone` meshes a flat tensile coupon: grips of width
`--grip-width`, a gage `--gage-width x --gage-length` in the middle, circular
shoulder fillets `--fillet-radius` tangent to the gage edge, total
`--length`, and a thickness profile along the gage given as measured
stations `--stations X:T,...` (X from the gage start; natural cubic spline
inside the gage, linear blend to `--grip-thickness` across each fillet,
constant grips). One mapped hexahedral block: cell size `--h` in the gage,
growing geometrically to `--h-grip` across the fillets, `--n-z` elements
through the thickness.

```bash
hexaform dogbone --length 25.4 --gage-length 8 --gage-width 1.85 --grip-width 4.8 \
  --fillet-radius 3.175 --grip-thickness 0.35 --stations 0:0.3399,2:0.3441,4:0.3540,6:0.3625,8:0.3715 \
  --h 0.2 -o output/specimen_L1.e --write-vtk --summary output/specimen_L1.json
```

Sets: node and side sets `x-`, `x+` (the grip ends); node sets `grips`,
`gage-`/`gage+` (the first and last station planes, for a gauge-length
strain), `station_k` (one per station), `edge-`/`edge+` (the free in-plane
edges) and `z-`/`z+` (the two faces). Blocks `1 = gage`, `2 = shoulder`,
`3 = grip`. Python: `dogbone_mesh(DogboneOptions(...))`, `dogbone_summary(mesh)`,
`half_width(opts, x)` and `thickness(opts, x)` for the analytic profiles.

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
