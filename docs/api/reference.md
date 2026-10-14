# API reference

Import surface, module by module. Every public function has a docstring; this is
the map, not a substitute for it.

---

## Top level

```python
from hexaform import particles_in_cube, run, RunConfig, Sphere, validate
```

### `particles_in_cube(...) -> RunResult`

The one-call entry point.

```python
particles_in_cube(
    length=1.0, n_particles=10, radius_mean=0.08, radius_std=0.0,
    mesh_density=32, seed=0, output="output/mesh.e", shape="sphere",
    element_type="HEX8", **kwargs,
)
```

`n_particles` accepts an int or a `(lo, hi)` tuple for a random count.
`element_type` accepts `HEX8`, `HEX20`, or `HEX27`.
`**kwargs` forwards to `InclusionConfig` fields plus `phase_field_length`,
`verbose`, `validate_mesh`.

### `run(config: RunConfig) -> RunResult`

`RunResult` carries `.mesh`, `.packing`, `.validation`, `.config`,
`.output_path`, `.report_path`, `.timings`, `.inclusions`.

---

## `hexaform.core`

### `HexMesh`

| Attribute | Type | Meaning |
| --- | --- | --- |
| `points` | `(n, 3) f8` | node coordinates |
| `cells` | `(m, 8/20/27) i8` | zero-based connectivity in Exodus order |
| `element_type` | `str` | homogeneous `HEX8`, `HEX20`, or `HEX27` |
| `block_id` | `(m,) i4` | element block per element |
| `block_names` | `{int: str}` | block labels |
| `node_sets` | `{str: i8[]}` | zero-based node indices |
| `side_sets` | `{str: (elem_idx, side_id)}` | side ids are 1..6 |
| `element_fields` | `{str: array}` | e.g. `particle_id` |
| `metadata` | `dict` | seed, timings, parameters |

Methods/properties: `n_nodes`, `n_elements`, `nodes_per_element`, `block_ids`, `bounding_box()`,
`add_side_set()`, `add_node_set()`, `summary()`.

`HEX8_SIDES`, `HEX20_SIDES`, and `HEX27_SIDES` give the Exodus local face-node ordering.

### `core.quality`

- `scaled_jacobian(points, cells) -> (m,)` — sampled isoparametric metric for
  HEX8/HEX20/HEX27. 1.0 is a perfect cube; ≤ 0 is inverted.
- `edge_ratio(points, cells) -> (m,)`
- `element_volumes(points, cells) -> (m,)`
- `assess(mesh, threshold=0.2) -> QualityReport`

### `core.topology`

- `weld(mesh, tol)` — merge coincident nodes, drop unused
- `boundary_faces(cells)`, `interface_faces(cells, block_id, a, b)`
- `faces_on_plane(mesh, axis, value)`, `nodes_on_plane(...)`
- `signed_volumes(points, cells)`, `fix_orientation(mesh)`
- `check_conformity(mesh) -> dict` — hanging nodes, non-manifold faces,
  degenerate and duplicate elements, unused nodes

### `core.validation`

- `validate(mesh, inclusions, cell_size, ...) -> ValidationReport`
- `ValidationThresholds(min_scaled_jacobian=0.2, max_edge_ratio=12.0,
  max_volume_error=0.05, min_ligament_elements=2, require_conforming=True)`
- `recommended_mesh_density(length, radius_mean, phase_field_length=None, ...)`

`ValidationReport` has `.passed`, `.failures`, `.quality`, `.topology`,
`.geometry`, `.ligaments`, `.phase_field`, `.as_dict()`.

---

## `hexaform.shapes`

Implement `StarShape` to add a body:

```python
class StarShape(Shape):
    @property
    def center(self) -> np.ndarray: ...
    def support_radius(self, directions) -> np.ndarray: ...   # the only required geometry
    def bounding_radius(self) -> float: ...
    def volume(self) -> float: ...
```

Provided for free: `surface_points()`, `project()`, `radial_scale()`, `sdf()`,
`surface_area()`.

Shipped: `Sphere(centre, radius)`, `Ellipsoid(centre, semi_axes, rotation=None)`,
`Superquadric(centre, semi_axes, exponent_xy, exponent_z, rotation=None)`,
`SphericalHarmonicGrain.random(centre, radius, rng, l_max, roughness)`.

---

## `hexaform.templates`

### `CubedSphere(n, spacing="equiangular", blend=1.0)`

`.directions (6n²+2, 3)`, `.cube_points`, `.index[face] -> (n+1, n+1)`,
`.quads (6n², 4)`, `.check()`. Faces: `"x+" "x-" "y+" "y-" "z+" "z-"`.

### `build_macroblock(shape, box_lo_idx, k, h, grid_origin, options) -> MacroBlock`

Returns `.points`, `.cells`, `.is_inclusion`, `.surface_nodes`,
`.box_face_nodes`, `.interface_cells`, `.interface_sides`, `.stats`.

`MacroBlockOptions`: `core_fraction` (0.55), `core_roundness` (0.35),
`n_inner` (2), `n_outer` (2), `inner_grading`, `outer_grading`,
`patch_spacing` ("blend"), `patch_blend` (0.6).

### `transfinite`

- `tfi_volume(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, wu=None, wv=None, ww=None)`
- `graded_weights(n, ratio=1.0, cluster="none")` — `"start"`, `"end"`, `"both"`

---

## `hexaform.background` / `hexaform.packing` / `hexaform.assembly`

```python
CartesianGrid.cube(length, divisions, origin=None)
grid.build(carve_boxes)          # -> points, cells, cell_index
grid.node_coordinates(); grid.node_id(idx); grid.point_to_index(x)

pack(grid, PackingSpec(...), rng, shape_factory=None) -> PackingResult
sample_radii(n, spec, h, rng); box_cells_for_radius(radius, h, box_ratio)

assemble(grid, inclusions, template_options=None, options=AssemblyOptions())
```

`PackingSpec` fields: `n_particles`, `radius_mean`, `radius_std`, `radius_min`,
`radius_max`, `box_ratio`, `gap_cells`, `wall_cells`, `min_radius_cells`,
`max_tries_per_particle`, `allow_shortfall`. See also :class:`CollectiveOptions`
for collective-rearrangement packing (denser populations, 35–45%+ volume fraction).

`:class:`CollectiveOptions` fields: `max_steps=2000`, `seed_tries=400`,
`moves_fraction=0.6`, `kinetic_fraction=0.5`, `jump=3`, `seat_tries=200`,
`stall_limit=200`. The collective packer is a drop-in replacement for RSA with
``--packer=collective`` or via ``hexaform.packing.collective.pack()``.

`AssemblyOptions`: `separate_inclusion_blocks`, `interface_side_set`,
`per_inclusion_interface_sets`, `boundary_node_sets`, block names.

Constants: `MATRIX_BLOCK_ID = 1`, `INCLUSION_BLOCK_BASE = 100`.

---

## `hexaform.plate`

### `plate_mesh(opts: PlateOptions) -> HexMesh`

Rectangular plate with one HEX8 through the thickness and an optional
conforming through-hole (`HoleSpec(kind, a, b, centre, angle)`; O-grid ring
with graded radial layers). `PlateOptions(width, height, thickness, h, hole,
transition, ring_cells, n_r, grading, origin)`. `plate_summary(mesh)` returns
the counts, hole-edge element sizes and quality numbers. See the user guide.

## `hexaform.io`

```python
exodus.write(mesh, path, title=..., element_variables=None, backend="auto")
exodus.read(path) -> HexMesh
exodus.round_trip_check(mesh, path) -> dict

report.build_report(config, mesh, packing, validation, timings) -> dict
report.write_report(path, **kwargs)
report.write_campaign_csv(path, reports); report.campaign_summary(reports)

vtk.write(mesh, path)
```

`backend` is `"auto"`, `"netcdf4"` or `"scipy"`.

---

## `hexaform.server`

```python
from hexaform.server import catalogue, validate_graph, compile_graph
compile_graph(graph) -> RunConfig     # raises ValueError with readable problems
```

See [`../GUI.md`](../GUI.md).

---

## Experimental adaptive octree primitives

> **Research-only:** these APIs build adaptive leaves, but do not yet insert
> 2:1 conforming transition templates. Surface projection can invert boundary
> cells. Validate every result; the production pipeline does not claim this as
> a solver-ready backend.

```python
from hexaform.background import (
    OctreeGrid, OctreeMesh, sphere_size_field, sphere_material_of,
    encode_location, decode_location,
)
```

### `OctreeGrid(origin, length, h_min, max_level, material_of=..., size_field=...)`

The adaptive analogue of `CartesianGrid`.  Refines only where the size
field at a cell centre demands a smaller cell, where the cell straddles
a material interface (mixed corner materials), or where the cell
centre is in a different material than its (uniform-material) corners.

* `material_of(x)` -- `(N, 3) -> (N,)` array of integer material ids.
  0 = matrix, > 0 = inclusion index.
* `size_field(x)` -- `(N, 3) -> (N,)` array of target cell sizes.
  Cells split until `h <= target`.
* `build(carve_boxes=())` -- returns `(points, cells, cell_index)` with
  the same shape as `CartesianGrid.build()`.

### Helpers

* `sphere_size_field(centres, radii, bulk_h, interface_h, band)` --
  the standard particulate-composite size field: bulk cells of size
  `bulk_h`, interface cells of size `interface_h` within `band` of any
  particle surface.
* `sphere_material_of(centres, radii)` -- the matching material
  classifier: 0 outside any particle, `i + 1` inside particle `i`.

### `OctreeAssemblyOptions` and `assemble_with_octree`

```python
from hexaform.assembly.octree_assembly import (
    OctreeAssemblyOptions, assemble_with_octree, project_surface_nodes,
)
```

* `assemble_with_octree(shapes, domain_length, domain_origin, options)`
  -- build an experimental mesh from octree leaves. One block per material;
  matrix is block 1, inclusions are blocks 100, 101, ...  Snaps every
  interface node onto the analytic body if `project_surface_nodes` is
  `True`.
* `project_surface_nodes(mesh, shapes, tol, projectors=None)` --
  standalone surface projection, useful for Sculpt-style workflows
  where the mesher and the projection are separate steps.

## Constrained mesh optimisation

```python
from hexaform.core.smoothing import (
    SmoothingOptions, SmoothingResult, smooth_surface, detect_surface_nodes,
)
```

* `smooth_surface(mesh, surface_nodes, options, surface_projector, fixed_mask=None)`
  -- Knupp-style Jacobian-barrier optimisation with surface projection
  after every step.
* `detect_surface_nodes(mesh, shapes_or_inclusions, tol=1e-6)` -- find
  every node lying on the analytic surface of any shape or inclusion.

## Interface pillowing

```python
from hexaform.assembly.pillow import (
    PillowOptions, pillow_interface, pillow_layer, detect_interface_elements,
)
```

* `pillow_layer(mesh, options, matrix_block_id, inclusion_block_id, shapes)`
  -- insert one hex layer between the matrix and the inclusion block.
  ``shapes`` is the list of analytic bodies; when supplied, the per-corner
  offset uses the SDF gradient (essential for curved interfaces, where
  the average face normal gives a degenerate hex).
* `pillow_interface(mesh, block_a, block_b, options, shapes)` -- the
  general entry point.
* `detect_interface_elements(mesh, block_a, block_b)` -- indices of
  elements that touch a material interface.

## Non-star-shaped geometry utilities

These utilities define and project the geometry. They do not imply that the
experimental octree assembler produces a conforming or valid torus mesh.

```python
from hexaform.shapes import Torus, project_to_torus
```

* `Torus(centre, major_radius, minor_radius, axis=None)` -- the
  prototypical non-star body.  Implements `sdf`, `bounding_radius`,
  `volume`, `surface_area`, and `project` (closed-form projection).
* `project_to_torus(x, centre, R, r, rotation)` -- standalone
  projection.  ``rotation`` is the body-to-world matrix whose columns
  are ``(e1, e2, axis)``.
