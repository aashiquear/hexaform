# Notch and pre-crack insertion

This is the single detailed reference for the notch feature. Overall release
status and evidence live in the roadmap and technical report.

Phase-field fracture simulations require an initial crack or notch to seed crack initiation.
Hexaform provides two complementary approaches for notch insertion, both fully integrated
into the mesh generation pipeline.

## Overview

When simulating fracture using phase-field methods, an initial flaw is needed to nucleate
cracks. Hexaform supports:

1. **Field method** (default): Initialize a phase-field damage variable to 1.0 in elements
   along the notch path. The mesh topology remains intact.

2. **Geometric method**: Mark elements as void (`particle_id = 0`) in the notch region,
   effectively removing material.

Both methods produce numerically equivalent results for most solvers, but the field method
preserves the mesh topology and is generally preferred.

## Supported Notch Geometries

### Edge Notch
A straight crack starting from a domain face, commonly used in three-point bend tests.

```python
from hexaform.assembly.notch import create_edge_notch

notch = create_edge_notch(
    length=0.3,           # Notch extends 0.3 units
    origin=(0.0, 0.5, 0.5),  # Start from mid-height of left face
    direction=(1.0, 0.0, 0.0),  # Propagate in +x direction
    depth=1.0,            # Through-thickness
)
```

### Center Notch
A crack centered in the domain interior, used in tension tests.

```python
from hexaform.assembly.notch import create_center_notch

notch = create_center_notch(
    length=0.4,
    origin=(0.5, 0.5, 0.5),  # Domain center
    direction=(1.0, 0.0, 0.0),
    depth=1.0,
)
```

### Chevron Notch
A V-shaped notch for mode-I fracture tests, promoting crack initiation at the vertex.

```python
from hexaform.assembly.notch import create_chevron_notch

notch = create_chevron_notch(
    length=0.4,
    origin=(0.0, 0.5, 0.5),
    direction=(1.0, 0.0, 0.0),  # Bisector of V
    depth=1.0,
    angle=45.0,  # Half-angle of V (degrees)
)
```

## Usage

### Command Line

```bash
# Field method (default)
hexaform particles-in-cube \
    --length 1.0 --n-particles 10 \
    --radius-mean 0.06 --mesh-density 40 --seed 42 \
    --notch-type edge --notch-length 0.3 \
    --method field \
    --phase-field-variable phase_field \
    -o notch_mesh.e

# Geometric method
hexaform particles-in-cube \
    --length 1.0 --n-particles 10 \
    --radius-mean 0.06 --mesh-density 40 --seed 42 \
    --notch-type edge --notch-length 0.3 \
    --method geometric \
    -o notch_geometric.e

# Chevron notch
hexaform particles-in-cube \
    --length 1.0 --n-particles 10 \
    --radius-mean 0.06 --mesh-density 40 --seed 42 \
    --notch-type chevron --notch-length 0.4 \
    --chevron-angle 45 \
    -o chevron_notch.e
```

### Python API

```python
from hexaform import RunConfig, run
from hexaform.pipeline.config import DomainConfig, InclusionConfig, NotchConfig, OutputConfig

config = RunConfig(
    domain=DomainConfig(length=1.0, mesh_density=40),
    inclusions=InclusionConfig(
        shape="sphere",
        n_particles=10,
        radius_mean=0.06,
    ),
    output=OutputConfig(path="notch_mesh.e"),
    seed=42,
    notch=NotchConfig(
        enabled=True,
        notch_type="edge",  # "edge", "center", or "chevron"
        origin=(0.0, 0.5, 0.5),
        direction=(1.0, 0.0, 0.0),
        length=0.3,
        depth=1.0,
        chevron_angle=45.0,  # Only used for chevron notch
        method="field",  # or "geometric"
        phase_field_value=1.0,
        phase_field_variable="phase_field",
    ),
)

result = run(config)

# Access the phase-field variable
if "phase_field" in result.mesh.element_fields:
    phase_field = result.mesh.element_fields["phase_field"]
    n_damaged = (phase_field > 0.5).sum()
    print(f"Notch contains {n_damaged} damaged elements")

# Check metadata
print(f"Notch type: {result.mesh.metadata['notch_type']}")
print(f"Notch method: {result.mesh.metadata['notch_method']}")
print(f"Notch elements: {result.mesh.metadata['n_notch_elements']}")
```

### Direct Insertion

You can also insert a notch into an existing mesh:

```python
from hexaform.assembly.notch import (
    create_edge_notch,
    NotchOptions,
    insert_notch,
)

# Create a mesh (with or without particles)
config = RunConfig(
    domain=DomainConfig(length=1.0, mesh_density=20),
    inclusions=InclusionConfig(n_particles=5, radius_mean=0.05),
    output=OutputConfig(path=""),
    validate_mesh=False,
)
result = run(config)
mesh = result.mesh

# Insert notch
notch_geom = create_edge_notch(length=0.3, origin=(0.0, 0.5, 0.5))
notch_opts = NotchOptions(
    method="field",
    phase_field_variable="phase_field",
    mark_elements=True,
)

mesh_with_notch = insert_notch(
    mesh,
    notch_geom,
    notch_opts,
    domain_extent=(1.0, 1.0, 1.0),
    domain_origin=(0.0, 0.0, 0.0),
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `False` | Enable notch insertion |
| `notch_type` | str | `"edge"` | Type: `"edge"`, `"center"`, or `"chevron"` |
| `origin` | tuple | `(0.0, 0.5, 0.5)` | Starting point (x, y, z) |
| `direction` | tuple | `(1.0, 0.0, 0.0)` | Propagation direction (dx, dy, dz) |
| `length` | float | `0.5` | Length along propagation direction |
| `depth` | float | `1.0` | Through-thickness depth |
| `chevron_angle` | float | `45.0` | Half-angle for chevron notch (degrees) |
| `method` | str | `"field"` | Insertion method: `"field"` or `"geometric"` |
| `phase_field_value` | float | `1.0` | Damage value to assign |
| `phase_field_variable` | str | `"phase_field"` | Element variable name |

## Output

### Element Variables

When using the field method, the mesh includes an element variable (default name: `phase_field`):
- Value `1.0`: Elements in the notch region (fully damaged)
- Value `0.0`: Elements outside the notch (undamaged)

This variable is written to the Exodus file at time step 1 and can be used to initialize
the phase-field in your solver.

### Element Fields

For the geometric method, elements in the notch region have `particle_id = 0`, indicating
void/inactive material.

### Metadata

The mesh metadata includes:
- `notch_type`: Type of notch inserted
- `notch_origin`: Starting coordinates
- `notch_direction`: Propagation direction
- `notch_length`: Notch length
- `notch_method`: Insertion method used
- `n_notch_elements`: Number of elements in the notch

### Element Indices

If `mark_elements=True`, the notch element indices are stored in
`mesh.element_fields["notch_element_ids"]` for easy access.

## Validation

The notch implementation includes comprehensive tests (`tests/test_notch.py`):
- Geometry creation and validation
- Field method insertion
- Geometric method insertion
- All notch types (edge, center, chevron)
- Pipeline integration
- Resolution scaling
- Length accuracy

Run tests with:
```bash
pytest tests/test_notch.py -v
```

## Example

See `examples/07_notch_insertion.py` for a complete working example demonstrating:
- All three notch types
- Both insertion methods
- Integration with particle inclusions
- Phase-field length reporting
- Output file generation

Run the example:
```bash
python examples/07_notch_insertion.py --notch-type edge --notch-length 0.3
```

## Implementation Details

The notch is inserted by:
1. Computing element centroids
2. Transforming centroids to the notch coordinate system
3. Identifying elements whose centroids lie within the notch region
4. Assigning phase-field values or marking as void

The field method is preferred because:
- Preserves mesh topology
- No element removal (better for solver stability)
- Direct initialization of phase-field variable
- Easier to modify notch geometry post-generation

## Solver Integration

Most phase-field fracture solvers can use the notch output directly:

**Abaqus**:
```
*Initial Conditions, type=Solution
ALL_ELEMENTS, phase_field
1.0, 0.0  # Read from element variable
```

**deal.II / FEniCS**:
Read the `phase_field` element variable from the Exodus file and use it as the
initial condition for the phase-field variable.

**Custom solvers**:
The `phase_field` element variable at time step 1 provides the initial damage field.
Elements with value 1.0 are fully damaged (crack), elements with 0.0 are intact.

## Limitations

- Notch width is effectively zero-thickness for the field method (single element layer)
- Geometric method removes elements, which may affect solver convergence
- Chevron notch approximation depends on mesh resolution
- Notch must be aligned with element boundaries for accurate representation

## Future Work

Planned enhancements:
- Curved notch geometries
- Multiple notches in a single mesh
- Notch width control for field method
- Automatic notch placement based on particle positions
- Integration with adaptive mesh refinement
