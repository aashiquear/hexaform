# Tutorial: first validated mesh

Complete [installation](INSTALLATION.md) first.

## 1. Generate from the CLI

```bash
hexaform particles-in-cube \
  --length 1.0 \
  --n-particles 3 \
  --radius-mean 0.10 \
  --radius-std 0.01 \
  --mesh-density 12 \
  --seed 7 \
  --write-vtk
```

Hexaform packs inclusions into disjoint grid-aligned macroboxes, creates one
cubed-sphere O-grid per inclusion, assembles the background, validates the
result, and writes:

```text
output/mesh.e
output/mesh.vtk
output/mesh.report.json
```

A successful run ends with `Hexaform validation: VALID`. The report preserves
the seed and complete configuration.

## 2. Inspect the output

```bash
hexaform inspect output/mesh.e
```

This reads Exodus back and reruns the file-recoverable topology and quality
checks. Analytic surface/ligament checks require the original shape data and
therefore remain in the sidecar report.

## 3. Generate in the browser

```bash
hexaform serve
```

Open `http://127.0.0.1:8000`, keep the quick preset, and choose **Generate
mesh**. Drag the canvas to orbit and scroll to zoom. The preview shows exterior
grid faces and material interfaces; interface color follows scaled Jacobian.

Download Exodus, VTK, the report, or the compiled config. Browser runs are
stored under `output/gui/<run-id>/`.

## 4. Run the same job from Python

```python
from hexaform import particles_in_cube

result = particles_in_cube(
    length=1.0,
    n_particles=3,
    radius_mean=0.10,
    radius_std=0.01,
    mesh_density=12,
    seed=7,
    output="output/tutorial/mesh.e",
    verbose=False,
)

if not result.validation.passed:
    raise RuntimeError(result.validation.failures)

print(result.mesh.n_elements, result.mesh.element_type)
```

## 5. Try higher-order geometry

```bash
hexaform particles-in-cube \
  --n-particles 1 --radius-mean 0.12 --mesh-density 12 \
  --element-type HEX27 -o output/tutorial/hex27.e
```

The element count is unchanged, but edge/face nodes are shared globally and
interface higher-order nodes are projected to the analytic shape. Confirm your
target solver uses curved isoparametric geometry before relying on it.

## 6. Preserve a reusable configuration

```python
result.config.to_json("output/tutorial/run.json")
```

Rerun it with:

```bash
hexaform run output/tutorial/run.json -o output/tutorial/repeated.e
```

Continue with the [user guide](USER_GUIDE.md) for shapes, campaigns, periodic
boundaries, notches, and configuration details.
