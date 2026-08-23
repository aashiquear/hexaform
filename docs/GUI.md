# Browser Mesh Studio

## Start

```bash
hexaform serve --host 127.0.0.1 --port 8000 --output-dir output/gui
```

Open `http://127.0.0.1:8000`. The server binds to loopback by default and has no
authentication; do not expose it to an untrusted network.

## Workflow

1. Choose domain size, density, shape population, packing, element type, and
   optional post-processing.
2. Select **Generate mesh**.
3. Orbit/zoom the canvas and optionally hide the domain boundary.
4. Inspect element/node counts, minimum scaled Jacobian, and validation status.
5. Download Exodus, VTK, JSON report, or the compiled configuration.

The renderer extracts exterior and material-interface quads. Material faces are
colored by the owning element's scaled Jacobian. Large previews are sampled
deterministically with material interfaces prioritized.

## Architecture

The browser creates a typed block graph. The server validates it, compiles it to
`RunConfig`, and calls `pipeline.run.run`; there is no second meshing
implementation. Generated artifacts are stored under a random 32-character run
identifier in the configured output directory.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Version/liveness. |
| `GET /api/blocks` | Block catalogue and parameter schemas. |
| `GET /api/preset` | Small default graph. |
| `POST /api/graph/validate` | Validate structure, types, bounds, and statuses. |
| `POST /api/graph/compile` | Return inspectable `RunConfig` JSON. |
| `POST /api/graph/run` | Generate, validate, preview, and persist artifacts. |
| `GET /api/runs/{token}/{artifact}` | Download an allowed artifact. |

Interactive OpenAPI documentation is at `/docs`.

## Operational limits

- Runs are synchronous in 0.2; use moderate densities.
- Output retention is manual. Remove old `output/gui/<run-id>` directories when
  no longer needed.
- The server is local-trust software, not a multi-user service.
- Planned/invalid algorithm blocks are rejected by graph validation.

The next server milestone is process-isolated jobs with progress, cancellation,
resource limits, and retention policies.
