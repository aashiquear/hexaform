# Installation

## Requirements

- Python 3.10 or newer;
- a POSIX shell for `scripts/setup.sh` (manual commands work on other systems);
- Git for source installation and versioned technical reports.

Core meshing requires NumPy and SciPy. Exodus files can be written through the
SciPy netCDF fallback; `netCDF4` is recommended for larger files.

## One-command development setup

```bash
git clone git@github.com:aashiquear/hexaform.git
cd hexaform
./scripts/setup.sh
source .venv/bin/activate
```

The script creates `.venv`, upgrades packaging tools, installs the project with
all optional dependencies, and runs an import/CLI check. It is safe to rerun.

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"
```

Available extras:

| Extra | Packages | Purpose |
| --- | --- | --- |
| `exodus` | `netCDF4` | Faster/larger Exodus II I/O. |
| `yaml` | `PyYAML` | YAML run configurations. |
| `server` | `fastapi`, `uvicorn` | Local browser Mesh Studio. |
| `dev` | `pytest`, `pytest-cov`, `httpx` | Tests, coverage, and API checks. |
| `all` | all of the above | Development and full feature validation. |

For a minimal core install:

```bash
python -m pip install -e .
```

## Verify

```bash
hexaform --version
python -m pip check
python -m pytest
```

Start the browser GUI:

```bash
hexaform serve
```

Then open `http://127.0.0.1:8000`.

## Uninstall

```bash
python -m pip uninstall hexaform
```

Remove `.venv` only if you also want to discard the project-local environment.
Generated meshes live in `output/` and are not removed by uninstalling.

## Troubleshooting

`hexaform: command not found`
: Activate `.venv` or run `.venv/bin/hexaform`.

`ImportError` mentioning FastAPI or uvicorn
: Install `.[server]` or `.[all]` in the active environment.

No `netCDF4`
: The SciPy backend remains available. Use `exodus_backend="scipy"` for a
  deterministic fallback.

Large mesh exhausts memory
: Lower `--mesh-density` or particle count. The 0.2 GUI is synchronous and
  builds the browser preview after the full mesh.
