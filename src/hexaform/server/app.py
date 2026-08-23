"""Local FastAPI application and browser GUI for Hexaform.

The browser compiles the same block graph into the same :class:`RunConfig` used
by the CLI. Generated Exodus, VTK, report, and configuration artifacts are kept
under ``output/gui`` by default instead of leaking into temporary directories.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import numpy as np

from ..core.elements import side_nodes
from ..core.mesh import HexMesh
from ..core.quality import scaled_jacobian
from .blocks import catalogue, compile_graph, validate_graph

_TOKEN_RE = re.compile(r"[0-9a-f]{32}")
_ARTIFACTS = {
    "exodus": "mesh.e",
    "vtk": "mesh.vtk",
    "report": "mesh.report.json",
    "config": "config.json",
}


def create_app(output_dir: str | Path = "output/gui"):
    """Build the FastAPI app while keeping web dependencies optional."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
    except ImportError as exc:  # pragma: no cover - exercised without server extra
        raise ImportError(
            "the browser GUI needs FastAPI and uvicorn: pip install 'hexaform[server]'"
        ) from exc

    from ..io import report as report_io
    from ..pipeline.run import run

    static_dir = Path(__file__).with_name("static")
    run_root = Path(output_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Hexaform",
        description="Browser-based hexahedral mesh generation and inspection",
        version=_version(),
    )
    @app.get("/", include_in_schema=False)
    async def index():
        return Response((static_dir / "index.html").read_bytes(), media_type="text/html")

    @app.get("/static/{filename}", include_in_schema=False)
    async def static_asset(filename: str):
        media_types = {"styles.css": "text/css", "app.js": "text/javascript"}
        if filename not in media_types:
            raise HTTPException(status_code=404, detail="asset not found")
        return Response(
            (static_dir / filename).read_bytes(),
            media_type=media_types[filename],
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "version": _version()}

    @app.get("/api/blocks")
    async def blocks() -> Dict[str, Any]:
        return {"blocks": catalogue()}

    @app.get("/api/preset")
    async def preset() -> Dict[str, Any]:
        return {"graph": default_graph()}

    @app.post("/api/graph/validate")
    async def validate_graph_endpoint(graph: Dict[str, Any]) -> Dict[str, Any]:
        problems = validate_graph(graph)
        if not problems:
            try:
                compile_graph(graph)
            except ValueError as exc:
                problems.append(str(exc))
        return {"valid": not problems, "problems": problems}

    @app.post("/api/graph/compile")
    async def compile_endpoint(graph: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {"config": compile_graph(graph).to_dict()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/graph/run")
    async def run_endpoint(graph: Dict[str, Any]) -> Dict[str, Any]:
        try:
            config = compile_graph(graph)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        token = uuid4().hex
        run_dir = run_root / token
        run_dir.mkdir(parents=False, exist_ok=False)
        config.output.path = str(run_dir / _ARTIFACTS["exodus"])
        config.output.report_path = str(run_dir / _ARTIFACTS["report"])
        config.output.write_report = True
        config.output.write_vtk = True
        config.output.vtk_path = str(run_dir / _ARTIFACTS["vtk"])
        config.verbose = False
        config.to_json(run_dir / _ARTIFACTS["config"])
        try:
            result = run(config)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        report = report_io.build_report(
            config=config,
            mesh=result.mesh,
            packing=result.packing,
            validation=result.validation,
            timings=result.timings,
        )
        downloads = {
            name: f"/api/runs/{token}/{name}"
            for name, filename in _ARTIFACTS.items()
            if (run_dir / filename).is_file()
        }
        return {
            "token": token,
            "report": report,
            "preview": mesh_preview(result.mesh),
            "downloads": downloads,
        }

    @app.get("/api/runs/{token}/{artifact}")
    async def download(token: str, artifact: str):
        if _TOKEN_RE.fullmatch(token) is None or artifact not in _ARTIFACTS:
            raise HTTPException(status_code=404, detail="artifact not found")
        path = run_root / token / _ARTIFACTS[artifact]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        async def chunks():
            with path.open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    yield data

        return StreamingResponse(
            chunks(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    return app


def default_graph() -> Dict[str, Any]:
    """Small, quick preset used by the browser and API examples."""
    return {
        "nodes": [
            {"id": "domain", "block": "cube_domain", "params": {
                "length": 1.0, "mesh_density": 12,
            }},
            {"id": "shapes", "block": "sphere_source", "params": {
                "n_particles": 3, "radius_mean": 0.10, "radius_std": 0.01,
            }},
            {"id": "packing", "block": "rsa_packing", "params": {
                "packer": "rsa", "box_ratio": 1.35, "gap_cells": 2,
                "wall_cells": 1, "seed": 7,
            }},
            {"id": "template", "block": "macroblock_template", "params": {
                "element_type": "HEX8", "n_inner": 2, "n_outer": 2,
            }},
            {"id": "quality", "block": "quality_gate", "params": {
                "min_scaled_jacobian": 0.2, "max_edge_ratio": 12.0,
                "max_volume_error": 0.05,
            }},
            {"id": "output", "block": "exodus_output", "params": {
                "path": "output/mesh.e", "write_report": True, "write_vtk": True,
            }},
        ],
        "edges": [
            {"from": "domain", "to": "packing"},
            {"from": "shapes", "to": "packing"},
            {"from": "packing", "to": "template"},
            {"from": "template", "to": "quality"},
            {"from": "quality", "to": "output"},
        ],
    }


def mesh_preview(mesh: HexMesh, max_faces: int = 12_000) -> Dict[str, Any]:
    """Extract exterior and material-interface quads for a compact web preview."""
    if mesh.n_elements == 0:
        return {"points": [], "faces": [], "kind": [], "quality": [], "bounds": []}
    local = np.asarray([s[:4] for s in side_nodes(mesh.element_type)], dtype=np.int64)
    faces = mesh.cells[:, local].reshape(-1, 4)
    owners = np.repeat(np.arange(mesh.n_elements, dtype=np.int64), 6)
    keys = np.sort(faces, axis=1)
    order = np.lexsort(keys.T[::-1])
    keys_s = keys[order]
    owners_s = owners[order]
    faces_s = faces[order]

    selected_faces: list[np.ndarray] = []
    selected_owners: list[int] = []
    selected_kind: list[int] = []  # 0 exterior, 1 material interface
    start = 0
    while start < keys_s.shape[0]:
        stop = start + 1
        while stop < keys_s.shape[0] and np.array_equal(keys_s[stop], keys_s[start]):
            stop += 1
        count = stop - start
        if count == 1:
            selected_faces.append(faces_s[start])
            selected_owners.append(int(owners_s[start]))
            selected_kind.append(0)
        elif count == 2:
            a, b = int(owners_s[start]), int(owners_s[start + 1])
            if mesh.block_id[a] != mesh.block_id[b]:
                selected_faces.append(faces_s[start])
                selected_owners.append(a)
                selected_kind.append(1)
        start = stop

    if not selected_faces:
        return {"points": [], "faces": [], "kind": [], "quality": [], "bounds": []}
    chosen = np.arange(len(selected_faces))
    if chosen.size > max_faces:
        kinds = np.asarray(selected_kind)
        interface = chosen[kinds == 1]
        exterior = chosen[kinds == 0]
        room = max(max_faces - interface.size, 0)
        if interface.size > max_faces:
            interface = interface[np.linspace(0, interface.size - 1, max_faces, dtype=int)]
            exterior = np.zeros(0, dtype=int)
        elif exterior.size > room:
            exterior = exterior[np.linspace(0, exterior.size - 1, room, dtype=int)]
        chosen = np.concatenate([interface, exterior])

    face_array = np.asarray(selected_faces, dtype=np.int64)[chosen]
    owner_array = np.asarray(selected_owners, dtype=np.int64)[chosen]
    kind_array = np.asarray(selected_kind, dtype=np.int8)[chosen]
    used, inverse = np.unique(face_array, return_inverse=True)
    preview_points = mesh.points[used]
    quality = scaled_jacobian(mesh.points, mesh.cells)[owner_array]
    bounds = np.stack([mesh.points.min(axis=0), mesh.points.max(axis=0)])
    return {
        "points": np.round(preview_points, 10).tolist(),
        "faces": inverse.reshape(-1, 4).tolist(),
        "kind": kind_array.tolist(),
        "quality": np.round(quality, 6).tolist(),
        "bounds": bounds.tolist(),
        "sampled": len(selected_faces) > len(chosen),
        "n_visible_faces": len(selected_faces),
    }


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    output_dir: str | Path = "output/gui",
) -> None:  # pragma: no cover - interactive entry point
    import uvicorn

    uvicorn.run(create_app(output_dir), host=host, port=port)


def _version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0"


__all__ = ["create_app", "default_graph", "mesh_preview", "serve"]
