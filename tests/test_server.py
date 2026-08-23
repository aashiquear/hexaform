"""Browser API, graph-validation, preview, and output regressions."""

from pathlib import Path

import numpy as np
import pytest

from hexaform import RunConfig, run
from hexaform.assembly.octree_assembly import OctreeAssemblyOptions, assemble_with_octree
from hexaform.core.validation import validate
from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig
from hexaform.server.app import default_graph, mesh_preview
from hexaform.server.blocks import compile_graph, validate_graph
from hexaform.shapes.torus import Torus


def test_default_browser_graph_is_valid_and_writes_vtk():
    graph = default_graph()
    assert validate_graph(graph) == []
    config = compile_graph(graph)
    assert config.output.path == "output/mesh.e"
    assert config.output.write_vtk is True


def test_graph_rejects_duplicate_nodes_unknown_parameters_and_bad_ranges():
    graph = default_graph()
    graph["nodes"].append(dict(graph["nodes"][0]))
    graph["nodes"][1]["params"]["not_a_parameter"] = 1
    graph["nodes"][2]["params"]["box_ratio"] = 0.5
    problems = validate_graph(graph)
    assert any("duplicate node id" in problem for problem in problems)
    assert any("unknown parameter" in problem for problem in problems)
    assert any("below" in problem for problem in problems)


def test_graph_rejects_malformed_edges_without_crashing():
    graph = default_graph()
    graph["edges"] = ["not-an-edge"]
    assert any("edge must be an object" in problem for problem in validate_graph(graph))


def test_non_star_gui_block_is_truthfully_blocked():
    graph = default_graph()
    graph["nodes"][1] = {
        "id": "shapes",
        "block": "non_star_body",
        "params": {"shape": "torus", "major_radius": 0.25, "minor_radius": 0.1},
    }
    assert any("not implemented" in problem for problem in validate_graph(graph))


def test_optional_graph_blocks_enable_their_pipeline_stages():
    graph = default_graph()
    graph["nodes"].extend([
        {"id": "pillow", "block": "pillow_layer", "params": {"thickness_fraction": 0.1}},
        {"id": "smooth", "block": "quality_smoothing", "params": {"n_iterations": 3}},
    ])
    graph["edges"] = [
        {"from": "domain", "to": "packing"},
        {"from": "shapes", "to": "packing"},
        {"from": "packing", "to": "template"},
        {"from": "template", "to": "pillow"},
        {"from": "pillow", "to": "smooth"},
        {"from": "smooth", "to": "quality"},
        {"from": "quality", "to": "output"},
    ]
    config = compile_graph(graph)
    assert config.pillow.enabled is True
    assert config.smoothing.enabled is True
    assert config.smoothing.n_iterations == 3


def test_pipeline_honours_vtk_output_flag(tmp_path):
    config = RunConfig(
        domain=DomainConfig(length=1.0, mesh_density=8),
        inclusions=InclusionConfig(n_particles=1, radius_mean=0.12),
        output=OutputConfig(path=str(tmp_path / "mesh.e"), write_vtk=True),
        seed=2,
        verbose=False,
    )
    result = run(config)
    assert result.vtk_path == tmp_path / "mesh.vtk"
    assert result.vtk_path.is_file()
    assert result.vtk_path.read_text().startswith("# vtk DataFile")


def test_browser_preview_contains_exterior_and_interface_faces(tmp_path):
    config = RunConfig(
        domain=DomainConfig(length=1.0, mesh_density=8),
        inclusions=InclusionConfig(n_particles=1, radius_mean=0.12),
        output=OutputConfig(path=""),
        seed=4,
        verbose=False,
    )
    result = run(config)
    preview = mesh_preview(result.mesh)
    assert preview["points"]
    assert preview["faces"]
    assert 0 in preview["kind"]
    assert 1 in preview["kind"]
    assert len(preview["quality"]) == len(preview["faces"])


def test_octree_request_discloses_cartesian_fallback(tmp_path):
    config = RunConfig(
        domain=DomainConfig(length=0.4, mesh_density=8, backend="octree"),
        inclusions=InclusionConfig(n_particles=1, radius_mean=0.05),
        output=OutputConfig(path=""),
        seed=8,
        verbose=False,
    )
    result = run(config)
    assert result.validation.passed
    assert result.mesh.metadata["backend_requested"] == "octree"
    assert result.mesh.metadata["background"] == "cartesian"
    assert result.mesh.metadata["adaptive_refinement_applied"] is False


def test_non_star_octree_prototype_does_not_mask_invalidity():
    torus = Torus(np.array([0.5, 0.5, 0.5]), major_radius=0.25, minor_radius=0.1)
    mesh = assemble_with_octree(
        [torus],
        options=OctreeAssemblyOptions(
            bulk_h=0.125, interface_h=0.0625, interface_band=0.05, max_level=4,
        ),
    )
    report = validate(mesh)
    assert mesh.metadata["experimental"] is True
    assert mesh.metadata["conforming_transition_templates"] is False
    assert report.passed is False
    assert report.topology["n_hanging_nodes"] > 0
    assert report.quality.n_inverted > 0


def test_browser_http_end_to_end(tmp_path):
    import asyncio

    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from hexaform.server.app import create_app

    async def exercise_api():
        graph = default_graph()
        graph["nodes"][0]["params"]["mesh_density"] = 8
        graph["nodes"][1]["params"]["n_particles"] = 1
        transport = httpx.ASGITransport(app=create_app(tmp_path / "gui"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/")).status_code == 200
            assert (await client.get("/api/health")).json()["status"] == "ok"
            response = await client.post("/api/graph/run", json=graph)
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["report"]["success"] is True
            assert set(data["downloads"]) == {"exodus", "vtk", "report", "config"}
            for href in data["downloads"].values():
                assert (await client.get(href)).status_code == 200
            assert (await client.get("/api/runs/not-a-token/exodus")).status_code == 404

    asyncio.run(exercise_api())
