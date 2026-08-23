"""End-to-end behaviour, including the CLI and the GUI block graph."""

import json

import numpy as np
import pytest

from hexaform import particles_in_cube
from hexaform.cli import main
from hexaform.core.validation import recommended_mesh_density
from hexaform.pipeline.config import DomainConfig, InclusionConfig, OutputConfig, RunConfig
from hexaform.pipeline.run import run
from hexaform.server import compile_graph, validate_graph


def _config(tmp_path, **kw):
    return RunConfig(
        domain=DomainConfig(length=1.0, mesh_density=kw.pop("density", 16)),
        inclusions=InclusionConfig(n_particles=4, radius_mean=0.10,
                                   radius_std=0.02, **kw.pop("incl", {})),
        output=OutputConfig(path=str(tmp_path / "m.e")),
        seed=kw.pop("seed", 3),
        verbose=False,
        **kw,
    )


def test_end_to_end_run_passes_validation(tmp_path):
    result = run(_config(tmp_path))
    assert result.validation.passed, result.validation.failures
    assert result.output_path.is_file()
    assert result.report_path.is_file()


def test_report_json_is_complete(tmp_path):
    result = run(_config(tmp_path))
    data = json.loads(result.report_path.read_text())
    assert data["success"] is True
    assert data["mesh"]["element_type"] == "HEX8"
    assert len(data["inclusions"]) == result.packing.count
    assert data["config"]["seed"] == 3
    for inc in data["inclusions"]:
        assert all(c >= 4 for c in inc["macrobox_cells"])  # per-axis (kx, ky, kz)


def test_runs_are_reproducible(tmp_path):
    a = run(_config(tmp_path / "a"))
    b = run(_config(tmp_path / "b"))
    assert np.array_equal(a.mesh.points, b.mesh.points)
    assert np.array_equal(a.mesh.cells, b.mesh.cells)


def test_seed_changes_the_mesh(tmp_path):
    a = run(_config(tmp_path / "a", seed=1))
    b = run(_config(tmp_path / "b", seed=2))
    assert a.mesh.points.shape != b.mesh.points.shape or not np.array_equal(
        a.mesh.points, b.mesh.points
    )


def test_mesh_density_controls_element_count(tmp_path):
    low = run(_config(tmp_path / "l", density=12))
    high = run(_config(tmp_path / "h", density=24))
    assert high.mesh.n_elements > 4 * low.mesh.n_elements


@pytest.mark.parametrize("shape", ["sphere", "ellipsoid", "superquadric", "sh_grain"])
def test_every_shape_family_runs_end_to_end(tmp_path, shape):
    cfg = _config(tmp_path / shape, incl={"shape": shape,
                                          "aspect_ratios": (1.0, 0.8, 0.65)},
                  density=24)
    result = run(cfg)
    assert result.validation.passed, result.validation.failures


def test_convenience_helper(tmp_path):
    result = particles_in_cube(
        length=1.0, n_particles=3, radius_mean=0.11, mesh_density=14,
        seed=1, output=str(tmp_path / "p.e"), verbose=False,
    )
    assert result.validation.passed
    assert result.output_path.is_file()


def test_phase_field_metrics_are_reported(tmp_path):
    cfg = _config(tmp_path)
    cfg.phase_field_length = 0.03
    result = run(cfg)
    pf = result.validation.phase_field
    assert pf["phase_field_length"] == pytest.approx(0.03)
    assert pf["h_over_ell_bulk"] > 0


def test_config_json_round_trip(tmp_path):
    cfg = _config(tmp_path)
    path = cfg.to_json(tmp_path / "cfg.json")
    back = RunConfig.load(path)
    assert back.seed == cfg.seed
    assert back.domain.mesh_density == cfg.domain.mesh_density
    assert back.inclusions.radius_mean == cfg.inclusions.radius_mean
    assert back.template.n_inner == cfg.template.n_inner


def test_config_ignores_unknown_fields_for_forward_compatibility():
    cfg = RunConfig.from_dict({
        "future_top_level": {"enabled": True},
        "domain": {"mesh_density": 8, "future_domain_field": 1},
    })
    assert cfg.domain.mesh_density == 8


@pytest.mark.parametrize("change, message", [
    (lambda c: setattr(c.inclusions, "n_particles", (-1, 2)), "n_particles"),
    (lambda c: setattr(c.inclusions, "aspect_ratios", (1.0, 0.0, 1.0)), "aspect_ratios"),
    (lambda c: setattr(c.inclusions, "gap_cells", -1), "gap_cells"),
    (lambda c: setattr(c, "phase_field_length", 0.0), "phase_field_length"),
    (lambda c: setattr(c.output, "exodus_backend", "unknown"), "exodus_backend"),
])
def test_config_validation_rejects_invalid_physical_inputs(change, message):
    cfg = RunConfig()
    change(cfg)
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_recommended_density_follows_the_binding_constraint():
    geometry_bound = recommended_mesh_density(1.0, 0.08)
    physics_bound = recommended_mesh_density(1.0, 0.08, phase_field_length=0.01)
    assert physics_bound > geometry_bound


# ---- CLI -------------------------------------------------------------
def test_cli_particles_in_cube(tmp_path):
    code = main([
        "particles-in-cube", "--length", "1.0", "--n-particles", "3",
        "--radius-mean", "0.11", "--mesh-density", "14", "--seed", "2",
        "-o", str(tmp_path / "cli.e"), "--quiet",
    ])
    assert code == 0
    assert (tmp_path / "cli.e").is_file()


def test_cli_accepts_a_particle_count_range(tmp_path):
    code = main([
        "particles-in-cube", "--n-particles", "2-5", "--radius-mean", "0.10",
        "--mesh-density", "14", "--seed", "8", "-o", str(tmp_path / "r.e"), "--quiet",
    ])
    assert code == 0


def test_cli_inspect_round_trips(tmp_path, capsys):
    main(["particles-in-cube", "--n-particles", "2", "--radius-mean", "0.12",
          "--mesh-density", "12", "-o", str(tmp_path / "i.e"), "--quiet"])
    assert main(["inspect", str(tmp_path / "i.e")]) == 0
    out = capsys.readouterr().out
    assert "HEX8" in out and "VALID" in out


def test_cli_suggest_density(capsys):
    assert main(["suggest-density", "--radius-mean", "0.08",
                 "--phase-field-length", "0.01"]) == 0
    assert "--mesh-density" in capsys.readouterr().out


def test_cli_run_from_config(tmp_path):
    cfg = _config(tmp_path)
    path = cfg.to_json(tmp_path / "c.json")
    assert main(["run", str(path), "-o", str(tmp_path / "fromcfg.e")]) == 0
    assert (tmp_path / "fromcfg.e").is_file()


# ---- GUI block graph --------------------------------------------------
def _graph(path):
    return {
        "nodes": [
            {"id": "d", "block": "cube_domain",
             "params": {"length": 1.0, "mesh_density": 14}},
            {"id": "s", "block": "sphere_source",
             "params": {"n_particles": 3, "radius_mean": 0.11}},
            {"id": "p", "block": "rsa_packing", "params": {"seed": 6}},
            {"id": "t", "block": "macroblock_template",
             "params": {"n_inner": 2, "n_outer": 2}},
            {"id": "q", "block": "quality_gate", "params": {"min_scaled_jacobian": 0.2}},
            {"id": "o", "block": "exodus_output", "params": {"path": path}},
        ],
        "edges": [{"from": "d", "to": "p"}, {"from": "s", "to": "p"},
                  {"from": "p", "to": "t"}, {"from": "t", "to": "q"},
                  {"from": "q", "to": "o"}],
    }


def test_valid_graph_compiles_and_runs(tmp_path):
    graph = _graph(str(tmp_path / "g.e"))
    assert validate_graph(graph) == []
    cfg = compile_graph(graph)
    cfg.verbose = False
    assert cfg.domain.mesh_density == 14
    assert cfg.inclusions.n_particles == 3
    assert cfg.seed == 6
    assert run(cfg).validation.passed


def test_graph_missing_an_output_block_is_rejected(tmp_path):
    graph = _graph(str(tmp_path / "g.e"))
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != "o"]
    assert any("output" in p for p in validate_graph(graph))


def test_unimplemented_block_is_flagged(tmp_path):
    """A block marked ``planned`` in the registry must be flagged by validate_graph."""
    from hexaform.server.blocks import BlockSpec, register, REGISTRY

    # Inject a temporary planned block; the test then verifies the validation hook.
    register(
        BlockSpec(
            key="test_planned_block",
            title="A test planned block",
            category="assembly",
            description="",
            outputs=[],
            status="planned",
        )
    )
    try:
        graph = _graph(str(tmp_path / "g.e"))
        graph["nodes"][0] = {"id": "d", "block": "test_planned_block", "params": {}}
        assert any("not implemented" in p for p in validate_graph(graph))
        with pytest.raises(ValueError):
            compile_graph(graph)
    finally:
        REGISTRY.pop("test_planned_block", None)


def test_unknown_block_is_rejected(tmp_path):
    graph = _graph(str(tmp_path / "g.e"))
    graph["nodes"].append({"id": "x", "block": "does_not_exist", "params": {}})
    assert any("unknown block" in p for p in validate_graph(graph))


def test_validation_omits_checks_it_cannot_perform(tmp_path):
    """Re-reading an Exodus file loses the analytic shapes; report must not lie."""
    from hexaform.core.validation import validate
    from hexaform.io import exodus

    result = run(_config(tmp_path))
    text = str(validate(exodus.read(result.output_path)))
    assert "nan" not in text
    assert "skipped" in text
    # with the shapes supplied, the checks are present again
    assert "volume error" in str(result.validation)
