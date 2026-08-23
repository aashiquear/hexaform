# Roadmap

This is the single implementation-status document. Detailed evidence and
review findings belong in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md); user
instructions belong in the tutorial and user guide.

## Released in 0.2

- Cartesian background and exact integer macrobox carving.
- Cubed-sphere O-grid templates for analytic star-shaped inclusions.
- Anisotropic macroboxes and shape-aware packing.
- RSA and collective rearrangement packing.
- Conforming HEX8 plus global HEX20/HEX27 promotion.
- Topology, Jacobian, edge-ratio, geometry, ligament, and phase-field metrics.
- Native Exodus II read/write with blocks, side sets, node sets, and fields.
- VTK output, JSON/YAML configs, CLI, examples, and campaigns.
- Periodic Cartesian RVE node pairing.
- Notch/phase-field insertion, interface pillowing, and constrained smoothing.
- Local browser Mesh Studio with interactive preview and artifact downloads.
- Consolidated `output/`, package installation, CI, governance, and technical
  report generation.

## Active engineering priorities

### 1. Adaptive octree transition templates

The current octree can construct adaptive leaves and analytic projection
experiments, but leaves are not made conforming across level transitions.
Projection can invert interface cells. Completion requires:

1. 2:1 balancing;
2. all-hex transition templates or a dual/hybrid-octree extraction;
3. interface reconstruction based on each shape's actual SDF;
4. constrained Jacobian optimization;
5. zero hanging nodes, zero inverted cells, and geometry convergence tests;
6. periodic-octree support or an explicit incompatibility contract.

Until all gates pass, star-shaped `backend=octree` requests use the conforming
Cartesian macroblock path and record that fallback in mesh metadata.

### 2. General non-star-shaped bodies

`Torus` supplies a correct SDF and analytic projector, but it is not a
solver-ready meshing path. It depends on the octree work above. The browser
catalogue marks this block as planned and refuses to compile it.

### 3. Long-running browser jobs

The 0.2 GUI runs synchronously. Before exposing mesh densities above ordinary
desktop sizes, add process-isolated jobs, progress/status endpoints,
cancellation, quotas, retention cleanup, and preview streaming.

### 4. Independent solver benchmarks

Add externally reproducible mechanics regressions:

- stress concentration around one spherical inclusion;
- phase-field crack benchmark and mesh-convergence study;
- target-solver verification of curved HEX20/HEX27 quadrature;
- scaled Monte Carlo success curves with published configurations.

## Deferred by design

General plastering/Whisker Weaving, tet-to-hex subdivision, volumetric
polycube/frame-field parameterization, and immersed/cut-cell methods are not
planned for the production path. The research synthesis explains why these
either solve a materially different problem or carry disproportionate risk.

## Release gates

Every release must pass:

- the full test suite and smoke examples;
- package installation from a fresh virtual environment;
- Exodus and VTK artifact checks;
- browser API and interaction checks;
- generated documentation/report checks;
- no unreviewed claim of production completeness for experimental algorithms.
