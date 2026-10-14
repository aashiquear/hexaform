# Changelog

## Unreleased

- Added `hexaform plate` / `hexaform.plate.plate_mesh`: rectangular plates with
  one hexahedron through the thickness and a conforming circular or elliptical
  through-hole meshed by a graded O-grid ring (node/side sets `hole`, `outer`,
  blocks `plate` and `ring`), with tests, user-guide and example 12.

## 0.2.0 - 2026-08-23

- Added a packaged local browser Mesh Studio with interactive quality preview
  and Exodus, VTK, report, and configuration downloads.
- Made VTK output and graph optional stages effective and validated.
- Disclosed the Cartesian fallback for requested star-shape octree runs and
  marked non-star octree assembly experimental.
- Added browser/API/fallback regression tests and all-example smoke execution.
- Consolidated documentation, generated artifacts, installation, governance,
  and reproducible technical-report workflows.

## 0.1.0 - 2026-08-14

- Initial Cartesian cubed-sphere macroblock implementation for star-shaped
  inclusions with Exodus II output and validation.
