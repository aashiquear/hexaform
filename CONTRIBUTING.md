# Contributing

Hexaform accepts focused changes that preserve the restricted, validity-first
scope described in [the roadmap](docs/roadmap.md).

## Development workflow

1. Create a feature branch from `master`.
2. Run `./scripts/setup.sh` once, then make the smallest coherent change.
3. Add tests for behavior and failure modes; never weaken a validity gate to
   make an experimental mesh pass.
4. Run `make test`, `make examples`, and `make report` when documentation or
   architecture changes.
5. Open a pull request using the repository template. One approving review and
   green CI are required before merge.

Use conventional commit subjects where practical (`feat:`, `fix:`, `docs:`,
`test:`, `refactor:`). Generated files belong under `output/` and must not be
committed. Research-source reviews are preserved evidence; corrections belong
in the synthesis or a new, cited source note.

## Definition of done

- public behavior and limitations are documented;
- unit/regression tests cover the change;
- all mesh outputs pass topology and scaled-Jacobian validation;
- the full suite and example smoke suite pass;
- roadmap/report claims match executable behavior.
