# Research

The design of Hexaform is drawn from two independent AI-assisted literature
reviews of hexahedral mesh generation for random spherical-particle
microstructures, both commissioned for this project. They are preserved here
verbatim, together with a synthesis explaining which of their recommendations
became code and which did not.

| File | Contents |
| --- | --- |
| [`sources/scholargpt-review.md`](sources/scholargpt-review.md) | ScholarGPT review. Algorithm-family survey, tool assessment, a Cubit/Sculpt-first workflow, Exodus guidance, phase-field and K-dominance considerations, 19 numbered references (S1–S19). |
| [`sources/scispace-review.md`](sources/scispace-review.md) | SciSpace review. Why random spheres are topologically hard, unconstrained plastering and Whisker Weaving background, octree/polycube/cuboid-decomposition families, a proposed module decomposition, validation metrics, 8 linked sources. |
| [`synthesis.md`](synthesis.md) | Where the two reviews agree, where they differ, and how each recommendation maps onto a module in this repository. **Start here.** |
| [`bibliography.md`](bibliography.md) | Consolidated, de-duplicated reference list from both reviews. |

## Provenance and how to read these

Both documents are **AI-generated literature reviews**, not peer-reviewed
publications. They were produced by different systems (ScholarGPT and SciSpace)
working from the same problem statement, which makes their points of agreement
unusually informative: two independent syntheses of the same literature
converging on the same recommendation is a stronger signal than either alone.

They should be treated as a well-organised map of the primary literature rather
than as authorities in themselves. Claims that materially shaped the
architecture were checked against the underlying papers listed in
[`bibliography.md`](bibliography.md), and where a claim could not be checked it
is flagged as such in [`synthesis.md`](synthesis.md).

Neither review has been edited. Their section numbering is referenced throughout
the code and documentation — for example
`hexaform/templates/macroblock.py` cites the SciSpace review's section 18 on the
distinction between voxel topology and voxelated geometry, and
`hexaform/io/exodus.py` cites the ScholarGPT review's warning about generic
Exodus converters.

## The one-line summary

> Both reviews independently conclude that no robust general-purpose all-hex
> mesher exists, and that the correct response is **not to build one** — but to
> exploit the fact that every internal surface here is analytic, using
> particle-local cubed-sphere O-grid templates embedded in a background grid.

Hexaform is an implementation of exactly that recommendation.
