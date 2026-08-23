# Synthesis: from two literature reviews to one architecture

This document reconciles the two source reviews and records which of their
recommendations became code, which were deferred, and which were declined.

---

## 1. Where the reviews agree

These points appear independently in both documents. Convergence between two
separately produced syntheses is the strongest evidence available here, and each
of these agreements is load-bearing in the implementation.

### 1.1 There is no general-purpose all-hex mesher, and building one is the wrong goal

ScholarGPT: automatic general-purpose all-hex meshing remains an open problem;
the 2026 literature is still investigating AI-assisted block decomposition,
polycubes and parameterisation rather than reporting a solved problem.

SciSpace: robust boundary-conforming all-hex meshing of arbitrary 3-D geometry
is substantially harder than tetrahedral meshing; there is still no
"TetGen for hexes".

Both then make the same move: *do not attack this as a general problem*, because
the geometry here is not arbitrary — every internal surface is analytic.

> **Implemented as:** the entire architecture. Hexaform meshes a restricted
> class (star-shaped bodies in a cuboidal domain) exactly, rather than an
> unrestricted class approximately.

### 1.2 The cubed sphere, not latitude/longitude

Both reviews single out the six-patch cubed sphere as the correct surface
parameterisation, because a polar parameterisation puts a coordinate singularity
on the interface — precisely where element quality matters most.

> **Implemented as:** `hexaform/templates/cubed_sphere.py`.
> Verified: node count is exactly `6n² + 2` at every resolution (shared patch
> edges weld exactly), all quads wind outward, and equiangular spacing is
> measurably more uniform than gnomonic.

### 1.3 A radial O-grid shell around every inclusion

ScholarGPT calls these "particle-local cubed-sphere macroblocks"; SciSpace calls
them "capsule/O-grid blocks" with an exterior offset surface `S⁺`. The
construction described is the same: connect corresponding cubed-sphere nodes
radially to obtain `n_r` clean hexahedral layers.

> **Implemented as:** the inner and outer shells in
> `hexaform/templates/macroblock.py`, with independent layer counts and grading
> on each side of the interface.

### 1.4 A HEX8 mesh cannot contain a mathematically exact sphere

Both reviews devote a section to this, and both reach the same reformulation of
the requirement: *nodes exactly on the analytic surface, planar facets between
them, controlled geometric error, no staircase.*

> **Implemented as:** interface nodes are placed by evaluating
> `support_radius`, never by snapping. Measured node radial error is
> **< 1e-14** — round-off, not discretisation. Facet-induced volume error is
> reported separately and converges at second order (see §4).
>
> This is stated plainly in the README rather than buried, because "perfectly
> spherical" is the one place where the user's stated requirement and what HEX8
> can deliver genuinely diverge.

### 1.5 Separate packing from meshing

ScholarGPT: store every particle as `(id, cx, cy, cz, radius)`; random packing
is a solved subproblem. SciSpace: "the mesher should not care how that list was
generated."

> **Implemented as:** `hexaform/packing/` emits shapes and macro-boxes;
> `hexaform/assembly/` consumes them and never asks where they came from.

### 1.6 Element quality must be treated as a first-class output

Both reviews stress that valid connectivity is only half the problem, and that a
topologically correct mesh can still contain isolated inverted elements.
ScholarGPT gives the priority ordering explicitly: *validity first, quality
second, geometric fidelity third, element count fourth*.

> **Implemented as:** `hexaform/core/quality.py` (Verdict-style scaled Jacobian)
> and `hexaform/core/validation.py`, whose gates are ordered exactly as above.

### 1.7 Exodus must be a data model, not a conversion target

ScholarGPT cites published experience where a generic converter's ExodusII
output did not produce a working file and the authors implemented output via
netCDF directly. SciSpace's section 22 insists blocks and sets be *designed*
before export rather than reconstructed from coordinates afterwards.

> **Implemented as:** `hexaform/io/exodus.py`, a native writer over netCDF-3 with
> two interchangeable backends, plus a reader used for round-trip validation.
> Blocks, side sets and node sets are created in `assembly/builder.py` by
> construction, not recovered geometrically.

### 1.8 Ligaments matter more than sphere smoothness

Both reviews argue that for a fracture problem the thin matrix ligaments between
close particles — likely crack-initiation sites — deserve more attention than
making the sphere look round. SciSpace formalises it as
`g_min ≳ n_g · h_target`.

> **Implemented as:** the packer's `gap_cells` parameter, which is an *integer
> count of guaranteed matrix elements* across the narrowest ligament rather than
> a length tolerance.

### 1.9 Phase-field length may dominate the element count

Both note that for `ℓ ≪ R` it is fracture physics, not sphere curvature, that
sets `h` — and both frame this as the *reason* removing staircasing is
worthwhile, since it lets physics rather than geometric artefact determine
resolution.

> **Implemented as:** `recommended_mesh_density()` and the `h/ℓ` block of the
> validation report. Note the deliberate restraint: supplying
> `phase_field_length` **reports** `h/ℓ` but never silently changes the mesh.

---

## 2. Where the reviews differ

### 2.1 Which tool to build on

| | ScholarGPT | SciSpace |
| --- | --- | --- |
| First choice | Cubit/Sculpt (overlay-grid + projection) | CUBIT + custom cubed-sphere decomposition |
| Custom Python route | ranked #2, "best custom algorithm" | ranked #2, "best research prototype" |
| Gmsh | usable only if you supply your own topology | "not my first strict-all-hex choice" |

Both rank a commercial, licence-bound tool first, and the pure-Python
cubed-sphere route second.

**Decision: implement the second choice, not the first.** The stated goal is a
Python-based, extensible, open repository with a future browser GUI. A
Cubit-dependent core cannot be any of those. Both reviews describe the
custom route as viable; ScholarGPT explicitly says that if an agent is going to
implement a mesher, it should be given "the cubed-sphere macroblock + adaptive
octree fallback specification".

The cost of this decision is honest and worth naming: Cubit's Sculpt would
handle *touching or overlapping* particles that Hexaform's packer currently
rejects. That gap is the octree backend's job (§3.1).

### 2.2 How to fill the matrix between shells

ScholarGPT proposes a Cartesian/octree background that the macro-boxes plug
into. SciSpace's Strategy 1 proposes CAD decomposition of the matrix into
sweepable volumes, keeping octree as Strategy 2.

**Decision: ScholarGPT's structure, made exact.** SciSpace correctly identifies
matrix decomposition as "the hardest implementation component" — so Hexaform
sidesteps it. The macro-box is aligned to whole background cells and the patch
resolution is set to `n = k`, which means the shell's outermost layer *is* the
background grid on that face. The hard problem is not solved; it is defined
away.

This is the single most important implementation decision in the repository, and
neither review states it explicitly — it emerges from combining them.

### 2.3 Fixed vs adaptive shell thickness

SciSpace's section 9 proposes adapting each particle's O-grid radius to its
neighbourhood via `t_i ≤ α·d_i`, driven by a particle proximity graph.
ScholarGPT proposes refining the background locally instead.

**Decision: neither, for now — a stronger constraint replaces both.** Because
macro-boxes are integer cell boxes, "shells must not collide" becomes
"macro-boxes must be disjoint", which is exact, trivially testable, and enforced
*during packing*. Anything that packs is guaranteed meshable, so the
pack → mesh → fail → repack retry loop that SciSpace's section 20 designs around
does not arise.

The proximity graph is still the right structure for the octree backend, where
boxes *will* be allowed to interact. It is deferred, not rejected.

---

## 3. Recommendation status in 0.2

This table is an architectural mapping, not a release-status log. The current
implementation sequence is maintained in the roadmap.

| Recommendation | Source | Status |
| --- | --- | --- |
| Adaptive octree background with all-hex transition templates | both | **Incomplete research prototype.** Adaptive leaves exist; conforming transitions and robust boundary fitting do not. |
| Pillowing at material interfaces | ScholarGPT step 8 | Implemented and tested as an optional post-process; every result still passes through validation. |
| Compound macro-blocks for close particle pairs | ScholarGPT | Needs the proximity graph; deferred with the octree backend. |
| Jacobian/condition-number barrier optimisation | ScholarGPT step 7 (Knupp) | Implemented as optional constrained smoothing; not a substitute for missing octree transitions. |
| HEX20/HEX27 curved elements | both | Implemented with globally shared nodes and analytic interface projection; gated on solver support. |
| Periodic RVE boundary conditions | SciSpace (Elbana et al.) | Implemented for the Cartesian production path; octree periodicity is unsupported. |
| Monte Carlo success-rate campaigns | SciSpace §26–27 | **Partly done**: `hexaform campaign` and the CSV/summary writer exist. What remains is running one at scale and publishing the curve. |

---

## 4. Recommendations declined

**Unconstrained plastering / Whisker Weaving.** Both reviews describe these as
essential background but poor first implementations; SciSpace calls plastering
"a research-grade meshing project by itself". Declined, and unlikely to be
revisited — the geometry does not require them.

**Tetrahedral-to-hexahedral subdivision.** ScholarGPT's assessment is decisive:
it multiplies element count, produces mediocre quality and inherits a
tet-approximated surface, defeating the goal. Declined.

**Polycube / frame-field parameterisation.** Elegant, and both reviews find it
academically compelling, but each internal closed surface adds singularity
structure to the volumetric parameterisation. Declined for v0.1; the relevant
insight — that a grid in *parameter* space is compatible with curved *physical*
geometry — is already exploited without the machinery.

**Immersed / cut-cell / finite-cell methods.** ScholarGPT lists these as an
alternative formulation that preserves the sphere analytically without
body-fitted meshing. Declined because they change the FE formulation, and the
target solver expects a conforming Exodus mesh.

---

## 5. What was verified, and what was not

Claims that shaped the architecture were tested numerically rather than taken on
trust. Measured on the current implementation:

| Claim | Source | Verification |
| --- | --- | --- |
| Cubed sphere avoids polar singularities and welds consistently | both | `n_nodes = 6n²+2` exactly for n = 1…16; all patches wind outward. `tests/test_cubed_sphere.py` |
| Nodes can lie exactly on the analytic surface | both | Max radial error **< 1e-14** for spheres, ellipsoids, superquadrics and SH grains. `tests/test_macroblock.py` |
| HEX8 facets leave a residual geometric error | both | Volume error 6.6% → 1.7% → 0.19% for k = 4, 8, 24; ratio confirms **second-order** convergence. |
| An O-grid gives good element quality | both | Min scaled Jacobian **0.577 = 1/√3**, attained at macro-box corners, with no smoothing. Well above the 0.2 acceptance floor both reviews imply. |
| Grid-aligned templates weld without hanging nodes | inferred | Zero hanging nodes, zero non-manifold faces, and total element volume equal to the domain volume to 1e-12. `tests/test_assembly.py` |
| Generic Exodus converters are unreliable | ScholarGPT [S16] | **Not independently verified.** Hexaform writes Exodus natively regardless, so the claim is not load-bearing — the cost of following the advice is low and the cost of ignoring it, if true, is high. |
| Sculpt-quality grid meshes match pave-and-sweep in elastoplastic analysis | ScholarGPT [S3] | **Not verified here** — this is a solver-side result. It supports the overall approach but no Hexaform behaviour depends on it. |
| `h ≤ ℓ/2` for phase-field fracture | ScholarGPT [S18] | **Not verified.** Reported as a default in `recommended_mesh_density`, explicitly flagged in both the source and this repo as requiring a convergence study for a specific elastoplastic formulation. Treat it as an order of magnitude, not a rule. |

The last three deserve emphasis: they are inherited from the reviews, they are
plausible, and they are **not** established by anything in this repository. The
`h ≤ ℓ/2` heuristic in particular should be replaced by your own convergence
study before it is trusted for production runs.

---

## 6. One thing neither review addressed

Both reviews treat the problem as "spheres in a cube". Neither asks what the
*next* geometry would be, which matters for a repository intended to grow.

The generalisation Hexaform adopts is that the macro-block template never needs
a radius — it needs a **support radius function** `r(d̂)`. Any star-shaped body
supplies one. This costs nothing at the sphere-only use case and means
ellipsoids, superquadrics and spherical-harmonic grains are already supported
through the identical code path, verified by parameterised tests.

The boundary of the method is now sharp and statable: **star-shaped bodies work;
non-star-shaped bodies (tori, branching pore networks, interlocking fibres) do
not, and are the octree backend's problem.** That is a more useful frontier than
"spheres work, other things might".
