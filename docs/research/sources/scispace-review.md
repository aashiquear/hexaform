# Research brief: non-voxelated all-hexahedral meshing of random spherical-particle microstructures

## 🧠 Executive assessment

Your problem sits in one of the genuinely difficult corners of automatic mesh generation:

$$
\text{cube} - \bigcup_{i=1}^{N}\text{spherical interfaces}
\quad\longrightarrow\quad
\text{conformal, low-distortion, all-hex mesh}
$$

where (N), particle centers, and potentially particle radii change from realization to realization.

The literature is quite clear on an important point: **fully automatic, robust, boundary-conforming all-hexahedral meshing of arbitrary 3-D geometry remains substantially harder than tetrahedral meshing**. Advancing-front/plastering, whisker weaving, octree/dual methods, frame-field parameterization, polycubes, and geometric decomposition all solve portions of the problem, but there is still no universally reliable "TetGen for hexes." Staten et al. explicitly describe arbitrary all-hex generation as an open problem, while later polycube work still identifies robust conformal adaptive all-hex generation as challenging ([Staten et al., 2010](https://scispace.com/papers/unconstrained-plastering-hexahedral-mesh-generation-via-21a5v6p6r3?utm_source=chatgpt); [Hu & Zhang, 2016](https://scispace.com/papers/centroidal-voronoi-tessellation-based-polycube-construction-1500d9cj6w?utm_source=chatgpt)).

For **your particular geometry**, however, I would *not* start by trying to implement a completely general all-hex algorithm. Spheres give you valuable geometric structure. The strongest engineering route is:

> **Create a curved, structured all-hex "capsule" or O-grid around every sphere, then automatically decompose and hex-mesh the remaining matrix region, with local refinement driven by particle separation and phase-field length scale.**

For a code-generating AI agent, I would investigate **CUBIT/PyCubIT first**, followed by a custom sphere-template + decomposition algorithm. A pure Gmsh solution is much less attractive if *strictly all HEX8* is non-negotiable. For the final Exodus database, **CUBIT is particularly natural because Exodus is native to its Sandia/finite-element ecosystem**.

There is also a deeper modeling point: a geometrically exact mathematical sphere cannot literally be represented by planar-faced first-order HEX8 elements. You can have either (1) nodes lying exactly on the analytical sphere but planar element faces between them, or (2) **curved higher-order hex faces** whose geometry approximates the sphere to much higher order. So the useful target is not "perfect sphere with HEX8," but **boundary-conforming spherical geometry to a specified geometric error**, potentially using HEX20/HEX27 if supported by the fracture code.

---

# 1. Why random spheres make the topology difficult

A single centered sphere inside a cube is straightforward because you can impose symmetry and construct something resembling a spherical O-grid:

$$
(r,\theta,\phi)\rightarrow (x,y,z),
$$

and transition that regular shell into the surrounding cube.

Once spheres are randomly positioned, four problems appear.

### 1.1 There is no global sweep direction

Classical mapped/swept hex meshing works beautifully when the volume is topologically equivalent to an extrusion. A cube containing several randomly placed internal surfaces generally is not.

### 1.2 Every particle introduces an internal boundary

With (N) particles, the matrix is essentially a multiply connected domain. A useful conceptual form is

$$
\Omega_m =
\Omega_{\rm cube}\setminus
\bigcup_{i=1}^{N}\Omega_i.
$$

Every (\partial\Omega_i) needs a conforming quadrilateral surface mesh if matrix and particle share nodes.

Polycube research has explicitly demonstrated handling disconnected boundaries/voids, but increasing topology still complicates parameterization and singularity placement ([Fang et al., 2016](https://scispace.com/papers/all-hex-meshing-using-closed-form-induced-polycube-1kg6d4ys3t?utm_source=chatgpt)).

### 1.3 Close particle pairs create thin ligaments

For particles (i,j),

$$
g_{ij} =
|\mathbf c_i-\mathbf c_j|-R_i-R_j.
$$

As (g_{ij}\rightarrow0), maintaining several good-quality hexahedral layers between the surfaces becomes increasingly difficult.

This is likely more important for your phase-field problem than simply reproducing spherical curvature.

### 1.4 Hex topology is globally coupled

A tetrahedral mesh permits many local topological modifications. Hex connectivity is much more constrained. Staten et al. emphasize that relatively small local changes can have global consequences because every hex must maintain its eight vertices, twelve edges, six quadrilateral faces, and acceptable warping/skew ([Staten et al., 2010](https://scispace.com/papers/unconstrained-plastering-hexahedral-mesh-generation-via-21a5v6p6r3?utm_source=chatgpt)).

That is why an algorithm that says "snap Cartesian nodes onto every sphere" eventually produces terrible elements.

---

# 2. Algorithm families worth investigating

## A. Geometry decomposition + mapping/sweeping

This is probably the most immediately useful family for your application.

The idea is:

1. construct exact CAD spheres;
2. create partitions around each sphere;
3. decompose the matrix into sweepable/mappable subvolumes;
4. map quadrilateral meshes to surfaces;
5. sweep those surfaces into HEX8/HEX27 volumes.

Traditional decomposition methods use geometry features or medial surfaces to divide complicated bodies into simple regions that can subsequently be mapped or swept. Their weakness is that decomposition itself can require substantial human intervention for arbitrary geometry ([Staten et al., 2010](https://scispace.com/papers/unconstrained-plastering-hexahedral-mesh-generation-via-21a5v6p6r3?utm_source=chatgpt)).

### Relevance to random spheres: **very high**

Unlike arbitrary CAD geometry, you know beforehand that every inclusion is a sphere. Therefore you can write a deterministic decomposition template for each particle.

A very promising construction is:

$$
\text{spherical surface}
\rightarrow
\text{cubed-sphere topology}
\rightarrow
\text{radial/O-grid hex shell}.
$$

A **cubed sphere** divides a sphere into six logically quadrilateral patches. This avoids the pole singularities of ordinary latitude-longitude parameterization.

Each sphere can therefore have:

* 6 quad patches,
* structured (n\times n) quads per patch,
* (n_r) radial layers around the particle,
* exact radial projection of boundary nodes onto (r=R_i).

This creates an excellent local particle mesh independently of particle position.

The difficult part becomes **joining all of those particle blocks into the cube matrix**.

---

# 3. Unconstrained plastering

Unconstrained plastering is one of the most important historical algorithms for your literature review.

Rather than first imposing a complete quadrilateral mesh on the boundary and then trying to fill the volume, it creates an advancing-front geometric decomposition. Successive front advances correspond to inserting fundamental sheets of hexahedra.

Staten et al. report that this produces meshes that are:

* all-hexahedral,
* boundary sensitive,
* relatively orientation insensitive,
* applicable to arbitrary volumetric bodies in demonstrated cases.

([Staten et al., 2010](https://scispace.com/papers/unconstrained-plastering-hexahedral-mesh-generation-via-21a5v6p6r3?utm_source=chatgpt))

The original motivation was precisely the failure mode of traditional plastering: a pre-existing boundary quad mesh can **over-constrain the interior topology** ([Staten, Owen & Blacker, 2005](https://scispace.com/papers/unconstrained-paving-plastering-a-new-idea-for-all-4piyn45wbk?utm_source=chatgpt)).

### Relevance: **high conceptually, moderate as an implementation target**

I would study it carefully, but I would **not recommend that another AI agent implement unconstrained plastering from scratch as version 1**. It is a research-grade meshing project by itself.

Its major lesson for your implementation is more useful:

> Do not completely freeze independently generated quadrilateral meshes on every sphere and then expect an arbitrary all-hex filler to reconcile them effortlessly.

Your particle templates and matrix decomposition should be constructed jointly.

---

# 4. Whisker Weaving / Spatial Twist Continuum

Whisker Weaving attacks the topology before the geometry.

It uses the **dual structure of the hexahedral mesh**, represented through the Spatial Twist Continuum (STC). Dual curves/surfaces determine how hexahedra must connect; physical node positions can then be optimized afterward.

Folwell & Mitchell developed curve-contraction algorithms that make this process substantially more reliable ([Folwell & Mitchell, 1999](https://scispace.com/papers/reliable-whisker-weaving-via-curve-contraction-1m7fm2r3ci?utm_source=chatgpt)).

This work is particularly informative because it demonstrates why all-hex generation remains hard: even when valid topology is obtained, isolated regions can contain poor-quality or even negative-Jacobian elements. Their work therefore combines topology operations with smoothing and local geometric/connectivity rules.

### Relevance: **excellent research background, poor first implementation choice**

It is useful if you eventually want to develop a truly general-purpose all-hex generator. It is probably excessive for "cube + spheres."

---

# 5. Octree and dual-octree all-hex meshing

This family is much closer to the voxel idea—but importantly **does not have to retain a stair-stepped boundary**.

The basic process is:

$$
\text{geometry}
\rightarrow
\text{adaptive octree}
\rightarrow
\text{boundary treatment}
\rightarrow
\text{dualization}
\rightarrow
\text{all-hex mesh}.
$$

Adaptive cells concentrate near interfaces, while coarse cells remain far away.

Hu & Zhang describe an improved hybrid-octree method where adaptive all-hex meshes are obtained from the dual of a hybrid octree. Their method combines grid robustness with a polycube mapping to improve boundary quality ([Hu & Zhang, 2016](https://scispace.com/papers/centroidal-voronoi-tessellation-based-polycube-construction-1500d9cj6w?utm_source=chatgpt)).

The literature they summarize is directly relevant to your choice: grid-based methods are robust for arbitrary geometry but traditionally produce poorer elements near curved boundaries; mapping/polyclube methods improve boundary conformity.

### Relevance: **very high**

If your principal objection to voxelization is **uniformly huge element counts**, adaptive octree meshing is probably the closest conceptual replacement.

Instead of

$$
N_{\rm elem}\sim (L/h)^3
$$

everywhere, you refine primarily where

$$
|\phi_i(\mathbf x)| =
\left||\mathbf x-\mathbf c_i|-R_i\right|
$$

is small.

Thus the fine resolution follows spherical interfaces rather than occupying the whole cube.

---

# 6. Polycube methods

Polycube algorithms deform the physical domain into a collection of axis-aligned cube-like regions. A structured hex grid is generated in parameter space and mapped back to the physical geometry.

Conceptually:

$$
\Omega
\xrightarrow{F}
\Omega_{\rm polycube}
\xrightarrow{\text{Cartesian hex grid}}
H_p
\xrightarrow{F^{-1}}
H.
$$

The great attraction is that mesh validity can be tied to obtaining a bijective parameterization.

Fang et al. extended polycube methods using **closed-form-induced frame fields**, reducing distortion and supporting models with complex topology and disconnected boundaries ([Fang et al., 2016](https://scispace.com/papers/all-hex-meshing-using-closed-form-induced-polycube-1kg6d4ys3t?utm_source=chatgpt)).

Hu & Zhang instead use centroidal Voronoi tessellation for automatic polycube construction and combine this with octree subdivision and adaptive dual extraction ([Hu & Zhang, 2016](https://scispace.com/papers/centroidal-voronoi-tessellation-based-polycube-construction-1500d9cj6w?utm_source=chatgpt)).

### Relevance: **high for a research implementation**

A cube containing multiple spherical boundaries is actually a compelling test problem for this class because geometry is smooth and analytically known.

But implementing a robust volumetric bijective parameterization, integer-grid extraction, singularity handling, and quality optimization is considerably harder than scripting CUBIT.

---

# 7. Modern cuboid-decomposition methods

Li et al. developed an interactive all-hex method that represents the polycube as a **union of cuboids** rather than requiring the input to deform directly into a conventional polycube.

Their workflow roughly consists of:

$$
\text{input tet mesh}
\rightarrow
\text{near-polycube deformation}
\rightarrow
\text{cuboid decomposition}
\rightarrow
\text{voxelized polycube}
\rightarrow
\text{inversion-free pullback}.
$$

The resulting physical mesh is therefore **not a voxel representation of the original boundary**. The regular grid exists in parameter space and is mapped onto the curved physical geometry.

They report robust results on more than 100 test shapes and mesh quality competitive with contemporary methods, although user guidance remains useful ([Li et al., 2021/2022](https://scispace.com/papers/interactive-all-hex-meshing-via-cuboid-decomposition-3lsdvzqc6b?utm_source=chatgpt)).

This distinction is important:

> **A method may internally use voxels without producing voxelated spheres.**

A mapped Cartesian topology is entirely compatible with a curved physical sphere.

### Relevance: **high as an algorithmic direction**

This is one of the more interesting modern approaches for an AI agent to study if the goal eventually becomes developing your own mesher.

---

# 8. The approach I would prototype first: particle-centered cubed-sphere/O-grid blocks

For your specific geometry, exploiting the analytical sphere is likely to outperform trying to solve general CAD hex meshing.

For particle (i),

$$
S_i =
{\mathbf x:
|\mathbf x-\mathbf c_i|=R_i}.
$$

Construct a six-patch quadrilateral sphere and an exterior offset surface

$$
S_i^{+} =
{\mathbf x:
|\mathbf x-\mathbf c_i|=R_i+t_i}.
$$

Connect corresponding quadrilateral nodes radially.

You immediately obtain (n_r) layers of clean hexahedra:

$$
R_i=r_0<r_1<\cdots<r_{n_r}=R_i+t_i.
$$

This gives you:

* excellent particle/interface conformity;
* controllable radial element size;
* no stair-step geometry;
* predictable HEX8 connectivity;
* straightforward material-block assignment;
* independent handling of arbitrary (N).

The outer surface (S_i^+) becomes the interface to the general matrix mesher.

---

# 9. The crucial innovation: adapt each particle's O-grid radius to its neighborhood

A fixed-thickness shell will fail when particles become close.

Compute

$$
d_i =
\min\left[
d(\text{particle }i,\partial\Omega_{\rm cube}),
;
\min_{j\neq i}
(|\mathbf c_i-\mathbf c_j|-R_i-R_j)
\right].
$$

Then choose, for example,

$$
t_i \le \alpha d_i,\qquad 0<\alpha<\frac12.
$$

This guarantees that particle-centered shell regions do not collide.

For close pairs, the shell thickness shrinks and matrix refinement increases in the ligament.

This neighborhood graph should be an explicit object in the implementation:

* vertex = particle;
* edge (i-j) if (g_{ij}<g_{\rm threshold});
* edge weight = (g_{ij}).

That graph can drive both decomposition and refinement.

---

# 10. Why this matters for phase-field fracture

For phase-field fracture, geometrical fidelity is only half the story. The mesh also needs to resolve the regularized crack.

If the phase-field length parameter is (\ell), you generally need a mesh sufficiently fine relative to (\ell); the exact requirement depends on the formulation, element interpolation, degradation function, and convergence criterion.

Therefore a sensible size field is not simply

$$
h=h(\text{sphere curvature}).
$$

It should resemble

$$
h(\mathbf x)
============

\min
\left[
h_{\rm bulk},
h_{\rm interface}(\mathbf x),
h_{\ell},
h_{\rm ligament}(\mathbf x)
\right].
$$

For your application, **particle-particle and particle-boundary ligaments are plausible crack-initiation/localization regions**, so those regions deserve special treatment.

This is one reason I would avoid measuring success solely by "number of elements needed to make the sphere look round."

---

# 11. K-dominance consideration

Your concern about the (K)-dominant region is important.

Near a crack tip in LEFM, the asymptotic field is conventionally represented by

$$
\sigma_{ij}(r,\theta)
=====================

\frac{K}{\sqrt{2\pi r}}
f_{ij}(\theta)
+\text{higher-order terms}.
$$

In elastoplastic fracture and phase-field fracture, the relevant near-tip/process-zone scales become more complicated, but the numerical principle remains: **do not allow mesh-transition artifacts, badly distorted hexes, or particle-interface discretization errors to contaminate the region from which fracture behavior is inferred.**

Thus I would impose two independent constraints:

1. **geometry constraint:** accurately represent particles;
2. **fracture constraint:** maintain controlled size, aspect ratio and Jacobian around likely crack paths/process zones.

The second can be stricter than the first.

---

# 12. Practical software ranking

For the exact problem you described, I would investigate the toolchain in approximately this order:

| Tool / approach                           |                  All-hex potential |                Curved CAD |        Python automation |                  Exodus route | Suitability                                    |
| ----------------------------------------- | ---------------------------------: | ------------------------: | -----------------------: | ----------------------------: | ---------------------------------------------- |
| **CUBIT / Coreform Cubit + Python**       |                          Excellent |                 Excellent |                Excellent |             **Native/strong** | **Best first choice**                          |
| Custom cubed-sphere + CUBIT decomposition |                          Excellent |                 Excellent |                Excellent |                    **Native** | **Best research prototype**                    |
| Custom octree/dual + geometry projection  | Excellent if implemented carefully |                      Good |                Excellent |          Via writer/converter | Strong long-term option                        |
| Polycube implementation                   |                          Excellent |   Excellent after mapping |                 Possible |                 Custom export | Research-intensive                             |
| Cuboid-decomposition implementation       |                          Excellent |  Excellent after pullback |                 Possible |                 Custom export | Advanced research option                       |
| Gmsh transfinite/recombine                |       Good on decomposable volumes |      Excellent CAD kernel | **Excellent Python API** |             Conversion needed | Useful, but not my first strict-all-hex choice |
| Abaqus scripting                          |       Geometry/RVE automation good |                      Good |                     Good | Not naturally Exodus-centered | Secondary                                      |
| Voxel/octree without boundary fitting     |                 Excellent topology | Poor at coarse resolution |                     Easy |                          Easy | Baseline only                                  |

The literature on modern cuboid decomposition explicitly characterizes **CUBIT as the industry-standard interactive hex-meshing tool**, operating directly on CAD geometry with sweeping operations ([Li et al., 2021](https://scispace.com/papers/interactive-all-hex-meshing-via-cuboid-decomposition-3lsdvzqc6b?utm_source=chatgpt)).

---

# 13. Recommended CUBIT/Python architecture

I would have the coding agent build the project as a pipeline rather than a monolithic script.

```text
microstructure_generator.py
    |
    +-- generate particle centers/radii
    +-- collision checking
    +-- periodicity if required
    |
geometry.py
    |
    +-- create cube
    +-- create exact spheres
    +-- Boolean/imprint operations
    |
particle_topology.py
    |
    +-- cubed-sphere decomposition
    +-- O-grid shell
    +-- radial sizing
    |
matrix_decomposition.py
    |
    +-- proximity graph
    +-- particle-particle partitions
    +-- particle-wall partitions
    +-- sweepability tests
    |
mesh.py
    |
    +-- interval assignment
    +-- surface schemes
    +-- sweep/map volumes
    +-- smoothing
    |
quality.py
    |
    +-- Jacobian/scaled Jacobian
    +-- aspect ratio
    +-- skew
    +-- interface conformity
    +-- geometric sphere error
    |
blocks.py
    |
    +-- matrix block
    +-- particle blocks
    +-- interface sidesets
    +-- external BC sidesets
    |
export.py
    |
    +-- Exodus II
```

That modularity is especially important because random geometry means the algorithm must detect failure and retry.

---

# 14. Random-particle generation is itself a solved subproblem

Do not couple packing and meshing unnecessarily.

A recent particulate-composite study uses Monte Carlo together with collision-driven molecular dynamics to create randomized spherical RVEs and reports particle/matrix volume ratios up to approximately 50% while maintaining periodicity. It also provides an automated Abaqus RVE generator ([Elbana et al., 2024](https://scispace.com/papers/multiscale-modelling-of-particulate-composites-with-4g7y7q2za8?utm_source=chatgpt)).

Thus split the problem:

$$
\boxed{\text{random packing}}
\quad\rightarrow\quad
\boxed{\text{geometry}}
\quad\rightarrow\quad
\boxed{\text{hex topology}}
\quad\rightarrow\quad
\boxed{\text{quality optimization}}.
$$

The packing module should output only

```python
particles = [
    {"center": (x1, y1, z1), "radius": r1},
    {"center": (x2, y2, z2), "radius": r2},
    ...
]
```

The mesher should not care how that list was generated.

---

# 15. Sphere representation: use a cubed sphere, not latitude/longitude

This deserves special emphasis.

A conventional spherical parameterization has coordinate singularities at the poles. That is undesirable for an all-quad surface mesh and consequently for swept hexes.

Instead start with the six faces of a cube and map each to the sphere:

$$
\mathbf x_s =
\mathbf c+
R\frac{\mathbf q}{|\mathbf q|},
$$

where (\mathbf q) lies on a cube face.

This produces six structured quadrilateral patches with no polar singularity.

For a radial layer at (r_k),

$$
\mathbf x_k =
\mathbf c+
r_k\frac{\mathbf q}{|\mathbf q|}.
$$

Corresponding quads at (r_k) and (r_{k+1}) form hexahedra directly.

This is simple enough that an AI coding agent could implement and test it independently before dealing with arbitrary particle arrangements.

---

# 16. Exactness: HEX8 versus higher-order hexes

There is an important geometrical limitation in your phrase "perfectly spherical."

A first-order hexahedron has planar/bilinear faces. Even if every boundary node satisfies

$$
|\mathbf x-\mathbf c|=R,
$$

the element face between those nodes is not the analytical spherical surface.

Define radial geometry error as

$$
e_g =
\max_{\mathbf x\in\Gamma_h}
\left|
|\mathbf x-\mathbf c|-R
\right|.
$$

For linear faces, (e_g) decreases as the surface is refined.

If your solver supports second-order elements, placing midside/face nodes on the sphere can drastically reduce geometric error without the same increase in element count.

So another research question for the downstream AI agent should be:

> **Does the elastoplastic phase-field implementation accept HEX20/HEX27, and are curved isoparametric interfaces correctly used in quadrature and constitutive calculations?**

If yes, higher-order geometry may solve much of the "perfect sphere versus element count" problem more efficiently than topology alone.

---

# 17. Handling the matrix between particle shells

This is the hardest implementation component.

I see three realistic strategies.

### Strategy 1 — Automatic CAD decomposition + sweep/map

Create sphere-centered shells, imprint their outer surfaces into the matrix, then partition the matrix into sweepable volumes.

**Advantages:** CAD-exact; directly compatible with CUBIT; easiest path to Exodus.

**Disadvantage:** robust automatic decomposition for arbitrary close particle configurations is difficult.

### Strategy 2 — Adaptive octree background + boundary fitting

Build an adaptive Cartesian/octree matrix mesh, refine near (S_i^+), replace intersected cells with conformal hex templates, project boundary nodes to the shell, and optimize.

**Advantages:** naturally handles arbitrary (N); excellent automation; localized refinement.

**Disadvantage:** transition-template implementation is substantial, and boundary elements require aggressive quality checks.

### Strategy 3 — Parameterization/polyclube

Tet-mesh the matrix temporarily, compute a volumetric frame field/polyclube parameterization, extract a hex grid in parameter space, map back.

**Advantages:** mathematically elegant; handles arbitrary geometry.

**Disadvantage:** by far the largest research-code investment.

For your immediate project I would use **Strategy 1**, while keeping Strategy 2 as the longer-term custom mesher.

---

# 18. Do not reject octrees because you reject voxels

This is perhaps the biggest conceptual distinction emerging from the literature.

There are three different things:

$$
\text{voxel topology},
\qquad
\text{adaptive Cartesian topology},
\qquad
\text{voxelated physical boundary}.
$$

They are not equivalent.

You can use an octree/grid to determine **connectivity** and subsequently move/project boundary nodes so that

$$
\mathbf x_b
\rightarrow
\mathbf c_i+
R_i
\frac{\mathbf x_b-\mathbf c_i}
{|\mathbf x_b-\mathbf c_i|}.
$$

Then optimize interior nodes subject to the fixed curved boundary.

Modern polycube/cuboid methods similarly use grid-like parameter domains but map them back onto curved geometry ([Li et al., 2021](https://scispace.com/papers/interactive-all-hex-meshing-via-cuboid-decomposition-3lsdvzqc6b?utm_source=chatgpt); [Hu & Zhang, 2016](https://scispace.com/papers/centroidal-voronoi-tessellation-based-polycube-construction-1500d9cj6w?utm_source=chatgpt)).

So **"non-voxelated" should be a requirement on physical geometry, not necessarily on computational topology.**

---

# 19. Mesh optimization should be treated as mandatory

Generating connectivity is only half of an all-hex algorithm.

Let the element mapping be

$$
\mathbf x(\xi,\eta,\zeta)
=========================

\sum_{a=1}^{8}
N_a(\xi,\eta,\zeta)\mathbf x_a.
$$

Then

$$
J=
\frac{\partial\mathbf x}
{\partial(\xi,\eta,\zeta)}
$$

must satisfy

$$
\det J>0
$$

throughout every element.

A practical acceptance pipeline should evaluate at least:

* minimum Jacobian;
* scaled Jacobian;
* aspect ratio;
* skew;
* warpage;
* edge-length ratio;
* sphere/interface geometric error;
* element-size gradient;
* number of elements across narrow ligaments.

Whisker Weaving research explicitly illustrates why topology alone is insufficient: complex meshes can contain isolated poor-quality or negative-Jacobian regions even after a valid connectivity construction ([Folwell & Mitchell, 1999](https://scispace.com/papers/reliable-whisker-weaving-via-curve-contraction-1m7fm2r3ci?utm_source=chatgpt)).

---

# 20. A robust automated algorithm should have failure/retry logic

Randomized microstructures mean some realizations will be pathological.

The generator should therefore work like:

```text
generate random realization
        ↓
compute gaps / neighborhood graph
        ↓
reject impossible geometry?
        ↓
construct particle shell topology
        ↓
construct matrix decomposition
        ↓
mesh
        ↓
quality evaluation
        ↓
acceptable? ── yes → Exodus
     │
     no
     ↓
local refinement / shell shrink / repartition
        ↓
remesh
```

This is much more realistic than assuming one meshing command will work for every random seed.

---

# 21. Geometry admissibility should be explicit

Before meshing, enforce something like

$$
|\mathbf c_i-\mathbf c_j|

>

R_i+R_j+g_{\min}
$$

and

$$
d(\mathbf c_i,\partial\Omega_{\rm cube})

>

R_i+g_{\rm wall}.
$$

The values (g_{\min}) and (g_{\rm wall}) should not be arbitrary geometry tolerances. They should reflect how many elements are required in the ligament:

$$
g_{\min}\gtrsim n_g h_{\rm target}.
$$

Otherwise you can generate geometrically valid particle configurations that are essentially impossible to mesh with acceptable HEX8 quality.

---

# 22. Exodus data model

The final mesh should probably contain separate Exodus element blocks such as

```text
block_1 : matrix
block_2 : particle material   # if all particles same material
```

or

```text
block_1001 : particle_001
block_1002 : particle_002
...
```

if particles require separate tracking.

Useful sidesets include:

```text
X_MINUS
X_PLUS
Y_MINUS
Y_PLUS
Z_MINUS
Z_PLUS
PARTICLE_MATRIX_INTERFACE
```

and potentially particle-specific interface sets.

Node sets may also be useful for:

* corner constraints;
* periodic boundary correspondence;
* crack initialization;
* measurement/probe regions.

This should be designed **before** export, not reconstructed from coordinates afterward.

---

# 23. Recommended software workflow

For an immediately executable research project, I would specify:

```text
Python
  │
  ├── NumPy/SciPy
  │      random packing
  │      nearest-neighbor search
  │      gap calculations
  │      graph construction
  │
  ├── CUBIT Python API
  │      exact cube/sphere CAD
  │      Boolean/imprint
  │      decomposition
  │      mesh schemes
  │      smoothing
  │      blocks/sidesets
  │
  ├── mesh-quality validation
  │
  └── Exodus II
```

For very large (N), use a KD-tree to avoid an (O(N^2)) full particle-pair search.

---

# 24. Where Gmsh fits

Gmsh is still worth investigating because its Python API is excellent and its OpenCASCADE geometry kernel makes random CAD generation easy.

It is especially useful for:

* sphere/cube Boolean geometry;
* distance-based size fields;
* transfinite structured subdomains;
* recombination experiments;
* prototype geometry;
* temporary tetrahedral meshes for parameterization algorithms.

But there is an important distinction between **recombining a tetrahedral mesh** and obtaining a guaranteed high-quality **all-hexahedral conformal mesh**. For the latter, CUBIT's decomposition/sweep ecosystem is a stronger starting point.

I would therefore use Gmsh primarily as a research/prototyping tool unless testing demonstrates that your restricted sphere geometry can be reliably decomposed into transfinite volumes.

---

# 25. Evidence synthesis

| No. | Source                                                                                                                                                                                                                                              | Algorithm / relevance                             | Key insight for your problem                                                                                                                    |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Unconstrained plastering—Hexahedral mesh generation via advancing-front geometry decomposition (Staten et al., 2010)](https://scispace.com/papers/unconstrained-plastering-hexahedral-mesh-generation-via-21a5v6p6r3?utm_source=chatgpt)           | Advancing-front geometry decomposition            | Demonstrates boundary-sensitive, orientation-insensitive all-hex generation but also why general automatic all-hex meshing remains difficult.   |
| 2   | [Unconstrained Paving & Plastering (Staten, Owen & Blacker, 2005)](https://scispace.com/papers/unconstrained-paving-plastering-a-new-idea-for-all-4piyn45wbk?utm_source=chatgpt)                                                                    | Early unconstrained plastering                    | Avoiding a completely prescribed boundary mesh can prevent the interior hex problem from becoming over-constrained.                             |
| 3   | [Reliable Whisker Weaving via Curve Contraction (Folwell & Mitchell, 1999)](https://scispace.com/papers/reliable-whisker-weaving-via-curve-contraction-1m7fm2r3ci?utm_source=chatgpt)                                                               | Dual/STC topology                                 | Shows that valid hex connectivity and good geometric quality are distinct problems; smoothing/topological correction remains essential.         |
| 4   | [Centroidal Voronoi tessellation based polycube construction for adaptive all-hexahedral mesh generation (Hu & Zhang, 2016)](https://scispace.com/papers/centroidal-voronoi-tessellation-based-polycube-construction-1500d9cj6w?utm_source=chatgpt) | Polycube + adaptive hybrid octree                 | Particularly relevant: combines grid robustness, adaptive all-hex generation and improved conformal boundary quality.                           |
| 5   | [All-hex meshing using closed-form induced polycube (Fang et al., 2016)](https://scispace.com/papers/all-hex-meshing-using-closed-form-induced-polycube-1kg6d4ys3t?utm_source=chatgpt)                                                              | Frame-field-guided polycube                       | Supports complex topology and disconnected boundaries while reducing distortion/singularities; useful conceptual basis for multiple inclusions. |
| 6   | [Interactive All-Hex Meshing via Cuboid Decomposition (Li et al., 2021)](https://scispace.com/papers/interactive-all-hex-meshing-via-cuboid-decomposition-3lsdvzqc6b?utm_source=chatgpt)                                                            | Cuboid decomposition + inversion-free pullback    | Modern robust direction: grid/polyclube topology is mapped back to the curved physical domain rather than leaving a voxelized surface.          |
| 7   | [Multiscale modelling of particulate composites with spherical inclusions (Elbana et al., 2024)](https://scispace.com/papers/multiscale-modelling-of-particulate-composites-with-4g7y7q2za8?utm_source=chatgpt)                                     | Random spherical RVE generation                   | Demonstrates automated random sphere packing and RVE generation; useful for separating the packing problem from the meshing problem.            |
| 8   | [Finite element modelling of complex 3D image data with quantification and analysis (Chakkour, 2024)](https://scispace.com/papers/finite-element-modelling-of-complex-3d-image-data-with-4k38t0gq9d?utm_source=chatgpt)                             | Conforming versus voxelized microstructure meshes | Reinforces the computational advantage of conforming interface representations over direct voxelization for heterogeneous microstructures.      |

---

# 26. Concrete specification I would give the coding AI

The first prototype should **not** attempt arbitrary particle counts immediately.

### Milestone 1 — One sphere

Generate:

$$
[-L/2,L/2]^3
$$

with one off-center sphere—not centered.

Requirements:

* cubed-sphere quadrilateral topology;
* radial all-hex shell;
* mapped/swept outer matrix;
* HEX8;
* Exodus output;
* positive Jacobian everywhere.

The sphere should deliberately be **off-center**, because a centered sphere hides decomposition assumptions.

### Milestone 2 — Two unequal spheres

Random positions and different radii.

This forces development of:

* neighbor detection;
* variable shell thickness;
* matrix transition topology;
* narrow-ligament refinement.

### Milestone 3 — 5–20 random spheres

Introduce the particle proximity graph and automated decomposition.

### Milestone 4 — Monte Carlo campaign

Run hundreds of seeds and record:

$$
P_{\rm success}
===============

\frac{N_{\rm valid\ meshes}}
{N_{\rm realizations}}.
$$

A mesh algorithm for stochastic fracture simulations should be judged by **success rate across random realizations**, not by one attractive mesh screenshot.

---

# 27. Validation metrics for the research code

For every generated realization, save a JSON/CSV summary alongside Exodus:

```text
random_seed
particle_count
particle_volume_fraction
minimum_particle_gap
minimum_wall_gap
element_count
node_count
minimum_scaled_jacobian
mean_scaled_jacobian
maximum_aspect_ratio
maximum_sphere_geometry_error
minimum_elements_per_ligament
meshing_time
retry_count
success/failure
```

Then perform convergence studies against:

1. sphere geometry error;
2. phase-field response;
3. crack trajectory;
4. load-displacement curve;
5. dissipated fracture energy;
6. plastic dissipation.

That tells you whether the sophisticated curved hex mesh actually produces a meaningful improvement over the voxel baseline.

---

# 28. A potentially better alternative to demanding globally uniform-quality all-hex topology

If your fracture solver absolutely requires hexes but **does not require them to be globally structured**, concentrate mesh sophistication where physics requires it:

$$
\boxed{\text{high-quality structured curved hexes near particles}}
+
\boxed{\text{unstructured/adaptive all-hex matrix}}
$$

rather than demanding one elegant global block structure.

The particle neighborhood can be highly regular while singular hex topology is pushed into comparatively benign bulk regions away from particle interfaces and expected fracture localization.

This is analogous to deliberately controlling where irregular vertices occur in quadrilateral meshing.

---

# 29. Recommended research direction

I would divide the project into two tracks.

**Track A — production solution:** Python + CUBIT, analytical sphere CAD, cubed-sphere/O-grid particle blocks, automatic geometry decomposition, sweep/map meshing, local refinement, mesh-quality retry logic, and native Exodus export.

**Track B — research mesher:** adaptive octree background, sphere-aware refinement, conformal boundary reconstruction/projection, dual all-hex extraction, optimization under analytical spherical constraints, and eventually local sheet/topology operations.

Track A is much more likely to get the phase-field simulations running. Track B could become a publishable meshing contribution if you make it reliably handle stochastic particulate microstructures.

---

## 🧾 Conclusion

For a random number of randomly positioned spherical particles, **there is currently no universally reliable push-button algorithm that guarantees a low-count, high-quality, curved, conformal all-hexahedral mesh for every realization**. The literature progresses from plastering and Whisker Weaving toward octree/dual, frame-field, polycube, and modern cuboid-decomposition approaches, but topology and element quality remain coupled challenges.

For your restricted geometry, the most promising practical solution is **not conventional voxelization**. I recommend a **six-patch cubed-sphere + radial O-grid around every analytical particle, coupled to automated matrix decomposition/adaptive all-hex meshing**, with CUBIT controlled from Python and Exodus as the native output. If CUBIT cannot robustly close the random matrix regions, the next algorithm to implement should be a **sphere-aware adaptive octree/dual-hex method with boundary projection and constrained quality optimization**, rather than simply increasing voxel resolution.
