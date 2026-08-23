## Collective conclusion

For a cube containing an arbitrary number of randomly positioned spherical inclusions, there is still **no universally robust, fully automatic algorithm that takes arbitrary CAD and always returns a high-quality conforming all-hexahedral mesh**. The 2022 state-of-the-art survey still identifies automatic general-purpose all-hex meshing as an open problem, and the 2026 literature is still investigating AI-assisted block decomposition, polycubes, and parameterization rather than reporting a solved production-grade problem [S1, S10].

For your particular geometry, however, the problem is much more tractable than arbitrary CAD because every internal surface is an analytic sphere. I would not attack it as a general-purpose all-hex problem. I would exploit that structure.

The most promising routes are:

| Priority                                 | Approach                                                                                                                | Assessment for random spherical particles                                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **1 — production route**                 | **Cubit/Sculpt-style overlay-grid + interface reconstruction + constrained sphere projection + pillowing/optimization** | Best combination of automation, arbitrary particle count, all-hex topology, Python scripting, and native Exodus output    |
| **2 — best custom Python algorithm**     | **Particle-local cubed-sphere/O-grid templates inserted into a Cartesian/octree background**                            | Particularly attractive for your exact problem; avoids solving general all-hex meshing                                    |
| **3 — research-grade general algorithm** | **Adaptive octree/isocontouring such as HybridOctree_Hex, followed by analytic surface projection and optimization**    | Very promising if another AI is going to implement a mesher from papers                                                   |
| 4                                        | Polycube / cuboid decomposition / frame-field parameterization                                                          | Elegant and potentially low-element-count, but difficult with many internal closed spherical boundaries                   |
| 5                                        | Plastering / whisker weaving / H-Morph                                                                                  | Historically important, but robust closure for arbitrary topology remains difficult                                       |
| Alternative formulation                  | Immersed/cut-cell/finite-cell/shifted-boundary methods                                                                  | Can preserve the sphere analytically without body-fitted meshing, but changes the FE formulation and probably your solver |

There is also an important issue hidden in the phrase **“perfectly spherical.”**

### A HEX8 mesh cannot contain a mathematically exact spherical boundary

A standard linear hexahedron has planar faces. Therefore, even if every surface vertex satisfies

$$
|\mathbf{x}_i-\mathbf{c}|=R,
$$

the quadrilateral between four vertices is planar and lies partly inside the true sphere.

So there are three meanings of “perfect sphere”:

| Desired property                                                      | Possible with coarse HEX8?                                                                                       |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Every interface node lies exactly on radius (R)                       | **Yes**                                                                                                          |
| Very small geometric error with relatively few elements               | **Yes**, especially with good quad topology                                                                      |
| The finite-element boundary itself is mathematically the exact sphere | **No** for polynomial HEX8; generally no finite polynomial Lagrange representation is exactly rational-spherical |

HEX20/HEX27 curved elements can approximate the sphere dramatically better per element. Exact CAD geometry requires something such as NURBS/isogeometric geometry, an exact-geometry FE treatment, or an immersed implicit surface [S17].

For a conventional Exodus/HEX8 fracture code, I would define the requirement as:

> **Analytic sphere input, interface vertices projected exactly onto that sphere, with controlled Hausdorff/volume error and no voxel staircase.**

That is achievable.

---

# 1. Algorithm families that matter

### Mapping, submapping and sweeping

These are why your “one particle exactly in the middle” case is easy. The domain is manually decomposed into topological blocks, after which every block can be mapped or swept.

The difficulty with random particles is not generating hexes inside one sphere. It is finding a consistent global block decomposition of the matrix surrounding (N) independently located closed surfaces.

For a small known number of particles, manual or scripted decomposition can work extremely well. For arbitrary (N), decomposition becomes the bottleneck.

**Verdict:** excellent quality, poor general automation.

---

### Plastering and whisker weaving

Plastering grows layers of hexahedra inward from a quadrilateral boundary. Whisker weaving works in the dual Spatial Twist Continuum and constructs hex connectivity from a closed quad surface [S2].

These methods were important because they attempted true **unstructured all-hexahedral meshing** rather than mapping a pre-existing block decomposition.

Their fundamental difficulty is front closure. As advancing fronts collide, difficult topological configurations such as self-intersections, wedges, blind chords, and incompatible sheets appear.

Whisker weaving's original paper explicitly reported success on relatively simple geometries while noting limitations in complexity and mesh size [S2].

**Verdict:** foundational research, but I would not build your production workflow around a new plastering implementation.

---

### Tetrahedral-to-hexahedral conversion

One can first generate a robust tetrahedral mesh and subdivide each tetrahedron into hexahedra through edge midpoints, face centers and a tetrahedron center.

This can guarantee hexahedral topology, but typically creates:

* many more elements,
* awkward singularity structure,
* mediocre element quality,
* a surface inherited from the tetrahedral approximation,
* and substantially more DOFs than the structured alternatives.

It largely defeats your objective of achieving spherical fidelity with fewer elements.

**Verdict:** useful fallback for topology, poor choice for your goal.

---

### Overlay-grid / Sculpt-type methods

This family is extremely relevant.

Start with a regular Cartesian hexahedral grid and classify material occupancy relative to the geometry. Boundary cells are modified, interface nodes are repositioned, and topology/quality operations such as smoothing and pillowing are applied.

Sandia's **Sculpt** is the best-known production implementation. Owen and Shelton evaluated Sculpt meshes in both linear and nonlinear elastic-plastic solid-mechanics calculations and found grid-based meshes capable of results equivalent or superior to traditionally generated pave-and-sweep meshes in the cases tested [S3].

Later work explicitly addresses relocating Sculpt/interface-reconstruction nodes to improve agreement with the reference geometry while retaining acceptable hex quality [S4].

This is particularly well suited to your problem because particle count and placement do not fundamentally change the algorithm. Twenty random particles are conceptually almost as straightforward as five.

The limitation is geometric resolution. A raw overlay-grid interface is still coupled to background resolution. The improvement for your case is to exploit the fact that the exact interface equation is known:

$$
\phi_i(\mathbf{x})
==================

|\mathbf{x}-\mathbf{c}_i|-R_i=0.
$$

After the correct interface topology has been created, sphere-interface vertices can be projected analytically:

$$
\mathbf{x}' =
\mathbf{c}_i+
R_i
\frac{\mathbf{x}-\mathbf{c}_i}
{|\mathbf{x}-\mathbf{c}_i|}.
$$

Then nearby interior nodes are optimized while the spherical nodes remain constrained.

**Verdict:** probably the fastest path to a robust usable Exodus mesh.

---

### Octree-based all-hex meshing and isocontouring

This is probably the most interesting family if another AI will write the mesher itself.

Zhang and Bajaj developed adaptive quadrilateral/hexahedral meshing from volumetric data using octree/isocontouring ideas [S5]. Qian and Zhang subsequently extended octree-based methods to B-Rep CAD assemblies with feature preservation, adaptive refinement, pillowing, smoothing and optimization [S6].

The 2024 **HybridOctree_Hex** work is especially relevant. It combines adaptive octree concepts with all-hexahedral transition mechanisms and explicit Jacobian control [S7].

For your problem, an octree could refine only:

* near particle surfaces,
* in thin particle-particle ligaments,
* around a prescribed initial crack/notch,
* and where (h) must resolve the phase-field length scale.

Large regions far from interfaces remain coarse.

This is much more efficient than globally refining a voxel representation until every sphere appears smooth.

**Verdict:** strongest open research direction for a custom general solution.

---

### Polycubes, cuboid decomposition and frame fields

Polycube methods map a complicated solid to an orthogonal block structure and pull a structured hex mesh back to the physical geometry. Gregson, Sheffer and Zhang demonstrated volumetric polycube deformation for all-hex generation [S8].

More recent work includes cut-enhanced polycube mappings, safer embedded cell-complex operations, volume parameterization quantization, and automatic/learning-assisted polycube construction [S9].

These methods can produce excellent coarse meshes.

But your matrix is a cube containing (N) internal closed surfaces. Each sphere adds nontrivial topology and corresponding singularity structure to the volumetric parameterization. Automatically obtaining a low-distortion valid map for arbitrary random arrangements remains difficult.

The 2026 literature includes diffusion-model polycube generation and AI/graph-network-assisted geometry decomposition. These are interesting, but I would regard them as **research accelerators rather than mature production algorithms** [S10].

**Verdict:** academically compelling, not my first implementation target for your application.

---

# 2. A tailored algorithm that may outperform a general mesher

Your geometry allows an alternative that is considerably simpler than arbitrary all-hex meshing:

## **Particle-local cubed-sphere macroblocks**

Give every particle its own local hexahedral template.

For each sphere, conceptually surround it with a small rectangular macrobox that conforms to the background Cartesian/octree mesh.

The sphere is divided into six quadrilateral patches corresponding to the six faces of a cube. On the (+x) patch, for example,

$$
\mathbf{s}(u,v)
===============

\mathbf{c}
+
R
\frac{(1,u,v)}
{\sqrt{1+u^2+v^2}},
\qquad -1\le u,v\le1.
$$

Analogous maps define the other five faces.

This gives a **cubed-sphere quadrangulation**: all nodes lie exactly on the analytic sphere and there are no polar singularities like those created by longitude-latitude meshes.

Between each spherical patch and the corresponding face of an enclosing box, construct a structured swept block. Each one is topologically a hexahedral volume.

Inside the spherical particle, use a bijective cube-to-ball mapping to transform a structured cube mesh into a spherical one.

The result is approximately:

```text
Cartesian matrix
        |
   outer macrobox
   +-----------+
   | \       / |
   |  \     /  |
   |   sphere  |
   |  /     \  |
   | /       \ |
   +-----------+
        |
six structured sphere-to-box transition blocks
```

The key idea is that the background never has to conform directly to the sphere. It only has to conform to the outer rectangular macrobox.

That converts an arbitrary curved-interface problem into a collection of structured block problems.

### Why this is attractive

The number of particles may change without changing the fundamental topology generator.

A sphere can have perhaps tens or hundreds of boundary quads instead of the thousands of voxel faces that would otherwise be needed for similar curvature resolution.

It is naturally scriptable.

It can be generated using Python plus Gmsh's transfinite/sweep capabilities, or with Cubit.

It gives direct control of radial layers near the inclusion-matrix interface, which is useful for high plastic strain/stress gradients.

### Primary difficulty

Two particle macroboxes must not overlap.

For moderate particle volume fractions this is manageable. For close particles, the algorithm should either:

* refine the background and shrink the local box;
* generate a two-particle compound macroblock;
* or fall back to the octree/Sculpt algorithm for that local region.

This tailored route is the one I would investigate most seriously if the objective is eventually to have a **fully Python-controlled meshing code**.

---

# 3. Software/tool assessment

| Tool                                      |                         Python control |                                                 General arbitrary all-hex? |                                      Exact sphere CAD |                           Exodus | Suitability                                                      |
| ----------------------------------------- | -------------------------------------: | -------------------------------------------------------------------------: | ----------------------------------------------------: | -------------------------------: | ---------------------------------------------------------------- |
| **Coreform Cubit / CUBIT + Sculpt**       |                          **Excellent** |                          Sculpt is highly automated grid-based hex meshing | Yes for geometry; final mesh remains FE approximation |                       **Native** | **Best overall production choice**                               |
| **Gmsh + OpenCASCADE kernel**             |                          **Excellent** |                              No general arbitrary-volume all-hex algorithm |                                               **Yes** |              Conversion required | **Best open tool for custom block/template approach**            |
| **pythonOCC / OpenCASCADE**               |                              Excellent |                                                              Geometry only |                                               **Yes** |                               No | Useful for CAD construction/Boolean operations                   |
| **SALOME/SMESH**                          |                        Good Python API | Strong for decomposed/sweepable volumes; arbitrary all-hex still difficult |                                                   Yes |      Usually conversion workflow | Secondary option                                                 |
| **MeshGems / Hexotic family**             | Mostly commercial/proprietary workflow |                                       Strong commercial meshing capability |                                                   Yes |           Depends on integration | Worth evaluating if licensing is acceptable                      |
| **ANSYS ICEM/blocking tools**             |     Scriptable, but less Python-native |                                    Powerful manual/semi-automatic blocking |                                                   Yes |      Solver-dependent conversion | Good engineering tool, poorer fit for autonomous Python research |
| **meshio**                                |                              Excellent |                                                               Not a mesher |                                                   N/A |      Can handle Exodus workflows | Good converter, but see warning below                            |
| **SEACAS / Exodus Python or netCDF APIs** |                              Excellent |                                                               Not a mesher |                                                   N/A | **Native format implementation** | Preferred low-level exporter                                     |

Gmsh itself is a mature open-source meshing/CAD environment with programmatic control [S14]. Its weakness for your problem is not geometry generation. Creating `occ.addSphere()` objects and Boolean-fragmenting them with a cube is straightforward. Its weakness is that **recombine/transfinite operations do not magically turn arbitrary multiply-connected 3-D geometry into a valid all-hex mesh**.

So I would use Gmsh as the meshing engine only if you supply your own topology—such as the local cubed-sphere macroblocks.

---

# 4. Cubit/Sculpt workflow I would prototype first

For a proof of concept, this is the lowest-risk route.

The complete workflow for an AI coding agent should look approximately like this:

1. **Generate the particle realization.** Store every particle as `(id, cx, cy, cz, radius)`. Use random sequential addition or Poisson-disk-like placement for low/moderate fractions. Maintain a prescribed minimum surface-to-surface ligament. For high fractions, move to collective rearrangement/optimization rather than simple rejection sampling. Richter's Mote3D and Schneider et al. are useful references for random periodic particulate microstructures and Python-driven RVE generation [S12, S13].

2. **Keep the geometry analytic.** Do not voxelize the particle locations. Represent each particle using its center and radius or an exact CAD sphere.

3. **Build a background hex grid or adaptive octree.** Let the bulk size be (h_b), interface size (h_s), and fracture-region size (h_f). The local target should conceptually satisfy
   $$
   h(\mathbf{x})=\min(h_b,h_s,h_f,h_{\rm gap}).
   $$

4. **Classify material occupancy analytically.** For every background cell, sphere membership is determined from the implicit functions (\phi_i(\mathbf{x})), rather than a raster image.

5. **Create conforming material-interface topology.** Use Sculpt/interface reconstruction or an octree all-hex method. Assign matrix and inclusion elements to distinct element blocks.

6. **Project every particle-interface vertex onto its exact sphere.** Associate each interface vertex with particle (i), then apply
   $$
   \mathbf{x}'=\mathbf{c}_i+R_i(\mathbf{x}-\mathbf{c}_i)/|\mathbf{x}-\mathbf{c}_i|.
   $$

7. **Optimize nearby hexes while locking interface vertices to the analytic surface.** Use barrier-based Jacobian/condition-number optimization rather than unconstrained Laplacian smoothing. Knupp's work remains foundational for Jacobian-based volume-mesh optimization [S11].

8. **Insert one or more pillow layers at matrix-particle interfaces when necessary.** This generally improves element orientation and prevents poorly shaped elements from terminating directly on a material interface. Sculpt literature specifically discusses pillowing and optimization for grid-based solid-mechanics meshes [S3].

9. **Refine particle-particle ligaments adaptively.** A pair of spheres separated by a small gap should not force global mesh refinement. Refine only their local octree cells or macroblocks. If the physical ligament is thinner than the desired numerical resolution, reject that particle realization or flag it for special treatment.

10. **Create Exodus metadata explicitly.** Matrix and particles should become element blocks. The six exterior cube boundaries should become sidesets/nodesets. Matrix-particle interfaces may be stored as sidesets if your fracture/contact/material formulation needs them. If individual particle identity matters, also preserve a particle-ID element field or separate blocks.

11. **Write Exodus and perform a round-trip validation.** Verify node numbering, HEX8/HEX20/HEX27 connectivity, blocks, side orientation, nodesets, material IDs, and positive element Jacobians after re-reading the file.

12. **Run geometry and mechanics regression tests.** Compare particle volume, surface location, minimum Jacobian, stress concentration around a single sphere, and a phase-field benchmark before moving to hundreds of random realizations.

---

# 5. Exodus output: what I would use

If Cubit generates the mesh, let **Cubit write Exodus directly**. Walters et al. describe a microstructure-processing workflow in which Sculpt produces Exodus II meshes and Cubit's Python interface is used for downstream processing [S15].

For a standalone Python implementation, I would not make `meshio` the only Exodus backend.

There is published experience where `meshio`'s ExodusII output did not produce the required working file and the authors instead implemented output using Python `netCDF` [S16].

That does not mean meshio is unusable. It means that for a research code where **element blocks, sidesets, nodesets and solver compatibility are essential**, Exodus should be treated as a first-class data model rather than merely a file conversion target.

Recommended hierarchy:

```text
Cubit mesh
    -> Cubit native Exodus export

Custom Python mesh
    -> SEACAS/Exodus Python API or direct Exodus-compatible netCDF writer
    -> optional meshio conversion for diagnostics/interchange
```

Internally, the code should maintain something like:

```python
points              # (N, 3)
hex_connectivity    # (M, 8), or HEX20/HEX27
element_block_id    # matrix / particle material
particle_id         # optional per-element identity
nodesets
sidesets
interface_faces
sphere_metadata
```

The Exodus writer should be the final thin layer, not where topology is reconstructed.

---

# 6. Phase-field fracture changes the mesh requirement

This part is important because simply making the sphere smoother may not reduce your final model size as much as expected.

In phase-field fracture, the regularized crack has a physical/numerical width controlled by the phase-field length parameter (\ell).

Finite-element studies commonly require several elements over that regularization zone; (h\leq \ell/2) is a frequently used starting point in the literature [S18]. It should not be treated as an immutable rule—your elastoplastic formulation needs a convergence study—but it gives an order of magnitude.

Thus your true mesh-size function should look more like

$$
h
\leq
\min
\left(
C_R R,
C_\ell\ell,
C_g g,
h_{\text{bulk}}
\right),
$$

where (R) is local particle curvature radius and (g) is the nearest particle-particle ligament.

If

$$
\ell \ll R,
$$

phase-field resolution rather than sphere geometry may dominate your element count.

This is precisely why eliminating voxel staircasing is worthwhile: it lets **fracture physics**, rather than artificial geometric approximation, determine (h).

---

# 7. K-dominance requires a separate check

Mesh quality by itself cannot guarantee a K-dominant region.

For classical LEFM, K-dominance requires an annular region around the crack tip in which higher-order geometric terms, plasticity/process-zone effects and boundary effects are small.

In an elastoplastic phase-field model you introduce at least two additional characteristic zones:

$$
\text{phase-field zone}\sim O(\ell)
$$

and a plastic/process zone governed by yielding.

Random particle interfaces introduce another heterogeneity length.

Consequently, a physically meaningful K-dominant annulus roughly requires the observation radius to lie outside the phase-field/plastic zone but sufficiently inside the specimen-scale and heterogeneity-scale boundaries.

If a stiff particle lies directly inside what would otherwise be the K-dominant annulus, the homogeneous K-field is physically perturbed. That is not a meshing failure.

For verification, I would have the code/solver evaluate:

$$
J(r_1), J(r_2), J(r_3), \ldots
$$

over several contours and examine path independence away from the diffuse-crack and particle-interface zones. That is much stronger evidence than relying on a mesh-density criterion alone.

---

# 8. What I would tell the coding AI to build

Rather than asking another agent to “write an arbitrary all-hexahedral mesher,” I would constrain the problem:

> Develop an all-hexahedral conforming mesher specialized for an axis-aligned cuboidal domain containing (N) mutually nonintersecting analytic spheres of arbitrary centers and radii.

Then implement **two interchangeable backends**.

### Backend A — local macroblock mesher

This should be the first open/Python implementation.

Each particle gets a six-patch cubed-sphere interface and a structured sphere-to-box shell. The box boundary connects directly to background Cartesian cells.

This can produce exceptionally clean meshes when particle separation is adequate.

### Backend B — adaptive octree/interface mesher

Use this for configurations where local boxes overlap or particles approach closely.

The design should borrow from Qian/Zhang-style isocontouring and particularly HybridOctree_Hex [S5–S7]:

```text
adaptive octree
      ↓
material/interface classification
      ↓
balanced refinement
      ↓
all-hex transition templates
      ↓
interface reconstruction
      ↓
analytic sphere projection
      ↓
pillowing
      ↓
Jacobian-constrained optimization
      ↓
Exodus
```

The API should make the macroblock backend the fast path and octree backend the general fallback.

That would be much more realistic than attempting a frame-field or whisker-weaving implementation first.

---

# 9. Validation targets for the mesher

The code should automatically report at least these quantities:

| Category         | Test                                                    |                           |                            |
| ---------------- | ------------------------------------------------------- | ------------------------- | -------------------------- |
| Topology         | Every volume element is a hex                           |                           |                            |
| Conformity       | No hanging nodes unless explicitly supported            |                           |                            |
| Interface        | Matrix and particle share identical interface nodes     |                           |                            |
| Sphere location  | (                                                       | |\mathbf{x}-\mathbf{c}|-R | ) for every interface node |
| Geometric volume | Numerical particle volume versus (4\pi R^3/3)           |                           |                            |
| Validity         | Jacobian (>0) everywhere                                |                           |                            |
| Quality          | Minimum scaled Jacobian / condition number / distortion |                           |                            |
| Ligaments        | Elements across each minimum matrix ligament            |                           |                            |
| Phase field      | (h/\ell) statistics                                     |                           |                            |
| Exterior BCs     | Correct six cube sidesets                               |                           |                            |
| Exodus           | Successful read-back with block/set counts unchanged    |                           |                            |
| Mechanics        | Single-sphere elastic stress-concentration benchmark    |                           |                            |
| Fracture         | Mesh-orientation and (h/\ell) convergence               |                           |                            |
| Randomness       | Repeat across many seeds and varying (N)                |                           |                            |

A useful acceptance philosophy is **validity first, quality second, geometric fidelity third, element count fourth**. A beautiful sphere surrounded by a few inverted hexes is unusable.

---

# 10. Recommended development order

I would invest effort in this order:

**First prototype:** Cubit/Sculpt + Python-generated random spheres + analytic projection + Exodus. This establishes whether Sculpt-quality meshes solve your actual elastoplastic phase-field model.

**Second prototype:** Pure Python cubed-sphere macroblock mesher using Gmsh/OpenCASCADE. This determines how much element count you can save by exploiting spherical geometry explicitly.

**Third:** Add adaptive background/octree transitions and a fallback for close-particle configurations.

**Only after those:** investigate general polycube/frame-field decomposition.

I would not start with whisker weaving, a general-purpose polycube solver, or a machine-learning geometry decomposition system. Those dramatically increase implementation risk without exploiting the strongest fact available to you: **all inclusions are analytic spheres inside a cube**.

---

## Key literature

| ID  | Reference                                                                                                                                                                                                                                                                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S1  | *Pietroni, N., Campen, M., Sheffer, A., Cherchi, G., et al. (2022). Hex-mesh generation and processing: a survey. ACM Transactions on Graphics. [https://doi.org/10.1145/3554920](https://doi.org/10.1145/3554920)*                                                                                                                                                                              |
| S2  | *Tautges, T. J., Blacker, T., & Mitchell, S. (1996). The whisker weaving algorithm: a connectivity-based method for constructing all-hexahedral finite element meshes. International Journal for Numerical Methods in Engineering, 39(19). [Publisher link](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0207%2819961015%2939:19%3C3327::AID-NME2%3E3.0.CO;2-H)*               |
| S3  | *Owen, S. J., & Shelton, T. R. (2015). Evaluation of grid-based hex meshes for solid mechanics. Engineering with Computers. [https://doi.org/10.1007/s00366-014-0368-8](https://link.springer.com/article/10.1007/s00366-014-0368-8)*                                                                                                                                                            |
| S4  | *Le Goff, N., Ledoux, F., & Owen, S. (2018). Hexahedral mesh modification to preserve volume. Computer-Aided Design. [Publisher link](https://www.sciencedirect.com/science/article/pii/S0010448518303981)*                                                                                                                                                                                      |
| S5  | *Zhang, Y., & Bajaj, C. (2006). Adaptive and quality quadrilateral/hexahedral meshing from volumetric data. Computer Methods in Applied Mechanics and Engineering. [Publisher link](https://www.sciencedirect.com/science/article/pii/S0045782505001374)*                                                                                                                                        |
| S6  | *Qian, J., & Zhang, Y. (2012). Automatic unstructured all-hexahedral mesh generation from B-Reps for non-manifold CAD assemblies. Engineering with Computers. [https://doi.org/10.1007/s00366-011-0232-z](https://link.springer.com/article/10.1007/s00366-011-0232-z)*                                                                                                                          |
| S7  | *Tong, H., Halilaj, E., & Zhang, Y. J. (2024). HybridOctree_Hex: Hybrid octree-based adaptive all-hexahedral mesh generation with Jacobian control. Journal of Computational Science. [Publisher link](https://www.sciencedirect.com/science/article/pii/S1877750324000711)*                                                                                                                     |
| S8  | *Gregson, J., Sheffer, A., & Zhang, E. (2011). All-hex mesh generation via volumetric polycube deformation. Computer Graphics Forum. [Publisher link](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2011.02015.x)*                                                                                                                                                                 |
| S9  | *Brückler, H., Bommes, D., & Campen, M. (2022). Volume parametrization quantization for hexahedral meshing. ACM Transactions on Graphics. [https://doi.org/10.1145/3528223.3530123](https://dl.acm.org/doi/abs/10.1145/3528223.3530123)*                                                                                                                                                         |
| S10 | *Owen, S., Brown, N., Chrisochoides, N., Garimella, R., et al. (2026). AI for Mesh Generation and Model Preparation: A Review. Proceedings of the 2026 SIAM International Meshing Roundtable. [SIAM link](https://epubs.siam.org/doi/abs/10.1137/1.9781611979138.1)*                                                                                                                             |
| S11 | *Knupp, P. M. (2000). Achieving finite element mesh quality via optimization of the Jacobian matrix norm and associated quantities, Part II: framework for volume mesh optimization. International Journal for Numerical Methods in Engineering, 48(8). [Publisher link](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0207%2820000720%2948:8%3C1165::AID-NME940%3E3.0.CO;2-Y)* |
| S12 | *Richter, H. (2017). Mote3D: an open-source toolbox for modelling periodic random particulate microstructures. Modelling and Simulation in Materials Science and Engineering. [IOP link](https://iopscience.iop.org/article/10.1088/1361-651X/aa629a/meta)*                                                                                                                                      |
| S13 | *Schneider, K., Klusemann, B., & Bargmann, S. (2016). Automatic three-dimensional geometry and mesh generation of periodic representative volume elements for matrix-inclusion composites. Advances in Engineering Software. [Publisher link](https://www.sciencedirect.com/science/article/pii/S0965997816301247)*                                                                              |
| S14 | *Geuzaine, C., & Remacle, J.-F. (2009). Gmsh: A 3-D finite element mesh generator with built-in pre- and post-processing facilities. International Journal for Numerical Methods in Engineering, 79(11), 1309–1331. [https://doi.org/10.1002/nme.2579](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2579)*                                                                                |
| S15 | *Walters, D. J., Luscher, D. J., & Yeager, J. D. (2020). Volumetric analysis and mesh generation of real and artificial microstructural geometries. MethodsX. [Publisher link](https://www.sciencedirect.com/science/article/pii/S2215016120300753)*                                                                                                                                             |
| S16 | *Henrich, M., Fehlemann, N., Bexter, F., Neite, M., Kong, L., et al. (2023). DRAGen—A deep learning supported RVE generator framework for complex microstructure models. Heliyon. [Publisher link](https://www.cell.com/heliyon/fulltext/S2405-8440%2823%2906211-4)*                                                                                                                             |
| S17 | *Heltai, L., Bangerth, W., Kronbichler, M., et al. (2021). Propagating geometry information to finite element computations. ACM Transactions on Mathematical Software. [https://doi.org/10.1145/3468428](https://dl.acm.org/doi/abs/10.1145/3468428)*                                                                                                                                            |
| S18 | *Cajuhi, T., Sanavia, L., & De Lorenzis, L. (2018). Phase-field modeling of fracture in variably saturated porous media. Computational Mechanics. [https://doi.org/10.1007/s00466-017-1459-3](https://link.springer.com/article/10.1007/s00466-017-1459-3)*                                                                                                                                      |
| S19 | *Ambati, M., Gerasimov, T., & De Lorenzis, L. (2015). Phase-field modeling of ductile fracture. Computational Mechanics. [https://doi.org/10.1007/s00466-015-1151-4](https://link.springer.com/article/10.1007/s00466-015-1151-4)*                                                                                                                                                               |

### The strongest recommendation

If another AI agent is going to turn this research into code, give it the **cubed-sphere macroblock + adaptive-octree fallback specification**, not the vague task “generate arbitrary unstructured hex meshes.”

That changes the problem from an unsolved general meshing problem into a specialized algorithm with a realistic path to Python, analytically located sphere interfaces, controlled element counts, positive-Jacobian optimization, and reliable Exodus II output.
