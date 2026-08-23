# Bibliography

Consolidated from both source reviews and de-duplicated. IDs prefixed `S` are
the ScholarGPT review's own numbering, retained so its text can be followed;
entries with no `S` id appear only in the SciSpace review.

References are grouped by the role they play in Hexaform's design rather than
alphabetically, so the list doubles as a reading order.

---

## Surveys and problem statements

**[S1] Pietroni, N., Campen, M., Sheffer, A., Cherchi, G., et al. (2022).**
*Hex-mesh generation and processing: a survey.* ACM Transactions on Graphics.
<https://doi.org/10.1145/3554920>
— The standard reference for why automatic all-hex meshing remains open. Read
first if you want one paper.

**[S10] Owen, S., Brown, N., Chrisochoides, N., Garimella, R., et al. (2026).**
*AI for Mesh Generation and Model Preparation: A Review.* Proceedings of the
2026 SIAM International Meshing Roundtable.
<https://epubs.siam.org/doi/abs/10.1137/1.9781611979138.1>
— Current state of learning-assisted decomposition. Treated by both reviews as a
research accelerator, not a production method.

---

## Advancing-front and dual/topological methods

*Background for why Hexaform does not use them; see `synthesis.md` §4.*

**Staten, M. L., Owen, S. J., & Blacker, T. D. (2005).** *Unconstrained Paving
and Plastering: a new idea for all-hexahedral mesh generation.*
<https://scispace.com/papers/unconstrained-paving-plastering-a-new-idea-for-all-4piyn45wbk>
— Source of the key warning: a fully prescribed boundary quad mesh
over-constrains the interior.

**Staten, M. L., et al. (2010).** *Unconstrained plastering — hexahedral mesh
generation via advancing-front geometry decomposition.*
<https://scispace.com/papers/unconstrained-plastering-hexahedral-mesh-generation-via-21a5v6p6r3>
— Also the source of the observation that hex connectivity is *globally*
coupled, which is why node-snapping schemes degrade.

**[S2] Tautges, T. J., Blacker, T., & Mitchell, S. (1996).** *The whisker
weaving algorithm.* International Journal for Numerical Methods in Engineering,
39(19).
<https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0207(19961015)39:19%3C3327::AID-NME2%3E3.0.CO;2-H>

**Folwell, N., & Mitchell, S. (1999).** *Reliable Whisker Weaving via Curve
Contraction.*
<https://scispace.com/papers/reliable-whisker-weaving-via-curve-contraction-1m7fm2r3ci>
— Demonstrates that valid topology and good geometry are *distinct* problems.
Directly motivates Hexaform's separate quality gate.

---

## Grid-based and octree methods

*The family Hexaform's background grid belongs to, and the basis of the planned
octree backend.*

**[S3] Owen, S. J., & Shelton, T. R. (2015).** *Evaluation of grid-based hex
meshes for solid mechanics.* Engineering with Computers.
<https://doi.org/10.1007/s00366-014-0368-8>
— Grid-based meshes matched or beat pave-and-sweep in elastic and
elastic-plastic tests. The empirical licence for this whole approach.

**[S4] Le Goff, N., Ledoux, F., & Owen, S. (2018).** *Hexahedral mesh
modification to preserve volume.* Computer-Aided Design.
<https://www.sciencedirect.com/science/article/pii/S0010448518303981>

**[S5] Zhang, Y., & Bajaj, C. (2006).** *Adaptive and quality
quadrilateral/hexahedral meshing from volumetric data.* CMAME.
<https://www.sciencedirect.com/science/article/pii/S0045782505001374>

**[S6] Qian, J., & Zhang, Y. (2012).** *Automatic unstructured all-hexahedral
mesh generation from B-Reps for non-manifold CAD assemblies.* Engineering with
Computers. <https://doi.org/10.1007/s00366-011-0232-z>

**[S7] Tong, H., Halilaj, E., & Zhang, Y. J. (2024).** *HybridOctree_Hex: hybrid
octree-based adaptive all-hexahedral mesh generation with Jacobian control.*
Journal of Computational Science.
<https://www.sciencedirect.com/science/article/pii/S1877750324000711>
— **The reference implementation to study before writing Hexaform's octree
backend.**

---

## Polycube and cuboid decomposition

**[S8] Gregson, J., Sheffer, A., & Zhang, E. (2011).** *All-hex mesh generation
via volumetric polycube deformation.* Computer Graphics Forum.
<https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2011.02015.x>

**[S9] Brückler, H., Bommes, D., & Campen, M. (2022).** *Volume parametrization
quantization for hexahedral meshing.* ACM TOG.
<https://doi.org/10.1145/3528223.3530123>

**Fang, X., et al. (2016).** *All-hex meshing using closed-form induced
polycube.*
<https://scispace.com/papers/all-hex-meshing-using-closed-form-induced-polycube-1kg6d4ys3t>
— Notable for explicitly handling disconnected boundaries and voids.

**Hu, K., & Zhang, Y. J. (2016).** *Centroidal Voronoi tessellation based
polycube construction for adaptive all-hexahedral mesh generation.*
<https://scispace.com/papers/centroidal-voronoi-tessellation-based-polycube-construction-1500d9cj6w>

**Li, L., et al. (2021).** *Interactive All-Hex Meshing via Cuboid
Decomposition.*
<https://scispace.com/papers/interactive-all-hex-meshing-via-cuboid-decomposition-3lsdvzqc6b>
— Source of the crucial distinction: *a method may internally use voxels without
producing voxelated geometry.* This is the idea Hexaform exploits.

---

## Quality and optimisation

**[S11] Knupp, P. M. (2000).** *Achieving finite element mesh quality via
optimization of the Jacobian matrix norm and associated quantities, Part II.*
IJNME, 48(8).
<https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0207(20000720)48:8%3C1165::AID-NME940%3E3.0.CO;2-Y>
— The reference for the smoothing step Hexaform has not yet needed.

Also relevant, though cited by neither review: the **Verdict** library
(Sandia SAND2007-1751) defines the scaled-Jacobian and edge-ratio metrics
implemented in `hexaform/core/quality.py`.

---

## Random particulate microstructures and RVE generation

**[S12] Richter, H. (2017).** *Mote3D: an open-source toolbox for modelling
periodic random particulate microstructures.* MSMSE.
<https://iopscience.iop.org/article/10.1088/1361-651X/aa629a/meta>

**[S13] Schneider, K., Klusemann, B., & Bargmann, S. (2016).** *Automatic 3-D
geometry and mesh generation of periodic RVEs for matrix-inclusion composites.*
Advances in Engineering Software.
<https://www.sciencedirect.com/science/article/pii/S0965997816301247>

**[S15] Walters, D. J., Luscher, D. J., & Yeager, J. D. (2020).** *Volumetric
analysis and mesh generation of real and artificial microstructural geometries.*
MethodsX. <https://www.sciencedirect.com/science/article/pii/S2215016120300753>

**[S16] Henrich, M., et al. (2023).** *DRAGen — a deep learning supported RVE
generator framework.* Heliyon.
<https://www.cell.com/heliyon/fulltext/S2405-8440(23)06211-4>
— Cited by ScholarGPT for the report that a generic Exodus writer did not
produce a working file. See `synthesis.md` §5: not independently verified.

**Elbana, A., et al. (2024).** *Multiscale modelling of particulate composites
with spherical inclusions.*
<https://scispace.com/papers/multiscale-modelling-of-particulate-composites-with-4g7y7q2za8>
— Reports particle volume fractions up to ~50% with periodicity, well above what
Hexaform's RSA packer currently reaches.

**Chakkour, T. (2024).** *Finite element modelling of complex 3D image data with
quantification and analysis.*
<https://scispace.com/papers/finite-element-modelling-of-complex-3d-image-data-with-4k38t0gq9d>
— Conforming vs voxelised microstructure meshes.

---

## Geometry representation in FE

**[S17] Heltai, L., Bangerth, W., Kronbichler, M., et al. (2021).**
*Propagating geometry information to finite element computations.* ACM TOMS.
<https://doi.org/10.1145/3468428>
— Directly relevant to the HEX20/HEX27 upgrade path: how to keep exact geometry
attached to a mesh rather than baked into node positions.

---

## Phase-field fracture

**[S18] Cajuhi, T., Sanavia, L., & De Lorenzis, L. (2018).** *Phase-field
modeling of fracture in variably saturated porous media.* Computational
Mechanics. <https://doi.org/10.1007/s00466-017-1459-3>
— Cited for the `h ≤ ℓ/2` starting point. Both the source review and this
repository flag it as an order of magnitude requiring your own convergence
study, **not** a rule.

**[S19] Ambati, M., Gerasimov, T., & De Lorenzis, L. (2015).** *Phase-field
modeling of ductile fracture.* Computational Mechanics.
<https://doi.org/10.1007/s00466-015-1151-4>
— The closest match to the elastoplastic formulation Hexaform's meshes are
intended for.

---

## Software

**[S14] Geuzaine, C., & Remacle, J.-F. (2009).** *Gmsh: a 3-D finite element mesh
generator.* IJNME, 79(11), 1309–1331. <https://doi.org/10.1002/nme.2579>

**Coreform Cubit / Sandia CUBIT + Sculpt** — ranked first by both reviews;
declined as a dependency for the reasons in `synthesis.md` §2.1.

**SEACAS / Exodus II** — the format specification underlying
`hexaform/io/exodus.py`. <https://sandialabs.github.io/seacas-docs/>
