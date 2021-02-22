# asimov-contact
Contact models for ASiMoV


## Plan for contact

1. Rigid plane Nitsche unilateral contact (frictionless/friction), Tresca/Coloumb.
2. Matching meshes frictionless
 - Handwritten custom integration kernel for surface integrals on arbitrary cell type.
   - Python/numba
 - Consider parallel computing implications. Remeshing/graph partitioning? How should we do this for realistic problems?
 - Should we consider self-contact or contact between multiple surface
3. Non-matching meshes frictionless.
 - How to ensure sufficiently accurate quadrature rule.
 - Subtriangulation of a submesh of facets
 - Dig into how to do the integration (check GETFEM and other Nitsche based literature).
4. Unbiased approach (how to make the inverse of the distance function)
