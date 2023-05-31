// Copyright (C) 2021-2022 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "KernelData.h"
#include "QuadratureRule.h"
#include "SubMesh.h"
#include "contact_kernels.h"
#include "elasticity.h"
#include "geometric_quantities.h"
#include "meshtie_kernels.h"
#include "utils.h"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>

using mat_set_fn = const std::function<int(
    const std::span<const std::int32_t>&, const std::span<const std::int32_t>&,
    const std::span<const PetscScalar>&)>;

namespace dolfinx_contact
{

class Contact
{
public:
  /// Constructor
  /// @param[in] markers List of meshtags defining the contact surfaces
  /// @param[in] surfaces Adjacency list. Links of i contains meshtag values
  /// associated with ith meshtag in markers
  /// @param[in] contact_pairs list of pairs (i, j) marking the ith and jth
  /// surface in surfaces->array() as a contact pair
  /// @param[in] V The functions space
  /// @param[in] q_deg The quadrature degree.
  Contact(const std::vector<
              std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
          std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
              surfaces,
          const std::vector<std::array<int, 2>>& contact_pairs,
          std::shared_ptr<dolfinx::fem::FunctionSpace<double>> V,
          const int q_deg = 3, ContactMode mode = ContactMode::ClosestPoint);

  /// Return meshtag value for surface with index surface
  /// @param[in] surface - the index of the surface
  int surface_mt(int surface) const { return _surfaces[surface]; }

  /// Return contact pair
  /// @param[in] pair - the index of the contact pair
  const std::array<int, 2>& contact_pair(int pair) const
  {
    return _contact_pairs[pair];
  }

  // Return active entities for surface s
  std::span<const std::int32_t> active_entities(int s) const
  {
    return _cell_facet_pairs->links(s);
  }

  // Return number of facets in surface s owned by the process
  std::size_t local_facets(int s) const { return _local_facets[s]; }

  // set quadrature rule
  void set_quadrature_rule(QuadratureRule q_rule)
  {
    _quadrature_rule = std::make_shared<QuadratureRule>(q_rule);
  }

  // set search radius for ray-tracing
  void set_search_radius(double r) { _radius = r; }

  /// return size of coefficients vector per facet on s
  /// @param[in] meshtie - Type of constraint,meshtie if true, unbiased contact
  /// if false
  std::size_t coefficients_size(bool meshtie);

  /// return distance map (adjacency map mapping quadrature points on surface
  /// to closest facet on other surface)
  /// @param[in] surface - index of the surface
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  facet_map(int surface) const
  {
    return _facet_maps[surface];
  }

  /// Return the quadrature points on physical facet for each facet on surface
  /// @param[in] surface The index of the surface (0 or 1).
  /// @returns The quadrature points and shape (num_facets, num_q_points, gdim).
  /// The points are flattened row major.
  std::pair<std::vector<double>, std::array<std::size_t, 3>>
  qp_phys(int surface);

  /// Return the submesh of all cells containing facets of the contact surface
  const SubMesh& submesh() const { return _submesh; }

  // Return mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh() const
  {
    return _V->mesh();
  }

  /// @brief Create a PETSc matrix with contact sparsity pattern
  ///
  /// Create a PETSc matrix with the sparsity pattern of the input form and the
  /// coupling contact interfaces
  ///
  /// @param[in] The bilinear form
  /// @param[in] The matrix type, see:
  /// https://petsc.org/main/docs/manualpages/Mat/MatType.html#MatType for
  /// available types
  /// @returns Mat The PETSc matrix
  Mat create_petsc_matrix(const dolfinx::fem::Form<PetscScalar>& a,
                          const std::string& type);

  /// Assemble matrix over exterior facets (for contact facets)
  ///
  /// @param[in] mat_set the function for setting the values in the matrix
  /// @param[in] bcs List of Dirichlet BCs
  /// @param[in] pair index of contact pair
  /// @param[in] kernel The integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_matrix(
      const mat_set_fn& mat_set,
      const std::vector<
          std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
      int pair, const kernel_fn<PetscScalar>& kernel,
      const std::span<const PetscScalar> coeffs, int cstride,
      const std::span<const PetscScalar>& constants);

  /// Assemble vector over exterior facet (for contact facets)
  /// @param[in] b The vector
  /// @param[in] pair index of contact pair
  /// @param[in] kernel The integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_vector(std::span<PetscScalar> b, int pair,
                       const kernel_fn<PetscScalar>& kernel,
                       const std::span<const PetscScalar>& coeffs, int cstride,
                       const std::span<const PetscScalar>& constants);

  /// @brief Generate contact kernel
  ///
  /// The kernel will expect input on the form
  /// @param[in] type The kernel type (Either `Jac` or `Rhs`).
  /// @returns Kernel function that takes in a vector (b) to assemble into, the
  /// coefficients (`c`), the constants (`w`), the local facet entity (`entity
  /// _local_index`), the quadrature permutation and the number of cells on the
  /// other contact boundary coefficients are extracted from.
  /// @note The ordering of coefficients are expected to be `mu`, `lmbda`, `h`,
  /// `gap`, `normals` `test_fn`, `u`, `u_opposite`.
  /// @note The scalar valued coefficients `mu`,`lmbda` and `h` are expected to
  /// be DG-0 functions, with a single value per facet.
  /// @note The coefficients `gap`, `normals`,`test_fn` and `u_opposite` is
  /// packed at quadrature points. The coefficient `u` is packed at dofs.
  /// @note The vector valued coefficents `gap`, `test_fn`, `u`, `u_opposite`
  /// has dimension `bs == gdim`.
  kernel_fn<PetscScalar> generate_kernel(Kernel type);

  /// Compute push forward of quadrature points _qp_ref_facet to the
  /// physical facet for each facet in _facet_"origin_meshtag" Creates and
  /// fills _qp_phys_"origin_meshtag"
  /// @param[in] origin_meshtag flag to choose the surface
  void create_q_phys(int origin_meshtag);

  /// Compute maximum number of links
  /// I think this should actually be part of create_distance_map
  /// which should be easier after the rewrite of contact
  /// It is therefore called inside create_distance_map
  void max_links(int pair);

  /// For a given contact pair, for quadrature point on the first surface
  /// compute the closest candidate facet on the second surface.
  /// @param[in] pair The index of the contact pair
  /// @note This function alters _facet_maps[pair], _max_links[pair],
  /// _qp_phys, _phi_ref_facets
  void create_distance_map(int pair);

  /// Compute and pack the gap function for each quadrature point the set of
  /// facets. For a set of facets; go through the quadrature points on each
  /// facet find the closest facet on the other surface and compute the
  /// distance vector
  /// @param[in] pair - surface on which to integrate
  /// @param[out] c - gap packed on facets. c[i*cstride +  gdim * k+ j]
  /// contains the jth component of the Gap on the ith facet at kth
  /// quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap(int pair);

  /// Compute test functions on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int> pack_test_functions(int pair);

  /// Compute gradient of test functions on opposite surface (initial
  /// configuration) at quadrature points of facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[in] u_packed -u packed on opposite surface per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_grad_test_functions(int pair, const std::span<const PetscScalar>& gap,
                           const std::span<const PetscScalar>& u_packed);

  /// Compute function on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_u_contact(int pair,
                 std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u);

  /// Compute gradient of function on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[in] u_packed -u packed on opposite surface per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_grad_u_contact(int pair,
                      std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
                      const std::span<const PetscScalar> gap,
                      const std::span<const PetscScalar> u_packed);

  /// Compute outward surface normal at x
  /// @param[in] pair - index of contact pair
  /// @returns c - (normals, cstride) ny packed on facets.
  std::pair<std::vector<PetscScalar>, int> pack_nx(int pair);

  /// Compute inward surface normal at Pi(x)
  /// @param[in] pair - index of contact pair
  /// @returns c - normals ny packed on facets.
  std::pair<std::vector<PetscScalar>, int> pack_ny(int pair);

  /// Pack gap with rigid surface defined by x[gdim-1] = -g.
  /// g_vec = zeros(gdim), g_vec[gdim-1] = -g
  /// Gap = x - g_vec
  /// @param[in] pair - index of contact pair
  /// @param[in] g - defines location of plane
  /// @param[out] c - gap packed on facets. c[i, gdim * k+ j] contains the
  /// jth component of the Gap on the ith facet at kth quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap_plane(int pair, double g);

  /// This function updates the submesh geometry for all submeshes using
  /// a function given on the parent mesh
  /// @param[in] u - displacement
  void update_submesh_geometry(dolfinx::fem::Function<PetscScalar>& u);

private:
  std::shared_ptr<QuadratureRule> _quadrature_rule; // quadrature rule
  std::vector<int> _surfaces; // meshtag values for surfaces
  // store index of candidate_surface for each quadrature_surface
  std::vector<std::array<int, 2>> _contact_pairs;
  std::shared_ptr<dolfinx::fem::FunctionSpace<double>> _V; // Function space
  // _facets_maps[i] = adjacency list of closest facet on candidate surface
  // for every quadrature point in _qp_phys[i] (quadrature points on every
  // facet of ith surface)
  std::vector<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>>
      _facet_maps;
  // reference points of the contact points on the opposite surface for each
  // surface output of compute_distance_map
  std::vector<std::vector<double>> _reference_contact_points;
  // shape  associated with _reference_contact_points
  std::vector<std::array<std::size_t, 2>> _reference_contact_shape;
  //  _qp_phys[i] contains the quadrature points on the physical facets for
  //  each facet on ith surface in _surfaces
  std::vector<std::vector<double>> _qp_phys;
  // quadrature points on facets of reference cell
  std::vector<double> _reference_basis;
  std::array<std::size_t, 4> _reference_shape;
  // maximum number of cells linked to a cell on ith surface
  std::vector<std::size_t> _max_links;
  // submesh containing all cells linked to facets on any of the contact
  // surfaces
  SubMesh _submesh;
  // Adjacency list linking facets as (cell, facet) pairs to the index of the
  // surface. The pairs are flattened row-major
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      _cell_facet_pairs;
  // number of facets owned by process for each surface
  std::vector<std::size_t> _local_facets;

  // Contact search mode
  ContactMode _mode;
  // Search radius for ray-tracing
  double _radius = -1;
};
} // namespace dolfinx_contact
