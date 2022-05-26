// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "SubMesh.h"
#include "geometric_quantities.h"
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
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xindex_view.hpp>

using mat_set_fn = const std::function<int(
    const xtl::span<const std::int32_t>&, const xtl::span<const std::int32_t>&,
    const xtl::span<const PetscScalar>&)>;

namespace dolfinx_contact
{
enum class Kernel
{
  Rhs,
  Jac
};

namespace
{
/// Tabulate the coordinate element basis functions at quadrature points
///
/// @param[in] cmap The coordinate element
/// @param[in] cell_type The cell type of the coordiante element
/// @param[in] q_rule The quadrature rule
std::vector<xt::xtensor<double, 2>>
tabulate(const dolfinx::fem::CoordinateElement& cmap,
         const dolfinx::mesh::CellType cell_type,
         const dolfinx_cuas::QuadratureRule& q_rule)
{
  const int tdim = dolfinx::mesh::cell_dim(cell_type);
  assert(tdim - 1 == q_rule.dim());

  const int num_facets = dolfinx::mesh::cell_num_entities(cell_type, tdim - 1);

  // Check that we only have one cell type in quadrature rule
  const dolfinx::mesh::CellType f_type = q_rule.cell_type(0);
  for (int i = 0; i < num_facets; i++)
    if (q_rule.cell_type(i) != f_type)
      throw std::invalid_argument("Prism meshes are currently not supported.");

  const std::vector<xt::xarray<double>>& q_points = q_rule.points_ref();
  assert((std::size_t)num_facets == q_points.size());

  const std::array<std::size_t, 4> tab_shape
      = cmap.tabulate_shape(0, q_points[0].shape(0));

  xt::xtensor<double, 4> basis_values(tab_shape);
  xt::xtensor<double, 2> phi_i({tab_shape[1], tab_shape[2]});

  // Tabulate basis functions at quadrature points  for each
  // facet of the reference cell.
  std::vector<xt::xtensor<double, 2>> phi;
  phi.reserve(num_facets);
  for (int i = 0; i < num_facets; ++i)
  {
    cmap.tabulate(0, q_points[i], basis_values);
    phi_i = xt::view(basis_values, 0, xt::all(), xt::all(), 0);
    phi.push_back(phi_i);
  }
  return phi;
}
} // namespace

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
  Contact(const std::vector<
              std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
          std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
              surfaces,
          const std::vector<std::array<int, 2>>& contact_pairs,
          std::shared_ptr<dolfinx::fem::FunctionSpace> V);

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
  const std::vector<std::pair<std::int32_t, int>>& active_entities(int s) const
  {
    return _cell_facet_pairs[s];
  }
  // return quadrature degree
  int quadrature_degree() const { return _quadrature_degree; }
  void set_quadrature_degree(int deg) { _quadrature_degree = deg; }

  // return size of coefficients vector per facet on s
  std::size_t coefficients_size();

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
  std::vector<xt::xtensor<double, 2>> qp_phys(int surface)
  {
    return _qp_phys[surface];
  }

  /// Return the submesh corresponding to surface
  /// @param[in] surface The index of the surface (0 or 1).
  SubMesh submesh(int surface) { return _submeshes[surface]; }
  // Return mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh() const { return _V->mesh(); }
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
      const xtl::span<const PetscScalar> coeffs, int cstride,
      const xtl::span<const PetscScalar>& constants);

  /// Assemble vector over exterior facet (for contact facets)
  /// @param[in] b The vector
  /// @param[in] pair index of contact pair
  /// @param[in] kernel The integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_vector(xtl::span<PetscScalar> b, int pair,
                       const kernel_fn<PetscScalar>& kernel,
                       const xtl::span<const PetscScalar>& coeffs, int cstride,
                       const xtl::span<const PetscScalar>& constants);

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
  kernel_fn<PetscScalar> generate_kernel(dolfinx_contact::Kernel type)
  {
    // Extract mesh data
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
    assert(mesh);
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();         // topological dimension
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
    bool affine = cmap.is_affine();
    const int num_coordinate_dofs = cmap.dim();

    std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
    if (const bool needs_dof_transformations
        = element->needs_dof_transformations();
        needs_dof_transformations)
    {
      throw std::invalid_argument("Contact-kernels are not supporting finite "
                                  "elements requiring dof transformations.");
    }

    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::size_t ndofs_cell = dofmap->element_dof_layout().num_dofs();
    const std::size_t bs = dofmap->bs();
    if (bs != gdim)
    {
      throw std::invalid_argument(
          "The geometric dimension of the mesh is not equal to the block size "
          "of the function space.");
    }

    // NOTE: Assuming same number of quadrature points on each cell
    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
    const std::size_t max_links
        = *std::max_element(_max_links.begin(), _max_links.end());

    // Structures needed for basis function tabulation
    // phi and grad(phi) at quadrature points
    const std::size_t num_facets = dolfinx::mesh::cell_num_entities(
        mesh->topology().cell_type(), tdim - 1);
    assert(num_facets == _qp_ref_facet.size());

    std::vector<xt::xtensor<double, 2>> phi;
    phi.reserve(num_facets);
    std::vector<xt::xtensor<double, 3>> dphi;
    phi.reserve(num_facets);
    std::vector<xt ::xtensor<double, 3>> dphi_c;
    dphi_c.reserve(num_facets);

    // Temporary structures used in loop
    xt::xtensor<double, 4> cell_tab(
        {(std::size_t)tdim + 1, num_q_points, ndofs_cell, bs});
    xt::xtensor<double, 2> phi_i({num_q_points, ndofs_cell});
    xt::xtensor<double, 3> dphi_i(
        {(std::size_t)tdim, num_q_points, ndofs_cell});
    std::array<std::size_t, 4> tabulate_shape
        = cmap.tabulate_shape(1, num_q_points);
    xt::xtensor<double, 4> c_tab(tabulate_shape);
    assert(tabulate_shape[0] - 1 == (std::size_t)tdim);
    xt::xtensor<double, 3> dphi_ci(
        {tabulate_shape[0] - 1, tabulate_shape[1], tabulate_shape[2]});

    // Tabulate basis functions and first order derivatives for each facet in
    // the reference cell. This tabulation is done both for the finite element
    // of the unknown and the coordinate element (which might differ)
    std::for_each(_qp_ref_facet.cbegin(), _qp_ref_facet.cend(),
                  [&](const auto& q_facet)
                  {
                    assert(q_facet.shape(0) == num_q_points);
                    element->tabulate(cell_tab, q_facet, 1);

                    phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
                    phi.push_back(phi_i);

                    dphi_i = xt::view(cell_tab, xt::range(1, tabulate_shape[0]),
                                      xt::all(), xt::all(), 0);
                    dphi.push_back(dphi_i);

                    // Tabulate coordinate element of reference cell
                    cmap.tabulate(1, q_facet, c_tab);
                    dphi_ci = xt::view(c_tab, xt::range(1, tabulate_shape[0]),
                                       xt::all(), xt::all(), 0);
                    dphi_c.push_back(dphi_ci);
                  });

    // Coefficient offsets
    // Expecting coefficients in following order:
    // mu, lmbda, h, gap, normals, test_fn, u, u_opposite
    std::array<std::size_t, 8> cstrides
        = {1,
           1,
           1,
           num_q_points * gdim,
           num_q_points * gdim,
           num_q_points * ndofs_cell * bs * max_links,
           ndofs_cell * bs,
           num_q_points * bs};
    std::array<std::size_t, 9> offsets;
    offsets[0] = 0;
    std::partial_sum(cstrides.cbegin(), cstrides.cend(),
                     std::next(offsets.begin()));

    // As reference facet and reference cell are affine, we do not need to
    // compute this per quadrature point
    basix::cell::type basix_cell
        = dolfinx::mesh::cell_type_to_basix_type(mesh->topology().cell_type());
    xt::xtensor<double, 3> ref_jacobians
        = basix::cell::facet_jacobians(basix_cell);

    // Get facet normals on reference cell
    xt::xtensor<double, 2> facet_normals
        = basix::cell::facet_outward_normals(basix_cell);

    // Get update Jacobian function (for each quadrature point)
    auto update_jacobian
        = dolfinx_contact::get_update_jacobian_dependencies(cmap);

    // Get update FacetNormal function (for each quadrature point)
    auto update_normal = dolfinx_contact::get_update_normal(cmap);

    // Get quadrature weights;
    const std::vector<std::vector<double>>& qweights = _qw_ref_facet;

    /// @brief Assemble kernel for RHS of unbiased contact problem
    ///
    /// Assemble of the residual of the unbiased contact problem into vector
    /// `b`.
    /// @param[in,out] b The vector to assemble the residual into
    /// @param[in] c The coefficients used in kernel. Assumed to be
    /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
    /// @param[in] w The constants used in kernel. Assumed to be ordered as
    /// `gamma`, `theta`.
    /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed to
    /// be padded to 3D, (shape (num_nodes, 3)).
    kernel_fn<PetscScalar> unbiased_rhs
        = [num_coordinate_dofs, gdim, tdim, ref_jacobians, dphi_c,
           facet_normals, phi, dphi, qweights, ndofs_cell, offsets, affine,
           update_jacobian, update_normal, bs, num_q_points](
              std::vector<std::vector<PetscScalar>>& b, const PetscScalar* c,
              const PetscScalar* w, const double* coordinate_dofs,
              const int facet_index, const std::size_t num_links)
    {
      // NOTE: DOLFINx has 3D input coordinate dofs
      // FIXME: These array should be views (when compute_jacobian doesn't use
      // xtensor)
      std::array<std::size_t, 2> shape = {(std::size_t)num_coordinate_dofs, 3};
      xt::xtensor<double, 2> coord = xt::adapt(
          coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

      // Extract the first derivative of the coordinate element (cell) of
      // degrees of freedom on the facet
      const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];

      // Create data structures for jacobians
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, (std::size_t)tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({(std::size_t)tdim, gdim});
      // J_f facet jacobian, J_tot = J * J_f
      xt::xtensor<double, 2> J_f
          = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
      xt::xtensor<double, 2> J_tot
          = xt::zeros<double>({J.shape(0), J_f.shape(1)});
      double detJ;
      auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

      // Normal vector on physical facet at a single quadrature point
      xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});

      // Pre-compute jacobians and normals for affine meshes
      if (affine)
      {
        detJ = dolfinx_contact::compute_facet_jacobians(0, J, K, J_tot, J_f,
                                                        dphi_fc, coord);
        dolfinx_contact::physical_facet_normal(
            n_phys, K, xt::row(facet_normals, facet_index));
      }

      // Extract constants used inside quadrature loop
      double gamma = c[2] / w[0];     // h/gamma
      double gamma_inv = w[0] / c[2]; // gamma/h
      double theta = w[1];
      double mu = c[0];
      double lmbda = c[1];

      // Extract reference to the tabulated basis function at the local
      // facet
      const xt::xtensor<double, 2>& phi_f = phi[facet_index];
      const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];

      // Extract reference to quadrature weights for the local facet
      const std::vector<double>& weights = qweights[facet_index];
      assert(weights.size() == num_q_points);

      // Temporary data structures used inside quadrature loop
      std::array<double, 3> n_surf = {0, 0, 0};
      xt::xtensor<double, 2> tr({ndofs_cell, gdim});
      xt::xtensor<double, 2> epsn({ndofs_cell, gdim});
      for (std::size_t q = 0; q < weights.size(); q++)
      {

        // Update Jacobian and physical normal
        detJ = update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord);
        update_normal(n_phys, K, facet_normals, facet_index);
        double n_dot = 0;
        double gap = 0;
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        for (std::size_t i = 0; i < gdim; i++)
        {

          n_surf[i] = -c[offsets[4] + q * gdim + i];
          n_dot += n_phys(i) * n_surf[i];
          gap += c[offsets[3] + q * gdim + i] * n_surf[i];
        }

        // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
        std::fill(tr.begin(), tr.end(), 0.0);
        std::fill(epsn.begin(), epsn.end(), 0.0);
        for (std::size_t j = 0; j < ndofs_cell; j++)
        {
          for (std::size_t l = 0; l < bs; l++)
          {
            for (int k = 0; k < tdim; k++)
            {
              tr(j, l) += K(k, l) * dphi_f(k, q, j);
              for (std::size_t s = 0; s < gdim; s++)
              {
                epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                              * (n_phys(s) * n_surf[l] + n_phys(l) * n_surf[s]);
              }
            }
          }
        }
        // compute tr(eps(u)), epsn at q
        double tr_u = 0;
        double epsn_u = 0;
        double jump_un = 0;
        for (std::size_t i = 0; i < ndofs_cell; i++)
        {
          std::size_t block_index = offsets[6] + i * bs;
          for (std::size_t j = 0; j < bs; j++)
          {
            PetscScalar coeff = c[block_index + j];
            tr_u += coeff * tr(i, j);
            epsn_u += coeff * epsn(i, j);
            jump_un += coeff * phi_f(q, i) * n_surf[j];
          }
        }
        std::size_t offset_u_opp = offsets[7] + q * bs;
        for (std::size_t j = 0; j < bs; ++j)
          jump_un += -c[offset_u_opp + j] * n_surf[j];
        double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
        const double w0 = weights[q] * detJ;
        double Pn_u
            = dolfinx_contact::R_plus((jump_un - gap) - gamma * sign_u) * w0;
        sign_u *= w0;

        // Fill contributions of facet with itself

        for (std::size_t i = 0; i < ndofs_cell; i++)
        {
          for (std::size_t n = 0; n < bs; n++)
          {
            double v_dot_nsurf = n_surf[n] * phi_f(q, i);
            double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
            // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
            double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
            b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;
            // 0.5 * (-theta * gamma * sign_v * sign_u + Pn_u * Pn_v);

            // entries corresponding to v on the other surface
            for (std::size_t k = 0; k < num_links; k++)
            {
              std::size_t index = offsets[5]
                                  + k * num_q_points * ndofs_cell * bs
                                  + i * num_q_points * bs + q * bs + n;
              double v_n_opp = c[index] * n_surf[n];

              b[k + 1][n + i * bs] -= 0.5 * gamma_inv * v_n_opp * Pn_u;
            }
          }
        }
      }
    };

    /// @brief Assemble kernel for Jacobian (LHS) of unbiased contact
    /// problem
    ///
    /// Assemble of the residual of the unbiased contact problem into matrix
    /// `A`.
    /// @param[in,out] A The matrix to assemble the Jacobian into
    /// @param[in] c The coefficients used in kernel. Assumed to be
    /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
    /// @param[in] w The constants used in kernel. Assumed to be ordered as
    /// `gamma`, `theta`.
    /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
    /// to be padded to 3D, (shape (num_nodes, 3)).
    kernel_fn<PetscScalar> unbiased_jac
        = [=](std::vector<std::vector<PetscScalar>>& A, const double* c,
              const double* w, const double* coordinate_dofs,
              const int facet_index, const std::size_t num_links)
    {
      // Reshape coordinate dofs to two dimensional array
      // NOTE: DOLFINx has 3D input coordinate dofs
      std::array<std::size_t, 2> shape = {(std::size_t)num_coordinate_dofs, 3};

      // FIXME: These array should be views (when compute_jacobian doesn't
      // use xtensor)
      xt::xtensor<double, 2> coord = xt::adapt(
          coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

      // Extract the first derivative of the coordinate element (cell) of
      // degrees of freedom on the facet
      const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];

      // Create data structures for jacobians
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, (std::size_t)tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({(std::size_t)tdim, gdim});
      // J_f facet jacobian, J_tot = J * J_f
      xt::xtensor<double, 2> J_f
          = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
      xt::xtensor<double, 2> J_tot
          = xt::zeros<double>({J.shape(0), J_f.shape(1)});
      double detJ;
      auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

      // Normal vector on physical facet at a single quadrature point
      xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});

      // Pre-compute jacobians and normals for affine meshes
      if (affine)
      {
        detJ = dolfinx_contact::compute_facet_jacobians(0, J, K, J_tot, J_f,
                                                        dphi_fc, coord);
        dolfinx_contact::physical_facet_normal(
            n_phys, K, xt::row(facet_normals, facet_index));
      }

      // Extract scaled gamma (h/gamma) and its inverse
      double gamma = c[2] / w[0];
      double gamma_inv = w[0] / c[2];

      double theta = w[1];
      double mu = c[0];
      double lmbda = c[1];

      const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
      const xt::xtensor<double, 2>& phi_f = phi[facet_index];
      const std::vector<double>& weights = _qw_ref_facet[facet_index];
      xt::xtensor<double, 1> n_surf = xt::zeros<double>({gdim});
      xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
      xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
      for (std::size_t q = 0; q < weights.size(); q++)
      {
        // Update Jacobian and physical normal
        detJ = update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord);
        update_normal(n_phys, K, facet_normals, facet_index);

        double n_dot = 0;
        double gap = 0;
        for (std::size_t i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf(i) = -c[offsets[4] + q * gdim + i];
          n_dot += n_phys(i) * n_surf(i);
          gap += c[offsets[3] + q * gdim + i] * n_surf(i);
        }

        // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
        std::fill(tr.begin(), tr.end(), 0.0);
        std::fill(epsn.begin(), epsn.end(), 0.0);
        for (std::size_t j = 0; j < ndofs_cell; j++)
        {
          for (std::size_t l = 0; l < bs; l++)
          {
            for (int k = 0; k < tdim; k++)
            {
              tr(j, l) += K(k, l) * dphi_f(k, q, j);
              for (std::size_t s = 0; s < gdim; s++)
              {
                epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                              * (n_phys(s) * n_surf(l) + n_phys(l) * n_surf(s));
              }
            }
          }
        }

        // compute tr(eps(u)), epsn at q
        double tr_u = 0;
        double epsn_u = 0;
        double jump_un = 0;

        for (std::size_t i = 0; i < ndofs_cell; i++)
        {
          std::size_t block_index = offsets[6] + i * bs;
          for (std::size_t j = 0; j < bs; j++)
          {
            tr_u += c[block_index + j] * tr(i, j);
            epsn_u += c[block_index + j] * epsn(i, j);
            jump_un += c[block_index + j] * phi_f(q, i) * n_surf(j);
          }
        }
        std::size_t offset_u_opp = offsets[7] + q * bs;
        for (std::size_t j = 0; j < bs; ++j)
          jump_un += -c[offset_u_opp + j] * n_surf(j);
        double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
        double Pn_u
            = dolfinx_contact::dR_plus((jump_un - gap) - gamma * sign_u);

        // Fill contributions of facet with itself
        const double w0 = weights[q] * detJ;
        for (std::size_t j = 0; j < ndofs_cell; j++)
        {
          for (std::size_t l = 0; l < bs; l++)
          {
            double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
            double Pn_du
                = (phi_f(q, j) * n_surf(l) - gamma * sign_du) * Pn_u * w0;

            sign_du *= w0;
            for (std::size_t i = 0; i < ndofs_cell; i++)
            {
              for (std::size_t b = 0; b < bs; b++)
              {
                double v_dot_nsurf = n_surf(b) * phi_f(q, i);
                double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
                double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
                A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * Pn_du * Pn_v;

                // entries corresponding to u and v on the other surface
                for (std::size_t k = 0; k < num_links; k++)
                {
                  std::size_t index = offsets[5]
                                      + k * num_q_points * ndofs_cell * bs
                                      + j * num_q_points * bs + q * bs + l;
                  double du_n_opp = c[index] * n_surf(l);

                  du_n_opp *= w0 * Pn_u;
                  index = offsets[5] + k * num_q_points * ndofs_cell * bs
                          + i * num_q_points * bs + q * bs + b;
                  double v_n_opp = c[index] * n_surf(b);
                  A[3 * k + 1][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                      -= 0.5 * du_n_opp * Pn_v;
                  A[3 * k + 2][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                      -= 0.5 * gamma_inv * Pn_du * v_n_opp;
                  A[3 * k + 3][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                      += 0.5 * gamma_inv * du_n_opp * v_n_opp;
                }
              }
            }
          }
        }
      }
    };
    switch (type)
    {
    case dolfinx_contact::Kernel::Rhs:
      return unbiased_rhs;
    case dolfinx_contact::Kernel::Jac:
      return unbiased_jac;
    default:
      throw std::invalid_argument("Unrecognized kernel");
    }
  }

  /// Compute push forward of quadrature points _qp_ref_facet to the
  /// physical facet for each facet in _facet_"origin_meshtag" Creates and
  /// fills _qp_phys_"origin_meshtag"
  /// @param[in] origin_meshtag flag to choose the surface
  void create_q_phys(int origin_meshtag)
  {
    // Get information depending on surface
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh
        = _submeshes[origin_meshtag].mesh();
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> cell_map
        = _submeshes[origin_meshtag].cell_map();
    const std::vector<std::pair<std::int32_t, int>>& puppet_facets
        = _cell_facet_pairs[origin_meshtag];

    // Geometrical info
    const dolfinx::mesh::Geometry& geometry = mesh->geometry();
    xtl::span<const double> mesh_geometry = geometry.x();
    const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = geometry.dofmap();
    const int gdim = geometry.dim();
    const std::size_t num_dofs_g = cmap.dim();

    // Create storage for output quadrature points
    const std::vector<xt::xarray<double>>& ref_facet_points = _qp_ref_facet;
    xt::xtensor<double, 2> q_phys(
        {ref_facet_points[0].shape(0), (std::size_t)gdim});

    // Check that we have a constant number of points per facet
    const dolfinx::mesh::Topology& topology = mesh->topology();
    const int tdim = topology.dim();
    const std::size_t num_facets
        = dolfinx::mesh::cell_num_entities(topology.cell_type(), tdim - 1);
    assert(num_facets == ref_facet_points.size());
    for (std::size_t i = 0; i < num_facets; ++i)
      if (ref_facet_points[0].shape(0) != ref_facet_points[i].shape(0))
        throw std::invalid_argument("Quadrature rule on prisms not supported");

    // Temporary data array
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});

    // Push forward of quadrature points _qp_ref_facet to physical facet
    // for each facet in _facet_"origin_meshtag"
    _qp_phys[origin_meshtag].reserve(puppet_facets.size());
    _qp_phys[origin_meshtag].clear();
    std::for_each(puppet_facets.cbegin(), puppet_facets.cend(),
                  [&](const auto& facet_pair)
                  {
                    auto [cell, local_index] = facet_pair;

                    // Extract local coordinate dofs
                    auto submesh_cells = cell_map->links(cell);
                    assert(submesh_cells.size() == 1);
                    auto x_dofs = x_dofmap.links(submesh_cells[0]);
                    assert(x_dofs.size() == num_dofs_g);
                    for (std::size_t i = 0; i < num_dofs_g; ++i)
                    {
                      std::copy_n(
                          std::next(mesh_geometry.begin(), 3 * x_dofs[i]), gdim,
                          std::next(coordinate_dofs.begin(), i * gdim));
                    }

                    // push forward of quadrature points _qp_ref_facet to the
                    // physical facet
                    cmap.push_forward(q_phys, coordinate_dofs,
                                      _phi_ref_facets[local_index]);
                    _qp_phys[origin_meshtag].push_back(q_phys);
                  });
  }

  /// Compute maximum number of links
  /// I think this should actually be part of create_distance_map
  /// which should be easier after the rewrite of contact
  /// It is therefore called inside create_distance_map
  void max_links(int pair)
  {
    std::size_t max_links = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    const std::array<int, 2>& contact_pair = _contact_pairs[pair];
    const std::vector<std::pair<std::int32_t, int>>& active_facets
        = _cell_facet_pairs[contact_pair[0]];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
        = _facet_maps[pair];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
        = _submeshes[contact_pair[1]].facet_map();
    for (std::size_t i = 0; i < active_facets.size(); i++)
    {
      std::vector<std::int32_t> linked_cells;
      const tcb::span<const int> links = map->links(i);
      for (auto link : links)
      {
        const tcb::span<const int> facet_pair = facet_map->links(link);
        linked_cells.push_back(facet_pair[0]);
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      max_links = std::max(max_links, linked_cells.size());
    }
    _max_links[pair] = max_links;
  }

  /// Compute closest candidate_facet for each quadrature point in
  /// _qp_phys[puppet_mt]
  /// This is saved as an adjacency list in _facet_maps[puppet_mt]
  /// and an xtensor containing cell_facet_pairs in
  /// _cell_maps[pair]
  void create_distance_map(int pair)
  {
    int puppet_mt = _contact_pairs[pair][0];
    int candidate_mt = _contact_pairs[pair][1];

    // Mesh info
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
    assert(mesh);

    const int gdim = mesh->geometry().dim();
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
    const dolfinx::mesh::Topology& topology = mesh->topology();

    const int tdim = topology.dim();
    const int fdim = tdim - 1;
    const dolfinx::mesh::CellType cell_type = topology.cell_type();
    if ((cell_type == dolfinx::mesh::CellType::pyramid)
        or (cell_type == dolfinx::mesh::CellType::prism))
    {
      throw std::invalid_argument("Pyramid and prism meshes are not supported");
    }

    // submesh info
    std::shared_ptr<const dolfinx::mesh::Mesh> candidate_mesh
        = _submeshes[candidate_mt].mesh();
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
        = candidate_mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    // Create _qp_ref_facet (quadrature points on reference facet)
    dolfinx_cuas::QuadratureRule q_rule(cell_type, _quadrature_degree, fdim);
    _qp_ref_facet = q_rule.points();
    _qw_ref_facet = q_rule.weights();

    _phi_ref_facets = tabulate(cmap, cell_type, q_rule);

    // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
    create_q_phys(puppet_mt);

    // assign puppet_ and candidate_facets
    const std::vector<std::pair<std::int32_t, int>>& candidate_facets
        = _cell_facet_pairs[candidate_mt];
    const std::vector<std::pair<std::int32_t, int>>& puppet_facets
        = _cell_facet_pairs[puppet_mt];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> cell_map
        = _submeshes[candidate_mt].cell_map();
    const std::vector<xt::xtensor<double, 2>>& qp_phys = _qp_phys[puppet_mt];
    std::vector<std::int32_t> submesh_facets(candidate_facets.size());
    for (std::size_t i = 0; i < candidate_facets.size(); ++i)
    {
      const std::pair<std::int32_t, int>& facet_pair = candidate_facets[i];
      submesh_facets[i] = c_to_f->links(
          cell_map->links(facet_pair.first)[0])[facet_pair.second];
    }
    // Create midpoint tree as compute_closest_entity will be called many
    // times
    dolfinx::geometry::BoundingBoxTree master_bbox(*candidate_mesh, fdim,
                                                   submesh_facets);
    dolfinx::geometry::BoundingBoxTree master_midpoint_tree
        = dolfinx::geometry::create_midpoint_tree(*candidate_mesh, fdim,
                                                  submesh_facets);

    // Copy quadrature points to 2D structure
    const std::size_t num_facets = qp_phys.size();
    const std::size_t num_q_points = qp_phys[0].shape(0);
    xt::xtensor<double, 2> quadrature_points
        = xt::zeros<double>({num_facets * num_q_points, (std::size_t)3});
    for (std::size_t i = 0; i < num_facets; ++i)
    {
      assert(qp_phys[i].shape(1) == (std::size_t)gdim);
      for (std::size_t j = 0; j < num_q_points; ++j)
        for (std::size_t k = 0; k < (std::size_t)gdim; ++k)
          quadrature_points(i * num_q_points + j, k) = qp_phys[i](j, k);
    }

    // Create structures used to create adjacency list of closest entity
    std::vector<std::int32_t> offset(puppet_facets.size() + 1, 0);
    for (std::size_t i = 0; i < puppet_facets.size(); ++i)
      offset[i + 1] = std::int32_t((i + 1) * num_q_points);

    std::vector<std::int32_t> closest_entity
        = dolfinx::geometry::compute_closest_entity(
            master_bbox, master_midpoint_tree, *candidate_mesh,
            quadrature_points);
    _facet_maps[pair]
        = std::make_shared<const dolfinx::graph::AdjacencyList<std::int32_t>>(
            closest_entity, offset);

    max_links(pair);
  }

  /// Compute and pack the gap function for each quadrature point the set of
  /// facets. For a set of facets; go through the quadrature points on each
  /// facet find the closest facet on the other surface and compute the
  /// distance vector
  /// @param[in] pair - surface on which to integrate
  /// @param[out] c - gap packed on facets. c[i*cstride +  gdim * k+ j]
  /// contains the jth component of the Gap on the ith facet at kth
  /// quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap(int pair)
  {
    int puppet_mt = _contact_pairs[pair][0];
    int candidate_mt = _contact_pairs[pair][1];
    // Mesh info
    const std::shared_ptr<const dolfinx::mesh::Mesh>& candidate_mesh
        = _submeshes[candidate_mt].mesh();
    assert(candidate_mesh);
    // Get information about submesh geometry and topology
    const dolfinx::mesh::Geometry& geometry = candidate_mesh->geometry();
    const int gdim = geometry.dim();
    xtl::span<const double> mesh_geometry = geometry.x();
    const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
    const dolfinx::fem::ElementDofLayout layout = cmap.create_dof_layout();
    const int tdim = candidate_mesh->topology().dim();
    const int fdim = tdim - 1;

    // Select which side of the contact interface to loop from and get the
    // correct map
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
        = _facet_maps[pair];
    assert(map);
    const std::vector<xt::xtensor<double, 2>>& qp_phys = _qp_phys[puppet_mt];
    const std::size_t num_facets = _cell_facet_pairs[puppet_mt].size();
    const std::size_t num_q_point = _qp_ref_facet[0].shape(0);

    // Get information aboute cell type and number of closure dofs on the facet
    // NOTE: Assumption that we do not have variable facet types (prism/pyramid
    // cell)
    const std::vector<std::int32_t>& closure_dofs
        = layout.entity_closure_dofs(fdim, 0);
    const std::size_t num_facet_dofs = closure_dofs.size();

    // Get all connected facets for each quadrature point
    const std::vector<std::int32_t> master_facets = map->array();
    assert(master_facets.size() == num_facets * num_q_point);

    // Get the geometry dofs for each of the facets for each quadrature point on
    // the opposite surface
    const dolfinx::graph::AdjacencyList<std::int32_t> master_facets_geometry
        = dolfinx_contact::entities_to_geometry_dofs(*candidate_mesh, fdim,
                                                     master_facets);

    // Temporary data structures used in loops
    xt::xtensor<double, 2> point = {{0, 0, 0}};
    xt::xtensor_fixed<double, xt::xshape<3>> dist_vec;
    xt::xtensor<double, 2> master_coords
        = xt::zeros<double>({num_facet_dofs, std::size_t(3)});

    // Pack gap function for each quadrature point on each facet
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = (int)num_q_point * gdim;
    for (std::size_t i = 0; i < num_facets; ++i)
    {
      int offset = (int)i * cstride;
      for (std::size_t q = 0; q < num_q_point; ++q)
      {
        // Get quadrature points in physical space for the ith facet, qth
        // quadrature point
        for (int k = 0; k < gdim; k++)
          point(0, k) = qp_phys[i](q, k);

        // Get the geometry dofs for the ith facet, qth quadrature point
        const tcb::span<const int> master_facet
            = master_facets_geometry.links(int(i * num_q_point + q));
        assert(num_facet_dofs == master_facet.size());

        // Get the coordinates of the geometry on the other interface,
        // and compute the distance of the convex hull created by the points
        for (std::size_t l = 0; l < num_facet_dofs; ++l)
        {
          const int pos = 3 * master_facet[l];
          for (int k = 0; k < gdim; ++k)
            master_coords(l, k) = mesh_geometry[pos + k];
        }

        dist_vec
            = dolfinx::geometry::compute_distance_gjk(master_coords, point);

        // Add distance vector to coefficient array
        for (int k = 0; k < gdim; k++)
          c[offset + q * gdim + k] += dist_vec(k);
      }
    }

    return {std::move(c), cstride};
  }

  /// Compute test functions on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_test_functions(int pair, const xtl::span<const PetscScalar>& gap)
  {
    int puppet_mt = _contact_pairs[pair][0];
    int candidate_mt = _contact_pairs[pair][1];
    // Mesh info
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh
        = _submeshes[candidate_mt].mesh(); // mesh
    assert(mesh);
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
    const dolfinx::graph::AdjacencyList<int>& x_dofmap
        = mesh->geometry().dofmap();
    xtl::span<const double> mesh_geometry = mesh->geometry().x();
    std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
    const std::uint32_t bs = element->block_size();
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint32_t> permutation_info
        = mesh->topology().get_cell_permutation_info();

    // Select which side of the contact interface to loop from and get the
    // correct map
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
        = _facet_maps[pair];
    const std::vector<xt::xtensor<double, 2>>& qp_phys = _qp_phys[puppet_mt];
    const std::vector<std::pair<std::int32_t, int>>& puppet_facets
        = _cell_facet_pairs[puppet_mt];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
        = _submeshes[candidate_mt].facet_map();
    const std::size_t max_links
        = *std::max_element(_max_links.begin(), _max_links.end());
    const std::size_t num_facets = puppet_facets.size();
    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
    const std::int32_t ndofs = _V->dofmap()->cell_dofs(0).size();
    std::vector<PetscScalar> c(
        num_facets * num_q_points * max_links * ndofs * bs, 0.0);
    const int cstride = int(num_q_points * max_links * ndofs * bs);
    xt::xtensor<double, 2> q_points
        = xt::zeros<double>({std::size_t(num_q_points), std::size_t(gdim)});
    xt::xtensor<double, 2> dphi;
    xt::xtensor<double, 3> J = xt::zeros<double>(
        {std::size_t(num_q_points), std::size_t(gdim), std::size_t(tdim)});
    xt::xtensor<double, 3> K = xt::zeros<double>(
        {std::size_t(num_q_points), std::size_t(tdim), std::size_t(gdim)});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_q_points});
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
    std::vector<std::int32_t> perm(num_q_points);
    std::vector<std::int32_t> linked_cells(num_q_points);

    // Loop over all facets
    for (std::size_t i = 0; i < num_facets; i++)
    {
      const tcb::span<const int> links = map->links((int)i);
      assert(links.size() == num_q_points);

      // Compute Pi(x) form points x and gap funtion Pi(x) - x
      for (std::size_t j = 0; j < num_q_points; j++)
      {
        const tcb::span<const int> linked_pair = facet_map->links(links[j]);
        linked_cells[j] = linked_pair[0];
        for (int k = 0; k < gdim; k++)
          q_points(j, k)
              = qp_phys[i](j, k) + gap[i * gdim * num_q_points + j * gdim + k];
      }

      // Sort linked cells
      assert(linked_cells.size() == num_q_points);
      std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
          sorted_cells = dolfinx_contact::sort_cells(
              xtl::span(linked_cells.data(), linked_cells.size()),
              xtl::span(perm.data(), perm.size()));
      const std::vector<std::int32_t>& unique_cells = sorted_cells.first;
      const std::vector<std::int32_t>& offsets = sorted_cells.second;

      // Loop over sorted array of unique cells
      for (std::size_t j = 0; j < unique_cells.size(); ++j)
      {

        std::int32_t linked_cell = unique_cells[j];
        // Extract indices of all occurances of cell in the unsorted cell
        // array
        auto indices
            = xtl::span(perm.data() + offsets[j], offsets[j + 1] - offsets[j]);
        // Extract local dofs
        const tcb::span<const int> x_dofs = x_dofmap.links(linked_cell);
        const std::size_t num_dofs_g = x_dofs.size();
        xt::xtensor<double, 2> coordinate_dofs
            = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});
        for (std::size_t k = 0; k < num_dofs_g; ++k)
        {

          std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[k]), gdim,
                      std::next(coordinate_dofs.begin(), k * gdim));
        }
        // Extract all physical points Pi(x) on a facet of linked_cell
        auto qp = xt::view(q_points, xt::keep(indices), xt::all());
        // Compute values of basis functions for all y = Pi(x) in qp
        xt::xtensor<double, 3> test_fn = dolfinx_contact::get_basis_functions(
            J, K, detJ, qp, coordinate_dofs, linked_cell,
            permutation_info[linked_cell], element, cmap);

        // Insert basis function values into c
        for (std::int32_t k = 0; k < ndofs; k++)
          for (std::size_t q = 0; q < test_fn.shape(0); ++q)
            for (std::size_t l = 0; l < bs; l++)
              c[i * cstride + j * ndofs * bs * num_q_points
                + k * bs * num_q_points + indices[q] * bs + l]
                  = test_fn(q, k * bs + l, l);
      }
    }

    return {std::move(c), cstride};
  }

  /// Compute function on opposite surface at quadrature points of
  /// facets
  /// @param[in] pair - index of contact pair
  /// @param[in] - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_u_contact(int pair,
                 std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
                 const xtl::span<const PetscScalar> gap)
  {
    int puppet_mt = _contact_pairs[pair][0];
    int candidate_mt = _contact_pairs[pair][1];
    dolfinx::common::Timer t("Pack contact u");
    // Mesh info
    dolfinx_contact::SubMesh submesh = _submeshes[candidate_mt];
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = submesh.mesh(); // mesh
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    const std::size_t bs_element = _V->element()->block_size();

    // Select which side of the contact interface to loop from and get the
    // correct map
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
        = _facet_maps[pair];
    const std::vector<xt::xtensor<double, 2>>& qp_phys = _qp_phys[puppet_mt];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
        = submesh.facet_map();
    const std::size_t num_facets = _cell_facet_pairs[puppet_mt].size();
    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
    auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace>(
        submesh.create_functionspace(_V));
    dolfinx::fem::Function<PetscScalar> u_sub(V_sub);
    std::shared_ptr<const dolfinx::fem::DofMap> sub_dofmap = V_sub->dofmap();
    assert(sub_dofmap);
    const int bs_dof = sub_dofmap->bs();

    std::array<std::size_t, 3> b_shape
        = evaulate_basis_shape(*V_sub, num_facets * num_q_points);
    xt::xtensor<double, 3> basis_values(b_shape);
    std::fill(basis_values.begin(), basis_values.end(), 0);
    std::vector<std::int32_t> cells(num_facets * num_q_points, -1);
    {
      // Copy function from parent mesh
      submesh.copy_function(*u, u_sub);

      xt::xtensor<double, 2> points
          = xt::zeros<double>({num_facets * num_q_points, gdim});
      for (std::size_t i = 0; i < num_facets; ++i)
      {
        const tcb::span<const int> links = map->links(i);
        assert(links.size() == num_q_points);
        for (std::size_t q = 0; q < num_q_points; ++q)
        {
          const std::size_t row = i * num_q_points;
          for (std::size_t j = 0; j < gdim; ++j)
          {
            points(row + q, j)
                = qp_phys[i](q, j) + gap[row * gdim + q * gdim + j];
            const tcb::span<const int> linked_pair = facet_map->links(links[q]);
            cells[row + q] = linked_pair[0];
          }
        }
      }

      evaluate_basis_functions(*u_sub.function_space(), points, cells,
                               basis_values);
    }

    const xtl::span<const PetscScalar>& u_coeffs = u_sub.x()->array();

    // Output vector
    std::vector<PetscScalar> c(num_facets * num_q_points * bs_element, 0.0);

    // Create work vector for expansion coefficients
    const int cstride = int(num_q_points * bs_element);
    const std::size_t num_basis_functions = basis_values.shape(1);
    const std::size_t value_size = basis_values.shape(2);
    std::vector<PetscScalar> coefficients(num_basis_functions * bs_element);
    for (std::size_t i = 0; i < num_facets; ++i)
    {
      for (std::size_t q = 0; q < num_q_points; ++q)
      {
        // Get degrees of freedom for current cell
        xtl::span<const std::int32_t> dofs
            = sub_dofmap->cell_dofs(cells[i * num_q_points + q]);
        for (std::size_t j = 0; j < dofs.size(); ++j)
          for (int k = 0; k < bs_dof; ++k)
            coefficients[bs_dof * j + k] = u_coeffs[bs_dof * dofs[j] + k];

        // Compute expansion
        for (std::size_t k = 0; k < bs_element; ++k)
        {
          for (std::size_t l = 0; l < num_basis_functions; ++l)
          {
            for (std::size_t m = 0; m < value_size; ++m)
            {
              c[cstride * i + q * bs_element + k]
                  += coefficients[bs_element * l + k]
                     * basis_values(num_q_points * i + q, l, m);
            }
          }
        }
      }
    }
    t.stop();
    return {std::move(c), cstride};
  }

  /// Compute inward surface normal at Pi(x)
  /// @param[in] pair - index of contact pair
  /// @param[in] gap - gap function: Pi(x)-x packed at quadrature points,
  /// where Pi(x) is the chosen projection of x onto the contact surface of
  /// the body coming into contact
  /// @param[out] c - normals ny packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_ny(int pair, const xtl::span<const PetscScalar> gap);

  /// Pack gap with rigid surface defined by x[gdim-1] = -g.
  /// g_vec = zeros(gdim), g_vec[gdim-1] = -g
  /// Gap = x - g_vec
  /// @param[in] pair - index of contact pair
  /// @param[in] g - defines location of plane
  /// @param[out] c - gap packed on facets. c[i, gdim * k+ j] contains the
  /// jth component of the Gap on the ith facet at kth quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap_plane(int pair, double g)
  {
    int puppet_mt = _contact_pairs[pair][0];
    // Mesh info
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
    assert(mesh);

    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;

    // Create _qp_ref_facet (quadrature points on reference facet)
    const dolfinx::mesh::CellType cell_type = mesh->topology().cell_type();
    dolfinx_cuas::QuadratureRule facet_quadrature(cell_type, _quadrature_degree,
                                                  fdim);
    _qp_ref_facet = facet_quadrature.points();
    _qw_ref_facet = facet_quadrature.weights();

    // Tabulate basis function on reference cell (_phi_ref_facets)
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
    _phi_ref_facets = tabulate(cmap, cell_type, facet_quadrature);

    // Compute quadrature points on physical facet _qp_phys_"puppet_mt"
    create_q_phys(puppet_mt);
    const std::vector<xt::xtensor<double, 2>> qp_phys = _qp_phys[puppet_mt];

    const std::size_t num_facets = _cell_facet_pairs[puppet_mt].size();
    // FIXME: This does not work for prism meshes
    std::size_t num_q_point = _qp_ref_facet[0].shape(0);
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = (int)num_q_point * gdim;
    for (std::size_t i = 0; i < num_facets; i++)
    {
      int offset = (int)i * cstride;
      for (std::size_t k = 0; k < num_q_point; k++)
        c[offset + (k + 1) * gdim - 1] = g - qp_phys[i](k, gdim - 1);
    }
    return {std::move(c), cstride};
  }

private:
  int _quadrature_degree = 3;
  std::vector<int> _surfaces; // meshtag values for surfaces
  // store index of candidate_surface for each puppet_surface
  std::vector<std::array<int, 2>> _contact_pairs;
  std::shared_ptr<dolfinx::fem::FunctionSpace> _V; // Function space
  // _facets_maps[i] = adjacency list of closest facet on candidate surface
  // for every quadrature point in _qp_phys[i] (quadrature points on every
  // facet of ith surface)
  std::vector<
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>>
      _facet_maps;
  //  _qp_phys[i] contains the quadrature points on the physical facets for
  //  each facet on ith surface in _surfaces
  std::vector<std::vector<xt::xtensor<double, 2>>> _qp_phys;
  // quadrature points on reference facet
  std::vector<xt::xarray<double>> _qp_ref_facet;
  // quadrature weights
  std::vector<std::vector<double>> _qw_ref_facet;
  // quadrature points on facets of reference cell
  std::vector<xt::xtensor<double, 2>> _phi_ref_facets;
  // maximum number of cells linked to a cell on ith surface
  std::vector<std::size_t> _max_links;
  // submeshes for contact surface
  std::vector<SubMesh> _submeshes;
  // facets as (cell, facet) pairs
  std::vector<std::vector<std::pair<std::int32_t, int>>> _cell_facet_pairs;
};
} // namespace dolfinx_contact
