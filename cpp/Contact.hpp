// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_contact/geometric_quantities.h>
#include <dolfinx_contact/utils.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/math.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <iostream>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtl/xspan.hpp>

using contact_kernel_fn = std::function<void(
    std::vector<std::vector<PetscScalar>>&, const double*, const double*,
    const double*, const int*, const std::uint8_t*, const std::int32_t)>;
namespace dolfinx_contact
{
enum Kernel
{
  Rhs,
  Jac
};
class Contact
{
public:
  /// Constructor
  /// @param[in] marker The meshtags defining the contact surfaces
  /// @param[in] surfaces Array of the values of the meshtags marking the
  /// surfaces
  /// @param[in] V The functions space
  Contact(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker,
          std::array<int, 2> surfaces,
          std::shared_ptr<dolfinx::fem::FunctionSpace> V)
      : _marker(marker), _surfaces(surfaces), _V(V)
  {
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet
    const dolfinx::mesh::Topology& topology = mesh->topology();
    auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    _facets[0] = marker->find(surfaces[0]);
    _facets[1] = marker->find(surfaces[1]);

    auto get_cell_indices = [c_to_f, f_to_c](std::vector<std::int32_t> facets)
    {
      int num_facets = facets.size();
      xt::xtensor<std::int32_t, 2> cell_facet_pairs
          = xt::zeros<std::int32_t>({num_facets, 2});
      for (int i = 0; i < num_facets; i++)
      {
        std::int32_t facet = facets[i];
        auto cells = f_to_c->links(facet);
        const std::int32_t cell = cells[0];
        // Find local facet index
        auto local_facets = c_to_f->links(cell);
        const auto it
            = std::find(local_facets.begin(), local_facets.end(), facet);
        assert(it != local_facets.end());
        const int facet_index = std::distance(local_facets.begin(), it);
        cell_facet_pairs(i, 0) = cell;
        cell_facet_pairs(i, 1) = facet_index;
      }
      return cell_facet_pairs;
    };

    // Replace with loop once it is generalized to an arbitrary number of
    // surfaces
    _cell_facet_pairs[0] = get_cell_indices(_facets[0]);
    _cell_facet_pairs[1] = get_cell_indices(_facets[1]);
  }

  /// Return facets belonging to surface with index surface
  /// @param[in] surface - the index of the surface
  const std::vector<int32_t>& facets(int surface) const
  {
    return _facets[surface];
  }

  /// Return meshtag value for surface with index surface
  /// @param[in] surface - the index of the surface
  const int surface_mt(int surface) const { return _surfaces[surface]; }

  // return quadrature degree
  const int quadrature_degree() const { return _quadrature_degree; }
  void set_quadrature_degree(int deg) { _quadrature_degree = deg; }

  /// return distance map (adjacency map mapping quadrature points on surface
  /// to closest facet on other surface)
  /// @param[in] surface - index of the surface
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
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

  // Return meshtags
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> meshtags() const
  {
    return _marker;
  }

  Mat create_matrix(const dolfinx::fem::Form<PetscScalar>& a, std::string type)
  {

    // Build standard sparsity pattern
    dolfinx::la::SparsityPattern pattern
        = dolfinx::fem::create_sparsity_pattern(a);

    // facet to cell connectivity
    auto mesh = _marker->mesh();
    auto dofmap = a.function_spaces().at(0)->dofmap();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    // assumes same number of quadrature points on each facet
    const std::int32_t num_qp = _cell_maps[0].shape(1);
    for (int s = 0; s < 2; ++s)
    {
      for (std::int32_t i = 0; i < _cell_facet_pairs[s].shape(0); i++)
      {
        auto cell = _cell_facet_pairs[s](i, 0);
        auto cell_dofs = dofmap->cell_dofs(cell);
        std::vector<std::int32_t> linked_dofs;
        for (std::int32_t j = 0; j < num_qp; j++)
        {
          auto linked_cell = _cell_maps[s](i, j, 0);
          auto linked_cell_dofs = dofmap->cell_dofs(linked_cell);
          for (std::int32_t k = 0; k < linked_cell_dofs.size(); k++)
          {
            linked_dofs.push_back(linked_cell_dofs[k]);
          }
        }
        // Remove duplicates
        std::sort(linked_dofs.begin(), linked_dofs.end());
        linked_dofs.erase(std::unique(linked_dofs.begin(), linked_dofs.end()),
                          linked_dofs.end());

        pattern.insert(cell_dofs, linked_dofs);
        pattern.insert(linked_dofs, cell_dofs);
      }
    }
    // Finalise communication
    pattern.assemble();

    return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
  }

  /// Assemble matrix over exterior facets
  /// Provides easier interface to dolfinx::fem::impl::assemble_exterior_facets
  /// @param[in] mat_set the function for setting the values in the matrix
  /// @param[in] V the function space
  /// @param[in] active_facets list of indices (local to process) of facets to
  /// be integrated over
  /// @param[in] kernel the custom integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_matrix(
      const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                              const std::int32_t*, const PetscScalar*)>&
          mat_set,
      const std::vector<
          std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
      int origin_meshtag, contact_kernel_fn& kernel,
      const xtl::span<const PetscScalar> coeffs, int cstride,
      const xtl::span<const PetscScalar>& constants)
  {
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = mesh->geometry().x();

    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::int32_t ndofs_cell = dofmap->cell_dofs(0).size();
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
    const int bs = dofmap->bs();

    // FIXME: Need to reconsider facet permutations for jump integrals
    std::uint8_t perm = 0;
    std::size_t max_links = std::max(_max_links[0], _max_links[1]);
    auto active_facets = _cell_facet_pairs[origin_meshtag];
    auto map = _cell_maps[origin_meshtag];
    // Data structures used in assembly
    std::vector<double> coordinate_dofs(3 * num_dofs_g);
    std::vector<std::vector<PetscScalar>> Ae_vec(
        3 * max_links + 1,
        std::vector<PetscScalar>(bs * ndofs_cell * bs * ndofs_cell));
    for (int i = 0; i < active_facets.shape(0); i++)
    {
      auto cell = active_facets(i, 0);
      // Get cell coordinates/geometry
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), gdim,
                    std::next(coordinate_dofs.begin(), i * gdim));
      }
      std::int32_t local_index = active_facets(i, 1);
      std::vector<std::int32_t> linked_cells;
      std::int32_t num_qp = map.shape(1);
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        linked_cells.push_back(map(i, j, 0));
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      const int num_linked_cells = linked_cells.size();

      std::fill(Ae_vec[0].begin(), Ae_vec[0].end(), 0);
      for (std::int32_t j = 0; j < num_linked_cells; j++)
      {
        std::fill(Ae_vec[3 * j + 1].begin(), Ae_vec[3 * j + 1].end(), 0);
        std::fill(Ae_vec[3 * j + 2].begin(), Ae_vec[3 * j + 2].end(), 0);
        std::fill(Ae_vec[3 * j + 3].begin(), Ae_vec[3 * j + 3].end(), 0);
      }

      kernel(Ae_vec, coeffs.data() + i * cstride, constants.data(),
             coordinate_dofs.data(), &local_index, &perm, num_linked_cells);

      // NOTE: Normally dof transform needs to be applied to the elements in
      // Ae_vec at this stage This is not need for the function spaces we
      // currently consider
      auto dmap_cell = dofmap->cell_dofs(cell);
      mat_set(dmap_cell.size(), dmap_cell.data(), dmap_cell.size(),
              dmap_cell.data(), Ae_vec[0].data());

      for (std::int32_t j = 0; j < num_linked_cells; j++)
      {
        auto dmap_linked = dofmap->cell_dofs(linked_cells[j]);
        mat_set(dmap_cell.size(), dmap_cell.data(), dmap_linked.size(),
                dmap_linked.data(), Ae_vec[3 * j + 1].data());
        mat_set(dmap_linked.size(), dmap_linked.data(), dmap_cell.size(),
                dmap_cell.data(), Ae_vec[3 * j + 2].data());
        mat_set(dmap_linked.size(), dmap_linked.data(), dmap_linked.size(),
                dmap_linked.data(), Ae_vec[3 * j + 3].data());
      }
    }
  }

  void assemble_vector(xtl::span<PetscScalar> b, int origin_meshtag,
                       contact_kernel_fn& kernel,
                       const xtl::span<const PetscScalar> coeffs, int cstride,
                       const xtl::span<const PetscScalar>& constants)
  {
    // Extract mesh
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = mesh->geometry().x();

    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::int32_t ndofs_cell = dofmap->cell_dofs(0).size();
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
    const int bs = dofmap->bs();

    // FIXME: Need to reconsider facet permutations for jump integrals
    std::uint8_t perm = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    auto active_facets = _cell_facet_pairs[origin_meshtag];
    auto map = _cell_maps[origin_meshtag];
    std::size_t max_links = std::max(_max_links[0], _max_links[1]);
    // Data structures used in assembly
    std::vector<double> coordinate_dofs(3 * num_dofs_g);
    std::vector<std::vector<PetscScalar>> be_vec(
        max_links + 1, std::vector<PetscScalar>(bs * ndofs_cell));

    for (int i = 0; i < active_facets.shape(0); i++)
    {
      auto cell = active_facets(i, 0);
      // Get cell coordinates/geometry
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), gdim,
                    std::next(coordinate_dofs.begin(), i * gdim));
      }
      std::int32_t local_index = active_facets(i, 1);
      std::vector<std::int32_t> linked_cells;
      std::int32_t num_qp = map.shape(1);
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        linked_cells.push_back(map(i, j, 0));
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      const int num_linked_cells = linked_cells.size();
      std::fill(be_vec[0].begin(), be_vec[0].end(), 0);
      for (std::int32_t j = 0; j < num_linked_cells; j++)
      {
        std::fill(be_vec[j + 1].begin(), be_vec[j + 1].end(), 0);
      }
      kernel(be_vec, coeffs.data() + i * cstride, constants.data(),
             coordinate_dofs.data(), &local_index, &perm, num_linked_cells);
      // NOTE: Normally dof transform needs to be applied to the elements in
      // Ae_vec at this stage This is not need for the function spaces we
      // currently consider

      // Add element vector to global vector
      auto dofs_cell = dofmap->cell_dofs(cell);
      for (int j = 0; j < ndofs_cell; ++j)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs_cell[j] + k] += be_vec[0][bs * j + k];
      for (int l = 0; l < num_linked_cells; ++l)
      {
        auto dofs_linked = dofmap->cell_dofs(linked_cells[l]);
        for (int j = 0; j < ndofs_cell; ++j)
          for (int k = 0; k < bs; ++k)
            b[bs * dofs_linked[j] + k] += be_vec[l + 1][bs * j + k];
      }
    }
  }

  contact_kernel_fn generate_kernel(dolfinx_contact::Kernel type)
  {
    // mesh data
    auto mesh = _marker->mesh();
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    const std::size_t tdim = mesh->topology().dim(); // topological dimension
    const std::size_t fdim = tdim - 1;
    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
    const std::size_t bs = dofmap->bs();

    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
    int max_links = std::max(_max_links[0], _max_links[1]);

    // Create coordinate elements (for facet and cell) _marker->mesh()
    const basix::FiniteElement basix_element
        = dolfinx_cuas::mesh_to_basix_element(mesh, tdim);
    const int num_coordinate_dofs = basix_element.dim();
    // Structures needed for basis function tabulation
    // phi and grad(phi) at quadrature points
    std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
    auto facets = basix::cell::topology(
        basix_element.cell_type())[tdim - 1]; // Topology of basix facets
    const std::uint32_t num_facets = facets.size();
    std::vector<xt::xtensor<double, 2>> phi;
    phi.reserve(num_facets);
    std::vector<xt::xtensor<double, 3>> dphi;
    phi.reserve(num_facets);
    std::vector<xt ::xtensor<double, 3>> dphi_c;
    dphi_c.reserve(num_facets);

    for (std::size_t i = 0; i < num_facets; ++i)
    {
      // Push quadrature points forward
      auto facet = facets[i];
      const xt::xarray<double>& q_facet = _qp_ref_facet[i];

      xt::xtensor<double, 4> cell_tab({tdim + 1, num_q_points, ndofs_cell, bs});
      element->tabulate(cell_tab, q_facet, 1);
      xt::xtensor<double, 2> phi_i
          = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      phi.push_back(phi_i);
      xt::xtensor<double, 3> dphi_i
          = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
      dphi.push_back(dphi_i);

      // Tabulate coordinate element of reference cell
      auto c_tab = basix_element.tabulate(1, q_facet);
      xt::xtensor<double, 3> dphi_ci
          = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
      dphi_c.push_back(dphi_ci);
    }
    // coefficient offsets
    // expecting coefficients in following order:
    // mu, lmbda, h, gap, normals, test_fn, u, u_opposite
    // mu, lmbda, h - DG0 one value per  facet
    // gap, normals, test_fn,  u_opposite
    // packed at quadrature points
    // u packed at dofs
    // mu, lmbda, h scalar
    // gap vector valued gdim
    // test_fn, u, u_opposite vector valued bs (should be bs = gdim)
    const std::size_t num_coeffs = 8;
    std::vector<std::size_t> cstrides
        = {1,
           1,
           1,
           num_q_points * gdim,
           num_q_points * gdim,
           num_q_points * ndofs_cell * bs * max_links,
           ndofs_cell * bs,
           num_q_points * bs};
    // As reference facet and reference cell are affine, we do not need to
    // compute this per quadrature point
    auto ref_jacobians
        = basix::cell::facet_jacobians(basix_element.cell_type());

    // Get facet normals on reference cell
    auto facet_normals
        = basix::cell::facet_outward_normals(basix_element.cell_type());

    // right hand side kernel
    contact_kernel_fn unbiased_rhs
        = [=](std::vector<std::vector<PetscScalar>>& b, const double* c,
              const double* w, const double* coordinate_dofs,
              const int* entity_local_index,
              const std::uint8_t* quadrature_permutation,
              const std::int32_t num_links)
    {
      // assumption that the vector function space has block size tdim
      assert(bs == gdim);
      std::size_t facet_index = size_t(*entity_local_index);

      // Reshape coordinate dofs to two dimensional array
      // NOTE: DOLFINx has 3D input coordinate dofs
      std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

      // FIXME: These array should be views (when compute_jacobian doesn't use
      // xtensor)
      xt::xtensor<double, 2> coord = xt::adapt(
          coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

      // Extract the first derivative of the coordinate element (cell) of
      // degrees of freedom on the facet
      const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
      const xt::xtensor<double, 2>& dphi0_c = xt::view(
          dphi_fc, xt::all(), 0,
          xt::all()); // FIXME: Assumed constant, i.e. only works for simplices
      // NOTE: Affine cell assumption
      // Compute Jacobian and determinant at first quadrature point
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
      auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
      dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
      dolfinx_cuas::math::compute_inv(J, K);

      // Compute normal of physical facet using a normalized covariant Piola
      // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
      // DOI: 10.1137/08073901X
      xt::xarray<double> n_phys = xt::zeros<double>({gdim});
      auto facet_normal = xt::row(facet_normals, facet_index);
      for (std::size_t i = 0; i < gdim; i++)
        for (std::size_t j = 0; j < tdim; j++)
          n_phys[i] += K(j, i) * facet_normal[j];
      double n_norm = 0;
      for (std::size_t i = 0; i < gdim; i++)
        n_norm += n_phys[i] * n_phys[i];
      n_phys /= std::sqrt(n_norm);

      double gamma = c[2] / w[0]; // This is h/gamma,  gamma = w[0];
      double gamma_inv = w[0] / c[2];
      double theta = w[1];
      double mu = c[0];
      double lmbda = c[1];

      // Compute det(J_C J_f) as it is the mapping to the reference facet
      xt::xtensor<double, 2> J_f
          = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
      xt::xtensor<double, 2> J_tot
          = xt::zeros<double>({J.shape(0), J_f.shape(1)});
      dolfinx::math::dot(J, J_f, J_tot);
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

      const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
      const xt::xtensor<double, 2>& phi_f = phi[facet_index];
      // const std::vector<double>& weights = _qw_ref_facet[facet_index];
      xt::xarray<double> n_surf = xt::zeros<double>({gdim});
      std::size_t num_points = _qw_ref_facet[facet_index].size();
      for (std::size_t q = 0; q < num_points; q++)
      {
        double n_dot = 0;
        double gap = 0;
        const std::size_t gap_offset = 3;
        const std::size_t normal_offset = gap_offset + cstrides[3];
        for (int i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf(i) = -c[normal_offset + q * gdim + i];
          n_dot += n_phys(i) * n_surf(i);
          gap += c[gap_offset + q * gdim + i] * n_surf(i);
        }

        xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
        xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
        // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            for (int k = 0; k < tdim; k++)
            {
              tr(j, l) += K(k, l) * dphi_f(k, q, j);
              for (int s = 0; s < gdim; s++)
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
        std::size_t offset_u = cstrides[0] + cstrides[1] + cstrides[2]
                               + cstrides[3] + cstrides[4] + cstrides[5];

        for (int i = 0; i < ndofs_cell; i++)
        {
          std::size_t block_index = offset_u + i * bs;
          for (int j = 0; j < bs; j++)
          {
            tr_u += c[block_index + j] * tr(i, j);
            epsn_u += c[block_index + j] * epsn(i, j);
            jump_un += c[block_index + j] * phi_f(q, i) * n_surf(j);
          }
        }
        std::size_t offset_u_opp = offset_u + cstrides[6] + q * bs;
        for (int j = 0; j < bs; ++j)
          jump_un += -c[offset_u_opp + j] * n_surf(j);
        double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
        const double w0 = _qw_ref_facet[facet_index][q] * detJ;
        double Pn_u
            = dolfinx_contact::R_plus((jump_un - gap) - gamma * sign_u) * w0;
        sign_u *= w0;
        // Fill contributions of facet with itself

        for (int i = 0; i < ndofs_cell; i++)
        {
          for (int n = 0; n < bs; n++)
          {
            double v_dot_nsurf = n_surf(n) * phi_f(q, i);
            double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
            // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
            double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
            b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;
            // 0.5 * (-theta * gamma * sign_v * sign_u + Pn_u * Pn_v);

            // entries corresponding to v on the other surface
            for (int k = 0; k < num_links; k++)
            {
              int index = 3 + cstrides[3] + cstrides[4]
                          + k * num_q_points * ndofs_cell * bs
                          + i * num_q_points * bs + q * bs + n;
              double v_n_opp = c[index] * n_surf(n);

              b[k + 1][n + i * bs] -= 0.5 * gamma_inv * v_n_opp * Pn_u;
            }
          }
        }
      }
    };

    // jacobian kernel
    contact_kernel_fn unbiased_jac
        = [=](std::vector<std::vector<PetscScalar>>& A, const double* c,
              const double* w, const double* coordinate_dofs,
              const int* entity_local_index,
              const std::uint8_t* quadrature_permutation,
              const std::int32_t num_links)
    {
      // assumption that the vector function space has block size tdim
      assert(bs == gdim);
      std::size_t facet_index = size_t(*entity_local_index);

      // Reshape coordinate dofs to two dimensional array
      // NOTE: DOLFINx has 3D input coordinate dofs
      std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

      // FIXME: These array should be views (when compute_jacobian doesn't use
      // xtensor)
      xt::xtensor<double, 2> coord = xt::adapt(
          coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

      // Extract the first derivative of the coordinate element (cell) of
      // degrees of freedom on the facet
      const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
      const xt::xtensor<double, 2>& dphi0_c = xt::view(
          dphi_fc, xt::all(), 0,
          xt::all()); // FIXME: Assumed constant, i.e. only works for simplices
      // NOTE: Affine cell assumption
      // Compute Jacobian and determinant at first quadrature point
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
      auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
      dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
      dolfinx_cuas::math::compute_inv(J, K);

      // Compute normal of physical facet using a normalized covariant Piola
      // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
      // DOI: 10.1137/08073901X
      xt::xarray<double> n_phys = xt::zeros<double>({gdim});
      auto facet_normal = xt::row(facet_normals, facet_index);
      for (std::size_t i = 0; i < gdim; i++)
        for (std::size_t j = 0; j < tdim; j++)
          n_phys[i] += K(j, i) * facet_normal[j];
      double n_norm = 0;
      for (std::size_t i = 0; i < gdim; i++)
        n_norm += n_phys[i] * n_phys[i];
      n_phys /= std::sqrt(n_norm);

      double gamma = c[2] / w[0]; // This is h/gamma,  gamma = w[0];
      double gamma_inv = w[0] / c[2];

      double theta = w[1];
      double mu = c[0];
      double lmbda = c[1];

      // Compute det(J_C J_f) as it is the mapping to the reference facet
      xt::xtensor<double, 2> J_f
          = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
      xt::xtensor<double, 2> J_tot
          = xt::zeros<double>({J.shape(0), J_f.shape(1)});
      dolfinx::math::dot(J, J_f, J_tot);
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

      const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
      const xt::xtensor<double, 2>& phi_f = phi[facet_index];
      // const std::vector<double>& weights = _qw_ref_facet[facet_index];
      xt::xarray<double> n_surf = xt::zeros<double>({gdim});
      std::size_t num_points = _qw_ref_facet[facet_index].size();
      for (std::size_t q = 0; q < num_points; q++)
      {
        double n_dot = 0;
        double gap = 0;
        const std::size_t gap_offset = 3;
        const std::size_t normal_offset = gap_offset + cstrides[3];
        for (int i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf(i) = -c[normal_offset + q * gdim + i];
          n_dot += n_phys(i) * n_surf(i);
          gap += c[gap_offset + q * gdim + i] * n_surf(i);
        }

        xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
        xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
        // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            for (int k = 0; k < tdim; k++)
            {
              tr(j, l) += K(k, l) * dphi_f(k, q, j);
              for (int s = 0; s < gdim; s++)
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
        std::size_t offset_u = cstrides[0] + cstrides[1] + cstrides[2]
                               + cstrides[3] + cstrides[4] + cstrides[5];
        for (int i = 0; i < ndofs_cell; i++)
        {
          std::size_t block_index = offset_u + i * bs;
          for (int j = 0; j < bs; j++)
          {
            tr_u += c[block_index + j] * tr(i, j);
            epsn_u += c[block_index + j] * epsn(i, j);
            jump_un += c[block_index + j] * phi_f(q, i) * n_surf(j);
          }
        }
        std::size_t offset_u_opp = offset_u + cstrides[6] + q * bs;
        for (int j = 0; j < bs; ++j)
          jump_un += -c[offset_u_opp + j] * n_surf(j);
        double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
        double Pn_u
            = dolfinx_contact::dR_plus((jump_un - gap) - gamma * sign_u);

        // Fill contributions of facet with itself
        const double w0 = _qw_ref_facet[facet_index][q] * detJ;
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
            double Pn_du
                = (phi_f(q, j) * n_surf(l) - gamma * sign_du) * Pn_u * w0;

            sign_du *= w0;
            for (int i = 0; i < ndofs_cell; i++)
            {
              for (int b = 0; b < bs; b++)
              {
                double v_dot_nsurf = n_surf(b) * phi_f(q, i);
                double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
                double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
                A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * Pn_du * Pn_v;
                // 0.5 * (-theta * gamma * sign_du * sign_v + Pn_du * Pn_v);

                // entries corresponding to u and v on the other surface
                for (int k = 0; k < num_links; k++)
                {
                  int index = 3 + cstrides[3] + cstrides[4]
                              + k * num_q_points * ndofs_cell * bs
                              + j * num_q_points * bs + q * bs + l;
                  double du_n_opp = c[index] * n_surf(l);

                  du_n_opp *= w0 * Pn_u;
                  index = 3 + cstrides[3] + cstrides[4]
                          + k * num_q_points * ndofs_cell * bs
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
      throw std::runtime_error("Unrecognized kernel");
    }
  }
  /// Tabulate the basis function at the quadrature points _qp_ref_facet
  /// creates and fills _phi_ref_facets
  std::vector<xt::xtensor<double, 2>>
  tabulate_on_ref_cell(basix::FiniteElement element)
  {

    // Create _phi_ref_facets
    std::uint32_t num_facets = _qp_ref_facet.size();
    std::uint32_t num_local_dofs = element.dim();
    std::vector<xt::xtensor<double, 2>> phi;
    phi.reserve(num_facets);

    // Tabulate basis functions at quadrature points _qp_ref_facet for each
    // facet of the reference cell. Fill _phi_ref_facets
    for (int i = 0; i < num_facets; ++i)
    {
      auto cell_tab = element.tabulate(0, _qp_ref_facet[i]);
      const xt::xtensor<double, 2> _phi_i
          = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      phi.push_back(_phi_i);
    }
    return phi;
  }

  /// Compute push forward of quadrature points _qp_ref_facet to the physical
  /// facet for each facet in _facet_"origin_meshtag" Creates and fills
  /// _qp_phys_"origin_meshtag"
  /// @param[in] origin_meshtag flag to choose the surface
  void create_q_phys(int origin_meshtag)
  {
    // Mesh info
    auto mesh = _marker->mesh();
    xtl::span<const double> mesh_geometry = mesh->geometry().x();
    auto cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    auto puppet_facets = _cell_facet_pairs[origin_meshtag];
    _qp_phys[origin_meshtag].reserve(puppet_facets.shape(0));
    _qp_phys[origin_meshtag].clear();
    // push forward of quadrature points _qp_ref_facet to physical facet for
    // each facet in _facet_"origin_meshtag"
    for (int i = 0; i < puppet_facets.shape(0); ++i)
    {
      auto cell = puppet_facets(i, 0); // extract cell
      const std::int32_t local_index = puppet_facets(i, 1);

      // extract local dofs
      auto x_dofs = x_dofmap.links(cell);
      const std::size_t num_dofs_g = x_dofmap.num_links(cell);
      xt::xtensor<double, 2> coordinate_dofs
          = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[i]), gdim,
                    std::next(coordinate_dofs.begin(), i * gdim));
      }
      xt::xtensor<double, 2> q_phys({_qp_ref_facet[local_index].shape(0),
                                     _qp_ref_facet[local_index].shape(1)});

      // push forward of quadrature points _qp_ref_facet to the physical facet
      cmap.push_forward(q_phys, coordinate_dofs, _phi_ref_facets[local_index]);
      _qp_phys[origin_meshtag].push_back(q_phys);
    }
  }
  /// Compute maximum number of links
  /// I think this should actually be part of create_distance_map
  /// which should be easier after the rewrite of contact
  /// It is therefore called inside create_distance_map
  void max_links(int origin_meshtag)
  {
    // facet to cell connectivity
    auto mesh = _marker->mesh();
    auto dofmap = _V->dofmap();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    std::size_t max_links = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    auto active_facets = _cell_facet_pairs[origin_meshtag];
    auto map = _cell_maps[origin_meshtag];

    std::int32_t num_qp = map.shape(1);
    for (std::int32_t i = 0; i < active_facets.shape(0); i++)
    {
      auto cell = active_facets(i, 0);
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::vector<std::int32_t> linked_cells;
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        linked_cells.push_back(map(i, j, 0));
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      max_links = std::max(max_links, linked_cells.size());
    }
    _max_links[origin_meshtag] = max_links;
  }
  /// Compute closest candidate_facet for each quadrature point in
  /// _qp_phys[origin_meshtag]
  /// This is saved as an adjacency list in _facet_maps[origin_meshtag]
  /// and an xtensor containing cell_facet_pairs in  _cell_maps[origin_mesthtag]
  void create_distance_map(int puppet_mt, int candidate_mt)
  {

    // Mesh info
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;

    // Create _qp_ref_facet (quadrature points on reference facet)
    dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(),
                                        _quadrature_degree, fdim);
    _qp_ref_facet = q_rule.points();
    _qw_ref_facet = q_rule.weights();

    // Tabulate basis function on reference cell (_phi_ref_facets)// Create
    // coordinate element
    // FIXME: For higher order geometry need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto dolfinx_cell = mesh->topology().cell_type();
    auto coordinate_element = basix::create_element(
        basix::element::family::P,
        dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
        basix::element::lagrange_variant::gll_warped);
    _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);

    // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
    create_q_phys(puppet_mt);

    xt::xtensor<double, 2> point = xt::zeros<double>({1, 3});
    point[2] = 0;

    // assign puppet_ and candidate_facets
    auto candidate_facets = _facets[candidate_mt];
    auto puppet_facets = _facets[puppet_mt];
    auto qp_phys = _qp_phys[puppet_mt];
    // Create midpoint tree as compute_closest_entity will be called many
    // times
    dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim,
                                                   candidate_facets);
    auto master_midpoint_tree = dolfinx::geometry::create_midpoint_tree(
        *mesh, fdim, candidate_facets);

    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);
    mesh->topology_mutable().create_connectivity(tdim, fdim);
    auto c_to_f = mesh->topology().connectivity(tdim, fdim);
    auto get_cell_indices = [c_to_f, f_to_c](std::vector<std::int32_t> facets)
    {
      int num_facets = facets.size();
      xt::xtensor<std::int32_t, 2> cell_facet_pairs
          = xt::zeros<std::int32_t>({num_facets, 2});
      for (int i = 0; i < num_facets; i++)
      {
        std::int32_t facet = facets[i];
        auto cells = f_to_c->links(facet);
        const std::int32_t cell = cells[0];
        // Find local facet index
        auto local_facets = c_to_f->links(cell);
        const auto it
            = std::find(local_facets.begin(), local_facets.end(), facet);
        assert(it != local_facets.end());
        const int facet_index = std::distance(local_facets.begin(), it);
        cell_facet_pairs(i, 0) = cell;
        cell_facet_pairs(i, 1) = facet_index;
      }
      return cell_facet_pairs;
    };
    std::vector<std::int32_t> data; // will contain closest candidate facet
    std::vector<std::int32_t> offset(1);
    offset[0] = 0;
    const int num_qp = qp_phys[0].shape(0);
    const int num_facets = puppet_facets.size();
    std::vector<std::int32_t> data2(
        num_qp); // will contain closest candidate facet
    xt::xtensor<std::int32_t, 3> map
        = xt::zeros<std::int32_t>({num_facets, num_qp, 2});
    xt::xtensor<std::int32_t, 2> old_map
        = xt::zeros<std::int32_t>({num_facets, num_qp});
    for (int i = 0; i < puppet_facets.size(); ++i)
    {
      // FIXME: This does not work for prism meshes
      for (int j = 0; j < qp_phys[0].shape(0); ++j)
      {
        for (int k = 0; k < gdim; ++k)
          point(0, k) = qp_phys[i](j, k);

        // Find closest facet to point
        std::vector<std::int32_t> search_result
            = dolfinx::geometry::compute_closest_entity(
                master_bbox, master_midpoint_tree, *mesh, point);
        data.push_back(search_result[0]);
        data2[j] = search_result[0];
      }
      offset.push_back(data.size());
      xt::view(map, i, xt::all(), xt::all()) = get_cell_indices(data2);
    }
    // save maps
    _cell_maps[puppet_mt] = map;
    _facet_maps[puppet_mt]
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data,
                                                                        offset);
    max_links(puppet_mt);
  }

  /// Compute and pack the gap function for each quadrature point the set of
  /// facets. For a set of facets; go through the quadrature points on each
  /// facet find the closest facet on the other surface and compute the
  /// distance vector
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[out] c - gap packed on facets. c[i*cstride +  gdim * k+ j]
  /// contains the jth component of the Gap on the ith facet at kth quadrature
  /// point
  std::pair<std::vector<PetscScalar>, int> pack_gap(int origin_meshtag)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    xtl::span<const double> mesh_geometry = mesh->geometry().x();

    // Select which side of the contact interface to loop from and get the
    // correct map
    auto puppet_facets = _facets[origin_meshtag];
    auto map = _facet_maps[origin_meshtag];
    auto qp_phys = _qp_phys[origin_meshtag];
    const std::int32_t num_facets = puppet_facets.size();
    const std::int32_t num_q_point = _qp_ref_facet[0].shape(0);

    // Pack gap function for each quadrature point on each facet
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = num_q_point * gdim;
    xt::xtensor<double, 2> point = {{0, 0, 0}};

    for (int i = 0; i < num_facets; ++i)
    {
      auto master_facets = map->links(i);
      auto master_facet_geometry = dolfinx::mesh::entities_to_geometry(
          *mesh, fdim, master_facets, false);
      int offset = i * cstride;
      for (int j = 0; j < map->num_links(i); ++j)
      {
        // Get quadrature points in physical space for the ith facet, jth
        // quadrature point
        for (int k = 0; k < gdim; k++)
          point(0, k) = qp_phys[i](j, k);

        // Get the coordinates of the geometry on the other interface, and
        // compute the distance of the convex hull created by the points
        auto master_facet = xt::view(master_facet_geometry, j, xt::all());
        std::size_t num_facet_dofs = master_facet_geometry.shape(1);
        xt::xtensor<double, 2> master_coords
            = xt::zeros<double>({num_facet_dofs, std::size_t(3)});
        for (std::size_t l = 0; l < num_facet_dofs; ++l)
        {
          const int pos = 3 * master_facet[l];
          for (std::size_t k = 0; k < gdim; ++k)
            master_coords(l, k) = mesh_geometry[pos + k];
        }
        auto dist_vec
            = dolfinx::geometry::compute_distance_gjk(master_coords, point);

        // Add distance vector to coefficient array
        for (int k = 0; k < gdim; k++)
          c[offset + j * gdim + k] += dist_vec(k);
      }
    }
    return {std::move(c), cstride};
  }

  /// Compute test functions on opposite surface at quadrature points of
  /// facets
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[in] gap - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_test_functions(int origin_meshtag,
                      const xtl::span<const PetscScalar>& gap)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    auto cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    xtl::span<const double> mesh_geometry = mesh->geometry().x();
    auto element = _V->element();
    const std::uint32_t bs = element->block_size();
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint32_t> permutation_info
        = mesh->topology().get_cell_permutation_info();

    // Select which side of the contact interface to loop from and get the
    // correct map
    auto map = _cell_maps[origin_meshtag];
    auto qp_phys = _qp_phys[origin_meshtag];
    auto puppet_facets = _cell_facet_pairs[origin_meshtag];
    std::size_t max_links = std::max(_max_links[0], _max_links[1]);
    const std::int32_t num_facets = map.shape(0);
    const std::int32_t num_q_points = _qp_ref_facet[0].shape(0);
    const std::int32_t ndofs = _V->dofmap()->cell_dofs(0).size();
    std::vector<PetscScalar> c(
        num_facets * num_q_points * max_links * ndofs * bs, 0.0);
    const int cstride = num_q_points * max_links * ndofs * bs;
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
    for (int i = 0; i < num_facets; i++)
    {
      auto cell = puppet_facets(i, 0); // extract cell

      // Compute Pi(x) form points x and gap funtion Pi(x) - x
      for (int j = 0; j < num_q_points; j++)
      {
        linked_cells[j] = map(i, j, 0);
        for (int k = 0; k < gdim; k++)
          q_points(j, k)
              = qp_phys[i](j, k) + gap[i * gdim * num_q_points + j * gdim + k];
      }

      // Sort linked cells
      assert(map.shape(1) == num_q_points);
      assert(linked_cells.size() == num_q_points);
      std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
          sorted_cells = dolfinx_contact::sort_cells(
              xtl::span(linked_cells.data(), linked_cells.size()),
              xtl::span(perm.data(), perm.size()));
      auto unique_cells = sorted_cells.first;
      auto offsets = sorted_cells.second;

      // Loop over sorted array of unique cells
      for (int j = 0; j < unique_cells.size(); ++j)
      {

        std::int32_t linked_cell = unique_cells[j];
        // Extract indices of all occurances of cell in the unsorted cell array
        auto indices
            = xtl::span(perm.data() + offsets[j], offsets[j + 1] - offsets[j]);
        // Extract local dofs
        auto x_dofs = x_dofmap.links(linked_cell);
        const std::size_t num_dofs_g = x_dofmap.num_links(linked_cell);
        xt::xtensor<double, 2> coordinate_dofs
            = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[i]), gdim,
                      std::next(coordinate_dofs.begin(), i * gdim));
        }
        // Extract all physical points Pi(x) on a facet of linked_cell
        auto qp = xt::view(q_points, xt::keep(indices), xt::all());
        // Compute values of basis functions for all y = Pi(x) in qp
        auto test_fn = dolfinx_contact::get_basis_functions(
            J, K, detJ, qp, coordinate_dofs, linked_cell,
            permutation_info[linked_cell], element, cmap);

        // Insert basis function values into c
        for (std::size_t k = 0; k < ndofs; k++)
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
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[in] - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_u_contact(int origin_meshtag,
                 std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
                 const xtl::span<const PetscScalar> gap)
  {

    // Mesh info
    auto mesh = _marker->mesh();                     // mesh
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    const std::size_t bs = _V->element()->block_size();

    // Select which side of the contact interface to loop from and get the
    // correct map
    auto map = _cell_maps[origin_meshtag];
    auto qp_phys = _qp_phys[origin_meshtag];
    const std::size_t num_facets = map.shape(0);
    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
    std::vector<PetscScalar> c(num_facets * num_q_points * bs, 0.0);
    const int cstride = num_q_points * bs;

    xt::xtensor<double, 2> points = xt::zeros<double>({num_q_points, gdim});
    std::vector<std::int32_t> cells(num_q_points, 0);
    xt::xtensor<PetscScalar, 2> vals
        = xt::zeros<PetscScalar>({num_q_points, bs});
    for (std::size_t i = 0; i < num_facets; ++i)
    {
      for (std::size_t q = 0; q < num_q_points; ++q)
      {
        for (std::size_t j = 0; j < gdim; ++j)
        {
          points(q, j)
              = qp_phys[i](q, j) + gap[i * gdim * num_q_points + q * gdim + j];
          cells[q] = map(i, q, 0);
        }
      }
      vals.fill(0);
      u->eval(points, cells, vals);
      for (std::size_t q = 0; q < num_q_points; ++q)
      {
        for (std::size_t j = 0; j < bs; ++j)
        {
          c[i * cstride + q * bs + j] = vals(q, j);
        }
      }
    }

    return {std::move(c), cstride};
  }

  /// Compute inward surface normal at Pi(x)
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[in] gap - gap function: Pi(x)-x packed at quadrature points, where
  /// Pi(x) is the chosen projection of x onto the contact surface of the body
  /// coming into contact
  /// @param[out] c - normals ny packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_ny(int origin_meshtag, const xtl::span<const PetscScalar> gap)
  {

    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    auto cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    xtl::span<const double> mesh_geometry = mesh->geometry().x();
    auto element = _V->element();
    auto cell_type
        = dolfinx::mesh::cell_type_to_basix_type(mesh->topology().cell_type());
    // Get facet normals on reference cell
    auto facet_normals = basix::cell::facet_outward_normals(cell_type);

    // Select which side of the contact interface to loop from and get the
    // correct map
    auto map = _cell_maps[origin_meshtag];
    auto puppet_facets = _cell_facet_pairs[origin_meshtag];
    auto qp_phys = _qp_phys[origin_meshtag];
    const std::int32_t num_facets = map.shape(0);
    const std::int32_t num_q_points = _qp_ref_facet[0].shape(0);
    std::vector<PetscScalar> c(num_facets * num_q_points * gdim, 0.0);
    const int cstride = num_q_points * gdim;
    xt::xtensor<double, 2> point = xt::zeros<double>(
        {std::size_t(1), std::size_t(gdim)}); // To store Pi(x)

    // Needed for pull_back in get_facet_normals
    xt::xtensor<double, 3> J = xt::zeros<double>(
        {std::size_t(num_q_points), std::size_t(gdim), std::size_t(tdim)});
    xt::xtensor<double, 3> K = xt::zeros<double>(
        {std::size_t(1), std::size_t(tdim), std::size_t(gdim)});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({std::size_t(1)});
    xt::xtensor<std::int32_t, 1> facet_indices
        = xt::zeros<std::int32_t>({std::size_t(1)});

    // Loop over quadrature points
    for (int i = 0; i < num_facets; i++)
    {
      auto cell = puppet_facets(i, 0); // extract cell
      for (int q = 0; q < num_q_points; ++q)
      {
        // Extract linked cell and facet at quadrature point q
        std::int32_t linked_cell = map(i, q, 0);
        facet_indices(0) = map(i, q, 1);

        // Compute Pi(x) from x, and gap = Pi(x) - x
        for (int k = 0; k < gdim; ++k)
          point(0, k)
              = qp_phys[i](q, k) + gap[i * gdim * num_q_points + q * gdim + k];

        // extract local dofs
        auto x_dofs = x_dofmap.links(linked_cell);
        const std::size_t num_dofs_g = x_dofmap.num_links(linked_cell);
        xt::xtensor<double, 2> coordinate_dofs
            = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});
        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[i]), gdim,
                      std::next(coordinate_dofs.begin(), i * gdim));
        }

        // Compute outward unit normal in point = Pi(x)
        // Note: in the affine case potential gains can be made
        //       if the cells are sorted like in pack_test_functions
        assert(linked_cell >= 0);
        xt::xtensor<double, 2> normals
            = dolfinx_contact::push_forward_facet_normal(
                point, J, K, coordinate_dofs, facet_indices, cmap,
                facet_normals);

        // Copy normal into c
        for (std::size_t l = 0; l < gdim; l++)
        {
          c[i * cstride + q * gdim + l] = normals(0, l);
        }
      }
    }

    return {std::move(c), cstride};
  }

  /// Pack gap with rigid surface defined by x[gdim-1] = -g.
  /// g_vec = zeros(gdim), g_vec[gdim-1] = -g
  /// Gap = x - g_vec
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[in] g - defines location of plane
  /// @param[out] c - gap packed on facets. c[i, gdim * k+ j] contains the
  /// jth component of the Gap on the ith facet at kth quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap_plane(int origin_meshtag,
                                                          double g)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    xtl::span<const double> mesh_geometry = mesh->geometry().x();
    // Create _qp_ref_facet (quadrature points on reference facet)
    dolfinx_cuas::QuadratureRule facet_quadrature(
        _marker->mesh()->topology().cell_type(), _quadrature_degree, fdim);
    _qp_ref_facet = facet_quadrature.points();
    _qw_ref_facet = facet_quadrature.weights();

    // Tabulate basis function on reference cell (_phi_ref_facets)// Create
    // coordinate element
    // FIXME: For higher order geometry need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto dolfinx_cell = _marker->mesh()->topology().cell_type();
    auto coordinate_element = basix::create_element(
        basix::element::family::P,
        dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
        basix::element::lagrange_variant::gll_warped);

    _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);
    // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
    create_q_phys(origin_meshtag);
    auto puppet_facets = _cell_facet_pairs[origin_meshtag];
    auto qp_phys = _qp_phys[origin_meshtag];

    int32_t num_facets = puppet_facets.shape(0);
    // FIXME: This does not work for prism meshes
    int32_t num_q_point = _qp_ref_facet[0].shape(0);
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = num_q_point * gdim;
    for (int i = 0; i < num_facets; i++)
    {
      int offset = i * cstride;
      for (int k = 0; k < num_q_point; k++)
      {
        c[offset + (k + 1) * gdim - 1] = g - qp_phys[i](k, gdim - 1);
      }
    }
    return {std::move(c), cstride};
  }

private:
  int _quadrature_degree = 3;
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> _marker;
  std::array<int, 2> _surfaces; // meshtag values for surfaces
  std::shared_ptr<dolfinx::fem::FunctionSpace> _V; // Function space
  // For the ith surface in _surfaces _cell_maps[i](j, k, l) returns
  // the cell (l=0) or local facet index (l=1) of the facets closest
  // to the jth facet on the ith surface at the kth quadrature point
  std::array<xt::xtensor<std::int32_t, 3>, 2> _cell_maps;
  // _facets_maps[i] = adjacency list of closest facet on candidate surface for
  // every quadrature point in _qp_phys[i] (quadrature points on every facet of
  // ith surface)
  std::array<std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>, 2>
      _facet_maps; // FIXME: should be made redundant
  //  _qp_phys[i] contains the quadrature points on the physical facets for each
  //  facet on ith surface in _surfaces
  std::array<std::vector<xt::xtensor<double, 2>>, 2> _qp_phys;
  // quadrature points on reference facet
  std::vector<xt::xarray<double>> _qp_ref_facet;
  // quadrature weights
  std::vector<std::vector<double>> _qw_ref_facet;
  // quadrature points on facets of reference cell
  std::vector<xt::xtensor<double, 2>> _phi_ref_facets;
  // _facets[i] = facets in ith surface
  std::array<std::vector<int32_t>, 2> _facets;
  // cell_facets_pairs[i](j, k) is the cell (k=0) or local
  // facet index (k=1) of the jth facet in the ith surface
  std::array<xt::xtensor<std::int32_t, 2>, 2> _cell_facet_pairs;
  // maximum number of cells linked to a cell on ith surface
  std::array<std::size_t, 2> _max_links = {0, 0};
};
} // namespace dolfinx_contact
