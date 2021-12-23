// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xtensor.hpp>
namespace dolfinx_contact
{

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as an array, where j is the index of the local cell
/// and c[j*cstride + q * (block_size * value_size) + k + c] = sum_i c^i[k] *
/// phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth vector component,
/// phi^i(x_q)[c] is the ith basis function's c-th value compoenent at the
/// quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[in] q_degree The quadrature degree
/// @returns c The packed coefficients and the number of coeffs per cell
std::pair<std::vector<PetscScalar>, int> pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree)
{
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const dolfinx::fem::FiniteElement* element
      = coeff->function_space()->element().get();
  const xtl::span<const double>& data = coeff->x()->array();

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh
      = coeff->function_space()->mesh();
  assert(mesh);
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::int32_t num_cells
      = mesh->topology().index_map((int)tdim)->size_local()
        + mesh->topology().index_map((int)tdim)->num_ghosts();

  // Get dof transformations
  const bool needs_dof_transformations = element->needs_dof_transformations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    mesh->topology_mutable().create_entity_permutations();
  }
  const std::function<void(const xtl::span<PetscScalar>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  auto cell_type = mesh->topology().cell_type();
  const std::size_t num_dofs = element->space_dimension();
  const std::size_t bs = dofmap->bs();
  const std::size_t vs
      = element->reference_value_size() / element->block_size();

  // Tabulate function at quadrature points
  auto [points, weights] = basix::quadrature::make_quadrature(
      basix::quadrature::type::Default,
      dolfinx::mesh::cell_type_to_basix_type(cell_type), q_degree);
  const std::size_t num_points = weights.size();
  xt::xtensor<double, 4> coeff_basis({1, num_points, num_dofs, vs});
  element->tabulate(coeff_basis, points, 0);
  std::vector<PetscScalar> c(num_cells * vs * bs * num_points, 0.0);
  const auto cstride = int(vs * bs * num_points);
  auto reference_basis_values
      = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());

  if (needs_dof_transformations)
  {
    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values({num_points, num_dofs / bs, vs});
    xt::xtensor<double, 2> point_basis_values({num_dofs / bs, vs});

    // Get geometry data
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = mesh->geometry().x();

    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points, tdim});
    xt::xtensor<double, 3> J = xt::zeros<double>({num_points, gdim, tdim});
    xt::xtensor<double, 3> K = xt::zeros<double>({num_points, tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_points});
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, gdim});

    // Get coordinate map
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

    // Compute first derivative of basis function of coordinate map
    xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, points);
    xt::xtensor<double, 4> dphi_c
        = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                   xt::all(), xt::all(), xt::all());

    // Get push forward function
    using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                          xt::xall<std::size_t>, xt::xall<std::size_t>>;
    using U_t = xt::xview<decltype(point_basis_values)&, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {

      // NOTE Add two separate loops here, one for and one without dof
      // transforms

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g[pos + j];
      }
      // NOTE: This can be simplified in affine case
      for (std::size_t q = 0; q < num_points; ++q)
      {
        J.fill(0);
        xt::xtensor<double, 2> dphi
            = xt::view(dphi_c, xt::all(), q, xt::all(), 0);
        auto _J = xt::view(J, q, xt::all(), xt::all());
        cmap.compute_jacobian(dphi, coordinate_dofs, _J);
        auto _K = xt::view(K, q, xt::all(), xt::all());
        cmap.compute_jacobian_inverse(_J, _K);
        detJ[q] = cmap.compute_jacobian_determinant(_J);

        // Permute the reference values to account for the cell's orientation
        point_basis_values
            = xt::view(reference_basis_values, q, xt::all(), xt::all());
        transformation(xtl::span(point_basis_values.data(), num_dofs / bs * vs),
                       cell_info, cell, (int)vs);

        // Push basis forward to physical element
        auto _u = xt::view(basis_values, q, xt::all(), xt::all());
        auto _U = xt::view(point_basis_values, xt::all(), xt::all());
        push_forward_fn(_u, _U, _J, detJ[q], _K);
      }

      // Sum up quadrature contributions
      int offset = cstride * cell;
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = (int)bs * dofs[i];

        for (std::size_t q = 0; q < num_points; ++q)
          for (std::size_t k = 0; k < bs; ++k)
            for (std::size_t j = 0; j < vs; j++)
              c[offset + q * (bs * vs) + k + j]
                  += basis_values(q, i, j) * data[pos_v + k];
      }
    }
  }
  else
  {
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {

      // Sum up quadrature contributions
      int offset = cstride * cell;
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = (int)bs * dofs[i];

        for (std::size_t q = 0; q < num_points; ++q)
          for (std::size_t k = 0; k < bs; ++k)
            for (std::size_t j = 0; j < vs; j++)
              c[offset + q * (bs * vs) + k + j]
                  += reference_basis_values(q, i, j) * data[pos_v + k];
      }
    }
  }
  return {std::move(c), cstride};
}

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as an array, where j corresponds to the jth facet in
/// active_facets and c[j*cstride + q * (block_size * value_size) + k + c] =
/// sum_i c^i[k] * phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth
/// vector component, phi^i(x_q)[c] is the ith basis function's c-th value
/// compoenent at the quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[in] active_facets List of active facets
/// @param[in] q_degree the quadrature degree
/// @param[out] c The packed coefficients and the number of coeffs per facet
std::pair<std::vector<PetscScalar>, int> pack_coefficient_facet(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    int q_degree, const xtl::span<const std::int32_t>& active_facets)
{
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const dolfinx::fem::FiniteElement* element
      = coeff->function_space()->element().get();
  const xtl::span<const double> data = coeff->x()->array();

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh
      = coeff->function_space()->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();
  const int fdim = tdim - 1;
  const std::size_t num_facets = active_facets.size();

  // Connectivity to evaluate at quadrature points
  // FIXME: Move create_connectivity out of this function and call before
  // calling the function...
  mesh->topology_mutable().create_connectivity(fdim, tdim);
  auto f_to_c = mesh->topology().connectivity(fdim, tdim);
  mesh->topology_mutable().create_connectivity(tdim, fdim);
  auto c_to_f = mesh->topology().connectivity(tdim, fdim);

  // Get dof transformations
  const bool needs_dof_transformations = element->needs_dof_transformations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    mesh->topology_mutable().create_entity_permutations();
  }
  const std::function<void(const xtl::span<PetscScalar>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  const std::string cell_type
      = dolfinx::mesh::to_string(mesh->topology().cell_type());
  const std::size_t num_dofs = element->space_dimension();
  const std::size_t bs = dofmap->bs();
  const int vs = element->reference_value_size() / element->block_size();

  // Tabulate function at quadrature points
  dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(), q_degree,
                                      fdim);
  // FIXME: This does not work for prism elements
  const std::vector<double> weights = q_rule.weights()[0];
  const std::vector<xt::xarray<double>> points = q_rule.points();

  const std::size_t num_points = weights.size();
  const std::size_t num_local_facets = points.size();
  xt::xtensor<double, 4> coeff_basis(
      {1, num_points, num_dofs / bs, (std::size_t)vs});
  xt::xtensor<double, 4> reference_basis_values(
      {num_local_facets, num_points, num_dofs / bs, (std::size_t)vs});

  for (std::size_t i = 0; i < num_local_facets; i++)
  {
    const xt::xarray<double>& q_facet = points[i];
    element->tabulate(coeff_basis, q_facet, 0);
    auto basis_ref
        = xt::view(reference_basis_values, i, xt::all(), xt::all(), xt::all());
    basis_ref = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());
  }

  std::vector<PetscScalar> c(num_facets * vs * bs * num_points, 0.0);
  const std::size_t cstride = vs * bs * num_points;
  if (needs_dof_transformations)
  {
    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values(
        {num_points, num_dofs / bs, (std::size_t)vs});
    xt::xtensor<double, 2> point_basis_values(
        {basis_values.shape(0), basis_values.shape(1)});

    // Get geometry data
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = mesh->geometry().x();

    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points, (std::size_t)tdim});
    xt::xtensor<double, 3> J
        = xt::zeros<double>({num_points, (std::size_t)gdim, (std::size_t)tdim});
    xt::xtensor<double, 3> K
        = xt::zeros<double>({num_points, (std::size_t)tdim, (std::size_t)gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_points});
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, (std::size_t)gdim});

    // Get coordinate map
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

    xt::xtensor<double, 5> dphi_c(
        {num_local_facets, (std::size_t)tdim, num_points, num_dofs_g / bs, 1});
    for (std::size_t i = 0; i < num_local_facets; i++)
    {
      const xt::xarray<double>& q_facet = points[i];
      xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, q_facet);
      auto dphi_ci
          = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all(), xt::all());
      dphi_ci = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                         xt::all(), xt::all(), xt::all());
    }

    // Get push forward function
    using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                          xt::xall<std::size_t>, xt::xall<std::size_t>>;
    using U_t = xt::xview<decltype(point_basis_values)&, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                          xt::xall<std::size_t>>;
    auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();

    for (std::size_t facet = 0; facet < num_facets; facet++)
    {

      // NOTE Add two separate loops here, one for and one without dof
      // transforms

      // FIXME: Assuming exterior facets
      // get cell/local facet index
      int global_facet = active_facets[facet]; // extract facet
      auto cells = f_to_c->links(global_facet);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell

      // find local index of facet
      auto cell_facets = c_to_f->links(cell);
      auto local_facet
          = std::find(cell_facets.begin(), cell_facets.end(), global_facet);
      const auto local_index = std::distance(cell_facets.data(), local_facet);
      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);

      for (std::size_t i = 0; i < num_dofs_g; ++i)
      {
        const int pos = 3 * x_dofs[i];
        for (int j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g[pos + j];
      }
      auto dphi_ci = xt::view(dphi_c, local_index, xt::all(), xt::all(),
                              xt::all(), xt::all());

      // NOTE: This can be simplified in affine case
      for (std::size_t q = 0; q < num_points; ++q)
      {
        J.fill(0);
        xt::xtensor<double, 2> dphi
            = xt::view(dphi_ci, xt::all(), q, xt::all(), 0);
        auto _J = xt::view(J, q, xt::all(), xt::all());
        cmap.compute_jacobian(dphi, coordinate_dofs, _J);
        auto _K = xt::view(K, q, xt::all(), xt::all());
        cmap.compute_jacobian_inverse(_J, _K);
        detJ[q] = cmap.compute_jacobian_determinant(_J);
        // Permute the reference values to account for the cell's orientation
        point_basis_values = xt::view(reference_basis_values, local_index, q,
                                      xt::all(), xt::all());
        transformation(xtl::span(point_basis_values.data(), num_dofs / bs * vs),
                       cell_info, cell, vs);

        // Push basis forward to physical element
        auto _u = xt::view(basis_values, q, xt::all(), xt::all());
        auto _U = xt::view(point_basis_values, xt::all(), xt::all());
        push_forward_fn(_u, _U, _J, detJ[q], _K);
      }

      // Sum up quadrature contributions
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = (int)bs * dofs[i];

        for (std::size_t q = 0; q < num_points; ++q)
          for (std::size_t k = 0; k < bs; ++k)
            for (int j = 0; j < vs; j++)
              c[cstride * facet + q * (bs * vs) + k + j]
                  += basis_values(q, i, j) * data[pos_v + k];
      }
    }
  }
  else
  {

    for (std::size_t facet = 0; facet < num_facets; facet++)
    { // Sum up quadrature contributions
      // FIXME: Assuming exterior facets
      // get cell/local facet index
      int global_facet = active_facets[facet]; // extract facet
      auto cells = f_to_c->links(global_facet);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell

      // find local index of facet
      auto cell_facets = c_to_f->links(cell);
      auto local_facet
          = std::find(cell_facets.begin(), cell_facets.end(), global_facet);
      const auto local_index = std::distance(cell_facets.data(), local_facet);

      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = (int)bs * dofs[i];

        for (std::size_t q = 0; q < num_points; ++q)
          for (std::size_t k = 0; k < bs; ++k)
            for (int l = 0; l < vs; l++)
            {
              c[cstride * facet + q * (bs * vs) + k + l]
                  += reference_basis_values(local_index, q, i, l)
                     * data[pos_v + k];
            }
      }
    }
  }
  return {std::move(c), cstride};
}

/// Prepare circumradii of triangle/tetrahedron for assembly with custom
/// kernels by packing them as an array, where the j*cstride to the ith facet
/// int active_facets.
/// @param[in] mesh
/// @param[in] active_facets List of active facets
/// @param[out] c The packed coefficients and the number of coeffs per facet
std::pair<std::vector<PetscScalar>, int>
pack_circumradius_facet(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        const xtl::span<const std::int32_t>& active_facets)
{
  const int tdim = mesh->topology().dim();
  const int gdim = mesh->geometry().dim();
  const int fdim = tdim - 1;
  const std::size_t num_facets = active_facets.size();

  // Connectivity to evaluate at quadrature points
  // FIXME: Move create_connectivity out of this function and call before
  // calling the function...
  mesh->topology_mutable().create_connectivity(fdim, tdim);
  auto f_to_c = mesh->topology().connectivity(fdim, tdim);
  mesh->topology_mutable().create_connectivity(tdim, fdim);
  auto c_to_f = mesh->topology().connectivity(tdim, fdim);

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  const dolfinx::mesh::CellType cell_type = mesh->topology().cell_type();

  // Quadrature points for piecewise constant
  dolfinx_cuas::QuadratureRule q_rule(cell_type, 0, fdim);

  // FIXME: This does not work for prism elements
  const std::vector<double> weights = q_rule.weights()[0];
  const std::vector<xt::xarray<double>> points = q_rule.points();

  const std::size_t num_points = weights.size();
  const std::size_t num_local_facets = points.size();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  xtl::span<const double> x_g = mesh->geometry().x();

  // Prepare geometry data structures
  xt::xtensor<double, 3> J = xt::zeros<double>(
      {std::size_t(1), (std::size_t)gdim, (std::size_t)tdim});
  xt::xtensor<double, 1> detJ = xt::zeros<double>({std::size_t(1)});
  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, (std::size_t)gdim});

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  xt::xtensor<double, 5> dphi_c(
      {num_local_facets, (std::size_t)tdim, num_points, num_dofs_g, 1});
  for (std::size_t i = 0; i < num_local_facets; i++)
  {
    const xt::xarray<double>& q_facet = points[i];
    xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, q_facet);
    auto dphi_ci
        = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all(), xt::all());
    dphi_ci = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                       xt::all(), xt::all(), xt::all());
  }

  // Prepare output variables
  std::vector<PetscScalar> circumradius;
  circumradius.reserve(num_facets);
  const int cstride = 1;

  for (auto facet : active_facets)
  {

    // NOTE Add two separate loops here, one for and one without dof
    // transforms

    // FIXME: Assuming exterior facets
    // get cell/local facet index
    auto cells = f_to_c->links(facet);
    // since the facet is on the boundary it should only link to one cell
    assert(cells.size() == 1);
    auto cell = cells[0]; // extract cell

    // find local index of facet
    auto cell_facets = c_to_f->links(cell);
    auto local_facet = std::find(cell_facets.begin(), cell_facets.end(), facet);
    const auto local_index = std::distance(cell_facets.data(), local_facet);

    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[pos + j];
    }

    // Compute determinant of Jacobian which is used to compute the area/volume
    // of the cell
    auto dphi_ci = xt::view(dphi_c, local_index, xt::all(), xt::all(),
                            xt::all(), xt::all());
    J.fill(0);
    auto _J = xt::view(J, 0, xt::all(), xt::all());
    xt::xtensor<double, 2> dphi
        = xt::view(dphi_c, local_index, xt::all(), 0, xt::all(), 0);
    cmap.compute_jacobian(dphi, coordinate_dofs, _J);
    detJ[0] = cmap.compute_jacobian_determinant(_J);

    double h = 0;
    switch (cell_type)
    {
    case dolfinx::mesh::CellType::triangle:
    {
      // Formula for circumradius of a triangle with sides with length a, b, c
      // is R = a b c / (4 A) where A is the area of the triangle
      const double ref_area = basix::cell::volume(basix::cell::type::triangle);
      double area = ref_area * std::abs(detJ[0]);

      // Compute the lenghts of each side of the cell
      std::array<double, 3> sides
          = {0, 0, 0}; // Array to hold lenghts of sides of triangle
      for (int i = 0; i < gdim; i++)
      {
        sides[0] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(1, i), 2);
        sides[1] += std::pow(coordinate_dofs(1, i) - coordinate_dofs(2, i), 2);
        sides[2] += std::pow(coordinate_dofs(2, i) - coordinate_dofs(0, i), 2);
      }
      std::for_each(sides.begin(), sides.end(),
                    [](double& side) { side = std::sqrt(side); });

      h = sides[0] * sides[1] * sides[2] / (4 * area);
      break;
    }
    case dolfinx::mesh::CellType::tetrahedron:
    {
      // Formula for circunradius of a tetrahedron with volume V.
      // Given three edges meeting at a vertex with length a, b, c,
      // and opposite edges with corresponding length A, B, C we have that the
      // circumradius
      // R = sqrt((aA + bB + cC)(aA+bB-cC)(aA-bB+cC)(-aA +bB+cC))/24V
      const double ref_volume
          = basix::cell::volume(basix::cell::type::tetrahedron);
      double cellvolume = detJ[0] * ref_volume;

      // Edges ordered as a, b, c, A, B, C
      std::array<double, 6> edges = {0, 0, 0, 0, 0, 0};
      for (int i = 0; i < gdim; i++)
      {
        // Accummulate a^2, b^2, c^2
        edges[0] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(1, i), 2);
        edges[1] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(2, i), 2);
        edges[2] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(3, i), 2);

        // Accumulate A^2, B^2, C^2
        edges[3] += std::pow(coordinate_dofs(2, i) - coordinate_dofs(3, i), 2);
        edges[4] += std::pow(coordinate_dofs(1, i) - coordinate_dofs(3, i), 2);
        edges[5] += std::pow(coordinate_dofs(1, i) - coordinate_dofs(2, i), 2);
      }
      // Compute length of each edge
      std::for_each(edges.begin(), edges.end(),
                    [](double& edge) { edge = std::sqrt(edge); });

      // Compute temporary variables
      const double aA = edges[0] * edges[3];
      const double bB = edges[1] * edges[4];
      const double cC = edges[2] * edges[5];

      // Compute circumradius
      h = std::sqrt((aA + bB + cC) * (aA + bB - cC) * (aA - bB + cC)
                    * (-aA + bB + cC))
          / (24 * cellvolume);
      break;
    }
    default:
      throw std::runtime_error("Unsupported cell_type "
                               + dolfinx::mesh::to_string(cell_type));
    }

    // Sum up quadrature contributions
    circumradius.push_back(h);
  }
  return {std::move(circumradius), cstride};
}
/// This function computes the pull back for a set of points x on a cell
/// described by coordinate_dofs as well as the corresponding Jacobian, their
/// inverses and their determinants
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in] x: points on physical element
/// @param[in ,out] X: pull pack of x (points on reference element)
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] element: the corresponding finite element
/// @param[in] cmap: the coordinate element
//-----------------------------------------------------------------------------
void pull_back(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
               xt::xtensor<double, 1>& detJ, const xt::xtensor<double, 2>& x,
               xt::xtensor<double, 2>& X,
               const xt::xtensor<double, 2>& coordinate_dofs,
               std::shared_ptr<const dolfinx::fem::FiniteElement> element,
               const dolfinx::fem::CoordinateElement& cmap)
{
  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) >= num_points);
  assert(K.shape(0) >= num_points);
  assert(detJ.shape(0) >= num_points);

  // Get mesh data from input
  const size_t tdim = K.shape(1);

  // -- Lambda function for affine pull-backs
  auto pull_back_affine
      = [&cmap, tdim,
         X0 = xt::xtensor<double, 2>(xt::zeros<double>({std::size_t(1), tdim})),
         data = xt::xtensor<double, 4>(cmap.tabulate_shape(1, 1)),
         dphi = xt::xtensor<double, 2>({tdim, cmap.tabulate_shape(1, 1)[2]})](
            auto&& X, const auto& cell_geometry, auto&& J, auto&& K,
            const auto& x) mutable
  {
    cmap.tabulate(1, X0, data);
    dphi = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
    cmap.compute_jacobian(dphi, cell_geometry, J);
    cmap.compute_jacobian_inverse(J, K);
    cmap.pull_back_affine(X, K, cmap.x0(cell_geometry), x);
  };

  xt::xtensor<double, 2> dphi;

  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
  if (cmap.is_affine())
  {
    J.fill(0);
    pull_back_affine(X, coordinate_dofs, xt::view(J, 0, xt::all(), xt::all()),
                     xt::view(K, 0, xt::all(), xt::all()), x);
    detJ[0] = cmap.compute_jacobian_determinant(
        xt::view(J, 0, xt::all(), xt::all()));
    for (std::size_t p = 1; p < num_points; ++p)
    {
      xt::view(J, p, xt::all(), xt::all())
          = xt::view(J, 0, xt::all(), xt::all());
      xt::view(K, p, xt::all(), xt::all())
          = xt::view(K, 0, xt::all(), xt::all());
      detJ[p] = detJ[0];
    }
  }
  else
  {
    cmap.pull_back_nonaffine(X, x, coordinate_dofs);
    cmap.tabulate(1, X, phi);
    dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
    J.fill(0);
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      auto _J = xt::view(J, p, xt::all(), xt::all());
      cmap.compute_jacobian(dphi, coordinate_dofs, _J);
      cmap.compute_jacobian_inverse(_J, xt::view(K, p, xt::all(), xt::all()));
      detJ[p] = cmap.compute_jacobian_determinant(_J);
    }
  }
}
//-----------------------------------------------------------------------------
/// This function computes the basis function values on a given cell at a
/// given set of points
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in] x: points on physical element
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] index: the index of the cell (local to process)
/// @param[in] perm: permutation infor for cell
/// @param[in] element: the corresponding finite element
/// @param[in] cmap: the coordinate element
xt::xtensor<double, 3>
get_basis_functions(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
                    xt::xtensor<double, 1>& detJ,
                    const xt::xtensor<double, 2>& x,
                    const xt::xtensor<double, 2>& coordinate_dofs,
                    const std::int32_t index, const std::int32_t perm,
                    std::shared_ptr<const dolfinx::fem::FiniteElement> element,
                    const dolfinx::fem::CoordinateElement& cmap)
{
  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) >= num_points);
  assert(K.shape(0) >= num_points);
  assert(detJ.shape(0) >= num_points);

  // Get mesh data from input
  const size_t tdim = K.shape(1);

  // Get element data
  const size_t block_size = element->block_size();
  const size_t reference_value_size
      = element->reference_value_size() / block_size;
  const size_t value_size = element->value_size() / block_size;
  const size_t space_dimension = element->space_dimension() / block_size;

  xt::xtensor<double, 2> X({x.shape(0), tdim});

  // Skip negative cell indices
  xt::xtensor<double, 3> basis_array = xt::zeros<double>(
      {num_points, space_dimension * block_size, value_size * block_size});
  if (index < 0)
    return basis_array;

  pull_back(J, K, detJ, x, X, coordinate_dofs, element, cmap);
  // Prepare basis function data structures
  xt::xtensor<double, 4> tabulated_data(
      {1, num_points, space_dimension, reference_value_size});
  xt::xtensor<double, 3> basis_values(
      {num_points, space_dimension, value_size});

  // Get push forward function
  xt::xtensor<double, 2> point_basis_values({space_dimension, value_size});
  using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(point_basis_values)&, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  // FIXME: These should be moved out of this function
  auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();
  const std::function<void(const xtl::span<PetscScalar>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  // Compute basis on reference element
  element->tabulate(tabulated_data, X, 0);
  for (std::size_t q = 0; q < num_points; ++q)
  {
    // Permute the reference values to account for the cell's orientation
    point_basis_values = xt::view(tabulated_data, 0, q, xt::all(), xt::all());
    element->apply_dof_transformation(
        xtl::span<double>(point_basis_values.data(), point_basis_values.size()),
        perm, 1);
    // Push basis forward to physical element
    auto _J = xt::view(J, q, xt::all(), xt::all());
    auto _K = xt::view(K, q, xt::all(), xt::all());
    auto _u = xt::view(basis_values, q, xt::all(), xt::all());
    auto _U = xt::view(point_basis_values, xt::all(), xt::all());
    push_forward_fn(_u, _U, _J, detJ[q], _K);
  }

  // Expand basis values for each dof
  for (std::size_t p = 0; p < num_points; ++p)
  {
    for (std::size_t block = 0; block < block_size; ++block)
    {
      for (std::size_t i = 0; i < space_dimension; ++i)
      {
        for (std::size_t j = 0; j < value_size; ++j)
        {
          basis_array(p, i * block_size + block, j * block_size + block)
              = basis_values(p, i, j);
        }
      }
    }
  }
  return basis_array;
}
//-----------------------------------------------------------------------------
/// This function computes the outward unit normal on a given cell at a
/// given set of points on the facets of the cell
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in] x: points on facets physical element
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] index: the index of the cell (local to process)
/// @param[in] facet_indices: local facet index corresponding to each point
/// @param[in] element: the corresponding finite element
/// @param[in] cmap: the coordinate element
/// @param[in] facet_normals: The facet normals on the reference cell
xt::xtensor<double, 2>
get_facet_normals(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
                  xt::xtensor<double, 1>& detJ, const xt::xtensor<double, 2>& x,
                  const xt::xtensor<double, 2>& coordinate_dofs,
                  const std::int32_t index,
                  const xt::xtensor<std::int32_t, 1>& facet_indices,
                  std::shared_ptr<const dolfinx::fem::FiniteElement> element,
                  const dolfinx::fem::CoordinateElement& cmap,
                  xt::xtensor<double, 2> facet_normals);

double R_plus(double x) { return 0.5 * (std::abs(x) + x); }
double dR_plus(double x) { return double(x > 0); }

/// See
/// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c/10133700
template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

/// @param[in] cells: the cells to be sorted
/// @param[in, out] perm: the permutation for the sorted cells
/// @param[out] pair(unique_cells, offsets): unique_cells is a vector of
/// sorted cells with all duplicates deleted, offsets contains the start and
/// end for each unique value in the sorted vector with all duplicates
// Example: cells = [5, 7, 6, 5]
//          unique_cells = [5, 6, 7]
//          offsets = [0, 2, 3, 4]
//          perm = [0, 3, 2, 1]
// Then given a cell and its index ("i") in unique_cells, one can recover the
// indices for its occurance in cells with perm[k], where
// offsets[i]<=k<offsets[i+1]. In the example if i = 0, then perm[k] = 0 or
// perm[k] = 3.
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
sort_cells(const xtl::span<const std::int32_t>& cells,
           const xtl::span<std::int32_t>& perm)
{
  assert(perm.size() == cells.size());

  const auto num_cells = (std::int32_t)cells.size();
  std::vector<std::int32_t> unique_cells(num_cells);
  std::vector<std::int32_t> offsets(num_cells + 1, 0);
  std::iota(perm.begin(), perm.end(), 0);
  dolfinx::argsort_radix<std::int32_t>(cells, perm);

  // Sort cells in accending order
  for (std::int32_t i = 0; i < num_cells; ++i)
    unique_cells[i] = cells[perm[i]];

  // Compute the number of identical cells
  std::int32_t index = 0;
  for (std::int32_t i = 0; i < num_cells - 1; ++i)
    if (unique_cells[i] != unique_cells[i + 1])
      offsets[++index] = i + 1;

  offsets[index + 1] = num_cells;
  unique_cells.erase(std::unique(unique_cells.begin(), unique_cells.end()),
                     unique_cells.end());
  offsets.resize(unique_cells.size() + 1);

  return std::make_pair(unique_cells, offsets);
}
} // namespace dolfinx_contact