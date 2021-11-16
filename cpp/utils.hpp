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
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx_cuas/QuadratureRule.hpp>

namespace dolfinx_contact
{

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as an array, where j is the index of the local cell
/// and c[j*cstride + q * (block_size * value_size) + k + c] = sum_i c^i[k] *
/// phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth vector component,
/// phi^i(x_q)[c] is the ith basis function's c-th value compoenent at the
/// quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[out] c The packed coefficients and the number of coeffs per cell
std::pair<std::vector<PetscScalar>, int> pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q)
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
      = mesh->topology().index_map(tdim)->size_local()
        + mesh->topology().index_map(tdim)->num_ghosts();

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
      dolfinx::mesh::cell_type_to_basix_type(cell_type), q);
  const std::size_t num_points = weights.size();
  xt::xtensor<double, 4> coeff_basis({1, num_points, num_dofs, vs});
  element->tabulate(coeff_basis, points, 0);
  std::vector<PetscScalar> c(num_cells * vs * bs * num_points, 0.0);
  const int cstride = vs * bs * num_points;
  auto basis_reference_values
      = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());

  if (needs_dof_transformations)
  {
    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values({num_points, num_dofs / bs, vs});
    xt::xtensor<double, 3> cell_basis_values({num_points, num_dofs / bs, vs});

    // Get geometry data
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

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

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {

      // NOTE Add two separate loops here, one for and one without dof
      // transforms

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g(x_dofs[i], j);
      // NOTE: This can be simplified in affine case
      for (std::size_t q = 0; q < num_points; ++q)
      {
        J.fill(0);
        auto _J = xt::view(J, q, xt::all(), xt::all());
        xt::xtensor<double, 2> dphi
            = xt::view(dphi_c, xt::all(), q, xt::all(), 0);
        cmap.compute_jacobian(dphi, coordinate_dofs, _J);
        cmap.compute_jacobian_inverse(_J, xt::view(K, q, xt::all(), xt::all()));
        detJ[q] = cmap.compute_jacobian_determinant(_J);
      }

      // Permute the reference values to account for the cell's orientation
      cell_basis_values = basis_reference_values;
      for (std::size_t q = 0; q < num_points; ++q)
      {
        transformation(
            xtl::span(cell_basis_values.data() + q * num_dofs / bs * vs,
                      num_dofs / bs * vs),
            cell_info, cell, vs);
      }
      // Push basis forward to physical element
      element->push_forward(basis_values, cell_basis_values, J, detJ, K);

      // Sum up quadrature contributions
      int offset = cstride * cell;
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int j = 0; j < vs; j++)
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
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int j = 0; j < vs; j++)
              c[offset + q * (bs * vs) + k + j]
                  += basis_reference_values(q, i, j) * data[pos_v + k];
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
/// @param[in] q the quadrature degree
/// @param[out] c The packed coefficients and the number of coeffs per facet
std::pair<std::vector<PetscScalar>, int> pack_coefficient_facet(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff, int q,
    const xtl::span<const std::int32_t>& active_facets)
{
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const dolfinx::fem::FiniteElement* element
      = coeff->function_space()->element().get();
  const xtl::span<const double> data = coeff->x()->array();

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh
      = coeff->function_space()->mesh();
  assert(mesh);
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t fdim = tdim - 1;
  const std::int32_t num_facets = active_facets.size();

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
  const std::size_t vs
      = element->reference_value_size() / element->block_size();

  // Tabulate function at quadrature points
  dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(), q, fdim);
  // FIXME: This does not work for prism elements
  const std::vector<double> weights = q_rule.weights()[0];
  const std::vector<xt::xarray<double>> points = q_rule.points();

  const std::size_t num_points = weights.size();
  const std::size_t num_local_facets = points.size();
  xt::xtensor<double, 4> coeff_basis({1, num_points, num_dofs / bs, vs});
  xt::xtensor<double, 4> basis_reference_values(
      {num_local_facets, num_points, num_dofs / bs, vs});

  for (int i = 0; i < num_local_facets; i++)
  {
    const xt::xarray<double>& q_facet = points[i];
    element->tabulate(coeff_basis, q_facet, 0);
    auto basis_ref
        = xt::view(basis_reference_values, i, xt::all(), xt::all(), xt::all());
    basis_ref = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());
  }

  std::vector<PetscScalar> c(num_facets * vs * bs * num_points, 0.0);
  const int cstride = vs * bs * num_points;
  if (needs_dof_transformations)
  {
    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values({num_points, num_dofs / bs, vs});
    xt::xtensor<double, 3> cell_basis_values({num_points, num_dofs / bs, vs});

    // Get geometry data
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points, tdim});
    xt::xtensor<double, 3> J = xt::zeros<double>({num_points, gdim, tdim});
    xt::xtensor<double, 3> K = xt::zeros<double>({num_points, tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_points});
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, gdim});

    // Get coordinate map
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

    xt::xtensor<double, 5> dphi_c(
        {num_local_facets, int(tdim), num_points, num_dofs_g / bs, 1});
    for (int i = 0; i < num_local_facets; i++)
    {
      const xt::xarray<double>& q_facet = points[i];
      xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, q_facet);
      auto dphi_ci
          = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all(), xt::all());
      dphi_ci = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                         xt::all(), xt::all(), xt::all());
    }

    for (int facet = 0; facet < num_facets; facet++)
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
      const std::int32_t local_index
          = std::distance(cell_facets.data(), local_facet);
      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);

      for (std::size_t i = 0; i < num_dofs_g; ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g(x_dofs[i], j);

      auto dphi_ci = xt::view(dphi_c, local_index, xt::all(), xt::all(),
                              xt::all(), xt::all());

      // NOTE: This can be simplified in affine case
      for (std::size_t q = 0; q < num_points; ++q)
      {
        J.fill(0);
        auto _J = xt::view(J, q, xt::all(), xt::all());
        xt::xtensor<double, 2> dphi
            = xt::view(dphi_ci, xt::all(), q, xt::all(), 0);
        cmap.compute_jacobian(dphi, coordinate_dofs, _J);
        cmap.compute_jacobian_inverse(_J, xt::view(K, q, xt::all(), xt::all()));
        detJ[q] = cmap.compute_jacobian_determinant(_J);
      }

      // Permute the reference values to account for the cell's orientation
      cell_basis_values = xt::view(basis_reference_values, local_index,
                                   xt::all(), xt::all(), xt::all());
      for (std::size_t q = 0; q < num_points; ++q)
      {
        transformation(
            xtl::span(cell_basis_values.data() + q * num_dofs / bs * vs,
                      num_dofs / bs * vs),
            cell_info, cell, vs);
      }
      // Push basis forward to physical element
      element->push_forward(basis_values, cell_basis_values, J, detJ, K);

      // Sum up quadrature contributions
      int offset = cstride * facet;
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int j = 0; j < vs; j++)
              c[offset + q * (bs * vs) + k + j]
                  += basis_values(q, i, j) * data[pos_v + k];
      }
    }
  }
  else
  {

    for (int facet = 0; facet < num_facets; facet++)
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
      const std::int32_t local_index
          = std::distance(cell_facets.data(), local_facet);

      int offset = cstride * facet;
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int l = 0; l < vs; l++)
            {
              c[offset + q * (bs * vs) + k + l]
                  += basis_reference_values(local_index, q, i, l)
                     * data[pos_v + k];
            }
      }
    }
  }
  return {std::move(c), cstride};
}

std::pair<std::vector<PetscScalar>, int> pack_coefficient_dofs(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    xt::xtensor<std::int32_t, 2>& active_facets)
{

  auto element = coeff->function_space()->element().get();
  auto dofmap = coeff->function_space()->dofmap().get();
  auto v = coeff->x()->array();
  // Get mesh
  auto mesh = coeff->function_space()->mesh();
  assert(mesh);
  const std::size_t num_facets = active_facets.shape(0);
  std::size_t bs = dofmap->bs();
  std::size_t ndofs = dofmap->cell_dofs(0).size();
  std::vector<PetscScalar> c(num_facets * ndofs * bs);
  const int cstride = ndofs * bs;

  bool needs_dof_transformations = false;
  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    needs_dof_transformations = true;
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  const std::function<void(const xtl::span<PetscScalar>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation
      = element->get_dof_transformation_function<PetscScalar>(false, true);

  for (std::size_t facet = 0; facet < num_facets; ++facet)
  {
    std::int32_t cell = active_facets(facet, 0);
    auto dofs = dofmap->cell_dofs(cell);
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      for (std::size_t k = 0; k < bs; ++k)
      {
        c[facet * cstride + bs * i + k] = v[bs * dofs[i] + k];
      }
    }
    transformation(
        xtl::span(c.data() + facet * cstride, element->space_dimension()),
        cell_info, cell, 1);
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
  // // Get mesh
  // std::shared_ptr<const dolfinx::mesh::Mesh> mesh =
  // coeff->function_space()->mesh(); assert(mesh);
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t fdim = tdim - 1;
  const std::int32_t num_facets = active_facets.size();

  // Connectivity to evaluate at quadrature points
  // FIXME: Move create_connectivity out of this function and call before
  // calling the function...
  mesh->topology_mutable().create_connectivity(fdim, tdim);
  auto f_to_c = mesh->topology().connectivity(fdim, tdim);
  mesh->topology_mutable().create_connectivity(tdim, fdim);
  auto c_to_f = mesh->topology().connectivity(tdim, fdim);

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  const std::string cell_type
      = dolfinx::mesh::to_string(mesh->topology().cell_type());

  // Quadrature points for piecewise constant
  dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(), 0, fdim);
  // FIXME: This does not work for prism elements
  const std::vector<double> weights = q_rule.weights()[0];
  const std::vector<xt::xarray<double>> points = q_rule.points();

  const std::size_t num_points = weights.size();
  const std::size_t num_local_facets = points.size();

  std::vector<PetscScalar> c(num_facets, 0.0);
  const int cstride = 1;

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const std::size_t num_dofs_g = x_dofmap.num_links(0);
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

  // Prepare geometry data structures
  // xt::xtensor<double, 2> X({num_points, tdim});
  xt::xtensor<double, 3> J = xt::zeros<double>({std::size_t(1), gdim, tdim});
  xt::xtensor<double, 3> K = xt::zeros<double>({std::size_t(1), tdim, gdim});
  xt::xtensor<double, 1> detJ = xt::zeros<double>({std::size_t(1)});
  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, gdim});

  // Get coordinate map
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

  xt::xtensor<double, 5> dphi_c(
      {num_local_facets, int(tdim), num_points, num_dofs_g, 1});
  for (int i = 0; i < num_local_facets; i++)
  {
    const xt::xarray<double>& q_facet = points[i];
    xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, q_facet);
    auto dphi_ci
        = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all(), xt::all());
    dphi_ci = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                       xt::all(), xt::all(), xt::all());
  }

  for (int facet = 0; facet < num_facets; facet++)
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
    const std::int32_t local_index
        = std::distance(cell_facets.data(), local_facet);
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell);

    for (std::size_t i = 0; i < num_dofs_g; ++i)
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(x_dofs[i], j);

    auto dphi_ci = xt::view(dphi_c, local_index, xt::all(), xt::all(),
                            xt::all(), xt::all());

    J.fill(0);
    auto _J = xt::view(J, 0, xt::all(), xt::all());
    xt::xtensor<double, 2> dphi
        = xt::view(dphi_c, local_index, xt::all(), 0, xt::all(), 0);
    cmap.compute_jacobian(dphi, coordinate_dofs, _J);
    cmap.compute_jacobian_inverse(_J, xt::view(K, 0, xt::all(), xt::all()));
    detJ[0] = cmap.compute_jacobian_determinant(_J);
    double h = 0;
    if (cell_type == "triangle")
    {
      double cellvolume
          = 0.5 * std::abs(detJ[0]); // reference triangle has area 0.5
      double a = 0, b = 0, c = 0;
      for (int i = 0; i < gdim; i++)
      {
        a += (coordinate_dofs(0, i) - coordinate_dofs(1, i))
             * (coordinate_dofs(0, i) - coordinate_dofs(1, i));
        b += (coordinate_dofs(1, i) - coordinate_dofs(2, i))
             * (coordinate_dofs(1, i) - coordinate_dofs(2, i));
        c += (coordinate_dofs(2, i) - coordinate_dofs(0, i))
             * (coordinate_dofs(2, i) - coordinate_dofs(0, i));
      }
      a = std::sqrt(a);
      b = std::sqrt(b);
      c = std::sqrt(c);
      h = a * b * c / (4 * cellvolume);
    }
    else if (cell_type == "tetrahedron")
    {
      double cellvolume
          = detJ[0] / 6; // reference tetrahedron has volume 1/6 = 0.5*1/3
      double a = 0, b = 0, c = 0, A = 0, B = 0, C = 0;
      for (int i = 0; i < gdim; i++)
      {
        a += (coordinate_dofs(0, i) - coordinate_dofs(1, i))
             * (coordinate_dofs(0, i) - coordinate_dofs(1, i));
        b += (coordinate_dofs(0, i) - coordinate_dofs(2, i))
             * (coordinate_dofs(0, i) - coordinate_dofs(2, i));
        c += (coordinate_dofs(0, i) - coordinate_dofs(3, i))
             * (coordinate_dofs(0, i) - coordinate_dofs(3, i));
        A += (coordinate_dofs(2, i) - coordinate_dofs(3, i))
             * (coordinate_dofs(2, i) - coordinate_dofs(3, i));
        B += (coordinate_dofs(1, i) - coordinate_dofs(3, i))
             * (coordinate_dofs(1, i) - coordinate_dofs(3, i));
        C += (coordinate_dofs(1, i) - coordinate_dofs(2, i))
             * (coordinate_dofs(1, i) - coordinate_dofs(2, i));
      }
      a = std::sqrt(a);
      b = std::sqrt(b);
      c = std::sqrt(c);
      A = std::sqrt(A);
      B = std::sqrt(B);
      C = std::sqrt(C);
      h = std::sqrt((a * A + b * B + c * C) * (a * A + b * B - c * C)
                    * (a * A - b * B + c * C) * (b * B + c * C - a * A))
          / (24 * cellvolume);
    }
    // Sum up quadrature contributions
    c[facet] = h;
  }

  return {std::move(c), cstride};
}
// helper functiion for pack_coefficients_facet and pack_circumradius_facet to
// work with dolfinx assembly routines should be made reduntant at a later
// stage
/// @param[in] mesh - the mesh
/// @param[in] active_facets - facet indices
/// @param[in] data - data to be converted
/// @param[in] num_cols - number of columns per facet
std::pair<std::vector<PetscScalar>, int>
facet_to_cell_data(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                   const xtl::span<const std::int32_t>& active_facets,
                   const xtl::span<const PetscScalar> data, int num_cols)
{
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t fdim = tdim - 1;
  const std::int32_t num_facets = active_facets.size();
  const std::int32_t num_cells
      = mesh->topology().index_map(tdim)->size_local()
        + mesh->topology().index_map(tdim)->num_ghosts();
  // Connectivity to evaluate at quadrature points
  // Assumes connectivity already created
  auto f_to_c = mesh->topology().connectivity(fdim, tdim);
  auto c_to_f = mesh->topology().connectivity(tdim, fdim);
  // get number of facets per cell. Assuming all cells are the same
  const std::size_t num_facets_c = c_to_f->num_links(0);

  std::vector<PetscScalar> c(num_cells * num_cols * num_facets_c, 0.0);
  const int cstride = num_cols * num_facets_c;
  for (int i = 0; i < num_facets; i++)
  {
    auto facet = active_facets[i];
    // assuming exterior facets
    auto cell = f_to_c->links(facet)[0];
    // find local index of facet
    auto cell_facets = c_to_f->links(cell);
    auto local_facet = std::find(cell_facets.begin(), cell_facets.end(), facet);
    const std::int32_t local_index
        = std::distance(cell_facets.data(), local_facet);
    int offset = cell * cstride;
    for (int j = 0; j < num_cols; j++)
    {
      c[offset + local_index * num_cols + j] = data[i * num_cols + j];
    }
  }
  return {std::move(c), cstride};
}

//-----------------------------------------------------------------------------
xt::xtensor<double, 3>
get_basis_functions(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
                    xt::xtensor<double, 1>& detJ,
                    const xt::xtensor<double, 2>& x,
                    xt::xtensor<double, 2> coordinate_dofs,
                    const std::int32_t index, const std::int32_t perm,
                    std::shared_ptr<const dolfinx::fem::FiniteElement> element,
                    const dolfinx::fem::CoordinateElement& cmap)
{

  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) == num_points);
  assert(K.shape(0) == num_points);
  assert(detJ.shape(0) == num_points);

  // Get mesh data from input
  const size_t gdim = coordinate_dofs.shape(1);
  const size_t num_dofs_g = coordinate_dofs.shape(0);
  const size_t tdim = K.shape(1);

  // Get element data
  const size_t block_size = element->block_size();
  const size_t reference_value_size
      = element->reference_value_size() / block_size;
  const size_t value_size = element->value_size() / block_size;
  const size_t space_dimension = element->space_dimension() / block_size;

  // Prepare basis function data structures
  xt::xtensor<double, 4> tabulated_data(
      {1, num_points, space_dimension, reference_value_size});
  auto reference_basis_values
      = xt::view(tabulated_data, 0, xt::all(), xt::all(), xt::all());
  xt::xtensor<double, 3> basis_values(
      {num_points, space_dimension, value_size});

  // Skip negative cell indices
  xt::xtensor<double, 3> basis_array = xt::zeros<double>(
      {num_points, space_dimension * block_size, value_size * block_size});
  if (index < 0)
    return basis_array;

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
  // FIXME: Move initialization out of J, detJ, K out of function
  xt::xtensor<double, 2> dphi;
  xt::xtensor<double, 2> X({x.shape(0), tdim});
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

  // Compute basis on reference element
  element->tabulate(tabulated_data, X, 0);

  element->apply_dof_transformation(
      xtl::span<double>(tabulated_data.data(), tabulated_data.size()), perm,
      reference_value_size);

  // Push basis forward to physical element
  element->push_forward(basis_values, reference_basis_values, J, detJ, K);

  // Expand basis values for each dof
  for (std::size_t p = 0; p < num_points; ++p)
  {
    for (int block = 0; block < block_size; ++block)
    {
      for (int i = 0; i < space_dimension; ++i)
      {
        for (int j = 0; j < value_size; ++j)
        {
          basis_array(p, i * block_size + block, j * block_size + block)
              = basis_values(p, i, j);
        }
      }
    }
  }
  return basis_array;
}
double R_plus(double x) { return 0.5 * (std::abs(x) + x); }
double dR_plus(double x) { return double(x > 0); }

/// See
/// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c/10133700
template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}
} // namespace dolfinx_contact