// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "coefficients.h"
#include <basix/quadrature.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xslice.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx_contact;

std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::pack_coefficient_quadrature(
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

std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::pack_coefficient_facet(
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
