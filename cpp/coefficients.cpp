// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "coefficients.h"
#include "geometric_quantities.h"
#include <basix/quadrature.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xslice.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx_contact;

std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree,
    std::variant<tcb::span<const std::int32_t>,
                 tcb::span<const std::pair<std::int32_t, int>>>
        active_entities)
{
  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh
      = coeff->function_space()->mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const dolfinx::mesh::Topology& topology = mesh->topology();

  // Create quadrature rule
  const int tdim = topology.dim();
  const int gdim = geometry.dim();
  const dolfinx::mesh::CellType cell_type = topology.cell_type();
  std::shared_ptr<dolfinx_cuas::QuadratureRule> q_rule;
  std::visit(
      [&q_rule, q_degree, tdim, cell_type](auto& entities)
      {
        using U = std::decay_t<decltype(entities)>;
        if constexpr (std::is_same_v<U, tcb::span<const std::int32_t>>)
        {
          q_rule = std::make_shared<dolfinx_cuas::QuadratureRule>(
              cell_type, q_degree, tdim);
        }
        else if constexpr (std::is_same_v<
                               U,
                               tcb::span<const std::pair<std::int32_t, int>>>)
        {
          q_rule = std::make_shared<dolfinx_cuas::QuadratureRule>(
              cell_type, q_degree, tdim - 1);
        }
        else
        {
          throw std::runtime_error("Could not pack coefficients. Input entity "
                                   "type is not supported.");
        }
      },
      active_entities);

  // Get the dofmap and finite element
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const dolfinx::fem::FiniteElement* element
      = coeff->function_space()->element().get();

  // Get the coeffs to pack
  const xtl::span<const double> data = coeff->x()->array();

  // Get dof transformations
  const bool needs_dof_transformations = element->needs_dof_transformations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = topology.get_cell_permutation_info();
  }
  const std::function<void(const xtl::span<PetscScalar>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  const std::size_t num_dofs = element->space_dimension();
  const std::size_t bs = dofmap->bs();
  const int vs = element->reference_value_size() / element->block_size();

  // Tabulate function at quadrature points (assuming no derivatives)
  auto weights = q_rule->weights_ref();
  auto points = q_rule->points_ref();
  const std::size_t num_entities = points.size();
  // NOTE: Does not work for facet integrals on prisms
  const std::size_t num_points = weights[0].size();
  xt::xtensor<double, 4> reference_basis_values(
      {num_entities, num_points, num_dofs / bs, (std::size_t)vs});

  // Temporary variable to fill in loop
  xt::xtensor<double, 4> coeff_basis(
      {1, num_points, num_dofs / bs, (std::size_t)vs});
  for (std::size_t i = 0; i < num_entities; i++)
  {
    const xt::xarray<double>& q_ent = points[i];
    element->tabulate(coeff_basis, q_ent, 0);
    auto basis_ref
        = xt::view(reference_basis_values, i, xt::all(), xt::all(), xt::all());
    basis_ref = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());
  }

  std::function<std::pair<std::int32_t, int>(std::size_t)> get_cell_info;
  std::size_t num_active_entities;
  // TODO see if this can be simplified with templating
  std::visit(
      [&num_active_entities, &get_cell_info](auto& entities)
      {
        using U = std::decay_t<decltype(entities)>;
        if constexpr (std::is_same_v<U, tcb::span<const std::int32_t>>)
        {
          num_active_entities = entities.size();
          // Iterate over coefficients
          get_cell_info = [&entities](auto i)
          {
            std::pair<std::int32_t, int> pair(entities[i], 0);
            return pair;
          };
        }
        else if constexpr (std::is_same_v<
                               U,
                               tcb::span<const std::pair<std::int32_t, int>>>)
        {
          num_active_entities = entities.size();
          // Create lambda function fetching cell index from exterior facet
          // entity
          get_cell_info = [&entities](auto i) { return entities[i]; };
        }
        else
        {
          throw std::runtime_error("Could not pack coefficient. Input entity "
                                   "type is not supported.");
        }
      },
      active_entities);

  // Create output array
  std::vector<PetscScalar> coefficients(
      num_active_entities * vs * bs * num_points, 0.0);
  const auto cstride = int(vs * bs * num_points);

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
    const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
    const std::size_t num_dofs_g = cmap.dim();
    xtl::span<const double> x_g = geometry.x();

    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points, (std::size_t)tdim});
    xt::xtensor<double, 3> J
        = xt::zeros<double>({num_points, (std::size_t)gdim, (std::size_t)tdim});
    xt::xtensor<double, 3> K
        = xt::zeros<double>({num_points, (std::size_t)tdim, (std::size_t)gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_points});
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, (std::size_t)gdim});

    xt::xtensor<double, 5> dphi_c(
        {num_entities, (std::size_t)tdim, num_points, num_dofs_g / bs, 1});
    for (std::size_t i = 0; i < num_entities; i++)
    {
      const xt::xarray<double>& q_ent = points[i];
      xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, q_ent);
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

    for (std::size_t i = 0; i < num_active_entities; i++)
    {
      auto [cell, entity_index] = get_cell_info(i);

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t k = 0; k < num_dofs_g; ++k)
      {
        const int pos = 3 * x_dofs[k];
        for (int j = 0; j < gdim; ++j)
          coordinate_dofs(k, j) = x_g[pos + j];
      }
      auto dphi_ci = xt::view(dphi_c, entity_index, xt::all(), xt::all(),
                              xt::all(), xt::all());

      // NOTE: This can be simplified in affine case
      for (std::size_t q = 0; q < num_points; ++q)
      {
        J.fill(0);
        xt::xtensor<double, 2> dphi
            = xt::view(dphi_ci, xt::all(), q, xt::all(), 0);
        auto _J = xt::view(J, q, xt::all(), xt::all());
        dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                          _J);
        auto _K = xt::view(K, q, xt::all(), xt::all());
        dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
        detJ[q]
            = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
        // Permute the reference values to account for the cell's orientation
        point_basis_values = xt::view(reference_basis_values, entity_index, q,
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
      for (std::size_t d = 0; d < dofs.size(); ++d)
      {
        const int pos_v = (int)bs * dofs[d];

        for (std::size_t q = 0; q < num_points; ++q)
          for (std::size_t k = 0; k < bs; ++k)
            for (int j = 0; j < vs; j++)
              coefficients[cstride * i + q * (bs * vs) + k + j]
                  += basis_values(q, d, j) * data[pos_v + k];
      }
    }
  }
  else
  {
    for (std::size_t i = 0; i < num_active_entities; i++)
    {
      auto [cell, entity_index] = get_cell_info(i);
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t d = 0; d < dofs.size(); ++d)
      {
        const int pos_v = (int)bs * dofs[d];

        for (std::size_t q = 0; q < num_points; ++q)
          for (std::size_t k = 0; k < bs; ++k)
            for (int l = 0; l < vs; l++)
            {
              coefficients[cstride * i + q * (bs * vs) + k + l]
                  += reference_basis_values(entity_index, q, d, l)
                     * data[pos_v + k];
            }
      }
    }
  }
  return {std::move(coefficients), cstride};
}
//-----------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int> dolfinx_contact::pack_circumradius(
    const dolfinx::mesh::Mesh& mesh,
    const tcb::span<const std::pair<std::int32_t, int>>& active_facets)
{
  if (!mesh.geometry().cmap().is_affine())
    throw std::runtime_error("Non-affine circumradius is not implemented");

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  const dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();

  // NOTE: This is not correct for non-affine geometries, then the quadrature
  // rule has to be passed in Quadrature points for piecewise constant
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  const dolfinx::mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  const int gdim = geometry.dim();
  const int fdim = tdim - 1;
  dolfinx_cuas::QuadratureRule q_rule(cell_type, 0, fdim);

  // FIXME: This does not work for prism elements
  const std::vector<double> weights = q_rule.weights()[0];
  const std::vector<xt::xarray<double>> points = q_rule.points();

  const std::size_t num_points = weights.size();
  const std::size_t num_local_facets = points.size();

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  xtl::span<const double> x_g = geometry.x();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Prepare geometry data structures
  xt::xtensor<double, 3> J = xt::zeros<double>(
      {std::size_t(1), (std::size_t)gdim, (std::size_t)tdim});
  xt::xtensor<double, 1> detJ = xt::zeros<double>({std::size_t(1)});
  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, (std::size_t)gdim});

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
  circumradius.reserve(active_facets.size());

  // Create tmp array to host dphi
  xt::xtensor<double, 2> dphi({(std::size_t)tdim, num_dofs_g});
  for (auto [cell, local_index] : active_facets)
  {
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
    J.fill(0);
    auto _J = xt::view(J, 0, xt::all(), xt::all());

    dphi = xt::view(dphi_c, local_index, xt::all(), 0, xt::all(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                      _J);
    detJ[0] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);

    // Sum up quadrature contributions
    // NOTE: Consider refactoring (moving in Jacobian computation when we start
    // supporting non-affine geoemtries)
    circumradius.push_back(
        compute_circumradius(mesh, detJ[0], coordinate_dofs));
  }
  return {std::move(circumradius), 1};
}