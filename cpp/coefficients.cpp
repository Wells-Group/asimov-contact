// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "coefficients.h"
#include "QuadratureRule.h"
#include "error_handling.h"
#include "geometric_quantities.h"
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>
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

  // Get topology data
  const dolfinx::mesh::Topology& topology = mesh->topology();
  const int tdim = topology.dim();
  const dolfinx::mesh::CellType cell_type = topology.cell_type();

  // Get what entity type we are integrating over
  int entity_dim;
  std::visit(
      [&entity_dim, tdim](auto& entities)
      {
        using U = std::decay_t<decltype(entities)>;
        if constexpr (std::is_same_v<U, tcb::span<const std::int32_t>>)
          entity_dim = tdim;
        else if constexpr (std::is_same_v<
                               U,
                               tcb::span<const std::pair<std::int32_t, int>>>)
        {
          entity_dim = tdim - 1;
        }
        else
        {
          throw std::invalid_argument(
              "Could not pack coefficients. Input entity "
              "type is not supported.");
        }
      },
      active_entities);

  // Create quadrature rule
  QuadratureRule q_rule(cell_type, q_degree, entity_dim);

  // Get element information
  const dolfinx::fem::FiniteElement* element
      = coeff->function_space()->element().get();
  const std::size_t bs = element->block_size();
  const std::size_t value_size = element->value_size();

  // Tabulate function at quadrature points (assuming no derivatives)
  dolfinx_contact::error::check_cell_type(cell_type);
  const xt::xtensor<double, 2>& q_points = q_rule.points();

  const basix::FiniteElement& basix_element = element->basix_element();
  std::array<std::size_t, 4> tab_shape
      = basix_element.tabulate_shape(0, q_points.shape(0));
  const std::size_t num_basis_functions = tab_shape[2];
  const std::size_t vs = tab_shape[3];
  assert(value_size / bs == vs);
  xt::xtensor<double, 4> reference_basis_values(tab_shape);
  element->tabulate(reference_basis_values, q_points, 0);

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
          throw std::invalid_argument(
              "Could not pack coefficient. Input entity "
              "type is not supported.");
        }
      },
      active_entities);

  // Create output array
  const std::vector<std::int32_t>& q_offsets = q_rule.offset();
  const std::size_t num_points_per_entity = q_offsets[1] - q_offsets[0];
  const auto cstride = int(value_size * num_points_per_entity);
  std::vector<PetscScalar> coefficients(num_active_entities * cstride, 0.0);

  // Get the coeffs to pack
  const xtl::span<const double> data = coeff->x()->array();

  // Get dofmap info
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const std::size_t dofmap_bs = dofmap->bs();

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

  if (needs_dof_transformations)
  {
    const auto num_points = (std::size_t)q_offsets.back();

    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values(
        {num_points_per_entity, num_basis_functions, vs});
    xt::xtensor<double, 2> element_basis_values({num_basis_functions, vs});

    // Get geometry data
    const dolfinx::mesh::Geometry& geometry = mesh->geometry();
    const int gdim = geometry.dim();
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();
    const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
    const std::size_t num_dofs_g = cmap.dim();
    xtl::span<const double> x_g = geometry.x();

    // Tabulate coordinate basis to compute Jacobian
    std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, num_points);
    xt::xtensor<double, 3> cmap_derivative(
        {(std::size_t)tdim, c_shape[1], c_shape[2]});
    {
      xt::xtensor<double, 4> cmap_basis_functions(c_shape);
      cmap.tabulate(1, q_points, cmap_basis_functions);
      cmap_derivative
          = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                     xt::all(), xt::all(), 0);
    }
    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points_per_entity, (std::size_t)tdim});
    xt::xtensor<double, 2> J
        = xt::zeros<double>({(std::size_t)gdim, (std::size_t)tdim});
    xt::xtensor<double, 2> K
        = xt::zeros<double>({(std::size_t)tdim, (std::size_t)gdim});
    xt::xtensor<double, 2> coordinate_dofs
        = xt::zeros<double>({num_dofs_g, (std::size_t)gdim});

    // Get push forward function
    using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                          xt::xall<std::size_t>, xt::xall<std::size_t>>;
    using U_t = xt::xview<decltype(element_basis_values)&,
                          xt::xall<std::size_t>, xt::xall<std::size_t>>;
    auto push_forward_fn
        = element->map_fn<u_t, U_t, decltype(J)&, decltype(K)&>();
    xt::xtensor<double, 2> dphi_q({(std::size_t)tdim, tab_shape[2]});

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
      if (cmap.is_affine())
      {
        std::fill(J.begin(), J.end(), 0);
        dphi_q = xt::view(cmap_derivative, xt::all(), q_offsets[entity_index],
                          xt::all());
        dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q,
                                                          coordinate_dofs, J);
        dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
        double detJ
            = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J);

        for (std::size_t q = 0; q < num_points_per_entity; ++q)
        {

          // Permute the reference values to account for the cell's
          // orientation
          element_basis_values
              = xt::view(reference_basis_values, 0, q_offsets[entity_index] + q,
                         xt::all(), xt::all());
          transformation(
              xtl::span(element_basis_values.data(), num_basis_functions * vs),
              cell_info, cell, (int)vs);

          // Push basis forward to physical element
          auto _u = xt::view(basis_values, q, xt::all(), xt::all());
          auto _U = xt::view(element_basis_values, xt::all(), xt::all());
          push_forward_fn(_u, _U, J, detJ, K);
        }
      }
      else
      {
        for (std::size_t q = 0; q < num_points_per_entity; ++q)
        {
          std::fill(J.begin(), J.end(), 0);
          dphi_q = xt::view(cmap_derivative, xt::all(),
                            q_offsets[entity_index] + q, xt::all());
          dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q,
                                                            coordinate_dofs, J);
          dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
          double detJ
              = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
                  J);
          // Permute the reference values to account for the cell's
          // orientation
          element_basis_values
              = xt::view(reference_basis_values, 0, q_offsets[entity_index] + q,
                         xt::all(), xt::all());
          transformation(
              xtl::span(element_basis_values.data(), num_basis_functions * vs),
              cell_info, cell, (int)vs);

          // Push basis forward to physical element
          auto _u = xt::view(basis_values, q, xt::all(), xt::all());
          auto _U = xt::view(element_basis_values, xt::all(), xt::all());
          push_forward_fn(_u, _U, J, detJ, K);
        }
      }
      // Sum up quadrature contributions
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t d = 0; d < dofs.size(); ++d)
      {
        const int pos_v = (int)dofmap_bs * dofs[d];

        for (std::size_t q = 0; q < num_points_per_entity; ++q)
          for (std::size_t k = 0; k < dofmap_bs; ++k)
            for (int j = 0; j < vs; j++)
              coefficients[cstride * i + q * value_size + k + j]
                  += basis_values(q, d, j) * data[pos_v + k];
      }
    }
  }
  else
  {
    // Loop over all entities
    for (std::size_t i = 0; i < num_active_entities; i++)
    {
      auto [cell, entity_index] = get_cell_info(i);
      auto dofs = dofmap->cell_dofs(cell);
      const std::int32_t q_offset = q_offsets[entity_index];

      // Loop over all dofs in cell
      for (std::size_t d = 0; d < dofs.size(); ++d)
      {
        const int pos_v = (int)dofmap_bs * dofs[d];
        // Unroll dofmap
        for (std::size_t b = 0; b < dofmap_bs; ++b)
        {
          auto coeff = data[pos_v + b];
          std::div_t pos = std::div(int(d * dofmap_bs + b), (int)bs);

          // Pack coefficients for each quadrature point
          for (std::size_t q = 0; q < num_points_per_entity; ++q)
          {
            // Access each component of the reference basis function (in the
            // case of vector spaces)
            for (int l = 0; l < vs; ++l)
            {
              coefficients[cstride * i + q * bs * vs + l + pos.rem]
                  += reference_basis_values(0, q_offset + q, pos.quot, l)
                     * coeff;
            }
          }
        }
      }
    }
  }
  return {std::move(coefficients), cstride};
}
//-----------------------------------------------------------------------------
std::vector<PetscScalar> dolfinx_contact::pack_circumradius(
    const dolfinx::mesh::Mesh& mesh,
    const tcb::span<const std::pair<std::int32_t, int>>& active_facets)
{
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  const dolfinx::mesh::Topology& topology = mesh.topology();
  if (!geometry.cmap().is_affine())
    throw std::invalid_argument("Non-affine circumradius is not implemented");

  // Tabulate element at quadrature points
  const dolfinx::mesh::CellType cell_type = topology.cell_type();
  dolfinx_contact::error::check_cell_type(cell_type);

  const int tdim = topology.dim();
  const int fdim = tdim - 1;
  const dolfinx_contact::QuadratureRule q_rule(cell_type, 0, fdim);
  const xt::xtensor<double, 2>& q_points = q_rule.points();
  const std::size_t num_q_points = q_points.shape(0);
  const std::vector<std::int32_t> q_offset = q_rule.offset();

  // Tabulate coordinate basis for Jacobian computation
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  std::array<std::size_t, 4> tab_shape = cmap.tabulate_shape(1, num_q_points);
  xt::xtensor<double, 4> coordinate_basis(tab_shape);
  assert(tab_shape[3] == 1);
  xt::xtensor<double, 3> dphi({(std::size_t)tdim, tab_shape[1], tab_shape[2]});
  cmap.tabulate(1, q_points, coordinate_basis);
  dphi = xt::view(coordinate_basis, xt::xrange(1, tdim + 1), xt::all(),
                  xt::all(), 0);

  // Prepare output variables
  std::vector<PetscScalar> circumradius;
  circumradius.reserve(active_facets.size());

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  xtl::span<const double> x_g = geometry.x();

  // Prepare temporary data structures data structures
  const int gdim = geometry.dim();
  const std::size_t num_dofs_g = cmap.dim();
  xt::xtensor<double, 2> J
      = xt::zeros<double>({(std::size_t)gdim, (std::size_t)tdim});

  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, (std::size_t)gdim});
  xt::xtensor<double, 2> dphi_q({(std::size_t)tdim, num_dofs_g});
  assert(num_dofs_g == tab_shape[2]);
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

    // Compute determinant of Jacobian which is used to compute the
    // area/volume of the cell
    std::fill(J.begin(), J.end(), 0);
    assert(q_offset[local_index + 1] - q_offset[local_index] == 1);
    dphi_q = xt::view(dphi, xt::all(), q_offset[local_index], xt::all());
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q, coordinate_dofs,
                                                      J);
    double detJ
        = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J);
    circumradius.push_back(compute_circumradius(mesh, detJ, coordinate_dofs));
  }
  assert(circumradius.size() == active_facets.size());
  return circumradius;
}