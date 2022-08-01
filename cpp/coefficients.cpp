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
using namespace dolfinx_contact;

std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree, std::span<const std::int32_t> active_entities,
    dolfinx::fem::IntegralType integral)
{
  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh
      = coeff->function_space()->mesh();
  assert(mesh);

  // Get topology data
  const dolfinx::mesh::Topology& topology = mesh->topology();
  const std::size_t tdim = topology.dim();
  const dolfinx::mesh::CellType cell_type = topology.cell_type();

  // Get what entity type we are integrating over
  std::size_t entity_dim;
  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
    entity_dim = tdim;
    break;
  case dolfinx::fem::IntegralType::exterior_facet:
    entity_dim = tdim - 1;
    break;
  default:
    throw std::invalid_argument("Unsupported integral type.");
  }
  // Create quadrature rule
  QuadratureRule q_rule(cell_type, q_degree, entity_dim);

  // Get element information
  const dolfinx::fem::FiniteElement* element
      = coeff->function_space()->element().get();
  const std::size_t bs = element->block_size();
  const std::size_t value_size = element->value_size();

  // Tabulate function at quadrature points (assuming no derivatives)
  dolfinx_contact::error::check_cell_type(cell_type);
  const std::vector<double>& q_points = q_rule.points();
  const std::vector<std::size_t>& q_offset = q_rule.offset();
  const std::size_t sum_q_points = q_offset.back();
  const std::size_t num_points_per_entity = q_offset[1] - q_offset[0];
  std::array<std::size_t, 2> p_shape = {sum_q_points, tdim};
  assert(q_rule.tdim() == tdim);

  const basix::FiniteElement& basix_element = element->basix_element();
  std::array<std::size_t, 4> tab_shape
      = basix_element.tabulate_shape(0, sum_q_points);
  std::vector<double> reference_basisb(
      std::reduce(tab_shape.cbegin(), tab_shape.cend(), 1, std::multiplies{}));
  element->tabulate(reference_basisb, q_points, p_shape, 0);
  cmdspan4_t reference_basis(reference_basisb.data(), tab_shape);

  assert(value_size / bs == tab_shape[3]);

  std::function<std::array<std::int32_t, 2>(std::size_t)> get_cell_info;
  std::size_t num_active_entities;
  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
    num_active_entities = active_entities.size();
    break;
  case dolfinx::fem::IntegralType::exterior_facet:
    num_active_entities = active_entities.size() / 2;
    break;
  default:
    throw std::invalid_argument("Unsupported integral type.");
  }

  // Create output array
  const std::vector<std::size_t>& q_offsets = q_rule.offset();
  const auto cstride = int(value_size * num_points_per_entity);
  std::vector<PetscScalar> coefficients(num_active_entities * cstride, 0.0);

  // Get the coeffs to pack
  const std::span<const double> data = coeff->x()->array();

  // Get dofmap info
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const std::size_t dofmap_bs = dofmap->bs();

  // Get dof transformations
  const bool needs_dof_transformations = element->needs_dof_transformations();
  std::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = topology.get_cell_permutation_info();
  }
  const std::function<void(const std::span<PetscScalar>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  if (needs_dof_transformations)
  {
    const auto num_points = q_offsets.back();

    // Get geometry data
    const dolfinx::mesh::Geometry& geometry = mesh->geometry();
    const int gdim = geometry.dim();
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();
    const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const double> x_g = geometry.x();

    // Tabulate coordinate basis to compute Jacobian
    std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, num_points);
    std::vector<double> c_basisb(
        std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{}));
    cmap.tabulate(1, q_points, p_shape, c_basisb);
    cmdspan4_t c_basis(c_basisb.data(), c_shape);

    // Prepare geometry data structures
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::vector<double> detJ_scratch(2 * gdim * tdim);
    std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
    mdspan2_t coordinate_dofs(coordinate_dofsb.data(), num_dofs_g, gdim);

    // Prepare transformation function
    std::vector<double> element_basisb(tab_shape[2] * tab_shape[3]);
    mdspan2_t element_basis(element_basisb.data(), tab_shape[2], tab_shape[3]);
    std::vector<double> basis_valuesb(num_points_per_entity * tab_shape[2]
                                      * tab_shape[3]);
    mdspan3_t basis_values(basis_valuesb.data(), num_points_per_entity,
                           tab_shape[2], tab_shape[3]);

    // Get push forward function
    using xu_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
    using xU_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    using xJ_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    using xK_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
    auto push_forward_fn
        = element->basix_element().map_fn<xu_t, xU_t, xJ_t, xK_t>();

    for (std::size_t i = 0; i < num_active_entities; i++)
    {
      // Get local cell info
      std::int32_t cell;
      std::int32_t entity_index;
      switch (integral)
      {
      case dolfinx::fem::IntegralType::cell:
        cell = active_entities[i];
        entity_index = 0;
        break;
      case dolfinx::fem::IntegralType::exterior_facet:
        cell = active_entities[2 * i];
        entity_index = active_entities[2 * i + 1];
        break;
      default:
        throw std::invalid_argument("Unsupported integral type.");
      }

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t j = 0; j < num_dofs_g; ++j)
      {
        auto pos = 3 * x_dofs[j];
        for (std::size_t k = 0; k < coordinate_dofs.extent(1); ++k)
          coordinate_dofs(j, k) = x_g[pos + k];
      }

      if (cmap.is_affine())
      {
        std::fill(Jb.begin(), Jb.end(), 0);
        auto dphi_q
            = stdex::submdspan(c_basis, std::pair{1, std::size_t(tdim + 1)},
                               q_offsets[entity_index], stdex::full_extent, 0);
        dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q,
                                                          coordinate_dofs, J);
        dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
        double detJ
            = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
                J, detJ_scratch);

        for (std::size_t q = 0; q < num_points_per_entity; ++q)
        {

          // Copy basis values prior to calling transformation
          for (std::size_t j = 0; j < element_basis.extent(0); ++j)
            for (std::size_t k = 0; k < element_basis.extent(1); ++k)
            {
              element_basis(j, k)
                  = reference_basis(0, q_offsets[entity_index] + q, j, k);
            }

          // Permute the reference values to account for the cell's
          // orientation
          transformation(element_basisb, cell_info, cell, (int)tab_shape[3]);

          // Push basis forward to physical element
          auto _u = stdex::submdspan(basis_values, q, stdex::full_extent,
                                     stdex::full_extent);
          push_forward_fn(_u, element_basis, J, detJ, K);
        }
      }
      else
      {
        for (std::size_t q = 0; q < num_points_per_entity; ++q)
        {
          std::fill(Jb.begin(), Jb.end(), 0);
          auto dphi_q = stdex::submdspan(
              c_basis, std::pair{1, std::size_t(tdim + 1)},
              q_offsets[entity_index] + q, stdex::full_extent, 0);
          dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q,
                                                            coordinate_dofs, J);
          dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
          double detJ
              = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
                  J, detJ_scratch);

          // Copy basis values prior to calling transformation
          for (std::size_t j = 0; j < element_basis.extent(0); ++j)
            for (std::size_t k = 0; k < element_basis.extent(1); ++k)
            {
              element_basis(j, k)
                  = reference_basis(0, q_offsets[entity_index] + q, j, k);
            }

          // Permute the reference values to account for the cell's
          // orientation
          transformation(element_basisb, cell_info, cell, (int)tab_shape[3]);

          // Push basis forward to physical element
          // Push basis forward to physical element
          auto _u = stdex::submdspan(basis_values, q, stdex::full_extent,
                                     stdex::full_extent);
          push_forward_fn(_u, element_basis, J, detJ, K);
        }
      }
      // Sum up quadrature contributions
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t d = 0; d < dofs.size(); ++d)
      {
        const int pos_v = (int)dofmap_bs * dofs[d];

        for (std::size_t q = 0; q < num_points_per_entity; ++q)
          for (std::size_t k = 0; k < dofmap_bs; ++k)
            for (std::size_t j = 0; j < tab_shape[3]; j++)
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
      // Get local cell info
      std::int32_t cell;
      std::int32_t entity_index;
      switch (integral)
      {
      case dolfinx::fem::IntegralType::cell:
        cell = active_entities[i];
        entity_index = 0;
        break;
      case dolfinx::fem::IntegralType::exterior_facet:
        cell = active_entities[2 * i];
        entity_index = active_entities[2 * i + 1];
        break;
      default:
        throw std::invalid_argument("Unsupported integral type.");
      }

      auto dofs = dofmap->cell_dofs(cell);
      const std::size_t q_offset = q_offsets[entity_index];

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
            for (std::size_t l = 0; l < tab_shape[3]; ++l)
            {
              coefficients[cstride * i + q * bs * tab_shape[3] + l + pos.rem]
                  += reference_basis(0, q_offset + q, pos.quot, l) * coeff;
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
    const std::span<const std::int32_t>& active_facets)
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

  // Get quadrature points on reference facets
  const std::vector<double>& q_points = q_rule.points();
  const std::vector<std::size_t>& q_offset = q_rule.offset();
  const std::size_t sum_q_points = q_offset.back();
  const std::array<std::size_t, 2> q_shape = {sum_q_points, (std::size_t)tdim};
  assert(q_rule.tdim() == (std::size_t)tdim);

  // Tabulate coordinate basis for Jacobian computation
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::array<std::size_t, 4> tab_shape
      = cmap.tabulate_shape(1, sum_q_points);
  std::vector<double> coordinate_basisb(
      std::reduce(tab_shape.cbegin(), tab_shape.cend(), 1, std::multiplies{}));
  assert(tab_shape.back() == 1);
  cmap.tabulate(1, q_points, q_shape, coordinate_basisb);
  cmdspan4_t coordinate_basis(coordinate_basisb.data(), tab_shape);

  // Prepare output variables
  std::vector<PetscScalar> circumradius;
  circumradius.reserve(active_facets.size() / 2);

  // Get geometry data
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  std::span<const double> x_g = geometry.x();

  // Prepare temporary data structures data structures
  const int gdim = geometry.dim();
  const std::size_t num_dofs_g = cmap.dim();

  std::vector<double> Jb(gdim * tdim);
  std::vector<double> Kb(tdim * gdim);
  mdspan2_t J(Jb.data(), gdim, tdim);
  mdspan2_t K(Kb.data(), tdim, gdim);
  std::vector<double> detJ_scratch(2 * gdim * tdim);
  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  mdspan2_t coordinate_dofs(coordinate_dofsb.data(), num_dofs_g, gdim);

  assert(num_dofs_g == tab_shape[2]);
  for (std::size_t i = 0; i < active_facets.size(); i += 2)
  {
    std::int32_t cell = active_facets[i];
    std::int32_t local_index = active_facets[i + 1];
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofsb.begin(), j * gdim));
    }

    // Compute determinant of Jacobian which is used to compute the
    // area/volume of the cell
    std::fill(Jb.begin(), Jb.end(), 0);
    assert(q_offset[local_index + 1] - q_offset[local_index] == 1);
    auto dphi_q = stdex::submdspan(
        coordinate_basis, std::pair{1, (std::size_t)tdim + 1},
        q_offset[local_index], stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q, coordinate_dofs,
                                                      J);
    double detJ = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
        J, detJ_scratch);
    circumradius.push_back(compute_circumradius(mesh, detJ, coordinate_dofs));
  }
  assert(circumradius.size() == active_facets.size() / 2);
  return circumradius;
}