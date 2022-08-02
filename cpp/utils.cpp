// Copyright (C) 2021-2022 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "utils.h"
#include "RayTracing.h"
#include "error_handling.h"
#include "geometric_quantities.h"
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>

//-----------------------------------------------------------------------------
void dolfinx_contact::pull_back(mdspan3_t J, dolfinx_contact::mdspan3_t K,
                                std::span<double> detJ, std::span<double> X,
                                dolfinx_contact::cmdspan2_t x,
                                dolfinx_contact::cmdspan2_t coordinate_dofs,
                                const dolfinx::fem::CoordinateElement& cmap)
{
  const std::size_t num_points = x.extent(0);
  assert(J.extent(0) >= num_points);
  assert(K.extent(0) >= num_points);
  assert(detJ.size() >= num_points);

  const size_t tdim = K.extent(1);
  const std::size_t gdim = K.extent(2);

  // Create working memory for determinant computation
  std::vector<double> detJ_scratch(2 * gdim * tdim);
  if (cmap.is_affine())
  {
    // Tabulate at reference coordinate origin
    std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, 1);
    std::vector<double> data(
        std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{}));
    std::array<double, 3> X0;
    cmap.tabulate(1, std::span(X0.data(), tdim), {1, tdim}, data);
    dolfinx_contact::cmdspan4_t c_basis(data.data(), c_shape);

    namespace stdex = std::experimental;
    auto dphi0 = stdex::submdspan(c_basis, std::pair{1, tdim + 1}, 0,
                                  stdex::full_extent, 0);

    // Only zero out first Jacobian as it is used to fill in the others
    for (std::size_t j = 0; j < J.extent(1); ++j)
      for (std::size_t k = 0; k < J.extent(2); ++k)
        J(0, j, k) = 0;

    // Compute Jacobian at origin of reference element
    auto J0 = stdex::submdspan(J, 0, stdex::full_extent, stdex::full_extent);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi0, coordinate_dofs,
                                                      J0);

    for (std::size_t j = 0; j < K.extent(1); ++j)
      for (std::size_t k = 0; k < K.extent(2); ++k)
        K(0, j, k) = 0;
    // Compute inverse Jacobian
    auto K0 = stdex::submdspan(K, 0, stdex::full_extent, stdex::full_extent);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J0, K0);

    // Compute determinant
    detJ[0] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
        J0, detJ_scratch);

    // Pull back all physical coordinates
    std::array<double, 3> x0 = {0, 0, 0};
    for (std::size_t i = 0; i < coordinate_dofs.extent(1); ++i)
      x0[i] += coordinate_dofs(0, i);
    dolfinx_contact::mdspan2_t Xs(X.data(), num_points, tdim);
    dolfinx::fem::CoordinateElement::pull_back_affine(Xs, K0, x0, x);

    // Copy Jacobian, inverse and determinant to all other inputs
    for (std::size_t p = 1; p < num_points; ++p)
    {
      for (std::size_t j = 0; j < J.extent(1); ++j)
        for (std::size_t k = 0; k < J.extent(2); ++k)
          J(p, j, k) = J0(j, k);
      for (std::size_t j = 0; j < K.extent(1); ++j)
        for (std::size_t k = 0; k < K.extent(2); ++k)
          K(p, j, k) = K0(j, k);
      detJ[p] = detJ[0];
    }
  }
  else
  {
    for (std::size_t i = 0; i < J.extent(0); ++i)
      for (std::size_t j = 0; j < J.extent(1); ++j)
        for (std::size_t k = 0; k < J.extent(2); ++k)
          J(i, j, k) = 0;

    dolfinx_contact::mdspan2_t Xs(X.data(), num_points, tdim);
    cmap.pull_back_nonaffine(Xs, x, coordinate_dofs);

    /// Tabulate coordinate basis at pull back points to compute the Jacobian,
    /// inverse and determinant

    const std::array<std::size_t, 4> c_shape
        = cmap.tabulate_shape(1, num_points);
    std::vector<double> basis_buffer(
        std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{}));
    cmap.tabulate(1, X, {num_points, tdim}, basis_buffer);
    dolfinx_contact::cmdspan4_t c_basis(basis_buffer.data(), c_shape);

    for (std::size_t p = 0; p < num_points; ++p)
    {
      for (std::size_t j = 0; j < J.extent(1); ++j)
        for (std::size_t k = 0; k < J.extent(2); ++k)
          J(p, j, k) = 0;
      auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
      auto dphi = stdex::submdspan(c_basis, std::pair{1, tdim + 1}, p,
                                   stdex::full_extent, 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);
      auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
      detJ[p] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
          _J, detJ_scratch);
    }
  }
}

//-----------------------------------------------------------------------------
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
dolfinx_contact::sort_cells(const std::span<const std::int32_t>& cells,
                            const std::span<std::int32_t>& perm)
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

//-------------------------------------------------------------------------------------
void dolfinx_contact::update_geometry(
    const dolfinx::fem::Function<PetscScalar>& u,
    std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace> V = u.function_space();
  assert(V);
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  assert(dofmap);
  // Check that mesh to be updated and underlying mesh of u are the same
  assert(mesh == V->mesh());

  // The Function and the mesh must have identical element_dof_layouts
  // (up to the block size)
  assert(dofmap->element_dof_layout()
         == mesh->geometry().cmap().create_dof_layout());

  const int tdim = mesh->topology().dim();
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = mesh->topology().index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();

  // Get dof array and retrieve u at the mesh dofs
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap_x
      = mesh->geometry().dofmap();
  const int bs = dofmap->bs();
  const auto& u_data = u.x()->array();
  std::span<double> coords = mesh->geometry().x();
  std::vector<double> dx(coords.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const std::span<const int> dofs = dofmap->cell_dofs(c);
    const std::span<const int> dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int j = 0; j < bs; ++j)
      {
        dx[3 * dofs_x[i] + j] = u_data[bs * dofs[i] + j];
      }
  }
  // add u to mesh dofs
  std::transform(coords.begin(), coords.end(), dx.begin(), coords.begin(),
                 std::plus<double>());
}
//-------------------------------------------------------------------------------------
double dolfinx_contact::R_plus(double x) { return 0.5 * (std::abs(x) + x); }
//-------------------------------------------------------------------------------------
double dolfinx_contact::R_minus(double x) { return 0.5 * (x - std::abs(x)); }
//-------------------------------------------------------------------------------------
double dolfinx_contact::dR_minus(double x) { return double(x < 0); }
//-------------------------------------------------------------------------------------

double dolfinx_contact::dR_plus(double x) { return double(x > 0); }
//-------------------------------------------------------------------------------------
std::array<std::size_t, 4>
dolfinx_contact::evaluate_basis_shape(const dolfinx::fem::FunctionSpace& V,
                                      const std::size_t num_points,
                                      const std::size_t num_derivatives)
{
  // Get element
  assert(V.element());
  std::size_t gdim = V.mesh()->geometry().dim();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t value_size = element->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;
  return {num_derivatives * gdim + 1, num_points, space_dimension, value_size};
}
//-----------------------------------------------------------------------------
void dolfinx_contact::evaluate_basis_functions(
    const dolfinx::fem::FunctionSpace& V, std::span<const double> x,
    std::span<const std::int32_t> cells, std::span<double> basis_values,
    std::size_t num_derivatives)
{

  assert(num_derivatives < 2);

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V.mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const dolfinx::mesh::Topology& topology = mesh->topology();
  const std::size_t tdim = topology.dim();
  const std::size_t gdim = geometry.dim();
  const std::size_t num_cells = cells.size();
  if (x.size() / gdim != num_cells)
  {
    throw std::invalid_argument(
        "Number of points and number of cells must be equal.");
  }

  if (x.size() == 0)
    return;

  // Get topology data
  std::shared_ptr<const dolfinx::common::IndexMap> map
      = topology.index_map((int)tdim);

  // Get geometry data
  std::span<const double> x_g = geometry.x();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Get element
  assert(V.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the sub
  // elements
  if (const int num_sub_elements = element->num_sub_elements();
      num_sub_elements > 1 && num_sub_elements != bs_element)
  {
    throw std::invalid_argument("Canot evaluate basis functions for mixed "
                                "function spaces. Extract subspaces.");
  }

  // Get dofmap
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  assert(dofmap);

  std::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(topology.get_cell_permutation_info());
  }

  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  dolfinx_contact::mdspan2_t coordinate_dofs(coordinate_dofsb.data(),
                                             num_dofs_g, gdim);

  // -- Lambda function for affine pull-backs
  const std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, 1);
  std::vector<double> datab(
      std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{}));
  std::array<double, 3> X0;
  cmap.tabulate(1, std::span(X0.data(), tdim), {1, tdim}, datab);
  dolfinx_contact::cmdspan4_t data(datab.data(), c_shape);
  auto dphi_0 = stdex::submdspan(data, std::pair{1, tdim + 1}, 0,
                                 stdex::full_extent, 0);
  auto pull_back_affine = [&dphi_0, x0 = std::array<double, 3>({0, 0, 0})](
                              auto&& X, const auto& cell_geometry, auto&& J,
                              auto&& K, const auto& x) mutable
  {
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_0, cell_geometry, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    for (std::size_t i = 0; i < cell_geometry.extent(1); ++i)
      x0[i] = cell_geometry(0, i);
    dolfinx::fem::CoordinateElement::pull_back_affine(X, K, x0, x);
  };

  // Create buffer for pull back
  std::vector<double> Xb(num_cells * tdim);
  std::vector<double> Jb(num_cells * gdim * tdim, 0);
  std::vector<double> Kb(num_cells * gdim * tdim);
  std::vector<double> detJ(num_cells);
  dolfinx_contact::mdspan3_t J(Jb.data(), num_cells, gdim, tdim);
  dolfinx_contact::mdspan3_t K(Kb.data(), num_cells, tdim, gdim);
  std::vector<double> detJ_scratch(2 * gdim * tdim);
  std::vector<double> basisb(
      std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{}));
  dolfinx_contact::cmdspan4_t basis(basisb.data(), c_shape);
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;
    assert(cell_index < x_dofmap.num_nodes());

    // Get cell geometry (coordinate dofs)
    const std::span<const int> x_dofs = x_dofmap.links(cell_index);
    for (std::size_t j = 0; j < num_dofs_g; ++j)
    {
      auto pos = 3 * x_dofs[j];
      for (std::size_t k = 0; k < coordinate_dofs.extent(1); ++k)
        coordinate_dofs(j, k) = x_g[pos + k];
    }

    auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
    auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
    dolfinx_contact::mdspan2_t Xp(Xb.data() + p * tdim, 1, tdim);
    dolfinx_contact::cmdspan2_t xp(x.data() + p * gdim, 1, gdim);
    // Compute reference coordinates X, and J, detJ and K
    if (cmap.is_affine())
    {
      pull_back_affine(Xp, coordinate_dofs, _J, _K, xp);
      detJ[p] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
          _J, detJ_scratch);
      assert(std::fabs(detJ[p]) > 1e-10);
    }
    else
    {
      cmap.pull_back_nonaffine(Xp, xp, coordinate_dofs);
      cmap.tabulate(1, std::span(Xb.data() + p * tdim, gdim), {1, tdim},
                    basisb);
      auto dphi = stdex::submdspan(basis, std::pair{1, tdim + 1}, 0,
                                   stdex::full_extent, 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
      detJ[p] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
          _J, detJ_scratch);
      assert(std::fabs(detJ[p]) > 1e-10);
    }
  }

  // Prepare basis function data structures
  const std::array<std::size_t, 4> reference_shape
      = element->basix_element().tabulate_shape(num_derivatives, num_cells);
  std::vector<double> basis_reference_valuesb(std::reduce(
      reference_shape.cbegin(), reference_shape.cend(), 1, std::multiplies{}));

  // Compute basis on reference element
  element->tabulate(basis_reference_valuesb, Xb, {num_cells, tdim},
                    (int)num_derivatives);
  dolfinx_contact::cmdspan4_t basis_reference_values(
      basis_reference_valuesb.data(), reference_shape);

  // We need a temporary data structure to apply push forward
  std::array<std::size_t, 4> shape
      = {1, num_cells, space_dimension, reference_value_size};
  if (num_derivatives == 1)
    shape[0] = tdim + 1;
  std::vector<double> tempb(
      std::reduce(shape.cbegin(), shape.cend(), 1, std::multiplies{}));
  dolfinx_contact::mdspan4_t temp(tempb.data(), shape);

  dolfinx_contact::mdspan4_t basis_span(basis_values.data(), shape);
  std::fill(basis_values.begin(), basis_values.end(), 0);

  using xu_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using xU_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using xJ_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using xK_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  auto push_forward_fn
      = element->basix_element().map_fn<xu_t, xU_t, xJ_t, xK_t>();
  const std::function<void(const std::span<double>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = element->get_dof_transformation_function<double>();
  const std::size_t num_basis_values = space_dimension * reference_value_size;

  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    auto _J = stdex::submdspan(J, p, stdex::full_extent, stdex::full_extent);
    auto _K = stdex::submdspan(K, p, stdex::full_extent, stdex::full_extent);
    /// NOTE: loop size correct for num_derivatives = 0,1
    for (std::size_t j = 0; j < num_derivatives * tdim + 1; ++j)
    {
      const int cell_index = cells[p];

      // Skip negative cell indices
      if (cell_index < 0)
        continue;

      // Permute the reference values to account for the cell's orientation
      apply_dof_transformation(
          std::span(basis_reference_valuesb.data()
                        + j * cells.size() * num_basis_values
                        + p * num_basis_values,
                    num_basis_values),
          cell_info, cell_index, (int)reference_value_size);

      // Push basis forward to physical element
      auto _U = stdex::submdspan(basis_reference_values, j, p,
                                 stdex::full_extent, stdex::full_extent);

      if (j == 0)
      {
        auto _u = stdex::submdspan(basis_span, j, p, stdex::full_extent,
                                   stdex::full_extent);
        push_forward_fn(_u, _U, _J, detJ[p], _K);
      }
      else
      {
        auto _u = stdex::submdspan(temp, j, p, stdex::full_extent,
                                   stdex::full_extent);
        push_forward_fn(_u, _U, _J, detJ[p], _K);
      }
    }

    for (std::size_t k = 0; k < gdim * num_derivatives; ++k)
    {
      auto du = stdex::submdspan(basis_span, k + 1, p, stdex::full_extent,
                                 stdex::full_extent);
      for (std::size_t j = 0; j < num_derivatives * tdim; ++j)
      {
        auto du_temp = stdex::submdspan(temp, j + 1, p, stdex::full_extent,
                                        stdex::full_extent);
        for (std::size_t m = 0; m < du.extent(0); ++m)
          for (std::size_t n = 0; n < du.extent(1); ++n)
            du(m, n) += _K(j, k) * du_temp(m, n);
      }
    }
  }
};

double dolfinx_contact::compute_facet_jacobian(
    dolfinx_contact::mdspan2_t J, dolfinx_contact::mdspan2_t K,
    dolfinx_contact::mdspan2_t J_tot, std::span<double> detJ_scratch,
    dolfinx_contact::cmdspan2_t J_f, dolfinx_contact::s_cmdspan2_t dphi,
    dolfinx_contact::cmdspan2_t coords)
{
  std::size_t gdim = J.extent(0);
  auto coordinate_dofs
      = stdex::submdspan(coords, stdex::full_extent, std::pair{0, gdim});
  for (std::size_t i = 0; i < J.extent(0); ++i)
    for (std::size_t j = 0; j < J.extent(1); ++j)
      J(i, j) = 0;
  dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs, J);
  dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
  for (std::size_t i = 0; i < J_tot.extent(0); ++i)
    for (std::size_t j = 0; j < J_tot.extent(1); ++j)
      J_tot(i, j) = 0;
  dolfinx::math::dot(J, J_f, J_tot);
  return std::fabs(
      dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
          J_tot, detJ_scratch));
}
//-------------------------------------------------------------------------------------
std::function<double(
    double, dolfinx_contact::mdspan2_t, dolfinx_contact::mdspan2_t,
    dolfinx_contact::mdspan2_t, std::span<double>, dolfinx_contact::cmdspan2_t,
    dolfinx_contact::s_cmdspan2_t, dolfinx_contact::cmdspan2_t)>
dolfinx_contact::get_update_jacobian_dependencies(
    const dolfinx::fem::CoordinateElement& cmap)
{
  if (cmap.is_affine())
  {
    // Return function that returns the input determinant
    return [](double detJ, [[maybe_unused]] dolfinx_contact::mdspan2_t J,
              [[maybe_unused]] dolfinx_contact::mdspan2_t K,
              [[maybe_unused]] dolfinx_contact::mdspan2_t J_tot,
              [[maybe_unused]] std::span<double> detJ_scratch,
              [[maybe_unused]] dolfinx_contact::cmdspan2_t J_f,
              [[maybe_unused]] dolfinx_contact::s_cmdspan2_t dphi,
              [[maybe_unused]] dolfinx_contact::cmdspan2_t coords)
    { return detJ; };
  }
  else
  {
    // Return function that returns the input determinant
    return [](double detJ, [[maybe_unused]] dolfinx_contact::mdspan2_t J,
              [[maybe_unused]] dolfinx_contact::mdspan2_t K,
              [[maybe_unused]] dolfinx_contact::mdspan2_t J_tot,
              [[maybe_unused]] std::span<double> detJ_scratch,
              [[maybe_unused]] dolfinx_contact::cmdspan2_t J_f,
              [[maybe_unused]] dolfinx_contact::s_cmdspan2_t dphi,
              [[maybe_unused]] dolfinx_contact::cmdspan2_t coords)
    {
      double new_detJ = dolfinx_contact::compute_facet_jacobian(
          J, K, J_tot, detJ_scratch, J_f, dphi, coords);
      return new_detJ;
    };
  }
}
//-------------------------------------------------------------------------------------
std::function<void(std::span<double>, dolfinx_contact::cmdspan2_t,
                   dolfinx_contact::cmdspan2_t, const std::size_t)>
dolfinx_contact::get_update_normal(const dolfinx::fem::CoordinateElement& cmap)
{
  if (cmap.is_affine())
  {
    // Return function that returns the input determinant
    return []([[maybe_unused]] std::span<double> n,
              [[maybe_unused]] dolfinx_contact::cmdspan2_t K,
              [[maybe_unused]] dolfinx_contact::cmdspan2_t n_ref,
              [[maybe_unused]] const std::size_t local_index)
    {
      // Do nothing
    };
  }
  else
  {
    // Return function that updates the physical normal based on K
    return [](std::span<double> n, dolfinx_contact::cmdspan2_t K,
              dolfinx_contact::cmdspan2_t n_ref, const std::size_t local_index)
    {
      std::fill(n.begin(), n.end(), 0);
      auto n_f = stdex::submdspan(n_ref, local_index, stdex::full_extent);
      dolfinx_contact::physical_facet_normal(n, K, n_f);
    };
  }
}
//-------------------------------------------------------------------------------------

/// Compute the active entities in DOLFINx format for a given integral type over
/// a set of entities If the integral type is cell, return the input, if it is
/// exterior facets, return a list of pairs (cell, local_facet_index), and if it
/// is interior facets, return a list of tuples (cell_0, local_facet_index_0,
/// cell_1, local_facet_index_1) for each entity.
/// @param[in] mesh The mesh
/// @param[in] entities List of mesh entities
/// @param[in] integral The type of integral
std::vector<std::int32_t> dolfinx_contact::compute_active_entities(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    std::span<const std::int32_t> entities, dolfinx::fem::IntegralType integral)
{

  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
  {
    std::vector<std::int32_t> active_entities(entities.size());
    std::transform(entities.begin(), entities.end(), active_entities.begin(),
                   [](std::int32_t cell) { return cell; });
    return active_entities;
  }
  case dolfinx::fem::IntegralType::exterior_facet:
  {
    std::vector<std::int32_t> active_entities(2 * entities.size());
    const dolfinx::mesh::Topology& topology = mesh->topology();
    int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    assert(c_to_f);
    for (std::size_t f = 0; f < entities.size(); f++)
    {
      assert(f_to_c->num_links(entities[f]) == 1);
      const std::int32_t cell = f_to_c->links(entities[f])[0];
      auto cell_facets = c_to_f->links(cell);

      auto facet_it
          = std::find(cell_facets.begin(), cell_facets.end(), entities[f]);
      assert(facet_it != cell_facets.end());
      active_entities[2 * f] = cell;
      active_entities[2 * f + 1]
          = (std::int32_t)std::distance(cell_facets.begin(), facet_it);
    }
    return active_entities;
  }
  case dolfinx::fem::IntegralType::interior_facet:
  {
    std::vector<std::int32_t> active_entities(4 * entities.size());
    const dolfinx::mesh::Topology& topology = mesh->topology();
    int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    if (!f_to_c)
      throw std::runtime_error("Facet to cell connectivity missing");
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    if (!c_to_f)
      throw std::runtime_error("Cell to facet connecitivty missing");
    for (std::size_t f = 0; f < entities.size(); f++)
    {
      assert(f_to_c->num_links(entities[f]) == 2);
      auto cells = f_to_c->links(entities[f]);
      for (std::int32_t i = 0; i < 2; i++)
      {
        auto cell_facets = c_to_f->links(cells[i]);
        auto facet_it
            = std::find(cell_facets.begin(), cell_facets.end(), entities[f]);
        assert(facet_it != cell_facets.end());
        active_entities[4 * f + 2 * i] = cells[i];
        active_entities[4 * f + 2 * i + 1]
            = (std::int32_t)std::distance(cell_facets.begin(), facet_it);
      }
    }
    return active_entities;
  }
  default:
    throw std::runtime_error("Unknown integral type");
  }
  return {};
}

//-------------------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_contact::entities_to_geometry_dofs(
    const dolfinx::mesh::Mesh& mesh, int dim,
    const std::span<const std::int32_t>& entity_list)
{

  // Get mesh geometry and topology data
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  const dolfinx::fem::ElementDofLayout layout
      = geometry.cmap().create_dof_layout();
  // FIXME: What does this return for prisms?
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);
  const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();

  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(tdim, dim);

  // Create arrays for the adjacency-list
  std::vector<std::int32_t> geometry_indices(
      num_entity_dofs * entity_list.size(), -1);
  std::vector<std::int32_t> offsets(entity_list.size() + 1, 0);
  for (std::size_t i = 0; i < entity_list.size(); ++i)
    offsets[i + 1] = std::int32_t((i + 1) * num_entity_dofs);

  // Fetch connectivities required to get entity dofs
  const std::vector<std::vector<std::vector<int>>>& closure_dofs
      = layout.entity_closure_dofs_all();
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> e_to_c
      = topology.connectivity(dim, tdim);
  assert(e_to_c);
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_e
      = topology.connectivity(tdim, dim);
  assert(c_to_e);
  for (std::size_t i = 0; i < entity_list.size(); ++i)
  {
    const std::int32_t idx = entity_list[i];
    // Skip if negative cell index
    if (idx < 0)
      continue;
    const std::int32_t cell = e_to_c->links(idx).front();
    const std::span<const int> cell_entities = c_to_e->links(cell);
    auto it = std::find(cell_entities.begin(), cell_entities.end(), idx);
    assert(it != cell_entities.end());
    const auto local_entity = std::distance(cell_entities.begin(), it);
    const std::vector<std::int32_t>& entity_dofs
        = closure_dofs[dim][local_entity];

    auto xc = xdofs.links(cell);
    assert(xc.size() == num_entity_dofs);
    for (std::size_t j = 0; j < num_entity_dofs; ++j)
      geometry_indices[i * num_entity_dofs + j] = xc[entity_dofs[j]];
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(geometry_indices, offsets);
}

//-------------------------------------------------------------------------------------
std::vector<std::int32_t> dolfinx_contact::find_candidate_surface_segment(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    const std::vector<std::int32_t>& puppet_facets,
    const std::vector<std::int32_t>& candidate_facets,
    const double radius = -1.)
{
  if (radius < 0)
  {
    // return all facets for negative radius / no radius
    return std::vector<std::int32_t>(candidate_facets);
  }
  // Find midpoints of puppet and candidate facets
  std::vector<double> puppet_midpoints = dolfinx::mesh::compute_midpoints(
      *mesh, mesh->topology().dim() - 1, puppet_facets);
  std::vector<double> candidate_midpoints = dolfinx::mesh::compute_midpoints(
      *mesh, mesh->topology().dim() - 1, candidate_facets);

  double r2 = radius * radius; // radius squared
  double dist; // used for squared distance between two midpoints
  double diff; // used for squared difference between two coordinates

  std::vector<std::int32_t> cand_patch;

  for (std::size_t i = 0; i < candidate_facets.size(); ++i)
  {
    for (std::size_t j = 0; j < puppet_facets.size(); ++j)
    {
      // compute distance betweeen midpoints of ith candidate facet
      // and jth puppet facet
      dist = 0;
      for (std::size_t k = 0; k < 3; ++k)
      {
        diff = std::abs(puppet_midpoints[j * 3 + k]
                        - candidate_midpoints[i * 3 + k]);
        dist += diff * diff;
      }
      if (dist < r2)
      {
        // if distance < radius add candidate_facet to output
        cand_patch.push_back(candidate_facets[i]);
        // break to avoid adding the same facet more than once
        break;
      }
    }
  }
  return cand_patch;
}

//-------------------------------------------------------------------------------------
void dolfinx_contact::compute_physical_points(
    const dolfinx::mesh::Mesh& mesh, std::span<const std::int32_t> facets,
    std::span<const std::size_t> offsets, dolfinx_contact::cmdspan4_t phi,
    std::span<double> qp_phys)
{
  // Geometrical info
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  std::span<const double> mesh_geometry = geometry.x();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  const int gdim = geometry.dim();

  // Create storage for output quadrature points
  // NOTE: Assume that all facets have the same number of quadrature points
  dolfinx_contact::error::check_cell_type(mesh.topology().cell_type());
  std::size_t num_q_points = offsets[1] - offsets[0];

  dolfinx_contact::mdspan3_t all_qps(qp_phys.data(),
                                     std::size_t(facets.size() / 2),
                                     num_q_points, (std::size_t)gdim);

  // Temporary data array
  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  dolfinx_contact::cmdspan2_t coordinate_dofs(coordinate_dofsb.data(),
                                              num_dofs_g, gdim);
  for (std::size_t i = 0; i < facets.size(); i += 2)
  {
    auto x_dofs = x_dofmap.links(facets[i]);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t j = 0; j < num_dofs_g; ++j)
    {
      std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofsb.begin(), j * gdim));
    }
    // push forward points on reference element
    const std::array<std::size_t, 2> range
        = {offsets[facets[i + 1]], offsets[facets[i + 1] + 1]};
    auto phi_f = stdex::submdspan(phi, (std::size_t)0,
                                  std::pair{range.front(), range.back()},
                                  stdex::full_extent, (std::size_t)0);
    auto qp = stdex::submdspan(all_qps, i / 2, stdex::full_extent,
                               stdex::full_extent);
    dolfinx::fem::CoordinateElement::push_forward(qp, coordinate_dofs, phi_f);
  }
}

//-------------------------------------------------------------------------------------
std::tuple<dolfinx::graph::AdjacencyList<std::int32_t>, std::vector<double>,
           std::array<std::size_t, 2>>
dolfinx_contact::compute_distance_map(
    const dolfinx::mesh::Mesh& quadrature_mesh,
    std::span<const std::int32_t> quadrature_facets,
    const dolfinx::mesh::Mesh& candidate_mesh,
    std::span<const std::int32_t> candidate_facets,
    const dolfinx_contact::QuadratureRule& q_rule,
    dolfinx_contact::ContactMode mode)
{

  const dolfinx::mesh::Geometry& geometry = quadrature_mesh.geometry();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t gdim = geometry.dim();
  const dolfinx::mesh::Topology& topology = quadrature_mesh.topology();
  const dolfinx::mesh::CellType cell_type = topology.cell_type();
  dolfinx_contact::error::check_cell_type(cell_type);

  const int tdim = topology.dim();
  assert(q_rule.dim() == tdim - 1);
  assert(q_rule.cell_type(0)
         == dolfinx::mesh::cell_entity_type(cell_type, fdim, 0));

  switch (mode)
  {
  case dolfinx_contact::ContactMode::ClosestPoint:
  {
    // Get quadrature points on reference facets
    const std::vector<double>& q_points = q_rule.points();
    const std::vector<std::size_t>& q_offset = q_rule.offset();
    const std::size_t num_q_points = q_offset[1] - q_offset[0];
    const std::size_t sum_q_points = q_offset.back();
    // Push forward quadrature points to physical element
    std::vector<double> quadrature_points(quadrature_facets.size() / 2
                                          * num_q_points * gdim);
    {
      // Tabulate coordinate element basis values
      std::array<std::size_t, 4> cmap_shape
          = cmap.tabulate_shape(0, sum_q_points);
      std::vector<double> c_basis(std::reduce(
          cmap_shape.cbegin(), cmap_shape.cend(), 1, std::multiplies{}));
      cmap.tabulate(0, q_points, {sum_q_points, (std::size_t)tdim}, c_basis);
      dolfinx_contact::cmdspan4_t reference_facet_basis_values(c_basis.data(),
                                                               cmap_shape);
      compute_physical_points(quadrature_mesh, quadrature_facets, q_offset,
                              reference_facet_basis_values, quadrature_points);
    }
    std::vector<std::int32_t> offsets(quadrature_points.size() + 1,
                                      num_q_points);
    for (std::size_t i = 0; i < offsets.size(); ++i)
      offsets[i] *= i;

    // Copy quadrature points to padded 3D structure
    std::vector<double> padded_qpsb(quadrature_facets.size() / 2 * num_q_points
                                    * 3);
    if (gdim == 2)
    {
      dolfinx_contact::mdspan3_t padded_qps(
          padded_qpsb.data(), quadrature_facets.size() / 2, num_q_points, 3);
      dolfinx_contact::cmdspan3_t qps(quadrature_points.data(),
                                      quadrature_facets.size() / 2,
                                      num_q_points, gdim);
      for (std::size_t i = 0; i < qps.extent(0); ++i)
        for (std::size_t j = 0; j < qps.extent(1); ++j)
          for (std::size_t k = 0; k < qps.extent(2); ++k)
            padded_qps(i, j, k) = qps(i, j, k);
    }
    else if (gdim == 3)
    {
      assert(quadrature_points.size() == padded_qpsb.size());
      std::copy(quadrature_points.begin(), quadrature_points.end(),
                padded_qpsb.begin());
    }
    else
      throw std::runtime_error("Invalid gdim: " + std::to_string(gdim));
    if (tdim == 2)
    {
      if (gdim == 2)
      {
        auto [closest_entities, reference_points, shape]
            = dolfinx_contact::compute_projection_map<2, 2>(
                candidate_mesh, candidate_facets, padded_qpsb);
        return {dolfinx::graph::AdjacencyList<std::int32_t>(closest_entities,
                                                            offsets),
                reference_points, shape};
      }
      else if (gdim == 3)
      {
        auto [closest_entities, reference_points, shape]
            = dolfinx_contact::compute_projection_map<2, 3>(
                candidate_mesh, candidate_facets, padded_qpsb);
        return {dolfinx::graph::AdjacencyList<std::int32_t>(closest_entities,
                                                            offsets),
                reference_points, shape};
      }
    }
    else if (tdim == 3)
    {
      auto [closest_entities, reference_points, shape]
          = dolfinx_contact::compute_projection_map<3, 3>(
              candidate_mesh, candidate_facets, padded_qpsb);
      return {dolfinx::graph::AdjacencyList<std::int32_t>(closest_entities,
                                                          offsets),
              reference_points, shape};
    }
    else
      throw std::runtime_error("Invalid tdim: " + std::to_string(tdim));
  }
  case dolfinx_contact::ContactMode::RayTracing:
  {

    if (tdim == 2)
    {
      if (gdim == 2)
      {
        return dolfinx_contact::compute_raytracing_map<2, 2>(
            quadrature_mesh, quadrature_facets, q_rule, candidate_mesh,
            candidate_facets);
      }
      else if (gdim == 3)
      {
        return dolfinx_contact::compute_raytracing_map<2, 3>(
            quadrature_mesh, quadrature_facets, q_rule, candidate_mesh,
            candidate_facets);
      }
      else
        throw std::runtime_error("Invalid gdim: " + std::to_string(gdim));
    }
    else if (tdim == 3)
    {
      return dolfinx_contact::compute_raytracing_map<3, 3>(
          quadrature_mesh, quadrature_facets, q_rule, candidate_mesh,
          candidate_facets);
    }
    else
      throw std::runtime_error("Invalid tdim: " + std::to_string(tdim));
  }
  default:
    throw std::runtime_error("Unsupported contact mode");
  }
}