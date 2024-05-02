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
void dolfinx_contact::pull_back(
    dolfinx_contact::mdspan3_t J, dolfinx_contact::mdspan3_t K,
    std::span<double> detJ, std::span<double> X, dolfinx_contact::cmdspan2_t x,
    dolfinx_contact::cmdspan2_t coordinate_dofs,
    const dolfinx::fem::CoordinateElement<double>& cmap)
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
    auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        c_basis, std::pair{1, tdim + 1}, 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

    // Only zero out first Jacobian as it is used to fill in the others
    for (std::size_t j = 0; j < J.extent(1); ++j)
      for (std::size_t k = 0; k < J.extent(2); ++k)
        J(0, j, k) = 0;

    // Compute Jacobian at origin of reference element
    auto J0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        J, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian(
        dphi0, coordinate_dofs, J0);

    for (std::size_t j = 0; j < K.extent(1); ++j)
      for (std::size_t k = 0; k < K.extent(2); ++k)
        K(0, j, k) = 0;
    // Compute inverse Jacobian
    auto K0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        K, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J0, K0);

    // Compute determinant
    detJ[0]
        = dolfinx::fem::CoordinateElement<double>::compute_jacobian_determinant(
            J0, detJ_scratch);

    // Pull back all physical coordinates
    std::array<double, 3> x0 = {0, 0, 0};
    for (std::size_t i = 0; i < coordinate_dofs.extent(1); ++i)
      x0[i] += coordinate_dofs(0, i);
    dolfinx_contact::mdspan2_t Xs(X.data(), num_points, tdim);
    dolfinx::fem::CoordinateElement<double>::pull_back_affine(Xs, K0, x0, x);

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
      auto _J = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          J, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          c_basis, std::pair{1, tdim + 1}, p,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
      dolfinx::fem::CoordinateElement<double>::compute_jacobian(
          dphi, coordinate_dofs, _J);
      auto _K = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          K, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(_J, _K);
      detJ[p] = dolfinx::fem::CoordinateElement<
          double>::compute_jacobian_determinant(_J, detJ_scratch);
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
    std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh)
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V
      = u.function_space();
  assert(V);
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  assert(dofmap);
  // Check that mesh to be updated and underlying mesh of u are the same
  assert(mesh == V->mesh());

  // The Function and the mesh must have identical element_dof_layouts
  // (up to the block size)
  assert(dofmap->element_dof_layout()
         == mesh->geometry().cmap().create_dof_layout());

  const int tdim = mesh->topology()->dim();
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = mesh->topology()->index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();

  // Get dof array and retrieve u at the mesh dofs
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      dofmap_x = mesh->geometry().dofmap();
  const int bs = dofmap->bs();
  const auto& u_data = u.x()->array();
  std::span<double> coords = mesh->geometry().x();
  std::vector<double> dx(coords.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const std::span<const int> dofs = dofmap->cell_dofs(c);
    auto dofs_x = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        dofmap_x, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
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
std::array<double, 3> dolfinx_contact::ball_projection(std::array<double, 3> x,
                                                       double alpha)
{
  // Compute norm of vector
  double norm = 0;
  std::for_each(x.cbegin(), x.cend(),
                [&norm](auto e) { norm += std::pow(e, 2); });
  norm = std::sqrt(norm);
  std::array<double, 3> proj = {0, 0, 0};

  // If x inside ball return x
  if (norm <= alpha)
    std::copy(x.cbegin(), x.cend(), proj.begin());
  else
    // If x outside ball return alpha*x/norm
    std::transform(x.cbegin(), x.cend(), proj.begin(),
                   [alpha, norm](auto& xi) { return alpha * xi / norm; });
  return proj;
}
//-------------------------------------------------------------------------------------
std::array<double, 9>
dolfinx_contact::d_ball_projection(std::array<double, 3> x, double alpha,
                                   std::size_t bs)
{
  std::array<double, 9> d_proj = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // avoid dividing by 0 if radius 0
  if (alpha < 1E-14)
    return d_proj;

  // Compute norm of vector
  double norm_squared = 0;
  std::for_each(x.cbegin(), x.cend(),
                [&norm_squared](auto e) { norm_squared += std::pow(e, 2); });
  double norm = std::sqrt(norm_squared);

  // If vector inside ball return identity matrix I
  if (norm < alpha)
    for (std::size_t j = 0; j < bs; ++j)
      d_proj[j * bs + j] = 1;
  else
    // If vector outside ball return alpha * I/norm - outer(x, x)/(norm**3)
    for (std::size_t i = 0; i < bs; ++i)
    {
      d_proj[i * bs + i] += alpha / norm;
      for (std::size_t j = 0; j < bs; ++j)
        d_proj[i * bs + j] -= alpha * x[i] * x[j] / (norm * norm_squared);
    }
  return d_proj;
}

std::array<double, 3>
dolfinx_contact::d_alpha_ball_projection(std::array<double, 3> x, double alpha,
                                         double d_alpha)
{
  // Compute norm of vector
  double norm = 0;
  std::for_each(x.cbegin(), x.cend(),
                [&norm](auto e) { norm += std::pow(e, 2); });
  norm = std::sqrt(norm);
  std::array<double, 3> d_alpha_proj = {0, 0, 0};

  // If x inside ball return 0
  if (norm > alpha)
    // If x outside ball return d_alpha*x/norm
    std::transform(x.cbegin(), x.cend(), d_alpha_proj.begin(),
                   [d_alpha, norm](auto& xi) { return d_alpha * xi / norm; });
  return d_alpha_proj;
}
//-------------------------------------------------------------------------------------
std::array<std::size_t, 4> dolfinx_contact::evaluate_basis_shape(
    const dolfinx::fem::FunctionSpace<double>& V, const std::size_t num_points,
    const std::size_t num_derivatives)
{
  // Get element
  assert(V.element());
  std::size_t gdim = V.mesh()->geometry().dim();
  std::shared_ptr<const dolfinx::fem::FiniteElement<double>> element
      = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t value_size = V.value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;
  return {num_derivatives * gdim + 1, num_points, space_dimension, value_size};
}
//-----------------------------------------------------------------------------
void dolfinx_contact::evaluate_basis_functions(
    const dolfinx::fem::FunctionSpace<double>& V, std::span<const double> x,
    std::span<const std::int32_t> cells, std::span<double> basis_values,
    std::size_t num_derivatives)
{

  assert(num_derivatives < 2);

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V.mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry<double>& geometry = mesh->geometry();
  auto topology = mesh->topology();
  const std::size_t tdim = topology->dim();
  const std::size_t gdim = geometry.dim();
  const std::size_t num_cells = cells.size();
  if (x.size() / tdim != num_cells)
  {
    throw std::invalid_argument(
        "Number of points and number of cells must be equal.");
  }

  if (x.size() == 0)
    return;

  // Get topology data
  std::shared_ptr<const dolfinx::common::IndexMap> map
      = topology->index_map((int)tdim);

  // Get geometry data
  std::span<const double> x_g = geometry.x();
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmap();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Get element
  assert(V.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement<double>> element
      = V.element();
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
    mesh->topology_mutable()->create_entity_permutations();
    cell_info = std::span(topology->get_cell_permutation_info());
  }

  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  dolfinx_contact::mdspan2_t coordinate_dofs(coordinate_dofsb.data(),
                                             num_dofs_g, gdim);

  // Prepare geometry data structures
  std::array<double, 9> Jb;
  std::array<double, 9> Kb;
  mdspan2_t J(Jb.data(), gdim, tdim);
  mdspan2_t K(Kb.data(), tdim, gdim);
  std::vector<double> detJ_scratch(2 * gdim * tdim);

  // Tabulate coordinate basis to compute Jacobian
  std::array<std::size_t, 4> c_shape2 = cmap.tabulate_shape(1, num_cells);
  std::vector<double> c_basisb(
      std::reduce(c_shape2.cbegin(), c_shape2.cend(), 1, std::multiplies{}));
  cmap.tabulate(1, x, {num_cells, tdim}, c_basisb);
  cmdspan4_t c_basis(c_basisb.data(), c_shape2);
  // Prepare basis function data structures
  const std::array<std::size_t, 4> reference_shape
      = element->basix_element().tabulate_shape(num_derivatives, num_cells);
  std::vector<double> basis_reference_valuesb(std::reduce(
      reference_shape.cbegin(), reference_shape.cend(), 1, std::multiplies{}));

  // Compute basis on reference element
  element->tabulate(basis_reference_valuesb, x, {num_cells, tdim},
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

  using xu_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xU_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xJ_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  using xK_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
  auto push_forward_fn
      = element->basix_element().map_fn<xu_t, xU_t, xJ_t, xK_t>();
  const std::function<void(const std::span<double>&,
                           const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation = element->dof_transformation_fn<double>(
          dolfinx::fem::doftransform::standard);
  const std::size_t num_basis_values = space_dimension * reference_value_size;

  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    // Skip negative cell indices
    const int cell_index = cells[p];
    if (cell_index < 0)
      continue;
    // Get cell geometry (coordinate dofs)
    auto x_dofs2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t j = 0; j < num_dofs_g; ++j)
    {
      auto pos = 3 * x_dofs2[j];
      for (std::size_t k = 0; k < coordinate_dofs.extent(1); ++k)
        coordinate_dofs(j, k) = x_g[pos + k];
    }

    std::fill(Jb.begin(), Jb.end(), 0);
    auto dphi_q = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        c_basis, std::pair{1, std::size_t(tdim + 1)}, p,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian(
        dphi_q, coordinate_dofs, J);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);
    double detJ
        = dolfinx::fem::CoordinateElement<double>::compute_jacobian_determinant(
            J, detJ_scratch);

    /// NOTE: loop size correct for num_derivatives = 0,1
    for (std::size_t j = 0; j < num_derivatives * tdim + 1; ++j)
    {
      // Permute the reference values to account for the cell's orientation
      apply_dof_transformation(
          std::span(basis_reference_valuesb.data()
                        + j * cells.size() * num_basis_values
                        + p * num_basis_values,
                    num_basis_values),
          cell_info, cell_index, (int)reference_value_size);

      // Push basis forward to physical element
      auto _U = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis_reference_values, j, p,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

      if (j == 0)
      {
        auto _u = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            basis_span, j, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        push_forward_fn(_u, _U, J, detJ, K);
      }
      else
      {
        auto _u = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            temp, j, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        push_forward_fn(_u, _U, J, detJ, K);
      }
    }

    for (std::size_t k = 0; k < gdim * num_derivatives; ++k)
    {
      auto du = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis_span, k + 1, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t j = 0; j < num_derivatives * tdim; ++j)
      {
        auto du_temp = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            temp, j + 1, p, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t m = 0; m < du.extent(0); ++m)
          for (std::size_t n = 0; n < du.extent(1); ++n)
            du(m, n) += K(j, k) * du_temp(m, n);
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
  auto coordinate_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      coords, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, std::pair{0, gdim});
  for (std::size_t i = 0; i < J.extent(0); ++i)
    for (std::size_t j = 0; j < J.extent(1); ++j)
      J(i, j) = 0;
  dolfinx::fem::CoordinateElement<double>::compute_jacobian(dphi,
                                                            coordinate_dofs, J);
  dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);
  for (std::size_t i = 0; i < J_tot.extent(0); ++i)
    for (std::size_t j = 0; j < J_tot.extent(1); ++j)
      J_tot(i, j) = 0;
  dolfinx::math::dot(J, J_f, J_tot);
  return std::fabs(
      dolfinx::fem::CoordinateElement<double>::compute_jacobian_determinant(
          J_tot, detJ_scratch));
}
//-------------------------------------------------------------------------------------
std::function<double(
    double, dolfinx_contact::mdspan2_t, dolfinx_contact::mdspan2_t,
    dolfinx_contact::mdspan2_t, std::span<double>, dolfinx_contact::cmdspan2_t,
    dolfinx_contact::s_cmdspan2_t, dolfinx_contact::cmdspan2_t)>
dolfinx_contact::get_update_jacobian_dependencies(
    const dolfinx::fem::CoordinateElement<double>& cmap)
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
dolfinx_contact::get_update_normal(
    const dolfinx::fem::CoordinateElement<double>& cmap)
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
      auto n_f = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          n_ref, local_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
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
/// @return list of active entities sorted by cell and size_local
std::pair<std::vector<std::int32_t>, std::size_t>
dolfinx_contact::compute_active_entities(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
    std::span<const std::int32_t> entities, dolfinx::fem::IntegralType integral)
{

  int tdim = mesh->topology()->dim();
  const int size_local = mesh->topology()->index_map(tdim)->size_local();
  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
  {
    std::vector<std::int32_t> active_entities(entities.size());
    std::transform(entities.begin(), entities.end(), active_entities.begin(),
                   [](std::int32_t cell) { return cell; });
    dolfinx::radix_sort(std::span<std::int32_t>(active_entities));
    auto cell_it = std::upper_bound(active_entities.begin(),
                                    active_entities.end(), size_local);
    std::size_t num_local = std::distance(active_entities.begin(), cell_it);
    return std::make_pair(active_entities, num_local);
  }
  case dolfinx::fem::IntegralType::exterior_facet:
  {
    std::vector<std::int32_t> active_entities(2 * entities.size());
    std::vector<std::int32_t> cells(entities.size());
    std::vector<std::int32_t> facets(entities.size());

    auto topology = mesh->topology();
    auto f_to_c = topology->connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = topology->connectivity(tdim, tdim - 1);
    assert(c_to_f);
    for (std::size_t f = 0; f < entities.size(); f++)
    {
      assert(f_to_c->num_links(entities[f]) == 1);
      const std::int32_t cell = f_to_c->links(entities[f])[0];
      auto cell_facets = c_to_f->links(cell);

      auto facet_it
          = std::find(cell_facets.begin(), cell_facets.end(), entities[f]);
      assert(facet_it != cell_facets.end());
      cells[f] = cell;
      facets[f] = (std::int32_t)std::distance(cell_facets.begin(), facet_it);
    }

    std::vector<std::int32_t> perm(entities.size());
    std::iota(perm.begin(), perm.end(), 0);
    dolfinx::argsort_radix<std::int32_t>(cells, perm);

    // Sort cells in accending order
    std::size_t num_local = 0;
    for (std::size_t f = 0; f < entities.size(); f++)
    {
      active_entities[2 * f] = cells[perm[f]];
      active_entities[2 * f + 1] = facets[perm[f]];
      if (cells[perm[f]] < size_local)
        num_local += 1;
    }

    return std::make_pair(active_entities, num_local);
  }
  default:
    throw std::invalid_argument(
        "Integral type not supported. Note that this function "
        "has not been implemented for interior facets.");
  }
  return {};
}

//-------------------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_contact::entities_to_geometry_dofs(
    const dolfinx::mesh::Mesh<double>& mesh, int dim,
    const std::span<const std::int32_t>& entity_list)
{

  // Get mesh geometry and topology data
  const dolfinx::mesh::Geometry<double>& geometry = mesh.geometry();
  const dolfinx::fem::ElementDofLayout layout
      = geometry.cmap().create_dof_layout();
  // FIXME: What does this return for prisms?
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      xdofs = geometry.dofmap();

  auto topology = mesh.topology();
  const int tdim = topology->dim();
  mesh.topology_mutable()->create_entities(dim);
  mesh.topology_mutable()->create_connectivity(dim, tdim);
  mesh.topology_mutable()->create_connectivity(tdim, dim);

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
      = topology->connectivity(dim, tdim);
  assert(e_to_c);
  const std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_e
      = topology->connectivity(tdim, dim);
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

    auto xc = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        xdofs, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    assert(num_entity_dofs <= xc.size());
    for (std::size_t j = 0; j < num_entity_dofs; ++j)
      geometry_indices[i * num_entity_dofs + j] = xc[entity_dofs[j]];
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(geometry_indices, offsets);
}
//--------------------------------------------------------------------------------------
std::vector<int32_t> dolfinx_contact::facet_indices_from_pair(
    std::span<const std::int32_t> facet_pairs,
    const dolfinx::mesh::Mesh<double>& mesh)
{
  // Convert (cell, local facet index) into facet index
  // (local to process) Convert cell,local_facet_index to
  // facet_index (local to proc)
  const int tdim = mesh.topology()->dim();
  std::vector<std::int32_t> facets(facet_pairs.size() / 2);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
      = mesh.topology()->connectivity(tdim, tdim - 1);
  if (!c_to_f)
  {
    throw std::runtime_error("Missing cell->facet connectivity on "
                             "mesh.");
  }

  for (std::size_t i = 0; i < facet_pairs.size(); i += 2)
  {
    auto local_facets = c_to_f->links(facet_pairs[i]);
    assert(!local_facets.empty());
    assert((std::size_t)facet_pairs[i + 1] < local_facets.size());
    facets[i / 2] = local_facets[facet_pairs[i + 1]];
  }
  return facets;
}
//-------------------------------------------------------------------------------------
std::vector<std::size_t> dolfinx_contact::find_candidate_facets(
    const dolfinx::mesh::Mesh<double>& quadrature_mesh,
    const dolfinx::mesh::Mesh<double>& candidate_mesh,
    const std::int32_t quadrature_facet,
    std::span<const std::int32_t> candidate_facets, const double radius = -1.)
{
  std::array<std::int32_t, 1> q_facet = {quadrature_facet};
  // Find midpoints of quadrature and candidate facets
  std::vector<double> quadrature_midpoints = dolfinx::mesh::compute_midpoints(
      quadrature_mesh, quadrature_mesh.topology()->dim() - 1,
      std::span<int32_t>(q_facet.data(), q_facet.size()));
  std::vector<double> candidate_midpoints = dolfinx::mesh::compute_midpoints(
      candidate_mesh, candidate_mesh.topology()->dim() - 1, candidate_facets);

  double r2 = radius * radius; // radius squared
  double dist; // used for squared distance between two midpoints
  double diff; // used for squared difference between two coordinates
  std::vector<std::size_t> cand_patch;
  std::vector<double> dists;
  for (std::size_t i = 0; i < candidate_facets.size(); ++i)
  {

    // compute distance betweeen midpoints of ith candidate facet
    // and jth quadrature facet
    dist = 0;
    for (std::size_t k = 0; k < 3; ++k)
    {
      diff = std::abs(quadrature_midpoints[k] - candidate_midpoints[i * 3 + k]);
      dist += diff * diff;
    }
    if (radius < 0 || dist < r2)
    {
      cand_patch.push_back(i); // save index of facet within facet array
      dists.push_back(dist);   // save distance for sorting
    }
  }
  // sort indices according to distance of facet
  std::vector<int> perm(cand_patch.size());
  std::iota(perm.begin(), perm.end(), 0); // Initializing
  std::sort(perm.begin(), perm.end(),
            [&](int i, int j) { return dists[i] < dists[j]; });
  std::vector<size_t> sorted_patch(cand_patch.size());
  for (std::size_t i = 0; i < cand_patch.size(); ++i)
    sorted_patch[i] = cand_patch[perm[i]];
  return sorted_patch;
}
//-------------------------------------------------------------------------------------
std::vector<std::int32_t> dolfinx_contact::find_candidate_surface_segment(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
    const std::vector<std::int32_t>& quadrature_facets,
    const std::vector<std::int32_t>& candidate_facets,
    const double radius = -1.)
{
  if (radius < 0)
  {
    // return all facets for negative radius / no radius
    return std::vector<std::int32_t>(candidate_facets);
  }
  // Find midpoints of quadrature and candidate facets
  std::vector<double> quadrature_midpoints = dolfinx::mesh::compute_midpoints(
      *mesh, mesh->topology()->dim() - 1, quadrature_facets);
  std::vector<double> candidate_midpoints = dolfinx::mesh::compute_midpoints(
      *mesh, mesh->topology()->dim() - 1, candidate_facets);

  double r2 = radius * radius; // radius squared
  double dist; // used for squared distance between two midpoints
  double diff; // used for squared difference between two coordinates

  std::vector<std::int32_t> cand_patch;

  for (std::size_t i = 0; i < candidate_facets.size(); ++i)
  {
    for (std::size_t j = 0; j < quadrature_facets.size(); ++j)
    {
      // compute distance betweeen midpoints of ith candidate facet
      // and jth quadrature facet
      dist = 0;
      for (std::size_t k = 0; k < 3; ++k)
      {
        diff = std::abs(quadrature_midpoints[j * 3 + k]
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
    const dolfinx::mesh::Mesh<double>& mesh,
    std::span<const std::int32_t> facets, std::span<const std::size_t> offsets,
    dolfinx_contact::cmdspan4_t phi, std::span<double> qp_phys)
{
  dolfinx::common::Timer timer("~Contact: Compute Physical points");

  // Geometrical info
  const dolfinx::mesh::Geometry<double>& geometry = mesh.geometry();
  std::span<const double> mesh_geometry = geometry.x();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmap();
  const int gdim = geometry.dim();

  // Create storage for output quadrature points
  // NOTE: Assume that all facets have the same number of quadrature points
  dolfinx_contact::error::check_cell_type(mesh.topology()->cell_type());
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
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, facets[i], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t j = 0; j < num_dofs_g; ++j)
    {
      std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofsb.begin(), j * gdim));
    }
    // push forward points on reference element
    const std::array<std::size_t, 2> range
        = {offsets[facets[i + 1]], offsets[facets[i + 1] + 1]};
    auto phi_f = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi, (std::size_t)0, std::pair{range.front(), range.back()},
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, (std::size_t)0);
    auto qp = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        all_qps, i / 2, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    dolfinx::fem::CoordinateElement<double>::push_forward(qp, coordinate_dofs,
                                                          phi_f);
  }
}

//-------------------------------------------------------------------------------------
std::tuple<dolfinx::graph::AdjacencyList<std::int32_t>, std::vector<double>,
           std::array<std::size_t, 2>>
dolfinx_contact::compute_distance_map(
    const dolfinx::mesh::Mesh<double>& quadrature_mesh,
    std::span<const std::int32_t> quadrature_facets,
    const dolfinx::mesh::Mesh<double>& candidate_mesh,
    std::span<const std::int32_t> candidate_facets,
    const dolfinx_contact::QuadratureRule& q_rule,
    dolfinx_contact::ContactMode mode, const double radius)
{
  dolfinx::common::Timer t("~Contact: compute distance map");
  const dolfinx::mesh::Geometry<double>& geometry = quadrature_mesh.geometry();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();

  const std::size_t gdim = geometry.dim();
  auto topology = quadrature_mesh.topology();
  const dolfinx::mesh::CellType cell_type = topology->cell_type();
  dolfinx_contact::error::check_cell_type(cell_type);

  const int tdim = topology->dim();
  assert(q_rule.dim() == tdim - 1);
  assert(q_rule.cell_type(0)
         == dolfinx::mesh::cell_entity_type(cell_type, tdim - 1, 0));

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
    std::vector<std::int32_t> offsets(quadrature_facets.size() / 2 + 1,
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
            candidate_facets, radius);
      }
      else if (gdim == 3)
      {
        return dolfinx_contact::compute_raytracing_map<2, 3>(
            quadrature_mesh, quadrature_facets, q_rule, candidate_mesh,
            candidate_facets, radius);
      }
      else
        throw std::runtime_error("Invalid gdim: " + std::to_string(gdim));
    }
    else if (tdim == 3)
    {
      return dolfinx_contact::compute_raytracing_map<3, 3>(
          quadrature_mesh, quadrature_facets, q_rule, candidate_mesh,
          candidate_facets, radius);
    }
    else
      throw std::runtime_error("Invalid tdim: " + std::to_string(tdim));
  }
  }

  throw std::runtime_error("Unsupported contact mode");
}
MatNullSpace dolfinx_contact::build_nullspace_multibody(
    const dolfinx::fem::FunctionSpace<double>& V,
    const dolfinx::mesh::MeshTags<std::int32_t>& mt,
    std::vector<std::int32_t>& tags)
{
  std::size_t gdim = V.mesh()->geometry().dim();
  std::size_t dim = (gdim == 2) ? 3 : 6;
  const std::size_t ndofs_cell = V.dofmap()->element_dof_layout().num_dofs();

  // Create vectors for nullspace basis
  // Need translations and rotations for each
  // component. number of components = tags.size()
  auto map = V.dofmap()->index_map;
  int bs = V.dofmap()->index_map_bs();
  std::vector<dolfinx::la::Vector<PetscScalar>> basis(
      dim * tags.size(), la::Vector<PetscScalar>(map, bs));

  // loop over components
  for (std::size_t j = 0; j < tags.size(); ++j)
  {
    // retrieve degrees of freedom
    std::vector<std::int32_t> cells = mt.find(tags[j]);
    std::vector<std::int32_t> dofs(cells.size() * ndofs_cell);
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      std::span<const int32_t> cell_dofs = V.dofmap()->cell_dofs(cells[c]);
      std::copy_n(cell_dofs.begin(), ndofs_cell, dofs.begin() + c * ndofs_cell);
    }
    // Remove duplicates
    dolfinx::radix_sort(std::span<std::int32_t>(dofs));
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());

    // Translations
    for (std::size_t k = 0; k < gdim; ++k)
    {
      std::span<PetscScalar> x = basis[j * dim + k].mutable_array();
      for (auto dof : dofs)
        x[gdim * dof + k] = 1.0;
    }

    // Rotations
    auto x1 = basis[j * dim + gdim].mutable_array();

    const std::vector<double> x = V.tabulate_dof_coordinates(false);
    if (gdim == 2)
    {
      for (auto dof : dofs)
      {
        std::span<const double, 3> xd(x.data() + 3 * dof, 3);
        x1[gdim * dof] = -xd[1];
        x1[gdim * dof + 1] = xd[0];
      }
    }
    else
    {
      auto x2 = basis[j * dim + 4].mutable_array();
      auto x3 = basis[j * dim + 5].mutable_array();
      for (auto dof : dofs)
      {
        std::span<const double, 3> xd(x.data() + 3 * dof, 3);
        x1[gdim * dof] = -xd[1];
        x1[gdim * dof + 1] = xd[0];

        x2[gdim * dof] = xd[2];
        x2[gdim * dof + 2] = -xd[0];

        x3[gdim * dof + 2] = xd[1];
        x3[gdim * dof + 1] = -xd[2];
      }
    }
  }

  // Orthonormalize basis
  dolfinx::la::orthonormalize<dolfinx::la::Vector<PetscScalar>>(
      std::vector<std::reference_wrapper<dolfinx::la::Vector<PetscScalar>>>(
          basis.begin(), basis.end()));
  if (!dolfinx::la::is_orthonormal(
          std::vector<
              std::reference_wrapper<const dolfinx::la::Vector<PetscScalar>>>(
              basis.begin(), basis.end())))
  {
    throw std::runtime_error("Space not orthonormal");
  }

  // Build PETSc nullspace object
  std::int32_t length = bs * map->size_local();
  std::vector<std::span<const PetscScalar>> basis_local;
  std::transform(basis.cbegin(), basis.cend(), std::back_inserter(basis_local),
                 [length](auto& x)
                 { return std::span(x.array().data(), length); });
  MPI_Comm comm = V.mesh()->comm();
  std::vector<Vec> v = la::petsc::create_vectors(comm, basis_local);
  MatNullSpace ns = la::petsc::create_nullspace(comm, v);
  std::for_each(v.begin(), v.end(), [](auto v0) { VecDestroy(&v0); });
  return ns;
}
