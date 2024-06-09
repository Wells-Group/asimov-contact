// Copyright (C) 2021-2022 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "geometric_quantities.h"

using namespace dolfinx_contact;

std::vector<double> dolfinx_contact::allocate_pull_back_nonaffine(
    const dolfinx::fem::CoordinateElement<double>& cmap, int gdim, int tdim)
{
  const std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, 1);
  const std::size_t basis_size
      = std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{});
  return std::vector<double>(2 * gdim * tdim + basis_size + 9);
}

void dolfinx_contact::pull_back_nonaffine(
    std::span<double> X, std::span<double> work_array,
    std::span<const double> x,
    const dolfinx::fem::CoordinateElement<double>& cmap,
    mdspan_t<const double, 2> cell_geometry, double tol, const int max_it)
{
  assert((std::size_t)cmap.dim() == cell_geometry.extent(0));
  // Temporary data structures for Newton iteration
  const std::size_t gdim = x.size();
  const std::size_t tdim = X.size();
  const std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, 1);
  const std::size_t basis_size
      = std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{});
  assert(work_array.size() >= basis_size + 2 * gdim * tdim);

  // Use work-array for views
  std::fill(work_array.begin(), std::next(work_array.begin(), gdim * tdim + 9),
            0);
  mdspan2_t J(work_array.data(), gdim, tdim);
  std::span<double, 3> Xk(work_array.data() + gdim * tdim, 3);
  std::span<double, 3> xk(work_array.data() + gdim * tdim + 3, 3);
  std::span<double, 3> dX(work_array.data() + gdim * tdim + 6, 3);
  mdspan2_t K(work_array.data() + gdim * tdim + 9, tdim, gdim);
  mdspan4_t basis_values(work_array.data() + 9 + 2 * gdim * tdim, c_shape);
  std::span basis_span(work_array.data() + 9 + 2 * gdim * tdim, basis_size);

  int k;
  for (k = 0; k < max_it; ++k)
  {
    // Tabulate coordinate basis at Xk
    cmap.tabulate(1, Xk.subspan(0, tdim), {1, tdim}, basis_span);

    // x = cell_geometry * phi
    auto phi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        basis_values, 0, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    std::fill(xk.begin(), xk.end(), 0.0);
    for (std::size_t i = 0; i < cell_geometry.extent(1); ++i)
      for (std::size_t j = 0; j < cell_geometry.extent(0); ++j)
        xk[i] += cell_geometry(j, i) * phi[j];

    // Compute Jacobian, its inverse and determinant
    std::fill(work_array.begin(), std::next(work_array.begin(), gdim * tdim),
              0.0);
    auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        basis_values, std::pair{1, tdim + 1}, 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian(dphi,
                                                              cell_geometry, J);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);

    // Compute dXk = K (x-xk)
    std::fill(dX.begin(), dX.end(), 0.0);
    for (std::size_t i = 0; i < K.extent(0); ++i)
      for (std::size_t j = 0; j < K.extent(1); ++j)
        dX[i] += K(i, j) * (x[j] - xk[j]);

    // Compute Xk += dX
    std::transform(dX.begin(), std::next(dX.begin(), tdim), Xk.begin(),
                   Xk.begin(), [](double a, double b) { return a + b; });

    // Compute dot(dX, dX)
    auto dX_squared = std::transform_reduce(
        dX.begin(), std::next(dX.begin(), tdim), 0.0, std::plus<double>(),
        [](const auto v) { return v * v; });
    if (std::sqrt(dX_squared) < tol)
      break;
  }

  std::copy(Xk.begin(), std::next(Xk.begin(), tdim), X.begin());
  if (k == max_it)
  {
    throw std::runtime_error(
        "Newton method failed to converge for non-affine geometry");
  }
}

std::array<double, 3> dolfinx_contact::push_forward_facet_normal(
    std::span<double> work_array, std::span<const double> x, std::size_t gdim,
    std::size_t tdim, mdspan_t<const double, 2> coordinate_dofs,
    const std::size_t facet_index,
    const dolfinx::fem::CoordinateElement<double>& cmap,
    mdspan_t<const double, 2> reference_normals)
{
  const std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, 1);
  const std::size_t basis_size
      = std::reduce(c_shape.cbegin(), c_shape.cend(), 1, std::multiplies{});
  assert(work_array.size() >= 3 + basis_size + 2 * gdim * tdim);

  // Use work-array for views
  std::fill(work_array.begin(), std::next(work_array.begin(), gdim * tdim + 9),
            0);
  mdspan2_t J(work_array.data(), gdim, tdim);
  std::span<double, 3> X(work_array.data() + gdim * tdim, 3);
  mdspan2_t K(work_array.data() + gdim * tdim + 3, tdim, gdim);
  cmdspan4_t basis_values(work_array.data() + 3 + 2 * gdim * tdim, c_shape);
  std::span<double> basis_span(work_array.data() + 3 + 2 * gdim * tdim,
                               basis_size);
  if (cmap.is_affine())
  {
    // Affine Jacobian can be computed at any point in the cell (0,0,0) in
    // the reference cell
    std::fill(X.begin(), X.end(), 0);
    cmap.tabulate(1, X.subspan(0, tdim), {1, tdim}, basis_span);

    auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        basis_values, std::pair{1, tdim + 1}, 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian(
        dphi, coordinate_dofs, J);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);
  }
  else
  {
    // For non-affine geometries we have to compute the point in the reference
    // cell, which is a nonlinear operation.
    pull_back_nonaffine(X.subspan(0, tdim), work_array, x.subspan(0, gdim),
                        cmap, coordinate_dofs);

    cmap.tabulate(1, X.subspan(0, tdim), {1, tdim}, basis_span);
    std::fill(work_array.begin(), std::next(work_array.begin(), gdim * tdim),
              0.0);
    auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        basis_values, std::pair{1, tdim + 1}, 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian(
        dphi, coordinate_dofs, J);
    dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);
  }
  // Push forward reference facet normal
  std::array<double, 3> normal = {0, 0, 0};
  auto facet_normal = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      reference_normals, facet_index,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
  physical_facet_normal(std::span(normal.data(), gdim), K, facet_normal);
  return normal;
}

//-----------------------------------------------------------------------------
double
dolfinx_contact::compute_circumradius(const dolfinx::mesh::Mesh<double>& mesh,
                                      double detJ,
                                      mdspan_t<const double, 2> coordinate_dofs)
{
  const dolfinx::mesh::CellType cell_type = mesh.topology()->cell_type();
  const int gdim = mesh.geometry().dim();

  switch (cell_type)
  {
  case dolfinx::mesh::CellType::triangle:
  {
    // Formula for circumradius of a triangle with sides with length a, b, c
    // is R = a b c / (4 A) where A is the area of the triangle
    const double ref_area
        = basix::cell::volume<double>(basix::cell::type::triangle);
    double area = ref_area * std::abs(detJ);

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

    return sides[0] * sides[1] * sides[2] / (4 * area);
  }
  case dolfinx::mesh::CellType::tetrahedron:
  {
    // Formula for circunradius of a tetrahedron with volume V.
    // Given three edges meeting at a vertex with length a, b, c,
    // and opposite edges with corresponding length A, B, C we have that the
    // circumradius
    // R = sqrt((aA + bB + cC)(aA+bB-cC)(aA-bB+cC)(-aA +bB+cC))/24V
    const double ref_volume
        = basix::cell::volume<double>(basix::cell::type::tetrahedron);
    double cellvolume = std::abs(detJ) * ref_volume;

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
    double h = std::sqrt((aA + bB + cC) * (aA + bB - cC) * (aA - bB + cC)
                         * (-aA + bB + cC))
               / (24 * cellvolume);
    return h;
  }
  default:
    throw std::invalid_argument("Unsupported cell_type "
                                + dolfinx::mesh::to_string(cell_type));
  }
}
