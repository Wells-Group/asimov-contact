
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "RayTracing.h"
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>

namespace
{
/// Get function that parameterizes a facet of a given cell
/// @param[in] cell_type The cell type
/// @param[in] facet_index The facet index (local to cell)
/// @returns Function that computes the coordinate parameterization of the local
/// facet on the reference cell.
std::function<std::array<double, 3>(std::array<double, 2>)>
get_3D_parameterization(dolfinx::mesh::CellType cell_type, int facet_index)
{

  std::function<std::array<double, 3>(std::array<double, 2>)> func;
  const int tdim = dolfinx::mesh::cell_dim(cell_type);
  const int num_facets = dolfinx::mesh::cell_num_entities(cell_type, tdim - 1);
  if (facet_index >= num_facets)
    throw std::invalid_argument(
        "Invalid facet index (larger than number of facets");

  switch (cell_type)
  {
  case dolfinx::mesh::CellType::tetrahedron:
    if (facet_index == 0)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], xi[1], 1 - xi[0] - xi[1]};
      };
    }
    else if (facet_index == 1)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {0, xi[0], xi[1]};
      };
    }
    else if (facet_index == 2)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], 0, xi[1]};
      };
    }
    else if (facet_index == 3)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], xi[1], 0};
      };
    }
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if (facet_index == 0)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], xi[1], 0};
      };
    }
    else if (facet_index == 1)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], 0, xi[1]};
      };
    }
    else if (facet_index == 2)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {0, xi[0], xi[1]};
      };
    }
    else if (facet_index == 3)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {1, xi[0], xi[1]};
      };
    }
    else if (facet_index == 4)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], 1, xi[1]};
      };
    }
    else if (facet_index == 5)
    {
      func = [](std::array<double, 2> xi) -> std::array<double, 3> {
        return {xi[0], xi[1], 1};
      };
    }
    break;
  default:
    throw std::invalid_argument("Unsupported cell type");
    break;
  }
  return func;
}

/// Get derivative of the parameterization with respect to the input parameters
/// @param[in] cell_type The cell type
/// @param[in] facet_index The facet index (local to cell)
/// @returns The Jacobian of the parameterization
std::array<std::array<double, 3>, 2>
get_parameterization_jacobian(dolfinx::mesh::CellType cell_type,
                              int facet_index)
{

  const int tdim = dolfinx::mesh::cell_dim(cell_type);
  const int num_facets = dolfinx::mesh::cell_num_entities(cell_type, tdim - 1);
  if (facet_index >= num_facets)
    throw std::invalid_argument(
        "Invalid facet index (larger than number of facets");

  switch (cell_type)
  {
  case dolfinx::mesh::CellType::tetrahedron:
    if (facet_index == 0)
      return {{{1, 0, -1}, {0, 1, -1}}};
    else if (facet_index == 1)
      return {{{0, 1, 0}, {0, 0, 1}}};
    else if (facet_index == 2)
      return {{{1, 0, 0}, {0, 0, 1}}};
    else if (facet_index == 3)
      return {{{1, 0, 0}, {0, 1, 0}}};
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if ((facet_index == 0) or (facet_index == 5))
      return {{{1, 0, 0}, {0, 1, 0}}};
    else if ((facet_index == 1) or (facet_index == 4))
      return {{{1, 0, 0}, {0, 0, 1}}};
    else if ((facet_index == 2) or (facet_index) == 3)
      return {{{0, 1, 0}, {0, 0, 1}}};
    break;
  default:
    throw std::invalid_argument("Unsupported cell type");
    break;
  }
  std::array<std::array<double, 3>, 2> output;
  return output;
}

} // namespace

std::tuple<int, std::array<double, 3>, std::array<double, 3>>
dolfinx_contact::compute_3D_ray(const dolfinx::mesh::Mesh& mesh,
                                const std::array<double, 3>& point,
                                const std::array<double, 3>& t1,
                                const std::array<double, 3>& t2, int cell,
                                int facet_index, const int max_iter,
                                const double tol)
{
  int status = -1;
  dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
  const int tdim = mesh.topology().dim();

  // Get parameterization function and jacobian
  auto xi = get_3D_parameterization(cell_type, facet_index);
  std::array<std::array<double, 3>, 2> dxi
      = get_parameterization_jacobian(cell_type, facet_index);

  const dolfinx::fem::CoordinateElement& cmap = mesh.geometry().cmap();
  const std::array<std::size_t, 4> basis_shape = cmap.tabulate_shape(1, 1);
  xt::xtensor<double, 4> basis_values(basis_shape);

  // Get cell coordinates/geometry
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  const int gdim = geometry.dim();
  xtl::span<const double> x_g = geometry.x();
  auto x_dofs = x_dofmap.links(cell);
  const std::size_t num_dofs_g = cmap.dim();
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, 3});
  for (std::size_t j = 0; j < x_dofs.size(); ++j)
  {
    std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), 3,
                std::next(coordinate_dofs.begin(), j * 3));
  }

  // Temporary variables
  xt::xtensor<double, 2> X_k({1, 3});
  xt::xtensor<double, 2> x_k({1, 3});
  std::array<double, 2> xi_k = {0.5, 0.25};
  std::array<double, 3> x;
  std::array<double, 3> X;
  xt::xtensor<double, 2> J({(std::size_t)gdim, (std::size_t)tdim});
  xt::xtensor<double, 2> dphi({(std::size_t)tdim, num_dofs_g});
  std::array<double, 2> G_k;
  for (int k = 0; k < max_iter; ++k)
  {
    // Evaluate reference coordinate at current iteration
    X = xi(xi_k);
    std::copy(X.cbegin(), X.cend(), X_k.begin());

    // Tabulate coordinate element basis function
    cmap.tabulate(1, X_k, basis_values);

    // Push forward reference coordinate
    cmap.push_forward(x_k, coordinate_dofs,
                      xt::view(basis_values, 0, xt::all(), xt::all(), 0));
    dphi = xt::view(basis_values, xt::xrange(1, tdim + 1), 0, xt::all(), 0);

    // Compute Jacobian
    cmap.compute_jacobian(dphi, coordinate_dofs, J);

    // Compute residual at current iteration
    std::fill(G_k.begin(), G_k.end(), 0);
    for (std::size_t i = 0; i < 3; ++i)
    {
      G_k[0] += (x_k(0, i) - point[i]) * t1[i];
      G_k[1] += (x_k(0, i) - point[i]) * t2[i];
    }

    // Check for convergence in first iteration
    if ((k == 0) and (std::abs(G_k[0]) < tol) and (std::abs(G_k[1]) < tol))
      break;

    /// Compute dG_k/dxi

    // Invert dG_k/dxi

    // Compute dxi and check for convergence

    // Update xi
  }

  // Check if converged  parameters are valid
  std::cout << X_k << " " << x_k << "\n";
  std::cout << xt::adapt(dxi[0]) << "\n;";
  std::tuple<int, std::array<double, 3>, std::array<double, 3>> output
      = std::make_tuple(status, x, X);
  return output;
};