
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "RayTracing.h"
#include <dolfinx/common/math.h>
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
std::function<xt::xtensor_fixed<double, xt::xshape<1, 3>>(
    xt::xtensor_fixed<double, xt::xshape<2>>)>
get_3D_parameterization(dolfinx::mesh::CellType cell_type, int facet_index)
{

  std::function<xt::xtensor_fixed<double, xt::xshape<1, 3>>(
      xt::xtensor_fixed<double, xt::xshape<2>>)>
      func;
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
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], xi[1], 1 - xi[0] - xi[1]}};
      };
    }
    else if (facet_index == 1)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{0, xi[0], xi[1]}};
      };
    }
    else if (facet_index == 2)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], 0, xi[1]}};
      };
    }
    else if (facet_index == 3)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], xi[1], 0}};
      };
    }
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if (facet_index == 0)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], xi[1], 0}};
      };
    }
    else if (facet_index == 1)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], 0, xi[1]}};
      };
    }
    else if (facet_index == 2)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{0, xi[0], xi[1]}};
      };
    }
    else if (facet_index == 3)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{1, xi[0], xi[1]}};
      };
    }
    else if (facet_index == 4)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], 1, xi[1]}};
      };
    }
    else if (facet_index == 5)
    {
      func = [](xt::xtensor_fixed<double, xt::xshape<2>> xi)
          -> xt::xtensor_fixed<double, xt::xshape<1, 3>> {
        return {{xi[0], xi[1], 1}};
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
xt::xtensor_fixed<double, xt::xshape<3, 2>>
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
      return {{1, 0}, {0, 1}, {-1, -1}};
    else if (facet_index == 1)
      return {{0, 0}, {1, 0}, {0, 1}};

    else if (facet_index == 2)
      return {{1, 0}, {0, 0}, {0, 1}};
    else if (facet_index == 3)
      return {{1, 0}, {0, 1}, {0, 0}};
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if ((facet_index == 0) or (facet_index == 5))
      return {{1, 0}, {0, 1}, {0, 0}};
    else if ((facet_index == 1) or (facet_index == 4))
      return {{1, 0}, {0, 0}, {0, 1}};
    else if ((facet_index == 2) or (facet_index) == 3)
      return {{0, 0}, {1, 0}, {0, 1}};
    break;
  default:
    throw std::invalid_argument("Unsupported cell type");
    break;
  }
  xt::xtensor_fixed<double, xt::xshape<3, 2>> output;
  return output;
}

} // namespace

std::tuple<int, xt::xtensor_fixed<double, xt::xshape<3>>,
           xt::xtensor_fixed<double, xt::xshape<3>>>
dolfinx_contact::compute_3D_ray(
    const dolfinx::mesh::Mesh& mesh,
    const xt::xtensor_fixed<double, xt::xshape<3>>& point,
    const xt::xtensor_fixed<double, xt::xshape<2, 3>>& tangents, int cell,
    int facet_index, const int max_iter, const double tol)
{
  int status = -1;
  dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
  const int tdim = mesh.topology().dim();

  // Get parameterization function and jacobian
  auto xi = get_3D_parameterization(cell_type, facet_index);
  xt::xtensor_fixed<double, xt::xshape<3, 2>> dxi
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
  if ((gdim != tdim) or (gdim != 3))
  {
    throw std::invalid_argument(
        "This raytracing algorithm is specialized for meshes with topological "
        "and geometrical dimension 3");
  }
  std::cout << "cell coords " << coordinate_dofs << "\n";
  // Temporary variables
  xt::xtensor<double, 2> X_k({1, 3});
  xt::xtensor<double, 2> x_k({1, 3});
  xt::xtensor_fixed<double, xt::xshape<2>> xi_k = {0.5, 0.25};
  xt::xtensor_fixed<double, xt::xshape<2>> dxi_k;

  xt::xtensor_fixed<double, xt::xshape<3, 3>> J;
  xt::xtensor<double, 2> dphi({(std::size_t)tdim, num_dofs_g});
  xt::xtensor_fixed<double, xt::xshape<3, 2>> dGk_tmp;
  xt::xtensor_fixed<double, xt::xshape<2, 2>> dGk;
  xt::xtensor_fixed<double, xt::xshape<2, 2>> dGk_inv;
  xt::xtensor_fixed<double, xt::xshape<2>> Gk;
  for (int k = 0; k < max_iter; ++k)
  {
    // Evaluate reference coordinate at current iteration
    X_k = xi(xi_k);

    // Tabulate coordinate element basis function
    cmap.tabulate(1, X_k, basis_values);

    // Push forward reference coordinate
    cmap.push_forward(x_k, coordinate_dofs,
                      xt::view(basis_values, 0, xt::all(), xt::all(), 0));
    dphi = xt::view(basis_values, xt::xrange(1, tdim + 1), 0, xt::all(), 0);

    // Compute Jacobian
    std::fill(J.begin(), J.end(), 0);
    cmap.compute_jacobian(dphi, coordinate_dofs, J);

    // Compute residual at current iteration
    std::fill(Gk.begin(), Gk.end(), 0);
    for (std::size_t i = 0; i < 3; ++i)
    {
      Gk[0] += (x_k(0, i) - point[i]) * tangents(0, i);
      Gk[1] += (x_k(0, i) - point[i]) * tangents(1, i);
    }

    // Check for convergence in first iteration
    if ((k == 0) and (std::abs(Gk[0]) < tol) and (std::abs(Gk[1]) < tol))
      break;

    /// Compute dGk/dxi
    std::fill(dGk_tmp.begin(), dGk_tmp.end(), 0);
    dolfinx::math::dot(J, dxi, dGk_tmp);
    std::fill(dGk.begin(), dGk.end(), 0);

    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        for (std::size_t l = 0; l < 3; ++l)
          dGk(i, j) += dGk_tmp(l, j) * tangents(i, l);

    // Invert dGk/dxi
    double det_dGk = dolfinx::math::det(dGk);
    if (std::abs(det_dGk) < tol)
    {
      status = -2;
      break;
    }
    dolfinx::math::inv(dGk, dGk_inv);

    // Compute dxi
    std::fill(dxi_k.begin(), dxi_k.end(), 0);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        dxi_k[i] += dGk_inv(i, j) * Gk[j];

    // Check for convergence
    if ((dxi_k[0] * dxi_k[0] + dxi_k[1] * dxi_k[1]) < tol * tol)
    {
      status = 1;
      break;
    }

    // Update xi
    std::transform(xi_k.cbegin(), xi_k.cend(), dxi_k.cbegin(), xi_k.begin(),
                   [](auto x, auto y) { return x - y; });
  }
  // Check if converged  parameters are valid
  switch (cell_type)
  {
  case dolfinx::mesh::CellType::tetrahedron:
    if ((xi_k[0] < 0) or (xi_k[0] > 1) or (xi_k[1] < 0)
        or (xi_k[1] > 1 - xi_k[0]))
    {
      status = -3;
    }
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if ((xi_k[0] < 0) or (xi_k[0] > 1) or (xi_k[1] < 0) or (xi_k[1] > 1))
    {
      status = -3;
    }
    break;
  default:
    throw std::invalid_argument("Invalid cell type");
  }

  std::tuple<int, xt::xtensor_fixed<double, xt::xshape<3>>,
             xt::xtensor_fixed<double, xt::xshape<3>>>
      output = std::make_tuple(status, x_k, X_k);
  return output;
};