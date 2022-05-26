
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace
{
/// Get function that parameterizes a facet of a given cell
///
/// @param[in] cell_type The cell type
/// @param[in] facet_index The facet index (local to cell)
/// @returns Function that computes the coordinate parameterization of the local
/// facet on the reference cell.
/// @tparam tdim The topological dimension of the cell
template <std::size_t tdim>
std::function<xt::xtensor_fixed<double, xt::xshape<1, tdim>>(
    xt::xtensor_fixed<double, xt::xshape<tdim - 1>>)>
get_parameterization(dolfinx::mesh::CellType cell_type, int facet_index)
{
  switch (cell_type)
  {
  case dolfinx::mesh::CellType::interval:
    throw std::invalid_argument("Unsupported cell type");
    break;
  case dolfinx::mesh::CellType::pyramid:
    throw std::invalid_argument("Unsupported cell type");
    break;
  case dolfinx::mesh::CellType::prism:
    throw std::invalid_argument("Unsupported cell type");
    break;
  default:
    break;
  }

  assert(dolfinx::mesh::cell_dim(cell_type) == tdim);

  if (const int num_facets
      = dolfinx::mesh::cell_num_entities(cell_type, tdim - 1);
      facet_index >= num_facets)
    throw std::invalid_argument(
        "Invalid facet index (larger than number of facets");

  // Get basix geometry information
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(cell_type);
  const xt::xtensor<double, 2> x = basix::cell::geometry(basix_cell);
  const std::vector<std::vector<int>> facets
      = basix::cell::topology(basix_cell)[tdim - 1];

  // Create parameterization function exploiting that the mapping between
  // reference geometries are affine
  std::function<xt::xtensor_fixed<double, xt::xshape<1, tdim>>(
      xt::xtensor_fixed<double, xt::xshape<tdim - 1>>)>
      func = [x, facet = facets[facet_index]](
                 xt::xtensor_fixed<double, xt::xshape<tdim>> xi)
      -> xt::xtensor_fixed<double, xt::xshape<1, tdim>>
  {
    auto x0 = xt::row(x, facet[0]);
    xt::xtensor_fixed<double, xt::xshape<1, tdim>> vals = x0;

    for (std::size_t i = 0; i < tdim; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        vals(0, i) += (xt::row(x, facet[j + 1])[i] - x0[i]) * xi[j];
    return vals;
  };
  return func;
}

/// Get derivative of the parameterization with respect to the input
/// parameters
/// @param[in] cell_type The cell type
/// @param[in] facet_index The facet index (local to cell)
/// @returns The Jacobian of the parameterization
/// @tparam tdim The topological dimension of the cell
template <std::size_t tdim>
xt::xtensor_fixed<double, xt::xshape<tdim, tdim - 1>>
get_parameterization_jacobian(dolfinx::mesh::CellType cell_type,
                              int facet_index)
{
  switch (cell_type)
  {
  case dolfinx::mesh::CellType::interval:
    throw std::invalid_argument("Unsupported cell type");
    break;
  case dolfinx::mesh::CellType::pyramid:
    throw std::invalid_argument("Unsupported cell type");
    break;
  case dolfinx::mesh::CellType::prism:
    throw std::invalid_argument("Unsupported cell type");
    break;
  default:
    break;
  }

  assert(dolfinx::mesh::cell_dim(cell_type) == tdim);

  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(cell_type);
  xt::xtensor<double, 3> facet_jacobians
      = basix::cell::facet_jacobians(basix_cell);

  xt::xtensor_fixed<double, xt::xshape<tdim, tdim - 1>> output;
  output = xt::view(facet_jacobians, facet_index, xt::all(), xt::all());
  return output;
}

} // namespace

namespace dolfinx_contact
{

template <std::size_t tdim>
struct newton_storage
{
  xt::xtensor_fixed<double, xt::xshape<tdim, tdim - 1>>
      dxi; // Jacobian of reference mapping
  xt::xtensor<double, 2>
      X_k; // Solution on reference domain (for Newton solver)
  xt::xtensor<double, 2> x_k; // Solution in physical space (for Newton solver)
  xt::xtensor_fixed<double, xt::xshape<tdim - 1>>
      xi_k; // Reference parameters (for Newton solver)
  xt::xtensor_fixed<double, xt::xshape<tdim - 1>>
      dxi_k; // Gradient of reference parameters (for Newton Solver)
  xt::xtensor_fixed<double, xt::xshape<tdim, tdim>> J; // Jacobian of the cell
  xt::xtensor_fixed<double, xt::xshape<tdim, tdim - 1>>
      dGk_tmp; // Temporary variable to invert Jacobian of Newton solver LHS
  xt::xtensor_fixed<double, xt::xshape<tdim - 1, tdim - 1>>
      dGk; // Newton solver LHS Jacobian
  xt::xtensor_fixed<double, xt::xshape<tdim - 1, tdim - 1>>
      dGk_inv; // Inverse of Newton solver LHS Jacobian
  xt::xtensor_fixed<double, xt::xshape<tdim - 1>>
      Gk; // Residual (RHS) of Newton solver
  xt::xtensor_fixed<double, xt::xshape<tdim - 1, tdim>>
      tangents;                                      // Tangents of ray
  xt::xtensor_fixed<double, xt::xshape<tdim>> point; // Point of origin for ray
};

/// @brief Compute the solution to the ray tracing problem for a single cell
///
/// The implementation solves dot(\Phi(\xi, \eta)-p, t_i)=0, i=1,..,, tdim-1
/// where \Phi(\xi,\eta) is the parameterized mapping from the reference
/// facet to the physical facet, p the point of origin of the ray, and t_i
/// is the ith tangents defining the ray. For more details, see
/// DOI: 10.1016/j.compstruc.2015.02.027 (eq 14).
///
/// @note The problem is solved using Newton's method
///
/// @param[in,out] storage Structure holding all memory required for
/// the newton iteration.
/// @note It is expected that the variables tangents, point, xi is filled with
/// appropriate input values
/// @note All other variables of the class is updated.
/// @param[in, out] basis_values Four-dimensional array to write basis values
/// into.
/// @param[in, out] dphi Two-dimensional matrix to write the derviative of the
/// basis functions into
/// @param[in] max_iter Maximum number of iterations for the Newton solver
/// @param[in] tol The tolerance for termination of the Newton solver
/// @param[in] cmap The coordinate element
/// @param[in] cell_type The cell type of the mesh
/// @param[in] coordinate_dofs The cell geometry
/// @param[in] reference_map Function mapping from reference parameters (xi,
/// eta) to the physical element
/// @tparam tdim The topological dimension of the cell
template <std::size_t tdim>
int raytracing_cell(
    newton_storage<tdim>& storage, xt::xtensor<double, 4>& basis_values,
    xt::xtensor<double, 2>& dphi, int max_iter, double tol,
    const dolfinx::fem::CoordinateElement& cmap,
    dolfinx::mesh::CellType cell_type,
    const xt::xtensor<double, 2>& coordinate_dofs,
    const std::function<xt::xtensor_fixed<double, xt::xshape<1, tdim>>(
        xt::xtensor_fixed<double, xt::xshape<tdim - 1>>)>& reference_map)
{

  // Set initial guess for Newton-iteration (midpoint of facet)
  int status = -1;
  if constexpr (tdim == 3)
  {
    storage.x_k = {{0, 0, 0}};
    storage.xi_k = {0.5, 0.25};
  }
  else if constexpr (tdim == 2)
  {
    storage.x_k = {{0, 0}};
    storage.xi_k = {0.5};
  }
  else
    throw std::invalid_argument("The topological dimenson has to be 2 or 3");

  for (int k = 0; k < max_iter; ++k)
  {
    // Evaluate reference coordinate at current iteration
    storage.X_k = reference_map(storage.xi_k);

    // Tabulate coordinate element basis function
    cmap.tabulate(1, storage.X_k, basis_values);

    // Push forward reference coordinate
    dolfinx::fem::CoordinateElement::push_forward(
        storage.x_k, coordinate_dofs,
        xt::view(basis_values, 0, xt::all(), xt::all(), 0));
    dphi = xt::view(basis_values, xt::xrange((std::size_t)1, tdim + 1), 0,
                    xt::all(), 0);

    // Compute Jacobian
    std::fill(storage.J.begin(), storage.J.end(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                      storage.J);

    // Compute residual at current iteration
    std::fill(storage.Gk.begin(), storage.Gk.end(), 0);
    for (std::size_t i = 0; i < tdim; ++i)
    {
      for (std::size_t j = 0; j < tdim - 1; ++j)
      {
        storage.Gk[j]
            += (storage.x_k(0, i) - storage.point[i]) * storage.tangents(j, i);
      }
    }

    // Check for convergence in first iteration
    if ((k == 0) and (std::abs(storage.Gk[0]) < tol)
        and (std::abs(storage.Gk[1]) < tol))
      break;

    /// Compute dGk/dxi
    std::fill(storage.dGk_tmp.begin(), storage.dGk_tmp.end(), 0);
    dolfinx::math::dot(storage.J, storage.dxi, storage.dGk_tmp);
    std::fill(storage.dGk.begin(), storage.dGk.end(), 0);

    for (std::size_t i = 0; i < tdim - 1; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        for (std::size_t l = 0; l < tdim; ++l)
          storage.dGk(i, j) += storage.dGk_tmp(l, j) * storage.tangents(i, l);

    // Invert dGk/dxi
    if (double det_dGk = dolfinx::math::det(storage.dGk);
        std::abs(det_dGk) < tol)
    {
      status = -2;
      break;
    }
    dolfinx::math::inv(storage.dGk, storage.dGk_inv);

    // Compute dxi
    std::fill(storage.dxi_k.begin(), storage.dxi_k.end(), 0);
    for (std::size_t i = 0; i < tdim - 1; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        storage.dxi_k[i] += storage.dGk_inv(i, j) * storage.Gk[j];

    // Check for convergence
    double norm_dxi = 0;
    for (std::size_t i = 0; i < tdim - 1; i++)
      norm_dxi += storage.dxi_k[i] * storage.dxi_k[i];
    if (norm_dxi < tol * tol)
    {
      status = 1;
      break;
    }

    // Update xi
    std::transform(storage.xi_k.cbegin(), storage.xi_k.cend(),
                   storage.dxi_k.cbegin(), storage.xi_k.begin(),
                   [](auto x, auto y) { return x - y; });
  }
  // Check if converged  parameters are valid
  switch (cell_type)
  {
  case dolfinx::mesh::CellType::tetrahedron:
    if ((storage.xi_k[0] < -tol) or (storage.xi_k[0] > 1 + tol)
        or (storage.xi_k[1] < -tol)
        or (storage.xi_k[1] > 1 - storage.xi_k[0] + tol))
    {
      status = -3;
    }
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if ((storage.xi_k[0] < -tol) or (storage.xi_k[0] > 1 + tol)
        or (storage.xi_k[1] < -tol) or (storage.xi_k[1] > 1 + tol))
    {
      status = -3;
    }
    break;
  case dolfinx::mesh::CellType::triangle:
    if ((storage.xi_k[0] < -tol) or (storage.xi_k[0] > 1 + tol))
    {
      status = -3;
    }
    break;
  case dolfinx::mesh::CellType::quadrilateral:
    if ((storage.xi_k[0] < -tol) or (storage.xi_k[0] > 1 + tol))
    {
      status = -3;
    }
    break;
  default:
    throw std::invalid_argument("Unsupported cell type");
  }
  return status;
}

/// @brief Compute the first intersection between a ray and a set of facets in
/// the mesh templated for the topological dimension.
///
/// @param[in] mesh The mesh
/// @param[in] point The point of origin for the ray
/// @param[in] tangents The tangents of the ray. Each row corresponds to a
/// tangent.
/// @param[in] cells List of tuples (cell, facet), where the cell index is
/// local to process and the facet index is local to the cell cell
/// @param[in] max_iter The maximum number of iterations to use for Newton's
/// method
/// @param[in] tol The tolerance for convergence in Newton's method
/// @returns A triplet (status, cell_idx, points), where x is the
/// convergence status, cell_idx is which entry in the input list the ray
/// goes through and point is the collision point in global and reference
/// coordinates.
/// @note The convergence status is >0 if converging, -1 if the facet is if
/// the maximum number of iterations are reached, -2 if the facet is
/// parallel with the tangent, -3 if the Newton solver finds a solution
/// outside the element.
/// @tparam tdim The topological dimension of the cell
template <std::size_t tdim>
std::tuple<int, std::int32_t, xt::xtensor_fixed<double, xt::xshape<2, tdim>>>
compute_ray(const dolfinx::mesh::Mesh& mesh,
            const xt::xtensor_fixed<double, xt::xshape<tdim>>& point,
            const xt::xtensor_fixed<double, xt::xshape<2, tdim>>& tangents,
            const std::vector<std::pair<std::int32_t, int>>& cells,
            const int max_iter = 25, const double tol = 1e-8)
{
  int status = -1;
  dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
  assert(mesh.topology().dim() == tdim);
  const dolfinx::fem::CoordinateElement& cmap = mesh.geometry().cmap();

  // Get cell coordinates/geometry
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  assert(geometry.dim() == tdim);
  xtl::span<const double> x_g = geometry.x();
  const std::size_t num_dofs_g = cmap.dim();
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, tdim});
  xt::xtensor<double, 2> dphi({(std::size_t)tdim, num_dofs_g});

  // Temporary variables
  const std::array<std::size_t, 4> basis_shape = cmap.tabulate_shape(1, 1);
  xt::xtensor<double, 4> basis_values(basis_shape);

  std::size_t cell_idx = -1;
  newton_storage<tdim> allocated_memory;
  allocated_memory.tangents = tangents;
  allocated_memory.point = point;

  for (std::size_t c = 0; c < cells.size(); ++c)
  {

    // Get cell geometry
    auto [cell, facet_index] = cells[c];
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      dolfinx::common::impl::copy_N<tdim>(
          std::next(x_g.begin(), 3 * x_dofs[j]),
          std::next(coordinate_dofs.begin(), tdim * j));
    }
    // Assign Jacobian of reference mapping
    allocated_memory.dxi
        = get_parameterization_jacobian<tdim>(cell_type, facet_index);

    // Get parameterization map
    auto reference_map = get_parameterization<tdim>(cell_type, facet_index);

    status = raytracing_cell<tdim>(allocated_memory, basis_values, dphi,
                                   max_iter, tol, cmap, cell_type,
                                   coordinate_dofs, reference_map);
    if (status > 0)
    {
      cell_idx = c;
      break;
    }
  }
  if (status < 0)
    LOG(WARNING) << "No ray through the facets have been found";

  xt::xtensor_fixed<double, xt::xshape<2, tdim>> output_coords;
  std::copy(allocated_memory.x_k.cbegin(), allocated_memory.x_k.cend(),
            output_coords.begin());
  std::copy(allocated_memory.X_k.cbegin(), allocated_memory.X_k.cend(),
            std::next(output_coords.begin(), tdim));

  std::tuple<int, std::int32_t, xt::xtensor_fixed<double, xt::xshape<2, tdim>>>
      output = std::make_tuple(status, cell_idx, output_coords);
  return output;
}

/// @brief Compute the first intersection between a ray and a set of facets in
/// the mesh.
///
/// @param[in] mesh The mesh
/// @param[in] point The point of origin for the ray
/// @param[in] tangents The tangents of the ray. Each row corresponds to a
/// tangent.
/// @param[in] cells List of tuples (cell, facet), where the cell index is
/// local to process and the facet index is local to the cell cell
/// @param[in] max_iter The maximum number of iterations to use for Newton's
/// method
/// @param[in] tol The tolerance for convergence in Newton's method
/// @returns A triplet (status, cell_idx, points), where x is the
/// convergence status, cell_idx is which entry in the input list the ray
/// goes through and point is the collision point in global and reference
/// coordinates.
/// @note The convergence status is >0 if converging, -1 if the facet is if
/// the maximum number of iterations are reached, -2 if the facet is
/// parallel with the tangent, -3 if the Newton solver finds a solution
/// outside the element.
std::tuple<int, std::int32_t, xt::xtensor<double, 2>>
raytracing(const dolfinx::mesh::Mesh& mesh, const xt::xtensor<double, 1>& point,
           const xt::xtensor<double, 2>& tangents,
           const std::vector<std::pair<std::int32_t, int>>& cells,
           const int max_iter = 25, const double tol = 1e-8);

} // namespace dolfinx_contact