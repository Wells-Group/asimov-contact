
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace dolfinx_contact
{

template <int tdim>
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

/// Get function that parameterizes a facet of a given cell
/// @param[in] cell_type The cell type
/// @param[in] facet_index The facet index (local to cell)
/// @returns Function that computes the coordinate parameterization of the local
/// facet on the reference cell.
template <int tdim>
std::function<xt::xtensor_fixed<double, xt::xshape<1, tdim>>(
    xt::xtensor_fixed<double, xt::xshape<2>>)>
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

  const int cell_dim = dolfinx::mesh::cell_dim(cell_type);
  assert(cell_dim == tdim);

  const int num_facets = dolfinx::mesh::cell_num_entities(cell_type, tdim - 1);
  if (facet_index >= num_facets)
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
template <int tdim>
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
  const int cell_dim = dolfinx::mesh::cell_dim(cell_type);
  assert(cell_dim == tdim);

  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(cell_type);
  xt::xtensor<double, 3> facet_jacobians
      = basix::cell::facet_jacobians(basix_cell);

  xt::xtensor_fixed<double, xt::xshape<tdim, tdim - 1>> output;
  output = xt::view(facet_jacobians, facet_index, xt::all(), xt::all());
  return output;
}

/// @brief Compute raytracing with no dynamic memory allocation
///
/// Implementation of 3D ray-tracing, using no dynamic memory allocation
///
/// @param[in,out] storage Structure holding all memory required for
/// the newton iteration.
/// @note It is expected that the variables tangents, point, xi is filled with
/// appropriate values
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
int allocated_3D_ray_tracing(
    newton_storage<3>& storage, xt::xtensor<double, 4>& basis_values,
    xt::xtensor<double, 2>& dphi, int max_iter, double tol,
    const dolfinx::fem::CoordinateElement& cmap,
    dolfinx::mesh::CellType cell_type,
    const xt::xtensor<double, 2>& coordinate_dofs,
    const std::function<xt::xtensor_fixed<double, xt::xshape<1, 3>>(
        xt::xtensor_fixed<double, xt::xshape<2>>)>& reference_map);

/// @brief Compute the intersection between a ray and a facet in the mesh.
///
/// The implementation solves dot(\Phi(\xi, \eta)-p, t_i)=0, i=1,2
/// where \Phi(\xi,\eta) is the parameterized mapping from the reference
/// facet to the physical facet, p the point of origin of the ray, and t_1,
/// t_2 the two tangents defining the ray. For more details, see
/// DOI: 10.1016/j.compstruc.2015.02.027 (eq 14).
///
/// @note The problem is solved using Newton's method
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
std::tuple<int, std::int32_t, xt::xtensor_fixed<double, xt::xshape<2, 3>>>
compute_3D_ray(const dolfinx::mesh::Mesh& mesh,
               const xt::xtensor_fixed<double, xt::xshape<3>>& point,
               const xt::xtensor_fixed<double, xt::xshape<2, 3>>& tangents,
               const std::vector<std::pair<std::int32_t, int>>& cells,
               const int max_iter = 25, const double tol = 1e-8);

} // namespace dolfinx_contact