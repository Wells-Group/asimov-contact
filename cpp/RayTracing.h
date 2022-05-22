
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xtensor.hpp>

namespace dolfinx_contact
{
struct newton_3D_storage
{
  xt::xtensor_fixed<double, xt::xshape<3, 2>>
      dxi; // Jacobian of reference mapping
  xt::xtensor<double, 2>
      X_k; // Solution on reference domain (for Newton solver)
  xt::xtensor<double, 2> x_k; // Solution in physical space (for Newton solver)
  xt::xtensor_fixed<double, xt::xshape<2>>
      xi_k; // Reference parameters (for Newton solver)
  xt::xtensor_fixed<double, xt::xshape<2>>
      dxi_k; // Gradient of reference parameters (for Newton Solver)
  xt::xtensor_fixed<double, xt::xshape<3, 3>> J; // Jacobian of the cell
  xt::xtensor_fixed<double, xt::xshape<3, 2>>
      dGk_tmp; // Temporary variable to invert Jacobian of Newton solver LHS
  xt::xtensor_fixed<double, xt::xshape<2, 2>> dGk; // Newton solver LHS Jacobian
  xt::xtensor_fixed<double, xt::xshape<2, 2>>
      dGk_inv; // Inverse of Newton solver LHS Jacobian
  xt::xtensor_fixed<double, xt::xshape<2>>
      Gk; // Residual (RHS) of Newton solver
  xt::xtensor_fixed<double, xt::xshape<2, 3>> tangents; // Tangents of ray
  xt::xtensor_fixed<double, xt::xshape<3>> point; // Point of origin for ray
};

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
    newton_3D_storage& storage, xt::xtensor<double, 4>& basis_values,
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