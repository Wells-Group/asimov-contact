
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/mesh/Mesh.h>

namespace dolfinx_contact
{

/// @brief Compute the intersection between a ray and a facet in the mesh.
///
/// The implementation solves dot(\Phi(\xi, \eta)-p, t_i)=0, i=1,2
/// where \Phi(\xi,\eta) is the parameterized mapping from the reference facet
/// to the physical facet, p the point of origin of the ray, and t_1, t_2 the
/// two tangents defining the ray. For more details, see
/// DOI: 10.1016/j.compstruc.2015.02.027 (eq 14).
///
/// @note The problem is solved using Newton's method
///
/// @param[in] mesh The mesh
/// @param[in] point The point of origin for the ray
/// @param[in] tangents The tangents of the ray. Each row corresponds to a
/// tangent.
/// @param[in] cells List of tuples (cell, facet), where the cell index is local
/// to process and the facet index is local to the cell cell
/// @param[in] max_iter The maximum number of iterations to use for Newton's
/// method
/// @param[in] tol The tolerance for convergence in Newton's method
/// @returns A triplet (status, cell_idx, points), where x is the convergence
/// status, cell_idx is which entry in the input list the ray goes through
/// and point is the collision point in global and reference coordinates.
/// @note The convergence status is >0 if converging, -1 if the facet is if the
/// maximum number of iterations are reached, -2 if the facet is parallel with
/// the tangent, -3 if the Newton solver finds a solution outside the element.
std::tuple<int, std::int32_t, xt::xtensor_fixed<double, xt::xshape<2, 3>>>
compute_3D_ray(const dolfinx::mesh::Mesh& mesh,
               const xt::xtensor_fixed<double, xt::xshape<3>>& point,
               const xt::xtensor_fixed<double, xt::xshape<2, 3>>& tangents,
               const std::vector<std::pair<std::int32_t, int>>& cells,
               const int max_iter = 25, const double tol = 1e-8);

} // namespace dolfinx_contact