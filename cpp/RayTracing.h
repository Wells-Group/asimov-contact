
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
/// two tangents defining the ray.
///
/// @note The problem is solved using Newton's method
///
/// @param[in] mesh The mesh
/// @param[in] point The point of origin for the ray
/// @param[in] t1 The first tangent of the ray
/// @param[in] t2 The second tangent of the ray
/// @param[in] cell The cell index (local to process)
/// @param[in] facet_index The facet index (local to cell)
/// @param[in] max_iter The maximum number of iterations to use for Newton's
/// method
/// @param[in] tol The tolerance for convergence in Newton's method
/// @returns A triplet (status, x, X), where x is the convergence status, x the
/// point in physical space, X the point in the reference cell.
/// @note The convergence status is >0 if converging, -1 if the facet is if the
/// maximum number of iterations are reached, -2 if the facet is parallel with
/// the tangent, -3 if the Newton solver finds a solution outside the element.
std::tuple<int, std::array<double, 3>, std::array<double, 3>> compute_3D_ray(
    const dolfinx::mesh::Mesh& mesh, const std::array<double, 3>& point,
    const std::array<double, 3>& t1, const std::array<double, 3>& t2, int cell,
    int facet_index, const int max_iter = 25, const double tol = 1e-8);

} // namespace dolfinx_contact