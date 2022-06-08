// Copyright (C) 2022  Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

// This file contains helper functions that are useful for writing elasticity
// kernels

#include <xtensor/xadapt.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xindex_view.hpp>

namespace dolfinx_contact
{
/// @brief compute dot(eps*n_2, n_1) and tr(eps)
///
/// Given the gradient of the basis functions, compute dot(eps*n_1, n_2) and
/// tr(eps) for the basis function
/// @param[in, out] epsn  dot(eps*n_1, n_2)
/// @param[in, out] tr    tr(eps)
/// @param[in] K          The jacobian at the quadrature point
/// @param[in] dphi       The gradients of the basis functions
/// @param[in] n_1        1st normal vector, typically n_surf
/// @param[in] n_2        2nd normal vector, typically n_phys
/// @param[in] q_pos      offset of quadrature point for accessing dphi
void compute_normal_strain_basis(xt::xtensor<double, 2>& epsn,
                                 xt::xtensor<double, 2>& tr,
                                 const xt::xtensor<double, 2>& K,
                                 const xt::xtensor<double, 3>& dphi,
                                 const std::array<double, 3>& n_1,
                                 const xt::xtensor<double, 1>& n_2,
                                 const std::size_t q_pos);



} // namespace dolfinx_contact