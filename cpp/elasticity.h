// Copyright (C) 2022  Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

// This file contains helper functions that are useful for writing elasticity
// kernels

#include "QuadratureRule.h"
#include <span>
#include <cmath>

namespace dolfinx_contact
{
/// @brief compute dot(eps(dphi(q_pos))*n_2, n_1) and tr(eps)
///
/// Given the gradient of the basis functions, compute dot(eps*n_1, n_2) and
/// tr(eps) for the basis function
/// @param[in, out] epsn  dot(eps*n_1, n_2)
/// @param[in, out] tr    tr(eps)
/// @param[in] K          The inverse jacobian at the quadrature point
/// @param[in] dphi       The gradients of the basis functions
/// @param[in] n_1        1st normal vector, typically n_surf
/// @param[in] n_2        2nd normal vector, typically n_phys
/// @param[in] q_pos      offset of quadrature point for accessing dphi
void compute_normal_strain_basis(mdspan2_t epsn, mdspan2_t tr, cmdspan2_t K,
                                 cmdspan3_t dphi,
                                 const std::array<double, 3>& n_1,
                                 std::span<const double> n_2,
                                 const std::size_t q_pos);

/// @brief Compute sigma(v)*n for all test functions in dphi at quadrature point
/// q_pos
///
/// @param[in, out] sig_n Variable to store sigma(v)*n (will be reinitialized)
/// Shape of sig_n is expected to be (ndofs_cell, gdim, gdim) (bs == gdim
/// assumed)
/// sig_n(i, j, k) contains the kth entry corresponding to the jth basis
/// function in the ith dof-node
/// @param[in] K     The inverse jacobian at the quadrature point
/// @param[in] dphi  The gradients of the basis functions
/// @param[in] n     The normal vector
/// @param[in] mu    The poisson ratio
/// @param[in] lmbda The 1st Lame parameter
/// @param[in] q_pos The offset of quadrature point for accessing dphi
void compute_sigma_n_basis(mdspan3_t sig_n, cmdspan2_t K, cmdspan3_t dphi,
                           std::span<const double> n, const double mu,
                           const double lmbda, const std::size_t q_pos);

/// @brief Compute sigma(u)*n from grad(u)
///
/// @param[in] sig_n_u Variable to store sigma(u)*n
/// @param[in] grad_u  The gradient of u
/// @param[in] n       The normal vector
/// @param[in] mu      The poisson ratio
/// @param[in] lmbda   The 1st Lame parameter
void compute_sigma_n_u(std::span<double> sig_n_u,
                       std::span<const double> grad_u,
                       std::span<const double> n, const double mu,
                       const double lmbda);

/// @brief Compute sigma(v)*n from the gradients of v evaluated on opposite
/// contact surface in qth quadrature point
///
/// @param[in, out] sig_n_opp Variable to store sigma(v)*n (will be
/// reinitialized) Shape of sig_n is expected to be (num_links, ndofs_cell,
/// gdim, gdim) (bs == gdim assumed) sig_n(i, j, k, l) contains the lth entry
/// corresponding to the kth basis function in the jth dof-node for the ith
/// linked facet
/// @param[in] grad_v The gradients of the test functions
/// @param[in] n      The normal vector
/// @param[in] mu     The poisson ratio
/// @param[in] lmbda  The 1st Lame parameter
/// @param[in] q_pos  The offset of quadrature point for accessing dphi
void compute_sigma_n_opp(mdspan4_t sig_n_opp, std::span<const double> grad_v,
                         std::span<const double> n, const double mu,
                         const double lmbda, const std::size_t q,
                         const std::size_t num_q_points);

/// @brief Compute contact pressure dot(sigma(u)*n_x, n_contact) from grad(u),
/// n_x, n_contact
///
/// all input data packed at quadrature points for each facet
/// @param[in] grad_u       The gradient of u
/// @param[in] n_x          The surface normal
/// @param[in] n_contact    The normal used in contact integrals (-n_x for raytracing, n_y for CPP)
/// @param[in] num_q_points The number of quadrature points
/// @param[in] num_facets   The number of facets
/// @param[in] gdim         The geometrical dimension of the mesh
/// @param[in] mu           The first Lame parameter
/// @param[in] lmbda        The second Lame parameter
/// @return vector containing surface pressure values for each quadrature point on each facet
std::vector<double>compute_contact_forces(
    std::span<const double> grad_u, std::span<const double> n_x, const std::size_t num_q_points,
    std::size_t num_facets, const std::size_t gdim, const double mu,
    const double lmbda);

} // namespace dolfinx_contact