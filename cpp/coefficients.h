// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>

namespace dolfinx_contact
{
/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as an array, where j is the index of the local cell
/// and c[j*cstride + q * (block_size * value_size) + k + c] = sum_i c^i[k] *
/// phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth vector component,
/// phi^i(x_q)[c] is the ith basis function's c-th value compoenent at the
/// quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[in] q_degree The quadrature degree
/// @returns c The packed coefficients and the number of coeffs per cell
std::pair<std::vector<PetscScalar>, int> pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree);

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as an array, where j corresponds to the jth facet in
/// active_facets and c[j*cstride + q * (block_size * value_size) + k + c] =
/// sum_i c^i[k] * phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth
/// vector component, phi^i(x_q)[c] is the ith basis function's c-th value
/// compoenent at the quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[in] active_facets List of active facets
/// @param[in] q_degree the quadrature degree
/// @param[out] c The packed coefficients and the number of coeffs per facet
std::pair<std::vector<PetscScalar>, int> pack_coefficient_facet(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    int q_degree, const xtl::span<const std::int32_t>& active_facets);

} // namespace dolfinx_contact