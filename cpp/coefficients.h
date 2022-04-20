// Copyright (C) 2021-2022 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <variant>
#include <xtl/xspan.hpp>
namespace dolfinx_contact
{

/// @brief Pack a coefficient at quadrature points.
///
/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as a vector. The vector is packed such that the
/// coefficients for the jth entry in active_entities is in the range
/// c[j*cstride:(j+1)*cstride] where
/// cstride=(num_quadrature_points)*block_size*value_size, where
/// block_size*value_size is the number of components in coeff.
///
/// @note For the `j`th entry, the coefficients are packed per quadrature point,
/// i.e. c[j*cstride + q * (block_size * value_size) + k + c] = sum_i c^i[k] *
/// phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth vector component,
/// phi^i(x_q)[c] is the ith basis function's c-th value component at the
/// quadrature point x_q.
///
/// @param[in] coeff The coefficient to pack
/// @param[in] q_degree The quadrature degree
/// @param[in] integral The integral type (cell or exterior facet)
/// @param[in] active_entities List of active entities (cells or exterior
/// facets)
/// @returns c The packed coefficients and the number of coeffs per entity
std::pair<std::vector<PetscScalar>, int> pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree,
    std::variant<tcb::span<const std::int32_t>,
                 tcb::span<const std::pair<std::int32_t, int>>>
        active_entities);

/// Prepare circumradii of triangle/tetrahedron for assembly with custom
/// kernels by packing them as an array, where the j*cstride to the ith facet
/// int active_facets.
/// @param[in] mesh The mesh
/// @param[in] active_facets List of (cell, local_facet_index) tuples
/// @returns[out] c The packed coefficients and the number of coeffs per facet
std::pair<std::vector<PetscScalar>, int> pack_circumradius(
    const dolfinx::mesh::Mesh& mesh,
    const tcb::span<const std::pair<std::int32_t, int>>& active_facets);
} // namespace dolfinx_contact