// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <xtl/xspan.hpp>

namespace dolfinx_contact
{

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom
/// kernels by packing them as an array, where j corresponds to the jth entity
/// active_entities and c[j*cstride + q * (block_size * value_size) + k + c] =
/// sum_i c^i[k] * phi^i(x_q)[c] where c^i[k] is the ith coefficient's kth
/// vector component, phi^i(x_q)[c] is the ith basis function's c-th value
/// compoenent at the quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[in] q_degree The quadrature degree
/// @param[in] integral The integral type (cell or exterior facet)
/// @param[in] active_entities List of active entities (cells or exterior
/// facets)
/// @returns c The packed coefficients and the number of coeffs per entity
std::pair<std::vector<PetscScalar>, int> pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree, dolfinx::fem::IntegralType integral,
    const xtl::span<const std::int32_t>& active_entities);

/// Prepare circumradii of triangle/tetrahedron for assembly with custom
/// kernels by packing them as an array, where the j*cstride to the ith facet
/// int active_facets.
/// @param[in] mesh
/// @param[in] active_facets List of active facets
/// @param[out] c The packed coefficients and the number of coeffs per facet
std::pair<std::vector<PetscScalar>, int>
pack_circumradius_facet(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        const xtl::span<const std::int32_t>& active_facets);
} // namespace dolfinx_contact