// Copyright (C) 2021-2022 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include "QuadratureRule.h"
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <variant>
namespace dolfinx_contact
{

/// @brief Apply dof transformations to basis functions and push forward to
/// physical space
/// @param[in] element The finite element
/// @param[in] reference_basis The basis functions tabulated on the reference
/// element
/// @param[in] element_basisb Storage for transformed basis
/// @param[in, out] _u The basis values in the physical space
/// @param[in] J The jacobian
/// @param[in] K The inverse of the jacobian
/// @param[in] detJ The determinant of the jacobian
/// @param[in] basis_offset The offset determining where to access the input
/// data
/// @param[in] q The qudrature point
/// @param[in] cell The index of the cell
/// @param[in] cell_info The cell info
void transformed_push_forward(const dolfinx::fem::FiniteElement* element,
                              cmdspan4_t reference_basis,
                              std::vector<double>& element_basisb,
                              mdspan3_t basis_values, mdspan2_t J, mdspan2_t K,
                              double detJ, std::size_t basis_offset,
                              std::size_t q, std::int32_t cell,
                              std::span<const std::uint32_t> cell_info);
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
/// @param[in] active_entities List of active entities.
/// @param[in] integral The integral type (cells or exterior facet)
/// @returns c The packed coefficients and the number of coeffs per entity
std::pair<std::vector<PetscScalar>, int> pack_coefficient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree, std::span<const std::int32_t> active_entities,
    dolfinx::fem::IntegralType integral);

/// @brief Pack the gradient of a coefficient at quadrature points.
///
/// Prepare a the gradient of a coefficient (dolfinx::fem::Function) for
/// assembly with custom kernels by packing them as a vector. The vector is
/// packed such that the derivatives of the coefficients for the jth entry in
/// active_entities is in the range c[j*cstride:(j+1)*cstride] where
/// cstride=(num_quadrature_points)*block_size*value_size*gdim, where
/// block_size*value_size is the number of components in coeff.
///
/// @note For the `j`th entry, the coefficients are packed per quadrature point,
/// i.e. c[j*cstride + q * (block_size * value_size)*gdim + gdim*(k + c) + l] =
/// sum_i c^i[k] * (dphi/dl)^i(x_q)[c] where c^i[k] is the ith coefficient's kth
/// vector component, (dphi/dl)^i(x_q)[c] is the c-th value component at the
/// quadrature point x_q of the derivative with respect to x_l of the ith basis
/// function.
///
/// @param[in] coeff The coefficient to pack
/// @param[in] q_degree The quadrature degree
/// @param[in] integral The integral type (cell or exterior facet)
/// @param[in] active_entities List of active entities.
/// @param[in] integral The integral type (cells or exterior facet)
/// @returns c The packed coefficients and the number of coeffs per entity
std::pair<std::vector<PetscScalar>, int> pack_gradient_quadrature(
    std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
    const int q_degree, std::span<const std::int32_t> active_entities,
    dolfinx::fem::IntegralType integral);

/// Prepare circumradii of triangle/tetrahedron for assembly with custom
/// kernels by packing them as an array, where the ith entry of the output
/// corresponds to the circumradius of the ith cell facet pair.
/// @param[in] mesh The mesh
/// @param[in] active_facets List of (cell, local_facet_index) tuples
/// @returns[out] The packed coefficients
/// @note Circumradius is constant and therefore the cstride is 1
std::vector<PetscScalar>
pack_circumradius(const dolfinx::mesh::Mesh& mesh,
                  const std::span<const std::int32_t>& active_facets);
} // namespace dolfinx_contact