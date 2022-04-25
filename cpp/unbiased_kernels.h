// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include "utils.h"
#include "geometric_quantities.h"
#include <dolfinx/common/math.h>

using contact_kernel_fn = std::function<void(
    std::vector<std::vector<PetscScalar>>&, const double*, const double*,
    const double*, const int*, const std::size_t, const std::int32_t*)>;
namespace dolfinx_contact
{

/// @brief Generate contact kernel
///
/// The kernel will expect input on the form
/// @param[in] type The kernel type (Either `Jac_variable_gap` or
/// `Rhs_variable_gap`).
/// @param[in] mesh the mesh
/// @param[in] V the function space
/// @param[in] qp_ref_facet the quadrature points on the reference facet
/// @param[in] qw_ref_facet the quadrature weights on the reference facet
/// @param[in] max_links maximum number of facets connected to a facet by
/// closest point projection
/// @returns Kernel function that takes in a vector (b) to assemble into, the
/// coefficients (`c`), the constants (`w`), the local facet entity (`entity
/// _local_index`), the quadrature permutation and the number of cells on the
/// other contact boundary coefficients are extracted from.
/// @note The ordering of coefficients are expected to be `mu`, `lmbda`, `h`,
/// `gap`, `normals` `test_fn`, `u`, `u_opposite`, `1st derivative of
/// transformation`, `2nd derivative of transformation`.
/// @note The scalar valued coefficients `mu`,`lmbda` and `h` are expected to
/// be DG-0 functions, with a single value per facet.
/// @note The coefficients `gap`, `normals`,`test_fn` and `u_opposite`,`1st
/// derivative of transformation`, `2nd derivative of transformation`  are
/// packed at quadrature points. The coefficient `u` is packed at dofs.
/// @note The vector valued coefficents `gap`, `test_fn`, `u`, `u_opposite`
/// have dimension `bs == gdim`.
contact_kernel_fn
generate_kernel(dolfinx_contact::Kernel type,
                std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                std::vector<xt::xarray<double>>& qp_ref_facet,
                std::vector<std::vector<double>>& qw_ref_facet,
                std::size_t max_links);
}