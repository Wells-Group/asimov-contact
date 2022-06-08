// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once
#include "KernelData.h"
#include "QuadratureRule.h"
#include "elasticity.h"
#include "geometric_quantities.h"
#include "utils.h"

/// @brief Generate contact kernel
///
/// The kernel will expect input on the form
/// @param[in] type The kernel type (Either `Jac` or `Rhs`).
/// @returns Kernel function that takes in a vector (b) to assemble into, the
/// coefficients (`c`), the constants (`w`), the local facet entity (`entity
/// _local_index`), the quadrature permutation and the number of cells on the
/// other contact boundary coefficients are extracted from.
/// @note The ordering of coefficients are expected to be `mu`, `lmbda`, `h`,
/// `gap`, `normals` `test_fn`, `u`, `u_opposite`.
/// @note The scalar valued coefficients `mu`,`lmbda` and `h` are expected to
/// be DG-0 functions, with a single value per facet.
/// @note The coefficients `gap`, `normals`,`test_fn` and `u_opposite` is
/// packed at quadrature points. The coefficient `u` is packed at dofs.
/// @note The vector valued coefficents `gap`, `test_fn`, `u`, `u_opposite`
/// has dimension `bs == gdim`.
namespace dolfinx_contact
{
dolfinx_contact::kernel_fn<PetscScalar>
generate_meshtie_kernel(dolfinx_contact::Kernel type,
                        std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                        std::shared_ptr<const dolfinx_contact::QuadratureRule>,
                        const std::size_t max_links);
} // namespace dolfinx_contact