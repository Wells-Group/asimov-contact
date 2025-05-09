// Copyright (C) 2023 Sarah Roggendorf
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

namespace dolfinx_contact
{
/// @brief Generate contact kernel
///
/// @param[in] type The kernel type (`Rhs`, `Jac`, `TrescaRhs`,
/// `TrescaJac`, CoulombRhs, CoulombJac). for the right hand side
/// vector/jacobian for frictionless contact or tresca friction
/// @param[in] V The function space
/// @param[in] quadrature_rule The quadrature rule
/// @param[in] max_links The maximum number of facets linked to one cell
/// @returns Kernel function that takes in a vector (b) to assemble
/// into, the coefficients (`c`), the constants (`w`), the local facet
/// entity (`entity _local_index`), the quadrature permutation and the
/// number of cells on the other contact boundary coefficients are
/// extracted from.
///
/// @note The ordering of coefficients are expected to be `mu`, `lmbda`,
/// `h`, `test_fn`, `grad(test_fn)`, `u`, `u_opposite`,
/// `grad(u_opposite)`.
/// @note The scalar valued coefficients `mu`,`lmbda`, the friciont
/// coefficient `s`, and `h` are expected to be DG-0 functions, with a
/// single value per facet.
/// @note The coefficients`test_fn`, `grad(test_fn)`,  `u_opposite`,
/// `grad(u_opposite)` are packed at quadrature points. The coefficient
/// `u` is packed at dofs.
/// @note The vector valued coefficents `test_fn`, `grad(test_fn)`, `u`,
/// `u_opposite`, `grad(u_opposite)` have dimension `bs == gdim`.
kernel_fn<PetscScalar> generate_contact_kernel(
    Kernel type, const dolfinx::fem::FunctionSpace<double>& V,
    const QuadratureRule& quadrature_rule, std::size_t max_links);
} // namespace dolfinx_contact