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

namespace dolfinx_contact
{
/// @brief Generate meshtie kernel for elasticity
///
/// @param[in] type The kernel type (Either `MeshTieJac` or`MeshTieRhs`).
/// @param[in] V The function space
/// @param[in] quadrature_rule The quadrature rule
/// @param[in] cstrides TODO
///
/// @note The ordering of coefficients are expected to be `mu`, `lmbda`,
/// `h`, `test_fn`, `grad(test_fn)`, `u`, `u_opposite`,
/// `grad(u_opposite)`.
/// @note The scalar valued coefficients `mu`,`lmbda` and `h` are
/// expected to be DG-0 functions, with a single value per facet.
/// @note The vector valued coefficents `test_fn`, `grad(test_fn)`, `u`,
/// @note  All other coefficients are packed at quadrature points.
/// `u_opposite`, `grad(u_opposite)` have dimension `bs == gdim`.
kernel_fn<PetscScalar>
generate_meshtie_kernel(Kernel type,
                        const dolfinx::fem::FunctionSpace<double>& V,
                        const QuadratureRule& quadrature_rule,
                        const std::vector<std::size_t>& cstrides);

/// @brief Generate meshtie kernel for poisson
///
/// @note The ordering of coefficients are expected to be `h`,
/// `test_fn`, `grad(test_fn)`, `T`, `grad(T)`, `T_opposite`,
/// `grad(T_opposite)`
/// @note The scalar valued coefficient `h` iis expected to be DG-0
/// functions, with a single value per facet.
/// @note All other coefficients are packed at quadrature points.
///
/// @param[in] type The kernel type (Either `Jac` or`Rhs`).
/// @param[in] V The function space
/// @param[in] quadrature_rule The quadrature rule
/// @param[in] cstrides TODO
kernel_fn<PetscScalar>
generate_poisson_kernel(Kernel type,
                        const dolfinx::fem::FunctionSpace<double>& V,
                        const QuadratureRule& quadrature_rule,
                        const std::vector<std::size_t>& cstrides);
} // namespace dolfinx_contact
