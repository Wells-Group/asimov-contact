// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
#include "utils.h"

namespace dolfinx_contact
{
/// Generate one-sided contact kernels
///
/// @param[in] V The function space of the trial/test function
/// @param[in] type The kernel type (Rhs or Jac)
/// @param[in] quadrature_rule The quadrature rule used in kernels
/// @param[in] constant_normal Boolean indicating if normal is constant
/// at every point of the cell.
/// @returns The integration kernel
kernel_fn<PetscScalar>
generate_rigid_surface_kernel(const dolfinx::fem::FunctionSpace<double>& V,
                              Kernel type, QuadratureRule& quadrature_rule,
                              bool constant_normal);
} // namespace dolfinx_contact
