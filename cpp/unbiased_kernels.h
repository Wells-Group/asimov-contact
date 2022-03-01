// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include "utils.h"

using contact_kernel_fn = std::function<void(
    std::vector<std::vector<PetscScalar>>&, const double*, const double*,
    const double*, const int*, const std::uint8_t*, const std::size_t)>;
namespace dolfinx_contact
{

contact_kernel_fn
generate_kernel(dolfinx_contact::Kernel type,
                std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                std::vector<xt::xarray<double>>& qp_ref_facet,
                std::vector<std::vector<double>>& qw_ref_facet,
                std::size_t max_links,
                const basix::FiniteElement basix_element);
}