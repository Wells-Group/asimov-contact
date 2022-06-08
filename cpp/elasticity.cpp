// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include "elasticity.h"

//-----------------------------------------------------------------------------
void dolfinx_contact::compute_normal_strain_basis(
    xt::xtensor<double, 2>& epsn, xt::xtensor<double, 2>& tr,
    const xt::xtensor<double, 2>& K, const xt::xtensor<double, 3>& dphi,
    const std::array<double, 3>& n_surf, const xt::xtensor<double, 1>& n_phys,
    const std::size_t q_pos)
{
  const std::size_t ndofs_cell = epsn.shape(0);
  const std::size_t bs = K.shape(1);
  const std::size_t tdim = K.shape(0);
  assert(K.shape(0) == dphi.shape(0));
  // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
  std::fill(tr.begin(), tr.end(), 0.0);
  std::fill(epsn.begin(), epsn.end(), 0.0);
  for (std::size_t j = 0; j < ndofs_cell; j++)
  {
    for (std::size_t l = 0; l < bs; l++)
    {
      for (std::uint32_t k = 0; k < tdim; k++)
      {
        tr(j, l) += K(k, l) * dphi(k, q_pos, j);
        for (std::size_t s = 0; s < bs; s++)
        {
          epsn(j, l) += K(k, s) * dphi(k, q_pos, j)
                        * (n_phys(s) * n_surf[l] + n_phys(l) * n_surf[s]);
        }
      }
    }
  }
}