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

//-----------------------------------------------------------------------------
void dolfinx_contact::compute_sigma_n_basis(xt::xtensor<double, 3>& sig_n,
                                            const xt::xtensor<double, 2>& K,
                                            const xt::xtensor<double, 3>& dphi,
                                            const xt::xtensor<double, 1>& n,
                                            const double mu, const double lmbda,
                                            const std::size_t q_pos)
{
  const std::size_t ndofs_cell = sig_n.shape(0);
  const std::size_t bs = K.shape(1);
  const std::size_t tdim = K.shape(0);
  assert(K.shape(0) == dphi.shape(0));

  // temp variable for grad(v)
  std::vector<double> grad_v(3);

  // Compute sig(v)n
  std::fill(sig_n.begin(), sig_n.end(), 0.0);
  for (std::size_t i = 0; i < ndofs_cell; ++i)
  {
    // Compute grad(v)
    std::fill(grad_v.begin(), grad_v.end(), 0.0);
    for (std::size_t j = 0; j < bs; ++j)
      for (std::size_t k = 0; k < tdim; ++k)
        grad_v[j] += K(k, j) * dphi(k, q_pos, i);

    // Compute dot(grad(v), n)
    double dv_dot_n = 0;
    for (std::size_t j = 0; j < bs; ++j)
      dv_dot_n += grad_v[j] * n(j);

    // Fill sig_n
    for (std::size_t j = 0; j < bs; ++j)
    {
      sig_n(i, j, j) += mu * dv_dot_n;
      for (std::size_t l = 0; l < bs; l++)
        sig_n(i, j, l) += lmbda * grad_v[j] * n(l) + mu * n(j) * grad_v[l];
    }
  }
}

//-----------------------------------------------------------------------------
void dolfinx_contact::compute_sigma_n_u(std::vector<double>& sig_n_u,
                                        xtl::span<const double> grad_u,
                                        const xt::xtensor<double, 1>& n,
                                        const double mu, const double lmbda)
{
  std::size_t gdim = sig_n_u.size();
  for (std::size_t i = 0; i < gdim; ++i)
    for (std::size_t j = 0; j < gdim; ++j)
      sig_n_u[i] += mu * (grad_u[j * gdim + i] + grad_u[i * gdim + j]) * n[j]
                    + lmbda * grad_u[j * gdim + j] * n[i];
}

//-----------------------------------------------------------------------------
void dolfinx_contact::compute_sigma_n_opp(xt::xtensor<double, 4>& sig_n_opp,
                                          xtl::span<const double> grad_v,
                                          const xt::xtensor<double, 1>& n,
                                          const double mu, const double lmbda,
                                          const int q, const int num_q_points)
{
  const std::size_t num_links = sig_n_opp.shape(0);
  const std::size_t ndofs_cell = sig_n_opp.shape(1);
  const std::size_t gdim = sig_n_opp.shape(2);

  // Compute sig(v)n
  std::fill(sig_n_opp.begin(), sig_n_opp.end(), 0.0);
  for (std::size_t i = 0; i < num_links; ++i)
    for (std::size_t j = 0; j < ndofs_cell; ++j)
    {
      std::size_t offset = i * std::size_t(num_q_points) * ndofs_cell * gdim
                           + j * std::size_t(num_q_points) * gdim
                           + std::size_t(q) * gdim;
      // Compute dot(grad(v), n)
      double dv_dot_n = 0;
      for (std::size_t k = 0; k < gdim; ++k)
        dv_dot_n += grad_v[offset + k] * n(k);

      // Fill sig_n
      for (std::size_t k = 0; k < gdim; ++k)
      {
        sig_n_opp(i, j, k, k) += mu * dv_dot_n;
        for (std::size_t l = 0; l < gdim; l++)
          sig_n_opp(i, j, k, l) += lmbda * grad_v[offset + k] * n(l)
                                   + mu * n(k) * grad_v[offset + l];
      }
    }
}