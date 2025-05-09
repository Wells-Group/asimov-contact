// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "elasticity.h"

namespace stdex = std::experimental;

//-----------------------------------------------------------------------------
void dolfinx_contact::compute_normal_strain_basis(
    mdspan_t<double, 2> epsn, mdspan_t<double, 2> tr,
    mdspan_t<const double, 2> K,
    mdspan_t<const double, 3, std::experimental::layout_stride> dphi,
    std::array<double, 3> n_surf, std::span<const double> n_phys,
    std::size_t q_pos)
{
  const std::size_t ndofs_cell = epsn.extent(0);
  const std::size_t bs = K.extent(1);
  const std::size_t tdim = K.extent(0);
  assert(K.extent(0) == dphi.extent(0));
  // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
  // Note could merge these loops
  for (std::size_t i = 0; i < tr.extent(0); ++i)
    for (std::size_t j = 0; j < tr.extent(1); ++j)
      tr(i, j) = 0;
  for (std::size_t i = 0; i < epsn.extent(0); ++i)
    for (std::size_t j = 0; j < epsn.extent(1); ++j)
      epsn(i, j) = 0;

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
                        * (n_phys[s] * n_surf[l] + n_phys[l] * n_surf[s]);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::compute_sigma_n_basis(
    mdspan_t<double, 3> sig_n, mdspan_t<const double, 2> K,
    mdspan_t<const double, 3, std::experimental::layout_stride> dphi,
    std::span<const double> n, double mu, double lmbda, std::size_t q_pos)
{
  const std::size_t ndofs_cell = sig_n.extent(0);
  const std::size_t bs = K.extent(1);
  const std::size_t tdim = K.extent(0);
  assert(K.extent(0) == dphi.extent(0));

  // temp variable for grad(v)
  std::array<double, 3> grad_v;

  // Compute sig(v)n
  for (std::size_t i = 0; i < sig_n.extent(0); ++i)
    for (std::size_t j = 0; j < sig_n.extent(1); ++j)
      for (std::size_t k = 0; k < sig_n.extent(2); ++k)
        sig_n(i, j, k) = 0;

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
      dv_dot_n += grad_v[j] * n[j];

    // Fill sig_n
    for (std::size_t j = 0; j < bs; ++j)
    {
      sig_n(i, j, j) += mu * dv_dot_n;
      for (std::size_t l = 0; l < bs; l++)
        sig_n(i, j, l) += lmbda * grad_v[j] * n[l] + mu * n[j] * grad_v[l];
    }
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::compute_sigma_n_u(std::span<double> sig_n_u,
                                        std::span<const double> grad_u,
                                        std::span<const double> n, double mu,
                                        double lmbda)
{
  std::size_t gdim = sig_n_u.size();
  for (std::size_t i = 0; i < gdim; ++i)
    for (std::size_t j = 0; j < gdim; ++j)
      sig_n_u[i] += mu * (grad_u[j * gdim + i] + grad_u[i * gdim + j]) * n[j]
                    + lmbda * grad_u[j * gdim + j] * n[i];
}
//-----------------------------------------------------------------------------
void dolfinx_contact::compute_sigma_n_opp(mdspan_t<double, 4> sig_n_opp,
                                          std::span<const double> grad_v,
                                          std::span<const double> n, double mu,
                                          double lmbda, std::size_t q,
                                          std::size_t num_q_points)
{
  const std::size_t num_links = sig_n_opp.extent(0);
  const std::size_t ndofs_cell = sig_n_opp.extent(1);
  const std::size_t gdim = sig_n_opp.extent(2);

  // Compute sig(v)n
  for (std::size_t i = 0; i < sig_n_opp.extent(0); ++i)
    for (std::size_t j = 0; j < sig_n_opp.extent(1); ++j)
      for (std::size_t k = 0; k < sig_n_opp.extent(2); ++k)
        for (std::size_t l = 0; l < sig_n_opp.extent(3); ++l)
          sig_n_opp(i, j, k, l) = 0;

  for (std::size_t i = 0; i < num_links; ++i)
    for (std::size_t j = 0; j < ndofs_cell; ++j)
    {
      std::size_t offset = i * num_q_points * ndofs_cell * gdim
                           + j * num_q_points * gdim + q * gdim;
      // Compute dot(grad(v), n)
      double dv_dot_n = 0;
      for (std::size_t k = 0; k < gdim; ++k)
        dv_dot_n += grad_v[offset + k] * n[k];

      // Fill sig_n
      for (std::size_t k = 0; k < gdim; ++k)
      {
        sig_n_opp(i, j, k, k) += mu * dv_dot_n;
        for (std::size_t l = 0; l < gdim; l++)
          sig_n_opp(i, j, k, l) += lmbda * grad_v[offset + k] * n[l]
                                   + mu * n[k] * grad_v[offset + l];
      }
    }
}
//-----------------------------------------------------------------------------
std::vector<double> dolfinx_contact::compute_contact_forces(
    std::span<const double> grad_u, std::span<const double> n_x,
    std::size_t num_q_points, std::size_t num_facets, std::size_t gdim,
    double mu, double lmbda)
{
  std::vector<double> sig_n_u(gdim);
  std::vector<double> sign(num_facets * num_q_points * gdim, 0.0);
  for (std::size_t f = 0; f < num_facets; ++f)
  {
    std::size_t f_offset = num_q_points * gdim * f;
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      // compute sig(u)*n_x
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(
          sig_n_u,
          grad_u.subspan(f_offset * gdim + q * gdim * gdim, gdim * gdim),
          n_x.subspan(f_offset + q * gdim, gdim), mu, lmbda);

      for (std::size_t j = 0; j < gdim; ++j)
        sign[f * num_q_points * gdim + q * gdim + j] = sig_n_u[j];
    }
  }
  return sign;
}
//-----------------------------------------------------------------------------
