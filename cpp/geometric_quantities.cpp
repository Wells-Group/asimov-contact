// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "geometric_quantities.h"
#include "utils.h"
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>

using namespace dolfinx_contact;

xt::xtensor<double, 2> dolfinx_contact::push_forward_facet_normal(
    const xt::xtensor<double, 2>& x, xt::xtensor<double, 3>& J,
    xt::xtensor<double, 3>& K, const xt::xtensor<double, 2>& coordinate_dofs,
    const xt::xtensor<std::int32_t, 1>& facet_indices,
    const dolfinx::fem::CoordinateElement& cmap,
    const xt::xtensor<double, 2>& reference_normals)
{
  assert(J.shape(0) >= x.shape(0));
  assert(K.shape(0) >= x.shape(0));

  // Shapes needed for computing the Jacobian inverse
  const std::size_t num_points = x.shape(0);
  const size_t gdim = coordinate_dofs.shape(1);
  const size_t tdim = K.shape(1);

  // Data structures for computing J inverse
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
  xt::xtensor<double, 2> dphi({tdim, cmap.tabulate_shape(1, 1)[2]});

  xt::xtensor<double, 2> X({num_points, tdim});

  // Compute Jacobian inverse
  if (cmap.is_affine())
  {
    J.fill(0);
    // Affine Jacobian can be computed at any point in the cell (0,0,0) in the
    // reference cell
    X.fill(0);
    auto _J = xt::view(J, 0, xt::all(), xt::all());
    auto _K = xt::view(K, 0, xt::all(), xt::all());
    cmap.tabulate(1, X, phi);
    dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                      _J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
    for (std::size_t p = 1; p < num_points; ++p)
    {
      xt::view(J, p, xt::all(), xt::all()) = _J;
      xt::view(K, p, xt::all(), xt::all()) = _K;
    }
  }
  else
  {
    // For non-affine geometries we have to compute the point in the reference
    // cell, which is a nonlinear operation. Internally cmap uses a
    // Newton-solver to get the reference coordinates X
    cmap.pull_back_nonaffine(X, x, coordinate_dofs);
    cmap.tabulate(1, X, phi);
    J.fill(0);
    for (std::size_t p = 0; p < num_points; ++p)
    {
      dphi = xt::view(phi, xt::range(1, tdim + 1), p, xt::all(), 0);
      auto _J = xt::view(J, p, xt::all(), xt::all());
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(
          _J, xt::view(K, p, xt::all(), xt::all()));
    }
  }

  xt::xtensor<double, 2> normals = xt::zeros<double>({num_points, gdim});
  for (std::size_t q = 0; q < num_points; ++q)
  {
    // Compute normal of physical facet using a normalized covariant Piola
    // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
    // DOI: 10.1137/08073901X
    auto _K = xt::view(K, q, xt::all(), xt::all());
    auto facet_normal = xt::row(reference_normals, facet_indices[q]);
    for (std::size_t i = 0; i < gdim; i++)
    {
      for (std::size_t j = 0; j < tdim; j++)
        normals(q, i) += _K(j, i) * facet_normal[j];
    }
    // Normalize vector
    auto n_q = xt::row(normals, q);
    auto n_norm = xt::norm_l2(n_q);
    n_q /= n_norm;
  }
  return normals;
}
