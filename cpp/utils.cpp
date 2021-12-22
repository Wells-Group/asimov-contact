// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include "utils.h"

using namespace dolfinx_contact;

xt::xtensor<double, 2> dolfinx_contact::get_facet_normals(
    xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
    xt::xtensor<double, 1>& detJ, const xt::xtensor<double, 2>& x,
    xt::xtensor<double, 2> coordinate_dofs, const std::int32_t index,
    const xt::xtensor<std::int32_t, 1> facet_indices,
    std::shared_ptr<const dolfinx::fem::FiniteElement> element,
    const dolfinx::fem::CoordinateElement& cmap,
    xt::xtensor<double, 2> facet_normals)
{
  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) >= num_points);
  assert(K.shape(0) >= num_points);
  assert(detJ.shape(0) >= num_points);

  // Get mesh data from input
  const size_t gdim = coordinate_dofs.shape(1);
  const size_t tdim = K.shape(1);

  // Get element data
  xt::xtensor<double, 2> X({x.shape(0), tdim});
  xt::xtensor<double, 2> normals = xt::zeros<double>({num_points, gdim});

  // Skip negative cell indices
  if (index >= 0)
  {
    pull_back(J, K, detJ, x, X, coordinate_dofs, element, cmap);

    for (std::size_t q = 0; q < num_points; ++q)
    {
      // Compute normal of physical facet using a normalized covariant Piola
      // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
      // DOI: 10.1137/08073901X
      auto _K = xt::view(K, q, xt::all(), xt::all());
      auto facet_normal = xt::row(facet_normals, facet_indices[q]);
      for (std::size_t i = 0; i < gdim; i++)
        for (std::size_t j = 0; j < tdim; j++)
          normals(q, i) += _K(j, i) * facet_normal[j];
      double n_norm = 0;
      for (std::size_t i = 0; i < gdim; i++)
        n_norm += normals(q, i) * normals(q, i);
      for (std::size_t i = 0; i < gdim; i++)
        normals(q, i) /= std::sqrt(n_norm);
    }
  }
  return normals;
}
