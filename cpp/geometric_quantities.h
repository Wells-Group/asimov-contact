// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xtensor.hpp>
namespace dolfinx_contact
{

//-----------------------------------------------------------------------------
/// Computes the outward unit normal for a set of points x located on
/// the surface of the cell. Each point x has a corresponding facet index,
/// relating to which local facet the point belongs to.
/// @note: The normal is computed using a covariant Piola transform
/// @note: The Jacobian J and its inverse K are computed internally, and
/// is only passed in to avoid dynamic memory allocation.
/// @param[in] x: points on facets physical element
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x.
/// @param[in, out] K: inverse of J at each point.
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] facet_indices: local facet index corresponding to each point
/// @param[in] cmap: the coordinate element
/// @param[in] reference_normals: The facet normals on the reference cell
xt::xtensor<double, 2>
push_forward_facet_normal(const xt::xtensor<double, 2>& x,
                          xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
                          const xt::xtensor<double, 2>& coordinate_dofs,
                          const xt::xtensor<std::int32_t, 1>& facet_indices,
                          const dolfinx::fem::CoordinateElement& cmap,
                          const xt::xtensor<double, 2>& reference_normals);

/// Compute circumradius for a cell with given coordinates and determinant of
/// Jacobian
/// @param[in] mesh The mesh
/// @param[in] detJ The determinant of the Jacobian of the mapping to the
/// reference cell
/// @param[in] coordinate_dofs The cell geometry
/// @returns The circumradius of the cell
double compute_circumradius(const dolfinx::mesh::Mesh& mesh, double detJ,
                            const xt::xtensor<double, 2>& coordinate_dofs);

/// @brief Push forward facet normal
///
/// Compute normal of physical facet using a normalized covariant Piola
/// transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref||
/// See: DOI: 10.1137/08073901X
/// @param[in, out] physical_normal The physical normal
/// @param[in] K The inverse of the Jacobian
/// @param[in] reference_normal The reference normal
template <class E, class F, class G>
void physical_facet_normal(E&& physical_normal, F&& K, G&& reference_normal)
{
  assert(physical_normal.shape(0) == K.shape(1));
  const std::size_t gdim = K.shape(0);
  const std::size_t tdim = K.shape(1);
  for (std::size_t i = 0; i < gdim; i++)
  {
    // FIXME: Replace with math-dot
    for (std::size_t j = 0; j < tdim; j++)
      physical_normal[i] += K(j, i) * reference_normal[j];
  }
  // Normalize vector
  double norm = 0;
  std::for_each(physical_normal.cbegin(), physical_normal.cend(),
                [&norm](auto ni) { norm += std::pow(ni, 2); });
  norm = std::sqrt(norm);
  std::for_each(physical_normal.begin(), physical_normal.end(),
                [norm](auto& ni) { ni = ni / norm; });
}

//-----------------------------------------------------------------------------

} // namespace dolfinx_contact