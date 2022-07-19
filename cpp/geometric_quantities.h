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
#include <xtensor/xview.hpp>
namespace dolfinx_contact
{

//-----------------------------------------------------------------------------
/// @brief Compute facet normal on physical cell
///
/// Computes the outward unit normal fora a point x located on
/// the surface of the physical cell. Each point x has a corresponding facet
/// index, relating to which local facet the point belongs to.
/// @note: The normal is computed using a covariant Piola transform
/// @note: The Jacobian J and its inverse K are computed internally, and
/// is only passed in to avoid dynamic memory allocation.
/// @param[in, out] J: Jacobian of transformation from reference element to
/// physical element. Shape = (gdim, tdim).
/// @param[in, out] K: Inverse of J. Shape = (tdim, gdim)
/// @param[in] x: The point on the facet of the physical cell(padded to 3D)
/// @param[in] coordinate_dofs: Geometry coordinates of cell
/// @param[in] facet_index: Local facet index
/// @param[in] cmap: The coordinate element
/// @param[in] reference_normals: The facet normals on the reference cell
std::array<double, 3>
push_forward_facet_normal(xt::xtensor<double, 2>& J, xt::xtensor<double, 2>& K,
                          const std::array<double, 3>& x,
                          const xt::xtensor<double, 2>& coordinate_dofs,
                          const std::size_t facet_index,
                          const dolfinx::fem::CoordinateElement& cmap,
                          const xt::xtensor<double, 2>& reference_normals);

/// @brief Pull back a single point of a non-affine cell to the reference cell.
///
/// Given a single cell and a point in the fell, pull it back to the reference
/// element. To compute this pull back Newton's method is employed.
/// @note The data structures `J`, `K` and `basis_values` does not correspond to
/// the Jacobian, its inverse and phi(X). They are sent in to avoid dynamic
/// memory allocation.
/// @param[in, out] X The point on the reference cell
/// @param[in, out] J The Jacobian
/// @param[in, out] K The inverse of the Jacobian
/// @param[in,out] basis Tabulated basis functions (and first order derivatives)
/// at a single point
/// @param[in] x The physical point
/// @param[in] cmap The coordinate element
/// @param[in] cell_geometry The cell geometry
/// @param[in] tol The tolerance for the Newton solver
/// @param[in] max_it The maximum number of Newton iterations
void pull_back_nonaffine(xt::xtensor<double, 2>& X, xt::xtensor<double, 2>& J,
                         xt::xtensor<double, 2>& K,
                         xt::xtensor<double, 4>& basis,
                         const std::array<double, 3>& x,
                         const dolfinx::fem::CoordinateElement& cmap,
                         const xt::xtensor<double, 2>& cell_geometry,
                         double tol = 1e-8, const int max_it = 10);

/// Compute circumradius for a cell with given coordinates and determinant
/// of Jacobian
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
  assert(physical_normal.size() == K.shape(1));
  const std::size_t tdim = K.shape(0);
  const std::size_t gdim = K.shape(1);
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